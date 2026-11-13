"""Tests for agent_host.gateway.SubAgentGateway.

The gateway runs for real (embedded uvicorn on a loopback port) and is driven
with the LangGraph SDK, i.e. the exact client the deepagents async-subagent
middleware uses. Hosts are FakeAgentServers behind an httpx.MockTransport
(tests/fake_agent_server.py), so no `langgraph dev` is launched; the real
server is covered by test_gateway_e2e.py (opt-in). Time is a fake clock so
pause-after-grace is deterministic; `tick_sync()` runs one maintenance pass.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fake_agent_server import FakeHosts, make_transport
from langgraph_sdk import get_sync_client
from langgraph_sdk.errors import NotFoundError

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    SubAgentGateway,
    async_subagent_spec,
)
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError

TOKEN = "gw-token"
RUN_START = "2026-09-03T00:01:00Z"


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def time(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def gw(tmp_path: Path) -> Iterator[tuple[SubAgentGateway, FakeHosts, FakeClock, str]]:
    hosts = FakeHosts()
    clock = FakeClock()
    gateway = SubAgentGateway(
        hosts,
        {
            "researcher": HostSpec(graph_id="researcher-graph"),
            "shared": HostSpec(graph_id="shared-graph", reuse="pooled"),
        },
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        webhook_base_url="http://gateway.test",
        pause_grace=45.0,
        idle_backstop=300.0,
        poll_interval=3600.0,  # background loop stays out of the way; tests call tick_sync()
        unreachable_threshold=2,
        upstream_transport=make_transport(hosts),
        clock=clock.time,
        iso_now=lambda: RUN_START,
    )
    url = gateway.start_embedded()
    try:
        yield gateway, hosts, clock, url
    finally:
        gateway.stop()


def _client(url: str):
    """SDK client with a short timeout so a gateway hang fails the test instead of stalling it."""
    return get_sync_client(url=url, api_key=TOKEN, timeout=httpx.Timeout(20.0))


def _launch(url: str, assistant: str = "researcher", content: str = "do research") -> tuple[dict, dict]:
    client = _client(url)
    thread = client.threads.create()
    run = client.runs.create(
        thread["thread_id"],
        assistant_id=assistant,
        input={"messages": [{"role": "user", "content": content}]},
    )
    return thread, run


def test_launch_allocates_a_host_and_forwards_the_run(gw) -> None:
    gateway, hosts, _clock, url = gw

    thread, run = _launch(url)

    tid = thread["thread_id"]
    assert hosts.spawned() == [tid]  # per-task: host id == thread id
    server = hosts.servers[tid]
    assert tid in server.threads  # created ON the host with the same id
    forwarded = server.runs[run["run_id"]]
    assert forwarded["assistant_id"] == "researcher-graph"  # mapped from the supervisor's name
    assert forwarded["kwargs"]["webhook"].startswith("http://gateway.test/hooks/run-done?secret=")
    assert forwarded["kwargs"]["input"]["messages"][0]["content"] == "do research"
    row = gateway.store.get_thread(tid)
    assert row is not None and row.host_id == tid and row.run_id == run["run_id"]
    assert gateway.store.get_host(tid).status == "busy"


def test_check_proxies_while_running_then_serves_from_cache_and_pauses_host(gw) -> None:
    gateway, hosts, clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    server = hosts.servers[tid]

    assert client.runs.get(tid, rid)["status"] == "pending"  # proxied
    server.complete(rid, "findings: 42")

    gateway.tick_sync()  # polling fallback notices completion
    assert gateway.store.get_thread(tid).run_status == "success"
    assert gateway.store.get_host(tid).status == "ready"
    assert hosts.paused() == []  # grace period not over

    clock.advance(46.0)
    gateway.tick_sync()
    assert hosts.paused() == [tid]

    upstream_before = len(server.requests)
    assert client.runs.get(tid, rid)["status"] == "success"
    state = client.threads.get(tid)
    assert state["values"]["messages"][-1]["content"] == "findings: 42"
    assert len(server.requests) == upstream_before  # served from cache: the paused host was never woken
    assert hosts.resumed() == []


def test_webhook_completes_a_run_without_polling(gw) -> None:
    gateway, hosts, _clock, url = gw
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    hosts.servers[tid].complete(rid, "via webhook")

    hook = f"{url}/hooks/run-done"
    payload = {"run_id": rid, "thread_id": tid, "status": "success", "values": {"messages": [{"type": "ai", "content": "via webhook"}]}}
    assert httpx.post(hook, params={"secret": "nope"}, json=payload).status_code == 401
    assert httpx.post(hook, params={"secret": gateway._hook_secret}, json=payload).status_code == 200

    row = gateway.store.get_thread(tid)
    assert row.run_status == "success"
    assert row.cached_values["messages"][-1]["content"] == "via webhook"
    client = _client(url)
    assert client.runs.get(tid, rid)["status"] == "success"


def test_dead_host_is_reported_as_an_error_run(gw) -> None:
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]

    hosts.kill(tid)

    checked = client.runs.get(tid, rid)
    assert checked["status"] == "error"
    assert "unreachable" in checked["error"]
    assert gateway.store.get_host(tid).status == "failed"
    # Stable on repeat, and no attempt to recreate the host.
    assert client.runs.get(tid, rid)["status"] == "error"
    assert hosts.spawned() == [tid]


def test_host_deleted_outside_the_gateway_fails_every_task_on_it(gw) -> None:
    """A deleted MicroVM's endpoint answers 404 at the edge: the inventory says the host is gone,
    so the host is marked failed and all its active tasks fail at once, as unreachable."""
    gateway, hosts, _clock, url = gw
    client = _client(url)
    first, run1 = _launch(url, "shared")
    second, run2 = _launch(url, "shared")
    host_id = gateway.store.get_thread(first["thread_id"]).host_id
    assert gateway.store.get_thread(second["thread_id"]).host_id == host_id  # pooled: one host

    hosts.delete_outside(host_id)

    checked = client.runs.get(first["thread_id"], run1["run_id"])
    assert checked["status"] == "error"
    assert "unreachable (gone)" in checked["error"]
    assert gateway.store.get_host(host_id).status == "failed"
    other = gateway.store.get_thread(second["thread_id"])
    assert other.run_status == "error" and "unreachable" in other.cached_run["error"]


def test_missing_run_on_a_live_host_fails_only_that_task(gw) -> None:
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    del hosts.servers[tid].runs[rid]  # the Agent Server lost the run; the MicroVM is fine

    checked = client.runs.get(tid, rid)
    assert checked["status"] == "error"
    assert "no longer exists" in checked["error"]
    assert gateway.store.get_host(tid).status != "failed"


def test_transient_blip_is_tolerated_until_the_threshold(gw) -> None:
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    server = hosts.servers[tid]

    server.dead = True  # unreachable, but the inventory still says READY
    assert client.runs.get(tid, rid)["status"] == "pending"  # first failure: last known status
    assert gateway.store.get_host(tid).failures == 1
    server.dead = False
    assert client.runs.get(tid, rid)["status"] == "pending"  # recovered
    assert gateway.store.get_host(tid).failures == 0

    server.dead = True
    client.runs.get(tid, rid)
    assert client.runs.get(tid, rid)["status"] == "error"  # threshold (2) reached


def test_restarted_host_orphans_its_run(gw) -> None:
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]

    hosts.restart(tid)
    gateway._record_handle(hosts._handle(tid), "researcher")  # the gateway learns the new boot time

    checked = client.runs.get(tid, rid)
    assert checked["status"] == "error"
    assert "restarted" in checked["error"]


def test_update_resumes_a_paused_host_and_forwards_interrupt(gw) -> None:
    gateway, hosts, clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    server = hosts.servers[tid]
    server.complete(rid, "first")
    gateway.tick_sync()
    clock.advance(60.0)
    gateway.tick_sync()
    assert hosts.paused() == [tid]

    run2 = client.runs.create(
        tid,
        assistant_id="researcher",
        input={"messages": [{"role": "user", "content": "more"}]},
        multitask_strategy="interrupt",
    )

    assert hosts.resumed() == [tid]
    assert server.runs[run2["run_id"]]["multitask_strategy"] == "interrupt"
    assert gateway.store.get_thread(tid).run_id == run2["run_id"]
    assert gateway.store.get_host(tid).status == "busy"
    # The superseded run is reported as interrupted, the new one proxied.
    assert client.runs.get(tid, rid)["status"] == "interrupted"
    assert client.runs.get(tid, run2["run_id"])["status"] == "pending"


def test_cancel_is_proxied_while_running_and_local_once_terminal(gw) -> None:
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    server = hosts.servers[tid]

    client.runs.cancel(tid, rid)  # running: proxied
    assert server.runs[rid]["status"] == "interrupted"
    assert gateway.store.get_thread(tid).run_status == "interrupted"

    client.runs.cancel(tid, rid)  # terminal: answered locally (the host would 404)
    thread2, run2 = _launch(url)
    hosts.servers[thread2["thread_id"]].complete(run2["run_id"], "done")
    gateway.tick_sync()
    client.runs.cancel(thread2["thread_id"], run2["run_id"])  # no NotFoundError
    with pytest.raises(NotFoundError):
        client.runs.cancel("no-such-thread", "no-such-run")


def test_pooled_graphs_share_one_host(gw) -> None:
    gateway, hosts, _clock, url = gw
    t1, _ = _launch(url, assistant="shared", content="a")
    t2, _ = _launch(url, assistant="shared", content="b")

    assert hosts.spawned() == ["pool-shared"]
    server = hosts.servers["pool-shared"]
    assert set(server.threads) == {t1["thread_id"], t2["thread_id"]}
    assert all(r["assistant_id"] == "shared-graph" for r in server.runs.values())


def test_gateway_requires_its_token_but_accepts_either_header(gw) -> None:
    _gateway, _hosts, _clock, url = gw

    assert httpx.post(f"{url}/threads", json={}).status_code == 401
    assert httpx.post(f"{url}/threads", json={}, headers={"x-api-key": "wrong"}).status_code == 401
    assert httpx.post(f"{url}/threads", json={}, headers={"x-api-key": TOKEN}).status_code == 200
    assert httpx.post(f"{url}/threads", json={}, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    # The SDK's auto-sent x-api-key may be the supervisor's LangSmith key; Authorization still wins.
    both = {"Authorization": f"Bearer {TOKEN}", "x-api-key": "lsv2_supervisor_key"}
    assert httpx.post(f"{url}/threads", json={}, headers=both).status_code == 200
    assert httpx.get(f"{url}/ok").status_code == 200
    assert httpx.get(f"{url}/gateway/status", headers={"x-api-key": TOKEN}).json()["hosts"] == []


def test_unknown_assistant_is_a_clear_404(gw) -> None:
    _gateway, hosts, _clock, url = gw
    client = _client(url)
    thread = client.threads.create()

    with pytest.raises(NotFoundError, match="Unknown assistant_id"):
        client.runs.create(thread["thread_id"], assistant_id="nope", input={"messages": []})
    assert hosts.spawned() == []


def test_async_subagent_spec_points_at_the_gateway() -> None:
    spec = async_subagent_spec(
        name="researcher",
        description="Researches things",
        graph_id="researcher",
        gateway_url="http://127.0.0.1:5555/",
        gateway_token=TOKEN,
    )
    assert spec["url"] == "http://127.0.0.1:5555"
    assert spec["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert spec["graph_id"] == "researcher"


def test_tick_sync_requires_a_running_gateway(tmp_path: Path) -> None:
    gateway = SubAgentGateway(FakeHosts(), {}, db_path=tmp_path / "g.sqlite", token=TOKEN)
    with pytest.raises(Exception, match="not running"):
        gateway.tick_sync()
    gateway.stop()
    time.sleep(0)  # nothing to wait for; stop() on a never-started gateway must be harmless


class ForgetfulHosts(FakeHosts):
    """A manager that never heard of some stored host (LocalDevServerHost after a supervisor restart)."""

    def is_idle(self, handle) -> bool:
        if handle.host_id not in self.servers:
            raise SandboxNotFoundError(f"No local host registered for host_id={handle.host_id!r}")
        return super().is_idle(handle)


def test_tick_marks_a_host_the_manager_forgot_as_failed_instead_of_crashing(tmp_path: Path) -> None:
    hosts = ForgetfulHosts()
    clock = FakeClock()
    gateway = SubAgentGateway(
        hosts,
        {"researcher": HostSpec(graph_id="researcher-graph")},
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        pause_grace=45.0,
        poll_interval=3600.0,
        upstream_transport=make_transport(hosts),
        clock=clock.time,
        iso_now=lambda: RUN_START,
    )
    url = gateway.start_embedded()
    try:
        thread, run = _launch(url)
        tid = thread["thread_id"]
        hosts.servers[tid].complete(run["run_id"], "done")
        gateway.tick_sync()  # run cached, host idle since now
        assert gateway.store.get_host(tid).status == "ready"

        # The supervisor restarted: the store remembers the host, the new manager does not.
        hosts.servers.pop(tid)
        hosts.status.pop(tid)
        clock.advance(46.0)

        gateway.tick_sync()  # used to raise out of the tick every poll interval

        assert gateway.store.get_host(tid).status == "failed"
        assert hosts.paused() == []
    finally:
        gateway.stop()


@pytest.mark.parametrize("old_run_answer", ["interrupted", "gone"])
def test_poll_of_a_superseded_run_does_not_overwrite_the_new_run(tmp_path: Path, old_run_answer: str) -> None:
    """A poll of a superseded run (its GET in flight while `update` starts a new run) never
    caches the old run's `interrupted` as the new run's result."""
    hosts = FakeHosts()
    inner = make_transport(hosts)
    race: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if race and request.method == "GET" and request.url.path.endswith(f"/runs/{race['old']}"):
            # The update lands on the gateway while this request is in flight.
            gateway.store.update_thread(race["tid"], run_id="run-new", run_status="pending", cached_run=None)
            if old_run_answer == "gone":
                return httpx.Response(404, json={"detail": "Run not found"})
            hosts.servers[race["tid"]].runs[race["old"]]["status"] = "interrupted"
        return await inner.handle_async_request(request)

    gateway = SubAgentGateway(
        hosts,
        {"researcher": HostSpec(graph_id="researcher-graph")},
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        poll_interval=3600.0,
        upstream_transport=httpx.MockTransport(handler),
        iso_now=lambda: RUN_START,
    )
    url = gateway.start_embedded()
    try:
        thread, run = _launch(url)
        race.update(tid=thread["thread_id"], old=run["run_id"])

        gateway.tick_sync()  # the poll that races with the update

        row = gateway.store.get_thread(thread["thread_id"])
        assert (row.run_id, row.run_status, row.cached_run) == ("run-new", "pending", None)
    finally:
        gateway.stop()


def test_a_cached_result_is_only_served_for_its_own_run(gw) -> None:
    """A row whose cache belongs to an older run (written before the fix above) heals on the next check."""
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    gateway.store.update_thread(tid, run_status="interrupted", cached_run={"run_id": "older", "status": "interrupted"})

    assert client.runs.get(tid, rid)["status"] == "pending"  # proxied to the host, not the stale cache


APPROVAL = {"type": "action_approval", "tool": "digitalocean_restore-droplet", "arguments": {"ID": 1, "ImageID": 2}}


def test_a_run_that_waits_for_a_human_is_cached_as_waiting(gw) -> None:
    """A run that hit `interrupt()` ends `success`; the cached thread keeps `interrupted` and the interrupt, without the host."""
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, rid = thread["thread_id"], run["run_id"]
    server = hosts.servers[tid]
    server.pause_for_human(rid, APPROVAL)

    gateway.tick_sync()
    assert client.runs.get(tid, rid)["status"] == "success"
    asked = len(server.requests)
    cached = client.threads.get(tid)
    assert cached["status"] == "interrupted"
    assert [i["value"] for items in cached["interrupts"].values() for i in items] == [APPROVAL]
    assert len(server.requests) == asked  # served from the cache: a waiting task's check does not wake its host


def test_an_approvers_resume_is_followed_through_the_runs_list(gw) -> None:
    """Another client resumes the task's thread; the middleware's `runs.list` finds that run, then the task's answer."""
    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid, old = thread["thread_id"], run["run_id"]
    server = hosts.servers[tid]
    server.pause_for_human(old, APPROVAL)
    gateway.tick_sync()

    resumed = client.runs.create(tid, assistant_id="researcher", command={"resume": {"interrupt-1": "approve"}})
    assert server.runs[resumed["run_id"]]["kwargs"]["command"] == {"resume": {"interrupt-1": "approve"}}
    assert client.runs.get(tid, old)["status"] == "interrupted"  # the tracked run was replaced
    assert [r["run_id"] for r in client.runs.list(tid, limit=1)] == [resumed["run_id"]]
    assert client.threads.get(tid)["status"] == "busy"  # proxied while the new run works

    server.complete(resumed["run_id"], "restored")
    gateway.tick_sync()
    latest = client.runs.list(tid, limit=1)
    assert [(r["run_id"], r["status"]) for r in latest] == [(resumed["run_id"], "success")]
    done = client.threads.get(tid)
    assert done["status"] == "idle" and done["interrupts"] == {}
    assert done["values"]["messages"][-1]["content"] == "restored"
    assert client.runs.list(tid, status="running") == []


def test_runs_list_of_a_thread_without_runs_is_empty(gw) -> None:
    _gateway, _hosts, _clock, url = gw
    client = _client(url)
    thread = client.threads.create()
    assert client.runs.list(thread["thread_id"]) == []
    with pytest.raises(NotFoundError):
        client.runs.list("no-such-thread")


def _middleware_follows_resumes() -> bool:
    """deepagents >= 0.7.30: its async-subagent check follows a superseded run through `runs.list` (0.7.29 reads the
    thread's status only). deepagents-code pins older 0.7 releases on some Pythons; there tasks never report waiting."""
    import deepagents

    major, minor, patch = (int(part) for part in deepagents.__version__.split(".")[:3])
    return (major, minor, patch) >= (0, 7, 30)


@pytest.mark.skipif(not _middleware_follows_resumes(), reason="this deepagents does not follow resumed tasks")
def test_contract_the_async_subagent_check_reads_waiting_and_follows_a_resume(gw) -> None:
    """The gateway's cached `interrupted` thread and its runs list are what the deepagents check relies on.

    `_observe` and `_task_status` are private: if deepagents renames them or stops reading the thread, fail here.
    """
    from deepagents.middleware import async_subagents as mw

    gateway, hosts, _clock, url = gw
    client = _client(url)
    thread, run = _launch(url)
    tid = thread["thread_id"]
    task = {"task_id": tid, "agent_name": "researcher", "thread_id": tid, "run_id": run["run_id"], "status": "running"}
    server = hosts.servers[tid]
    server.pause_for_human(run["run_id"], APPROVAL)
    gateway.tick_sync()

    observed, seen = mw._observe(client, task)
    assert mw._task_status(observed, seen) == "waiting"
    assert [i["value"] for i in mw._pending_interrupts(seen)] == [APPROVAL]

    resumed = client.runs.create(tid, assistant_id="researcher", command={"resume": {"interrupt-1": "approve"}})
    observed, seen = mw._observe(client, task)  # the tracked run is superseded: the check follows the newest one
    assert observed["run_id"] == resumed["run_id"]
    assert mw._task_status(observed, seen) in {"pending", "running"}

    server.complete(resumed["run_id"], "restored")
    gateway.tick_sync()
    observed, seen = mw._observe(client, {**task, "run_id": resumed["run_id"]})
    assert mw._task_status(observed, seen) == "success"
