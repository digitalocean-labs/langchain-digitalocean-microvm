"""Lifecycle (TTL garbage collection): hosts do not outlive their usefulness.

Scripted timelines on a fake clock, counting pause/resume/delete calls:

- the gateway's tick() pauses an idle host after the grace, deletes it once it
  has been unused for its TTL (`release_after` per-task, `paused_ttl` pooled),
  never while a run is active, and deletes FAILED hosts on the next tick;
- any use (an update, an artifact fetch) restarts the TTL, which is therefore
  also the artifact retention window; after deletion `check` is still served
  from the cache, an update is refused and an artifact fetch answers 410 with
  the reason;
- the same timeline on a real MicroVMAgentHost over FakeMicroVMClient;
- the sweeper behind scripts/gc_microvms.py: what it deletes, keeps and marks.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from fake_agent_server import FakeAgentServer, FakeHosts, make_transport
from fake_client import FakeMicroVMClient
from test_gateway import FakeClock
from test_gateway_artifacts import FilesystemFakeHosts

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    MicroVMAgentHost,
    SubAgentGateway,
)
from langchain_digitalocean_microvm.agent_host.host import GOLDEN_TAG
from langchain_digitalocean_microvm.agent_host.models import (
    HOST_MANAGED_TAG,
    host_graph_tag,
    host_id_tag,
)
from langchain_digitalocean_microvm.agent_host.store import GatewayStore, HostRow
from langchain_digitalocean_microvm.agent_host.sweeper import (
    apply_sweep,
    delete_checkpoint_patiently,
    plan_sweep,
)
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import MicroVMAPIError
from langchain_digitalocean_microvm.models import MicroVMSize

TOKEN = "gw-lifecycle-token"
GRAPHS = {"echo": HostSpec(graph_id="echo"), "shared": HostSpec(graph_id="echo", reuse="pooled")}


def make_gateway(tmp_path: Path, hosts, clock: FakeClock, **kwargs) -> SubAgentGateway:
    return SubAgentGateway(
        hosts,
        GRAPHS,
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        pause_grace=45.0,
        poll_interval=kwargs.pop("poll_interval", 3600.0),  # ticks are driven by the test
        release_after=kwargs.pop("release_after", 3600.0),
        paused_ttl=kwargs.pop("paused_ttl", 86400.0),
        upstream_transport=kwargs.pop("upstream_transport", None) or make_transport(hosts),
        clock=clock.time,
        **kwargs,
    )


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[tuple[SubAgentGateway, FakeHosts, FakeClock, httpx.Client]]:
    hosts = FakeHosts()
    clock = FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock)
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        yield gateway, hosts, clock, http
    finally:
        http.close()
        gateway.stop()


def launch(http: httpx.Client, assistant: str = "echo", tid: str | None = None) -> tuple[str, str]:
    if tid is None:
        tid = http.post("/threads", json={}).json()["thread_id"]
    response = http.post(f"/threads/{tid}/runs", json={"assistant_id": assistant, "input": {"messages": []}})
    assert response.status_code == 200, response.text
    return tid, response.json()["run_id"]


def ops(hosts: FakeHosts, host_id: str) -> list[str]:
    return [op for op, h in hosts.calls if h == host_id]


def finish(gateway: SubAgentGateway, server: FakeAgentServer, run_id: str, content: str = "done") -> None:
    server.complete(run_id, content)
    gateway.tick_sync()  # result cached, host idle from now


# ---------------------------------------------------------------- gateway tick


def test_an_idle_per_task_host_is_paused_then_deleted_after_its_ttl(setup) -> None:
    gateway, hosts, clock, http = setup
    tid, rid = launch(http)
    finish(gateway, hosts.servers[tid], rid, "echo: one")

    clock.advance(45.0)
    gateway.tick_sync()
    assert ops(hosts, tid) == ["get_or_create", "pause"]

    clock.advance(3600.0 - 46.0)  # 1 s short of release_after since the run finished
    gateway.tick_sync()
    assert gateway.store.get_host(tid).status == "paused"

    clock.advance(1.0)
    gateway.tick_sync()
    gateway.tick_sync()  # idempotent
    assert ops(hosts, tid) == ["get_or_create", "pause", "delete"]
    row = gateway.store.get_host(tid)
    assert (row.status, row.retired_reason) == ("destroyed", "ttl")

    # The supervisor can still check the task; an update is refused with the reason.
    assert http.get(f"/threads/{tid}/runs/{rid}").json()["status"] == "success"
    assert http.get(f"/threads/{tid}").json()["values"]["messages"][-1]["content"] == "echo: one"
    update = http.post(f"/threads/{tid}/runs", json={"assistant_id": "echo", "input": {"messages": []}})
    assert update.status_code == 503
    assert "time-to-live" in update.json()["detail"] and "cannot be updated" in update.json()["detail"]
    assert gateway.status().hosts[0]["retired_reason"] == "ttl"


def test_a_host_with_an_active_run_is_never_paused_or_deleted(setup) -> None:
    gateway, hosts, clock, http = setup
    tid, rid = launch(http)
    for _ in range(5):
        clock.advance(86400.0)
        gateway.tick_sync()
    assert ops(hosts, tid) == ["get_or_create"]
    assert gateway.store.get_host(tid).status == "busy"
    assert http.get(f"/threads/{tid}/runs/{rid}").json()["status"] == "pending"


def test_an_update_restarts_the_ttl(setup) -> None:
    gateway, hosts, clock, http = setup
    tid, rid = launch(http)
    finish(gateway, hosts.servers[tid], rid)
    clock.advance(3000.0)
    gateway.tick_sync()  # paused

    _, rid2 = launch(http, tid=tid)  # update: resumes the host
    finish(gateway, hosts.servers[tid], rid2)
    clock.advance(3000.0)  # 6000 s after the first run, 3000 s after the update
    gateway.tick_sync()
    assert ops(hosts, tid) == ["get_or_create", "pause", "resume", "pause"]

    clock.advance(600.0)
    gateway.tick_sync()
    assert ops(hosts, tid)[-1] == "delete"


def test_pooled_hosts_use_paused_ttl_and_are_recreated_on_demand(tmp_path) -> None:
    hosts, clock = FakeHosts(), FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock, release_after=10.0, paused_ttl=100.0)
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        tid, rid = launch(http, "shared")
        finish(gateway, hosts.servers["pool-shared"], rid)
        clock.advance(50.0)  # past release_after, which only applies to per-task hosts
        gateway.tick_sync()
        assert gateway.store.get_host("pool-shared").status == "paused"
        clock.advance(50.0)
        gateway.tick_sync()
        assert gateway.store.get_host("pool-shared").retired_reason == "ttl"
        assert ops(hosts, "pool-shared") == ["get_or_create", "pause", "delete"]

        tid2, _ = launch(http, "shared")  # a new task on that graph gets a fresh pooled host
        assert ops(hosts, "pool-shared")[-1] == "get_or_create"
        assert gateway.store.get_host("pool-shared").status == "busy"
        assert http.get(f"/threads/{tid}/runs/{rid}").json()["status"] == "success"
    finally:
        http.close()
        gateway.stop()


def test_ttls_of_none_never_delete(tmp_path) -> None:
    hosts, clock = FakeHosts(), FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock, release_after=None, paused_ttl=None)
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        tid, rid = launch(http)
        finish(gateway, hosts.servers[tid], rid)
        clock.advance(10 * 86400.0)
        gateway.tick_sync()
        gateway.tick_sync()
        assert ops(hosts, tid) == ["get_or_create", "pause"]
    finally:
        http.close()
        gateway.stop()
    with pytest.raises(ValueError, match="release_after"):
        make_gateway(tmp_path, FakeHosts(), clock, release_after=0.0)


def test_a_failed_host_is_deleted_on_the_next_tick(setup) -> None:
    gateway, hosts, _clock, http = setup
    tid, rid = launch(http)
    hosts.kill(tid)
    assert http.get(f"/threads/{tid}/runs/{rid}").json()["status"] == "error"
    assert gateway.store.get_host(tid).status == "failed"

    gateway.tick_sync()
    assert ops(hosts, tid) == ["get_or_create", "delete"]
    assert gateway.store.get_host(tid).retired_reason == "failed"


def test_a_ready_host_that_reports_work_in_flight_is_not_deleted(tmp_path) -> None:
    hosts, clock = FakeHosts(), FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock)
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        tid, rid = launch(http)
        finish(gateway, hosts.servers[tid], rid)
        hosts.is_idle = lambda handle: False  # e.g. a run the gateway does not track
        clock.advance(7200.0)
        gateway.tick_sync()
        gateway.tick_sync()
        assert ops(hosts, tid) == ["get_or_create"]
        assert gateway.store.get_host(tid).status == "ready"
    finally:
        http.close()
        gateway.stop()


def test_a_host_used_after_the_ttl_decision_is_left_alone(setup) -> None:
    gateway, hosts, clock, http = setup
    tid, rid = launch(http)
    finish(gateway, hosts.servers[tid], rid)
    stale = gateway.store.get_host(tid)
    gateway.store.update_host(tid, last_active_at=clock.t + 1)  # e.g. an artifact fetch in between

    assert asyncio.run_coroutine_threadsafe(gateway._retire_host(stale, reason="ttl"), gateway._loop).result(5) is False
    assert ops(hosts, tid) == ["get_or_create"]


# ---------------------------------------------------------------- artifact retention


def test_artifacts_stay_fetchable_until_the_ttl_and_then_answer_410_with_the_reason(tmp_path) -> None:
    hosts, clock = FilesystemFakeHosts(tmp_path / "hosts"), FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock)
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        tid, rid = launch(http)
        hosts.save(tid, tid, "report.md", "# findings")
        finish(gateway, hosts.servers[tid], rid)
        clock.advance(3500.0)
        gateway.tick_sync()  # paused, not yet expired

        fetched = http.get(f"/artifacts/{tid}/files/report.md")  # resumes the host and counts as activity
        assert fetched.content == b"# findings"
        gateway.tick_sync()  # idle again from now (the poll loop does this within poll_interval)
        clock.advance(3500.0)
        gateway.tick_sync()
        assert gateway.store.get_host(tid).status == "paused"  # 7000 s after the run, 3500 s after the fetch

        clock.advance(100.0)
        gateway.tick_sync()
        gone = http.get(f"/artifacts/{tid}")
        assert gone.status_code == 410
        assert "time-to-live" in gone.json()["detail"]
        assert ops(hosts, tid) == ["get_or_create", "pause", "resume", "pause", "delete"]
    finally:
        http.close()
        gateway.stop()


# ---------------------------------------------------------------- on MicroVMAgentHost + FakeMicroVMClient


class EndpointServers:
    """Upstream transport and metrics for MicroVMAgentHost hosts: one FakeAgentServer per MicroVM endpoint."""

    def __init__(self, client: FakeMicroVMClient) -> None:
        self.client = client
        self.servers: dict[str, FakeAgentServer] = {}

    def server(self, hostname: str, token: str) -> FakeAgentServer:
        microvm_id = hostname.split(".", 1)[0]
        state = self.client.get(microvm_id)
        if state.status.value != "running":
            raise httpx.ConnectError(f"{microvm_id} is {state.status.value}")
        return self.servers.setdefault(hostname, FakeAgentServer(token=token))

    def transport(self) -> httpx.MockTransport:
        async def handler(request: httpx.Request) -> httpx.Response:
            server = self.server(request.url.host, request.headers.get("x-api-key", ""))
            return await server.handle(request)

        return httpx.MockTransport(handler)

    def metrics(self, url: str, token: str, timeout: float) -> dict:
        active = self.server(urlparse(url).hostname or "", token).active_runs()
        return {"queue": {"n_pending": len(active), "n_running": 0}}


def test_ttl_timeline_on_microvm_agent_hosts_counts_api_calls(tmp_path) -> None:
    client = FakeMicroVMClient()
    upstream = EndpointServers(client)
    tokens = count(1)
    config = MicroVMConfig(transport="exec", api_token="t", access_token="a", agent_image="img", idle_timeout="10m")
    hosts = MicroVMAgentHost(
        config, client=client, probe=lambda u, t, s: True, metrics_fetcher=upstream.metrics,  # type: ignore[arg-type]
        token_factory=lambda: f"tok-{next(tokens)}",
    )
    clock = FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock, poll_interval=60.0, upstream_transport=upstream.transport())
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        tid, rid = launch(http)
        (server,) = upstream.servers.values()
        server.complete(rid, "echo")
        gateway.tick_sync()
        assert (client.create_calls, client.pause_calls, client.delete_calls) == (1, 0, 0)

        clock.advance(45.0)
        gateway.tick_sync()
        assert client.pause_calls == 1 and client.live()[0].status.value == "paused"

        _, rid2 = launch(http, tid=tid)  # update: resume (0.5 s on a real MicroVM), not a new host
        assert (client.create_calls, client.resume_calls) == (1, 1)
        server.complete(rid2, "echo again")
        gateway.tick_sync()
        clock.advance(45.0)
        gateway.tick_sync()
        assert client.pause_calls == 2

        clock.advance(3600.0)
        gateway.tick_sync()
        assert (client.create_calls, client.pause_calls, client.resume_calls, client.delete_calls) == (1, 2, 1, 1)
        assert client.live() == []
        assert gateway.store.get_host(tid).retired_reason == "ttl"
    finally:
        http.close()
        gateway.stop()


# ---------------------------------------------------------------- sweeper (scripts/gc_microvms.py)

CREATED = datetime(2026, 10, 1, 22, 46, 36, tzinfo=UTC)  # FakeMicroVMClient's created_at
HOUR = 3600.0


def _agent_host(client: FakeMicroVMClient, host_id: str) -> str:
    state = client.create(
        name=f"microvm-agenthost-{host_id}", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048),
        tags=[HOST_MANAGED_TAG, host_id_tag(host_id), host_graph_tag("echo")],
    )
    return state.id


def _row(store: GatewayStore, host_id: str, microvm_id: str, *, status: str = "paused", last_active: float) -> None:
    store.upsert_host(HostRow(host_id, microvm_id, "echo", status, "https://x", "tok", "", last_active, None, 0))


def test_the_sweep_deletes_stale_hosts_and_checkpoints_and_keeps_the_rest(tmp_path) -> None:
    client = FakeMicroVMClient()
    store = GatewayStore(tmp_path / "gateway.sqlite")
    now = CREATED.timestamp() + 48 * HOUR

    old = _agent_host(client, "old")
    _row(store, "old", old, last_active=now - 25 * HOUR)
    recent = _agent_host(client, "recent")
    _row(store, "recent", recent, last_active=now - 2 * HOUR)
    busy = _agent_host(client, "busy")
    _row(store, "busy", busy, status="busy", last_active=now - 30 * HOUR)
    store.create_thread("t-busy", now=now - 30 * HOUR)
    store.update_thread("t-busy", host_id="busy", run_id="r", run_status="running")
    failed = _agent_host(client, "failed")
    _row(store, "failed", failed, status="failed", last_active=now - 60.0)
    orphan = _agent_host(client, "orphan")  # no row: created 48 h ago
    _row(store, "vanished", "01gone", last_active=now - HOUR)  # its MicroVM no longer exists
    sandbox = client.create(name="microvm-sandbox-x", oci_ref="img", tags=["microvm-sandbox-id:x"]).id
    golden_vm = client.create(name="microvm-golden-abc", oci_ref="img", tags=[GOLDEN_TAG]).id

    stale_cp = client.create_checkpoint(golden_vm, name="microvm-golden-111111111111")
    fresh_cp = client.create_checkpoint(golden_vm, name="microvm-golden-222222222222")
    failed_cp = client.create_checkpoint(golden_vm, name="microvm-golden-333333333333")
    other_cp = client.create_checkpoint(sandbox, name="my-own-checkpoint")
    client._checkpoints[stale_cp.id] = replace(stale_cp, created_at=CREATED)
    client._checkpoints[fresh_cp.id] = replace(fresh_cp, created_at=CREATED + timedelta(hours=47))
    client._checkpoints[failed_cp.id] = replace(failed_cp, status="CHECKPOINT_FAILED", created_at=CREATED)
    client._checkpoints[other_cp.id] = replace(other_cp, created_at=CREATED)

    plan = plan_sweep(client, store=store, now=now, host_ttl=24 * HOUR, checkpoint_ttl=24 * HOUR)
    assert client.delete_calls == 0  # planning only reads
    assert {i.id for i in plan.delete_microvms} == {old, failed, orphan, golden_vm}
    assert {i.id for i in plan.kept if i.kind == "host"} == {recent, busy}
    assert "in progress" in next(i.reason for i in plan.kept if i.id == busy)
    assert {i.id for i in plan.delete_checkpoints} == {stale_cp.id, failed_cp.id}
    assert [i.id for i in plan.retire_rows] == ["vanished"]

    result = apply_sweep(plan, client, store=store)
    assert result.errors == []
    assert client.delete_calls == 4
    assert {s.id for s in client.live()} == {recent, busy, sandbox}
    assert {c.id for c in client.list_checkpoints()} == {fresh_cp.id, other_cp.id}
    assert {r.host_id: (r.status, r.retired_reason) for r in store.list_hosts()} == {
        "old": ("destroyed", "gc"),
        "recent": ("paused", None),
        "busy": ("busy", None),
        "failed": ("destroyed", "gc"),
        "vanished": ("destroyed", "vanished"),
    }

    again = plan_sweep(client, store=store, now=now, host_ttl=24 * HOUR, checkpoint_ttl=24 * HOUR)
    assert again.empty
    store.close()


def test_without_a_store_young_hosts_are_kept_and_old_ones_are_orphans() -> None:
    client = FakeMicroVMClient()
    host = _agent_host(client, "h")
    young = plan_sweep(client, store=None, now=CREATED.timestamp() + HOUR, host_ttl=24 * HOUR)
    assert young.empty and young.kept[0].id == host
    old = plan_sweep(client, store=None, now=CREATED.timestamp() + HOUR, host_ttl=24 * HOUR, orphan_ttl=30 * 60.0)
    assert [i.id for i in old.delete_microvms] == [host]
    assert "not in the gateway store" in old.delete_microvms[0].reason


def test_checkpoint_deletion_waits_out_the_capture() -> None:
    class Capturing(FakeMicroVMClient):
        refusals = 2

        def delete_checkpoint(self, checkpoint_id: str) -> None:
            if self.refusals:
                self.refusals -= 1
                raise MicroVMAPIError(400, "checkpoint is still being created")
            super().delete_checkpoint(checkpoint_id)

    client = Capturing()
    sleeps: list[float] = []
    delete_checkpoint_patiently(client, "cp-1", sleep=sleeps.append)
    assert sleeps == [5.0, 5.0]
    client.refusals = 5
    with pytest.raises(MicroVMAPIError):
        delete_checkpoint_patiently(client, "cp-1", attempts=3, sleep=sleeps.append)


def test_store_adds_the_retired_reason_column_to_an_existing_file(tmp_path) -> None:
    path = tmp_path / "m6.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE hosts (host_id TEXT PRIMARY KEY, microvm_id TEXT, graph_id TEXT, status TEXT, url TEXT,"
        " token TEXT, booted_at TEXT, last_active_at REAL, idle_since REAL, failures INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO hosts (host_id, status, last_active_at) VALUES ('old', 'paused', 1);"
    )
    conn.close()

    store = GatewayStore(path)
    try:
        assert store.get_host("old").retired_reason is None
        store.retire_host("old", "gc")
        assert (store.get_host("old").status, store.get_host("old").retired_reason) == ("destroyed", "gc")
    finally:
        store.close()
