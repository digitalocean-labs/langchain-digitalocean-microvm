"""Processes sharing one gateway store never resume and delete the same MicroVM.

A gateway (or a supervisor with subagent sandboxes) claims a row before it
resumes or launches (`claim_host` / `claim_sandbox`); scripts/gc_microvms.py
and the gateway's own tick mark a row retiring before they delete
(`begin_retire_host` / `begin_retire_sandbox`). Each decision is one
`BEGIN IMMEDIATE` transaction, so whichever comes first wins:

- a claim between the sweep's plan and its application: not deleted, reported;
- retiring first: the claim fails and the gateway starts a fresh host instead
  of resuming the one being deleted (a task update is refused);
- a run started between plan and apply: skipped;
- a sweep that crashed after retiring: the next sweep deletes the host, and a
  gateway that needs it finishes the deletion itself;
- the same for subagent sandboxes;
- a multiprocessing stress test: claim and retire never both win.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

import httpx
import pytest
from fake_client import FakeMicroVMClient
from scripted_model import ScriptedChatModel
from test_gateway import FakeClock
from test_lifecycle import (
    TOKEN,
    EndpointServers,
    _agent_host,
    _row,
    launch,
    make_gateway,
)
from test_subagents import _lifecycle, _specs

from langchain_digitalocean_microvm import subagents
from langchain_digitalocean_microvm.agent_host import MicroVMAgentHost, SubAgentGateway
from langchain_digitalocean_microvm.agent_host.store import GatewayStore, HostRow
from langchain_digitalocean_microvm.agent_host.sweeper import apply_sweep, plan_sweep
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.subagents import (
    provision_subagent_sandboxes,
    subagent_sandbox_id,
)

HOUR = 3600.0


class Setup:
    """A gateway on MicroVMAgentHost over FakeMicroVMClient, and a second store on the same file (the sweeper's)."""

    def __init__(self, tmp_path: Path, **gateway_kwargs) -> None:
        self.client = FakeMicroVMClient()
        self.upstream = EndpointServers(self.client)
        tokens = count(1)
        config = MicroVMConfig(transport="exec", api_token="t", access_token="a", agent_image="img", idle_timeout="10m")
        hosts = MicroVMAgentHost(
            config, client=self.client, probe=lambda u, t, s: True, metrics_fetcher=self.upstream.metrics,  # type: ignore[arg-type]
            token_factory=lambda: f"tok-{next(tokens)}",
        )
        self.clock = FakeClock()
        self.gateway: SubAgentGateway = make_gateway(
            tmp_path, hosts, self.clock, poll_interval=60.0, upstream_transport=self.upstream.transport(),
            release_after=None, paused_ttl=None, **gateway_kwargs,  # only the sweeper deletes here
        )
        self.http = httpx.Client(base_url=self.gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"},
                                 timeout=30.0)
        self.sweeper_store = GatewayStore(tmp_path / "gateway.sqlite")

    def close(self) -> None:
        self.http.close()
        self.gateway.stop()
        self.sweeper_store.close()

    def finished_and_paused(self, assistant: str = "echo") -> tuple[str, str, str]:
        """Launch a task, let it finish and its host pause: (thread id, host id, MicroVM id)."""
        tid, rid = launch(self.http, assistant)
        host_id = self.gateway.store.get_thread(tid).host_id
        microvm_id = self.gateway.store.get_host(host_id).microvm_id
        self.server(microvm_id).complete(rid, "done")
        self.gateway.tick_sync()
        self.clock.advance(45.0)
        self.gateway.tick_sync()
        assert self.gateway.store.get_host(host_id).status == "paused"
        return tid, host_id, microvm_id

    def server(self, microvm_id: str):
        return next(s for host, s in self.upstream.servers.items() if host.startswith(microvm_id))

    def plan(self):
        return plan_sweep(self.client, store=self.sweeper_store, now=self.clock.t, host_ttl=HOUR)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Setup]:
    setup = Setup(tmp_path)
    try:
        yield setup
    finally:
        setup.close()


# ---------------------------------------------------------------- agent hosts


def test_a_host_claimed_between_plan_and_apply_is_not_deleted(env: Setup) -> None:
    tid, host_id, microvm_id = env.finished_and_paused()
    env.clock.advance(2 * HOUR)
    plan = env.plan()
    assert [i.id for i in plan.delete_microvms] == [microvm_id]

    _, rid2 = launch(env.http, tid=tid)  # an update: the gateway claims, then resumes
    result = apply_sweep(plan, env.client, store=env.sweeper_store)

    assert [i.id for i in result.skipped] == [microvm_id] and result.deleted_microvms == []
    assert env.client.delete_calls == 0 and env.client.resume_calls == 1
    assert env.gateway.store.get_host(host_id).status == "busy"
    env.server(microvm_id).complete(rid2, "again")
    assert env.http.get(f"/threads/{tid}/runs/{rid2}").json()["status"] == "success"


def test_a_retiring_pooled_host_is_replaced_not_resumed(env: Setup) -> None:
    _, host_id, old = env.finished_and_paused("shared")
    env.clock.advance(2 * HOUR)
    (item,) = env.plan().delete_microvms
    # The sweeper process has marked the row and is deleting the MicroVM (it takes a moment).
    assert env.sweeper_store.begin_retire_host(host_id, old, idle_before=item.idle_before, reason="gc")
    assert not env.gateway.store.claim_host(host_id, now=env.clock.t)

    def sweeper_finishes() -> None:
        time.sleep(1.0)
        env.client.delete(old)
        env.sweeper_store.retire_host(host_id, "gc", microvm_id=old)

    worker = threading.Thread(target=sweeper_finishes)
    worker.start()
    started = time.monotonic()
    launch(env.http, "shared")  # waits for the deletion, then creates a fresh pooled host
    waited = time.monotonic() - started
    worker.join()

    row = env.gateway.store.get_host(host_id)
    assert row.microvm_id != old and row.status == "busy"
    assert (env.client.create_calls, env.client.resume_calls) == (2, 0)
    assert waited >= 0.9


def test_a_task_whose_host_is_retiring_cannot_be_updated(env: Setup) -> None:
    tid, host_id, microvm_id = env.finished_and_paused()
    env.clock.advance(2 * HOUR)
    (item,) = env.plan().delete_microvms
    assert env.sweeper_store.begin_retire_host(host_id, microvm_id, idle_before=item.idle_before, reason="gc")

    update = env.http.post(f"/threads/{tid}/runs", json={"assistant_id": "echo", "input": {"messages": []}})
    assert update.status_code == 503
    assert "is being deleted by scripts/gc_microvms.py" in update.json()["detail"]
    assert "cannot be updated" in update.json()["detail"]
    assert env.client.resume_calls == 0
    assert env.http.get(f"/artifacts/{tid}").status_code == 410
    assert env.http.get(f"/threads/{tid}").status_code == 200  # the cached result is still there


def test_a_run_started_between_plan_and_apply_is_skipped(tmp_path: Path) -> None:
    client = FakeMicroVMClient()
    store = GatewayStore(tmp_path / "gateway.sqlite")
    other = GatewayStore(tmp_path / "gateway.sqlite")  # the gateway's process
    now = 100 * HOUR
    microvm_id = _agent_host(client, "h")
    _row(store, "h", microvm_id, last_active=now - 2 * HOUR)
    plan = plan_sweep(client, store=store, now=now, host_ttl=HOUR)
    assert [i.id for i in plan.delete_microvms] == [microvm_id]

    other.create_thread("t", now=now - 2 * HOUR)
    other.update_thread("t", host_id="h", run_id="r", run_status="pending")
    result = apply_sweep(plan, client, store=store)

    assert [i.id for i in result.skipped] == [microvm_id] and client.delete_calls == 0
    assert store.get_host("h").status == "paused"
    store.close()
    other.close()


def test_a_sweep_that_crashed_after_retiring_is_finished_by_the_next(tmp_path: Path) -> None:
    client = FakeMicroVMClient()
    store = GatewayStore(tmp_path / "gateway.sqlite")
    now = 100 * HOUR
    microvm_id = _agent_host(client, "h")
    _row(store, "h", microvm_id, last_active=now - 2 * HOUR)
    assert store.begin_retire_host("h", microvm_id, idle_before=now - HOUR, reason="gc")  # ... and the process dies

    assert not store.claim_host("h", now=now)
    plan = plan_sweep(client, store=store, now=now, host_ttl=10 * HOUR)  # not past this TTL: retiring decides
    assert [(i.id, i.reason) for i in plan.delete_microvms] == [(microvm_id, "the gateway store marks it retiring")]
    result = apply_sweep(plan, client, store=store)
    assert client.delete_calls == 1 and result.skipped == []
    assert (store.get_host("h").status, store.get_host("h").retired_reason) == ("destroyed", "gc")

    # Crashed after the delete, before recording it: the row is marked when its MicroVM is gone.
    second = _agent_host(client, "h2")
    _row(store, "h2", second, last_active=now - 2 * HOUR)
    assert store.begin_retire_host("h2", second, idle_before=now - HOUR, reason="gc")
    client.delete(second)
    result = apply_sweep(plan_sweep(client, store=store, now=now), client, store=store)
    assert result.retired_rows == ["h2"]
    assert (store.get_host("h2").status, store.get_host("h2").retired_reason) == ("destroyed", "gc")
    store.close()


def test_a_gateway_finishes_a_crashed_sweeps_deletion_itself(tmp_path: Path) -> None:
    env = Setup(tmp_path, retire_wait=0.5)
    try:
        _, host_id, old = env.finished_and_paused("shared")
        assert env.sweeper_store.begin_retire_host(host_id, old, idle_before=None, reason="gc")  # then crashed

        launch(env.http, "shared")
        assert env.client.delete_calls == 1 and env.client.resume_calls == 0
        row = env.gateway.store.get_host(host_id)
        assert row.microvm_id != old and row.status == "busy"
    finally:
        env.close()


def test_the_gateway_tick_uses_the_same_compare_and_set(env: Setup) -> None:
    tid, rid = launch(env.http)
    host_id = env.gateway.store.get_thread(tid).host_id
    env.server(env.gateway.store.get_host(host_id).microvm_id).complete(rid, "done")
    env.gateway.tick_sync()
    stale = env.gateway.store.get_host(host_id)
    assert stale.status == "ready"

    def idle_while_another_process_claims(handle) -> bool:
        assert env.sweeper_store.claim_host(host_id, now=env.clock.t + 1)  # lands during the gateway's await
        return True

    env.gateway.hosts.is_idle = idle_while_another_process_claims  # type: ignore[method-assign]
    retired = asyncio.run_coroutine_threadsafe(env.gateway._retire_host(stale, reason="ttl"), env.gateway._loop)
    assert retired.result(5) is False
    assert env.client.delete_calls == 0 and env.gateway.store.get_host(host_id).status == "ready"


# ---------------------------------------------------------------- subagent sandboxes

CREATED = datetime(2026, 10, 1, 22, 46, 36, tzinfo=UTC).timestamp()  # FakeMicroVMClient's created_at
DAY = 24 * HOUR


def _provision(lifecycle, store: GatewayStore, at: float):
    return provision_subagent_sandboxes(lifecycle, _specs()[:1], thread_id="t", default_model=ScriptedChatModel(),
                                        store=store, clock=lambda: at)


def test_a_sandbox_claimed_between_plan_and_apply_is_not_deleted(tmp_path: Path) -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    store = GatewayStore(tmp_path / "gateway.sqlite")
    sweeper = GatewayStore(tmp_path / "gateway.sqlite")
    first = _provision(lifecycle, store, CREATED).backends["analyst"].id
    plan = plan_sweep(client, store=sweeper, now=CREATED + 2 * DAY, sandbox_ttl=DAY)
    assert [i.id for i in plan.delete_microvms] == [first]

    again = _provision(lifecycle, store, CREATED + 2 * DAY).backends["analyst"].id  # a new turn
    result = apply_sweep(plan, client, store=sweeper)

    assert again == first and [i.id for i in result.skipped] == [first] and client.delete_calls == 0
    assert store.get_sandbox(subagent_sandbox_id("t", "analyst")).status == "active"
    store.close()
    sweeper.close()


def test_a_retiring_sandbox_is_replaced_not_resumed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subagents, "_RETIRE_POLL", 0.05)
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    store = GatewayStore(tmp_path / "gateway.sqlite")
    sweeper = GatewayStore(tmp_path / "gateway.sqlite")
    sandbox_id = subagent_sandbox_id("t", "analyst")
    old = _provision(lifecycle, store, CREATED).backends["analyst"].id
    assert sweeper.begin_retire_sandbox(sandbox_id, old, idle_before=CREATED + DAY, reason="gc")

    def sweeper_finishes() -> None:
        time.sleep(0.5)
        client.delete(old)
        sweeper.retire_sandbox(sandbox_id, "gc", microvm_id=old)

    worker = threading.Thread(target=sweeper_finishes)
    worker.start()
    fresh = _provision(lifecycle, store, CREATED + 2 * DAY).backends["analyst"].id
    worker.join()

    assert fresh != old and client.resume_calls == 0
    row = store.get_sandbox(sandbox_id)
    assert (row.microvm_id, row.status, row.retired_reason) == (fresh, "active", None)
    store.close()
    sweeper.close()


def test_a_crashed_sandbox_sweep_is_finished_by_the_next_provision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subagents, "RETIRE_WAIT", 0.3)
    monkeypatch.setattr(subagents, "_RETIRE_POLL", 0.05)
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    store = GatewayStore(tmp_path / "gateway.sqlite")
    sandbox_id = subagent_sandbox_id("t", "analyst")
    old = _provision(lifecycle, store, CREATED).backends["analyst"].id
    assert store.begin_retire_sandbox(sandbox_id, old, idle_before=None, reason="gc")  # then crashed

    fresh = _provision(lifecycle, store, CREATED + 2 * DAY).backends["analyst"].id
    assert fresh != old and client.delete_calls == 1 and client.resume_calls == 0
    assert store.get_sandbox(sandbox_id).status == "active"

    # The sweeper treats a retiring row like a destroyed one.
    assert store.begin_retire_sandbox(sandbox_id, fresh, idle_before=None, reason="gc")
    plan = plan_sweep(client, store=store, now=CREATED + 2 * DAY, sandbox_ttl=10 * DAY)
    assert [(i.id, i.reason) for i in plan.delete_microvms] == [(fresh, "the gateway store marks it retiring")]
    store.close()


# ---------------------------------------------------------------- stress: separate processes

ROWS = 150


def _claimer(path: str, barrier, out) -> None:
    store = GatewayStore(path)
    won = []
    for i in range(ROWS):
        barrier.wait()  # both processes go for row i at the same moment
        won.append(store.claim_host(f"h{i}", now=10.0))
    out.put(("claim", won))
    store.close()


def _retirer(path: str, barrier, out) -> None:
    store = GatewayStore(path)
    won = []
    for i in range(ROWS):
        barrier.wait()
        won.append(store.begin_retire_host(f"h{i}", f"vm{i}", idle_before=1.0, reason="gc"))
    out.put(("retire", won))
    store.close()


def test_claim_and_retire_never_both_win_across_processes(tmp_path: Path) -> None:
    path = str(tmp_path / "gateway.sqlite")
    store = GatewayStore(path)
    for i in range(ROWS):
        store.upsert_host(HostRow(f"h{i}", f"vm{i}", "echo", "paused", "https://x", "tok", "", 1.0, None, 0))
    store.close()

    ctx = multiprocessing.get_context("spawn")
    barrier, out = ctx.Barrier(2), ctx.Queue()
    workers = [ctx.Process(target=fn, args=(path, barrier, out)) for fn in (_claimer, _retirer)]
    for worker in workers:
        worker.start()
    results = dict(out.get(timeout=60) for _ in workers)
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    pairs = list(zip(results["claim"], results["retire"], strict=True))
    assert all(claimed != retired for claimed, retired in pairs), "exactly one side wins each row"
    store = GatewayStore(path)
    for i, (claimed, _) in enumerate(pairs):
        row = store.get_host(f"h{i}")
        assert (row.status, row.last_active_at) == (("paused", 10.0) if claimed else ("retiring", 1.0))
    store.close()
