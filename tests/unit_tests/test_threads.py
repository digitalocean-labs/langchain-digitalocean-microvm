"""Thread sandboxes: one MicroVM per conversation thread (threads.ThreadSandboxes).

On `FakeMicroVMClient` (create/pause/resume counters) and in local mode: get-or-create on
enter and pause on exit, a template clone for new threads, the bound on concurrent
creates, a teardown that never masks the run, store rows the sweeper expires, and the
claim that replaces a sandbox a sweep is deleting instead of resuming it.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient
from test_subagents import _lifecycle

from langchain_digitalocean_microvm import (
    MicroVMLifecycle,
    TemplateCache,
    ThreadSandboxes,
    subagents,
)
from langchain_digitalocean_microvm.agent_host.store import GatewayStore
from langchain_digitalocean_microvm.agent_host.sweeper import apply_sweep, plan_sweep
from langchain_digitalocean_microvm.client import (
    MANAGED_TAG,
    THREAD_SANDBOX_TAG,
    id_tag,
)
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.status import MicroVMStatus

CREATED = datetime(2026, 10, 1, 22, 46, 36, tzinfo=UTC).timestamp()  # FakeMicroVMClient's created_at
DAY = 24 * 3600.0


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    """LocalTransport keys its directories by MicroVM id ("01fake0001"...), which repeat across tests."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def test_a_thread_gets_its_sandbox_created_then_resumed_and_paused_after_each_run() -> None:
    client = FakeMicroVMClient()
    threads = ThreadSandboxes(_lifecycle(client), prefix="sales")
    assert threads.sandbox_id("t1") == "sales-t1"

    with threads.sandbox("t1") as backend:
        backend.write("/outputs/report.md", "hi")
        first = backend.id
        assert client.get(first).status == MicroVMStatus.RUNNING
    assert client.get(first).status == MicroVMStatus.PAUSED
    assert set(client.get(first).tags) >= {MANAGED_TAG, THREAD_SANDBOX_TAG, id_tag("sales-t1")}

    with threads.sandbox("t1") as backend:  # the next run of the thread: same MicroVM, same files
        assert backend.id == first
        assert backend.read("/outputs/report.md").file_data["content"] == "hi"
    assert (client.create_calls, client.resume_calls, client.pause_calls) == (1, 1, 2)

    with threads.sandbox("t2") as other:
        assert other.id != first


def test_asandbox_runs_off_the_event_loop() -> None:
    client = FakeMicroVMClient(create_delay=0.3)
    threads = ThreadSandboxes(_lifecycle(client))

    async def main() -> float:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        async with threads.asandbox("t") as backend:
            sandbox = backend.id
        task.cancel()
        assert client.get(sandbox).status == MicroVMStatus.PAUSED
        return ticks

    assert asyncio.run(main()) >= 10  # the loop kept running during the 0.3 s create


def test_a_missing_thread_id_is_an_error() -> None:
    threads = ThreadSandboxes(_lifecycle(FakeMicroVMClient()))
    with pytest.raises(ValueError, match="thread id is required"), threads.sandbox(""):
        pytest.fail("must not enter the body")
    with pytest.raises(ValueError, match="at least 1"):
        ThreadSandboxes(_lifecycle(FakeMicroVMClient()), max_concurrent_creates=0)


def test_concurrent_creates_are_bounded_per_instance() -> None:
    client = FakeMicroVMClient(create_delay=0.2)
    threads = ThreadSandboxes(_lifecycle(client), max_concurrent_creates=2)

    def run(thread_id: str) -> str:
        with threads.sandbox(thread_id) as backend:
            return backend.id

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(run, [f"t{i}" for i in range(6)]))
    assert len(set(ids)) == 6
    assert client.peak_concurrent_creates == 2


def test_the_run_error_propagates_and_the_sandbox_is_still_paused() -> None:
    client = FakeMicroVMClient()
    threads = ThreadSandboxes(_lifecycle(client))
    with pytest.raises(RuntimeError, match="agent failed"), threads.sandbox("t") as backend:
        raise RuntimeError("agent failed")
    assert client.get(backend.id).status == MicroVMStatus.PAUSED


def test_a_failed_pause_is_logged_not_raised(caplog) -> None:
    client = FakeMicroVMClient()
    threads = ThreadSandboxes(_lifecycle(client))
    with threads.sandbox("t") as backend:
        client.delete(backend.id)  # gone behind our back; pause will fail
    assert "Could not pause the sandbox of thread t" in caplog.text


def test_new_threads_are_cloned_from_the_template_in_fake_mode() -> None:
    lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local"))
    cache = TemplateCache(lambda: lifecycle.build_template("assistant", seed_files=[("/AGENTS.md", b"# manual")]))
    threads = ThreadSandboxes(lifecycle, prefix="assistant", template=cache)

    with threads.sandbox("a") as backend:
        assert backend.read("/AGENTS.md").file_data["content"] == "# manual"
    lifecycle.delete_template(cache.get())  # the checkpoint is deleted while the server runs
    with threads.sandbox("b") as backend:  # rebuilt once, then cloned
        assert backend.read("/AGENTS.md").file_data["content"] == "# manual"
    assert set(lifecycle._local_states.values()) == {MicroVMStatus.PAUSED}


def test_the_store_records_each_run_and_the_sweeper_expires_unused_threads() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    store = GatewayStore(":memory:")
    now = [CREATED]
    threads = ThreadSandboxes(lifecycle, prefix="sales", store=store, clock=lambda: now[0])

    with threads.sandbox("old") as stale:
        (row,) = store.list_sandboxes()
        assert (row.sandbox_id, row.thread_id, row.subagent, row.status) == ("sales-old", "old", "", "active")
    assert store.get_sandbox("sales-old").status == "paused"
    now[0] = CREATED + 2 * DAY
    with threads.sandbox("new") as fresh:
        pass
    orphan = ThreadSandboxes(lifecycle, prefix="lost")  # no store: expired by age
    with orphan.sandbox("x") as lost:
        pass
    plain = lifecycle.get_or_create(sandbox_id="m2-demo")  # not a thread sandbox: never swept

    plan = plan_sweep(client, store=store, now=CREATED + 2 * DAY + 60, sandbox_ttl=DAY, orphan_ttl=DAY)
    assert {item.id for item in plan.delete_microvms} == {stale.id, lost.id}
    kept = {item.id: item.reason for item in plan.kept}
    assert kept[fresh.id].startswith("thread new, unused for")
    assert plain.id not in kept

    result = apply_sweep(plan, client, store=store)
    assert not result.errors and store.get_sandbox("sales-old").retired_reason == "gc"
    with threads.sandbox("old") as again:  # the thread comes back: a fresh sandbox, the row revived
        assert again.id != stale.id
    row = store.get_sandbox("sales-old")
    assert (row.microvm_id, row.status, row.retired_reason) == (again.id, "paused", None)


def test_a_sandbox_being_swept_is_replaced_not_resumed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subagents, "_RETIRE_POLL", 0.05)
    client = FakeMicroVMClient()
    store = GatewayStore(tmp_path / "gateway.sqlite")
    sweeper = GatewayStore(tmp_path / "gateway.sqlite")
    threads = ThreadSandboxes(_lifecycle(client), store=store, clock=lambda: CREATED + 2 * DAY)
    with threads.sandbox("t") as backend:
        old = backend.id
    assert sweeper.begin_retire_sandbox("thread-t", old, idle_before=CREATED + 3 * DAY, reason="gc")

    def sweeper_finishes() -> None:
        time.sleep(0.5)
        client.delete(old)
        sweeper.retire_sandbox("thread-t", "gc", microvm_id=old)

    worker = threading.Thread(target=sweeper_finishes)
    worker.start()
    with threads.sandbox("t") as backend:
        fresh = backend.id
    worker.join()

    assert fresh != old and client.resume_calls == 0
    assert store.get_sandbox("thread-t").microvm_id == fresh
    store.close()
    sweeper.close()
