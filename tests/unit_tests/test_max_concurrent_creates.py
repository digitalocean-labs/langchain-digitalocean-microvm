"""`MicroVMConfig.max_concurrent_creates`: one bound on get-or-creates in flight, whoever calls them."""

from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from scripted_model import ScriptedChatModel

from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.provider import MicroVMLifecycle
from langchain_digitalocean_microvm.subagents import provision_subagent_sandboxes
from langchain_digitalocean_microvm.threads import ThreadSandboxes


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


class SlowLifecycle(MicroVMLifecycle):
    """Local transport; records how many creates run inside the gate at once."""

    def __init__(self, limit: int | None) -> None:
        super().__init__(MicroVMConfig(transport="local", max_concurrent_creates=limit))
        self.peak = self._inside = 0
        self._count = threading.Lock()

    def _local_get_or_create(self, **kwargs: Any) -> Any:
        with self._count:
            self._inside += 1
            self.peak = max(self.peak, self._inside)
        try:
            time.sleep(0.05)
            return super()._local_get_or_create(**kwargs)
        finally:
            with self._count:
                self._inside -= 1


def test_the_limit_is_validated_and_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MicroVMConfig().max_concurrent_creates is None
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_concurrent_creates"):
            MicroVMConfig(max_concurrent_creates=bad)
    monkeypatch.setenv("DO_MICROVM_MAX_CONCURRENT_CREATES", "3")
    assert MicroVMConfig.from_env().max_concurrent_creates == 3
    monkeypatch.setenv("DO_MICROVM_MAX_CONCURRENT_CREATES", "")
    assert MicroVMConfig.from_env().max_concurrent_creates is None


def test_without_a_limit_creates_run_at_once() -> None:
    lifecycle = SlowLifecycle(None)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: lifecycle.get_or_create(sandbox_id=f"s{i}"), range(8)))
    assert lifecycle.peak > 2


def test_direct_callers_share_the_bound() -> None:
    lifecycle = SlowLifecycle(2)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: lifecycle.get_or_create(sandbox_id=f"s{i}"), range(8)))
    assert lifecycle.peak == 2


def test_thread_sandboxes_and_subagent_provisioning_share_one_bound() -> None:
    """Callers with bounds of their own (4 each) still stay within the lifecycle's."""
    lifecycle = SlowLifecycle(2)
    threads = ThreadSandboxes(lifecycle, prefix="t", max_concurrent_creates=4)
    specs = [{"name": f"agent{i}", "description": "d", "system_prompt": "s", "permissions": []} for i in range(4)]
    model = ScriptedChatModel()

    def one_thread(i: int) -> None:
        with threads.sandbox(f"th{i}"):
            pass

    def one_turn(i: int) -> None:
        provision_subagent_sandboxes(lifecycle, specs, thread_id=f"turn{i}", default_model=model)

    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(one_thread, i) for i in range(4)] + [pool.submit(one_turn, i) for i in range(2)]
        for future in futures:
            future.result()
    assert lifecycle.peak == 2


def test_a_failed_create_releases_its_slot() -> None:
    lifecycle = SlowLifecycle(1)
    original = lifecycle._local_get_or_create

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("no")

    lifecycle._local_get_or_create = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        lifecycle.get_or_create(sandbox_id="a")
    lifecycle._local_get_or_create = original  # type: ignore[method-assign]
    assert lifecycle.get_or_create(sandbox_id="b").id  # would block forever if the slot leaked
