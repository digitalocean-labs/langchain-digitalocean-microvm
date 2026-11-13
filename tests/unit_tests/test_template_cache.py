"""TemplateCache survives a deleted checkpoint (local mode).

A `langgraph dev` server outlives a `scripts/delete_all.py --include-templates`, which deletes the template's
checkpoint: the next new thread's clone fails, the cache rebuilds the template once and the
thread is created from the new one. Threads that fail together cause a single rebuild; other
errors are not retried. Subagent sandboxes take a cache in place of a template (example 08).
"""

from __future__ import annotations

import tempfile
import threading

import pytest
from scripted_model import ScriptedChatModel

from langchain_digitalocean_microvm import MicroVMLifecycle, TemplateCache
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import (
    MicroVMRateLimitError,
    SandboxError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.subagents import provision_subagent_sandboxes

SEEDS = [("/AGENTS.md", b"# manual"), ("/data/table.csv", b"a,b\n1,2\n")]


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


class Counting:
    """build_template as the example calls it, counting the calls that really built."""

    def __init__(self, lifecycle: MicroVMLifecycle) -> None:
        self.lifecycle = lifecycle
        self.calls = 0
        self.built = 0

    def __call__(self):
        self.calls += 1
        template = self.lifecycle.build_template("sales-assistant", seed_files=SEEDS)
        self.built += template.built
        return template


def _setup() -> tuple[MicroVMLifecycle, Counting, TemplateCache]:
    lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local"))
    build = Counting(lifecycle)
    return lifecycle, build, TemplateCache(build)


def _thread(lifecycle: MicroVMLifecycle, cache: TemplateCache, thread_id: str):
    return cache.create(lambda template: lifecycle.get_or_create(sandbox_id=f"sales-{thread_id}", template=template))


def test_a_new_thread_after_the_checkpoint_was_deleted_rebuilds_once() -> None:
    lifecycle, build, cache = _setup()
    a = _thread(lifecycle, cache, "a")
    assert a.read("/AGENTS.md").file_data["content"] == "# manual"
    assert (build.calls, build.built) == (1, 1)

    lifecycle.delete_template(cache.get())  # what delete_all.py does to the checkpoint
    b = _thread(lifecycle, cache, "b")
    assert b.read("/data/table.csv").file_data["content"].startswith("a,b\n1,2")
    assert (build.calls, build.built) == (2, 2)

    _thread(lifecycle, cache, "c")  # the rebuilt template is cached
    assert _thread(lifecycle, cache, "a").read("/AGENTS.md").file_data["content"] == "# manual"  # a resume
    assert build.calls == 2


def test_a_refresh_reuses_a_checkpoint_that_still_exists() -> None:
    lifecycle, build, cache = _setup()
    stale = cache.get()
    assert cache.refresh(stale).checkpoint_id == stale.checkpoint_id
    assert (build.calls, build.built) == (2, 1)


def test_a_second_failure_is_raised() -> None:
    lifecycle, build, cache = _setup()
    cache.get()
    failures = []

    def always_fails(template):
        failures.append(template.checkpoint_id)
        raise SandboxNotFoundError("checkpoint not found")

    with pytest.raises(SandboxNotFoundError):
        cache.create(always_fails)
    assert len(failures) == 2 and build.calls == 2


@pytest.mark.parametrize("error", [MicroVMRateLimitError(429, "slow down"), SandboxError("not ready")])
def test_errors_a_rebuild_cannot_fix_are_raised_at_once(error) -> None:
    lifecycle, build, cache = _setup()

    def fails(template):
        raise error

    with pytest.raises(type(error)):
        cache.create(fails)
    assert build.calls == 1


def test_on_demand_subagent_sandboxes_rebuild_through_the_cache() -> None:
    lifecycle, build, cache = _setup()
    cache.get()
    lifecycle.delete_template(cache.get())
    specs = [{"name": "analyst", "description": "Analyses.", "system_prompt": "You analyse."}]
    sandboxes = provision_subagent_sandboxes(lifecycle, specs, thread_id="t", default_model=ScriptedChatModel(),
                                             templates={"analyst": cache}, on_demand=True)
    assert build.calls == 1  # nothing provisioned yet
    assert sandboxes.backends["analyst"].read("/AGENTS.md").file_data["content"] == "# manual"
    assert (build.calls, build.built) == (2, 2)


def test_two_concurrent_new_threads_cause_one_rebuild() -> None:
    lifecycle, build, cache = _setup()
    _thread(lifecycle, cache, "a")
    lifecycle.delete_template(cache.get())
    both_hold_the_stale_template = threading.Barrier(2)
    errors: list[BaseException] = []

    def new_thread(thread_id: str) -> None:
        def create(template):
            if template.built and build.calls == 1:  # the first attempt, from the stale template
                both_hold_the_stale_template.wait(timeout=10)
            return lifecycle.get_or_create(sandbox_id=f"sales-{thread_id}", template=template)

        try:
            backend = cache.create(create)
            assert backend.read("/AGENTS.md").file_data["content"] == "# manual"
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    workers = [threading.Thread(target=new_thread, args=(tid,)) for tid in ("b", "c")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert errors == []
    assert (build.calls, build.built) == (2, 2)
