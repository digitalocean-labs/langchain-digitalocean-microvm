"""Thread sandboxes: one MicroVM per conversation thread, for a per-thread graph factory.

A server (`langgraph dev`, a deployment) enters its graph factory for every run of a
thread. `ThreadSandboxes.asandbox(thread_id)` gets or creates that thread's sandbox
(cloned from a template the first time, resumed afterwards) and pauses it when the run
ends; a failed pause is logged, never raised, so it cannot mask the run's own outcome
(the MicroVM also pauses itself when idle).

    threads = ThreadSandboxes(lifecycle, prefix="sales", template=templates, store=store)

    @contextlib.asynccontextmanager
    async def make_graph(config, runtime):
        async with threads.asandbox(config["configurable"]["thread_id"]) as backend:
            yield create_deep_agent(..., backend=backend)

Create one instance per process: it bounds how many get-or-create calls (creates and
resumes) run at once, so a burst of new threads queues in the process instead of
reaching the API's rate limits and the team's MicroVM limit together.

Lifecycle: sandboxes carry `THREAD_SANDBOX_TAG`. With a `GatewayStore`, each run
claims the sandbox's row before get-or-create (a sweep that is deleting it is waited
out and a fresh sandbox created, as for subagent sandboxes) and records its use, so
scripts/gc_microvms.py deletes threads unused past `--sandbox-ttl`. Without a store
the sweeper deletes them by age (`--orphan-ttl`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import THREAD_SANDBOX_TAG
from langchain_digitalocean_microvm.provider import MicroVMLifecycle, SandboxTemplate
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.subagents import _claim_sandbox
from langchain_digitalocean_microvm.template_cache import TemplateCache

if TYPE_CHECKING:
    from langchain_digitalocean_microvm.agent_host.store import GatewayStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_CREATES = 4


class ThreadSandboxes:
    """One sandbox per thread id: get-or-create on enter, pause on exit."""

    def __init__(
        self,
        lifecycle: MicroVMLifecycle,
        *,
        prefix: str = "thread",
        template: SandboxTemplate | TemplateCache | None = None,
        max_concurrent_creates: int = DEFAULT_MAX_CONCURRENT_CREATES,
        store: GatewayStore | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """`prefix` names the sandboxes (`<prefix>-<thread id>`); `template` clones new ones (a
        `TemplateCache` is rebuilt once if its checkpoint was deleted); `max_concurrent_creates`
        bounds this instance's get-or-create calls in flight; `store` records use for the sweeper."""
        if max_concurrent_creates < 1:
            raise ValueError(f"max_concurrent_creates must be at least 1, got {max_concurrent_creates}")
        self.lifecycle = lifecycle
        self.prefix = prefix
        self.template = template
        self.store = store
        self.clock = clock
        self._creates = threading.BoundedSemaphore(max_concurrent_creates)

    def sandbox_id(self, thread_id: str) -> str:
        """The sandbox_id of `thread_id`'s sandbox (the MicroVM name is derived and hashed if long)."""
        if not thread_id:
            raise ValueError("a thread id is required (config['configurable']['thread_id'])")
        return f"{self.prefix}-{thread_id}"

    @contextlib.contextmanager
    def sandbox(self, thread_id: str) -> Iterator[MicroVMSandbox]:
        """The thread's sandbox for one run, paused on exit."""
        backend = self._get(thread_id)
        try:
            yield backend
        finally:
            self._pause(thread_id, backend)

    @contextlib.asynccontextmanager
    async def asandbox(self, thread_id: str) -> AsyncIterator[MicroVMSandbox]:
        """`sandbox` off the event loop, for a graph factory."""
        backend = await asyncio.to_thread(self._get, thread_id)
        try:
            yield backend
        finally:
            await asyncio.to_thread(self._pause, thread_id, backend)

    def _get(self, thread_id: str) -> MicroVMSandbox:
        sandbox_id = self.sandbox_id(thread_id)
        if self.store is not None:
            _claim_sandbox(self.lifecycle, self.store, sandbox_id, clock=self.clock)

        def create(template: SandboxTemplate | None) -> MicroVMSandbox:
            return self.lifecycle.get_or_create(sandbox_id=sandbox_id, tags=(THREAD_SANDBOX_TAG,), template=template)

        with self._creates:
            template = self.template
            backend = template.create(create) if isinstance(template, TemplateCache) else create(template)
        self._touch(thread_id, backend, "active")
        return backend

    def _pause(self, thread_id: str, backend: MicroVMSandbox) -> None:
        try:
            state = self.lifecycle.pause(backend)
            paused = state is None or state.status == MicroVMStatus.PAUSED  # None: local mode
            if not paused:
                logger.warning("The sandbox of thread %s is %s after pausing; it pauses itself when idle",
                               thread_id, state.status.value if state else "?")
            self._touch(thread_id, backend, "paused" if paused else "active")
        except Exception:  # noqa: BLE001 - teardown must not mask the run's own outcome
            logger.exception("Could not pause the sandbox of thread %s", thread_id)

    def _touch(self, thread_id: str, backend: MicroVMSandbox, status: str) -> None:
        if self.store is not None:
            self.store.touch_sandbox(self.sandbox_id(thread_id), microvm_id=backend.id, thread_id=thread_id,
                                     subagent="", status=status, now=self.clock())


__all__ = ["DEFAULT_MAX_CONCURRENT_CREATES", "THREAD_SANDBOX_TAG", "ThreadSandboxes"]
