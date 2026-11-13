"""A process-wide sandbox template that is rebuilt when its checkpoint disappears.

A server (`langgraph dev`, a deployment) serves many threads from one process, so a
template is built once (`MicroVMLifecycle.build_template`, or its checkpoint found) and
reused. Its checkpoint can be deleted behind the process's back (scripts/delete_all.py
--include-templates deletes every template this package made); the API then refuses the next clone with a
404 (`MicroVMNotFoundError`). `TemplateCache.create` catches that, calls `refresh(stale)`
and retries once. Under the lock, the template is rebuilt only if the cache still holds
the stale one, so threads that fail together cause one rebuild, and `build_template`
reuses the checkpoint if it is still there.

    templates = TemplateCache(lambda: lifecycle.build_template("analyst", setup=..., seed_files=...))
    backend = templates.create(lambda t: lifecycle.get_or_create(sandbox_id=sid, template=t))

Subagent sandboxes accept a cache in place of a template (`templates={name: cache}`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from langchain_digitalocean_microvm.exceptions import (
    MicroVMNotFoundError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.provider import SandboxTemplate

logger = logging.getLogger(__name__)

T = TypeVar("T")

# What a clone from a deleted checkpoint raises: the API's 404, or local mode's missing template.
DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (MicroVMNotFoundError, SandboxNotFoundError)


class TemplateCache:
    """One template per process, built on first use and rebuilt once when a clone from it fails."""

    def __init__(
        self,
        build: Callable[[], SandboxTemplate],
        *,
        retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_ON,
    ) -> None:
        """`build` makes the template (or finds its checkpoint); `retry_on`: the errors a refresh may fix."""
        self._build = build
        self._retry_on = retry_on
        self._template: SandboxTemplate | None = None
        self._lock = threading.Lock()

    def get(self) -> SandboxTemplate:
        """The cached template, built on first use. Blocking: call it off the event loop."""
        with self._lock:
            if self._template is None:
                self._template = self._build()
                logger.info("template %s %s", self._template.checkpoint_name,
                            "built" if self._template.built else "reused")
            return self._template

    def refresh(self, stale: SandboxTemplate) -> SandboxTemplate:
        """Rebuild after a clone from `stale` failed, unless another thread already replaced it."""
        with self._lock:
            if self._template is None or self._template is stale:
                self._template = self._build()
                logger.warning("template %s %s after a clone from it failed", self._template.checkpoint_name,
                               "rebuilt" if self._template.built else "found again")
            return self._template

    def create(self, create: Callable[[SandboxTemplate], T]) -> T:
        """`create(template)`; if it fails with a `retry_on` error, refresh and try once more (a second error is raised)."""
        template = self.get()
        try:
            return create(template)
        except self._retry_on as exc:
            logger.warning("creating from template %s failed (%s: %s); refreshing it",
                           template.checkpoint_name, type(exc).__name__, exc)
            return create(self.refresh(template))
