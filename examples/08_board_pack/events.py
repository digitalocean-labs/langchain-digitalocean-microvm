"""Makes every MicroVM lifecycle step visible: an event log for the dashboard, a span for LangSmith.

`DemoLifecycle` is a `MicroVMLifecycle` that times each get-or-create (clone / create / resume /
attach) and pause, then

- appends one JSON line to outputs/events.jsonl, which dashboard.py tails (the only source of
  status in local mode; in real mode it adds timings to what the MicroVMs API reports), and
- records a LangSmith span (when tracing is on) inside the run that caused it, so a trace shows
  "MicroVM clone · analyst-europe · 1.4 s" right where the subagent first touched its sandbox.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from langchain_digitalocean_microvm import MicroVMLifecycle
from langchain_digitalocean_microvm.status import MicroVMStatus

EVENTS = Path(__file__).resolve().parent / "outputs" / "events.jsonl"
_lock = threading.Lock()


def log_event(**event: Any) -> None:
    """Append one timestamped JSON line; the lock keeps lines whole when several threads log at once."""
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.time(), **event})
    with _lock, EVENTS.open("a") as f:
        f.write(line + "\n")


def _span(name: str, inputs: dict[str, Any]):
    """A LangSmith span under the current run, or a no-op when tracing is off."""
    try:
        from langsmith import trace
        from langsmith.utils import tracing_is_enabled
    except ImportError:  # pragma: no cover
        tracing_is_enabled = None
    if tracing_is_enabled is None or not tracing_is_enabled():
        return contextlib.nullcontext(None)
    return trace(name=name, run_type="tool", inputs=inputs, tags=["microvm"])


class DemoLifecycle(MicroVMLifecycle):
    """MicroVMLifecycle with an event per lifecycle step (see the module docstring)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sandbox_ids: dict[str, str] = {}  # microvm_id -> sandbox_id, to log a pause by backend

    def _status_of(self, sandbox_id: str) -> MicroVMStatus | None:
        """The sandbox's current MicroVM status, or None if it does not exist (yet)."""
        if self.config.transport == "local":
            return self._local_states.get(sandbox_id)
        state = self._client_or_raise().get_by_name(self._name(sandbox_id))
        return state.status if state else None

    def get_or_create(self, *, sandbox_id: str, template=None, **kwargs: Any):  # type: ignore[override]
        # What get_or_create is about to do depends on the state it finds: none (or a dead VM) means a
        # new one (cloned if there is a template), paused means resume, running means attach.
        before = self._status_of(sandbox_id)
        if before is None or before in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED, MicroVMStatus.FAILED):
            action = "clone" if template is not None else "create"
        elif before == MicroVMStatus.PAUSED:
            action = "resume"
        else:
            action = "attach"
        log_event(event=f"{action}-start", sandbox_id=sandbox_id)
        started = time.monotonic()
        with _span(f"MicroVM {action}", {"sandbox_id": sandbox_id, "template": getattr(template, "checkpoint_name", None)}) as run:
            backend = super().get_or_create(sandbox_id=sandbox_id, template=template, **kwargs)
            seconds = round(time.monotonic() - started, 2)
            if run is not None:
                run.end(outputs={"microvm_id": backend.id, "seconds": seconds, "was": before.value if before else None})
        self._sandbox_ids[backend.id] = sandbox_id
        log_event(event=action, sandbox_id=sandbox_id, microvm_id=backend.id, seconds=seconds)
        return backend

    def pause(self, backend=None, *, sandbox_id: str | None = None):  # type: ignore[override]
        started = time.monotonic()
        state = super().pause(backend, sandbox_id=sandbox_id)
        microvm_id = backend.id if backend is not None else (state.id if state else sandbox_id)
        if sandbox_id is None and backend is not None:
            sandbox_id = self._sandbox_ids.get(backend.id, backend.id)
        log_event(event="pause", sandbox_id=sandbox_id, microvm_id=microvm_id,
                  seconds=round(time.monotonic() - started, 2))
        return state

    def delete(self, *, sandbox_id: str) -> None:
        super().delete(sandbox_id=sandbox_id)
        log_event(event="delete", sandbox_id=sandbox_id)
