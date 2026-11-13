"""Subagent sandboxes: every sync subagent keeps its LLM loop in the supervisor but gets its own MicroVM.

Design: see docs/design.md. deepagents 0.7.0 builds
each `SubAgent` spec's middleware stack from the PARENT's backend in three
places (`graph.py:668-678`): `FilesystemMiddleware` (file tools + `execute`,
with the spec's `permissions`; `_PERMISSIONS_KWARG` below), the
summarization middleware (evicted history goes to
`/conversation_history/<thread>.md` on that backend) and, when the spec has
`skills`, `SkillsMiddleware`. `_apply_custom_middleware` replaces a base
middleware in place when the spec's own `middleware` list has one with the
same `.name` (the class name), so `per_subagent_specs` rebuilds those three
against the subagent's MicroVMSandbox and the subagent never touches the
parent's files. `tests/test_subagents.py` pins these names (contract
test): an upgrade that changes them fails there, not silently at runtime.

Three layers, use whichever fits:

- `per_subagent_specs(specs, backends=..., default_model=...)`: pure spec
  rewrite, for callers that already hold one backend per subagent.
- `provision_subagent_sandboxes` / `pause_subagent_sandboxes`: get-or-create
  one sandbox per subagent for a thread, in parallel on a bounded thread pool
  (each create/resume blocks on the MicroVMs API), and pause them after the
  turn. Async twins run the same code via `asyncio.to_thread`, so a graph
  factory never blocks its event loop.
- `asubagent_sandboxes(...)` / `subagent_sandboxes(...)`: context managers
  for a per-thread graph factory (`make_graph(config, runtime)`): provision on
  enter, pause on exit. MicroVMSandbox instances are rebuilt every turn (the
  transport is bound at build), which the factory pattern does naturally.

Two options shape provisioning:

- `templates={name: SandboxTemplate}`: a subagent's sandbox is cloned from a
  prepared checkpoint (`MicroVMLifecycle.build_template`) instead of booted
  from the image and seeded: its packages, tools and reference data are there
  from the first second, and every clone starts from the same known state.
  A `TemplateCache` in place of the template is built on first use and
  rebuilt once if its checkpoint was deleted (the clone is then retried).
- `on_demand=True`: nothing is created or resumed when the turn starts. Each
  subagent gets an `OnDemandSandbox` that gets or creates its MicroVM on its
  first file operation or command, so a turn only wakes the subagents it
  actually delegates to, and subagents the supervisor calls in parallel start
  their MicroVMs in parallel. Teardown pauses only the sandboxes that started.

- `per_task={name: PerTask(...)}`: that subagent gets one sandbox per `task`
  call (or per key) instead of one per thread. Its spec gets a `PerTaskSandbox`
  that resolves the running task's sandbox on every call; the parent agent
  needs `sb.middleware` (a `TaskKeyMiddleware`), which sets the running task,
  applies `max_sessions` / `max_concurrent`, and deletes or pauses the sandbox
  when the task returns.

The parent sees each subagent's files through `SubagentSandboxes.mount()`, a
`CompositeBackend` routing `/subagents/<name>/` to that subagent's sandbox.
CompositeBackend sends `execute` only to its default backend, so the bridge is
files-only and commands stay on the parent's backend.

Budget: with the default endpoint transport, sandbox commands and files go to
each sandbox's endpoint and exec is used once per new sandbox. With
DO_MICROVM_TRANSPORT=exec, N subagents share the team's exec limits
(60 commands/min, 32 concurrent) and every sandbox file operation is at least
one exec call; the client backs off on 429, so a chatty fan-out slows down
rather than fails. Paused MicroVMs count against the team's MicroVM limit.

Lifecycle: sandboxes carry `SUBAGENT_SANDBOX_TAG`, so scripts/delete_all.py
(every MicroVM this package created) and scripts/gc_microvms.py find them.
Pass a `GatewayStore` and each provision/pause records the sandbox's last use
in its `sandboxes` table; the sweeper deletes sandboxes unused past its
`--sandbox-ttl` (default 24 h), and ones no store knows by age. Provisioning
claims the row first (`GatewayStore.claim_sandbox`): a sweep that has already
started deleting the sandbox is waited out (`RETIRE_WAIT` seconds, then the
deletion is finished here) and a fresh one is created.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphInterrupt

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import (
    PER_TASK_SANDBOX_TAG,
    SUBAGENT_SANDBOX_TAG,
)
from langchain_digitalocean_microvm.exceptions import SandboxError, SandboxNotFoundError
from langchain_digitalocean_microvm.provider import MicroVMLifecycle, SandboxTemplate
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.template_cache import TemplateCache

# FilesystemMiddleware takes the rules as `permissions` since deepagents 0.7.31; before, only as the private
# `_permissions` (deprecated from 0.7.31, to be removed in 0.9.0). Both are stored as `._permissions`.
_PERMISSIONS_KWARG = (
    "permissions" if "permissions" in inspect.signature(FilesystemMiddleware.__init__).parameters else "_permissions"
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from langchain_digitalocean_microvm.agent_host.store import GatewayStore

logger = logging.getLogger(__name__)

DEFAULT_MOUNT_PREFIX = "/subagents/"
# Creates/resumes run in parallel up to this many at once per provisioning
# call (eager mode; on-demand sandboxes start when their subagent first acts).
DEFAULT_MAX_WORKERS = 4
# How long provisioning waits for another process to finish deleting a sandbox it claims.
RETIRE_WAIT = 60.0
_RETIRE_POLL = 0.5

SeedFiles = Mapping[str, Sequence[tuple[str, bytes]]]
Templates = Mapping[str, "SandboxTemplate | TemplateCache"]


def subagent_sandbox_id(thread_id: str, name: str) -> str:
    """The sandbox_id for subagent `name` in `thread_id` (the MicroVM name is derived and hashed if long)."""
    return f"{thread_id}-{name}"


def is_sync_subagent(spec: Mapping[str, Any]) -> bool:
    """A plain `SubAgent` (no `runnable`, no `graph_id`): the only kind whose stack deepagents builds."""
    return "runnable" not in spec and "graph_id" not in spec


def per_subagent_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    backends: Mapping[str, BackendProtocol],
    default_model: str | BaseChatModel,
    parent_permissions: list[Any] | None = None,
) -> list[Any]:
    """Rewrite `specs` so each subagent named in `backends` uses that backend instead of the parent's.

    For each such spec, the spec's `middleware` gains a `FilesystemMiddleware`
    (with the spec's `permissions`, else `parent_permissions`, as deepagents
    resolves them, and the model's harness-profile tool descriptions), the
    summarization middleware and, if the spec has `skills`, a
    `SkillsMiddleware`, all on the subagent's backend. deepagents swaps them
    in for its defaults by name. Middleware already in the spec comes after
    and wins on a name clash. Specs not in `backends`, compiled subagents and
    async subagents pass through unchanged (they keep the parent's backend).
    `default_model` is what deepagents gives a spec without `model` (the
    supervisor's model).

    deepagents 0.7.0 refuses permission rules on a backend that can
    `execute` (a MicroVMSandbox), so a subagent that has or inherits rules
    raises ValueError here; `permissions: []` on the spec opts out.
    """
    out: list[Any] = []
    for spec in specs:
        backend = backends.get(spec["name"]) if is_sync_subagent(spec) else None
        if backend is None:
            out.append(spec)
            continue
        raw_model = spec.get("model", default_model)
        model = _resolve_model(raw_model)
        try:
            filesystem = FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=_tool_description_overrides(model, raw_model),
                **{_PERMISSIONS_KWARG: spec.get("permissions", parent_permissions)},
            )
        except NotImplementedError as exc:  # deepagents 0.7.0: no permissions on a backend that executes
            raise ValueError(
                f"Subagent {spec['name']!r}: deepagents cannot enforce filesystem permissions on a sandbox "
                "(its `execute` would bypass them), and the spec has or inherits some. The MicroVM is the "
                "boundary: set `permissions: []` on the spec to run it unrestricted inside its VM."
            ) from exc
        middleware: list[Any] = [filesystem, create_summarization_middleware(model, backend)]
        if spec.get("skills"):
            middleware.append(SkillsMiddleware(backend=backend, sources=spec["skills"]))
        out.append({**spec, "middleware": [*middleware, *spec.get("middleware", [])]})
    return out


class OnDemandSandbox(BaseSandbox):
    """A subagent's sandbox whose MicroVM is gotten or created on first use, not when the turn starts.

    `provision` (a blocking call returning a connected MicroVMSandbox, i.e.
    `MicroVMLifecycle.get_or_create`) runs once, on the first command or
    file operation; concurrent first uses wait for the same call. The async
    paths run it off the event loop. A provisioning failure is returned to the
    agent as a failed command / per-file error (the next use retries), so a
    subagent whose VM cannot start reports it instead of crashing the run.
    """

    def __init__(self, *, name: str, sandbox_id: str, provision: Callable[[], MicroVMSandbox]) -> None:
        self.name = name
        self.sandbox_id = sandbox_id
        self._provision = provision
        self._backend: MicroVMSandbox | None = None
        self._lock = threading.Lock()
        self.enable_capture_offload = False  # decided by the MicroVMSandbox, once it exists

    @property
    def started(self) -> MicroVMSandbox | None:
        """The connected sandbox if this turn has used it, else None."""
        return self._backend

    @property
    def id(self) -> str:
        return self._backend.id if self._backend is not None else self.sandbox_id

    def get(self) -> MicroVMSandbox:
        """The sandbox, getting or creating its MicroVM now if this is the first use (blocking)."""
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = self._provision()
                    self.enable_capture_offload = self._backend.enable_capture_offload
        return self._backend

    async def aget(self) -> MicroVMSandbox:
        return self._backend if self._backend is not None else await asyncio.to_thread(self.get)

    def _failure(self, exc: Exception) -> str:
        logger.warning("The sandbox of subagent %r could not be started: %s", self.name, exc)
        return f"Error: the sandbox of subagent {self.name!r} could not be started: {exc}"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            backend = self.get()
        except Exception as exc:  # noqa: BLE001 - reported to the agent, retried on the next use
            return ExecuteResponse(output=self._failure(exc), exit_code=1, truncated=False)
        return backend.execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        try:
            backend = await self.aget()
        except Exception as exc:  # noqa: BLE001
            return ExecuteResponse(output=self._failure(exc), exit_code=1, truncated=False)
        return await backend.aexecute(command, timeout=timeout)

    def execute_with_offload(self, command: str, capture_path: str, **kwargs: Any) -> Any:
        try:
            backend = self.get()
        except Exception:  # noqa: BLE001 - execute() reports it
            return super().execute_with_offload(command, capture_path, **kwargs)
        return backend.execute_with_offload(command, capture_path, **kwargs)

    async def aexecute_with_offload(self, command: str, capture_path: str, **kwargs: Any) -> Any:
        try:
            backend = await self.aget()
        except Exception:  # noqa: BLE001
            return await super().aexecute_with_offload(command, capture_path, **kwargs)
        return await backend.aexecute_with_offload(command, capture_path, **kwargs)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        try:
            backend = self.get()
        except Exception as exc:  # noqa: BLE001
            error = self._failure(exc)
            return [FileUploadResponse(path=path, error=error) for path, _ in files]
        return backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        try:
            backend = self.get()
        except Exception as exc:  # noqa: BLE001
            error = self._failure(exc)
            return [FileDownloadResponse(path=path, content=None, error=error) for path in paths]
        return backend.download_files(paths)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()


@dataclass(frozen=True)
class TaskCall:
    """A `task` tool call, as `PerTask.key` sees it."""

    id: str  # the tool call id
    subagent_type: str
    description: str
    args: Mapping[str, Any]  # the call's arguments, as the model sent them


@dataclass(frozen=True)
class PerTask:
    """How a subagent gets one sandbox per `task` call instead of one per thread.

    - `key`: maps a call to the key of its sandbox; calls with the same key in
      a thread use the same sandbox (one at a time). Default: the tool call id,
      i.e. one sandbox per call. A key function that raises or returns an empty
      key refuses the call with that message.
    - `max_sessions`: sandboxes of this subagent the thread may hold at once,
      running or paused (paused ones count against the team's quota). A call
      that would start one more is refused with a tool message the model reads.
      Counting across turns uses the store's records.
    - `max_concurrent`: tasks of this subagent that run at once; further calls
      wait.
    - `when_done`: what happens to the sandbox when its task returns:
      `"delete"` (frees its quota at once) or `"pause"` (kept, resumed by the
      next task with the same key, in this turn or a later one). A task
      interrupted inside the subagent (an approval) is always paused, so its
      replay on resume finds the same sandbox.
    """

    key: Callable[[TaskCall], str] | None = None
    max_sessions: int | None = 8
    max_concurrent: int = 4
    when_done: str = "delete"

    def __post_init__(self) -> None:
        if self.when_done not in ("delete", "pause"):
            raise ValueError(f"when_done must be 'delete' or 'pause', got {self.when_done!r}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be at least 1, got {self.max_concurrent}")
        if self.max_sessions is not None and self.max_sessions < 1:
            raise ValueError(f"max_sessions must be at least 1 or None, got {self.max_sessions}")

_LIVE_ROW_STATUSES = frozenset({"active", "paused"})
_POLL = 0.02  # async waits on a per-key lock / the concurrency gate


@dataclass(frozen=True)
class _ActiveTask:
    subagent: str
    key: str
    sandbox: OnDemandSandbox


# Set by TaskKeyMiddleware around a per-task `task` call; the subagent runs inside that call, so its tools see it.
_ACTIVE_TASK: contextvars.ContextVar[_ActiveTask | None] = contextvars.ContextVar(
    "microvm_active_task", default=None
)


class PerTaskSandbox(BaseSandbox):
    """The backend of a per-task subagent: every call goes to the sandbox of the task that is running.

    deepagents builds a subagent's file tools, `execute`, summarization and
    skills against one backend, once; this one resolves the sandbox on every
    call, from the task the parent's `TaskKeyMiddleware` is running. Custom
    tools take it as their backend like any other, or call `current()` /
    `acurrent()` for the task's connected MicroVMSandbox. Used outside a task
    of its subagent, it raises SandboxError: never a silently shared sandbox.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def _active(self) -> _ActiveTask:
        task = _ACTIVE_TASK.get()
        if task is None or task.subagent != self.name:
            raise SandboxError(
                f"The per-task sandbox of subagent {self.name!r} was used outside a task of that subagent: "
                "it has a sandbox only while the parent's `task` tool runs it (add `sb.middleware` to the agent)."
            )
        return task

    def _task(self) -> OnDemandSandbox:
        return self._active().sandbox

    @property
    def key(self) -> str:
        """The running task's key."""
        return self._active().key

    def current(self) -> MicroVMSandbox:
        """The running task's sandbox, started (created, cloned or resumed) now if this is its first use (blocking)."""
        return self._task().get()

    async def acurrent(self) -> MicroVMSandbox:
        return await self._task().aget()

    @property
    def id(self) -> str:
        task = _ACTIVE_TASK.get()
        if task is None or task.subagent != self.name:
            return f"per-task:{self.name}"
        return task.sandbox.id

    @property
    def enable_capture_offload(self) -> bool:  # type: ignore[override]
        task = _ACTIVE_TASK.get()
        return bool(task is not None and task.subagent == self.name and task.sandbox.enable_capture_offload)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._task().execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        return await self._task().aexecute(command, timeout=timeout)

    def execute_with_offload(self, command: str, capture_path: str, **kwargs: Any) -> Any:
        return self._task().execute_with_offload(command, capture_path, **kwargs)

    async def aexecute_with_offload(self, command: str, capture_path: str, **kwargs: Any) -> Any:
        return await self._task().aexecute_with_offload(command, capture_path, **kwargs)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._task().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._task().download_files(paths)


class _Refused(Exception):
    """A `task` call refused before it runs; the message goes to the model."""


class _PerTaskPool:
    """One per-task subagent's sandboxes in one thread, for one turn: admission, caps, locks, end-of-task rules."""

    def __init__(
        self,
        *,
        name: str,
        rules: PerTask,
        thread_id: str,
        lifecycle: MicroVMLifecycle,
        store: GatewayStore | None,
        provision: Callable[[str], MicroVMSandbox],
        clock: Callable[[], float],
    ) -> None:
        self.name, self.rules, self.thread_id = name, rules, thread_id
        self._lifecycle, self._store, self._provision, self._clock = lifecycle, store, provision, clock
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._gate = threading.BoundedSemaphore(rules.max_concurrent)
        self._sandboxes: dict[str, OnDemandSandbox] = {}  # key -> this turn's sandbox, while it may be live
        self._held: set[str] = set()  # sandbox_ids admitted this turn and not deleted

    def sandbox_id(self, key: str) -> str:
        return subagent_sandbox_id(self.thread_id, f"{self.name}-{key}")

    def started(self) -> list[tuple[str, MicroVMSandbox]]:
        """(sandbox_id, sandbox) of the sandboxes this turn started that are still live."""
        with self._lock:
            sandboxes = list(self._sandboxes.values())
        return [(s.sandbox_id, s.started) for s in sandboxes if s.started is not None]

    # -- admission ------------------------------------------------------------

    def key_for(self, call: TaskCall) -> str:
        try:
            key = call.id if self.rules.key is None else self.rules.key(call)
        except Exception as exc:  # noqa: BLE001 - refused, the model reads why
            raise _Refused(f"the task key could not be computed: {exc}") from exc
        if not isinstance(key, str) or not key.strip():
            raise _Refused(f"the task key is empty (got {key!r})")
        return key.strip()

    def _held_in_store(self) -> set[str]:
        if self._store is None:
            return set()
        prefix = f"{self.thread_id}-{self.name}-"
        return {
            row.sandbox_id for row in self._store.list_sandboxes()
            if row.thread_id == self.thread_id and row.subagent == self.name
            and row.status in _LIVE_ROW_STATUSES and row.sandbox_id.startswith(prefix)
        }

    def admit(self, key: str) -> OnDemandSandbox:
        """Count the key's sandbox against `max_sessions` (once) and return its sandbox for this turn."""
        sandbox_id = self.sandbox_id(key)
        with self._lock:
            held = self._held | self._held_in_store()
            limit = self.rules.max_sessions
            if sandbox_id not in held and limit is not None and len(held) >= limit:
                raise _Refused(
                    f"subagent {self.name!r} already holds {len(held)} sandboxes in this thread (the limit is "
                    f"{limit}); wait for a running task to finish, or do this work in a task that already ran"
                )
            self._held.add(sandbox_id)
            sandbox = self._sandboxes.get(key)
            if sandbox is None:
                sandbox = OnDemandSandbox(
                    name=self.name, sandbox_id=sandbox_id, provision=functools.partial(self._provision, sandbox_id),
                )
                self._sandboxes[key] = sandbox
            return sandbox

    def key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    @property
    def gate(self) -> threading.BoundedSemaphore:
        return self._gate

    # -- end of a task --------------------------------------------------------

    def finish(self, key: str, *, interrupted: bool) -> None:
        """Delete or pause the task's sandbox (never raises: the task's own result must reach the parent)."""
        with self._lock:
            sandbox = self._sandboxes.pop(key, None)
        if sandbox is None:
            return
        mode = "pause" if interrupted else self.rules.when_done
        try:
            if mode == "delete":
                self._delete(sandbox)
            else:
                self._pause(sandbox)
        except Exception as exc:  # noqa: BLE001 - the sandbox's idle timeout / the sweeper catch what stays
            logger.warning("Could not %s the sandbox of subagent %r task %r (thread %s): %s",
                           mode, self.name, key, self.thread_id, exc)

    def _delete(self, sandbox: OnDemandSandbox) -> None:
        live = sandbox.started
        with self._lock:
            known = sandbox.sandbox_id in self._held_in_store()
        if live is not None:
            live.close()
        if live is not None or known:  # `known`: paused in an earlier turn, resumed but not used this time
            with contextlib.suppress(SandboxNotFoundError):
                self._lifecycle.delete(sandbox_id=sandbox.sandbox_id)
            if self._store is not None:
                self._store.retire_sandbox(sandbox.sandbox_id, "deleted")
        with self._lock:
            self._held.discard(sandbox.sandbox_id)

    def _pause(self, sandbox: OnDemandSandbox) -> None:
        live = sandbox.started
        if live is None:
            return
        state = self._lifecycle.pause(live)
        paused = state is None or state.status == MicroVMStatus.PAUSED
        if self._store is not None:
            self._store.touch_sandbox(
                sandbox.sandbox_id, microvm_id=live.id, thread_id=self.thread_id, subagent=self.name,
                status="paused" if paused else "active", now=self._clock(),
            )


def _refusal(call: Mapping[str, Any], message: str) -> ToolMessage:
    return ToolMessage(content=f"Error: task refused: {message}", tool_call_id=call["id"], name="task",
                       status="error")


class TaskKeyMiddleware(AgentMiddleware):
    """The parent's middleware for per-task subagents: runs each of their `task` calls in its own sandbox.

    Around a `task` call for a per-task subagent it computes the key, applies
    `max_sessions` (refusal) and `max_concurrent` (wait), holds the key's lock
    (two tasks never share a live sandbox), sets the task for the subagent's
    `PerTaskSandbox`, and on return deletes or pauses the sandbox. Other tool
    calls and other subagent types pass through untouched.
    """

    def __init__(self, pools: Mapping[str, _PerTaskPool]) -> None:
        super().__init__()
        self._pools = dict(pools)

    def _route(self, request: Any) -> tuple[_PerTaskPool, TaskCall] | None:
        call = request.tool_call
        if call.get("name") != "task":
            return None
        args = call.get("args") or {}
        pool = self._pools.get(args.get("subagent_type", ""))
        if pool is None:
            return None
        return pool, TaskCall(id=call["id"], subagent_type=args["subagent_type"],
                              description=str(args.get("description", "")), args=MappingProxyType(dict(args)))

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        routed = self._route(request)
        if routed is None:
            return handler(request)
        pool, call = routed
        try:
            key = pool.key_for(call)
        except _Refused as exc:
            return _refusal(request.tool_call, str(exc))
        with pool.key_lock(key):
            try:
                sandbox = pool.admit(key)
            except _Refused as exc:
                return _refusal(request.tool_call, str(exc))
            with pool.gate:
                token = _ACTIVE_TASK.set(_ActiveTask(pool.name, key, sandbox))
                interrupted = False
                try:
                    return handler(request)
                except GraphInterrupt:
                    interrupted = True
                    raise
                finally:
                    _ACTIVE_TASK.reset(token)
                    pool.finish(key, interrupted=interrupted)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        routed = self._route(request)
        if routed is None:
            return await handler(request)
        pool, call = routed
        try:
            key = pool.key_for(call)
        except _Refused as exc:
            return _refusal(request.tool_call, str(exc))
        lock = pool.key_lock(key)
        await _acquire(lock)
        try:
            try:
                sandbox = await asyncio.to_thread(pool.admit, key)
            except _Refused as exc:
                return _refusal(request.tool_call, str(exc))
            await _acquire(pool.gate)
            try:
                token = _ACTIVE_TASK.set(_ActiveTask(pool.name, key, sandbox))
                interrupted = False
                try:
                    return await handler(request)
                except GraphInterrupt:
                    interrupted = True
                    raise
                finally:
                    _ACTIVE_TASK.reset(token)
                    await asyncio.to_thread(pool.finish, key, interrupted=interrupted)
            finally:
                pool.gate.release()
        finally:
            lock.release()


async def _acquire(lock: threading.Lock | threading.BoundedSemaphore) -> None:
    """Acquire a threading lock without blocking the event loop (and without leaking it on cancellation)."""
    while not lock.acquire(blocking=False):
        await asyncio.sleep(_POLL)


@dataclass(frozen=True)
class SubagentSandboxes:
    """One thread's subagent sandboxes, as provisioned for a turn."""

    thread_id: str
    backends: Mapping[str, MicroVMSandbox | OnDemandSandbox]  # subagent name -> its sandbox (fixed subagents)
    sandbox_ids: Mapping[str, str]  # subagent name -> sandbox_id (fixed subagents)
    subagents: tuple[Any, ...] = ()  # the specs rewritten by per_subagent_specs
    per_task: Mapping[str, PerTaskSandbox] = field(default_factory=lambda: MappingProxyType({}))
    middleware: tuple[Any, ...] = ()  # for the parent agent: a TaskKeyMiddleware when `per_task` is set
    _store: GatewayStore | None = field(default=None, repr=False, compare=False)
    _pools: Mapping[str, _PerTaskPool] = field(default_factory=lambda: MappingProxyType({}), repr=False,
                                               compare=False)

    def started(self) -> dict[str, MicroVMSandbox]:
        """The fixed sandboxes that exist this turn: all of them, or the on-demand ones used so far."""
        out: dict[str, MicroVMSandbox] = {}
        for name, backend in self.backends.items():
            live = backend.started if isinstance(backend, OnDemandSandbox) else backend
            if live is not None:
                out[name] = live
        return out

    def started_per_task(self) -> dict[str, dict[str, MicroVMSandbox]]:
        """`{subagent: {sandbox_id: sandbox}}`: the per-task sandboxes this turn started that are still live."""
        return {name: dict(pool.started()) for name, pool in self._pools.items()}

    def _live(self) -> list[tuple[str, str, MicroVMSandbox]]:
        """(subagent, sandbox_id, sandbox) of every live sandbox of this turn, fixed and per-task."""
        out = [(name, self.sandbox_ids[name], backend) for name, backend in self.started().items()]
        for name, pool in self._pools.items():
            out.extend((name, sandbox_id, backend) for sandbox_id, backend in pool.started())
        return out

    def routes(self, prefix: str = DEFAULT_MOUNT_PREFIX) -> dict[str, BackendProtocol]:
        """`{"/subagents/<name>/": sandbox}` for a CompositeBackend."""
        base = "/" + prefix.strip("/") + "/"
        return {f"{base}{name}/": backend for name, backend in self.backends.items()}

    def mount(self, parent_backend: BackendProtocol, prefix: str = DEFAULT_MOUNT_PREFIX) -> CompositeBackend:
        """The parent's backend with each subagent's files under `<prefix><name>/` (files only)."""
        return CompositeBackend(default=parent_backend, routes=self.routes(prefix))


def provision_subagent_sandboxes(
    lifecycle: MicroVMLifecycle,
    specs: Sequence[Mapping[str, Any]],
    *,
    thread_id: str,
    default_model: str | BaseChatModel,
    parent_permissions: list[Any] | None = None,
    seed_files: SeedFiles | None = None,
    templates: Templates | None = None,
    on_demand: bool = False,
    per_task: Mapping[str, PerTask] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    store: GatewayStore | None = None,
    clock: Callable[[], float] = time.time,
) -> SubagentSandboxes:
    """Get or create one sandbox per sync subagent in `specs` for `thread_id`, in parallel.

    Paused sandboxes are resumed; `seed_files[name]` (skills, memory, inputs)
    is uploaded only into a sandbox this call created. `templates[name]`
    clones a created sandbox from that template (seeds are applied on top),
    or from a `TemplateCache`'s, rebuilt once if the clone finds it deleted.
    If any sandbox fails, the ones that succeeded are paused again and the
    first error is raised.

    With `on_demand=True` nothing is provisioned here: each subagent gets an
    `OnDemandSandbox` that does the same get-or-create on its first use.

    `per_task[name]` gives that subagent one sandbox per `task` call (or per
    `PerTask.key`) instead of one per thread: its spec gets a `PerTaskSandbox`,
    and the parent agent needs `sb.middleware`. Each task's sandbox starts on
    its first file operation or command (seeds and template as above), under
    the sandbox_id `subagent_sandbox_id(thread_id, f"{name}-{key}")`.
    """
    names = [spec["name"] for spec in specs if is_sync_subagent(spec)]
    if len(set(names)) != len(names):
        raise ValueError(f"subagent names must be unique, got {names}")
    unknown = set(templates or {}) - set(names)
    if unknown:
        raise ValueError(f"templates for unknown subagents: {sorted(unknown)}")
    per_task = dict(per_task or {})
    unknown = set(per_task) - set(names)
    if unknown:
        raise ValueError(f"per_task for unknown (or not sync) subagents: {sorted(unknown)}")
    fixed = [name for name in names if name not in per_task]
    sandbox_ids = {name: subagent_sandbox_id(thread_id, name) for name in fixed}
    seeds = seed_files or {}
    templates = templates or {}

    def provision(name: str, sandbox_id: str, tags: tuple[str, ...]) -> MicroVMSandbox:
        seed = seeds.get(name)
        if store is not None:
            _claim_sandbox(lifecycle, store, sandbox_id, clock=clock)
        template = templates.get(name)

        def create(template: SandboxTemplate | None) -> MicroVMSandbox:
            return lifecycle.get_or_create(
                sandbox_id=sandbox_id,
                seed_files=list(seed) if seed else None,
                tags=tags,
                template=template,
            )

        # A cache rebuilds a template whose checkpoint was deleted and retries the clone once.
        backend = template.create(create) if isinstance(template, TemplateCache) else create(template)
        if store is not None:
            store.touch_sandbox(
                sandbox_id, microvm_id=backend.id, thread_id=thread_id, subagent=name,
                status="active", now=clock(),
            )
        return backend

    def one(name: str) -> MicroVMSandbox:
        return provision(name, sandbox_ids[name], (SUBAGENT_SANDBOX_TAG,))

    pools = {
        name: _PerTaskPool(
            name=name, rules=rules, thread_id=thread_id, lifecycle=lifecycle, store=store, clock=clock,
            provision=functools.partial(
                lambda n, sandbox_id: provision(n, sandbox_id, (SUBAGENT_SANDBOX_TAG, PER_TASK_SANDBOX_TAG)), name),
        )
        for name, rules in per_task.items()
    }
    routers = {name: PerTaskSandbox(name) for name in pools}
    per_task_fields: dict[str, Any] = {
        "per_task": MappingProxyType(routers),
        "middleware": (TaskKeyMiddleware(pools),) if pools else (),
        "_pools": MappingProxyType(pools),
    }

    if on_demand:
        lazy = {
            name: OnDemandSandbox(name=name, sandbox_id=sandbox_ids[name], provision=functools.partial(one, name))
            for name in fixed
        }
        return SubagentSandboxes(
            thread_id=thread_id,
            backends=MappingProxyType(lazy),
            sandbox_ids=MappingProxyType(sandbox_ids),
            subagents=tuple(per_subagent_specs(
                specs, backends={**lazy, **routers}, default_model=default_model,
                parent_permissions=parent_permissions,
            )),
            _store=store,
            **per_task_fields,
        )

    backends: dict[str, MicroVMSandbox] = {}
    errors: list[BaseException] = []
    if fixed:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(fixed))),
                                thread_name_prefix="microvm-provision") as pool:
            futures = {name: pool.submit(one, name) for name in fixed}
        for name, future in futures.items():
            exc = future.exception()
            if exc is None:
                backends[name] = future.result()
            else:
                errors.append(exc)
    sandboxes = SubagentSandboxes(
        thread_id=thread_id,
        backends=MappingProxyType(backends),
        sandbox_ids=MappingProxyType({n: sandbox_ids[n] for n in backends}),
        _store=store,
    )
    if errors:
        pause_subagent_sandboxes(lifecycle, sandboxes, max_workers=max_workers, clock=clock)
        raise errors[0]
    specs_out = per_subagent_specs(
        specs, backends={**backends, **routers}, default_model=default_model, parent_permissions=parent_permissions,
    )
    return SubagentSandboxes(
        thread_id=thread_id,
        backends=sandboxes.backends,
        sandbox_ids=sandboxes.sandbox_ids,
        subagents=tuple(specs_out),
        _store=store,
        **per_task_fields,
    )


def _claim_sandbox(
    lifecycle: MicroVMLifecycle, store: GatewayStore, sandbox_id: str, *, clock: Callable[[], float]
) -> None:
    """Claim the sandbox's store row before get-or-create; if a sweep is deleting it, let that finish first.

    get-or-create no longer finds a deleted MicroVM, so it creates a fresh
    one. A row still retiring after `RETIRE_WAIT` (the sweep was interrupted)
    is deleted here.
    """
    if store.claim_sandbox(sandbox_id, now=clock()):
        return
    deadline = time.monotonic() + RETIRE_WAIT
    while time.monotonic() < deadline:
        time.sleep(_RETIRE_POLL)
        if store.claim_sandbox(sandbox_id, now=clock()):
            return
    row = store.get_sandbox(sandbox_id)
    logger.warning("Sandbox %s is still being deleted after %gs; deleting it here", sandbox_id, RETIRE_WAIT)
    with contextlib.suppress(SandboxNotFoundError):
        lifecycle.delete(sandbox_id=sandbox_id)
    if row is not None:
        store.retire_sandbox(sandbox_id, row.retired_reason or "gc", microvm_id=row.microvm_id)


def pause_subagent_sandboxes(
    lifecycle: MicroVMLifecycle,
    sandboxes: SubagentSandboxes,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    clock: Callable[[], float] = time.time,
) -> list[BaseException]:
    """Pause every sandbox in `sandboxes` in parallel (closing its transport). Errors are logged and returned.

    On-demand sandboxes that were never used this turn are left as they are;
    per-task sandboxes still live (a task that did not return) are paused too.
    Never raises: this runs in teardown, where an exception would mask the
    turn's own outcome. A sandbox that failed to pause still pauses itself
    after the MicroVM's idle timeout.
    """
    items = sandboxes._live()
    if not items:
        return []
    store = sandboxes._store

    def one(name: str, sandbox_id: str, backend: MicroVMSandbox) -> None:
        state = lifecycle.pause(backend)
        paused = state is None or state.status == MicroVMStatus.PAUSED  # None: local mode
        if not paused:
            logger.warning("The sandbox of subagent %r (thread %s) is %s after pausing; it pauses itself when idle",
                           name, sandboxes.thread_id, state.status.value if state else "?")
        if store is not None:
            store.touch_sandbox(
                sandbox_id, microvm_id=backend.id, thread_id=sandboxes.thread_id,
                subagent=name, status="paused" if paused else "active", now=clock(),
            )

    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items))),
                            thread_name_prefix="microvm-pause") as pool:
        futures = [(name, pool.submit(one, name, sandbox_id, backend)) for name, sandbox_id, backend in items]
    for name, future in futures:
        exc = future.exception()
        if exc is not None:
            logger.warning("Could not pause the sandbox of subagent %r (thread %s): %s",
                           name, sandboxes.thread_id, exc)
            errors.append(exc)
    return errors


async def aprovision_subagent_sandboxes(
    lifecycle: MicroVMLifecycle, specs: Sequence[Mapping[str, Any]], **kwargs: Any
) -> SubagentSandboxes:
    """`provision_subagent_sandboxes` off the event loop."""
    return await asyncio.to_thread(provision_subagent_sandboxes, lifecycle, specs, **kwargs)


async def apause_subagent_sandboxes(
    lifecycle: MicroVMLifecycle, sandboxes: SubagentSandboxes, **kwargs: Any
) -> list[BaseException]:
    """`pause_subagent_sandboxes` off the event loop."""
    return await asyncio.to_thread(pause_subagent_sandboxes, lifecycle, sandboxes, **kwargs)


@contextlib.asynccontextmanager
async def asubagent_sandboxes(
    lifecycle: MicroVMLifecycle,
    specs: Sequence[Mapping[str, Any]],
    *,
    thread_id: str,
    default_model: str | BaseChatModel,
    **kwargs: Any,
) -> AsyncIterator[SubagentSandboxes]:
    """Provision for one turn, pause afterwards. For a graph factory:

        @contextlib.asynccontextmanager
        async def make_graph(config, runtime):
            async with asubagent_sandboxes(lifecycle, SPECS, thread_id=..., default_model=model) as sb:
                yield create_deep_agent(model=model, subagents=list(sb.subagents),
                                        backend=sb.mount(StateBackend()))

    `kwargs` go to `provision_subagent_sandboxes` (`seed_files`, `templates`,
    `on_demand`, `per_task`, `store`, `parent_permissions`, `max_workers`, `clock`).
    """
    sandboxes = await aprovision_subagent_sandboxes(
        lifecycle, specs, thread_id=thread_id, default_model=default_model, **kwargs
    )
    try:
        yield sandboxes
    finally:
        await apause_subagent_sandboxes(
            lifecycle, sandboxes, **{k: v for k, v in kwargs.items() if k in ("max_workers", "clock")}
        )


@contextlib.contextmanager
def subagent_sandboxes(
    lifecycle: MicroVMLifecycle,
    specs: Sequence[Mapping[str, Any]],
    *,
    thread_id: str,
    default_model: str | BaseChatModel,
    **kwargs: Any,
) -> Iterator[SubagentSandboxes]:
    """Synchronous `asubagent_sandboxes`, for scripts."""
    sandboxes = provision_subagent_sandboxes(
        lifecycle, specs, thread_id=thread_id, default_model=default_model, **kwargs
    )
    try:
        yield sandboxes
    finally:
        pause_subagent_sandboxes(
            lifecycle, sandboxes, **{k: v for k, v in kwargs.items() if k in ("max_workers", "clock")}
        )


# ----------------------------------------------------------------------------
# deepagents internals, each behind one function so an upgrade breaks here
# ----------------------------------------------------------------------------


def _resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    # What create_deep_agent uses for spec["model"].
    from deepagents._models import resolve_model

    return resolve_model(model)


def _tool_description_overrides(model: BaseChatModel, raw: object) -> Mapping[str, str] | None:
    """The harness profile's tool descriptions deepagents gives this subagent's FilesystemMiddleware."""
    try:
        from deepagents.profiles.harness.harness_profiles import (
            _harness_profile_for_model,
        )
    except ImportError:  # moved in a later deepagents; descriptions fall back to the defaults
        return None
    profile = _harness_profile_for_model(model, raw if isinstance(raw, str) else None)
    return profile.tool_description_overrides or None


__all__ = [
    "DEFAULT_MOUNT_PREFIX",
    "SUBAGENT_SANDBOX_TAG",
    "OnDemandSandbox",
    "SubagentSandboxes",
    "apause_subagent_sandboxes",
    "aprovision_subagent_sandboxes",
    "asubagent_sandboxes",
    "is_sync_subagent",
    "pause_subagent_sandboxes",
    "per_subagent_specs",
    "provision_subagent_sandboxes",
    "subagent_sandbox_id",
    "subagent_sandboxes",
]
