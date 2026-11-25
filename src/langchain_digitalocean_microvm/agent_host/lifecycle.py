"""Host lifecycle for SubAgentGateway: allocation, claims, pause, garbage collection, eviction.

The gateway (gateway.py) serves the Agent Protocol routes and talks to the
hosts' Agent Servers; this module decides which MicroVM serves a task and
when one is paused or deleted (see docs/design.md):

- Allocation: `per-task` specs get one host per thread (host id = thread id),
  `pooled` specs one shared host per graph (`pool-<assistant>`). An existing
  row is claimed in the store before it is resumed or receives a run, so a
  sweep (`scripts/gc_microvms.py`) that decided on an older reading can no
  longer delete it; a row another process is deleting is waited for
  (`retire_wait`), then replaced by a fresh MicroVM.
- Pause: a READY host with no active run is paused once it has been idle for
  `pause_grace` seconds (or unused for `idle_backstop`), never a BUSY one and
  never one whose manager reports work in flight.
- Garbage collection: a FAILED host is deleted on the next tick, and a
  READY/PAUSED host with no active run is deleted once it has been unused for
  its time-to-live (`release_after` for per-task hosts, `paused_ttl` for every
  host). A host whose thread waits for a human (its run ended in an
  `interrupt()`) holds that thread, so only `paused_ttl` applies to it. The store keeps the row as `destroyed` with a `retired_reason`.
- Eviction: when creating a host fails because the team is at its MicroVM
  limit (HostCapacityError), one idle host whose task results are already
  cached is deleted and the create is tried once more. A host is only evicted
  once it has been unused for `eviction_grace` seconds, and never while a
  thread on it waits for a human.
- Startup reconciliation: stored hosts that no longer exist are marked
  destroyed (and their active tasks failed); host tokens persisted in the
  store are handed back to the host manager.

`HostLifecycle` is a base class of SubAgentGateway: it keeps the state it
owns (host manager, graph specs, store, TTLs, per-host locks) on the gateway
instance, which provides `_fail_thread` (run bookkeeping).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from langchain_digitalocean_microvm.agent_host.models import (
    HostHandle,
    HostSpec,
    HostStatus,
)
from langchain_digitalocean_microvm.agent_host.store import (
    RETIRED_REASONS,
    GatewayStore,
    HostRow,
    ThreadRow,
)
from langchain_digitalocean_microvm.exceptions import (
    HostCapacityError,
    SandboxError,
    SandboxNotFoundError,
)

logger = logging.getLogger(__name__)

ACTIVE_RUN_STATUSES: frozenset[str] = frozenset({"pending", "running"})
_DEAD_HOST_STATUSES = {HostStatus.FAILED.value, HostStatus.DESTROYED.value, HostStatus.RETIRING.value}

class HostManager(Protocol):
    """What the gateway needs from MicroVMAgentHost / LocalDevServerHost.

    Optional extras, used when present: `edge_headers() -> dict` (headers
    every request to a host must carry, e.g. a DigitalOcean access token),
    `remember_token(host_id, token)` (hand back tokens persisted in the
    store after a restart), `idle_timeout_seconds` (hosts pause after
    this long without inbound requests).
    """

    def get_or_create(self, *, host_id: str, spec: HostSpec) -> HostHandle: ...
    def resume(self, *, host_id: str, spec: HostSpec | None = None) -> HostHandle: ...
    def pause(self, *, host_id: str) -> None: ...
    def delete(self, *, host_id: str) -> None: ...
    def list_hosts(self, *, graph_id: str | None = None) -> list[HostHandle]: ...
    def is_idle(self, handle: HostHandle) -> bool: ...


class _HostRetired(SandboxError):
    """Another process (or this gateway's own tick) is deleting the host: it cannot be claimed."""

    def __init__(self, row: HostRow) -> None:
        super().__init__(f"agent host {row.host_id} {_retired_text(row)}")
        self.row = row

class HostLifecycle:
    """Which host serves a task, and when hosts are paused, deleted or evicted (base class of SubAgentGateway)."""

    hosts: HostManager
    graphs: dict[str, HostSpec]
    store: GatewayStore
    clock: Callable[[], float]

    def _init_lifecycle(
        self,
        *,
        pause_grace: float,
        idle_backstop: float,
        release_after: float | None,
        paused_ttl: float | None,
        eviction_grace: float,
        retire_wait: float,
    ) -> None:
        """Validate and keep the lifecycle settings; `hosts`, `graphs`, `store` and `clock` are set by the caller."""
        self.pause_grace = pause_grace
        self.idle_backstop = idle_backstop
        for name, value in (("release_after", release_after), ("paused_ttl", paused_ttl)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None (never), got {value}")
        self.release_after = release_after
        self.paused_ttl = paused_ttl
        self.eviction_grace = eviction_grace
        self.retire_wait = retire_wait
        self._locks: dict[str, asyncio.Lock] = {}

    def _fail_thread(self, thread: ThreadRow, message: str) -> dict[str, Any]:
        """Record `thread`'s run as failed with `message`; provided by the gateway."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # startup reconciliation
    # ------------------------------------------------------------------

    def _remember_tokens(self) -> None:
        """Hand the tokens persisted in the store back to a host manager that accepts them (after a restart)."""
        remember = getattr(self.hosts, "remember_token", None)
        if callable(remember):
            for row in self.store.list_hosts():
                if row.token and row.status not in _DEAD_HOST_STATUSES:
                    remember(row.host_id, row.token)

    def _reconcile_with_inventory(self) -> None:
        """Mark stored hosts that no longer exist as destroyed (and their active tasks as errors).

        The store outlives the process; the hosts may not (deleted by hand, by
        delete_all.py, or local servers that died with the previous process).
        Without this, the first task routed to a vanished host would fail.
        """
        rows = [r for r in self.store.list_hosts() if r.status not in _DEAD_HOST_STATUSES]
        if not rows:
            return
        try:
            live = {h.host_id: h.status.value for h in self.hosts.list_hosts()}
        except Exception:  # noqa: BLE001 - inventory failure must not stop the gateway
            logger.warning("Could not list agent hosts to reconcile the gateway store", exc_info=True)
            return
        for row in rows:
            status = live.get(row.host_id)
            if status is None or status in _DEAD_HOST_STATUSES:
                self.store.retire_host(row.host_id, "vanished")
                for thread in self.store.list_threads(host_id=row.host_id):
                    if thread.run_status in ACTIVE_RUN_STATUSES:
                        self._fail_thread(thread, f"agent host {row.host_id} no longer exists")
                logger.info("Agent host %s from a previous session is gone; marked destroyed", row.host_id)

    # ------------------------------------------------------------------
    # pause
    # ------------------------------------------------------------------

    async def _pause_idle_hosts(self) -> None:
        """Track idleness of READY/BUSY hosts; pause those idle past `pause_grace` (or unused past `idle_backstop`)."""
        now = self.clock()
        for host in self.store.list_hosts():
            if host.status not in (HostStatus.READY.value, HostStatus.BUSY.value):
                continue
            active = any(t.run_status in ACTIVE_RUN_STATUSES for t in self.store.list_threads(host_id=host.host_id))
            if active:
                if host.idle_since is not None or host.status != HostStatus.BUSY.value:
                    self.store.update_host(host.host_id, idle_since=None, status=HostStatus.BUSY.value)
                continue
            if host.idle_since is None:
                self.store.update_host(host.host_id, idle_since=now, status=HostStatus.READY.value)
                continue
            idle_for = now - host.idle_since
            unused_for = now - host.last_active_at
            if idle_for >= self.pause_grace or unused_for >= self.idle_backstop:
                await self._pause_host(host)

    # ------------------------------------------------------------------
    # host allocation and lifecycle
    # ------------------------------------------------------------------

    def _lock_for(self, host_id: str) -> asyncio.Lock:
        lock = self._locks.get(host_id)
        if lock is None:
            lock = self._locks[host_id] = asyncio.Lock()
        return lock

    def _host_id_for(self, thread: ThreadRow, assistant: str, spec: HostSpec) -> str:
        if spec.reuse == "pooled":
            return f"pool-{assistant}"
        return thread.thread_id

    def _spec_for(self, thread: ThreadRow) -> HostSpec | None:
        return self.graphs.get(thread.graph_id or "")

    async def _ensure_host_for_thread(self, thread: ThreadRow, assistant: str, spec: HostSpec) -> HostRow:
        host_id = thread.host_id or self._host_id_for(thread, assistant, spec)
        async with self._lock_for(host_id):
            row = self.store.get_host(host_id)
            if row is not None and row.status not in _DEAD_HOST_STATUSES:
                # Claim before resuming or sending a run: a sweep that decided on
                # an older reading can no longer delete it (store.claim_host).
                try:
                    if row.status == HostStatus.PAUSED.value:
                        return await self._resume_host_locked(row, spec)  # lock already held; claims
                    if not self.store.claim_host(host_id, now=self.clock()):
                        raise _HostRetired(self.store.get_host(host_id) or row)
                    return self.store.get_host(host_id) or row
                except _HostRetired as exc:
                    row = exc.row  # lost to a deletion: continue as for a dead row
            if row is not None and row.status in _DEAD_HOST_STATUSES and thread.host_id is not None:
                raise SandboxError(
                    f"agent host {host_id} for task {thread.thread_id} {_retired_text(row)}; "
                    "its result can still be checked, but the task cannot be updated. Start a new task instead."
                )
            if row is not None and row.status != HostStatus.DESTROYED.value:
                await self._finish_retirement_locked(row)  # never reuse a MicroVM that is (to be) deleted
            handle = await self._create_host(host_id, spec)
            return self._record_handle(handle, assistant)

    async def _finish_retirement_locked(self, row: HostRow) -> None:
        """Make sure the MicroVM of a failed or retiring row is gone before its host id gets a fresh one.

        A failed host is deleted here (what the next tick would do). A retiring
        one is being deleted by another process: wait up to `retire_wait` for
        its row to read destroyed, then delete it here (deleting twice is
        harmless; a crashed sweep leaves the row retiring). Caller holds the lock.
        """
        if row.status == HostStatus.RETIRING.value:
            deadline = time.monotonic() + self.retire_wait
            while True:
                current = self.store.get_host(row.host_id)
                if current is None or current.status == HostStatus.DESTROYED.value:
                    return
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0.5)
            logger.warning("Agent host %s is still being deleted after %gs; deleting it here", row.host_id,
                           self.retire_wait)
        elif not self.store.begin_retire_host(row.host_id, row.microvm_id, idle_before=None, reason="failed"):
            current = self.store.get_host(row.host_id)
            if current is None or current.status == HostStatus.DESTROYED.value:
                return
            raise SandboxError(f"agent host {row.host_id} changed while it was being replaced; try again")
        try:
            await asyncio.to_thread(self.hosts.delete, host_id=row.host_id)
        except SandboxNotFoundError:
            pass
        except SandboxError as exc:
            raise SandboxError(f"agent host {row.host_id} could not be deleted before its replacement: {exc}") from exc
        current = self.store.get_host(row.host_id)
        self.store.retire_host(row.host_id, (current.retired_reason if current else None) or "failed",
                               microvm_id=row.microvm_id)

    async def _create_host(self, host_id: str, spec: HostSpec) -> HostHandle:
        """get_or_create; at the team's MicroVM limit, evict one idle cached host and retry once."""
        try:
            return await asyncio.to_thread(self.hosts.get_or_create, host_id=host_id, spec=spec)
        except HostCapacityError as exc:
            now = self.clock()
            evictable, protected = self._eviction_candidates(exclude=host_id, now=now)
            for victim in evictable:
                logger.warning("At the MicroVM limit; deleting idle agent host %s to make room for %s", victim.host_id, host_id)
                if await self._retire_host(victim, reason="evicted"):
                    break
            else:
                if protected:
                    soonest = min(protected, key=lambda h: h.last_active_at)
                    wait = max(1, int(soonest.last_active_at + self.eviction_grace - now + 0.999))
                    raise SandboxError(
                        f"{exc} Agent host {soonest.host_id} is idle, but its tasks' files are kept for "
                        f"{self.eviction_grace:g}s after its last activity so they can be fetched "
                        f"(fetch_subagent_artifacts); try again in about {wait}s."
                    ) from exc
                raise SandboxError(
                    f"{exc} Every existing agent host still has a task in progress or waiting for a human; wait for one to finish "
                    "or cancel a task, then try again."
                ) from exc
            return await asyncio.to_thread(self.hosts.get_or_create, host_id=host_id, spec=spec)

    def _eviction_candidates(self, *, exclude: str, now: float) -> tuple[list[HostRow], list[HostRow]]:
        """Idle hosts whose every task is terminal (results cached): (past `eviction_grace`, still inside it).

        The first list is ordered paused first, then least recently active.
        """
        evictable: list[HostRow] = []
        protected: list[HostRow] = []
        for host in self.store.list_hosts():
            if host.host_id == exclude or host.status in _DEAD_HOST_STATUSES:
                continue
            if self._has_active_run(host.host_id) or self._has_waiting_thread(host.host_id):
                continue
            (evictable if now - host.last_active_at >= self.eviction_grace else protected).append(host)
        evictable.sort(key=lambda h: (h.status != HostStatus.PAUSED.value, h.last_active_at))
        return evictable, protected

    def _has_active_run(self, host_id: str) -> bool:
        return any(t.run_id and not t.terminal for t in self.store.list_threads(host_id=host_id))

    def _has_waiting_thread(self, host_id: str) -> bool:
        """A thread on the host waits for a human: deleting the host would lose the thread and its interrupt."""
        return any(t.waiting for t in self.store.list_threads(host_id=host_id))

    # ------------------------------------------------------------------
    # garbage collection
    # ------------------------------------------------------------------

    def _ttl_for(self, host: HostRow) -> float | None:
        """Seconds this host may stay unused: `paused_ttl` for pooled hosts and hosts whose thread waits for a human,
        also `release_after` for other per-task ones."""
        spec = self.graphs.get(host.graph_id)
        limits = [self.paused_ttl]
        if (spec is None or spec.reuse != "pooled") and not self._has_waiting_thread(host.host_id):
            limits.append(self.release_after)
        set_limits = [v for v in limits if v is not None]
        return min(set_limits) if set_limits else None

    async def _collect_garbage(self) -> None:
        """Delete FAILED hosts, and READY/PAUSED hosts unused for longer than their TTL (never one with an active run)."""
        now = self.clock()
        for host in self.store.list_hosts():
            if host.status == HostStatus.FAILED.value:
                await self._retire_host(host, reason="failed")
                continue
            if host.status == HostStatus.RETIRING.value:  # a deletion that failed or was interrupted
                await self._retire_host(host, reason=host.retired_reason or "gc")
                continue
            if host.status not in (HostStatus.READY.value, HostStatus.PAUSED.value):
                continue
            ttl = self._ttl_for(host)
            if ttl is not None and now - host.last_active_at >= ttl:
                logger.info("Agent host %s unused for %.0fs (TTL %gs); deleting it", host.host_id, now - host.last_active_at, ttl)
                await self._retire_host(host, reason="ttl")

    async def _retire_host(self, host: HostRow, *, reason: str) -> bool:
        """Delete `host` and record why; False when it was used meanwhile, has work, or the delete failed.

        Takes the host's lock and re-reads the row: a host that received a run,
        an update or an artifact fetch since the caller looked is left alone.
        """
        async with self._lock_for(host.host_id):
            current = self.store.get_host(host.host_id)
            if current is None or current.status == HostStatus.DESTROYED.value:
                return False
            if current.status != host.status or current.last_active_at != host.last_active_at:
                return False
            if self._has_active_run(host.host_id):
                return False
            if reason == "evicted" and self._has_waiting_thread(host.host_id):
                return False
            if current.status == HostStatus.READY.value and reason != "failed":
                try:
                    idle = await asyncio.to_thread(self.hosts.is_idle, self._handle_from_row(current))
                except SandboxNotFoundError:
                    idle = True  # unknown to the manager: nothing left to protect
                except SandboxError:
                    idle = False
                if not idle:
                    logger.info("Host %s reports work in flight; not deleting it", host.host_id)
                    return False
            # Another process (a supervisor's subagent claims, gc_microvms.py) shares the store:
            # decide in one transaction, against what this method read.
            if not self.store.begin_retire_host(
                host.host_id, current.microvm_id, idle_before=current.last_active_at, reason=reason
            ):
                return False
            try:
                await asyncio.to_thread(self.hosts.delete, host_id=host.host_id)
            except SandboxNotFoundError:
                pass  # already gone; record it
            except SandboxError:
                logger.warning("Deleting agent host %s (%s) failed; will retry", host.host_id, reason, exc_info=True)
                return False  # the row stays retiring; the next tick deletes it again
            self.store.retire_host(host.host_id, reason, microvm_id=current.microvm_id)
            return True

    async def _resume_host(self, host: HostRow, spec: HostSpec | None) -> HostRow:
        async with self._lock_for(host.host_id):
            return await self._resume_host_locked(host, spec)

    async def _resume_host_locked(self, host: HostRow, spec: HostSpec | None) -> HostRow:
        """Resume `host`; the caller must hold `_lock_for(host.host_id)` (asyncio.Lock is not reentrant)."""
        current = self.store.get_host(host.host_id) or host
        if current.status != HostStatus.PAUSED.value:
            return current
        if not self.store.claim_host(host.host_id, now=self.clock()):
            raise _HostRetired(self.store.get_host(host.host_id) or current)
        handle = await asyncio.to_thread(self.hosts.resume, host_id=host.host_id, spec=spec)
        return self._record_handle(handle, current.graph_id)

    async def _pause_host(self, host: HostRow) -> None:
        async with self._lock_for(host.host_id):
            current = self.store.get_host(host.host_id) or host
            if current.status not in (HostStatus.READY.value, HostStatus.BUSY.value):
                return
            handle = self._handle_from_row(current)
            try:
                idle = await asyncio.to_thread(self.hosts.is_idle, handle)
            except SandboxNotFoundError:
                # The manager has never heard of this host (typically a supervisor
                # restart with local hosts): same verdict as `_live_status` -> gone.
                logger.warning("Host %s is unknown to the host manager; marking it failed", host.host_id)
                self.store.update_host(host.host_id, status=HostStatus.FAILED.value)
                return
            except SandboxError:
                logger.warning("Could not tell whether host %s is idle; not pausing", host.host_id, exc_info=True)
                self.store.update_host(host.host_id, idle_since=self.clock())
                return
            if not idle:
                logger.info("Host %s reports work in flight; not pausing", host.host_id)
                self.store.update_host(host.host_id, idle_since=self.clock())
                return
            try:
                await asyncio.to_thread(self.hosts.pause, host_id=host.host_id)
            except SandboxError:
                logger.warning("Pausing host %s failed", host.host_id, exc_info=True)
                return
            self.store.update_host(host.host_id, status=HostStatus.PAUSED.value)

    def _record_handle(self, handle: HostHandle, graph_id: str) -> HostRow:
        now = self.clock()
        row = HostRow(
            host_id=handle.host_id,
            microvm_id=handle.microvm_id,
            graph_id=graph_id,
            status=HostStatus.READY.value if handle.status == HostStatus.READY else handle.status.value,
            url=handle.url,
            token=handle.token,
            booted_at=handle.booted_at,
            last_active_at=now,
            idle_since=None,
            failures=0,
        )
        self.store.upsert_host(row)
        return row

    @staticmethod
    def _handle_from_row(row: HostRow) -> HostHandle:
        try:
            status = HostStatus(row.status)
        except ValueError:
            status = HostStatus.UNKNOWN
        return HostHandle(
            host_id=row.host_id,
            microvm_id=row.microvm_id,
            graph_id=row.graph_id,
            url=row.url,
            token=row.token,
            status=status,
            booted_at=row.booted_at,
        )

    async def _live_status(self, host_id: str) -> str | None:
        try:
            handles = await asyncio.to_thread(self.hosts.list_hosts)
        except Exception:  # noqa: BLE001 - inventory failure must not mask the real error
            logger.warning("list_hosts failed while checking host %s", host_id, exc_info=True)
            return "unknown"
        for handle in handles:
            if handle.host_id == host_id:
                return handle.status.value
        return None


def _retired_text(host: HostRow) -> str:
    """'was deleted after ...' for a destroyed/failed/retiring host row, for messages to the supervisor."""
    if host.status == HostStatus.DESTROYED.value and host.retired_reason in RETIRED_REASONS:
        return f"was {RETIRED_REASONS[host.retired_reason]}"
    if host.status == HostStatus.RETIRING.value:
        why = RETIRED_REASONS.get(host.retired_reason or "", "deleted")
        return f"is being {why}"
    if host.status == HostStatus.FAILED.value:
        return "failed"
    return f"was {host.status} (deleted, evicted at the MicroVM limit, or lost)"
