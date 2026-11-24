"""Find and delete agent hosts and warm-start checkpoints that outlived their use.

While a supervisor runs, its gateway's `tick()` deletes failed hosts and hosts
unused past their TTL. Everything that outlives the supervisor needs a sweep
from outside, which is what `scripts/gc_microvms.py` runs:

- agent hosts (MicroVMs tagged `microvm-agenthost`) whose gateway-store row
  says they were destroyed or failed, or that have had no active run and no
  activity for `host_ttl` (the store's `last_active_at`; the MicroVMs API has
  no last-activity field and tags cannot change after create);
- agent hosts no store row describes (another store, a lost store file, a
  create that raced a crash), once they are older than `orphan_ttl`;
- leftover golden-checkpoint builds (MicroVMs tagged `microvm-golden`; a
  build deletes its VM when done, so one that lives past `golden_build_timeout`
  was interrupted);
- golden checkpoints (`microvm-golden-<hash>`) that failed, are stuck
  capturing, or are older than `checkpoint_ttl`. Nothing records which
  configuration still uses a checkpoint (the name is a hash, and clones
  report the image, not the checkpoint, as their source), so age is the only
  sound test; a deleted golden checkpoint is rebuilt on its next use (~35 s);
- Subagent and thread sandboxes (MicroVMs tagged `microvm-subagent-sandbox`,
  subagents.py, or `microvm-thread-sandbox`, threads.py) the store's
  `sandboxes` table marks destroyed or that were last used more than
  `sandbox_ttl` ago, and ones no row describes once older than `orphan_ttl`.
  A sandbox's files live only in it, so this is also how long a thread's
  files survive between turns. Other sandboxes
  (plain MicroVMLifecycle users such as the m2 demo) are never touched;
- store rows (hosts and sandboxes) whose MicroVM no longer exists, marked
  destroyed (`vanished`).

A host with an active run in the store is never deleted. `plan_sweep` only
reads; `apply_sweep` deletes and records each deleted host as destroyed with
`retired_reason="gc"`, so a gateway sharing the store answers `check` from its
cache and artifact fetches with a 410 that says why.

A gateway (or a supervisor with subagent sandboxes) may use a host or sandbox
between the plan and its application. Before deleting one the store describes,
`apply_sweep` re-checks the plan's decision in the store and marks the row
`retiring` in the same transaction (`GatewayStore.begin_retire_host` /
`begin_retire_sandbox`); one used since the plan is skipped and reported, and
a gateway that finds a row `retiring` waits for the deletion and starts a
fresh host. A row left `retiring` by an interrupted sweep is deleted by the next.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from langchain_digitalocean_microvm.agent_host.models import (
    HOST_MANAGED_TAG,
    HostStatus,
    host_id_from_tags,
)
from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    HostRow,
    SandboxRow,
)
from langchain_digitalocean_microvm.agent_host.warm import GOLDEN_PREFIX, GOLDEN_TAG
from langchain_digitalocean_microvm.client import (
    SUBAGENT_SANDBOX_TAG,
    THREAD_SANDBOX_TAG,
    MicroVMClient,
    sandbox_id_from_tags,
)
from langchain_digitalocean_microvm.exceptions import (
    MicroVMAPIError,
    MicroVMClientError,
    MicroVMNotFoundError,
)
from langchain_digitalocean_microvm.models import Checkpoint, MicroVMState
from langchain_digitalocean_microvm.namespace import (
    in_scope,
    namespace_from_name,
    namespace_from_tags,
)
from langchain_digitalocean_microvm.owner import in_owner_scope
from langchain_digitalocean_microvm.status import MicroVMStatus

DEFAULT_HOST_TTL = 24 * 3600.0
DEFAULT_CHECKPOINT_TTL = 24 * 3600.0
DEFAULT_GOLDEN_BUILD_TIMEOUT = 30 * 60.0
_GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)
_GONE_CHECKPOINT = ("CHECKPOINT_DELETING", "CHECKPOINT_DELETED")
_GONE_ROW = (HostStatus.DESTROYED.value, HostStatus.FAILED.value)  # rows retire_rows leaves alone
_RETIRED = (HostStatus.DESTROYED.value, HostStatus.FAILED.value, HostStatus.RETIRING.value)

ItemKind = Literal["host", "golden_vm", "checkpoint", "row", "sandbox", "sandbox_row"]


@dataclass(frozen=True, slots=True)
class SweepItem:
    """One thing the sweep deletes (or keeps), and why. `id` is a MicroVM, checkpoint or host id."""

    kind: ItemKind
    id: str
    name: str
    reason: str
    host_id: str | None = None
    sandbox_id: str | None = None
    idle_before: float | None = None  # delete only if the store row was last active at or before this


@dataclass(frozen=True, slots=True)
class SweepPlan:
    delete_microvms: tuple[SweepItem, ...] = ()
    delete_checkpoints: tuple[SweepItem, ...] = ()
    retire_rows: tuple[SweepItem, ...] = ()  # store rows (kind row / sandbox_row) whose MicroVM is gone
    kept: tuple[SweepItem, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.delete_microvms or self.delete_checkpoints or self.retire_rows)


@dataclass(slots=True)
class SweepResult:
    deleted_microvms: list[SweepItem] = field(default_factory=list)
    deleted_checkpoints: list[SweepItem] = field(default_factory=list)
    retired_rows: list[str] = field(default_factory=list)
    skipped: list[SweepItem] = field(default_factory=list)  # used since the plan: kept
    errors: list[str] = field(default_factory=list)


def plan_sweep(
    client: MicroVMClient,
    *,
    store: GatewayStore | None,
    now: float | None = None,
    host_ttl: float = DEFAULT_HOST_TTL,
    orphan_ttl: float | None = None,
    checkpoint_ttl: float = DEFAULT_CHECKPOINT_TTL,
    golden_build_timeout: float = DEFAULT_GOLDEN_BUILD_TIMEOUT,
    sandbox_ttl: float | None = None,
    namespace: str | None = None,
    all_namespaces: bool = False,
    all_owners: bool = True,
) -> SweepPlan:
    """Decide what to delete. Read-only. `now` is epoch seconds; `orphan_ttl` and `sandbox_ttl` default to `host_ttl`.

    Only MicroVMs and golden checkpoints of `namespace` (None: the default namespace, i.e. those without a
    namespace tag) are considered, or every namespace's with `all_namespaces`. `store` should be that namespace's store.
    With `all_owners=False`, an agent host or sandbox that no store row describes is kept unless it carries this
    user's owner tag (owner.py): a MicroVM the store does not know may belong to someone else on the team.
    """
    now = time.time() if now is None else now
    orphan_ttl = host_ttl if orphan_ttl is None else orphan_ttl
    sandbox_ttl = host_ttl if sandbox_ttl is None else sandbox_ttl
    delete: list[SweepItem] = []
    kept: list[SweepItem] = []
    live_hosts: set[tuple[str, str]] = set()  # (host_id, microvm_id)

    for state in _scoped(client.list(tag=HOST_MANAGED_TAG), namespace, all_namespaces):
        if state.status in _GONE:
            continue
        host_id = host_id_from_tags(state.tags)
        if host_id is not None:
            live_hosts.add((host_id, state.id))
        row = store.get_host(host_id) if store is not None and host_id is not None else None
        if row is None and not in_owner_scope(state.tags, all_owners=all_owners):
            kept.append(SweepItem("host", state.id, state.name, _NOT_YOURS))
            continue
        verdict, reason = _host_verdict(state, row, store, now=now, host_ttl=host_ttl, orphan_ttl=orphan_ttl)
        item = SweepItem("host", state.id, state.name, reason, host_id=host_id, idle_before=now - host_ttl)
        (delete if verdict else kept).append(item)

    # Golden checkpoints first: a golden MicroVM kept for a checkpoint with credentials lives as long as it.
    golden: list[tuple[Checkpoint, str | None]] = []
    for checkpoint in client.list_checkpoints():
        if not checkpoint.name.startswith(f"{GOLDEN_PREFIX}-") or checkpoint.status in _GONE_CHECKPOINT:
            continue
        if not in_scope(namespace_from_name(checkpoint.name, (GOLDEN_PREFIX,)), namespace, all_namespaces=all_namespaces):
            continue
        golden.append((checkpoint, _checkpoint_reason(checkpoint, now=now, ttl=checkpoint_ttl, build_timeout=golden_build_timeout)))
    origins = {cp.microvm_id: (cp, reason) for cp, reason in golden if cp.has_credentials and cp.microvm_id}

    for state in _scoped(client.list(tag=GOLDEN_TAG), namespace, all_namespaces):
        if state.status in _GONE:
            continue
        if state.id in origins:
            cp, cp_reason = origins[state.id]
            if cp_reason:
                delete.append(SweepItem("golden_vm", state.id, state.name, f"its warm-start checkpoint is deleted ({cp_reason})"))
            else:
                kept.append(SweepItem("golden_vm", state.id, state.name, f"keeps warm-start checkpoint {cp.name} restorable"))
            continue
        age = _age(now, state)
        if age is not None and age >= golden_build_timeout:
            delete.append(SweepItem("golden_vm", state.id, state.name, f"golden-checkpoint build left running {_fmt(age)}"))
        else:
            kept.append(SweepItem("golden_vm", state.id, state.name, "golden-checkpoint build in progress"))

    live_sandboxes: set[tuple[str, str]] = set()  # (sandbox_id, microvm_id)
    for state in _scoped([*client.list(tag=SUBAGENT_SANDBOX_TAG), *client.list(tag=THREAD_SANDBOX_TAG)], namespace, all_namespaces):
        if state.status in _GONE:
            continue
        sandbox_id = sandbox_id_from_tags(state.tags)
        if sandbox_id is not None:
            live_sandboxes.add((sandbox_id, state.id))
        srow = store.get_sandbox(sandbox_id) if store is not None and sandbox_id is not None else None
        if srow is None and not in_owner_scope(state.tags, all_owners=all_owners):
            kept.append(SweepItem("sandbox", state.id, state.name, _NOT_YOURS))
            continue
        verdict, reason = _sandbox_verdict(state, srow, now=now, ttl=sandbox_ttl, orphan_ttl=orphan_ttl)
        item = SweepItem("sandbox", state.id, state.name, reason, sandbox_id=sandbox_id, idle_before=now - sandbox_ttl)
        (delete if verdict else kept).append(item)

    checkpoints: list[SweepItem] = []
    for checkpoint, reason in golden:
        item = SweepItem("checkpoint", checkpoint.id, checkpoint.name, reason or "in use (rebuilt on demand)")
        (checkpoints if reason else kept).append(item)

    rows: list[SweepItem] = []
    if store is not None:
        for row in store.list_hosts():
            if row.status not in _GONE_ROW and (row.host_id, row.microvm_id) not in live_hosts:
                rows.append(SweepItem("row", row.host_id, row.microvm_id, "its MicroVM no longer exists", host_id=row.host_id))
        for srow in store.list_sandboxes():
            if srow.status not in _GONE_ROW and (srow.sandbox_id, srow.microvm_id) not in live_sandboxes:
                rows.append(SweepItem("sandbox_row", srow.sandbox_id, srow.microvm_id, "its MicroVM no longer exists",
                                      sandbox_id=srow.sandbox_id))

    return SweepPlan(tuple(delete), tuple(checkpoints), tuple(rows), tuple(kept))


def apply_sweep(
    plan: SweepPlan,
    client: MicroVMClient,
    *,
    store: GatewayStore | None,
    sleep: Callable[[float], None] = time.sleep,
) -> SweepResult:
    """Delete what `plan` lists and record it in the store. Errors are collected, not raised."""
    result = SweepResult()
    for item in plan.delete_microvms:
        if store is not None and not _begin_retire(store, item):
            result.skipped.append(item)
            continue
        try:
            client.delete(item.id)
        except MicroVMNotFoundError:
            pass
        except MicroVMClientError as exc:
            result.errors.append(f"MicroVM {item.name} ({item.id}): {exc}")
            continue  # a row marked retiring stays so: the next sweep deletes it again
        result.deleted_microvms.append(item)
        if store is not None and item.host_id is not None:
            store.retire_host(item.host_id, "gc", microvm_id=item.id)
        if store is not None and item.sandbox_id is not None:
            store.retire_sandbox(item.sandbox_id, "gc", microvm_id=item.id)
    for item in plan.delete_checkpoints:
        try:
            delete_checkpoint_patiently(client, item.id, sleep=sleep)
        except MicroVMClientError as exc:
            result.errors.append(f"checkpoint {item.name} ({item.id}): {exc}")
            continue
        result.deleted_checkpoints.append(item)
    if store is not None:
        for item in plan.retire_rows:  # item.name is the row's MicroVM: a row since given a new one is left alone
            if item.kind == "sandbox_row":
                retiring = (row := store.get_sandbox(item.id)) is not None and row.status == HostStatus.RETIRING.value
                done = store.retire_sandbox(item.id, "gc" if retiring else "vanished", microvm_id=item.name)
            else:
                retiring = (hrow := store.get_host(item.id)) is not None and hrow.status == HostStatus.RETIRING.value
                done = store.retire_host(item.id, "gc" if retiring else "vanished", microvm_id=item.name)
            if done:
                result.retired_rows.append(item.id)
    return result


_NOT_YOURS = "no store row and not created by this user (--all-owners selects it)"


def _scoped(states: list[MicroVMState], namespace: str | None, all_namespaces: bool) -> list[MicroVMState]:
    return [s for s in states if in_scope(namespace_from_tags(s.tags), namespace, all_namespaces=all_namespaces)]


def _begin_retire(store: GatewayStore, item: SweepItem) -> bool:
    """Re-check the plan's decision on a MicroVM the store describes, and mark its row retiring; False: keep it.

    MicroVMs no row describes (orphans, replaced ones, golden builds) need no
    check: nothing resumes a MicroVM without a row naming it.
    """
    if item.host_id is not None:
        row = store.get_host(item.host_id)
        if row is None or row.microvm_id != item.id or row.status == HostStatus.DESTROYED.value:
            return True
        return store.begin_retire_host(item.host_id, item.id, idle_before=item.idle_before, reason="gc")
    if item.sandbox_id is not None:
        srow = store.get_sandbox(item.sandbox_id)
        if srow is None or srow.microvm_id != item.id or srow.status == HostStatus.DESTROYED.value:
            return True
        return store.begin_retire_sandbox(item.sandbox_id, item.id, idle_before=item.idle_before, reason="gc")
    return True


def delete_checkpoint_patiently(
    client: MicroVMClient, checkpoint_id: str, *, attempts: int = 24, delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """DELETE a checkpoint, waiting out the 400 the API answers while it is still being captured."""
    for attempt in range(attempts):
        try:
            client.delete_checkpoint(checkpoint_id)
            return
        except MicroVMNotFoundError:
            return
        except MicroVMAPIError as exc:
            if exc.status_code != 400 or attempt == attempts - 1:
                raise
            sleep(delay)


def _host_verdict(
    state: MicroVMState, row: HostRow | None, store: GatewayStore | None, *, now: float, host_ttl: float,
    orphan_ttl: float,
) -> tuple[bool, str]:
    """(delete?, why) for one agent-host MicroVM."""
    if row is not None and row.microvm_id == state.id:
        if row.status in _RETIRED:
            return True, f"the gateway store marks it {row.status}"
        assert store is not None
        active = [t.thread_id for t in store.list_threads(host_id=row.host_id) if t.run_id and not t.terminal]
        if active:
            return False, f"task {active[0]} is in progress"
        unused = now - row.last_active_at
        if unused >= host_ttl:
            return True, f"unused for {_fmt(unused)} (TTL {_fmt(host_ttl)})"
        return False, f"unused for {_fmt(unused)}, TTL {_fmt(host_ttl)}"
    age = _age(now, state)
    where = "not in the gateway store" if row is None else "replaced in the gateway store by another MicroVM"
    if age is not None and age >= orphan_ttl:
        return True, f"{where}, created {_fmt(age)} ago"
    return False, f"{where}, created {_fmt(age) if age is not None else '?'} ago (orphan TTL {_fmt(orphan_ttl)})"


def _sandbox_verdict(
    state: MicroVMState, row: SandboxRow | None, *, now: float, ttl: float, orphan_ttl: float
) -> tuple[bool, str]:
    """(delete?, why) for one subagent or thread sandbox. Activity comes from the store (the API has none)."""
    if row is not None and row.microvm_id == state.id:
        if row.status in (HostStatus.DESTROYED.value, HostStatus.RETIRING.value):
            return True, f"the gateway store marks it {row.status}"
        unused = now - row.last_active_at
        owner = f"subagent {row.subagent!r} of thread {row.thread_id}" if row.subagent else f"thread {row.thread_id}"
        if unused >= ttl:
            return True, f"{owner}, unused for {_fmt(unused)} (TTL {_fmt(ttl)})"
        return False, f"{owner}, unused for {_fmt(unused)}, TTL {_fmt(ttl)}"
    age = _age(now, state)
    where = "not in the gateway store" if row is None else "replaced in the gateway store by another MicroVM"
    if age is not None and age >= orphan_ttl:
        return True, f"{where}, created {_fmt(age)} ago"
    return False, f"{where}, created {_fmt(age) if age is not None else '?'} ago (orphan TTL {_fmt(orphan_ttl)})"


def _checkpoint_reason(checkpoint: Checkpoint, *, now: float, ttl: float, build_timeout: float) -> str | None:
    age = now - checkpoint.created_at.timestamp() if checkpoint.created_at else None
    if checkpoint.status == "CHECKPOINT_FAILED":
        return "capture failed"
    if checkpoint.status == "CHECKPOINT_CREATING":
        return f"stuck capturing for {_fmt(age)}" if age is not None and age >= build_timeout else None
    if age is not None and age >= ttl:
        return f"older than {_fmt(ttl)} ({_fmt(age)}); rebuilt on next use"
    return None


def _age(now: float, state: MicroVMState) -> float | None:
    return now - state.created_at.timestamp() if state.created_at else None


def _fmt(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"
