"""SQLite-backed state for the SubAgentGateway: hosts and threads (plus subagent sandboxes).

Deliberately tiny and synchronous (stdlib sqlite3, one connection, one lock):
every operation is a point lookup or a single-row upsert, so it never blocks
an event loop for longer than a disk sync. The schema is the one in
docs/design.md; JSON columns hold the cached terminal
run and the thread's final `values` so `check_async_task` can be answered
without waking a paused host.

The `sandboxes` table belongs to subagent sandboxes (subagents: one MicroVMSandbox per
subagent per thread) and thread sandboxes (threads: one per thread, `subagent` empty).
Nothing in the gateway reads it; it records when each
sandbox was last used so agent_host/sweeper.py can expire it, since the
MicroVMs API has no last-activity field and tags are fixed at create.

Several processes may share one store file (a gateway, a supervisor using
subagent sandboxes, scripts/gc_microvms.py). Deleting and reusing a MicroVM
are therefore decided by compare-and-set, each in one `BEGIN IMMEDIATE`
transaction: `begin_retire_*` marks a row `retiring` only if it is still
unused, `claim_*` records use only if the row is not being retired. Whoever
wins decides; a MicroVM is never resumed and deleted at once.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_APP_DIR = "langchain_digitalocean_microvm"


def state_dir() -> Path:
    """The per-user directory for the gateway store: the platform's state directory.

    macOS: `~/Library/Application Support/langchain_digitalocean_microvm`; Windows:
    `%LOCALAPPDATA%\\langchain_digitalocean_microvm`; elsewhere `$XDG_STATE_HOME` (else
    `~/.local/state`)`/langchain_digitalocean_microvm`.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / _APP_DIR


def legacy_state_dir() -> Path:
    """`~/.langchain_digitalocean_microvm`, where earlier versions kept the store."""
    return Path.home() / f".{_APP_DIR}"


def store_dirs() -> list[Path]:
    """The directories that may hold gateway stores: `state_dir()`, then the legacy one."""
    return [state_dir(), legacy_state_dir()]


def default_db_path(host_mode: Literal["local", "microvm"], namespace: str | None = None) -> Path:
    """`gateway.sqlite` (real MicroVMs) or `gateway-local.sqlite` in `state_dir()`.

    With a namespace the file is `gateway-<namespace>.sqlite` (`gateway-local-<namespace>.sqlite`), so
    two namespaces of one user never share rows. A store that earlier versions left in
    `legacy_state_dir()` keeps being used until a file of that name exists in `state_dir()`
    (move it there to adopt the new location); nothing is moved or copied.
    """
    stem = "gateway-local" if host_mode == "local" else "gateway"
    filename = f"{stem}-{namespace}.sqlite" if namespace else f"{stem}.sqlite"
    path = state_dir() / filename
    legacy = legacy_state_dir() / filename
    if not path.exists() and legacy.exists():
        logger.info("Using the gateway store %s; move it to %s to adopt the new location", legacy, path)
        return legacy
    return path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (
    host_id        TEXT PRIMARY KEY,
    microvm_id     TEXT,
    graph_id       TEXT,
    status         TEXT,
    url            TEXT,
    token          TEXT,
    booted_at      TEXT,
    last_active_at REAL,
    idle_since     REAL,
    failures       INTEGER NOT NULL DEFAULT 0,
    retired_reason TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    thread_id      TEXT PRIMARY KEY,
    graph_id       TEXT,
    host_id        TEXT,
    run_id         TEXT,
    run_status     TEXT,
    run_started_at TEXT,
    host_microvm_id TEXT,
    cached_run     TEXT,
    cached_values  TEXT,
    created_at     REAL,
    last_active_at REAL,
    cached_thread  TEXT
);
CREATE INDEX IF NOT EXISTS threads_host ON threads(host_id);
CREATE TABLE IF NOT EXISTS sandboxes (
    sandbox_id     TEXT PRIMARY KEY,
    microvm_id     TEXT,
    thread_id      TEXT,
    subagent       TEXT,
    status         TEXT,
    last_active_at REAL,
    retired_reason TEXT
);
"""
# Columns added after a store file may already exist: (table, column, type).
_ADDED_COLUMNS = (
    ("threads", "host_microvm_id", "TEXT"),  # which MicroVM ran the task (artifacts)
    ("hosts", "retired_reason", "TEXT"),  # why a destroyed host was deleted (RETIRED_REASONS)
    ("threads", "cached_thread", "TEXT"),  # a finished run's thread status and interrupts when it waits for a human
)

TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"success", "error", "timeout", "interrupted", "cancelled"})
# hosts.status values a claim cannot revive (agent_host.models.HostStatus): being deleted, deleted, failed.
_UNCLAIMABLE_HOST_STATUSES = ("retiring", "destroyed", "failed")
# Retirement needs no idle time for these: the row already says the MicroVM is to go.
_DOOMED_STATUSES = ("retiring", "failed")
_BUSY_TIMEOUT = 30.0  # seconds a transaction waits for another process's write lock

# hosts.retired_reason -> how the gateway explains a deleted host (410 on artifacts, 503 on update).
RETIRED_REASONS: dict[str, str] = {
    "ttl": "deleted after being idle longer than the gateway's time-to-live",
    "evicted": "deleted to make room for another task at the team's MicroVM limit",
    "failed": "deleted after it failed",
    "gc": "deleted by scripts/gc_microvms.py",
    "deleted": "deleted by scripts/delete_all.py",
    "vanished": "deleted outside the gateway",
    "closed": "stopped with the supervisor that started it",
}


@dataclass
class HostRow:
    host_id: str
    microvm_id: str
    graph_id: str
    status: str
    url: str
    token: str
    booted_at: str
    last_active_at: float
    idle_since: float | None
    failures: int
    retired_reason: str | None = None  # set with status=destroyed; key of RETIRED_REASONS


# sandboxes.status values: `active` while a turn uses the sandbox, `paused` after
# the turn's teardown, `destroyed` once deleted (with a RETIRED_REASONS key).
# `retiring` (set only by begin_retire_sandbox) marks a deletion in progress.
SANDBOX_STATUSES: frozenset[str] = frozenset({"active", "paused", "destroyed"})


@dataclass
class SandboxRow:
    sandbox_id: str
    microvm_id: str
    thread_id: str
    subagent: str
    status: str
    last_active_at: float
    retired_reason: str | None = None


@dataclass
class ThreadRow:
    thread_id: str
    graph_id: str | None
    host_id: str | None
    run_id: str | None
    run_status: str | None
    run_started_at: str | None
    host_microvm_id: str | None  # the MicroVM the latest run was sent to; its files live (only) there
    cached_run: dict[str, Any] | None
    cached_values: dict[str, Any] | None
    created_at: float
    last_active_at: float
    # A finished run's thread when it is not simply idle: {"status": "interrupted", "interrupts": {...}} for a run that
    # paused for a human (it ends `success`; only the thread shows the wait). None otherwise.
    cached_thread: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.run_status in TERMINAL_RUN_STATUSES

    @property
    def waiting(self) -> bool:
        """The latest run paused for a human (an `interrupt()`): the thread, held by its host, waits for a resume."""
        return isinstance(self.cached_thread, dict) and self.cached_thread.get("status") == "interrupted"


# Columns `update_host` / `update_thread` may set: the row fields minus the key. Their SET
# clause is built from argument names, so anything else is refused before any SQL is built.
_HOST_COLUMNS = frozenset(f.name for f in dataclass_fields(HostRow)) - {"host_id"}
_THREAD_COLUMNS = frozenset(f.name for f in dataclass_fields(ThreadRow)) - {"thread_id"}


def _check_columns(table: str, names: Iterable[str], allowed: frozenset[str]) -> None:
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unknown {table} columns: {unknown}")


def _loads(raw: str | None) -> dict[str, Any] | None:
    return json.loads(raw) if raw else None


class GatewayStore:
    def __init__(self, path: Path | str) -> None:
        """Open (or create) the store. The file holds host tokens: owner-only (0600).

        A missing parent directory is created owner-only (0700); an existing
        one is left as it is.
        """
        self.path = Path(path)
        in_memory = str(self.path) == ":memory:"
        if not in_memory and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, mode=0o700)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=_BUSY_TIMEOUT
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            for table, column, kind in _ADDED_COLUMNS:
                existing = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        if not in_memory:
            self.path.chmod(0o600)  # also tightens a store created by an older version

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ hosts

    def upsert_host(self, row: HostRow) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO hosts (host_id, microvm_id, graph_id, status, url, token,
                                      booted_at, last_active_at, idle_since, failures, retired_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(host_id) DO UPDATE SET
                     microvm_id=excluded.microvm_id, graph_id=excluded.graph_id, status=excluded.status,
                     url=excluded.url, token=excluded.token, booted_at=excluded.booted_at,
                     last_active_at=excluded.last_active_at, idle_since=excluded.idle_since,
                     failures=excluded.failures, retired_reason=excluded.retired_reason""",
                (
                    row.host_id, row.microvm_id, row.graph_id, row.status, row.url, row.token,
                    row.booted_at, row.last_active_at, row.idle_since, row.failures,
                    row.retired_reason,
                ),
            )

    def get_host(self, host_id: str) -> HostRow | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM hosts WHERE host_id = ?", (host_id,))
            raw = cur.fetchone()
        return self._host_row(raw) if raw else None

    def list_hosts(self, *, graph_id: str | None = None) -> list[HostRow]:
        with self._lock:
            if graph_id is None:
                cur = self._conn.execute("SELECT * FROM hosts ORDER BY host_id")
            else:
                cur = self._conn.execute("SELECT * FROM hosts WHERE graph_id = ? ORDER BY host_id", (graph_id,))
            rows = cur.fetchall()
        return [self._host_row(r) for r in rows]

    def update_host(self, host_id: str, **fields: Any) -> None:
        if not fields:
            return
        _check_columns("hosts", fields, _HOST_COLUMNS)
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE hosts SET {assignments} WHERE host_id = ?",  # noqa: S608 - column names allowlisted
                (*fields.values(), host_id),
            )

    def retire_host(self, host_id: str, reason: str, *, microvm_id: str | None = None) -> bool:
        """Record that a host was deleted (status `destroyed`) and why (a RETIRED_REASONS key).

        With `microvm_id`, only while the row still describes that MicroVM and
        is not destroyed yet. True when the row was updated.
        """
        if microvm_id is None:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE hosts SET status = 'destroyed', retired_reason = ? WHERE host_id = ?", (reason, host_id)
                )
        else:
            with self._lock:
                cur = self._conn.execute(
                    """UPDATE hosts SET status = 'destroyed', retired_reason = ?
                       WHERE host_id = ? AND microvm_id = ? AND status != 'destroyed'""",
                    (reason, host_id, microvm_id),
                )
        return cur.rowcount > 0

    def begin_retire_host(self, host_id: str, microvm_id: str, *, idle_before: float | None, reason: str) -> bool:
        """Mark a host `retiring` before deleting its MicroVM; False if it may not be deleted now.

        One transaction re-checks what the caller decided on: the row still
        describes `microvm_id`, is not destroyed, has no active run and, unless
        it is failed or already retiring, was last active at or before
        `idle_before` (None: no time condition). `reason` (a RETIRED_REASONS key)
        is kept in `retired_reason` so whoever finishes the deletion records it.
        After True, delete the MicroVM, then `retire_host`. A row left
        `retiring` (the deleter crashed) stays unclaimable and is deleted again
        by the next sweep or gateway tick.
        """
        with self._transaction() as conn:
            raw = conn.execute("SELECT * FROM hosts WHERE host_id = ?", (host_id,)).fetchone()
            if raw is None or raw["microvm_id"] != microvm_id or raw["status"] == "destroyed":
                return False
            doomed = raw["status"] in _DOOMED_STATUSES
            if not doomed and idle_before is not None and (raw["last_active_at"] or 0.0) > idle_before:
                return False
            active = conn.execute(
                f"""SELECT 1 FROM threads WHERE host_id = ? AND run_id IS NOT NULL AND run_id != ''
                    AND (run_status IS NULL OR run_status NOT IN ({",".join("?" * len(TERMINAL_RUN_STATUSES))}))
                    LIMIT 1""",  # noqa: S608 - placeholders only
                (host_id, *sorted(TERMINAL_RUN_STATUSES)),
            ).fetchone()
            if active is not None:
                return False
            conn.execute(
                "UPDATE hosts SET status = 'retiring', retired_reason = ? WHERE host_id = ?", (reason, host_id)
            )
            return True

    def claim_host(self, host_id: str, *, now: float) -> bool:
        """Record use of a host about to be resumed or sent a run; False if it is retiring, destroyed or failed.

        A True claim moves `last_active_at` past any sweep's cutoff, so a
        `begin_retire_host` decided on an older reading fails.
        """
        with self._transaction() as conn:
            cur = conn.execute(
                f"""UPDATE hosts SET last_active_at = ?
                    WHERE host_id = ? AND status NOT IN ({",".join("?" * len(_UNCLAIMABLE_HOST_STATUSES))})""",  # noqa: S608
                (now, host_id, *_UNCLAIMABLE_HOST_STATUSES),
            )
            return cur.rowcount > 0

    def delete_host(self, host_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM hosts WHERE host_id = ?", (host_id,))

    # ---------------------------------------------------------------- threads

    def create_thread(self, thread_id: str, *, now: float) -> ThreadRow:
        with self._lock:
            self._conn.execute(
                "INSERT INTO threads (thread_id, created_at, last_active_at) VALUES (?, ?, ?)",
                (thread_id, now, now),
            )
        return ThreadRow(thread_id, None, None, None, None, None, None, None, None, now, now)

    def get_thread(self, thread_id: str) -> ThreadRow | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,))
            raw = cur.fetchone()
        return self._thread_row(raw) if raw else None

    def list_threads(self, *, host_id: str | None = None) -> list[ThreadRow]:
        with self._lock:
            if host_id is None:
                cur = self._conn.execute("SELECT * FROM threads ORDER BY created_at")
            else:
                cur = self._conn.execute(
                    "SELECT * FROM threads WHERE host_id = ? ORDER BY created_at", (host_id,)
                )
            rows = cur.fetchall()
        return [self._thread_row(r) for r in rows]

    def update_thread(self, thread_id: str, **fields: Any) -> None:
        if not fields:
            return
        _check_columns("threads", fields, _THREAD_COLUMNS)
        encoded = {
            name: (json.dumps(value) if name in ("cached_run", "cached_values", "cached_thread") and value is not None else value)
            for name, value in fields.items()
        }
        assignments = ", ".join(f"{name} = ?" for name in encoded)
        with self._lock:
            self._conn.execute(
                f"UPDATE threads SET {assignments} WHERE thread_id = ?",  # noqa: S608 - column names allowlisted
                (*encoded.values(), thread_id),
            )

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    # -------------------------------------------------------------- sandboxes

    def touch_sandbox(
        self, sandbox_id: str, *, microvm_id: str, thread_id: str, subagent: str, status: str, now: float
    ) -> None:
        """Record use of a subagent or thread sandbox (insert or update; clears any retired_reason)."""
        if status not in SANDBOX_STATUSES:
            raise ValueError(f"sandbox status must be one of {sorted(SANDBOX_STATUSES)}, got {status!r}")
        with self._lock:
            self._conn.execute(
                """INSERT INTO sandboxes (sandbox_id, microvm_id, thread_id, subagent, status,
                                          last_active_at, retired_reason)
                   VALUES (?, ?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(sandbox_id) DO UPDATE SET
                     microvm_id=excluded.microvm_id, thread_id=excluded.thread_id,
                     subagent=excluded.subagent, status=excluded.status,
                     last_active_at=excluded.last_active_at, retired_reason=NULL""",
                (sandbox_id, microvm_id, thread_id, subagent, status, now),
            )

    def get_sandbox(self, sandbox_id: str) -> SandboxRow | None:
        with self._lock:
            raw = self._conn.execute("SELECT * FROM sandboxes WHERE sandbox_id = ?", (sandbox_id,)).fetchone()
        return self._sandbox_row(raw) if raw else None

    def list_sandboxes(self) -> list[SandboxRow]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sandboxes ORDER BY sandbox_id").fetchall()
        return [self._sandbox_row(r) for r in rows]

    def retire_sandbox(self, sandbox_id: str, reason: str, *, microvm_id: str | None = None) -> bool:
        """Record that a sandbox was deleted (status `destroyed`) and why (a RETIRED_REASONS key).

        With `microvm_id`, only while the row still describes that MicroVM and
        is not destroyed yet. True when the row was updated.
        """
        with self._lock:
            if microvm_id is None:
                cur = self._conn.execute(
                    "UPDATE sandboxes SET status = 'destroyed', retired_reason = ? WHERE sandbox_id = ?",
                    (reason, sandbox_id),
                )
            else:
                cur = self._conn.execute(
                    """UPDATE sandboxes SET status = 'destroyed', retired_reason = ?
                       WHERE sandbox_id = ? AND microvm_id = ? AND status != 'destroyed'""",
                    (reason, sandbox_id, microvm_id),
                )
        return cur.rowcount > 0

    def begin_retire_sandbox(self, sandbox_id: str, microvm_id: str, *, idle_before: float | None, reason: str) -> bool:
        """Mark a sandbox `retiring` before deleting its MicroVM; False if it may not be deleted now.

        Same contract as `begin_retire_host`, without the run check (the
        store does not know a sandbox's turns, only when it was last used).
        """
        with self._transaction() as conn:
            raw = conn.execute("SELECT * FROM sandboxes WHERE sandbox_id = ?", (sandbox_id,)).fetchone()
            if raw is None or raw["microvm_id"] != microvm_id or raw["status"] == "destroyed":
                return False
            doomed = raw["status"] in _DOOMED_STATUSES
            if not doomed and idle_before is not None and (raw["last_active_at"] or 0.0) > idle_before:
                return False
            conn.execute(
                "UPDATE sandboxes SET status = 'retiring', retired_reason = ? WHERE sandbox_id = ?",
                (reason, sandbox_id),
            )
            return True

    def claim_sandbox(self, sandbox_id: str, *, now: float) -> bool:
        """Record use of a sandbox about to be resumed or created; False only while it is retiring.

        A sandbox with no row or a destroyed one is claimable: its next
        get-or-create makes a fresh MicroVM (a deleted one is no longer found).
        """
        with self._transaction() as conn:
            raw = conn.execute("SELECT status FROM sandboxes WHERE sandbox_id = ?", (sandbox_id,)).fetchone()
            if raw is None:
                return True
            if raw["status"] == "retiring":
                return False
            conn.execute("UPDATE sandboxes SET last_active_at = ? WHERE sandbox_id = ?", (now, sandbox_id))
            return True

    # ---------------------------------------------------------------- helpers

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One `BEGIN IMMEDIATE` transaction: holds the file's write lock against other processes too."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    @staticmethod
    def _sandbox_row(raw: sqlite3.Row) -> SandboxRow:
        return SandboxRow(
            sandbox_id=raw["sandbox_id"],
            microvm_id=raw["microvm_id"] or "",
            thread_id=raw["thread_id"] or "",
            subagent=raw["subagent"] or "",
            status=raw["status"] or "",
            last_active_at=raw["last_active_at"] or 0.0,
            retired_reason=raw["retired_reason"],
        )

    @staticmethod
    def _host_row(raw: sqlite3.Row) -> HostRow:
        return HostRow(
            host_id=raw["host_id"],
            microvm_id=raw["microvm_id"] or "",
            graph_id=raw["graph_id"] or "",
            status=raw["status"] or "",
            url=raw["url"] or "",
            token=raw["token"] or "",
            booted_at=raw["booted_at"] or "",
            last_active_at=raw["last_active_at"] or 0.0,
            idle_since=raw["idle_since"],
            failures=raw["failures"] or 0,
            retired_reason=raw["retired_reason"],
        )

    @staticmethod
    def _thread_row(raw: sqlite3.Row) -> ThreadRow:
        return ThreadRow(
            thread_id=raw["thread_id"],
            graph_id=raw["graph_id"],
            host_id=raw["host_id"],
            run_id=raw["run_id"],
            run_status=raw["run_status"],
            run_started_at=raw["run_started_at"],
            host_microvm_id=raw["host_microvm_id"],
            cached_run=_loads(raw["cached_run"]),
            cached_values=_loads(raw["cached_values"]),
            created_at=raw["created_at"] or 0.0,
            last_active_at=raw["last_active_at"] or 0.0,
            cached_thread=_loads(raw["cached_thread"]),
        )
