"""`update_host` / `update_thread` set only the table's own columns.

Their SET clause is built from argument names, so a name that is not a column
(an injection attempt included) is refused before any SQL runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    HostRow,
)

INJECTION = "status = status WHERE 1=1 AND ? AND ? --"


@pytest.fixture
def store(tmp_path: Path) -> GatewayStore:
    store = GatewayStore(tmp_path / "store.sqlite3")
    for host_id in ("h1", "h2"):
        store.upsert_host(HostRow(host_id, f"s-{host_id}", "echo", "ready", "https://x", "tok", "", 1.0, None, 0))
    for thread_id in ("t1", "t2"):
        store.create_thread(thread_id, now=1.0)
    return store


def test_known_columns_are_updated(store: GatewayStore) -> None:
    store.update_host("h1", status="paused", failures=2)
    store.update_thread("t1", run_status="success", cached_values={"a": 1})

    host = store.get_host("h1")
    thread = store.get_thread("t1")
    assert host is not None and (host.status, host.failures) == ("paused", 2)
    assert thread is not None and (thread.run_status, thread.cached_values) == ("success", {"a": 1})


@pytest.mark.parametrize("name", ["nope", INJECTION])
def test_update_host_refuses_other_names(store: GatewayStore, name: str) -> None:
    with pytest.raises(ValueError, match="unknown hosts columns"):
        store.update_host("h1", **{name: "x"})
    assert [h.status for h in store.list_hosts()] == ["ready", "ready"]


@pytest.mark.parametrize("name", ["nope", INJECTION])
def test_update_thread_refuses_other_names(store: GatewayStore, name: str) -> None:
    with pytest.raises(ValueError, match="unknown threads columns"):
        store.update_thread("t1", run_status="success", **{name: "x"})
    assert [t.run_status for t in store.list_threads()] == [None, None]


def test_a_store_from_before_cached_thread_gains_the_column(tmp_path: Path) -> None:
    """A store created before `cached_thread` existed is upgraded when it is opened; its rows are kept."""
    import sqlite3

    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE threads (thread_id TEXT PRIMARY KEY, graph_id TEXT, host_id TEXT, run_id TEXT, run_status TEXT,
            run_started_at TEXT, host_microvm_id TEXT, cached_run TEXT, cached_values TEXT, created_at REAL,
            last_active_at REAL);
        INSERT INTO threads (thread_id, run_status, created_at, last_active_at) VALUES ('t-old', 'success', 1.0, 1.0);
    """)
    conn.close()

    store = GatewayStore(path)
    old = store.get_thread("t-old")
    assert old is not None and old.run_status == "success" and old.cached_thread is None
    store.update_thread("t-old", cached_thread={"status": "interrupted", "interrupts": {"n": [{"id": "i", "value": 1}]}})
    upgraded = store.get_thread("t-old")
    assert upgraded is not None
    assert upgraded.cached_thread == {"status": "interrupted", "interrupts": {"n": [{"id": "i", "value": 1}]}}
    store.close()
    GatewayStore(path).close()  # opening it again changes nothing
