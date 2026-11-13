"""End-to-end (marked `e2e`): the subagent-sandbox graph factory under a real `langgraph dev`.

The server enters `tests/unit_tests/fixtures/subagent_sandboxes/graph.py:make_graph` for every run,
so this checks what the unit tests cannot: that the Agent Server's
context-manager factory protocol provisions the thread's sandbox before the
run and pauses it after (store rows written by the server process), that the
subagent's file persists across runs in its (local) sandbox, and that the
parent reads it through the mount on a later run. A model-free policy model
drives the tools.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e
langgraph_sdk = pytest.importorskip("langgraph_sdk")
pytest.importorskip("langgraph_cli")

from langchain_digitalocean_microvm.agent_host.store import GatewayStore  # noqa: E402
from langchain_digitalocean_microvm.subagents import subagent_sandbox_id  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "subagent_sandboxes" / "graph.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_graph_factory_under_langgraph_dev(tmp_path) -> None:
    shutil.copy(FIXTURE, tmp_path / "graph.py")
    (tmp_path / "langgraph.json").write_text(json.dumps({"dependencies": ["."], "graphs": {"agent": "./graph.py:make_graph"}}))
    db = tmp_path / "store.sqlite"
    fake_tmp = tmp_path / "tmp"
    fake_tmp.mkdir()
    port = _free_port()
    env = {**os.environ, "PHASE2_E2E_DB": str(db), "TMPDIR": str(fake_tmp), "LANGGRAPH_DISABLE_FILE_PERSISTENCE": "true"}
    env.pop("LANGSMITH_API_KEY", None)
    log = (tmp_path / "server.log").open("w")
    server = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "langgraph_cli", "dev", "--config", str(tmp_path / "langgraph.json"), "--host", "127.0.0.1",
         "--port", str(port), "--no-browser", "--no-reload", "--allow-blocking"],
        cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/ok", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if server.poll() is not None:
                break
            time.sleep(0.5)
        else:
            pytest.fail("langgraph dev did not come up")
        assert server.poll() is None, (tmp_path / "server.log").read_text()[-3000:]

        client = langgraph_sdk.get_sync_client(url=url, api_key=None)
        thread_id = client.threads.create()["thread_id"]
        sandbox_id = subagent_sandbox_id(thread_id, "reviewer")

        out = client.runs.wait(thread_id, "agent", input={"messages": [{"role": "user", "content": "save hello from run one"}]})
        assert out["messages"][-1]["content"] == "delegated", out
        store = GatewayStore(db)
        row = store.get_sandbox(sandbox_id)
        assert row is not None and row.status == "paused" and row.thread_id == thread_id
        first_use = row.last_active_at
        note = fake_tmp / "microvm_sandbox_local" / sandbox_id / "notes" / "n.md"
        assert note.read_text() == "hello from run one"

        out = client.runs.wait(thread_id, "agent", input={"messages": [{"role": "user", "content": "read the note"}]})
        assert "hello from run one" in out["messages"][-1]["content"], out
        row = store.get_sandbox(sandbox_id)
        assert row is not None and row.status == "paused" and row.last_active_at > first_use
        assert len(store.list_sandboxes()) == 1  # same sandbox on the second run
        store.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
