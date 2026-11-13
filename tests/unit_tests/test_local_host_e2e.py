"""End-to-end: LocalDevServerHost really spawns `langgraph dev` from the template.

Marked `e2e` (deselected by default; `uv run pytest -m e2e`) because it boots an Agent Server (several seconds) and
needs the `agent-host` extra installed (`uv sync --extra agent-host`). It
proves, against the real in-memory server:

- the agent-host template loads and serves the `echo` graph;
- auth.py rejects a wrong or missing token (401) on Agent Protocol routes,
  including when the request carries `x-auth-scheme: langsmith` (the header
  the deepagents middleware sends, which would bypass custom auth without
  `disable_studio_auth: true`), and accepts the host token;
- the meta routes (/ok, /metrics) are reachable without a token, which is what
  readiness probes and idle detection rely on;
- the five Agent Protocol calls the deepagents middleware makes work through
  the LangGraph SDK against a host;
- SIGSTOP freezes the server (a request hangs), SIGCONT brings it back with
  its state intact.

The server runs from this environment by default. To check the agent-host
image's own locked dependencies instead, point AGENT_HOST_PYTHON at the
interpreter of that environment:

    uv sync --project images/agent-host
    AGENT_HOST_PYTHON=images/agent-host/.venv/bin/python \
        uv run pytest -m e2e tests/unit_tests/test_local_host_e2e.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    HostStatus,
    LocalDevServerHost,
)

pytestmark = pytest.mark.e2e

langgraph_sdk = pytest.importorskip("langgraph_sdk")
pytest.importorskip("langgraph_cli")


def _host(tmp_path) -> LocalDevServerHost:
    """A LocalDevServerHost; with AGENT_HOST_PYTHON, its `langgraph dev` runs on that interpreter (not resolved: a venv's python is a symlink)."""
    python = os.environ.get("AGENT_HOST_PYTHON")
    command = [str(Path(python).absolute()), "-m", "langgraph_cli", "dev"] if python else None
    return LocalDevServerHost(root_dir=tmp_path / "hosts", ready_timeout=120.0, command=command)


def _wait_run(client, thread_id: str, run_id: str, *, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        run = client.runs.get(thread_id=thread_id, run_id=run_id)
        if run["status"] in {"success", "error", "timeout", "interrupted"}:
            return run
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} still {run['status']} after {timeout}s")
        time.sleep(0.5)


def _search(url: str, headers: dict[str, str]) -> int:
    return httpx.post(f"{url}/threads/search", json={}, headers=headers, timeout=10).status_code


def test_echo_graph_over_agent_protocol_with_pause_and_resume(tmp_path) -> None:
    with _host(tmp_path) as host:
        handle = host.get_or_create(host_id="e2e", spec=HostSpec(graph_id="echo", n_jobs_per_worker=2))
        assert handle.status is HostStatus.READY

        # Meta routes are unauthenticated by design (readiness, idle detection).
        assert httpx.get(f"{handle.url}/ok", timeout=5).status_code == 200
        assert httpx.get(f"{handle.url}/metrics", params={"format": "json"}, timeout=5).status_code == 200

        # Agent Protocol routes require the host token -- and the Studio bypass is off.
        assert _search(handle.url, {}) == 401
        assert _search(handle.url, {"x-api-key": "wrong"}) == 401
        assert _search(handle.url, {"x-auth-scheme": "langsmith"}) == 401
        assert _search(handle.url, {"x-auth-scheme": "langsmith", "x-api-key": "wrong"}) == 401
        assert _search(handle.url, {"x-api-key": handle.token}) == 200
        assert _search(handle.url, {"x-auth-scheme": "langsmith", "x-api-key": handle.token}) == 200

        # The exact calls AsyncSubAgentMiddleware makes, via the SDK.
        client = langgraph_sdk.get_sync_client(url=handle.url, api_key=handle.token)
        thread = client.threads.create()
        run = client.runs.create(
            thread["thread_id"],
            assistant_id="echo",
            input={"messages": [{"role": "user", "content": "ping"}]},
        )
        finished = _wait_run(client, thread["thread_id"], run["run_id"])
        assert finished["status"] == "success"
        state = client.threads.get(thread["thread_id"])
        assert state["values"]["messages"][-1]["content"] == "echo: ping"

        # The artifact routes (artifacts_app.py) are served by the same server, behind the same token.
        tid = thread["thread_id"]
        listing = f"{handle.url}/microvm/artifacts/{tid}"
        for headers in ({}, {"x-api-key": "wrong"}, {"x-auth-scheme": "langsmith"}):
            assert httpx.get(listing, headers=headers, timeout=10).status_code == 401
        empty = httpx.get(listing, headers={"x-api-key": handle.token}, timeout=10)
        assert empty.status_code == 200 and empty.headers["x-microvm-artifacts"] == "1"
        outputs = host.workdir_for("e2e") / "tasks" / tid / "outputs"
        assert empty.json() == {"exists": outputs.is_dir(), "files": [], "truncated": False}
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "r.md").write_text("report")
        page = httpx.get(f"{listing}/files/r.md", headers={"x-api-key": handle.token, "Range": "bytes=2-"}, timeout=10)
        assert page.status_code == 206 and page.content == b"port" and page.headers["x-file-size"] == "6"

        # Pause: the process is SIGSTOPped, so a request hangs rather than failing.
        host.pause(host_id="e2e")
        with pytest.raises(httpx.TimeoutException):
            httpx.get(f"{handle.url}/ok", timeout=2)

        # Resume: same server, same state.
        resumed = host.resume(host_id="e2e")
        assert resumed.status is HostStatus.READY
        state = client.threads.get(thread["thread_id"])
        assert state["values"]["messages"][-1]["content"] == "echo: ping"

        # A second run on the same thread (what update_async_task does).
        run2 = client.runs.create(
            thread["thread_id"],
            assistant_id="echo",
            input={"messages": [{"role": "user", "content": "pong"}]},
            multitask_strategy="interrupt",
        )
        assert _wait_run(client, thread["thread_id"], run2["run_id"])["status"] == "success"
        assert client.threads.get(thread["thread_id"])["values"]["messages"][-1]["content"] == "echo: pong"

        # cancel on a finished run is a 404 from the Agent Server (nothing left to
        # cancel). The gateway must answer cancel_async_task locally for
        # terminal runs instead of proxying this call.
        with pytest.raises(langgraph_sdk.errors.NotFoundError):
            client.runs.cancel(thread["thread_id"], run2["run_id"])

        # is_idle reads the real /metrics endpoint.
        assert host.is_idle(resumed) is True

    assert host.list_hosts() == []


def test_echo_sleep_prefix_delays_the_answer_and_cancel_interrupts_it(tmp_path) -> None:
    """`sleep N:` is what lets a lab check a task while it runs; cancel must cut it short."""
    with _host(tmp_path) as host:
        handle = host.get_or_create(host_id="e2e-sleep", spec=HostSpec(graph_id="echo", n_jobs_per_worker=2))
        client = langgraph_sdk.get_sync_client(url=handle.url, api_key=handle.token)

        thread = client.threads.create()
        started = time.monotonic()
        run = client.runs.create(
            thread["thread_id"],
            assistant_id="echo",
            input={"messages": [{"role": "user", "content": "sleep 4: slow"}]},
        )
        assert client.runs.get(thread["thread_id"], run["run_id"])["status"] in {"pending", "running"}
        assert _wait_run(client, thread["thread_id"], run["run_id"])["status"] == "success"
        assert time.monotonic() - started >= 4.0
        assert client.threads.get(thread["thread_id"])["values"]["messages"][-1]["content"] == "echo: slow"

        other = client.threads.create()
        run2 = client.runs.create(
            other["thread_id"],
            assistant_id="echo",
            input={"messages": [{"role": "user", "content": "sleep 60: never"}]},
        )
        time.sleep(1.0)
        client.runs.cancel(other["thread_id"], run2["run_id"])
        assert _wait_run(client, other["thread_id"], run2["run_id"], timeout=20.0)["status"] == "interrupted"
