"""End-to-end: SDK -> embedded gateway -> LocalDevServerHost running `langgraph dev`.

Marked `e2e` (`uv run pytest -m e2e`). The whole agent-host chain with real processes: the gateway
allocates a local mode host for the `echo` graph, forwards the run, polls it to
completion, serves `check` from cache after pausing the host (SIGSTOP), wakes
it for an update, and turns a deleted host into an error run.
"""

from __future__ import annotations

import time

import pytest

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    LocalDevServerHost,
    SubAgentGateway,
)

pytestmark = pytest.mark.e2e

langgraph_sdk = pytest.importorskip("langgraph_sdk")
pytest.importorskip("langgraph_cli")

TOKEN = "gw-e2e-token"


def _wait_terminal(gateway: SubAgentGateway, client, thread_id: str, run_id: str, *, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        gateway.tick_sync()  # polling fallback (no webhook in this test)
        run = client.runs.get(thread_id=thread_id, run_id=run_id)
        if run["status"] in {"success", "error", "timeout", "interrupted"}:
            return run
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} still {run['status']} after {timeout}s")
        time.sleep(0.5)


def test_full_chain_with_a_real_local_agent_server(tmp_path) -> None:
    hosts = LocalDevServerHost(root_dir=tmp_path / "hosts", ready_timeout=120.0)
    gateway = SubAgentGateway(
        hosts,
        {"echo": HostSpec(graph_id="echo", n_jobs_per_worker=2)},
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        pause_grace=0.0,          # pause as soon as a tick sees the host idle
        poll_interval=3600.0,     # ticks are driven by the test
    )
    url = gateway.start_embedded()
    try:
        client = langgraph_sdk.get_sync_client(url=url, api_key=TOKEN)

        thread = client.threads.create()
        tid = thread["thread_id"]
        run = client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "ping"}]})
        assert hosts.list_hosts()[0].host_id == tid  # one real langgraph dev per task

        finished = _wait_terminal(gateway, client, tid, run["run_id"])
        assert finished["status"] == "success"
        assert client.threads.get(tid)["values"]["messages"][-1]["content"] == "echo: ping"

        # Idle host gets paused (SIGSTOP) by the maintenance tick; checks still work from cache.
        gateway.tick_sync()
        gateway.tick_sync()
        assert gateway.store.get_host(tid).status == "paused"
        assert client.runs.get(tid, run["run_id"])["status"] == "success"
        assert client.threads.get(tid)["values"]["messages"][-1]["content"] == "echo: ping"
        assert gateway.store.get_host(tid).status == "paused"

        # Update: the gateway wakes the host (SIGCONT) and forwards with interrupt semantics.
        run2 = client.runs.create(
            tid,
            assistant_id="echo",
            input={"messages": [{"role": "user", "content": "pong"}]},
            multitask_strategy="interrupt",
        )
        assert gateway.store.get_host(tid).status == "busy"
        assert _wait_terminal(gateway, client, tid, run2["run_id"])["status"] == "success"
        assert client.threads.get(tid)["values"]["messages"][-1]["content"] == "echo: pong"

        # A third run, then the host disappears mid-flight: the task becomes an error, not a hang.
        gateway.tick_sync()
        gateway.tick_sync()
        run3 = client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "gone"}]})
        hosts.delete(host_id=tid)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.runs.get(tid, run3["run_id"])["status"]
            if status == "error":
                break
            time.sleep(0.5)
        checked = client.runs.get(tid, run3["run_id"])
        assert checked["status"] == "error"
        assert "unreachable" in checked["error"] or "gone" in checked["error"]
    finally:
        gateway.stop()
        hosts.close()


def test_the_researcher_factory_gets_its_runs_thread_id(tmp_path) -> None:
    """Artifacts: the Agent Server builds `researcher` per run with the run's config, so its root is the task's directory.

    The model endpoint refuses connections, so the run fails after the factory ran; no model key is needed.
    """
    hosts = LocalDevServerHost(root_dir=tmp_path / "hosts", ready_timeout=120.0)
    spec = HostSpec(
        graph_id="researcher",
        env={"DO_MODEL_ACCESS_KEY": "not-a-key", "SUBAGENT_INFERENCE_BASE_URL": "http://127.0.0.1:9"},
    )
    gateway = SubAgentGateway(hosts, {"researcher": spec}, db_path=tmp_path / "gateway.sqlite", token=TOKEN,
                              pause_grace=3600.0, poll_interval=3600.0)
    url = gateway.start_embedded()
    try:
        client = langgraph_sdk.get_sync_client(url=url, api_key=TOKEN)
        tid = client.threads.create()["thread_id"]
        run = client.runs.create(tid, assistant_id="researcher", input={"messages": [{"role": "user", "content": "hi"}]})
        finished = _wait_terminal(gateway, client, tid, run["run_id"], timeout=120.0)
        assert finished["status"] == "error"  # no model behind the URL
        assert (hosts.workdir_for(tid) / "tasks" / tid / "outputs").is_dir()
        assert sorted(p.name for p in (hosts.workdir_for(tid) / "tasks").iterdir()) == [tid]
    finally:
        gateway.stop()
        hosts.close()


def test_an_idle_host_is_deleted_after_its_ttl_and_its_task_stays_checkable(tmp_path) -> None:
    """Lifecycle: the gateway's own poll loop pauses (SIGSTOP), then deletes, a real local host past `release_after`."""
    hosts = LocalDevServerHost(root_dir=tmp_path / "hosts", ready_timeout=120.0)
    gateway = SubAgentGateway(
        hosts,
        {"echo": HostSpec(graph_id="echo")},
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        pause_grace=0.0,
        poll_interval=0.5,     # the background loop does the work, as in the lab
        release_after=4.0,
    )
    url = gateway.start_embedded()
    try:
        client = langgraph_sdk.get_sync_client(url=url, api_key=TOKEN)
        tid = client.threads.create()["thread_id"]
        run = client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "save a.txt: kept"}]})
        assert _wait_terminal(gateway, client, tid, run["run_id"])["status"] == "success"
        http = langgraph_sdk.get_sync_client(url=url, api_key=TOKEN).http
        assert [f["path"] for f in http.get(f"/artifacts/{tid}")["files"]] == ["a.txt"]
        workdir = hosts.workdir_for(tid)
        assert workdir.is_dir()

        deadline = time.monotonic() + 30
        while gateway.store.get_host(tid).status != "destroyed":
            assert time.monotonic() < deadline, gateway.store.get_host(tid)
            time.sleep(0.25)

        assert gateway.store.get_host(tid).retired_reason == "ttl"
        assert hosts.list_hosts() == [] and not workdir.exists()  # process and files gone
        assert client.runs.get(tid, run["run_id"])["status"] == "success"  # still served from the cache
        assert client.threads.get(tid)["values"]["messages"][-1]["content"].startswith("echo:")
        with pytest.raises(langgraph_sdk.errors.APIStatusError) as excinfo:
            http.get(f"/artifacts/{tid}")
        assert excinfo.value.response.status_code == 410
        assert "time-to-live" in excinfo.value.response.text
    finally:
        gateway.stop()
        hosts.close()
