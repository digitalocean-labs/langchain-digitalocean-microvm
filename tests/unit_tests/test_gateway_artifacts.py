"""Tests for artifacts through the gateway: the /artifacts routes and the
supervisor's fetch_subagent_artifacts tool.

Hosts are FakeAgentServers (fake_agent_server.py) whose filesystems are
directories under tmp_path, served by the real artifact routes of the
agent-host image (artifacts_app.py, in-process), so the gateway runs the real
client and server. The tool runs inside a real
deepagents supervisor driven by a ScriptedChatModel and writes into its
StateBackend.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from deepagents import create_deep_agent
from deepagents.backends.protocol import BackendProtocol, FileUploadResponse
from fake_agent_server import FakeHosts, make_transport
from langgraph.checkpoint.memory import InMemorySaver
from scripted_model import ScriptedChatModel, last_tool_message, say, tool_call
from test_gateway import FakeClock

from langchain_digitalocean_microvm.agent_host import (
    GatewaySettings,
    HostSpec,
    HostStatus,
    LazyGateway,
    SubAgentGateway,
    fetch_artifacts_tool,
)
from langchain_digitalocean_microvm.agent_host.artifacts import outputs_dir
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError

TOKEN = "gw-artifacts-token"
CONFIG = {"configurable": {"thread_id": "supervisor"}}


class FilesystemFakeHosts(FakeHosts):
    """FakeHosts whose hosts have a work directory each and serve the artifact routes."""

    def __init__(self, root: Path) -> None:
        super().__init__(workdir_root=root)

    def save(self, host_id: str, thread_id: str, rel: str, content: bytes | str) -> None:
        """What the subagent does: write under its task's outputs/."""
        path = Path(outputs_dir(str(self.workdir(host_id)), thread_id)) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    def close(self) -> None:
        for host_id in list(self.servers):
            self.delete(host_id=host_id)


@pytest.fixture
def hosts(tmp_path: Path) -> FilesystemFakeHosts:
    return FilesystemFakeHosts(tmp_path / "hosts")


@pytest.fixture
def gw(tmp_path: Path, hosts: FilesystemFakeHosts) -> Iterator[tuple[SubAgentGateway, FakeClock, httpx.Client]]:
    clock = FakeClock()
    gateway = SubAgentGateway(
        hosts,
        {"researcher": HostSpec(graph_id="researcher-graph"), "shared": HostSpec(graph_id="shared-graph", reuse="pooled")},
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        pause_grace=45.0,
        poll_interval=3600.0,
        artifact_max_bytes=1000,
        upstream_transport=make_transport(hosts),
        clock=clock.time,
    )
    url = gateway.start_embedded()
    http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        yield gateway, clock, http
    finally:
        http.close()
        gateway.stop()


def _launch(http: httpx.Client, assistant: str = "researcher") -> tuple[str, str]:
    tid = http.post("/threads", json={}).json()["thread_id"]
    run = http.post(f"/threads/{tid}/runs", json={"assistant_id": assistant, "input": {"messages": []}}).json()
    return tid, run["run_id"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_list_and_fetch_files_of_a_running_task(gw, hosts) -> None:
    gateway, _clock, http = gw
    tid, _rid = _launch(http)
    hosts.save(tid, tid, "report.md", "# findings\n")
    hosts.save(tid, tid, "data/raw.bin", bytes(range(200)))

    listing = http.get(f"/artifacts/{tid}")
    assert listing.status_code == 200
    assert listing.json() == {
        "task_id": tid,
        "directory": "/outputs",
        "exists": True,
        "files": [{"path": "data/raw.bin", "size": 200}, {"path": "report.md", "size": 11}],
        "truncated": False,
        "max_file_bytes": 1000,
    }
    assert http.get(f"/artifacts/{tid}/files/report.md").content == b"# findings\n"
    raw = http.get(f"/artifacts/{tid}/files/data/raw.bin")
    assert raw.headers["content-type"] == "application/octet-stream" and raw.content == bytes(range(200))
    assert all(h.get("x-api-key") == f"tok-{tid}" for h in hosts.servers[tid].artifact_requests)
    assert gateway.store.get_thread(tid).run_status == "pending"  # reading files does not touch the run


def test_artifact_errors_map_to_http_statuses(gw, hosts) -> None:
    _gateway, _clock, http = gw
    tid, _ = _launch(http)
    hosts.save(tid, tid, "big.bin", b"x" * 1001)
    created_only = http.post("/threads", json={}).json()["thread_id"]

    assert httpx.get(f"{str(http.base_url).rstrip('/')}/artifacts/{tid}").status_code == 401  # no token
    assert http.get("/artifacts/nope").status_code == 404
    no_run = http.get(f"/artifacts/{created_only}")
    assert no_run.status_code == 404 and "has not started a run" in no_run.json()["detail"]
    assert http.get(f"/artifacts/{tid}/files/missing.md").status_code == 404
    too_big = http.get(f"/artifacts/{tid}/files/big.bin")
    assert too_big.status_code == 413 and too_big.json()["reason"] == "too_large"


def test_a_task_that_wrote_nothing_has_an_empty_listing(gw, hosts) -> None:
    _gateway, _clock, http = gw
    tid, _ = _launch(http)
    body = http.get(f"/artifacts/{tid}").json()
    assert body["exists"] is False and body["files"] == []


def test_paused_host_is_resumed_for_the_read_and_paused_again(gw, hosts) -> None:
    gateway, clock, http = gw
    tid, rid = _launch(http)
    hosts.save(tid, tid, "report.md", "done")
    hosts.servers[tid].complete(rid, "finished")
    gateway.tick_sync()
    clock.advance(46.0)
    gateway.tick_sync()
    assert hosts.status[tid] == HostStatus.PAUSED

    assert http.get(f"/artifacts/{tid}/files/report.md").content == b"done"
    assert ("resume", tid) in hosts.calls
    assert gateway.store.get_host(tid).status == "ready"
    gateway.tick_sync()
    clock.advance(46.0)
    gateway.tick_sync()
    assert hosts.status[tid] == HostStatus.PAUSED  # tick() put it back to sleep after the grace


def test_deleted_or_evicted_host_reports_the_artifacts_gone(gw, hosts) -> None:
    gateway, _clock, http = gw
    evicted, _ = _launch(http)
    vanished, _ = _launch(http)

    gateway.store.update_host(evicted, status=HostStatus.DESTROYED.value)  # what eviction records
    gone = http.get(f"/artifacts/{evicted}")
    assert gone.status_code == 410 and gone.json()["reason"] == "host_gone"
    assert "are gone" in gone.json()["detail"] and "destroyed" in gone.json()["detail"]

    hosts.delete(host_id=vanished)  # deleted behind the gateway's back (delete_all.py, by hand)
    gone = http.get(f"/artifacts/{vanished}/files/report.md")
    assert gone.status_code == 410 and "are gone" in gone.json()["detail"]
    assert gateway.store.get_host(vanished).status == "destroyed"


def test_a_paused_host_deleted_behind_the_gateways_back_reports_the_artifacts_gone(gw, hosts) -> None:
    gateway, _clock, http = gw
    tid, _ = _launch(http)
    gateway.store.update_host(tid, status=HostStatus.PAUSED.value)
    real_resume = hosts.resume

    def resume(*, host_id: str, spec=None):
        hosts.delete(host_id=host_id)
        raise SandboxNotFoundError(f"No MicroVM registered for host_id={host_id!r}")

    hosts.resume = resume  # type: ignore[method-assign]
    try:
        gone = http.get(f"/artifacts/{tid}")
    finally:
        hosts.resume = real_resume  # type: ignore[method-assign]
    assert gone.status_code == 410 and "are gone" in gone.json()["detail"]
    assert gateway.store.get_host(tid).status == "destroyed"


def test_a_pooled_host_replaced_since_the_task_ran_reports_the_artifacts_gone(gw, hosts) -> None:
    gateway, _clock, http = gw
    tid, _ = _launch(http, "shared")
    host_id = gateway.store.get_thread(tid).host_id
    assert host_id == "pool-shared" and gateway.store.get_thread(tid).host_microvm_id == "fake-pool-shared"
    assert http.get(f"/artifacts/{tid}").status_code == 200

    gateway.store.update_host(host_id, microvm_id="fake-pool-shared-2")  # evicted, then recreated under the same id
    gone = http.get(f"/artifacts/{tid}")
    assert gone.status_code == 410 and "replaced" in gone.json()["detail"]


def test_tasks_on_a_pooled_host_see_only_their_own_outputs(gw, hosts) -> None:
    _gateway, _clock, http = gw
    first, _ = _launch(http, "shared")
    second, _ = _launch(http, "shared")
    hosts.save("pool-shared", first, "a.md", "first")
    hosts.save("pool-shared", second, "b.md", "second")
    assert [f["path"] for f in http.get(f"/artifacts/{first}").json()["files"]] == ["a.md"]
    assert [f["path"] for f in http.get(f"/artifacts/{second}").json()["files"]] == ["b.md"]
    assert http.get(f"/artifacts/{first}/files/b.md").status_code == 404


def test_a_host_whose_image_lacks_the_routes_answers_502(tmp_path) -> None:
    hosts = FakeHosts()  # no work directories: the servers answer the artifact paths with a plain 404
    gateway = SubAgentGateway(hosts, {"r": HostSpec(graph_id="r")}, db_path=tmp_path / "g.sqlite", token=TOKEN,
                              poll_interval=3600.0, upstream_transport=make_transport(hosts))
    url = gateway.start_embedded()
    try:
        with httpx.Client(base_url=url, headers={"x-api-key": TOKEN}) as http:
            tid, _ = _launch(http, "r")
            answer = http.get(f"/artifacts/{tid}")
            assert answer.status_code == 502 and "rebuild it" in answer.json()["detail"]
            assert gateway.store.get_host(tid).status == "busy"  # the host itself is fine: not retired
    finally:
        gateway.stop()


def test_a_host_deleted_outside_the_gateway_answers_410_from_the_edge_404(gw, hosts) -> None:
    gateway, _clock, http = gw
    tid, _ = _launch(http)
    hosts.save(tid, tid, "report.md", "x")
    hosts.delete_outside(tid)  # the edge answers 404 (no marker), the inventory has no entry
    gone = http.get(f"/artifacts/{tid}/files/report.md")
    assert gone.status_code == 410 and gone.json()["reason"] == "host_gone"
    assert gateway.store.get_host(tid).status == "destroyed"


def test_an_unreachable_host_that_is_still_listed_answers_502(gw, hosts) -> None:
    gateway, _clock, http = gw
    tid, _ = _launch(http)
    hosts.servers[tid].dead = True  # connection refused, but the inventory still says READY
    answer = http.get(f"/artifacts/{tid}")
    assert answer.status_code == 502 and "ConnectError" in answer.json()["detail"]
    assert gateway.store.get_host(tid).status == "busy"  # not retired over one failed read


def test_large_files_come_in_range_pages(tmp_path, hosts) -> None:
    gateway = SubAgentGateway(hosts, {"researcher": HostSpec(graph_id="researcher-graph")}, db_path=tmp_path / "g.sqlite",
                              token=TOKEN, poll_interval=3600.0, artifact_page_bytes=300,
                              upstream_transport=make_transport(hosts))
    url = gateway.start_embedded()
    try:
        with httpx.Client(base_url=url, headers={"x-api-key": TOKEN}) as http:
            tid, _ = _launch(http)
            data = bytes(range(256)) * 4  # 1024 bytes: four pages of at most 300
            hosts.save(tid, tid, "blob.bin", data)
            assert http.get(f"/artifacts/{tid}/files/blob.bin").content == data
            ranges = [h["range"] for h in hosts.servers[tid].artifact_requests]
            assert ranges == ["bytes=0-299", "bytes=300-599", "bytes=600-899", "bytes=900-1199"]
    finally:
        gateway.stop()


# ---------------------------------------------------------------------------
# fetch_subagent_artifacts
# ---------------------------------------------------------------------------


class RecordingBackend(BackendProtocol):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.files.update(files)
        return [FileUploadResponse(path=path, error=None) for path, _ in files]


def test_tool_copies_selected_files_and_explains_the_rest(gw, hosts) -> None:
    gateway, _clock, http = gw
    tid, _ = _launch(http)
    hosts.save(tid, tid, "report.md", "# r\n")
    hosts.save(tid, tid, "chart.png", b"\x89PNG\x00\xff")
    hosts.save(tid, tid, "big.bin", b"x" * 1001)
    backend = RecordingBackend()
    tool = fetch_artifacts_tool(gateway_url=gateway.url, gateway_token=TOKEN, backend=backend, max_total_bytes=500)

    text = tool.invoke({"task_id": tid, "paths": ["/outputs/report.md", "chart.png", "big.bin", "nope.md", "../x"]})

    assert backend.files == {f"/subagents/{tid}/report.md": b"# r\n", f"/subagents/{tid}/chart.png": b"\x89PNG\x00\xff"}
    assert f"/subagents/{tid}/report.md (4 bytes)" in text
    assert "chart.png (6 bytes, binary: stored base64-encoded)" in text
    assert "big.bin: skipped, 1001 bytes would exceed" in text  # the tool's own budget, before the gateway's 413
    assert "nope.md: not found" in text and "../x: not a path inside /outputs/" in text

    roomy = fetch_artifacts_tool(gateway_url=gateway.url, gateway_token=TOKEN, backend=backend)
    assert "big.bin is 1001 bytes; the limit is 1000 (HTTP 413)" in roomy.invoke({"task_id": tid, "paths": ["big.bin"]})
    assert "Could not list" in fetch_artifacts_tool(gateway_url=gateway.url, gateway_token="wrong").invoke({"task_id": tid})


def make_lazy(tmp_path: Path, hosts: FakeHosts) -> LazyGateway:
    return LazyGateway(
        {"echo": HostSpec(graph_id="echo-graph")},
        settings=GatewaySettings(host_mode="local", token=TOKEN, db_path=tmp_path / "lazy.sqlite"),
        hosts_factory=lambda mode: hosts,
        register_atexit=False,
        export_sdk_api_key=False,
        poll_interval=3600.0,
        upstream_transport=make_transport(hosts),
    )


def _turn(agent, model: ScriptedChatModel, *steps, run_async: bool = False) -> dict[str, Any]:
    model.script(*steps)
    payload = {"messages": [{"role": "user", "content": "go"}]}
    if run_async:
        return asyncio.run(agent.ainvoke(payload, config=CONFIG))
    return agent.invoke(payload, config=CONFIG)


def test_supervisor_fetches_artifacts_into_its_own_filesystem(tmp_path, hosts) -> None:
    lazy = make_lazy(tmp_path, hosts)
    model = ScriptedChatModel()
    agent = create_deep_agent(
        model=model,
        subagents=[lazy.subagent(name="echo", description="Replies 'echo: <message>'")],
        middleware=[lazy.middleware()],
        tools=[lazy.artifacts_tool()],
        checkpointer=InMemorySaver(),
    )
    try:
        state = _turn(agent, model, tool_call("start_async_task", description="write a report", subagent_type="echo"), say("ok"))
        [tid] = state["async_tasks"]
        hosts.save(tid, tid, "report.md", "# report\nall good\n")
        hosts.save(tid, tid, "notes/n1.txt", "n1")

        state = _turn(agent, model, tool_call("fetch_subagent_artifacts", task_id=tid), say("fetched"))
        message = last_tool_message(state["messages"])
        assert f"Copied 2 file(s) from task {tid}" in message.content
        assert state["files"][f"/subagents/{tid}/report.md"]["content"] == "# report\nall good\n"
        assert state["files"][f"/subagents/{tid}/notes/n1.txt"]["content"] == "n1"

        # The supervisor can now read it with its own filesystem tools (async path this time).
        state = _turn(agent, model, tool_call("read_file", file_path=f"/subagents/{tid}/report.md"), say("read"), run_async=True)
        assert "all good" in last_tool_message(state["messages"]).content

        hosts.save(tid, tid, "late.md", "later")
        state = _turn(agent, model, tool_call("fetch_subagent_artifacts", task_id=tid, paths=["/outputs/late.md"]), say("ok"),
                      run_async=True)
        assert state["files"][f"/subagents/{tid}/late.md"]["content"] == "later"

        hosts.delete(host_id=tid)
        state = _turn(agent, model, tool_call("fetch_subagent_artifacts", task_id=tid), say("gone"))
        assert "are gone" in last_tool_message(state["messages"]).content
    finally:
        lazy.close()
