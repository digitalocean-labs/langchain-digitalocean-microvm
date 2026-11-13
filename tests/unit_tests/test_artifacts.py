"""Tests for artifacts below the gateway: the template's per-task directories,
the agent-host image's artifact routes (images/agent-host/artifacts_app.py),
their client (agent_host.artifacts), and the store migration.

The routes run in-process over ASGI against directories under tmp_path; the
client talks to them through the same `HostGet` the gateway builds. The
gateway routes and the supervisor tool are in test_gateway_artifacts.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from deepagents.backends import LocalShellBackend
from fake_agent_server import artifacts_app_module

from langchain_digitalocean_microvm.agent_host import artifacts as artifacts_module
from langchain_digitalocean_microvm.agent_host.artifacts import (
    ArtifactError,
    ArtifactUpstreamError,
    HostGet,
    list_artifacts,
    outputs_dir,
    read_artifact,
    safe_relative_path,
)
from langchain_digitalocean_microvm.agent_host.store import GatewayStore

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and FIFOs required")

AGENT_PATH = Path(__file__).resolve().parents[2] / "images" / "agent-host" / "agent.py"
MiB = 1024 * 1024
TOKEN = "host-token"


@pytest.fixture
def template(monkeypatch, tmp_path):
    """images/agent-host/agent.py, with SUBAGENT_WORKDIR under tmp_path."""
    monkeypatch.setenv("SUBAGENT_WORKDIR", str(tmp_path / "work"))
    spec = importlib.util.spec_from_file_location("subagent_agent_under_test", AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _app(workdir: Path, expected: str = TOKEN):  # type: ignore[no-untyped-def]
    async def expected_token() -> str:
        return expected

    return artifacts_app_module().create_app(workdir=str(workdir), expected_token=expected_token)


def host_get(workdir: Path, *, token: str = TOKEN, cut_over: int | None = None,
             seen: list[str] | None = None) -> HostGet:
    """The client's view of a host serving `workdir`; `cut_over`: responses larger than that never arrive."""
    app = _app(workdir)

    async def get(path: str, headers: Mapping[str, str]) -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://host") as client:
            response = await client.get(path, headers={"x-api-key": token, **headers})
        if seen is not None:
            seen.append(headers.get("Range", ""))
        if cut_over is not None and len(response.content) > cut_over:
            raise ArtifactUpstreamError("the response did not arrive in time")
        return response

    return get


def raw(workdir: Path, path: str, **headers: str) -> httpx.Response:
    async def go() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(workdir)), base_url="http://host") as c:
            return await c.get(path, headers={"x-api-key": TOKEN, **headers})

    return asyncio.run(go())


def listing(get: HostGet, tid: str = "t1", **kwargs):  # type: ignore[no-untyped-def]
    return asyncio.run(list_artifacts(get, tid, **kwargs))


def read(get: HostGet, path: str, tid: str = "t1", max_bytes: int = MiB, **kwargs) -> bytes:
    return asyncio.run(read_artifact(get, tid, path, max_bytes=max_bytes, **kwargs))


def local_sandbox(root: Path) -> LocalShellBackend:
    """A SandboxBackendProtocol that runs commands on this machine, paths as-is."""
    return LocalShellBackend(root_dir=root, virtual_mode=False, inherit_env=True)


@pytest.fixture
def outputs(tmp_path) -> Path:
    root = tmp_path / "work" / "tasks" / "t1" / "outputs"
    (root / "sub").mkdir(parents=True)
    (root / "report.md").write_text("# report\n")
    (root / "sub" / "data.bin").write_bytes(bytes(range(256)) * 4)
    return root.resolve()


# ---------------------------------------------------------------------------
# The template: per-task directories and echo's "save NAME:"
# ---------------------------------------------------------------------------


def test_template_layout_matches_the_helpers(template) -> None:
    assert (template.TASKS_DIR, template.OUTPUTS_DIR) == (artifacts_module.TASKS_DIR, artifacts_module.OUTPUTS_DIR)


def test_task_dir_is_per_thread_and_created_on_use(template, tmp_path) -> None:
    work = tmp_path / "work"
    assert template.task_dir(None) == work and not (work / "tasks").exists()  # schema calls create nothing
    path = template.task_dir("3f1c-thread")
    assert path == work / "tasks" / "3f1c-thread" and (path / "outputs").is_dir()
    assert str(path / "outputs") == outputs_dir(str(work), "3f1c-thread")
    for bad in ("..", "../x", "a/b", "", "/abs"):
        with pytest.raises(ValueError):
            template.task_dir(bad)


def test_parse_save_request(template) -> None:
    assert template.parse_save_request("save notes.md: hello world") == ("notes.md", "hello world")
    assert template.parse_save_request("SAVE a/b.txt:x") == ("a/b.txt", "x")
    assert template.parse_save_request("hello") == (None, "hello")
    for bad in ("save ../up.md: x", "save /etc/x: y", "save a/../../b: z"):
        with pytest.raises(ValueError):
            template.parse_save_request(bad)


def test_echo_saves_into_the_runs_task_directory(template, tmp_path) -> None:
    config = {"configurable": {"thread_id": "task-a"}}
    result = asyncio.run(template.echo.ainvoke({"messages": [("user", "save out/notes.md: hi there")]}, config))
    assert result["messages"][-1].content == "echo: hi there (saved /outputs/out/notes.md)"
    assert (tmp_path / "work" / "tasks" / "task-a" / "outputs" / "out" / "notes.md").read_text() == "hi there"

    plain = asyncio.run(template.echo.ainvoke({"messages": [("user", "ping")]}, {"configurable": {"thread_id": "task-b"}}))
    assert plain["messages"][-1].content == "echo: ping"
    refused = asyncio.run(template.echo.ainvoke({"messages": [("user", "save ../x: no")]}, config))
    assert refused["messages"][-1].content.startswith("echo: error:")


def test_researcher_backend_is_rooted_at_the_task_directory(template, tmp_path, monkeypatch) -> None:
    import deepagents.backends

    roots: list[Path] = []

    class RecordingBackend(LocalShellBackend):
        def __init__(self, *, root_dir, **kwargs) -> None:
            roots.append(Path(root_dir))
            super().__init__(root_dir=root_dir, **kwargs)

    monkeypatch.setattr(deepagents.backends, "LocalShellBackend", RecordingBackend)
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "not-used")  # the model is built, never called
    template.make_graph({"configurable": {"thread_id": "task-r"}})
    template.make_graph({"configurable": {}})  # schema/introspection: no thread id
    assert roots == [tmp_path / "work" / "tasks" / "task-r", tmp_path / "work"]


# ---------------------------------------------------------------------------
# The routes (artifacts_app.py) and their client (agent_host.artifacts)
# ---------------------------------------------------------------------------


def test_safe_relative_path_normalizes_and_refuses_escapes() -> None:
    assert safe_relative_path("/outputs/report.md") == "report.md"
    assert safe_relative_path("report.md") == "report.md"
    assert safe_relative_path("/sub//data.bin") == "sub/data.bin"
    assert safe_relative_path("./x") == "x"
    for bad in ("", "/", "..", "a/../b", "../etc/passwd"):
        with pytest.raises(ArtifactError) as info:
            safe_relative_path(bad)
        assert info.value.reason == "bad_path"


def test_outputs_dir_refuses_odd_task_ids() -> None:
    assert outputs_dir("/work", "abc-123") == "/work/tasks/abc-123/outputs"
    for bad in ("..", "a/b", "", "-x" * 100):
        with pytest.raises(ValueError):
            outputs_dir("/work", bad)


def test_the_routes_and_the_client_agree_on_their_constants() -> None:
    app = artifacts_app_module()
    assert (app.ROUTE_PREFIX, app.MARKER_HEADER) == (artifacts_module.ROUTE_PREFIX, artifacts_module.MARKER_HEADER)
    assert (app.TASKS_DIR, app.OUTPUTS_DIR, app.DEFAULT_LIMIT) == (
        artifacts_module.TASKS_DIR, artifacts_module.OUTPUTS_DIR, artifacts_module.DEFAULT_MAX_FILES)


def test_list_returns_regular_files_and_skips_symlinks_and_fifos(outputs, tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("host token")
    (outputs / "link.txt").symlink_to(secret)
    (outputs / "linkdir").symlink_to(tmp_path, target_is_directory=True)  # listed by fwalk, never entered
    os.mkfifo(outputs / "pipe")

    result = listing(host_get(tmp_path / "work"))

    assert result.exists and not result.truncated
    assert [(f.path, f.size) for f in result.files] == [("report.md", 9), ("sub/data.bin", 1024)]


def test_list_reports_a_missing_directory_and_truncates(outputs, tmp_path) -> None:
    get = host_get(tmp_path / "work")
    missing = listing(get, "t-never")
    assert not missing.exists and missing.files == ()
    capped = listing(get, max_files=1)
    assert capped.truncated and [f.path for f in capped.files] == ["report.md"]


@pytest.mark.parametrize("swapped", ["outputs", "task", "tasks"])
def test_a_symlink_anywhere_on_the_way_to_outputs_is_refused(tmp_path, swapped: str) -> None:
    """Every component below SUBAGENT_WORKDIR is opened without following a symlink."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "tasks" / "t1" / "outputs").mkdir(parents=True)
    (elsewhere / "tasks" / "t1" / "outputs" / "token").write_text("secret")
    work = tmp_path / "work"
    if swapped == "tasks":
        work.mkdir()
        (work / "tasks").symlink_to(elsewhere / "tasks", target_is_directory=True)
    elif swapped == "task":
        (work / "tasks").mkdir(parents=True)
        (work / "tasks" / "t1").symlink_to(elsewhere / "tasks" / "t1", target_is_directory=True)
    else:
        (work / "tasks" / "t1").mkdir(parents=True)
        (work / "tasks" / "t1" / "outputs").symlink_to(elsewhere / "tasks" / "t1" / "outputs", target_is_directory=True)
    get = host_get(work)
    with pytest.raises(ArtifactError) as info:
        listing(get)
    assert info.value.reason == "symlink" and info.value.status == 403
    with pytest.raises(ArtifactError) as info:
        read(get, "token")
    assert info.value.reason == "symlink"


def test_read_returns_exact_bytes_in_range_pages(outputs, tmp_path) -> None:
    seen: list[str] = []
    get = host_get(tmp_path / "work", seen=seen)
    assert read(get, "/outputs/report.md") == b"# report\n"
    seen.clear()
    assert read(get, "sub/data.bin", page_bytes=300) == bytes(range(256)) * 4
    assert seen == ["bytes=0-299", "bytes=300-599", "bytes=600-899", "bytes=900-1199"]
    (outputs / "empty").write_bytes(b"")
    assert read(get, "empty") == b""


def test_a_page_that_does_not_arrive_is_asked_for_at_half_the_size(outputs, tmp_path) -> None:
    data = os.urandom(5000)
    (outputs / "blob.bin").write_bytes(data)
    seen: list[str] = []
    get = host_get(tmp_path / "work", cut_over=1500, seen=seen)
    assert read(get, "blob.bin", page_bytes=4096, min_page_bytes=512) == data
    assert seen[:3] == ["bytes=0-4095", "bytes=0-2047", "bytes=0-1023"]
    with pytest.raises(ArtifactUpstreamError):  # even the smallest page does not get through
        read(host_get(tmp_path / "work", cut_over=100), "blob.bin", page_bytes=4096, min_page_bytes=512)


def test_read_refuses_symlinks_large_files_and_non_files(outputs, tmp_path) -> None:
    get = host_get(tmp_path / "work")
    (tmp_path / "secret.txt").write_text("host token")
    (outputs / "link.txt").symlink_to(tmp_path / "secret.txt")
    (outputs / "via").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(outputs / "pipe")

    def reason(path: str, max_bytes: int = MiB) -> str:
        with pytest.raises(ArtifactError) as info:
            read(get, path, max_bytes=max_bytes)
        return info.value.reason

    assert reason("link.txt") == "symlink"
    assert reason("via/secret.txt") == "symlink"
    assert reason("missing.md") == "not_found"
    assert reason("report.md/x") == "symlink"  # a file where a directory belongs
    assert reason("sub") == "not_a_file"
    assert reason("pipe") == "not_a_file"  # opened non-blocking, so a FIFO cannot hang the read
    assert reason("sub/data.bin", max_bytes=100) == "too_large"
    assert reason("../secret.txt") == "bad_path"


def test_the_routes_refuse_requests_without_the_host_token(outputs, tmp_path) -> None:
    for get in (host_get(tmp_path / "work", token="wrong"), host_get(tmp_path / "work", token="")):
        with pytest.raises(ArtifactError) as info:
            listing(get)
        assert info.value.reason == "unauthorized" and info.value.status == 401
    nobody = artifacts_app_module().create_app(workdir=str(tmp_path / "work"), expected_token=_no_token)

    async def go() -> int:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=nobody), base_url="http://host") as c:
            return (await c.get("/microvm/artifacts/t1", headers={"x-api-key": ""})).status_code

    assert asyncio.run(go()) == 401  # no token configured: nothing gets in


async def _no_token() -> str:
    return ""


def test_range_answers(outputs, tmp_path) -> None:
    work = tmp_path / "work"
    url = "/microvm/artifacts/t1/files/sub/data.bin"
    whole = raw(work, url)
    assert whole.status_code == 200 and len(whole.content) == 1024 and whole.headers["x-microvm-artifacts"] == "1"
    part = raw(work, url, range="bytes=1000-")
    assert part.status_code == 206 and part.content == (bytes(range(256)) * 4)[1000:]
    assert part.headers["content-range"] == "bytes 1000-1023/1024" and part.headers["x-file-size"] == "1024"
    assert raw(work, url, range="bytes=2000-2100").status_code == 416
    assert raw(work, url, range="bytes=5-1").status_code == 416
    assert raw(work, url, range="items=0-1").status_code == 416
    assert raw(work, "/microvm/artifacts/..%2Fx").status_code in (400, 404)
    assert raw(work, "/microvm/artifacts/t1/files/a%2F..%2F..%2Fsecret").json()["error"] == "bad_path"


def test_an_answer_without_the_marker_is_not_served(tmp_path) -> None:
    async def edge_404(path: str, headers: Mapping[str, str]) -> httpx.Response:
        return httpx.Response(404, text="<html>Not found</html>")

    with pytest.raises(ArtifactError) as info:
        listing(edge_404)
    assert info.value.reason == "not_served" and info.value.status == 404


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_adds_the_host_microvm_column_to_an_existing_file(tmp_path) -> None:
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE threads (thread_id TEXT PRIMARY KEY, graph_id TEXT, host_id TEXT, run_id TEXT, run_status TEXT,"
        " run_started_at TEXT, cached_run TEXT, cached_values TEXT, created_at REAL, last_active_at REAL);"
        "INSERT INTO threads (thread_id, created_at, last_active_at) VALUES ('old', 1, 1);"
    )
    conn.close()

    store = GatewayStore(path)
    try:
        assert store.get_thread("old").host_microvm_id is None
        store.update_thread("old", host_microvm_id="mv-1")
        assert store.get_thread("old").host_microvm_id == "mv-1"
    finally:
        store.close()
    GatewayStore(path).close()  # reopening an already-migrated file is a no-op
