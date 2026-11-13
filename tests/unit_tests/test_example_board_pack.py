"""examples/08_board_pack: the orchestrator's tools stay off the event loop.

`langgraph dev` runs graphs under blockbuster, which raises on blocking calls (disk I/O, sleeps,
subprocesses) made on the event loop. `run_script.py` has no such guard, so a blocking call only
shows up under the server; this test runs the tools with blockbuster active instead.
"""

from __future__ import annotations

import importlib.util
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

blockbuster = pytest.importorskip("blockbuster")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "08_board_pack"


def _load_tools():
    spec = importlib.util.spec_from_file_location("board_pack_tools", EXAMPLE / "tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Backend:
    """The slice of a sandbox backend the orchestrator's tools use, in memory."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = dict(files or {})

    async def aglob(self, pattern: str, path: str = "/"):
        return SimpleNamespace(matches=[{"path": p} for p in self.files if p.startswith(path)])

    def download_files(self, paths: list[str]):
        return [SimpleNamespace(path=p, content=self.files.get(p), error=None if p in self.files else "file_not_found")
                for p in paths]

    def upload_files(self, files: list[tuple[str, bytes]]):
        self.files.update(files)
        return [SimpleNamespace(path=p, error=None) for p, _ in files]


async def test_orchestrator_tools_do_not_block_the_event_loop(tmp_path, monkeypatch):
    tools = _load_tools()
    monkeypatch.setattr(tools, "HERE", tmp_path)
    monkeypatch.delenv("BOARD_PACK_OPEN", raising=False)
    opened: list[tuple[str, bool]] = []
    loop_thread = threading.current_thread()
    monkeypatch.setattr(tools.webbrowser, "open", lambda url: opened.append((url, threading.current_thread() is loop_thread)))

    analyst = _Backend({"/outputs/europe.md": b"# Europe", "/outputs/europe.png": b"\x89PNG", "/outputs/notes.txt": b"x"})
    publisher = _Backend({"/outputs/board-pack.html": b"<html>pack</html>"})
    sandboxes = SimpleNamespace(backends={"analyst-europe": analyst, "publisher": publisher})
    handoff_files, export_board_pack = tools.orchestrator_tools(sandboxes, "thread-1")

    with blockbuster.blockbuster_ctx():
        handed = await handoff_files.ainvoke({"source": "analyst-europe", "target": "publisher"})
        exported = await export_board_pack.ainvoke({})

    assert "Copied 2 file(s)" in handed
    assert publisher.files["/inputs/analyst-europe/europe.md"] == b"# Europe"
    target = tmp_path / "outputs" / "thread-1" / "board-pack.html"
    assert target.read_bytes() == b"<html>pack</html>"
    assert exported.endswith("and opened it in the browser.")
    assert opened == [(target.as_uri(), False)]  # opened, from a worker thread

    monkeypatch.setenv("BOARD_PACK_OPEN", "0")
    exported = await export_board_pack.ainvoke({})
    assert exported == f"Exported to {target} ({len(b'<html>pack</html>'):,} bytes)."
    assert len(opened) == 1


async def test_specialist_tool_in_fake_mode_does_not_block_the_event_loop(tmp_path, monkeypatch):
    tools = _load_tools()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # local roots live under the temp dir
    commands: list[str] = []

    class _Sandbox:
        id = "board-pack-test-sandbox"

        async def aexecute(self, command: str, timeout: int | None = None):
            commands.append(command)
            return SimpleNamespace(exit_code=0, output='{"rows": []}')

    query_chinook = tools.query_tool(_Sandbox(), local=True)
    with blockbuster.blockbuster_ctx():
        result = await query_chinook.ainvoke({"sql": "SELECT 1"})

    assert result == '{"rows": []}'
    assert f"{tools.root_for_host(_Sandbox.id)}/tools/board_sql.py" in commands[0]


async def test_blockbuster_catches_disk_io_on_the_loop(tmp_path):
    """The guard above is live: the same write made on the loop raises."""
    with blockbuster.blockbuster_ctx(), pytest.raises(blockbuster.BlockingError):
        (tmp_path / "x").mkdir()
