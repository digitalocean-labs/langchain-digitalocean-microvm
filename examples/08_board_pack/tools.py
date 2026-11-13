"""The board pack's tools: the specialists' run inside their MicroVMs, the orchestrator's move files between them.

Specialist tools are bound to one subagent's sandbox and run a helper script there (sandbox/,
seeded into the template under /tools/): SQL on the VM's own copy of the Chinook database,
charts with the VM's matplotlib, the HTML render. Arguments travel as one base64 JSON blob, so
model-written SQL and text never go through the shell's quoting.

The orchestrator has no shell. Its two tools copy files from one subagent's VM to another's
(`handoff_files`) and from the publisher's VM to this machine (`export_board_pack`).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import sys
import webbrowser
from pathlib import Path
from typing import Any

from langchain.tools import tool

from langchain_digitalocean_microvm.subagents import SubagentSandboxes
from langchain_digitalocean_microvm.transport.local import root_for_host

HERE = Path(__file__).resolve().parent


def _script(backend: Any, path: str, local: bool) -> str:
    """`path` as a command argument: a local sandbox is a local directory, so prefix its root."""
    return f"{root_for_host(backend.id)}{path}" if local else path


async def _run(backend: Any, script: str, args: list[str], *, local: bool, payload: dict | None = None) -> str:
    """Run a /tools/ helper in the sandbox and return its JSON output (or a JSON error)."""
    if hasattr(backend, "aget"):  # OnDemandSandbox: start the VM first, so a local root exists
        try:
            backend = await backend.aget()
        except Exception as exc:  # noqa: BLE001 - reported to the agent
            return json.dumps({"error": f"sandbox could not be started: {exc}"})
    argv = [*args]
    if payload is not None:
        argv.append(base64.b64encode(json.dumps(payload).encode()).decode())
    # local mode: this process's interpreter (a login shell's python3 may lack matplotlib/markdown), without
    # the local transport's path shim (it would also redirect matplotlib's own files; the helpers resolve
    # their paths from their location)
    python = f"env -u MICROVM_LOCAL_ROOT {shlex.quote(sys.executable)}" if local else "python3"
    # root_for_host creates the local root: disk I/O, so off the event loop (blockbuster under `langgraph dev`)
    path = await asyncio.to_thread(_script, backend, script, local) if local else script
    command = " ".join([python, shlex.quote(path), *map(shlex.quote, argv)])
    result = await backend.aexecute(command, timeout=120)
    if result.exit_code != 0:
        return json.dumps({"error": f"sandbox command failed ({result.exit_code}): {result.output.strip()[-800:]}"})
    return result.output.strip()


def query_tool(backend: Any, *, local: bool):
    """`query_chinook`, bound to one specialist's sandbox."""

    @tool
    async def query_chinook(sql: str) -> str:
        """Run one read-only SQL SELECT on the Chinook SQLite database in your VM; returns JSON rows (max 200)."""
        return await _run(backend, "/tools/board_sql.py", ["query"], local=local, payload={"sql": sql})

    return query_chinook


def chart_tool(backend: Any, *, local: bool):
    """`render_chart`, bound to one analyst's sandbox."""

    @tool
    async def render_chart(
        title: str, labels: list[str], series: dict[str, list[float]], filename: str, kind: str = "bar",
    ) -> str:
        """Render a chart in the board pack's style to /outputs/<filename>.png in your VM.

        labels: x-axis categories (e.g. years). series: {"Revenue": [one number per label]}, or several
        series to compare. kind: "bar" (default), "line" or "barh". filename: a slug without extension.
        """
        spec = {"title": title, "labels": labels, "series": series, "filename": filename, "kind": kind}
        return await _run(backend, "/tools/board_chart.py", [], local=local, payload=spec)

    return render_chart


def render_tool(backend: Any, *, local: bool):
    """`render_board_pack`, bound to the publisher's sandbox."""

    @tool
    async def render_board_pack(
        title: str,
        subtitle: str,
        summary: str,
        kpis: list[dict[str, str]],
        sections: list[dict[str, str]],
        audit_path: str = "",
    ) -> str:
        """Render the board pack to /outputs/board-pack.html in your VM (one self-contained file).

        summary: the executive summary, Markdown. kpis: [{"label", "value", "note", "tone": "good"|"bad"|"neutral"}].
        sections: one per region, [{"title", "markdown_path", "chart_path", "agent"}] with paths in your VM
        (e.g. /inputs/analyst-europe/europe.md). audit_path: the auditor's report in your VM.
        """
        spec = {"title": title, "subtitle": subtitle, "summary": summary, "kpis": kpis, "sections": sections,
                "audit_path": audit_path}
        return await _run(backend, "/tools/board_render.py", [], local=local, payload=spec)

    return render_board_pack


async def _list_outputs(backend: Any) -> list[str]:
    """Absolute paths of the files (not directories) under the sandbox's /outputs/."""
    found = (await backend.aglob("**", path="/outputs")).matches or []
    # deepagents >= 0.7.21 returns absolute paths; earlier 0.7.x, paths relative to `path`
    return [m["path"] if m["path"].startswith("/") else f"/outputs/{m['path']}" for m in found if not m.get("is_dir")]


def _save_and_open(target: Path, content: bytes) -> bool:
    """Write the board pack on this machine and open it unless BOARD_PACK_OPEN=0; True if opened.

    Blocking, so callers run it in a thread.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    if os.environ.get("BOARD_PACK_OPEN", "1") == "0":
        return False
    webbrowser.open(target.as_uri())
    return True


def orchestrator_tools(sandboxes: SubagentSandboxes, thread_id: str) -> list:
    """`handoff_files` and `export_board_pack`, bound to this turn's subagent sandboxes."""
    names = sorted(sandboxes.backends)

    @tool
    async def handoff_files(source: str, target: str, paths: list[str] | None = None) -> str:
        """Copy files from one specialist's VM into another's, byte for byte (charts included).

        source/target: subagent names. paths: files in the source VM; default: the .md and .png files
        under its /outputs/. They land in the target VM under /inputs/<source>/ (same file names). Call it for
        several sources in one turn to collect their work in parallel.
        """
        if source not in sandboxes.backends or target not in sandboxes.backends:
            return f"Unknown subagent. Known: {', '.join(names)}"
        src, dst = sandboxes.backends[source], sandboxes.backends[target]
        wanted = paths or [p for p in await _list_outputs(src) if p.endswith((".md", ".png"))]
        if not wanted:
            return f"{source} has no files under /outputs/ yet."
        # download_files/upload_files are blocking sandbox calls: run them in threads, off the event loop
        downloads = await asyncio.to_thread(src.download_files, wanted)
        files = [(f"/inputs/{source}/{Path(d.path).name}", d.content) for d in downloads if d.content is not None]
        missing = [d.path for d in downloads if d.content is None]
        uploads = await asyncio.to_thread(dst.upload_files, files) if files else []
        copied = [u.path for u in uploads if not u.error]
        failed = [f"{u.path} ({u.error})" for u in uploads if u.error]
        lines = [f"Copied {len(copied)} file(s) from {source} to {target}: {', '.join(copied) or '-'}"]
        if missing or failed:
            lines.append(f"Not copied: {', '.join(missing + failed)}")
        return "\n".join(lines)

    @tool
    async def export_board_pack() -> str:
        """Copy the publisher's /outputs/board-pack.html to the presenter's machine and open it."""
        if "publisher" not in sandboxes.backends:
            return "No publisher in this team."
        # One path in, so exactly one response out.
        (download,) = await asyncio.to_thread(sandboxes.backends["publisher"].download_files, ["/outputs/board-pack.html"])
        if download.content is None:
            return f"The publisher has not produced /outputs/board-pack.html yet ({download.error})."
        target = HERE / "outputs" / thread_id / "board-pack.html"
        # Off the event loop: under `langgraph dev`, blockbuster rejects disk I/O on it.
        opened = await asyncio.to_thread(_save_and_open, target, download.content)
        return f"Exported to {target} ({len(download.content):,} bytes)" + (" and opened it in the browser." if opened else ".")

    return [handoff_files, export_board_pack]
