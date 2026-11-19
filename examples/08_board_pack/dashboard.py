"""Live terminal dashboard: one tile per specialist MicroVM, as the team starts, works and pauses them.

Real mode polls the MicroVMs API (status of every VM tagged as a subagent sandbox, the template
checkpoint) and adds the timings agent.py logs to outputs/events.jsonl (clone in 1.4 s, paused
in 0.8 s). Local mode has no API, so it shows the event log alone.

RUN (from the repository root, in a second terminal next to the agent)
    uv run examples/08_board_pack/dashboard.py                # every board pack thread
    uv run examples/08_board_pack/dashboard.py --thread <id>  # one thread
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle
from langchain_digitalocean_microvm.client import (
    SUBAGENT_SANDBOX_TAG,
    sandbox_id_from_tags,
)
from langchain_digitalocean_microvm.provider import TEMPLATE_PREFIX

EVENTS = HERE / "outputs" / "events.jsonl"
# agent.py's SUBAGENTS, in display order (not imported: that would build the models)
ROLES = ["analyst-north-america", "analyst-europe", "analyst-latin-america", "analyst-asia-pacific", "auditor", "publisher"]
STYLE = {  # status -> (border, badge)
    "cloning": ("yellow", "bold black on yellow"),
    "creating": ("yellow", "bold black on yellow"),
    "resuming": ("yellow", "bold black on yellow"),
    "running": ("green", "bold black on green"),
    "pausing": ("cyan", "bold black on cyan"),
    "paused": ("blue", "bold white on blue"),
    "terminating": ("red", "bold white on red"),
    "deleted": ("grey37", "white on grey37"),
    "failed": ("red", "bold white on red"),
    "wrecked": ("red", "bold white on red"),
}
BUSY = {"cloning", "creating", "resuming", "running", "pausing"}


OVERVIEW = Text(  # what the run is about, in the header
    "Leadership needs a board pack: the headline numbers and how each region did. A team of AI agents "
    "builds it from the sales database: four analysts take a region each, an auditor re-checks the "
    "headline figures from its own copy of the data and a publisher assembles the document, each in a "
    "MicroVM of its own. The store, Chinook, and its data are fictional.",
    style="#5AB0FF", justify="center",
)


@dataclass
class Tile:
    sandbox_id: str
    role: str
    thread: str
    status: str = "creating"
    microvm_id: str | None = None
    note: str = ""
    since: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)


def split(sandbox_id: str) -> tuple[str, str] | None:
    """`<thread>-<role>` -> (thread, role), for the roles of this example only."""
    for role in ROLES:
        if sandbox_id.endswith(f"-{role}"):
            return sandbox_id[: -len(role) - 1], role
    return None


class State:
    """Everything the screen shows. Two sources feed it: the events log (read_events, from the main
    thread) and the MicroVMs API (poll_api, from a background thread); `lock` guards both."""

    def __init__(self, thread: str | None) -> None:
        self.thread = thread
        self.tiles: dict[str, Tile] = {}
        self.events: list[tuple[float, str]] = []
        self.template: str | None = None
        self.api_error: str | None = None
        self.lock = threading.Lock()
        self._offset = EVENTS.stat().st_size if EVENTS.exists() else 0  # only what happens from now on

    def tile(self, sandbox_id: str) -> Tile | None:
        parts = split(sandbox_id)
        if parts is None or (self.thread and parts[0] != self.thread):
            return None
        if sandbox_id not in self.tiles:
            self.tiles[sandbox_id] = Tile(sandbox_id, parts[1], parts[0])
        return self.tiles[sandbox_id]

    def set_status(self, tile: Tile, status: str) -> None:
        if tile.status != status:
            tile.status, tile.since = status, time.time()

    def read_events(self) -> None:
        """Apply the lines appended to events.jsonl since the last call (tail -f style, by byte offset)."""
        if not EVENTS.exists():
            return
        with EVENTS.open() as f:
            f.seek(self._offset)
            lines = f.readlines()
            self._offset = f.tell()
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            with self.lock:
                tile = self.tile(event.get("sandbox_id") or "")
                if tile is None:
                    continue
                kind, seconds = event["event"], event.get("seconds")
                if kind.endswith("-start"):
                    self.set_status(tile, {"clone": "cloning", "create": "creating", "resume": "resuming"}.get(kind[:-6], "running"))
                    continue
                if kind == "pause":
                    self.set_status(tile, "paused")
                elif kind == "delete":
                    self.set_status(tile, "deleted")
                elif kind == "chaos":
                    self.set_status(tile, "wrecked")
                else:
                    self.set_status(tile, "running")
                tile.microvm_id = event.get("microvm_id") or tile.microvm_id
                label = {"clone": "cloned from template", "create": "booted", "resume": "resumed",
                         "attach": "attached", "pause": "paused", "delete": "deleted",
                         "chaos": "wiped with rm -rf /"}.get(kind, kind)
                tile.note = f"{label} in {seconds:.1f} s" if seconds is not None else label
                self.events.append((event["ts"], f"{tile.role} ({tile.thread[:8]}) {label}"
                                    + (f" in {seconds:.1f} s" if seconds is not None else "")))
                self.events = self.events[-8:]

    def poll_api(self, lifecycle: MicroVMLifecycle) -> None:
        """Forever, once a second: mirror the API's view of the specialist VMs into the tiles."""
        client = lifecycle._client_or_raise()
        last_checkpoints = 0.0
        while True:
            try:
                states = client.list(tag=SUBAGENT_SANDBOX_TAG)
                if time.time() - last_checkpoints > 10:
                    names = [c.name for c in client.list_checkpoints() if c.available and c.name.startswith(f"{TEMPLATE_PREFIX}-board-pack")]
                    self.template, last_checkpoints = (names[0] if names else None), time.time()
                with self.lock:
                    seen = set()
                    for state in states:
                        sandbox_id = sandbox_id_from_tags(state.tags)
                        tile = self.tile(sandbox_id or "")
                        if tile is None:
                            continue
                        seen.add(sandbox_id)
                        tile.microvm_id = state.id
                        status = state.status.value
                        # The API says "running" as soon as the VM is up; keep the richer state from the
                        # event log (cloning/resuming/wrecked) until its own event replaces it.
                        if not (status == "running" and tile.status in ("cloning", "resuming", "creating", "wrecked")):
                            self.set_status(tile, status)
                    for sandbox_id, tile in self.tiles.items():
                        if sandbox_id not in seen and tile.status not in ("cloning", "creating"):
                            self.set_status(tile, "deleted")
                self.api_error = None
            except Exception as exc:  # noqa: BLE001 - shown in the header, retried
                self.api_error = str(exc)[:120]
            time.sleep(1.0)


def ago(seconds: float) -> str:
    """Seconds as '42s' or '3m07s'."""
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def render(state: State, mode: str) -> Group:
    """One frame: header, a grid of tiles (newest thread first, roles in team order), the lifecycle log."""
    with state.lock:
        tiles = sorted(
            (t for t in state.tiles.values() if t.status != "deleted" or time.time() - t.since < 20),
            key=lambda t: (-max(x.first_seen for x in state.tiles.values() if x.thread == t.thread), t.thread, ROLES.index(t.role)),
        )
        events = list(state.events)
    counts = {s: sum(1 for t in tiles if t.status == s) for s in ("running", "paused")}
    starting = sum(1 for t in tiles if t.status in ("cloning", "creating", "resuming"))

    title = Text.assemble(("DigitalOcean MicroVMs", "bold #0069FF"), "  ·  ", ("Deep Agents board pack", "bold"), f"  ·  {mode}")
    stats = Text.assemble(
        ("● ", "green"), f"{counts['running']} running   ", ("● ", "yellow"), f"{starting} starting   ",
        ("● ", "blue"), f"{counts['paused']} paused   ",
        ("template: ", "dim"), (state.template or ("local directory" if mode == "local" else "…"), "dim"),
    )
    if state.api_error:
        stats.append(f"   API: {state.api_error}", style="red")
    header = Panel(Group(Align.center(title), OVERVIEW, Align.center(stats)), border_style="#0069FF")

    grid = Table.grid(padding=(0, 1), expand=True)
    columns = 3
    for _ in range(columns):
        grid.add_column(ratio=1)
    cells = []
    for t in tiles:
        border, badge = STYLE.get(t.status, ("white", "reverse"))
        body = Table.grid()
        status = Text(f" {t.status.upper()} ", style=badge)
        if t.status in BUSY:
            body.add_row(Spinner("dots", text=status, style=border))
        else:
            body.add_row(status)
        body.add_row(Text(f"{t.note or ' '}", style="bold"))
        body.add_row(Text(f"{t.status} for {ago(time.time() - t.since)}", style="dim"))
        body.add_row(Text(f"vm {t.microvm_id[:13] if t.microvm_id else '…'}  ·  thread {t.thread[:8]}", style="dim"))
        cells.append(Panel(body, title=Text(t.role, style="bold"), title_align="left", border_style=border, height=6))
    if not cells:
        cells = [Panel(Text("No specialist VMs yet. Ask for the board pack.", style="dim"), height=6)]
    for i in range(0, len(cells), columns):
        row = cells[i:i + columns]
        grid.add_row(*row, *[""] * (columns - len(row)))

    log = Table.grid(padding=(0, 2))
    for ts, line in events[::-1]:
        log.add_row(Text(time.strftime("%H:%M:%S", time.localtime(ts)), style="dim"), Text(line))
    lifecycle = Panel(log, title="MicroVM lifecycle (cloned, resumed, paused, deleted)", title_align="left",
                      border_style="grey37", height=10)
    return Group(header, grid, lifecycle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", help="show only this thread's VMs")
    args = parser.parse_args()

    config = MicroVMConfig.from_env()
    state = State(args.thread)
    if config.transport != "local":
        threading.Thread(target=state.poll_api, args=(MicroVMLifecycle(config),), daemon=True).start()
    console = Console()
    with Live(render(state, config.transport), console=console, refresh_per_second=8, screen=True) as live:
        try:
            while True:
                state.read_events()
                live.update(render(state, config.transport))
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
