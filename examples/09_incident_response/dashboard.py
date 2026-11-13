"""Live terminal dashboard: the investigators' MicroVMs, the droplet, and every Action Gateway call.

Three panels. One tile per investigator MicroVM, as the team starts, works and pauses them (a pending approval shows
every tile paused). The droplet strip: its state and what its `/health` says, from the `world` events (local mode)
or by probing DO_INCIDENT_URL (real mode). The gateway audit: each call with the policy's answer (allow, ask, deny,
approved) and each human decision.

Real mode polls the MicroVMs API (status of every MicroVM tagged as a subagent sandbox, and the template's
checkpoint) and adds the timings agent.py logs to outputs/events.jsonl. Local mode has no API, so it shows the event log alone.

RUN (from the repository root, in a second terminal next to run_script.py)
    uv run examples/09_incident_response/dashboard.py                # every incident thread
    uv run examples/09_incident_response/dashboard.py --thread <id>  # one thread
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # examples/: the shared incident scenario (incident_common)
    sys.path.insert(0, str(HERE.parent))
from incident_common import OUTPUTS  # noqa: E402

load_dotenv(HERE.parents[1] / ".env")
load_dotenv(OUTPUTS / "incident.env")  # written by incident_common/setup_droplet.py (real mode)

import httpx
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

EVENTS = OUTPUTS / "events.jsonl"
DROPLET_NAME = "incident-svc"  # incident_common's name for it (world.py, setup_droplet.py)
# agent.py's SUBAGENTS, in display order (not imported: that would build the models), and what each is called on screen
ROLES = ["metrics", "changes", "infra", "verifier"]
ROLE_NAMES = {"metrics": "Metrics Analyst", "changes": "Change Analyst", "infra": "Infra Analyst", "verifier": "Recovery Validator"}
STYLE = {  # status -> (border, badge)
    "cloning": ("yellow", "bold black on yellow"),
    "creating": ("yellow", "bold black on yellow"),
    "resuming": ("yellow", "bold black on yellow"),
    "running": ("green", "bold black on green"),
    "pausing": ("cyan", "bold black on cyan"),
    "paused": ("blue", "bold white on blue"),
    "deleted": ("grey37", "white on grey37"),
    "failed": ("red", "bold white on red"),
    "terminating": ("red", "bold white on red"),
}
BUSY = {"cloning", "creating", "resuming", "running", "pausing"}


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

    def __init__(self, thread: str | None, *, from_start: bool = False) -> None:
        self.thread = thread
        self.tiles: dict[str, Tile] = {}
        self.events: list[tuple[float, str]] = []
        self.template: str | None = None
        self.api_error: str | None = None
        self.droplet: dict | None = None  # the last `world` event, or the last probe of DO_INCIDENT_URL
        self.audit: list[tuple[float, str, str, str]] = []  # (time, tone, tool, what)
        self.lock = threading.Lock()
        self._offset = 0 if from_start or not EVENTS.exists() else EVENTS.stat().st_size  # only what happens from now on

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
                if event.get("event") == "world":
                    self.droplet = {**event, "at": event["ts"]}
                    continue
                if event.get("event") in ("gateway", "decision"):
                    self.audit.append(self.audit_row(event))
                    self.audit = self.audit[-7:]
                    continue
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
                else:
                    self.set_status(tile, "running")
                tile.microvm_id = event.get("microvm_id") or tile.microvm_id
                label = {"clone": "cloned from template", "create": "booted", "resume": "resumed",
                         "attach": "attached", "pause": "paused", "delete": "deleted"}.get(kind, kind)
                tile.note = f"{label} in {seconds:.1f} s" if seconds is not None else label
                self.events.append((event["ts"], f"{ROLE_NAMES[tile.role]} ({tile.thread[:8]}) {label}"
                                    + (f" in {seconds:.1f} s" if seconds is not None else "")))
                self.events = self.events[-8:]

    @staticmethod
    def audit_row(event: dict) -> tuple[float, str, str, str]:
        if event["event"] == "decision":
            return event["ts"], "approve" if event["decision"] == "approve" else "deny", event["tool"], f"human: {event['decision']}"
        what = {"allow": "allowed", "ask": "waiting for approval", "approved": "ran after approval",
                "deny": "refused by policy", "denied": "denied by the human", "error": "error"}.get(event["status"], event["status"])
        return event["ts"], event["status"], event["tool"], (event.get("error") or what)[:90]

    def probe_droplet(self, url: str) -> None:
        """Forever, every 2 s (real mode): the service's own /health."""
        while True:
            try:
                body = httpx.get(f"{url.rstrip('/')}/health", timeout=5).json()
            except (httpx.HTTPError, ValueError):
                body = {"status": "unreachable"}
            with self.lock:
                self.droplet = {**body, "what": (self.droplet or {}).get("what", ""), "at": time.time()}
            time.sleep(2.0)

    def poll_api(self, lifecycle: MicroVMLifecycle) -> None:
        """Forever, once a second: mirror the API's view of the investigators' MicroVMs into the tiles."""
        client = lifecycle._client_or_raise()
        last_checkpoints = 0.0
        while True:
            try:
                states = client.list(tag=SUBAGENT_SANDBOX_TAG)
                if time.time() - last_checkpoints > 10:
                    names = [c.name for c in client.list_checkpoints() if c.available and c.name.startswith(f"{TEMPLATE_PREFIX}-incident")]
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
                        # The API says "running" as soon as the MicroVM is up; keep the richer state from the
                        # event log (cloning/resuming) until its own event replaces it.
                        if not (status == "running" and tile.status in ("cloning", "resuming", "creating")):
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
        droplet, audit = state.droplet, list(state.audit)
    counts = {s: sum(1 for t in tiles if t.status == s) for s in ("running", "paused")}
    starting = sum(1 for t in tiles if t.status in ("cloning", "creating", "resuming"))

    title = Text.assemble(("DigitalOcean MicroVMs", "bold #0069FF"), "  ·  ", ("Deep Agents incident response", "bold"), f"  ·  {mode}")
    stats = Text.assemble(
        ("● ", "green"), f"{counts['running']} running   ", ("● ", "yellow"), f"{starting} starting   ",
        ("● ", "blue"), f"{counts['paused']} paused   ",
        ("template: ", "dim"), (state.template or ("local directory" if mode == "local" else "…"), "dim"),
    )
    if state.api_error:
        stats.append(f"   API: {state.api_error}", style="red")
    header = Panel(Group(Align.center(title), Align.center(stats)), border_style="#0069FF")

    grid = Table.grid(padding=(0, 1), expand=True)
    columns = len(ROLES)
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
        body.add_row(Text(f"MicroVM {t.microvm_id[:13] if t.microvm_id else '…'}  ·  {t.thread[:8]}", style="dim"))
        cells.append(Panel(body, title=Text(ROLE_NAMES[t.role], style="bold"), title_align="left", border_style=border, height=6))
    if not cells:
        cells = [Panel(Text("No investigator MicroVMs yet. Send the alert.", style="dim"), height=6)]
    for i in range(0, len(cells), columns):
        row = cells[i:i + columns]
        grid.add_row(*row, *[""] * (columns - len(row)))

    log = Table.grid(padding=(0, 2))
    for ts, line in events[::-1]:
        log.add_row(Text(time.strftime("%H:%M:%S", time.localtime(ts)), style="dim"), Text(line))
    strip = Table.grid(padding=(0, 2))
    if droplet is None:
        strip.add_row(Text("no data yet", style="dim"))
    else:
        status = str(droplet.get("status", "?"))
        tone = {"healthy": "bold black on green", "failing": "bold white on red", "unreachable": "bold white on red"}.get(status, "reverse")
        rate = droplet.get("error_rate")
        strip.add_row(Text(f" {status.upper()} ", style=tone), Text(f"version {droplet.get('version', '?')}"),
                      Text(f"error rate {rate:.0%}" if isinstance(rate, (int, float)) else "error rate ?"),
                      Text(f"droplet {droplet.get('droplet', '?')}"), Text(str(droplet.get("what") or ""), style="dim"))
    droplet_panel = Panel(strip, title=f"droplet {DROPLET_NAME}", title_align="left", border_style="grey37")

    gateway_log = Table.grid(padding=(0, 2))
    colors = {"allow": "green", "approved": "green", "approve": "green", "ask": "yellow", "deny": "red", "denied": "red", "error": "red"}
    for ts, tone, tool, what in audit[::-1]:
        gateway_log.add_row(Text(time.strftime("%H:%M:%S", time.localtime(ts)), style="dim"),
                            Text(tone.upper(), style=f"bold {colors.get(tone, 'white')}"), Text(tool), Text(what, style="dim"))
    gateway_panel = Panel(gateway_log, title="Action Gateway", title_align="left", border_style="grey37", height=9)
    return Group(header, grid, droplet_panel, gateway_panel,
                 Panel(log, title="lifecycle", title_align="left", border_style="grey37", height=7))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", help="show only this thread's MicroVMs")
    parser.add_argument("--replay", action="store_true", help="start from the beginning of the event log, not from now")
    args = parser.parse_args()

    config = MicroVMConfig.from_env()
    state = State(args.thread, from_start=args.replay)
    if config.transport != "local":
        threading.Thread(target=state.poll_api, args=(MicroVMLifecycle(config),), daemon=True).start()
        if os.environ.get("DO_INCIDENT_URL"):
            threading.Thread(target=state.probe_droplet, args=(os.environ["DO_INCIDENT_URL"],), daemon=True).start()
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
