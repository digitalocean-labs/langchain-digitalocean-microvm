"""Live terminal dashboard: the locators' MicroVMs, every record of the five systems, and every Action Gateway call.

Three panels. One tile per locator MicroVM, as the team starts, works and pauses them (a pending approval shows every
tile paused). The systems strip: one column per system (all mocks), one line per record the locators found, with its
plan and what happened to it: awaiting approval, erased, redacted, kept, skipped, denied, a refused delete, and the
verifier's check. The gateway audit: each call that changes something, with the policy's answer, and each human
decision. Everything shown is an id or a status, never a name or an address: the event log holds no more.

Real mode polls the MicroVMs API (status of every MicroVM tagged as a subagent sandbox, and the template's
checkpoint) and adds the timings agent.py logs to outputs/events.jsonl. Local mode has no API, so it shows the event log alone.

RUN (from the repository root, in a second terminal next to run_script.py)
    uv run examples/10_erasure/dashboard.py                # every request thread
    uv run examples/10_erasure/dashboard.py --thread <id>  # one thread
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
ROLES = ["crm", "support", "billing", "marketing"]
SYSTEMS = {"crm": "CRM", "support": "support desk", "billing": "payments", "marketing": "email marketing", "analytics": "product analytics"}
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
RECORD_STYLE = {  # record state -> style
    "found": "bold white",
    "awaiting approval": "bold black on yellow",
    "erased": "bold green",
    "redacted": "bold green",
    "kept": "cyan",
    "skipped": "dim",
    "denied": "bold white on red",
}
PLANNED = {"erase": "found", "redact": "found", "retain": "kept", "skip": "skipped"}


OVERVIEW = Text(  # what the run is about, in the header
    "A customer asks us to erase her personal data, held in five systems. A team of AI agents finds every "
    "record, each search in a MicroVM of its own, and a human approves every change. Billing keeps its "
    "invoices, so her billing details are cleared, not deleted; another person with the same name is left "
    "alone. The agents then check it is gone and report. The systems are mocks.",
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


@dataclass
class Record:
    """One record of one system, as the event log describes it: ids and states only."""

    system: str
    kind: str
    id: str
    action: str = "?"  # the plan: erase / redact / retain / skip
    identity: str = ""
    state: str = "found"
    note: str = ""  # a refused delete, an error
    verified: bool | None = None


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
        self.records: dict[tuple[str, str, str], Record] = {}
        self.events: list[tuple[float, str]] = []
        self.template: str | None = None
        self.api_error: str | None = None
        self.outcome: str | None = None  # the verifier's verdict
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

    def record(self, target: list | tuple | None) -> Record | None:
        if not target or len(target) != 3:
            return None
        key = (str(target[0]), str(target[1]), str(target[2]))
        if key not in self.records:
            self.records[key] = Record(*key)
        return self.records[key]

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
                self.apply(event)

    def apply(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "manifest":
            for r in event.get("records", []):
                record = self.record((r.get("system"), r.get("kind"), r.get("id")))
                if record is not None:
                    record.action, record.identity = str(r.get("action")), str(r.get("identity") or "")
                    record.state = PLANNED.get(record.action, "found")  # a new request starts the record over
                    record.note, record.verified = "", None
            return
        if kind == "world":
            record = self.record((event.get("system"), event.get("kind"), event.get("record")))
            if record is not None:
                record.state = str(event.get("what"))
            return
        if kind == "verified":
            for r in event.get("records", []):
                record = self.record((r.get("system"), r.get("kind"), r.get("id")))
                if record is not None:
                    record.verified = bool(r.get("ok"))
                    if r.get("status") == "still present":
                        record.note = record.note or "still present"
            missed = len(event.get("missed", []))
            self.outcome = "COMPLETE" if event.get("complete") else \
                f"INCOMPLETE: {sum(1 for r in event.get('records', []) if not r.get('ok')) + missed} open"
            return
        if kind in ("gateway", "decision"):
            self.apply_gateway(event)
            return
        tile = self.tile(event.get("sandbox_id") or "")
        if tile is None or not kind:
            return
        seconds = event.get("seconds")
        if kind.endswith("-start"):
            self.set_status(tile, {"clone": "cloning", "create": "creating", "resume": "resuming"}.get(kind[:-6], "running"))
            return
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
        self.events.append((event["ts"], f"{tile.role} ({tile.thread[:8]}) {label}"
                            + (f" in {seconds:.1f} s" if seconds is not None else "")))
        self.events = self.events[-8:]

    def apply_gateway(self, event: dict) -> None:
        """A changing call or a decision moves its record; reads (all `allow`) are counted, not listed."""
        record = self.record(event.get("target"))
        if event["event"] == "decision":
            if record is not None and event["decision"] != "approve":
                record.state = "denied"
            tone = "approve" if event["decision"] == "approve" else "deny"
            self._audit(event["ts"], tone, event["tool"], f"human: {event['decision']}", record)
            return
        status = event["status"]
        if record is None:  # a read
            if status != "allow":
                self._audit(event["ts"], status, event["tool"], (event.get("error") or status)[:70], None)
            return
        if status == "ask":
            record.state = "awaiting approval"
        elif status == "deny":
            record.note = "delete refused by policy" if "delete" in event["tool"] else "refused by policy"
        elif status == "error":
            record.note = "error"
        what = {"ask": "waiting for approval", "approved": "ran after approval", "deny": "refused by policy",
                "denied": "denied by the human", "error": "error"}.get(status, status)
        self._audit(event["ts"], status, event["tool"], what, record)

    def _audit(self, ts: float, tone: str, tool: str, what: str, record: Record | None) -> None:
        where = f"{record.system} {record.kind} {record.id}" if record else ""
        self.audit.append((ts, tone, tool, f"{what}  {where}".strip()))
        self.audit = self.audit[-8:]

    def poll_api(self, lifecycle: MicroVMLifecycle) -> None:
        """Forever, once a second: mirror the API's view of the locators' MicroVMs into the tiles."""
        client = lifecycle._client_or_raise()
        last_checkpoints = 0.0
        while True:
            try:
                states = client.list(tag=SUBAGENT_SANDBOX_TAG)
                if time.time() - last_checkpoints > 10:
                    names = [c.name for c in client.list_checkpoints() if c.available and c.name.startswith(f"{TEMPLATE_PREFIX}-erasure")]
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


def record_line(r: Record) -> Text:
    """`contact 51  erase → ERASED ✔`, styled by state."""
    line = Text(f"{r.kind} {r.id}", style="bold" if r.action in ("erase", "redact") else "")
    line.append(f"  {r.action}" + (" (another person)" if r.identity == "different_person" else ""), style="dim")
    line.append("  ")
    line.append(f" {r.state.upper()} " if r.state in ("awaiting approval", "denied") else r.state.upper(),
                style=RECORD_STYLE.get(r.state, "white"))
    if r.verified is not None:
        line.append(" ✔" if r.verified else " ✘", style="green" if r.verified else "bold red")
    if r.note:
        line.append(f"\n  {r.note}", style="red")
    return line


def render(state: State, mode: str) -> Group:
    """One frame: header, the locator tiles, the systems strip, the gateway audit, the lifecycle log."""
    with state.lock:
        tiles = sorted(
            (t for t in state.tiles.values() if t.status != "deleted" or time.time() - t.since < 20),
            key=lambda t: (-max(x.first_seen for x in state.tiles.values() if x.thread == t.thread), t.thread, ROLES.index(t.role)),
        )
        records = list(state.records.values())
        events, audit, outcome = list(state.events), list(state.audit), state.outcome
    counts = {s: sum(1 for t in tiles if t.status == s) for s in ("running", "paused")}
    starting = sum(1 for t in tiles if t.status in ("cloning", "creating", "resuming"))
    waiting = sum(1 for r in records if r.state == "awaiting approval")

    title = Text.assemble(("DigitalOcean MicroVMs", "bold #0069FF"), "  ·  ", ("Deep Agents erasure request", "bold"), f"  ·  {mode}")
    stats = Text.assemble(
        ("● ", "green"), f"{counts['running']} running   ", ("● ", "yellow"), f"{starting} starting   ",
        ("● ", "blue"), f"{counts['paused']} paused   ",
        ("template: ", "dim"), (state.template or ("local directory" if mode == "local" else "…"), "dim"),
    )
    if waiting:
        stats.append(f"   {waiting} waiting for a human", style="bold yellow")
    if state.api_error:
        stats.append(f"   API: {state.api_error}", style="red")
    header = Panel(Group(Align.center(title), OVERVIEW, Align.center(stats)), border_style="#0069FF")

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
        cells.append(Panel(body, title=Text(t.role, style="bold"), title_align="left", border_style=border, height=6))
    if not cells:
        cells = [Panel(Text("No locator MicroVMs yet. Send the request.", style="dim"), height=6)]
    for i in range(0, len(cells), columns):
        row = cells[i:i + columns]
        grid.add_row(*row, *[""] * (columns - len(row)))

    strip = Table(expand=True, box=None, padding=(0, 1), show_edge=False)
    for system, product in SYSTEMS.items():
        strip.add_column(Text.assemble((system, "bold"), (f" {product}", "dim")), ratio=1)
    by_system = {s: [r for r in records if r.system == s] for s in SYSTEMS}
    if records:
        strip.add_row(*[Group(*[record_line(r) for r in by_system[s]]) if by_system[s] else Text("nothing found", style="dim")
                        for s in SYSTEMS])
    systems_title = "systems (mocks)" + (f"  ·  verified: {outcome}" if outcome else "")
    systems_panel = Panel(strip if records else Text("no records yet: the locators have not reported", style="dim"),
                          title=systems_title, title_align="left",
                          border_style="green" if outcome == "COMPLETE" else ("red" if outcome else "grey37"))

    gateway_log = Table.grid(padding=(0, 2))
    colors = {"allow": "green", "approved": "green", "approve": "green", "ask": "yellow", "deny": "red", "denied": "red", "error": "red"}
    for ts, tone, tool, what in audit[::-1]:
        gateway_log.add_row(Text(time.strftime("%H:%M:%S", time.localtime(ts)), style="dim"),
                            Text(tone.upper(), style=f"bold {colors.get(tone, 'white')}"), Text(tool), Text(what, style="dim"))
    gateway_panel = Panel(gateway_log, title="Action Gateway (changes and decisions; reads are allowed)", title_align="left",
                          border_style="grey37", height=10)

    log = Table.grid(padding=(0, 2))
    for ts, line in events[::-1]:
        log.add_row(Text(time.strftime("%H:%M:%S", time.localtime(ts)), style="dim"), Text(line))
    return Group(header, grid, systems_panel, gateway_panel,
                 Panel(log, title="MicroVM lifecycle (cloned, resumed, paused, deleted)", title_align="left",
                       border_style="grey37", height=7))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", help="show only this thread's MicroVMs")
    parser.add_argument("--replay", action="store_true", help="start from the beginning of the event log, not from now")
    args = parser.parse_args()

    config = MicroVMConfig.from_env()
    state = State(args.thread, from_start=args.replay)
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
