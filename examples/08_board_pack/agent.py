"""Board pack: a team of Deep Agents where every specialist works in its own DigitalOcean MicroVM.

THE IDEA
A board pack is the document a company's leadership sends its board of directors before a
board meeting: headline numbers, how each part of the business did, what changed and why,
and what the board should discuss. Here an orchestrator agent builds Chinook's (an online music
store's) FY2025 board pack with six specialists:

    analyst-north-america ┐
    analyst-europe        ├─ in parallel: each queries ITS OWN copy of the sales database,
    analyst-latin-america │  charts its region and writes its section
    analyst-asia-pacific  ┘
    auditor                ─ then: re-computes every headline figure independently
    publisher              ─ then: assembles the HTML board pack

Every specialist is a synchronous deepagents subagent whose filesystem and shell are a MicroVM
of its own, cloned on demand from one sandbox template (database, helper scripts, matplotlib),
so its first command finds everything in place. The orchestrator has no shell at all: it delegates,
and moves files between the VMs with `handoff_files`.

PER RUN (`make_graph`; design doc §5)
`langgraph dev` enters `make_graph` for every run of a thread. `asubagent_sandboxes(...,
templates=..., on_demand=True)` creates nothing up front: a subagent's MicroVM is cloned from
the template the first time that subagent acts (resumed if this thread already has one), so the
four analysts start their VMs in parallel when the orchestrator fans out, and the auditor's and
publisher's only when their turn comes. When the run ends, the VMs that were used are paused;
a follow-up question about one region wakes only that region's analyst.

MODES (.env)
- DO_MICROVM_TRANSPORT=local: sandboxes are local directories, no DigitalOcean account. The
  helper scripts run on this machine (charts need matplotlib installed locally).
- otherwise real MicroVMs: DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN and DO_MICROVM_IMAGE
  (scripts/build_sandbox_image.py).

RUN (from the repository root): see README.md
    cd examples/08_board_pack && uv run langgraph dev
or  uv run examples/08_board_pack/run_script.py
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")  # the shell wins over .env; before the modules below read the environment

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from events import DemoLifecycle
from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime
from models import model, strong_model
from tools import chart_tool, orchestrator_tools, query_tool, render_tool

from langchain_digitalocean_microvm import TemplateCache
from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    default_db_path,
)
from langchain_digitalocean_microvm.subagents import (
    SubagentSandboxes,
    asubagent_sandboxes,
)

logger = logging.getLogger(__name__)

lifecycle = DemoLifecycle()  # DO_MICROVM_* from the environment; logs every lifecycle step (events.py)
LOCAL = lifecycle.config.transport == "local"
# Real MicroVMs: record each sandbox's last use where scripts/gc_microvms.py looks.
store = None if LOCAL else GatewayStore(default_db_path("microvm", lifecycle.config.namespace))

DATABASE = HERE.parent / "07_sales_assistant" / "data" / "chinook.db"  # shared with example 07, not duplicated
SETUP = ["pip install --quiet --no-cache-dir matplotlib markdown"]

# ── The team ──────────────────────────────────────────────────────────────────

REGIONS: dict[str, tuple[str, str, list[str]]] = {  # subagent -> (region, slug, countries)
    "analyst-north-america": ("North America", "north-america", ["USA", "Canada"]),
    "analyst-europe": ("Europe", "europe", [
        "France", "Germany", "United Kingdom", "Czech Republic", "Portugal", "Ireland", "Hungary", "Austria",
        "Finland", "Netherlands", "Norway", "Sweden", "Spain", "Poland", "Italy", "Denmark", "Belgium",
    ]),
    "analyst-latin-america": ("Latin America", "latin-america", ["Brazil", "Chile", "Argentina"]),
    "analyst-asia-pacific": ("Asia-Pacific", "asia-pacific", ["India", "Australia"]),
}

SCHEMA = (
    "Chinook schema (SQLite): Invoice(InvoiceId, CustomerId, InvoiceDate 'YYYY-MM-DD HH:MM:SS', BillingCountry, Total); "
    "InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity); Track(TrackId, Name, AlbumId, GenreId); "
    "Genre(GenreId, Name); Customer(CustomerId, FirstName, LastName, Country). Revenue = SUM(Invoice.Total); "
    "a genre's revenue = SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity). Years: strftime('%Y', InvoiceDate). "
    "The data covers 2021-2025; FY2025 is calendar 2025. Amounts are plain US dollars (write $157.41, never $157.41M)."
)

NO_INSTALL = (
    "Never install packages (pip, apt): this sandbox runs on the presenter's machine."
    if LOCAL else "Everything you need is installed; do not install packages."
)


def _in(countries: list[str]) -> str:
    """A SQL IN list for the prompts, e.g. ('USA', 'Canada')."""
    return "(" + ", ".join(f"'{c}'" for c in countries) + ")"


def analyst_spec(name: str) -> dict[str, Any]:
    """One regional analyst's subagent spec; the prompt scripts its steps so the four run fast and alike."""
    region, slug, countries = REGIONS[name]
    return {
        "name": name,
        "description": f"Regional analyst for {region}, with its own VM and its own copy of the sales database. "
                       f"Ask it for the region's board pack section, or any question about {region}.",
        "system_prompt": (
            f"You are the {region} analyst preparing your region's section of Chinook's FY2025 board pack. "
            f"You work alone in your own Linux VM. {SCHEMA} "
            f"Your region is BillingCountry IN {_in(countries)}.\n\n"
            "For the board pack section, be fast and exact. Do not plan with write_todos, and create no files "
            "other than the two below. If a tool fails twice, carry on without it.\n"
            "1. In ONE turn, run these queries in parallel with `query_chinook`: revenue and invoice count by year "
            "(2021-2025) for the region; FY2025 and FY2024 revenue by country; FY2025 top 3 genres by revenue.\n"
            f"2. `render_chart` a bar chart of revenue by year, filename '{slug}-trend', title "
            f"'{region}: revenue by year'.\n"
            f"3. `write_file` /outputs/{slug}.md: a bold one-line headline (FY2025 revenue, change vs FY2024 in %), "
            "3-4 bullet findings (growth or decline drivers by country, top genres), and a compact Markdown table "
            "of FY2025 revenue by country. No H1 title. Round money to 2 decimals. Check that the countries add up "
            "to the regional total before you write.\n"
            "Answer with: FY2025 revenue, FY2024 revenue, change in %, the headline, and the two file paths.\n\n"
            "For any other question, query what you need and answer briefly with exact figures; chart only if asked. "
            + NO_INSTALL
        ),
        "model": model,
    }


AUDITOR = {
    "name": "auditor",
    "description": "Independent auditor with its own VM and its own copy of the database. Hand it the regional "
                   "reports first (handoff_files to 'auditor'); it re-computes every headline figure.",
    "system_prompt": (
        "You are the auditor of Chinook's FY2025 board pack, working alone in your own Linux VM. "
        f"{SCHEMA} Regions by BillingCountry: "
        + "; ".join(f"{region} = {_in(countries)}" for region, _, countries in REGIONS.values())
        + ".\n\nThe regional reports are under /inputs/<analyst>/ (read them with read_file). Re-compute the "
        "FY2025 and FY2024 revenue of every region yourself with ONE `query_chinook` query (a CASE on "
        "BillingCountry, grouped by region and year), compare with each report's headline (tolerance 0.01), "
        "and `write_file` /outputs/audit.md (your only file; no write_todos): a Markdown table region | reported FY2025 | recomputed FY2025 | "
        "status (✅ or ❌), then a one-line verdict. Answer with the verdict and any discrepancy. " + NO_INSTALL
    ),
    "model": model,
}

PUBLISHER = {
    "name": "publisher",
    "description": "Publisher with its own VM: turns the regional sections, charts and audit under its /inputs/ "
                   "into the HTML board pack. Hand it every analyst's files and the auditor's first.",
    "system_prompt": (
        "You are the publisher of Chinook's FY2025 board pack, working in your own Linux VM. Under /inputs/<agent>/ "
        "you have each regional analyst's section (.md) and chart (.png), and the auditor's /inputs/auditor/audit.md. "
        "Read each region's .md section and the audit with read_file (in one turn; skip .png and any other files), "
        "then call `render_board_pack` once; do not use write_todos. Never write HTML yourself (no write_file, no "
        "execute): `render_board_pack` is the only way to produce the board pack.\n"
        "- title 'Chinook Music · FY2025 Board Pack', subtitle 'Prepared for the board meeting · all figures audited';\n"
        "- summary: 3-4 crisp sentences for directors: the global result, what grew, what declined, and the one "
        "question the board should discuss;\n"
        "- kpis: global FY2025 revenue (with YoY %), the strongest region, the region to watch, the audit result "
        "(tone good/bad/neutral);\n"
        "- sections: one per region, largest first, with markdown_path, chart_path and agent (the analyst's name);\n"
        "- audit_path: /inputs/auditor/audit.md.\n"
        "Answer with the path of the board pack. " + NO_INSTALL
    ),
    "model": strong_model,  # writes the executive summary; follows the one-tool rule reliably
}

SUBAGENTS: list[dict[str, Any]] = [*(analyst_spec(name) for name in REGIONS), AUDITOR, PUBLISHER]


def specialist_spec(spec: dict[str, Any], backend: Any) -> dict[str, Any]:
    """`spec` with its bound tools; the publisher also loses `execute` and `write_file`, so the only way
    it can produce the board pack is `render_board_pack` (a prompt alone did not stop it from hand-writing HTML)."""
    out = {**spec, "tools": specialist_tools(spec["name"], backend)}
    if spec["name"] == "publisher":
        out["middleware"] = [*spec.get("middleware", []), FilesystemMiddleware(backend=backend, tools=["ls", "read_file"])]
    return out


def specialist_tools(name: str, backend: Any) -> list:
    """The tools each role gets, bound to its own sandbox: analysts query and chart, the auditor only
    queries, the publisher only renders. Nobody but the orchestrator can move files between VMs."""
    if name in REGIONS:
        return [query_tool(backend, local=LOCAL), chart_tool(backend, local=LOCAL)]
    if name == "auditor":
        return [query_tool(backend, local=LOCAL)]
    return [render_tool(backend, local=LOCAL)]


SYSTEM_PROMPT = (
    "You are the board pack orchestrator at Chinook, an online music distributor. You lead six specialists; each "
    "works in its own isolated VM with its own copy of the sales database. You have no shell yourself: you "
    "delegate with `task` and move files between the specialists' VMs with `handoff_files`.\n\n"
    "When asked for the board pack:\n"
    "1. FAN OUT: delegate to all four regional analysts AT ONCE, four `task` calls in a single message, each "
    "asking for its region's FY2025 board pack section.\n"
    "2. AUDIT: in one message, `handoff_files` from each analyst to 'auditor' (four calls); then delegate to the "
    "auditor.\n"
    "3. PUBLISH: in one message, `handoff_files` from each analyst and from the auditor to 'publisher' (five "
    "calls); then delegate to the publisher with a short task: have it read the sections and the audit and "
    "call its `render_board_pack` tool. Never ask it to write HTML or files itself.\n"
    "4. Call `export_board_pack`, then reply with: a 3-line executive summary, the audit verdict, and the local "
    "file path.\n\n"
    "For follow-up questions, delegate only to the specialist(s) the question needs (a question about one "
    "region goes to that region's analyst alone, several regions to several analysts at once), and do not "
    "rebuild the board pack unless asked. Keep replies short; use exact figures from your specialists."
)

# ── Template, sandboxes, orchestrator ─────────────────────────────────────────


def seed_files() -> list[tuple[str, bytes]]:
    """What every specialist's VM starts with: the database and the helper scripts."""
    files = [("/data/chinook.db", DATABASE.read_bytes())]
    for script in sorted((HERE / "sandbox").glob("*.py")):
        files.append((f"/tools/{script.name}", script.read_bytes()))
    return files


# Built (or its checkpoint found) once per process; rebuilt once if the checkpoint is deleted meanwhile.
templates = TemplateCache(lambda: lifecycle.build_template("board-pack", setup=SETUP, seed_files=seed_files()))


def build_orchestrator(sandboxes: SubagentSandboxes | None, thread_id: str, **kwargs: Any):
    """The orchestrator for one run; without sandboxes (introspection) everything sits on an in-memory backend."""
    if sandboxes is None:
        state = StateBackend()
        specs = [{**spec, "tools": specialist_tools(spec["name"], state)} for spec in SUBAGENTS]
        return create_deep_agent(model=strong_model, system_prompt=SYSTEM_PROMPT, subagents=specs,
                                 name="board-pack-orchestrator", **kwargs)
    # sandboxes.subagents are our specs with a per-thread sandbox backend attached; add the bound tools.
    specs = [specialist_spec(spec, sandboxes.backends[spec["name"]]) for spec in sandboxes.subagents]
    return create_deep_agent(
        model=strong_model,
        system_prompt=SYSTEM_PROMPT,
        subagents=specs,
        tools=orchestrator_tools(sandboxes, thread_id),
        backend=sandboxes.mount(StateBackend()),  # /subagents/<name>/ -> that specialist's VM, files only
        name="board-pack-orchestrator",
        **kwargs,
    )


@contextlib.asynccontextmanager
async def team_sandboxes(thread_id: str):
    """This thread's specialist sandboxes for one run: started on first use, paused when the run ends."""
    await asyncio.to_thread(templates.get)  # build it before the first specialist needs it
    async with asubagent_sandboxes(
        lifecycle, SUBAGENTS, thread_id=thread_id, default_model=strong_model,
        templates={spec["name"]: templates for spec in SUBAGENTS}, on_demand=True, store=store,
    ) as sandboxes:
        yield sandboxes


# https://docs.langchain.com/langsmith/graph-rebuild#context-manager-factory
@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    if not runtime.execution_runtime:
        yield build_orchestrator(None, "introspection")
        return
    thread_id = config.get("configurable", {}).get("thread_id")
    async with team_sandboxes(thread_id) as sandboxes:
        yield build_orchestrator(sandboxes, thread_id)
