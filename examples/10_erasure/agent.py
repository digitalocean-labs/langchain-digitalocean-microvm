"""Erasure request: a team of Deep Agents, one MicroVM per locator, and an Action Gateway that asks
a human before each irreversible step.

THE IDEA
A data-subject erasure request arrives: erase Jane Doe <jane.doe@example.com>. The team:

    erasure-coordinator  ─ no shell; plans from the locators' manifests, acts only through the Action Gateway
      ├─ crm        ┐
      ├─ support    ├─ in parallel, each in a MicroVM of its own cloned from one template: they pull the person's
      ├─ billing    │  records into their MicroVM (through the gateway, on this side), classify them with a script
      └─ marketing  ┘  there, and return a manifest: ids, categories, proposed actions; never a name or an address.
                     No shell: they read and write files there, and their tools run the script
    then the coordinator plans, and asks for each erasure. The gateway's policy:

      allow  searches and reads in every system
      ask    each erasure, and the billing redaction        -> `action_approval`: the run ends, the MicroVMs pause,
                                                              a human decides (run_script.py), the run resumes
      deny   deleting the billing customer, GDPR-deleting CRM files -> refused by the gateway, whoever asks

    and ends with `verify_erasure` (every system searched again) and `save_report` (an evidence report: ids, a hash).

Erasure is not "delete everything": a second Jane Doe (another address) is in the CRM and must stay, and billing keeps
its invoices (tax records), so the customer record is redacted, not deleted.

MODES (.env)
The five systems are mocks in memory (world.py) behind the package's local Action Gateway, in every mode. MicroVMs:
- DO_MICROVM_TRANSPORT=local: local directories, no account needed.
- otherwise: real MicroVMs (DO_MICROVM_API_TOKEN).

RUN (from the repository root): see README.md
    uv run examples/10_erasure/run_script.py
or  cd examples/10_erasure && uv run langgraph dev
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
from events import DemoLifecycle
from gateway import make_gateway
from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime
from models import model, strong_model
from tools import FilesOnly, Ledger, coordinator_tools, locator_tools
from world import SimulatedSystems

from langchain_digitalocean_microvm import TemplateCache
from langchain_digitalocean_microvm.subagents import (
    SubagentSandboxes,
    asubagent_sandboxes,
    per_subagent_specs,
)

logger = logging.getLogger(__name__)

# DO_MICROVM_* from the environment; logs every lifecycle step (events.py).
lifecycle = DemoLifecycle()
LOCAL = lifecycle.config.transport == "local"
systems = SimulatedSystems()  # the five mock systems, one set for every run of this process
gateway, audited = make_gateway(systems)  # opened on first use
ledger = Ledger(audited.audit)  # what the evidence report is built from

# The locators have no shell (tools.FilesOnly): their tools fetch the records and run the classifier in their MicroVMs.
NO_SHELL = "You have no shell: your tools fetch the records and classify them in your MicroVM."

# ── The team ──────────────────────────────────────────────────────────────────

LOCATED = {
    "crm": "the CRM: contacts and their notes",
    "support": "the support desk: contacts and their notes",
    "billing": "billing (payments): customers and their invoices",
    "marketing": "email marketing and product analytics: list members and profiles",
}


def _locator(name: str, what: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Locator for {what}, with its own MicroVM. Give it the requester's email address and name; it answers "
                       "with a manifest of the records it found: ids, identity match, personal-data categories, the proposed "
                       "action and the call that performs it. The manifest holds no personal data.",
        "system_prompt": (
            f"You are the {name} locator of an erasure request, working alone in your own Linux MicroVM; your systems: {what}. "
            "Call `fetch_records` with the email address and the name you are given, then `classify_records`. Answer with "
            "the JSON `classify_records` printed, exactly as printed, and nothing else. Never read the files under /data/ "
            "and never write a name, email address, phone number or address in your answer. Do not plan with write_todos. "
            "If a tool errors, say so. " + NO_SHELL
        ),
        "model": model,
    }


SUBAGENTS: list[dict[str, Any]] = [_locator(name, what) for name, what in LOCATED.items()]

SYSTEM_PROMPT = (
    "You are the erasure coordinator of a company that has received a data-subject erasure request. You lead four "
    "locators, each in an isolated MicroVM: `crm`, `support`, `billing` and `marketing` (email marketing and product "
    "analytics). You have no shell. Your only way to change a system is `action_invoke`, which calls an action by name "
    "through the Action Gateway; the gateway's policy, not you, decides what runs: every erasure waits for a human, some "
    "actions are refused. Never write a person's name, email address, phone number or address in a rationale, in your plan "
    "or in the report's notes: name records by system and id. If a human asks for an action, call it and report what the "
    "gateway answered, quoting a refusal as worded and never guessing why the policy refused; never look for another way "
    "around a refusal. To delete a billing customer the action is "
    "`stripe_delete_customers_customer` with arguments {customer: <id>}.\n\n"
    "When a request comes in:\n"
    "1. LOCATE: in ONE message, four `task` calls at once, one per locator, each given the email address and the name.\n"
    "2. PLAN: from the four manifests, a short table: system (crm, support, billing, marketing for a list `member`, "
    "analytics for a `profile`), kind, id, action (erase / redact / retain / skip) and why, as the manifests propose. A record whose "
    "identity is not `match` belongs to someone else: skip it. Retained records stay; their customer record is redacted.\n"
    "3. ERASE: for each record whose action is erase or redact, in the table's order and ONE AT A TIME (each waits for a "
    "human), call `action_invoke` with the record's `call` (tool and arguments exactly as the manifest gives them) and a "
    "rationale of at most 2 sentences: the system, the record id, that its email address matched the request exactly, and "
    "for a redaction what is kept and why. If one is denied, note it and go on with the next.\n"
    "4. VERIFY: call `verify_erasure` with the email address, the name and every record of the plan ({system, kind, id, "
    "action}), skipped and retained ones included.\n"
    "5. REPORT: call `save_report` with notes of at most 4 lines: what was erased, redacted, kept and skipped, and what was "
    "denied, by id.\n"
    "Finish with a 3-line summary: erased, kept (and why), open. Keep replies short; use exact ids."
)

# ── Template, sandboxes, coordinator ──────────────────────────────────────────


def seed_files() -> list[tuple[str, bytes]]:
    """What every locator's MicroVM starts with: the classifier."""
    return [(f"/tools/{script.name}", script.read_bytes()) for script in sorted((HERE / "sandbox").glob("*.py"))]


# Built (or its checkpoint found) once per process; rebuilt once if the checkpoint is deleted meanwhile.
templates = TemplateCache(lambda: lifecycle.build_template("erasure", seed_files=seed_files()))


def build_coordinator(sandboxes: SubagentSandboxes | None, **kwargs: Any):
    """The coordinator for one run; without sandboxes (introspection) everything sits on an in-memory backend."""
    if sandboxes is None:
        specs = [{**spec, "tools": []} for spec in SUBAGENTS]
        return create_deep_agent(model=strong_model, system_prompt=SYSTEM_PROMPT, subagents=specs,
                                 name="erasure-coordinator", **kwargs)
    # Each locator's file tools reach its MicroVM through FilesOnly, so deepagents gives it no `execute`; its own tools
    # hold the sandbox and run the classifier.
    files_only = {name: FilesOnly(backend) for name, backend in sandboxes.backends.items()}
    specs = [{**spec, "tools": locator_tools(spec["name"], sandboxes.backends[spec["name"]], gateway, local=LOCAL)}
             for spec in per_subagent_specs(SUBAGENTS, backends=files_only, default_model=strong_model)]
    return create_deep_agent(
        model=strong_model,
        system_prompt=SYSTEM_PROMPT,
        subagents=specs,
        tools=[*gateway.tools(), *coordinator_tools(gateway, ledger)],
        backend=sandboxes.mount(StateBackend()),  # /subagents/<name>/ -> that locator's MicroVM, files only
        name="erasure-coordinator",
        **kwargs,
    )


@contextlib.asynccontextmanager
async def team_sandboxes(thread_id: str):
    """This thread's locator sandboxes for one run: started on first use, paused when the run ends (an approval ends it)."""
    await asyncio.to_thread(templates.get)  # build it before the first locator needs it
    async with asubagent_sandboxes(
        lifecycle, SUBAGENTS, thread_id=thread_id, default_model=strong_model,
        templates={spec["name"]: templates for spec in SUBAGENTS}, on_demand=True,
    ) as sandboxes:
        yield sandboxes


# https://docs.langchain.com/langsmith/graph-rebuild#context-manager-factory
@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    if not runtime.execution_runtime:
        yield build_coordinator(None)
        return
    thread_id = config.get("configurable", {}).get("thread_id")
    async with team_sandboxes(thread_id) as sandboxes:
        yield build_coordinator(sandboxes)
