"""Incident response: a team of Deep Agents, one MicroVM per investigator, and an Action Gateway that asks a human
before anything irreversible.

THE IDEA
A checkout service on a droplet starts failing minutes after a deploy. An alert starts the team:

    incident-commander  ─ no shell, no action tools: it reads through the investigators and acts through the remediator
      ├─ metrics   ┐
      ├─ changes   ├─ in parallel, each in a MicroVM of its own cloned from one template: they pull evidence
      └─ infra     ┘  (metrics, commits, the droplet and its snapshots) and analyse it with scripts in the MicroVM;
                      no shell: they read and write files there, and their tools run the scripts
      ├─ remediator ─ the only agent with an action tool (`action_invoke`); no MicroVM, it calls the gateway
      └─ verifier   ─ after the fix, in a MicroVM of its own: samples the service's health and judges it (reads only)
    the commander names the cause, the remediator asks to restore the droplet from its snapshot, and the verifier,
    which did not do the restore, says whether the service is healthy again. The gateway's policy:

      allow  commit reads, the droplet, its snapshots
      ask    restore the droplet       -> `action_approval`: the run ends, the MicroVMs pause,
                                          a human decides (run_script.py), the run resumes
      deny   delete the droplet        -> refused by the gateway, whoever asks

Credentials stay with the Action Gateway: no MicroVM holds a token, and an approval is decided with the
operator's own token, never by an agent.

MODES (.env)
- DO_MICROVM_TRANSPORT=local: MicroVMs are local directories, the gateway and the droplet are in memory
  (incident_common/world.py). No account needed.
- otherwise: real MicroVMs (DO_MICROVM_API_TOKEN), the Action Gateway (DO_ACTION_GATEWAY_TOKEN,
  DO_ACTION_GATEWAY_ACTOR_ID) and a real droplet (incident_common/setup_droplet.py writes
  incident_common/outputs/incident.env). See README.md.

RUN (from the repository root): see README.md
    uv run examples/09_incident_response/run_script.py
or  cd examples/09_incident_response && uv run langgraph dev
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:  # examples/: the shared incident scenario (incident_common)
    sys.path.insert(0, str(HERE.parent))
from incident_common import OUTPUTS, SANDBOX  # noqa: E402

load_dotenv(HERE.parents[1] / ".env")  # the shell wins over .env; before the modules below read the environment
load_dotenv(OUTPUTS / "incident.env")  # written by incident_common/setup_droplet.py (real mode)

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from gateway import make_gateway
from incident_common.events import DemoLifecycle
from incident_common.models import model, strong_model
from incident_common.world import make_service
from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime
from tools import (
    FilesOnly,
    changes_tools,
    commander_tools,
    metrics_tools,
    verifier_tools,
)

from langchain_digitalocean_microvm import TemplateCache
from langchain_digitalocean_microvm.subagents import (
    SubagentSandboxes,
    asubagent_sandboxes,
    per_subagent_specs,
)

logger = logging.getLogger(__name__)

# DO_MICROVM_* from the environment; logs every lifecycle step (incident_common/events.py).
lifecycle = DemoLifecycle()
LOCAL = lifecycle.config.transport == "local"
service = make_service(LOCAL)  # the simulated world, or the real service named by DO_INCIDENT_*
gateway = make_gateway(service, local=LOCAL)  # opened on first use; one session for every run of this process

# The investigators have no shell (tools.FilesOnly): their tools fetch the evidence and run the scripts in their MicroVMs.
NO_SHELL = (
    "You have no shell: your tools fetch the evidence into /data/ and run the analysis in your MicroVM; credentials stay "
    "with the gateway. If their results are not enough, say what is missing."
)

# ── The team ──────────────────────────────────────────────────────────────────

SUBAGENTS: list[dict[str, Any]] = [
    {
        "name": "metrics",
        "description": "Metrics Analyst, with its own MicroVM. Ask when the errors started and how bad they got; it "
                       "answers with the onset time, the peak, and which version came in when.",
        "system_prompt": (
            "You are the metrics analyst of an incident, working alone in your own Linux MicroVM. In ONE turn call "
            "`fetch_metrics`, then `find_onset`, and answer from its JSON: the baseline error rate, the onset time (UTC), "
            "the peak error rate and latency, the version serving at the onset, the one before it, and when the version changed. Do not plan with write_todos. "
            "If a tool errors, say so. " + NO_SHELL
        ),
        "model": model,
    },
    {
        "name": "changes",
        "description": "Change Analyst, with its own MicroVM. Give it the repository (owner/name) and, if known, the "
                       "error onset; it answers with the change timeline and the commit just before the onset.",
        "system_prompt": (
            "You are the change analyst of an incident, working alone in your own Linux MicroVM. Call `fetch_commits` "
            "for the repository you are given, then `build_timeline` (pass the error onset if you have one), then "
            "`github_compare_commits` (base = the commit before, head = the last commit before the onset) to see which "
            "files the suspect commit changed. Answer with the timeline, the suspect commit (sha, message, minutes before "
            "the onset) and its changed files. Read no other files and do not plan with write_todos. " + NO_SHELL
        ),
        "model": model,
    },
    {
        "name": "infra",
        "description": "Infra Analyst, with its own MicroVM. Give it the droplet id; it answers with the droplet's state "
                       "and its snapshots (id, name, age).",
        "system_prompt": (
            "You are the infrastructure analyst of an incident, working alone in your own Linux MicroVM. In ONE turn "
            "call `digitalocean_droplet-get` for the droplet you are given and `digitalocean_image-list`. Answer with the "
            "droplet's status and size, and every snapshot: its numeric id, name and creation time, and whether it predates "
            "the deploy you are told about. Then `write_file` /outputs/infra.md with what you found (your report; your only file). "
            "You only read. Do not plan with write_todos. " + NO_SHELL
        ),
        "model": model,
    },
    {
        "name": "verifier",
        "description": "Recovery Validator, with its own MicroVM; it did not do the fix and only reads. Call it after the "
                       "restore is decided and done: it samples the service's health for a while and answers with a "
                       "verdict, healthy or not_healthy, and the evidence.",
        "system_prompt": (
            "You are the recovery validator of an incident, working alone in your own Linux MicroVM. You did not do the fix and you "
            "cannot change anything: you only check whether the service is healthy. Call `sample_health` with "
            "wait_seconds 30, then `judge_health`: its verdict is the one you report; do not read /data/health.json and judge it yourself. "
            "If the verdict is not_healthy, repeat both (at most 4 rounds in all: a "
            "restore can take a couple of minutes to come up). Then call `digitalocean_droplet-get` for the droplet you "
            "are given. Answer in 3 lines: the verdict (healthy or not_healthy, exactly one of these words), the latest sample "
            "(status, version, error rate) and the droplet's status. Report what the checks say, not what you expect. "
            "Do not plan with write_todos. " + NO_SHELL
        ),
        "model": model,
    },
]

# No MicroVM: it only calls the Action Gateway. Every action goes through it, so the policy (ask, deny) applies to it alone.
REMEDIATOR: dict[str, Any] = {
    "name": "remediator",
    "description": "The only agent that can change anything. Give it the exact action and its arguments (for the restore: "
                   "digitalocean_restore-droplet with ID and ImageID) and the reason, with the evidence in at most 3 sentences. It calls the "
                   "action through the gateway, which may wait for a human or refuse, and answers with what the gateway "
                   "answered. It does not verify the result.",
    "system_prompt": (
        "You are the remediator of an incident. You carry out one action that you are given, through `action_invoke` "
        "(tool name, arguments, and a `rationale` of at most 3 sentences with the evidence you were given). Action names "
        "are exact; if you are not sure of one, `action_search` first. Ids in an action's arguments are numbers, not text. "
        "The Action Gateway's policy decides what runs: some actions wait for a human, some are refused. Report exactly what the "
        "gateway answered (done, waiting, denied by the user, refused by policy). An answer that is the action itself (an action id "
        "with a status such as in-progress or completed) means it RAN, after a human approved it if one had to: report it as "
        "started, with its id and status. If the gateway refused the action, answer in one sentence: the action, and that the "
        "Action Gateway's policy refused it. Add nothing after it: no other way to do it and no one else who could. "
        "You do not verify the result and you do not investigate: another agent does that."
    ),
    "model": model,
}


def direct(*names: str) -> list:
    """The gateway's preloaded read tools with these names, for an investigator."""
    return [t for t in gateway.tools(meta=False) if t.name in names]


def specialist_tools(name: str, backend: Any) -> list:
    """The tools each role gets, bound to its own sandbox: nobody but the remediator can change anything."""
    if name == "metrics":
        return metrics_tools(backend, service, local=LOCAL)
    if name == "verifier":
        return [*verifier_tools(backend, service, local=LOCAL), *direct("digitalocean_droplet-get")]
    if name == "changes":
        return [*changes_tools(backend, gateway, local=LOCAL), *direct("github_compare_commits")]
    return direct("digitalocean_droplet-get", "digitalocean_image-list")


SYSTEM_PROMPT = (
    "You are the incident commander of a checkout service. You lead a team in isolated MicroVMs: three investigators "
    "(`metrics`, `changes`, `infra`), a `remediator` and a `verifier`. You have no shell and no action tools: you read "
    "through the investigators and you change things only by asking the `remediator`, which calls actions by name through the "
    "Action Gateway. The gateway's policy, not you or the remediator, decides what runs: some actions wait for a human, some "
    "are refused. If a human asks for an action, give it to the remediator with its exact arguments, whatever you think of "
    "it: the policy, not you, decides whether it runs. When the gateway refuses an action, your whole answer is one sentence: "
    "which action was refused, and that the Action Gateway's policy refused it. Add nothing after it: no other way to do "
    "it, no next step, and no one else (a role, team, person, process or tool) who could. Ids in an action's arguments are numbers, not text. The action "
    "to delete a droplet is `digitalocean_droplet-delete`.\n\n"
    "When an alert comes in:\n"
    "1. INVESTIGATE: in ONE message, three `task` calls at once: `metrics` (when did the errors start, how bad), `changes` "
    "(the repository from the alert; the last commit before the onset), `infra` (the droplet from the alert; its snapshots).\n"
    "2. DIAGNOSE: say in 3 short sentences what broke and why, citing the onset time, the commit and the minutes between.\n"
    "3. RESTORE: one `task` call to `remediator`: the action `digitalocean_restore-droplet`, arguments {ID: <droplet id>, "
    "ImageID: <the newest snapshot that predates the deploy>}, and the reason in at most 3 sentences: the cause with its "
    "evidence (the changed line), that the service is stateless so the restore loses no data, and which snapshot. A human "
    "approves it. Do nothing else until it is decided. If it is denied, stop and say so. If the remediator reports the restore "
    "as started (an action id, in-progress or completed), it was approved and ran: go on to VERIFY.\n"
    "4. VERIFY: one `task` call to `verifier` with the droplet id. It did not do the restore. Its verdict is the only thing that "
    "says the service is back: if it is not_healthy, say the restore did not bring the service back, and do not call the incident fixed.\n"
    "5. POSTMORTEM: call `save_postmortem` (a title and a Markdown body: summary, timeline, root cause, what was done, the verifier's "
    "verdict, follow-ups).\n"
    "Finish with a 3-line summary: cause, action taken, service state (the verifier's verdict). Keep replies short; use exact figures."
)

# ── Template, sandboxes, commander ────────────────────────────────────────────


def seed_files() -> list[tuple[str, bytes]]:
    """What every investigator's MicroVM starts with: the analysis helpers."""
    return [(f"/tools/{script.name}", script.read_bytes()) for script in sorted(SANDBOX.glob("*.py"))]


# Built (or its checkpoint found) once per process; rebuilt once if the checkpoint is deleted meanwhile.
templates = TemplateCache(lambda: lifecycle.build_template("incident", seed_files=seed_files()))


def build_commander(sandboxes: SubagentSandboxes | None, **kwargs: Any):
    """The commander for one run; without sandboxes (introspection) everything sits on an in-memory backend."""
    if sandboxes is None:
        specs = [{**spec, "tools": []} for spec in [*SUBAGENTS, REMEDIATOR]]
        return create_deep_agent(model=strong_model, system_prompt=SYSTEM_PROMPT, subagents=specs,
                                 name="incident-commander", **kwargs)
    # Each investigator's file tools reach its MicroVM through FilesOnly, so deepagents gives it no `execute`; the
    # role's own tools hold the sandbox and run the scripts.
    files_only = {name: FilesOnly(backend) for name, backend in sandboxes.backends.items()}
    specs = [{**spec, "tools": specialist_tools(spec["name"], sandboxes.backends[spec["name"]])}
             for spec in per_subagent_specs(SUBAGENTS, backends=files_only, default_model=strong_model)]
    meta = [t for t in gateway.tools() if t.name in ("action_invoke", "action_search")]
    specs.append({**REMEDIATOR, "tools": meta})  # the only holder of the action tools
    return create_deep_agent(
        model=strong_model,
        system_prompt=SYSTEM_PROMPT,
        subagents=specs,
        tools=commander_tools(),
        backend=sandboxes.mount(StateBackend()),  # /subagents/<name>/ -> that investigator's MicroVM, files only
        name="incident-commander",
        **kwargs,
    )


@contextlib.asynccontextmanager
async def team_sandboxes(thread_id: str):
    """This thread's investigator sandboxes for one run: started on first use, paused when the run ends (an approval ends it)."""
    await asyncio.to_thread(templates.get)  # build it before the first investigator needs it
    async with asubagent_sandboxes(
        lifecycle, SUBAGENTS, thread_id=thread_id, default_model=strong_model,
        templates={spec["name"]: templates for spec in SUBAGENTS}, on_demand=True,
    ) as sandboxes:
        yield sandboxes


# https://docs.langchain.com/langsmith/graph-rebuild#context-manager-factory
@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    # Built in a thread: in real mode the gateway's tools open its session with a synchronous request, which the
    # Agent Server refuses on its event loop unless it runs with --allow-blocking.
    if not runtime.execution_runtime:
        yield await asyncio.to_thread(build_commander, None)
        return
    thread_id = config.get("configurable", {}).get("thread_id")
    async with team_sandboxes(thread_id) as sandboxes:
        yield await asyncio.to_thread(build_commander, sandboxes)
