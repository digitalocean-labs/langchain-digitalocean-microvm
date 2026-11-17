"""The relay: a DigitalOcean Insights alert starts one incident commander in a MicroVM agent host of its own; it wakes the
commander when its specialists finish, brokers the read-only specialists' Action Gateway actions, and approvers decide here.

    uv run examples/12_incident_response_agents/relay.py serve --port 8787 --broker-port 8788 [--page v3]

Routes (the relay's port, for the console and approvers; keep it private):
    GET  /                                         the approvals page (web/v2, or --page / DO_INCIDENT_PAGE), its files
                                                   under /static
    POST /alerts                                   an Insights webhook delivery, signed (below): starts the incident
    GET  /incidents                                the commander hosts and what the relay last saw of each
    GET  /incidents/{id}                           the incident: pending approvals (with facts read here), last message, story
    DELETE /incidents/{id}                         close it: revoke its broker tokens, the commander host deletes its
                                                   specialists' hosts, then the relay deletes the commander host
    POST /incidents/{id}/decision                  {"interrupt_id", "decision": approve|deny|connected|cancel, "reason"?}
    POST /incidents/{id}/messages                  {"text"}: another turn (a follow-up from the on-call lead)
    GET  /incidents/{id}/runs/{run_id}/events      server-sent events: a run's tool calls and the story as it changes, then its end
    GET  /incidents/{id}/log?after=<offset>        the incident's event log from the commander's host: the Remediator's
                                                   gateway calls and the decisions
    GET  /incidents/{id}/postmortem                the postmortem as it stands: the recorded timeline and verdict, then the
                                                   commander's narrative (closing the incident writes it to a file first)
    GET  /service                                  the service's /health as the relay reads it (real mode; null in local mode)
    GET  /policy                                   the actions each role holds, and the Remediator's policy
Everything under /incidents, /service and /policy needs `Authorization: Bearer <DO_INCIDENT_OPERATOR_TOKEN>`: whoever holds it
can approve. The page asks for the token (or takes it from the URL fragment, which is never sent) and keeps it in the tab.

The broker (the broker's port, the only one the specialists' MicroVMs reach; in real mode at DO_INCIDENT_BROKER_URL):
    POST /incidents/{id}/actions/{role}            {"op": "list"} | {"op": "search", "query", "limit"} |
                                                   {"op": "invoke", "tool", "arguments", "rationale"} (the Remediator:
                                                   also "call", its tool call's id; no "list")
                                                   with `Authorization: Bearer <that role's token for that incident>`
The relay holds the gateway token (DO_ACTION_GATEWAY_TOKEN) and every gateway endpoint: one per read-only role (roles.py), each
allowing that role's actions and nothing else, and one per incident for the Remediator, whose policy asks a human before the
restore and refuses a delete. The Remediator's `invoke` of an `ask` action answers {"approval": {...}} and runs nothing: the
relay decides it (`Relay.decide`, after an approver's decision) and, once approved, runs the action itself. No token in a
specialist's MicroVM can approve anything or create a gateway endpoint of its own. A role's token for an incident is an HMAC of the incident and the role
under the broker key (DO_INCIDENT_BROKER_KEY, else a random key the relay keeps in outputs/broker.key; never derived from
the alerts' signing secret), so a restarted relay accepts the same tokens. Each token gets at most BROKER_RATE requests per
minute (429 beyond). A token works only for its own role in its own incident, and only while that incident is open: closing the
incident revokes it before any host is deleted. The relay gives the commander's host the tokens of its incident
(DO_INCIDENT_BROKER_TOKEN_<ROLE>); the commander's gateway gives each specialist's host its own role's token only.

For each new alert (signed with DO_INCIDENT_RELAY_SECRET: `X-DigitalOcean-Signature: t=<unix>,v1=<HMAC-SHA256 of "<unix>.<body>">`)
the relay answers 202 at once, gets an agent host `incident-<id>` (a MicroVM from DO_INCIDENT_HOST_IMAGE, built from host/; in
local mode a local `langgraph dev` serving host/), creates the incident's thread and starts the commander's first run. The
commander starts its specialists as async sub-agents; each runs in an agent host of its own, created by the commander's gateway.

What the relay adds for sub-agents on other servers:
- **Waking the commander.** A sub-agent on another server cannot wake its parent. While an incident has work open, the relay
  reads its tasks every WATCH_SECONDS through the commander host (`/incident/tasks`); when some finished that the commander has
  not seen, and the commander is idle, it starts a short turn on the commander's thread that says so. These reads also keep
  the commander's MicroVM from pausing while its specialists work (its gateway polls them).
- **Approvals in the sub-agent's thread.** The Remediator's `ask` ends ITS run with an `action_approval` interrupt; the commander
  is not involved. The relay lists those interrupts as the incident's pending approvals. On a decision it posts it to the
  gateway itself and, once approved, runs the action (the broker); then it resumes the Remediator's thread
  (`/incident/decision`, through the commander's gateway), whose repeated request reads the outcome from the broker.
- **Keeping hosts awake.** A MicroVM with no inbound requests is paused, even while it works. While a commander run is active
  the relay reads it every `keepalive` seconds. The relay records its hosts, their tokens and each incident's latest run in the
  package's gateway store, so a restarted relay resumes them (`Relay.adopt`) and `scripts/gc_microvms.py` knows them.

Local mode (DO_MICROVM_TRANSPORT=local) serves host/ and specialists/ from this checkout and the simulated world from this process;
the read-only specialists reach the simulated world directly (incident_common/world_server.py); the Remediator goes through the
broker on 127.0.0.1, as in real mode.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import contextlib
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE):  # examples/ (incident_common), and this example (roles.py, approvals.py)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from incident_common import OUTPUTS  # noqa: E402

load_dotenv(HERE.parents[1] / ".env")
load_dotenv(OUTPUTS / "incident.env")

import httpx
import postmortem
from approvals import facts_of, problems_of, read_target
from incident_common import events as event_log
from incident_common.events import log_event
from incident_common.sandbox import verdict as verdict_rules
from incident_common.world import DROPLET_NAME
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.errors import ConflictError
from roles import (
    BROKERED,
    BY_NAME,
    NAMES,
    REMEDIATOR,
    REMEDIATOR_ASK,
    REMEDIATOR_DENY,
    REMEDIATOR_TOOLS,
    ROLES,
    bare,
    broker_token_name,
    remediator_endpoint_prefix,
)

from langchain_digitalocean_microvm.agent_host import HostHandle, HostSpec, HostStatus
from langchain_digitalocean_microvm.agent_host.store import (
    TERMINAL_RUN_STATUSES,
    GatewayStore,
    HostRow,
    default_db_path,
)
from langchain_digitalocean_microvm.integrations.action_gateway import (
    ENDPOINT_PREFIX,
    PendingApproval,
)

logger = logging.getLogger(__name__)

GRAPH = "incident_commander"  # host/langgraph.json
SIGNATURE_HEADER = "x-digitalocean-signature"
MAX_AGE = 300  # seconds: an older signature is a replay
MAX_STREAM_DROPS, STREAM_RETRY_SECONDS = 100, 1.0  # a run's event stream from its host, picked up again after a drop
CLEANUP_TIMEOUT = 300.0  # seconds for the commander's host to delete its specialists' MicroVMs
READ_ATTEMPTS, READ_RETRY_SECONDS = 4, 0.5  # a read from a host that a dropped connection lost, repeated (with a growing pause)
KEEPALIVE = 30.0  # seconds between the relay's reads of an active run: keeps its host from pausing mid-run
WATCH_SECONDS = 3.0  # how often an incident with open work is read
SETTLE_SECONDS = 150.0  # after its last task ends, the incident is read until its specialists' hosts show paused, at most this long
OPEN = ("pending", "running", "waiting")  # a task that is not over
OVER = ("success", "error", "cancelled", "timeout")
# What the commander needs in its host: the MicroVMs API for its specialists' hosts, the models, the service, the specialists'
# image, and the Remediator's gateway (passed on to its host only).
# No DO_ACTION_GATEWAY_*: the gateway token stays in the relay (its broker), in neither the commander's host nor a specialist's.
HOST_ENV_PREFIXES = ("DO_MICROVM_", "DO_INCIDENT_", "DO_INFERENCE_", "EXAMPLES_")
HOST_ENV_NAMES = ("DO_MODEL_ACCESS_KEY", "ANTHROPIC_API_KEY")
# Not for the host: the agent-host settings (they configure this relay's hosts), the commander's image, the relay's secrets and
# the broker's settings (the relay sets the incident's broker variables itself, see Relay.spec_for).
HOST_ENV_EXCLUDED_PREFIXES = ("DO_MICROVM_AGENT_", "DO_MICROVM_SUBAGENT_", "DO_INCIDENT_HOST_", "DO_INCIDENT_RELAY_",
                              "DO_INCIDENT_BROKER_")
HOST_ENV_EXCLUDED = ("DO_MICROVM_HOST_MODE", "DO_MICROVM_WARM_START", "DO_MICROVM_GATEWAY_TOKEN", "DO_INCIDENT_OPERATOR_TOKEN",
                     "DO_INCIDENT_ID", "DO_INCIDENT_PAGE")
BROKER_BODY_LIMIT = 64 * 1024  # bytes of a broker request
BROKER_RATE = 60  # requests per minute per role and incident: an agent in a loop is stopped here, not at the gateway
BROKER_KEY_FILE = HERE / "outputs" / "broker.key"
RATIONALE_LIMIT = 500


class SignatureError(ValueError):
    """The delivery is not signed with our secret, or the signature is too old."""


def sign(secret: str, body: bytes, timestamp: int | None = None) -> str:
    """The header value Insights sends for `body` (also used to simulate a delivery)."""
    t = int(time.time()) if timestamp is None else timestamp
    digest = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={t},v1={digest}"


def verify(secret: str, header: str | None, body: bytes, *, now: float | None = None) -> int:
    """Raise SignatureError unless `header` signs `body` with `secret` within the last MAX_AGE seconds; returns when it was
    signed (unix time)."""
    if not header:
        raise SignatureError("no signature")
    fields = dict(part.split("=", 1) for part in header.split(",") if "=" in part)
    try:
        t = int(fields["t"])
    except (KeyError, ValueError):
        raise SignatureError("malformed signature") from None
    if abs((time.time() if now is None else now) - t) > MAX_AGE:
        raise SignatureError("signature too old")
    expected = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, fields.get("v1", "")):
        raise SignatureError("bad signature")
    return t


def incident_id(body: bytes) -> str:
    """The incident's id, from the alert body: a retried delivery of the same alert is the same incident."""
    return hashlib.sha256(body).hexdigest()[:8]


def host_id_for(incident: str) -> str:
    return f"incident-{incident}"


def thread_id_for(incident: str) -> str:
    """The incident's thread on its host: a UUID (the Agent Server takes no other thread id), the same for the same incident."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"microvm-incident-agents:{incident}"))


def host_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The relay's variables the commander needs in its host (the MicroVM's environment, set when it is created)."""
    environ = dict(os.environ if environ is None else environ)
    return {
        name: value for name, value in sorted(environ.items())
        if value and (name in HOST_ENV_NAMES or name.startswith(HOST_ENV_PREFIXES))
        and name not in HOST_ENV_EXCLUDED and not name.startswith(HOST_ENV_EXCLUDED_PREFIXES)
    }


def host_spec(local: bool, environ: dict[str, str] | None = None) -> HostSpec:
    """The commander's host: host/ from this checkout (local), else the image built from it (DO_INCIDENT_HOST_IMAGE)."""
    environ = dict(os.environ if environ is None else environ)
    if local:
        return HostSpec(graph_id=GRAPH, env={**host_env(environ), "DO_MICROVM_TRANSPORT": "local"})
    image = environ.get("DO_INCIDENT_HOST_IMAGE")
    if not image:
        raise SystemExit("real mode: set DO_INCIDENT_HOST_IMAGE (the commander's image: "
                         "images/agent-host/build_image.py --spec examples/12_incident_response_agents/host/image.json)")
    if not environ.get("DO_INCIDENT_SPECIALISTS_IMAGE"):
        raise SystemExit("real mode: set DO_INCIDENT_SPECIALISTS_IMAGE (the specialists' image: "
                         "images/agent-host/build_image.py --spec examples/12_incident_response_agents/specialists/image.json)")
    return HostSpec(graph_id=GRAPH, image=image, env=host_env(environ))


def make_hosts(local: bool) -> Any:
    """Local: a local `langgraph dev` per host, serving host/. Real: a MicroVM per host (MicroVMAgentHost)."""
    from langchain_digitalocean_microvm.agent_host import (
        LocalDevServerHost,
        MicroVMAgentHost,
    )

    return LocalDevServerHost(template_dir=HERE / "host") if local else MicroVMAgentHost()


def make_store(local: bool) -> GatewayStore:
    """The package's gateway store, the one scripts/gc_microvms.py reads (per host mode and namespace)."""
    from langchain_digitalocean_microvm.config import MicroVMConfig

    return GatewayStore(default_db_path("local" if local else "microvm", MicroVMConfig.from_env().namespace))


def host_client(handle: HostHandle, transport: httpx.AsyncBaseTransport | None, edge: dict[str, str] | None = None) -> LangGraphClient:
    """An Agent Protocol client for one host: the edge's credential (a MicroVM's endpoint) and the host's token."""
    http = httpx.AsyncClient(base_url=handle.url, transport=transport, headers={**(edge or {}), "x-api-key": handle.token},
                             timeout=httpx.Timeout(30.0, read=None))
    return LangGraphClient(http)


def _alert_of(body: bytes) -> dict[str, Any]:
    event = json.loads(body or b"{}")
    alert = event.get("data", event) if isinstance(event, dict) else {}
    return alert if isinstance(alert, dict) else {}


def first_message(body: bytes, service: Any) -> str:
    """The commander's first message from an Insights delivery (its alert is the body's `data`): the alert as DigitalOcean
    raised it, and where the service lives; nothing about the cause."""
    alert = _alert_of(body)
    value = alert.get("value")
    reading = f" at {value:.0f}" if isinstance(value, (int, float)) else ""
    return (f"ALERT from DigitalOcean Insights: rule {alert.get('alert_rule_name')} is {alert.get('state')} "
            f"({alert.get('severity')}): {alert.get('metric')}{reading} on {alert.get('resource_name')}.\n"
            f"It is a stateless service (orders are in an external database) on droplet {service.droplet_id} "
            f"({DROPLET_NAME}), code in {service.repo}. Find the cause and restore service.")


def simulated_alert(service: Any) -> dict[str, Any]:
    """An Insights alert's `data` for the service as it is now (console.py demo, tests), in the shape a real delivery has:
    a rule on the droplet's CPU utilization, an Insights metric (the service's error rate is not one). The bad deploy really
    does pin the CPU (service/app.py), but the service does not report its CPU, so the reading is illustrative: 98 percent
    while it fails, 12 otherwise."""
    failing = service.health().get("status") == "failing"
    return {"alert_rule_name": "incident-cpu", "state": "Triggered", "severity": "SEVERITY_CRITICAL",
            "metric": "do.droplets.cpu_utilization", "resource_urn": f"do:droplet:{service.droplet_id}",
            "resource_name": DROPLET_NAME, "value": 98.0 if failing else 12.0}


def alarm_summary(body: bytes, signed_at: float) -> dict[str, Any]:
    """What the page shows of the alert that started an incident: rule, metric, value, resource, and when it was sent."""
    alert = _alert_of(body)
    fields = {"rule": "alert_rule_name", "metric": "metric", "value": "value", "resource": "resource_name",
              "severity": "severity", "state": "state"}
    summary: dict[str, Any] = {
        name: value if isinstance(value, (int, float, str)) or value is None else str(value)
        for name, key in fields.items() for value in [alert.get(key)]
    }
    return {**summary, "at": signed_at}


# -- the story of an incident: what the specialists found, the verdict, the postmortem, and how far it got ------------------------

STEPS = ("alert", "investigate", "diagnosis", "approval", "restore", "verify", "report")
INVESTIGATORS = ("metrics", "changes", "infra")
FINDING_LIMIT, REPORT_LIMIT = 8000, 60000  # characters kept of a specialist's report and of the postmortem


def new_story() -> dict[str, Any]:
    """started: {task id: role}; sent: {tool run id: {"role", "at"}}; findings: {task id: {"role", "text", "at"}}; verdict,
    report, decision: None until known."""
    return {"started": {}, "sent": {}, "findings": {}, "verdict": None, "report": None, "decision": None}


def note_sent(story: dict[str, Any], run_id: str, role: str, at: float) -> bool:
    """A `start_async_task` call starting: the commander sent `role`. Returns whether it is new."""
    if not run_id or run_id in story["sent"]:
        return False
    story["sent"][run_id] = {"role": role, "at": at}
    return True


def verdict_of(text: str) -> str | None:
    """The Recovery Validator answers with its verdict first: healthy or not_healthy (exactly one of these words)."""
    match = re.search(r"\b(not_healthy|healthy)\b", text[:400])
    return match.group(1) if match else None


def relay_check(verdict: str, health: dict[str, Any] | None, at: float) -> dict[str, Any] | None:
    """The relay's own reading of the service against the Recovery Validator's verdict: what it sees now (one sample, judged by
    the validator's own rule) and whether it agrees. None when the relay does not probe the service. The validator is an agent
    in a MicroVM; this reading does not depend on it."""
    if health is None:
        return None
    seen = "healthy" if verdict_rules.is_healthy(health) else "not_healthy"
    return {"verdict": seen, "status": health.get("status"), "error_rate": health.get("error_rate"),
            "version": health.get("version"), "agrees": seen == verdict, "at": at}


def note_finding(story: dict[str, Any], task: str, role: str, text: str, at: float) -> bool:
    """A finished task's answer, once. Returns whether it is new."""
    story["started"].setdefault(task, role)
    if task in story["findings"]:
        return False
    story["findings"][task] = {"role": role, "text": text[:FINDING_LIMIT], "at": at}
    if role == "verifier" and (verdict := verdict_of(text)):
        story["verdict"] = {**(story["verdict"] or {}), "verdict": verdict, "at": at}
    return True


def note_report(story: dict[str, Any], title: Any, body: Any, at: float) -> None:
    story["report"] = {"title": str(title or "Postmortem")[:300], "body": str(body or "")[:REPORT_LIMIT], "at": at}


def story_from_messages(story: dict[str, Any], messages: list[dict[str, Any]], at: float) -> None:
    """The postmortem as the commander's thread holds it (the arguments of `save_postmortem`)."""
    for message in messages:
        if message.get("type") == "ai":
            for call in message.get("tool_calls") or []:
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                if call.get("name") == "save_postmortem" and not story["report"]:
                    note_report(story, args.get("title"), args.get("body"), at)


def progress(story: dict[str, Any], status: str | None, pending: int) -> list[dict[str, str]]:
    """Each step of the incident: done | active | waiting | failed | todo, from what the story holds."""
    started = list(story["started"].values())
    found = [f["role"] for f in story["findings"].values()]
    investigators = [r for r in started if r in INVESTIGATORS] or [r for r in found if r in INVESTIGATORS]
    verdict = (story["verdict"] or {}).get("verdict")
    busy = status == "busy"
    state: dict[str, str] = {"alert": "done"}
    later = ("remediator" in started + found or "verifier" in started + found or pending > 0 or bool(story["decision"])
             or bool(story["report"]))
    if investigators and all(found.count(r) >= investigators.count(r) for r in set(investigators)):
        state["investigate"] = "done"
    else:
        state["investigate"] = "done" if later else ("active" if busy or investigators else "todo")
    state["diagnosis"] = "done" if later else ("active" if state["investigate"] == "done" and busy else "todo")
    if pending:
        state["approval"] = "waiting"
    elif story["decision"] == "deny":
        state["approval"] = "failed"
    elif story["decision"] in ("approve", "connected") or "remediator" in found:
        state["approval"] = "done"
    else:
        state["approval"] = "todo"
    if "remediator" in found:
        state["restore"] = "done"
    else:
        state["restore"] = "active" if state["approval"] == "done" else "todo"
    if verdict == "not_healthy":
        state["verify"] = "failed"
    elif verdict == "healthy" or "verifier" in found:
        state["verify"] = "done"
    else:
        state["verify"] = "active" if "verifier" in started else "todo"
    state["report"] = "done" if story["report"] else ("active" if busy and state["verify"] in ("done", "failed") else "todo")
    return [{"step": step, "state": state[step]} for step in STEPS]


def public_story(story: dict[str, Any], status: str | None, pending: int) -> dict[str, Any]:
    """What the page shows: the roles sent, the findings in the order they came, the verdict, the postmortem, the steps."""
    findings = sorted(story["findings"].values(), key=lambda f: f["at"])
    sent = sorted(story.get("sent", {}).values(), key=lambda s: s["at"])
    return {"sent": sent, "findings": findings, "verdict": story["verdict"], "report": story["report"],
            "progress": progress(story, status, pending)}


_HOST_STATUS = {"provisioning": "creating", "booting": "creating", "ready": "running", "busy": "running", "paused": "paused",
                "retiring": "deleting", "destroyed": "deleted", "failed": "failed"}


def host_tile(view: dict[str, Any], previous: dict[str, Any] | None, now: float) -> dict[str, Any] | None:
    """A task's agent host as the page's tile: {"status", "note", "since", "ts"}; None before the gateway created it."""
    host = view.get("host")
    if not host:
        return {"status": "creating", "note": "its agent host is being created", "since": (previous or {}).get("since", now),
                "ts": now} if view.get("live") in ("pending", "running") else previous
    status = _HOST_STATUS.get(str(host.get("status")), str(host.get("status")))
    note = {"running": "working", "waiting": "waiting for a human", "success": "done", "error": "failed",
            "cancelled": "cancelled", "pending": "starting"}.get(str(view.get("live")), str(view.get("live")))
    since = previous["since"] if previous and previous.get("status") == status else now
    return {"status": status, "note": f"agent host · {note}", "since": since, "ts": now}


@dataclass
class Started:
    incident: str
    host_id: str
    thread_id: str
    run_id: str | None  # None: still starting (see Relay.wait_started), or a retried delivery
    duplicate: bool = False


_GONE = (HostStatus.DESTROYED.value, HostStatus.FAILED.value, HostStatus.RETIRING.value)  # store rows of hosts gone or going


class UnknownIncident(LookupError):
    """No host for this incident (never started, or deleted)."""


class NotPending(LookupError):
    """The approval is not waiting any more (decided, or the task moved on), or the thread is busy."""


WEB = HERE / "web"  # the approvals pages, one directory each: index.html at /, app.js and app.css under /static
DEFAULT_PAGE = "v2"


def pages() -> list[str]:
    """The approvals pages there are (web/<name>/index.html)."""
    return sorted(p.name for p in WEB.iterdir() if (p / "index.html").is_file())


def page_dir(page: str | None = None) -> Path:
    """The page to serve: `page`, else DO_INCIDENT_PAGE, else DEFAULT_PAGE."""
    name = page or os.environ.get("DO_INCIDENT_PAGE") or DEFAULT_PAGE
    if name not in pages():
        raise ValueError(f"no approvals page {name!r}: choose one of {', '.join(pages())}")
    return WEB / name


SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
                               "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
# approve/deny answer an `action_approval`; connected/cancel answer a `connect_required` (the actor opened the link, or not).
DECISIONS = ("approve", "deny", "connected", "cancel")
_INCIDENT_ID = re.compile(r"^[0-9a-f]{8}$")


def action_key(value: Any) -> str | None:
    """An approval's action, the tool and its arguments, as a comparable key; None for anything else (a connect link)."""
    if not isinstance(value, dict) or value.get("type") != "action_approval" or not value.get("tool"):
        return None
    return json.dumps([value["tool"], value.get("arguments") or {}], sort_keys=True, default=str)


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def notification(views: list[dict[str, Any]]) -> str:
    """The turn that wakes the commander: which sub-agents finished."""
    lines = [f"- `{v['role']}` (task {v['task_id']}): {v['live']}" for v in views]
    return ("Update from your sub-agents (sent by the relay, not by a person):\n" + "\n".join(lines)
            + "\nRead them with `check_async_task` and continue.")


async def start_run(client: LangGraphClient, thread_id: str, *, text: str) -> str:
    """A run on the commander's thread: a new message."""
    run = await client.runs.create(thread_id, GRAPH, input={"messages": [{"role": "user", "content": text}]},
                                   config={"recursion_limit": 200}, stream_mode=["events"], stream_resumable=True)
    return run["run_id"]


T = TypeVar("T")


async def reread(read: Callable[[], Awaitable[T]], *, attempts: int | None = None) -> T:
    """A read from a host, repeated when the connection drops it ("Server disconnected"): it changes nothing, so it is safe
    to repeat. Resuming a thread or starting a run is never repeated this way."""
    attempts = READ_ATTEMPTS if attempts is None else attempts
    for attempt in range(1, attempts + 1):
        try:
            return await read()
        except httpx.TransportError as exc:
            if attempt == attempts:
                raise
            logger.info("a read from a host was dropped (%s); reading again", exc)
            await asyncio.sleep(READ_RETRY_SECONDS * attempt)
    raise AssertionError("unreachable")


async def _active_run(client: LangGraphClient, thread_id: str) -> bool:
    """A run of the thread is pending or running (the thread's own status lags behind a run just created)."""
    for status in ("pending", "running"):
        if await reread(lambda status=status: client.runs.list(thread_id, status=status, limit=1)):
            return True
    return False


# -- the broker: the read-only specialists' actions, under each role's allow-only gateway endpoint ---------------------------------


class BrokerRefused(Exception):
    """A broker request refused: `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def broker_key(environ: Mapping[str, str] | None = None, path: Path | None = None) -> bytes:
    """The key the broker's tokens are derived with: DO_INCIDENT_BROKER_KEY, else a random key kept in `path` (default
    outputs/broker.key, created on first use, readable by its owner only). Not derived from the alerts' signing secret: whoever
    can sign an alert must not be able to mint broker tokens."""
    environ = os.environ if environ is None else environ
    if environ.get("DO_INCIDENT_BROKER_KEY"):
        return environ["DO_INCIDENT_BROKER_KEY"].encode()
    path = BROKER_KEY_FILE if path is None else path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):  # another relay process made it first: use that one
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as out:
                out.write(secrets.token_hex(32))
    key = path.read_text().strip()
    if len(key) < 32:
        raise ValueError(f"{path} holds no usable key: delete it (the incidents' broker tokens change) or set DO_INCIDENT_BROKER_KEY")
    return key.encode()


def broker_token(key: bytes, incident: str, role: str) -> str:
    """`role`'s token for `incident`: no other role, no other incident."""
    digest = hmac.new(key, f"broker\0{incident}\0{role}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


class Broker:
    """The gateway endpoints the specialists' actions run in (created on first use), and the tokens that reach them: one
    allow-only endpoint per read-only role, and one per incident for the Remediator (ask before the restore, refuse a delete),
    whose approvals only the relay decides (`decide`).

    `client_factory` gives each endpoint its own `GatewayClient` (the real one with the gateway token, or the simulated world's).
    `is_open(incident)` says whether an incident's tokens still work (the relay: its commander host is not gone or going).
    """

    def __init__(self, key: bytes, url: str, client_factory: Callable[[], Any], is_open: Callable[[str], bool],
                 pins: Mapping[str, Mapping[str, Any]] | None = None, rate: int = BROKER_RATE,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.key, self.url, self.client_factory, self.is_open = key, url.rstrip("/"), client_factory, is_open
        self.pins = {tool: dict(arguments) for tool, arguments in (pins or {}).items()}  # tool -> {argument: the one value allowed}
        self._gateways: dict[str, Any] = {}
        self._remediators: dict[str, Any] = {}  # incident -> the Remediator's gateway endpoint
        # The Remediator's requests that asked for an approval, by proposal id: {"incident", "call", "pending", "text"}; "text"
        # is the outcome once decided. A request repeated for the same tool call (a resumed run) finds its proposal again.
        self._proposals: dict[str, dict[str, Any]] = {}
        self._calls: dict[tuple[str, str], str] = {}  # (incident, tool call id) -> proposal id
        self._lock = threading.Lock()
        self.rate, self._clock = rate, clock
        self._recent: dict[tuple[str, str], collections.deque[float]] = {}  # (incident, role) -> its requests in the last minute

    def env(self, incident: str) -> dict[str, str]:
        """What the commander's host needs to give its specialists the broker: its URL and each role's token."""
        return {"DO_INCIDENT_BROKER_URL": self.url,
                **{broker_token_name(role): broker_token(self.key, incident, role) for role in BROKERED}}

    def gateway(self, role: str) -> Any:
        from langchain_digitalocean_microvm.integrations.action_gateway import (
            ActionGateway,
            ActionPolicy,
        )

        if role not in self._gateways:
            actions = list(BY_NAME[role].actions)
            self._gateways[role] = ActionGateway(ActionPolicy(allow=actions), tools=actions, preload=actions,
                                                 client=self.client_factory())
        return self._gateways[role]

    def check(self, incident: str, role: str, authorization: str | None) -> None:
        """Raise BrokerRefused unless `authorization` carries `role`'s token for `incident` and the incident is open."""
        if role not in BROKERED:
            raise BrokerRefused(404, f"no actions for the role {role!r}")
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not _INCIDENT_ID.match(incident) or not hmac.compare_digest(given.encode(), broker_token(self.key, incident, role).encode()):
            raise BrokerRefused(401, "invalid or missing broker token")
        if not self.is_open(incident):
            raise BrokerRefused(403, "the incident is closed: its broker tokens are revoked")
        now = self._clock()
        with self._lock:
            recent = self._recent.setdefault((incident, role), collections.deque())
            while recent and recent[0] <= now - 60:
                recent.popleft()
            if len(recent) >= self.rate:
                raise BrokerRefused(429, f"more than {self.rate} requests in a minute for the {role} role: slow down")
            recent.append(now)

    def remediator(self, incident: str) -> Any:
        """The incident's Remediator endpoint: the policy is the relay's, set here, not the agent's."""
        from langchain_digitalocean_microvm.integrations.action_gateway import (
            ActionGateway,
            ActionPolicy,
        )

        with self._lock:
            if incident not in self._remediators:
                policy = ActionPolicy(ask=list(REMEDIATOR_ASK), deny=list(REMEDIATOR_DENY))
                self._remediators[incident] = ActionGateway(policy, tools=list(REMEDIATOR_TOOLS), client=self.client_factory(),
                                                            endpoint_prefix=remediator_endpoint_prefix(incident))
            return self._remediators[incident]

    def answer(self, incident: str, role: str, body: Any) -> dict[str, Any]:
        """One request (already checked): list, search or invoke the role's actions. Blocking: run it in a thread."""
        if role == REMEDIATOR:
            return self._remediate(incident, body)
        if not isinstance(body, dict) or body.get("op") not in ("list", "search", "invoke"):
            raise BrokerRefused(400, 'a JSON object with "op": list, search or invoke is required')
        gateway = self.gateway(role)
        names = {bare(a) for a in BY_NAME[role].actions}
        if body["op"] == "list":
            actions = []
            for found in gateway.tools(meta=False):
                schema = found.args_schema if isinstance(found.args_schema, dict) else {}
                actions.append({"name": found.name, "description": found.description,
                                "input_schema": {k: v for k, v in schema.items() if k != "title"}})
            return {"actions": actions}
        if body["op"] == "search":
            limit = body.get("limit", 5)
            limit = min(max(limit, 1), 10) if isinstance(limit, int) else 5
            return {"actions": json.loads(gateway.search(str(body.get("query") or ""), limit))}
        tool, arguments = bare(str(body.get("tool") or "")), body.get("arguments") or {}
        if tool not in names:
            raise BrokerRefused(403, f"{tool or 'that'} is not one of the {role} role's actions")
        if not isinstance(arguments, dict):
            raise BrokerRefused(400, "arguments must be a JSON object")
        for name, wanted in self.pins.get(tool, {}).items():  # the agent picks its arguments: the incident's service is not its choice
            if str(arguments.get(name, "")).lower() != str(wanted).lower():
                raise BrokerRefused(403, f"{tool}: {name} must be {wanted!r}, the incident's own")
        rationale = str(body.get("rationale") or f"the {role} specialist of incident {incident}")[:RATIONALE_LIMIT]
        try:
            text = gateway.call(tool, arguments, rationale)
        except RuntimeError as exc:  # the gateway wanted a decision or a connection: an interrupt, outside a graph here
            log_event(event="broker", incident=incident, role=role, tool=tool, status="error", error=str(exc)[:300])
            raise BrokerRefused(502, f"{tool} needs the relay's gateway actor to connect an account first") from exc
        status = "error" if text.startswith("Error:") else "allow"
        log_event(event="broker", incident=incident, role=role, tool=tool, status=status, arguments=arguments,
                  error=text[:300] if status == "error" else "")
        return {"text": text}

    def _remediate(self, incident: str, body: Any) -> dict[str, Any]:
        """The Remediator's request: search its endpoint, or request an action. An `ask` action runs nothing here; it answers
        {"approval": {"proposal", "tool", "arguments", "rationale", "expires_at"}} until the relay decides it, then
        {"text": the outcome} to the same tool call."""
        if not isinstance(body, dict) or body.get("op") not in ("search", "invoke"):
            raise BrokerRefused(400, 'a JSON object with "op": search or invoke is required')
        gateway = self.remediator(incident)
        if body["op"] == "search":
            limit = body.get("limit", 5)
            limit = min(max(limit, 1), 10) if isinstance(limit, int) else 5
            return {"actions": json.loads(gateway.search(str(body.get("query") or ""), limit))}
        tool, arguments, call = bare(str(body.get("tool") or "")), body.get("arguments") or {}, str(body.get("call") or "")
        if tool not in {bare(a) for a in REMEDIATOR_TOOLS}:
            raise BrokerRefused(403, f"{tool or 'that'} is not one of the Remediator's actions")
        if not isinstance(arguments, dict):
            raise BrokerRefused(400, "arguments must be a JSON object")
        if not call or len(call) > 200:
            raise BrokerRefused(400, '"call", the tool call\'s id, is required')
        with self._lock:
            known = self._calls.get((incident, call))
            proposal = self._proposals.get(known) if known else None
        if proposal is not None:  # the same tool call again (its run resumed): the outcome, or the approval still waiting
            return {"text": proposal["text"]} if proposal["text"] is not None else {"approval": self._view(known, proposal)}
        rationale = str(body.get("rationale") or f"the Remediator of incident {incident}")[:RATIONALE_LIMIT]
        answer = gateway.request(tool, arguments, rationale)
        if isinstance(answer, PendingApproval):
            proposal_id = uuid.uuid4().hex
            proposal = {"incident": incident, "call": call, "pending": answer, "text": None}
            with self._lock:
                self._proposals[proposal_id] = proposal
                self._calls[(incident, call)] = proposal_id
            log_event(event="broker", incident=incident, role=REMEDIATOR, tool=tool, status="ask", arguments=arguments, error="")
            return {"approval": self._view(proposal_id, proposal)}
        status = ("deny" if "polic" in answer.lower() else "error") if answer.startswith(("Error:", "Not connected")) else "allow"
        log_event(event="broker", incident=incident, role=REMEDIATOR, tool=tool, status=status, arguments=arguments,
                  error=answer[:300] if status != "allow" else "")
        return {"text": answer}

    @staticmethod
    def _view(proposal_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
        pending = proposal["pending"]
        return {"proposal": proposal_id, "tool": pending.tool, "arguments": dict(pending.arguments),
                "rationale": pending.rationale, "expires_at": pending.expires_at}

    def decide(self, incident: str, proposal_id: str, decision: str, reason: str | None = None) -> str:
        """The relay's decision on one of the incident's proposals: posted to the gateway with the relay's token and, once
        approved, the action run by the relay. Returns the outcome (also what the Remediator's repeated request gets).
        Raises LookupError for a proposal it does not hold (another incident's, or one a restarted relay never saw)."""
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None or proposal["incident"] != incident:
                raise LookupError("no such proposal in this incident")
            if proposal["text"] is not None:
                return proposal["text"]
        text = self.remediator(incident).decide(proposal["pending"], decision, reason)
        with self._lock:
            proposal["text"] = text
        status = "denied" if text.startswith("The user denied") else (
            "error" if text.startswith(("Error:", "Not run", "Not connected", "The approval expired")) else "approved")
        log_event(event="broker", incident=incident, role=REMEDIATOR, tool=proposal["pending"].tool, status=status,
                  arguments=dict(proposal["pending"].arguments), error=text[:300] if status == "error" else "")
        return text

    def forget(self, incident: str) -> None:
        """A closed incident: delete its Remediator endpoint and drop its proposals."""
        with self._lock:
            gateway = self._remediators.pop(incident, None)
            for proposal_id in [p for p, v in self._proposals.items() if v["incident"] == incident]:
                proposal = self._proposals.pop(proposal_id)
                self._calls.pop((incident, proposal["call"]), None)
            for key in [k for k in self._recent if k[0] == incident]:
                del self._recent[key]
        if gateway is not None:
            gateway.close()

    def close(self) -> None:
        for gateway in (*self._gateways.values(), *self._remediators.values()):
            gateway.close()
        self._gateways.clear()
        self._remediators.clear()


class Relay:
    """Starts one commander per alert, wakes it when its specialists finish, keeps its host awake while it works, brokers the
    specialists' actions, and resumes their threads for approvers.

    `hosts` is a host manager (MicroVMAgentHost, or LocalDevServerHost in local mode); `store` the gateway store its hosts
    and their tokens are recorded in; `transport` reaches the hosts (tests). `gateway` is a read-only gateway endpoint of
    the relay's (the approver's facts come from it, not from the agents); None shows no facts. `broker`: None in local mode
    (the specialists reach the simulated world directly). `probe_service`: the relay reads the service's health itself (real
    mode). `keepalive`: seconds between the relay's reads of an active commander run (default: a third of the hosts' idle
    timeout, at most 30 s); `watch_seconds`: between the reads of an incident's tasks while it has work open. `endpoints`: a
    gateway client that lists and deletes endpoints (`list_endpoints`, `delete_endpoint`): the relay deletes the Remediator's
    endpoints of an incident it closes (`sweep`); None sweeps nothing. `postmortems`: the directory where closing an incident
    writes its postmortem (the recorded timeline and verdict, then the commander's narrative); None writes nothing.
    """

    def __init__(self, hosts: Any, secret: str, service: Any, *, spec: HostSpec, store: GatewayStore, gateway: Any = None,
                 broker: Broker | None = None, probe_service: bool = False, transport: httpx.AsyncBaseTransport | None = None,
                 keepalive: float | None = None, watch_seconds: float = WATCH_SECONDS, endpoints: Any = None,
                 postmortems: Path | None = None) -> None:
        if not secret:
            raise ValueError("the relay needs the channel's signing secret (DO_INCIDENT_RELAY_SECRET)")
        self.hosts, self.secret, self.service, self.spec, self.gateway = hosts, secret, service, spec, gateway
        self.store, self.transport, self.broker, self.endpoints = store, transport, broker, endpoints
        self.postmortems = postmortems  # where closing an incident writes its postmortem (None: nowhere)
        self.probe_service = probe_service  # the real service answers /health; a simulated world lives in this process
        idle = getattr(hosts, "idle_timeout_seconds", None)
        self.keepalive = keepalive if keepalive is not None else min(KEEPALIVE, idle / 3 if idle else KEEPALIVE)
        self.watch_seconds = watch_seconds
        edge = getattr(hosts, "edge_headers", None)
        self.edge: dict[str, str] = edge() if callable(edge) else {}
        self.seen: dict[str, dict[str, Any]] = {}  # incident -> what the relay last saw: status, pending approvals, run
        # What the relay passed on, kept for the page's fleet view so a paused host need not be read (memory only):
        self.alarms: dict[str, dict[str, Any]] = {}  # incident -> the alert that started it
        self.host_tiles: dict[str, dict[str, dict[str, Any]]] = {}  # incident -> role -> its latest task's agent host
        self.last_tool: dict[str, dict[str, Any]] = {}  # incident -> the latest tool call of a run streamed through here
        self.stories: dict[str, dict[str, Any]] = {}  # incident -> its story (new_story): findings, verdict, postmortem
        self.tasks: dict[str, list[dict[str, Any]]] = {}  # incident -> its tasks as last read
        self.pending: dict[str, list[dict[str, Any]]] = {}  # incident -> the approvals its sub-agents wait on
        self.subruns: dict[str, dict[str, dict[str, Any]]] = {}  # incident -> run id -> the task a decision resumed
        self._settling: dict[str, float] = {}  # incident -> when its last task ended while a specialist's host was still awake
        self._notified: dict[str, set[tuple[str, str, str]]] = {}  # incident -> (task, status, run) the commander was told
        self._locks: dict[str, asyncio.Lock] = {}
        # incident -> actions it has waiting / approved through this relay: {"waiting"|"approved": {action_key: unix time}}
        self._actions: dict[str, dict[str, dict[str, float]]] = {}
        self._decided: set[str] = set()  # interrupt ids answered through this relay
        self._closed: set[str] = set()  # incidents being closed: their broker tokens no longer work
        self._tasks: dict[str, asyncio.Task[Any]] = {}  # incident -> its start, then its active run's keepalive
        self._watchers: dict[str, asyncio.Task[Any]] = {}  # incident -> the reads of its tasks while work is open
        remember = getattr(hosts, "remember_token", None)
        for row in store.list_hosts(graph_id=GRAPH):  # hosts a previous relay started: their tokens are only here
            if row.status not in _GONE and callable(remember):
                remember(row.host_id, row.token)

    def is_open(self, incident: str) -> bool:
        """The incident's commander host is recorded and neither gone nor going, and nobody is closing it."""
        if incident in self._closed:
            return False
        row = self.store.get_host(host_id_for(incident))
        return row is not None and row.status not in _GONE

    def spec_for(self, incident: str) -> HostSpec:
        """The commander's host for one incident: the relay's spec, plus the incident's id and its broker variables."""
        env = {**self.spec.env, "DO_INCIDENT_ID": incident, **(self.broker.env(incident) if self.broker else {})}
        return dataclasses.replace(self.spec, env=env)

    # -- alerts -----------------------------------------------------------------------------------

    async def handle(self, body: bytes, signature: str | None) -> Started:
        """One delivery: check it, then start its incident in the background (or find it). Raises SignatureError."""
        signed_at = verify(self.secret, signature, body)
        incident = incident_id(body)
        self.alarms.setdefault(incident, alarm_summary(body, signed_at))
        return self.begin(incident, first_message(body, self.service))

    def begin(self, incident: str, text: str) -> Started:
        """Start the incident unless it is starting or started; answers at once (a cold host takes about a quarter minute)."""
        host_id, thread_id = host_id_for(incident), thread_id_for(incident)
        row = self.store.get_host(host_id)
        if incident in self._tasks or (row is not None and row.status not in _GONE):
            log_event(event="incident", incident=incident, host=host_id, duplicate=True)
            return Started(incident, host_id, thread_id, (self.seen.get(incident) or {}).get("run"), duplicate=True)
        self._closed.discard(incident)
        self.seen[incident] = {"status": "starting", "pending": 0, "run": None}
        self._tasks[incident] = asyncio.create_task(self._start(incident, text))
        return Started(incident, host_id, thread_id, None)

    async def wait_started(self, incident: str, timeout: float = 600.0) -> str:
        """The run the incident's start created (the console, tests); raises if the start failed."""
        deadline = time.monotonic() + timeout
        while True:
            seen = self.seen.get(incident) or {}
            if seen.get("run"):
                return seen["run"]
            if seen.get("status") == "error":
                raise RuntimeError(f"incident {incident} did not start: {seen.get('error')}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"incident {incident} did not start within {timeout:.0f} s")
            await asyncio.sleep(0.2)

    async def _start(self, incident: str, text: str) -> None:
        try:
            await self._start_inner(incident, text)
        finally:
            if self._tasks.get(incident) is asyncio.current_task():  # not replaced by the first run's keepalive
                self._tasks.pop(incident, None)

    async def _start_inner(self, incident: str, text: str) -> None:
        host_id, thread_id = host_id_for(incident), thread_id_for(incident)
        started = time.monotonic()
        try:
            handle = await asyncio.to_thread(self.hosts.get_or_create, host_id=host_id, spec=self.spec_for(incident))
            self._record_host(handle)
            log_event(event="host", incident=incident, host=host_id, seconds=round(time.monotonic() - started, 2))
            client = self._host_client(handle)
            try:
                try:
                    await client.threads.create(thread_id=thread_id, graph_id=GRAPH)
                except ConflictError:  # a host a previous relay started: the incident is already under way
                    self.seen[incident] = {"status": "unknown", "pending": 0, "run": None}
                    self.watch(incident)
                    return
                self.store.create_thread(thread_id, now=time.time())
                run_id = await start_run(client, thread_id, text=text)
            finally:
                await client.http.client.aclose()
        except Exception as exc:  # noqa: BLE001 - reported on the incident (GET /incidents) and in the log
            logger.exception("incident %s did not start", incident)
            self.seen[incident] = {"status": "error", "pending": 0, "run": None, "error": str(exc)[:300]}
            log_event(event="incident", incident=incident, host=host_id, error=str(exc)[:300])
            return
        self._began(incident, run_id)
        log_event(event="incident", incident=incident, host=host_id, thread=thread_id, run=run_id)

    # -- hosts, the store, the keepalive and the watchers ---------------------------------------------

    def _host_client(self, handle: HostHandle) -> LangGraphClient:
        return host_client(handle, self.transport, self.edge)

    def _record_host(self, handle: HostHandle) -> None:
        now = time.time()
        self.store.upsert_host(HostRow(
            host_id=handle.host_id, microvm_id=handle.microvm_id, graph_id=GRAPH, status=HostStatus.READY.value,
            url=handle.url, token=handle.token, booted_at=handle.booted_at, last_active_at=now, idle_since=None, failures=0))

    def _began(self, incident: str, run_id: str) -> None:
        """A run started on the commander's thread: record it (the sweeper keeps a host with an active run), keep its host
        awake until the run ends, and watch the incident's tasks."""
        self.seen[incident] = {**self.seen.get(incident, {}), "status": "busy", "run": run_id}
        self.store.update_thread(thread_id_for(incident), host_id=host_id_for(incident), graph_id=GRAPH, run_id=run_id,
                                 run_status="running", last_active_at=time.time())
        previous = self._tasks.pop(incident, None)
        if previous is not None and previous is not asyncio.current_task():  # the keepalive of the thread's previous run
            previous.cancel()
        self._tasks[incident] = asyncio.create_task(self._keep_awake(incident, run_id))
        self.watch(incident)

    async def _keep_awake(self, incident: str, run_id: str) -> None:
        """Read the run's status every `keepalive` seconds until it ends: a MicroVM with no inbound requests is paused,
        even while its commander works, and a model call frozen mid-flight fails when it is resumed."""
        thread_id = thread_id_for(incident)
        try:
            client = await self._client(incident)
        except UnknownIncident:
            return
        try:
            while True:
                await asyncio.sleep(self.keepalive)
                try:
                    run = await reread(lambda: client.runs.get(thread_id, run_id))
                except httpx.HTTPError as exc:
                    logger.info("incident %s: could not read run %s (%s); trying again", incident, run_id, exc)
                    continue
                self.store.claim_host(host_id_for(incident), now=time.time())
                if run["status"] in TERMINAL_RUN_STATUSES:
                    self.store.update_thread(thread_id, run_status=run["status"], last_active_at=time.time())
                    return
        finally:
            await client.http.client.aclose()
            if self._tasks.get(incident) is asyncio.current_task():
                self._tasks.pop(incident, None)

    def watch(self, incident: str) -> None:
        """Read the incident's tasks every `watch_seconds` while it has work open (started again by any turn or decision)."""
        task = self._watchers.get(incident)
        if task is None or task.done():
            self._watchers[incident] = asyncio.get_running_loop().create_task(self._watch(incident))

    async def _watch(self, incident: str) -> None:
        while True:
            await asyncio.sleep(self.watch_seconds)
            try:
                if not await self.refresh(incident):
                    return
            except UnknownIncident:
                return
            except Exception:  # noqa: BLE001 - one failed read; the next one may work
                logger.warning("incident %s: reading its tasks failed", incident, exc_info=True)

    async def adopt(self) -> list[str]:
        """Pick up the incidents a previous relay left open (a restarted relay; `serve` calls it on startup): keep the host
        of a commander run still active awake, and watch every open incident's tasks again (nothing else would wake its
        commander or keep its host awake while specialists work), and delete the Remediator's endpoints the closed incidents
    left (`sweep`). Returns the incidents adopted."""
        adopted, closed = [], []
        for row in self.store.list_hosts(graph_id=GRAPH):
            incident = row.host_id.removeprefix("incident-")
            if row.status in _GONE and _INCIDENT_ID.match(incident):
                closed.append(incident)
            if row.status in _GONE or not _INCIDENT_ID.match(incident) or incident in self._tasks:
                continue
            thread = self.store.get_thread(thread_id_for(incident))
            if thread is None:
                continue
            if thread.run_id is not None and not thread.terminal:
                self.seen.setdefault(incident, {"status": "busy", "pending": 0, "run": thread.run_id})
                self._tasks[incident] = asyncio.create_task(self._keep_awake(incident, thread.run_id))
            self.watch(incident)
            adopted.append(incident)
        if closed:  # a run killed before or while closing them left their Remediator's endpoints
            swept = await asyncio.to_thread(self.sweep, closed)
            if swept:
                log_event(event="relay", adopted=adopted, endpoints=swept)
        return adopted

    async def close(self) -> None:
        """Stop the background work (starts, keepalives, watchers); the hosts stay."""
        tasks = [*self._tasks.values(), *self._watchers.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._watchers.clear()

    async def _client(self, incident: str) -> LangGraphClient:
        """The incident's host, resumed if paused; never a new one (a new host would have lost the thread)."""
        from langchain_digitalocean_microvm.exceptions import SandboxError

        if not _INCIDENT_ID.match(incident):
            raise UnknownIncident(incident)
        host_id = host_id_for(incident)
        if not self.store.claim_host(host_id, now=time.time()):  # unknown here, or being deleted
            raise UnknownIncident(incident)
        try:
            handle = await asyncio.to_thread(self.hosts.resume, host_id=host_id)
        except SandboxError as exc:  # SandboxNotFoundError, or a host this relay holds no token for
            raise UnknownIncident(incident) from exc
        return self._host_client(handle)

    async def refresh(self, incident: str) -> bool:
        """Read the commander's thread and its tasks; wake the commander when tasks it has not seen are over. Returns whether
        the incident still has work open (the commander busy, or a task not over)."""
        async with self._locks.setdefault(incident, asyncio.Lock()):
            client = await self._client(incident)
            thread_id = thread_id_for(incident)
            try:
                thread = await reread(lambda: client.threads.get(thread_id))
                state = await reread(lambda: client.threads.get_state(thread_id))
                values = state.get("values") or {}
                tracked = values.get("async_tasks") or {}
                pairs = ",".join(f"{t['task_id']}:{t['agent_name']}" for t in tracked.values() if t.get("agent_name") in NAMES)
                response = await reread(lambda: client.http.client.get(
                    "/incident/tasks", params={"thread": thread_id, "tasks": pairs}))
                response.raise_for_status()
                views = response.json().get("tasks") or []
                busy = thread.get("status") == "busy" or await _active_run(client, thread_id)
                runs = await reread(lambda: client.runs.list(thread_id, limit=1))
                woken = None
                fresh = []
                if not busy:
                    notified = self._notified.setdefault(incident, set())
                    for view in views:
                        recorded = (tracked.get(view["task_id"]) or {}).get("status")
                        key = (view["task_id"], view["live"], str(view.get("run_id")))
                        if view["live"] in OVER and view["live"] != recorded and key not in notified:
                            fresh.append(view)
                            notified.add(key)
                    if fresh:
                        woken = await start_run(client, thread_id, text=notification(fresh))
            finally:
                await client.http.client.aclose()
        now = time.time()
        self.tasks[incident] = views
        story = self.stories.setdefault(incident, new_story())
        judged = False
        for view in views:
            story["started"].setdefault(view["task_id"], view["role"])
            if view["live"] == "success" and view.get("result"):
                if note_finding(story, view["task_id"], view["role"], view["result"], now) and view["role"] == "verifier":
                    judged = bool(story["verdict"])
        if judged:  # the validator just reported: read the service ourselves, once
            health = await self.service_health()
            check = relay_check(story["verdict"]["verdict"], health, time.time())
            if check is not None:
                story["verdict"]["relay_check"] = check
                log_event(event="relay", incident=incident, verifier_verdict=story["verdict"]["verdict"], relay_check=check)
        story_from_messages(story, values.get("messages") or [], now)
        pending = [{**stop, "task": view["task_id"], "role": view["role"]} for view in views for stop in view["interrupts"]
                   if stop["id"] not in self._decided]
        self.pending[incident] = pending
        tiles = self.host_tiles.setdefault(incident, {})
        for view in views:  # one tile per role: its latest task
            if (tile := host_tile(view, tiles.get(view["role"]), now)) is not None:
                tiles[view["role"]] = tile
        status = "busy" if busy or woken else ("interrupted" if pending else thread.get("status"))
        run = woken or (runs[0]["run_id"] if runs else None)
        self.seen[incident] = {"status": status, "pending": len(pending), "run": run,
                               "open": sum(1 for v in views if v["live"] in OPEN)}
        if woken:
            self._began(incident, woken)
            log_event(event="relay", incident=incident, woke=[v["task_id"] for v in fresh], run=woken)
        if busy or woken or any(v["live"] in OPEN for v in views):
            self._settling.pop(incident, None)
            return True
        # Nothing is open, but a specialist's host that just finished still reads "running" until its MicroVM pauses (a
        # minute after the run): read on, so the page shows the pause, not the last state seen, for a bounded time.
        if any(t["status"] in ("running", "creating", "resuming") for t in tiles.values()):
            return now - self._settling.setdefault(incident, now) < SETTLE_SECONDS
        self._settling.pop(incident, None)
        return False

    # -- approvers --------------------------------------------------------------------------------

    async def incidents(self) -> list[dict[str, Any]]:
        """The commander hosts (their MicroVM status) and what the relay last saw of each: commander status, pending
        approvals, latest run, the alert, its specialists' hosts and latest tool call; nothing is woken up."""
        handles = {h.host_id: h for h in await asyncio.to_thread(self.hosts.list_hosts, graph_id=GRAPH)}
        # Incidents still starting (or that failed to) have no host listed yet.
        early = [i for i, seen in self.seen.items() if seen.get("status") in ("starting", "error")]
        names = sorted({*handles, *(host_id_for(i) for i in early)})
        out = []
        for host_id in names:
            incident = host_id.removeprefix("incident-")
            if not (host_id.startswith("incident-") and _INCIDENT_ID.match(incident)):
                continue
            handle = handles.get(host_id)
            seen = self.seen.get(incident, {})
            story = self.stories.get(incident) or new_story()
            out.append({"incident": incident, "host_id": host_id,
                        "host": handle.status.value if handle else HostStatus.PROVISIONING.value, **seen,
                        "alarm": self.alarms.get(incident), "hosts": self.host_tiles.get(incident, {}),
                        "last_tool": self.last_tool.get(incident),
                        "progress": progress(story, seen.get("status"), seen.get("pending") or 0),
                        "verdict": (story["verdict"] or {}).get("verdict")})
        return out

    async def detail(self, incident: str) -> dict[str, Any]:
        """The incident as now: its tasks read again, the approvals its sub-agents wait on (with facts read here), the last
        message of the commander."""
        await self.refresh(incident)
        client = await self._client(incident)
        thread_id = thread_id_for(incident)
        try:
            state = await reread(lambda: client.threads.get_state(thread_id))
            runs = await reread(lambda: client.runs.list(thread_id, limit=1))  # the newest first
        finally:
            await client.http.client.aclose()
        pending = [dict(stop) for stop in self.pending.get(incident, [])]
        for stop in pending:
            value = stop["value"] if isinstance(stop["value"], dict) else {}
            target = await asyncio.to_thread(read_target, value, self.gateway) if self.gateway else None
            stop["facts"] = facts_of(target)
            stop["problems"] = self.restore_problems(incident, value, target)
        actions = self._actions.setdefault(incident, {"waiting": {}, "approved": {}})
        actions["waiting"] = {key: time.time() for stop in pending if (key := action_key(stop["value"]))}
        for stop in pending:
            stop["same_elsewhere"] = self.same_elsewhere(incident, stop["value"])
        messages = (state.get("values") or {}).get("messages") or []
        run = {"run_id": runs[0]["run_id"], "status": runs[0]["status"]} if runs else None
        seen = self.seen.get(incident, {})
        if seen.get("pending") or seen.get("open"):
            self.watch(incident)
        story = self.stories.setdefault(incident, new_story())
        return {"incident": incident, "thread_id": thread_id, "status": seen.get("status"), "pending": pending, "run": run,
                "last_message": text_of(messages[-1]) if messages else "", "tasks": self.tasks.get(incident, []),
                "story": public_story(story, seen.get("status"), len(pending))}

    def restore_problems(self, incident: str, value: dict[str, Any], target: dict[str, Any] | None) -> list[str]:
        """What stops a restore from being approved (approvals.problems_of): the relay's own facts about this incident, the
        droplet its service runs on and when its alert came, against what the Remediator was told to restore."""
        return problems_of(value, target, droplet_id=getattr(self.service, "droplet_id", None),
                           alerted_at=(self.alarms.get(incident) or {}).get("at"))

    def same_elsewhere(self, incident: str, value: Any) -> list[dict[str, Any]]:
        """Other open incidents with the same action (tool and arguments) waiting for a decision or approved through this
        relay, as far as it has seen them (it reads no other host for this): [{"incident", "state", "at"}]."""
        key = action_key(value)
        if key is None:
            return []
        found = []
        for other, actions in sorted(self._actions.items()):
            if other == incident:
                continue
            if key in actions["approved"]:
                found.append({"incident": other, "state": "approved", "at": actions["approved"][key]})
            elif key in actions["waiting"]:
                found.append({"incident": other, "state": "waiting", "at": actions["waiting"][key]})
        return found

    async def decide(self, incident: str, interrupt_id: str, decision: str, reason: str | None = None) -> str:
        """Settle an approval: the relay posts the decision to the gateway and, once approved, runs the action itself (the
        broker); then it resumes the sub-agent that waits on it (through the commander's gateway, which follows the new run),
        whose repeated request reads the outcome. Returns that run's id."""
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
        if interrupt_id in self._decided:
            raise NotPending("this approval is not waiting any more (decided, or the task moved on)")
        await self.refresh(incident)
        async with self._locks.setdefault(incident, asyncio.Lock()):  # two approvers at once: the second finds nothing
            stop = next((s for s in self.pending.get(incident, []) if s["id"] == interrupt_id), None)
            if stop is None or interrupt_id in self._decided:
                raise NotPending("this approval is not waiting any more (decided, or the task moved on)")
            value = stop["value"] if isinstance(stop["value"], dict) else {}
            if decision == "approve" and (action_key(value) is not None):  # a denial is always allowed
                target = await asyncio.to_thread(read_target, value, self.gateway) if self.gateway else None
                if problems := self.restore_problems(incident, value, target):
                    log_event(event="approver", incident=incident, decision="refused", task=stop["task"], problems=problems)
                    raise ValueError(f"approval refused: {'; '.join(problems)}. Deny it and start the incident's restore again.")
            self._decided.add(interrupt_id)
            # The decision is recorded before the relay acts on it, so the record reads in the order things happened.
            log_event(event="approver", incident=incident, decision=decision, task=stop["task"])
            if self.broker is not None and value.get("proposal") and decision in ("approve", "deny"):
                try:
                    await asyncio.to_thread(self.broker.decide, incident, str(value["proposal"]), decision, reason)
                except LookupError:  # a restarted relay never saw it: the resumed Remediator requests the action again
                    log_event(event="relay", incident=incident, rerequest=stop["task"])
                except BaseException as exc:
                    self._decided.discard(interrupt_id)
                    log_event(event="relay", incident=incident, decision_failed=stop["task"], error=str(exc)[:300])
                    raise
            answer = {"decision": decision, "reason": reason} if reason else decision
            client = await self._client(incident)
            try:
                response = await client.http.client.post("/incident/decision", json={
                    "thread": thread_id_for(incident), "task": stop["task"], "role": stop["role"],
                    "interrupt_id": interrupt_id, "answer": answer, "tool": value.get("tool")})
            finally:
                await client.http.client.aclose()
            if response.status_code >= 400:
                self._decided.discard(interrupt_id)
                raise NotPending(f"the sub-agent could not be resumed: {response.text[:300]}")
            run_id = str(response.json().get("run_id"))
        self.subruns.setdefault(incident, {})[run_id] = {"task": stop["task"], "role": stop["role"], "decision": decision}
        self.pending[incident] = [s for s in self.pending.get(incident, []) if s["id"] != interrupt_id]
        self.seen[incident] = {**self.seen.get(incident, {}), "pending": len(self.pending[incident])}
        self.stories.setdefault(incident, new_story())["decision"] = decision
        actions = self._actions.setdefault(incident, {"waiting": {}, "approved": {}})
        actions["waiting"] = {}
        if decision == "approve" and (key := action_key(value)):
            actions["approved"][key] = time.time()
        self.watch(incident)
        log_event(event="relay", incident=incident, resumed=stop["task"], run=run_id)
        return run_id

    async def message(self, incident: str, text: str) -> str:
        """Another turn on the commander's thread, when it is idle."""
        async with self._locks.setdefault(incident, asyncio.Lock()):
            client = await self._client(incident)
            thread_id = thread_id_for(incident)
            try:
                thread = await reread(lambda: client.threads.get(thread_id))
                if thread.get("status") == "busy" or await _active_run(client, thread_id):
                    raise NotPending("the commander is working: wait for it first")
                run_id = await start_run(client, thread_id, text=text)
            finally:
                await client.http.client.aclose()
        self._began(incident, run_id)
        return run_id

    async def _host_events(self, client: LangGraphClient, incident: str) -> tuple[list[dict[str, Any]], str | None]:
        """The incident's event log as the commander's host has it (the Remediator's gateway calls), every page; and a note when it
        could not be read (the postmortem says so rather than leave the calls out silently)."""
        found: list[dict[str, Any]] = []
        after = 0
        try:
            for _ in range(200):
                params = {"thread": thread_id_for(incident), "after": after}
                response = await reread(lambda: client.http.client.get("/incident/events", params=params))
                response.raise_for_status()
                body = response.json()
                found += [e for e in body.get("events") or [] if isinstance(e, dict)]
                if int(body.get("next", after)) <= after:
                    break
                after = int(body["next"])
        except Exception as exc:  # noqa: BLE001 - the relay's own record still makes a postmortem
            return found, f"The commander host's log could not be read ({type(exc).__name__}): the Remediator's gateway calls may be missing."
        return found, None

    async def _postmortem_text(self, incident: str, client: LangGraphClient) -> str:
        host_events, note = await self._host_events(client, incident)
        own = await asyncio.to_thread(postmortem.read_events, event_log.EVENTS, incident, thread_id_for(incident))
        return postmortem.render(incident=incident, alarm=self.alarms.get(incident), story=self.stories.get(incident) or new_story(),
                                 events=[*own, *host_events], notes=[note] if note else [])

    async def postmortem(self, incident: str) -> str:
        """The incident's postmortem as it stands: the recorded timeline and verdict, then the commander's narrative."""
        client = await self._client(incident)
        try:
            return await self._postmortem_text(incident, client)
        finally:
            await client.http.client.aclose()

    async def delete(self, incident: str) -> list[str]:
        """Close the incident: revoke its broker tokens, the commander's host deletes its specialists' hosts, then the relay
        deletes the commander's host. Refused while a commander run is in flight. Returns what was deleted."""
        async with self._locks.setdefault(incident, asyncio.Lock()):
            client = await self._client(incident)
            host_id = host_id_for(incident)
            try:
                if await _active_run(client, thread_id_for(incident)):
                    raise NotPending("a run is in flight: wait for it to end first")
                if self.postmortems is not None:  # before any host goes: the commander's host holds half of the record
                    text = await self._postmortem_text(incident, client)
                    path = await asyncio.to_thread(postmortem.write, self.postmortems, incident, text)
                    log_event(event="relay", incident=incident, postmortem=str(path))
                self._closed.add(incident)  # from here on the broker refuses the incident's tokens
                try:
                    response = await client.http.client.post("/incident/cleanup", params={"thread": thread_id_for(incident)},
                                                             timeout=CLEANUP_TIMEOUT)
                    response.raise_for_status()
                except BaseException:
                    self._closed.discard(incident)  # nothing was deleted: the incident goes on
                    raise
            finally:
                await client.http.client.aclose()
            deleted = list(response.json().get("deleted") or [])
            await asyncio.to_thread(self.hosts.delete, host_id=host_id)
            self.store.retire_host(host_id, "deleted")
            self.store.update_thread(thread_id_for(incident), run_status="cancelled")
            for running in (self._tasks.pop(incident, None), self._watchers.pop(incident, None)):
                if running is not None and running is not asyncio.current_task():
                    running.cancel()
            for kept in (self.seen, self._actions, self.alarms, self.host_tiles, self.last_tool, self.stories, self.tasks,
                         self.pending, self.subruns, self._notified):
                kept.pop(incident, None)
        if self.broker is not None:
            await asyncio.to_thread(self.broker.forget, incident)  # its Remediator endpoint and proposals
        swept = await asyncio.to_thread(self.sweep, [incident])
        log_event(event="relay", incident=incident, deleted=deleted, endpoints=swept)
        return [*deleted, host_id, *swept]

    def sweep(self, incidents: list[str]) -> list[str]:
        """Delete the Remediator's gateway endpoints of these closed incidents that are still listed: the broker deletes the
        incident's endpoint, but a killed relay never did. Returns the names deleted; a failure is logged, not raised (the hosts are gone)."""
        if self.endpoints is None or not incidents:
            return []
        prefixes = tuple(remediator_endpoint_prefix(i) for i in incidents)
        swept = []
        try:
            for endpoint in self.endpoints.list_endpoints(prefix=ENDPOINT_PREFIX):
                name = str(endpoint.get("name") or "")
                if name.startswith(prefixes) and endpoint.get("id"):
                    self.endpoints.delete_endpoint(str(endpoint["id"]))
                    swept.append(name)
        except Exception as exc:  # noqa: BLE001 - the incident is closed either way; say what was left
            log_event(event="relay", incidents=incidents, sweep="error", swept=swept, error=str(exc)[:300])
        return swept

    async def log(self, incident: str, after: int = 0) -> dict[str, Any]:
        """The incident's event log from the commander's host (the Remediator's gateway calls and the decisions)."""
        client = await self._client(incident)
        try:
            params = {"thread": thread_id_for(incident), "after": max(0, after)}
            response = await reread(lambda: client.http.client.get("/incident/events", params=params))
        finally:
            await client.http.client.aclose()
        response.raise_for_status()
        return response.json()

    async def service_health(self) -> dict[str, Any] | None:
        """The service's /health as the relay reads it (real mode), or None (local mode)."""
        return await asyncio.to_thread(self.service.health) if self.probe_service else None

    async def _subrun_events(self, incident: str, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """A run a decision started in a sub-agent's thread: shown as one step, then its end once the task is over."""
        sub = self.subruns[incident][run_id]
        yield {"type": "tool", "agent": sub["role"], "tool": "resumed after the decision", "input": {"decision": sub["decision"]}}
        view = None
        while True:
            await asyncio.sleep(self.watch_seconds)
            await self.refresh(incident)
            view = next((v for v in self.tasks.get(incident, []) if v["task_id"] == sub["task"]), None)
            if view is None or view["live"] not in OPEN:
                break
        yield {"type": "story", "story": public_story(self.stories[incident], "busy", len(self.pending.get(incident, [])))}
        yield {"type": "end", "status": view["live"] if view else "unknown", "pending": len(self.pending.get(incident, []))}

    async def events(self, incident: str, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """A run's tool calls as they happen ({"type": "tool", ...}), the story when it changes, then {"type": "end", ...}."""
        if run_id in self.subruns.get(incident, {}):
            async for event in self._subrun_events(incident, run_id):
                yield event
            return
        client = await self._client(incident)
        thread_id = thread_id_for(incident)
        last, drops = "-", 0
        try:
            while True:
                # A long stream through a host's endpoint can be dropped; the SDK reconnects only a few times per call,
                # so pick the stream up again after the last event seen until the run is over.
                try:
                    async for part in client.runs.join_stream(thread_id, run_id, last_event_id=last):
                        last = part.id or last
                        data = part.data if isinstance(part.data, dict) else {}
                        if part.event.startswith("events") and data.get("event") == "on_tool_start":
                            event = {"type": "tool",
                                     "agent": (data.get("metadata") or {}).get("lc_agent_name") or "incident-commander",
                                     "tool": data.get("name"), "input": (data.get("data") or {}).get("input")}
                            self.last_tool[incident] = {"agent": event["agent"], "tool": event["tool"], "at": time.time(),
                                                        "input": json.dumps(event["input"], default=str)[:160]}
                            yield event
                            given = event["input"] if isinstance(event["input"], dict) else {}
                            story = self.stories.setdefault(incident, new_story())
                            if (event["tool"] == "start_async_task" and given.get("subagent_type")
                                    and note_sent(story, str(data.get("run_id") or part.id or ""), str(given["subagent_type"]),
                                                  time.time())):
                                yield {"type": "story", "story": public_story(story, "busy", len(self.pending.get(incident, [])))}
                        elif (part.event.startswith("events") and data.get("event") == "on_tool_end"
                              and data.get("name") in ("check_async_task", "save_postmortem")):
                            await self.refresh(incident)
                            yield {"type": "story", "story": public_story(self.stories[incident], "busy",
                                                                          len(self.pending.get(incident, [])))}
                    break
                except httpx.HTTPError as exc:
                    drops += 1
                    if drops > MAX_STREAM_DROPS:
                        raise
                    if (await reread(lambda: client.runs.get(thread_id, run_id)))["status"] not in ("pending", "running"):
                        break
                    logger.info("run %s: event stream dropped (%s), picking it up again", run_id, exc)
                    await asyncio.sleep(STREAM_RETRY_SECONDS)
            await reread(lambda: client.runs.join(thread_id, run_id))
            run = await reread(lambda: client.runs.get(thread_id, run_id))
        finally:
            await client.http.client.aclose()
        await self.refresh(incident)
        self.watch(incident)
        pending = len(self.pending.get(incident, []))
        status = "interrupted" if pending else run["status"]
        yield {"type": "end", "status": status, "pending": pending}


def policy() -> dict[str, Any]:
    """What the page's Action Gateway tab shows: every role's actions in its tier, and which role holds each."""
    allow = sorted({a for role in ROLES for a in role.actions})
    return {"allow": allow, "ask": list(REMEDIATOR_ASK), "deny": list(REMEDIATOR_DENY),
            "roles": {role.name: [bare(a) for a in role.actions] for role in ROLES if role.actions}
            | {"remediator": [bare(a) for a in (*REMEDIATOR_ASK, *REMEDIATOR_DENY)]}}


def _security_headers() -> Any:
    """Every answer carries SECURITY_HEADERS: nothing it serves loads from elsewhere, and nothing frames it."""
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    class SecurityHeaders(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # noqa: ANN001, ANN201
            response = await call_next(request)
            response.headers.update(SECURITY_HEADERS)
            return response

    return Middleware(SecurityHeaders)


def make_app(relay: Relay, *, operator_token: str = "", page: str | None = None) -> Any:
    """The relay's routes and the approvals page (see the module docstring). Without an operator token /incidents answers 403."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    web = page_dir(page)

    async def alerts(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            started = await relay.handle(body, request.headers.get(SIGNATURE_HEADER))
        except SignatureError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except ValueError:
            return JSONResponse({"error": "the body is not JSON"}, status_code=400)
        return JSONResponse(started.__dict__, status_code=202)

    def refused(request: Request) -> JSONResponse | None:
        if not operator_token:
            return JSONResponse({"error": "approvals are off: set DO_INCIDENT_OPERATOR_TOKEN"}, status_code=403)
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), operator_token.encode()):
            return JSONResponse({"error": "operator token required"}, status_code=401)
        return None

    def guarded(handler):  # noqa: ANN001, ANN202 - a Starlette endpoint
        async def endpoint(request: Request):  # noqa: ANN202
            if (denied := refused(request)) is not None:
                return denied
            try:
                return await handler(request)
            except UnknownIncident:
                return JSONResponse({"error": "no such incident"}, status_code=404)
            except NotPending as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
        return endpoint

    async def body_of(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise ValueError("a JSON object is required")
        return body

    async def incidents(request: Request) -> JSONResponse:
        return JSONResponse({"incidents": await relay.incidents()})

    async def incident(request: Request) -> JSONResponse:
        return JSONResponse(await relay.detail(request.path_params["incident"]))

    async def decision(request: Request) -> JSONResponse:
        try:
            body = await body_of(request)
            run_id = await relay.decide(request.path_params["incident"], str(body.get("interrupt_id", "")),
                                        str(body.get("decision", "")), body.get("reason") or None)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"run_id": run_id}, status_code=202)

    async def messages(request: Request) -> JSONResponse:
        try:
            body = await body_of(request)
            if not str(body.get("text", "")).strip():
                raise ValueError("text is required")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"run_id": await relay.message(request.path_params["incident"], str(body["text"]))},
                            status_code=202)

    async def log(request: Request) -> JSONResponse:
        try:
            after = int(request.query_params.get("after", "0"))
        except ValueError:
            return JSONResponse({"error": "after must be a byte offset"}, status_code=400)
        return JSONResponse(await relay.log(request.path_params["incident"], after))

    async def service(request: Request) -> JSONResponse:
        return JSONResponse({"health": await relay.service_health()})

    async def policy_route(request: Request) -> JSONResponse:
        return JSONResponse(policy())

    async def close(request: Request) -> JSONResponse:
        return JSONResponse({"deleted": await relay.delete(request.path_params["incident"])})

    async def postmortem_route(request: Request) -> JSONResponse:
        return JSONResponse({"markdown": await relay.postmortem(request.path_params["incident"])})

    async def events(request: Request) -> StreamingResponse:
        stream = relay.events(request.path_params["incident"], request.path_params["run_id"])
        first = await anext(stream)  # an unknown incident fails here, before the 200

        async def sse() -> AsyncIterator[str]:
            yield f"data: {json.dumps(first, default=str)}\n\n"
            async for event in stream:
                yield f"data: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    async def index(request: Request) -> FileResponse:
        return FileResponse(web / "index.html")

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        if adopted := await relay.adopt():  # incidents a previous relay left open: watch them, keep their hosts awake
            logger.info("picked up incidents %s (started by a previous relay)", ", ".join(adopted))
        yield
        await relay.close()  # the background starts, keepalives and watchers; the hosts stay

    return Starlette(lifespan=lifespan, middleware=[_security_headers()], routes=[
        Route("/", index, methods=["GET"]),
        Mount("/static", StaticFiles(directory=web, html=False), name="static"),
        Route("/alerts", alerts, methods=["POST"]),
        Route("/incidents", guarded(incidents), methods=["GET"]),
        Route("/incidents/{incident}", guarded(incident), methods=["GET"]),
        Route("/incidents/{incident}", guarded(close), methods=["DELETE"]),
        Route("/incidents/{incident}/decision", guarded(decision), methods=["POST"]),
        Route("/incidents/{incident}/messages", guarded(messages), methods=["POST"]),
        Route("/incidents/{incident}/runs/{run_id}/events", guarded(events), methods=["GET"]),
        Route("/incidents/{incident}/log", guarded(log), methods=["GET"]),
        Route("/incidents/{incident}/postmortem", guarded(postmortem_route), methods=["GET"]),
        Route("/service", guarded(service), methods=["GET"]),
        Route("/policy", guarded(policy_route), methods=["GET"]),
    ])


def make_broker_app(broker: Broker) -> Any:
    """The broker's one route, on a port of its own (the only one the specialists' MicroVMs reach)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def actions(request: Request) -> JSONResponse:
        incident, role = request.path_params["incident"], request.path_params["role"]
        try:
            broker.check(incident, role, request.headers.get("authorization"))
            if int(request.headers.get("content-length") or 0) > BROKER_BODY_LIMIT:
                raise BrokerRefused(413, "the request is too large")
            raw = await request.body()
            if len(raw) > BROKER_BODY_LIMIT:
                raise BrokerRefused(413, "the request is too large")
            try:
                body = json.loads(raw or b"null")
            except ValueError:
                raise BrokerRefused(400, "the body is not JSON") from None
            answer = await asyncio.wait_for(asyncio.to_thread(broker.answer, incident, role, body), timeout=30)
        except BrokerRefused as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        except ValueError:
            return JSONResponse({"error": "content-length must be a number"}, status_code=400)
        except TimeoutError:
            return JSONResponse({"error": "the Action Gateway did not answer within 30 s"}, status_code=504)
        except Exception as exc:  # noqa: BLE001 - the gateway's error, for the specialist to report
            logger.warning("broker: %s %s failed", incident, role, exc_info=True)
            return JSONResponse({"error": f"the Action Gateway failed: {type(exc).__name__}: {str(exc)[:300]}"}, status_code=502)
        return JSONResponse(answer)

    return Starlette(middleware=[_security_headers()],
                     routes=[Route("/incidents/{incident}/actions/{role}", actions, methods=["POST"])])


def pins_for(service: Any) -> dict[str, dict[str, Any]]:
    """The arguments the broker fixes for the specialists' reads: the incident's repository and its droplet, nothing else
    the code-host connection or the account can see."""
    pins: dict[str, dict[str, Any]] = {}
    owner, _, repo = str(getattr(service, "repo", "")).partition("/")
    if owner and repo:
        pins["github_list_commits"] = pins["github_compare_commits"] = {"owner": owner, "repo": repo}
    if getattr(service, "droplet_id", None):
        pins["digitalocean_droplet-get"] = {"ID": service.droplet_id}
    return pins


READS = ["digitalocean_droplet-get", "digitalocean_image-list"]  # the relay's own reads: the approver's facts


def build(local: bool, broker_port: int | None = None) -> tuple[Relay, Any, Any, Any]:
    """The relay as `serve` and `console.py demo` run it: hosts, the store, the service (local: the simulated world, served to the
    hosts), a read-only gateway endpoint of its own for the approver's facts, and the broker (local mode: on 127.0.0.1 at
    `broker_port`, default DO_INCIDENT_BROKER_PORT or 8788, for the Remediator). Returns the relay, hosts, gateway and the
    world's server."""
    from langchain_digitalocean_microvm.integrations.action_gateway import (
        ActionGateway,
        ActionGatewayClient,
        ActionPolicy,
    )

    server = None
    if local:
        from incident_common.world import SimulatedWorld
        from incident_common.world_server import WorldEndpointsClient, serve

        from langchain_digitalocean_microvm.integrations.action_gateway_local import (
            LocalActionGatewayClient,
        )

        service: Any = SimulatedWorld()
        url, server = serve(service)
        # The hosts reach the one simulated world: its service, and the local gateway's actions.
        os.environ.update({"DO_INCIDENT_URL": url, "DO_INCIDENT_WORLD_URL": url, "DO_INCIDENT_DROPLET_ID": str(service.droplet_id),
                           "DO_INCIDENT_REPO": service.repo})
        client: Any = LocalActionGatewayClient(actions=service.actions())
        endpoints: Any = WorldEndpointsClient(url)  # the Remediators' simulated endpoints are recorded on the world server
        world_url = url
        port = broker_port or int(os.environ.get("DO_INCIDENT_BROKER_PORT", "8788"))
        broker_url = f"http://127.0.0.1:{port}"

        def broker_client() -> Any:  # the broker's endpoints, recorded on the world server like the real gateway's list
            return WorldEndpointsClient(world_url, actions=service.actions())
    else:
        from incident_common.world import make_service

        service = make_service(False)
        token = os.environ.get("DO_ACTION_GATEWAY_TOKEN")
        if not token:
            raise SystemExit("set DO_ACTION_GATEWAY_TOKEN: the relay holds the gateway for the approver's facts and the broker")
        url = os.environ.get("DO_INCIDENT_BROKER_URL", "")
        if not url.startswith("https://"):
            raise SystemExit("real mode: set DO_INCIDENT_BROKER_URL to the public https URL of the relay's broker port "
                             "(a tunnel to --broker-port; see README.md)")
        client = ActionGatewayClient(token)
        endpoints = ActionGatewayClient(token)
        broker_url = url

        def broker_client() -> Any:
            return ActionGatewayClient(token)
    gateway = ActionGateway(ActionPolicy(allow=READS), tools=READS, preload=READS, client=client)
    hosts = make_hosts(local)
    store = make_store(local)
    relay = Relay(hosts, os.environ.get("DO_INCIDENT_RELAY_SECRET", ""), service, spec=host_spec(local), store=store,
                  gateway=gateway, probe_service=not local, endpoints=endpoints,
                  postmortems=Path(os.environ.get("DO_INCIDENT_POSTMORTEMS") or HERE / "outputs" / "postmortems"))
    relay.broker = Broker(broker_key(), broker_url, broker_client, relay.is_open, pins=pins_for(service))
    return relay, hosts, gateway, server


def close_all(relay: Relay, hosts: Any, gateway: Any, server: Any) -> None:
    """Stop what this process started: local hosts are local processes; real hosts stay (their incidents go on)."""
    close = getattr(hosts, "close", None)
    if callable(close):
        close()
    gateway.close()
    if relay.broker is not None:
        relay.broker.close()
    if server is not None:
        server.should_exit = True


async def _serve(relay: Relay, *, host: str, port: int, broker_port: int, operator_token: str, page: str) -> None:
    import uvicorn

    app = make_app(relay, operator_token=operator_token, page=page)
    servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=port))]
    if relay.broker is not None:
        servers.append(uvicorn.Server(uvicorn.Config(make_broker_app(relay.broker), host="127.0.0.1", port=broker_port)))
    await asyncio.gather(*(s.serve() for s in servers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve_cmd = sub.add_parser("serve", help="serve the relay's routes, and the broker on a port of its own")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8787)
    serve_cmd.add_argument("--broker-port", type=int, default=int(os.environ.get("DO_INCIDENT_BROKER_PORT", "8788")),
                           help="the broker's port, on 127.0.0.1: point the tunnel at it (default DO_INCIDENT_BROKER_PORT or 8788)")
    serve_cmd.add_argument("--page", choices=pages(), help=f"the approvals page to serve (default: DO_INCIDENT_PAGE or {DEFAULT_PAGE})")
    args = parser.parse_args()

    token = os.environ.get("DO_INCIDENT_OPERATOR_TOKEN", "")
    if not token:
        raise SystemExit("set DO_INCIDENT_OPERATOR_TOKEN: approvers send it to decide")
    web = page_dir(args.page)
    relay, hosts, gateway, server = build(os.environ.get("DO_MICROVM_TRANSPORT") == "local", broker_port=args.broker_port)
    print(f"relay and approvals page {web.name}: http://{args.host}:{args.port}/ (it asks for DO_INCIDENT_OPERATOR_TOKEN)")
    if relay.broker is not None:
        print(f"broker: http://127.0.0.1:{args.broker_port}/ reached as {relay.broker.url}")
    try:
        asyncio.run(_serve(relay, host=args.host, port=args.port, broker_port=args.broker_port, operator_token=token,
                           page=web.name))
    finally:
        close_all(relay, hosts, gateway, server)


if __name__ == "__main__":
    main()
