"""The commander host's routes for the relay, served by its Agent Server (`"http"` in langgraph.json, behind the host token).

    GET  /incident/tasks?thread=<commander thread>&tasks=<task id>:<role>,...
            each task as its own host has it, read through the commander's gateway: its live status (the async sub-agent
            middleware's words: pending, running, waiting, success, error, cancelled), the interrupts it waits on, its answer
            once done, and its host's state. The Remediator's gateway calls are written to the event log once each.
    POST /incident/decision  {"thread", "task", "role", "interrupt_id", "answer"}
            resumes the sub-agent's thread with a human's answer (the approval waits there, not in the commander's thread)
    POST /incident/cleanup   deletes every specialist host the gateway created
    GET  /incident/events?thread=<id>&after=<offset>
            the incident's event log: the gateway audit and the decisions, and the offset to ask from next time

Every route checks the host token itself as well (auth.py of the agent-host image checks it first). Each answer is short: a
request through a MicroVM's endpoint has to be answered within about 15 seconds.
"""

from __future__ import annotations

import hmac
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

HERE = Path(__file__).resolve().parent
EXAMPLE, EXAMPLES = HERE.parent, HERE.parents[1]
for path in (EXAMPLE, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from incident_common.events import EVENTS, log_event  # noqa: E402
from roles import NAMES, REMEDIATOR_ASK, bare  # noqa: E402

AUTH = HERE.parents[2] / "images" / "agent-host" / "auth.py"  # langgraph.json's auth handler
LIMIT = 500  # log lines per answer
RESULT_LIMIT = 8000  # characters of a task's answer
ASKED = {bare(a) for a in REMEDIATOR_ASK}
_logged: set[tuple[str, str]] = set()  # (task, call id) of the gateway calls already in the log
_logged_lock = threading.Lock()


# -- what a task's thread says ---------------------------------------------------------------------------------------------------

def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return "" if content is None else str(content)


def live_status(run: dict[str, Any] | None, thread: dict[str, Any] | None) -> str:
    """As the async sub-agent middleware reports a task: a paused run ends `success`, only its thread shows the wait."""
    if not run:
        return "pending"
    if run.get("status") == "interrupted":
        return "cancelled"
    if run.get("status") != "success" or not thread:
        return str(run.get("status"))
    return {"interrupted": "waiting", "busy": "running", "error": "error"}.get(str(thread.get("status")), "success")


def interrupts_of(thread: dict[str, Any] | None) -> list[dict[str, Any]]:
    found = (thread or {}).get("interrupts") or {}
    return [{"id": i["id"], "value": i.get("value")} for items in found.values() for i in items or []]


def gateway_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Remediator's `action_invoke` calls with what the gateway answered: allow, approved, denied (a human), deny (policy), error."""
    asked: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("type") == "ai":
            for call in message.get("tool_calls") or []:
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                if call.get("name") == "action_invoke" and args.get("tool"):
                    asked[call["id"]] = {"call": call["id"], "tool": bare(str(args["tool"])), "arguments": args.get("arguments")}
    calls = []
    for message in messages:
        call = asked.get(message.get("tool_call_id")) if message.get("type") == "tool" else None
        if call is None:
            continue
        text = content_text(message.get("content"))
        if text.startswith("The user denied"):
            status = "denied"
        elif text.startswith("Error:") and "polic" in text.lower():
            status = "deny"
        elif text.startswith(("Error:", "Not run", "Not connected", "Still not connected", "The approval expired")):
            status = "error"
        else:
            status = "approved" if call["tool"] in ASKED else "allow"
        calls.append({**call, "status": status, "error": text[:300] if status in ("deny", "denied", "error") else ""})
    return calls


def task_view(lazy: Any, task: str, role: str) -> dict[str, Any]:
    """One task as its host has it, read through the gateway (which answers from its cache once the run is over)."""
    view: dict[str, Any] = {"task_id": task, "role": role, "live": "pending", "interrupts": [], "result": None, "host": None,
                            "calls": []}
    store = lazy.gateway.store if lazy.gateway is not None else None
    row = store.get_thread(task) if store is not None else None
    if row is None or not row.run_id:
        return view
    headers = {"x-api-key": lazy.token}
    with httpx.Client(base_url=lazy.url, headers=headers, timeout=30) as http:
        run = http.get(f"/threads/{task}/runs/{row.run_id}")
        thread = http.get(f"/threads/{task}")
    run_body = run.json() if run.status_code == 200 else None
    thread_body = thread.json() if thread.status_code == 200 else None
    view["live"] = live_status(run_body, thread_body)
    view["run_id"] = row.run_id
    view["interrupts"] = interrupts_of(thread_body) if view["live"] == "waiting" else []
    messages = ((thread_body or {}).get("values") or {}).get("messages") or []
    if view["live"] in ("success", "error") and messages:
        view["result"] = content_text(messages[-1].get("content"))[:RESULT_LIMIT]
    if role == "remediator":
        view["calls"] = gateway_calls(messages)
    host = store.get_host(row.host_id) if row.host_id else None
    if host is not None:
        view["host"] = {"host_id": host.host_id, "microvm_id": host.microvm_id, "status": host.status,
                        "since": host.idle_since or host.last_active_at}
    return view


def log_calls(thread: str, view: dict[str, Any]) -> None:
    """The Remediator's gateway calls (and the approval it waits on) into the event log, once each."""
    entries = [(c["call"], {"event": "gateway", "tool": c["tool"], "status": c["status"], "arguments": c["arguments"],
                            "error": c["error"]}) for c in view["calls"]]
    for stop in view["interrupts"]:
        value = stop["value"] if isinstance(stop["value"], dict) else {}
        if value.get("type") == "action_approval":
            entries.append((f"ask:{stop['id']}", {"event": "gateway", "tool": bare(str(value.get("tool"))), "status": "ask",
                                                  "arguments": value.get("arguments"), "error": ""}))
    for key, event in entries:
        with _logged_lock:
            if (view["task_id"], key) in _logged:
                continue
            _logged.add((view["task_id"], key))
        log_event(thread=thread, task=view["task_id"], **event)


def tasks(lazy: Any, thread: str, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    views = []
    for task, role in pairs:
        view = task_view(lazy, task, role)
        log_calls(thread, view)
        views.append(view)
    return views


def resume(lazy: Any, task: str, role: str, interrupt_id: str, answer: Any) -> dict[str, Any]:
    """A run on the sub-agent's thread that answers its interrupt; through the gateway, so it follows the new run."""
    if lazy.gateway is None:
        raise LookupError("no sub-agent has started yet")
    with httpx.Client(base_url=lazy.url, headers={"x-api-key": lazy.token}, timeout=60) as http:
        response = http.post(f"/threads/{task}/runs", json={"assistant_id": role, "command": {"resume": {interrupt_id: answer}}})
    if response.status_code >= 400:
        raise LookupError(f"the gateway refused the resume ({response.status_code}): {response.text[:300]}")
    return response.json()


# -- the event log -----------------------------------------------------------------------------------------------------------------

def read_events(path: Path, thread: str, after: int, limit: int = LIMIT) -> dict[str, Any]:
    """The incident's events after byte `after` (from the start if the log is shorter: it was replaced), and the next offset."""
    if not path.exists():
        return {"events": [], "next": 0}
    with path.open("rb") as f:
        if after > path.stat().st_size:
            after = 0
        f.seek(after)
        events, offset = [], after
        for _ in range(limit):
            line = f.readline()
            if not line.endswith(b"\n"):
                break
            offset += len(line)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("thread") == thread:
                events.append(event)
    return {"events": events, "next": offset}


# -- the app ---------------------------------------------------------------------------------------------------------------------

def _expected_token():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("incident_host_auth", AUTH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.expected_token


def _lazy() -> Any:
    import agent  # the commander's module (graph.py imported it): its gateway

    return agent.gateway


def _cleanup() -> list[str]:
    import agent

    return agent.cleanup()


def create_app(*, events: Path = EVENTS, expected_token=None, lazy=_lazy, cleanup=_cleanup) -> Starlette:  # noqa: ANN001
    token_source = expected_token or _expected_token()

    async def authorized(request: Request) -> bool:
        expected = await token_source()
        return bool(expected) and hmac.compare_digest(request.headers.get("x-api-key", ""), expected)

    def guarded(handler):  # noqa: ANN001, ANN202
        async def endpoint(request: Request) -> JSONResponse:
            if not await authorized(request):
                return JSONResponse({"error": "invalid or missing host token"}, status_code=401)
            try:
                return await handler(request)
            except (KeyError, ValueError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except LookupError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            except Exception as exc:  # noqa: BLE001 - the relay shows what went wrong in the host
                return JSONResponse({"error": f"{type(exc).__name__}: {str(exc)[:500]}"}, status_code=500)
        return endpoint

    async def task_list(request: Request) -> JSONResponse:
        thread = request.query_params["thread"]
        pairs = []
        for item in filter(None, request.query_params.get("tasks", "").split(",")):
            task, _, role = item.partition(":")
            if role not in NAMES:
                raise ValueError(f"unknown role {role!r}")
            pairs.append((task, role))
        return JSONResponse({"tasks": await run_in_threadpool(tasks, lazy(), thread, pairs)})

    async def decision(request: Request) -> JSONResponse:
        body = await request.json()
        if not isinstance(body, dict) or body.get("role") not in NAMES:
            raise ValueError("thread, task, role, interrupt_id and answer are required")
        run = await run_in_threadpool(resume, lazy(), str(body["task"]), str(body["role"]), str(body["interrupt_id"]),
                                      body.get("answer"))
        answer = body.get("answer")
        await run_in_threadpool(lambda: log_event(  # no file I/O on the server's event loop
            thread=str(body["thread"]), task=str(body["task"]), event="decision", tool=bare(str(body.get("tool") or "?")),
            decision=answer.get("decision") if isinstance(answer, dict) else answer))
        return JSONResponse({"run_id": run.get("run_id"), "task": body["task"]})

    async def clean(request: Request) -> JSONResponse:
        return JSONResponse({"deleted": await run_in_threadpool(cleanup)})

    async def log(request: Request) -> JSONResponse:
        after = max(0, int(request.query_params.get("after", "0")))
        return JSONResponse(await run_in_threadpool(read_events, events, request.query_params["thread"], after))

    return Starlette(routes=[Route("/incident/tasks", guarded(task_list), methods=["GET"]),
                             Route("/incident/decision", guarded(decision), methods=["POST"]),
                             Route("/incident/cleanup", guarded(clean), methods=["POST"]),
                             Route("/incident/events", guarded(log), methods=["GET"])])


app = create_app()
