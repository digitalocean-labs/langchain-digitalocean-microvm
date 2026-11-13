"""A specialist's Action Gateway actions, through the relay's broker (relay.py, `POST /incidents/{id}/actions/{role}`).

The relay holds the gateway token and one allow-only gateway session per role. This host holds only a token for its own role
in its own incident (DO_INCIDENT_BROKER_TOKEN, set by the commander's gateway when it created the host), valid while the
incident is open. The broker answers three requests, each a JSON object:

    {"op": "list"}                                         the role's actions: [{"name", "description", "input_schema"}]
    {"op": "search", "query", "limit"}                     the role's actions that fit a description
    {"op": "invoke", "tool", "arguments", "rationale"}     {"text": what the gateway answered}

`broker_tools(role)` lists the role's actions once and gives one LangChain tool per action, named after it and with its input
schema; each call is one `invoke`.

The Remediator's tools (`remediator_tools`) are `action_search` and `action_invoke` on its incident's session in the relay. An
`invoke` of an action the policy asks a human about runs nothing and answers {"approval": {...}}: the tool then pauses the run
with an `action_approval` interrupt. The relay shows it to a human, posts the decision to the gateway with its own token and,
once approved, runs the action; then it resumes the run, and the same request (the same tool call id) answers the outcome. The
Remediator's token can request an action but cannot approve one.

Environment: DO_INCIDENT_BROKER_URL (the relay's broker; https in real mode), DO_INCIDENT_ID, DO_INCIDENT_BROKER_TOKEN.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any

import httpx
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.types import interrupt

TIMEOUT = httpx.Timeout(60.0, connect=15.0)  # the broker waits up to 30 s for the gateway
RATIONALE_LIMIT = 500


class BrokerError(RuntimeError):
    """The broker refused the request (a bad or revoked token, an unknown role) or could not be reached."""


def settings(environ: dict[str, str] | None = None) -> tuple[str, str, str]:
    """(the role's broker endpoint without the role, the incident, the token) from the host's environment."""
    environ = dict(os.environ if environ is None else environ)
    missing = [n for n in ("DO_INCIDENT_BROKER_URL", "DO_INCIDENT_ID", "DO_INCIDENT_BROKER_TOKEN") if not environ.get(n)]
    if missing:
        raise BrokerError(f"{', '.join(missing)} not set in this host: the commander's gateway sets them when it creates it")
    url = environ["DO_INCIDENT_BROKER_URL"].rstrip("/")
    return f"{url}/incidents/{environ['DO_INCIDENT_ID']}/actions", environ["DO_INCIDENT_ID"], environ["DO_INCIDENT_BROKER_TOKEN"]


async def ask(role: str, body: dict[str, Any], *, environ: dict[str, str] | None = None,
              transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """One request to the broker for `role`; raises BrokerError on a refusal or a dropped connection."""
    endpoint, _incident, token = settings(environ)
    try:
        async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT,
                                     headers={"authorization": f"Bearer {token}"}) as http:
            response = await http.post(f"{endpoint}/{role}", json=body)
    except httpx.HTTPError as exc:
        raise BrokerError(f"the relay's broker could not be reached: {exc}") from exc
    try:
        answer = response.json()
    except ValueError:
        answer = {}
    if response.status_code >= 400 or not isinstance(answer, dict):
        error = answer.get("error") if isinstance(answer, dict) else None
        raise BrokerError(f"the relay's broker refused the request ({response.status_code}): {error or response.text[:300]}")
    return answer


def _tool(role: str, listed: dict[str, Any], *, environ: dict[str, str] | None,
          transport: httpx.AsyncBaseTransport | None) -> BaseTool:
    name = str(listed["name"])
    schema = listed.get("input_schema") if isinstance(listed.get("input_schema"), dict) else {}
    schema = {"type": "object", "properties": {}, **schema, "title": name}

    async def run(**arguments: Any) -> str:
        try:
            answer = await ask(role, {"op": "invoke", "tool": name, "arguments": arguments, "rationale": ""},
                               environ=environ, transport=transport)
        except BrokerError as exc:
            return f"Error: {exc}"
        return str(answer.get("text", ""))

    return StructuredTool(name=name, description=str(listed.get("description") or f"Action Gateway action {name}."),
                          args_schema=schema, coroutine=run)


async def broker_tools(role: str, names: set[str], *, environ: dict[str, str] | None = None,
                       transport: httpx.AsyncBaseTransport | None = None) -> list[BaseTool]:
    """The role's actions `names` (bare) as tools. A run without them fails here rather than answer without evidence."""
    listed = (await ask(role, {"op": "list"}, environ=environ, transport=transport)).get("actions") or []
    tools = [_tool(role, item, environ=environ, transport=transport) for item in listed
             if isinstance(item, dict) and item.get("name") in names]
    missing = names - {t.name for t in tools}
    if missing:
        raise BrokerError(f"the relay's broker does not offer {', '.join(sorted(missing))} to the {role} role")
    return tools


REMEDIATOR = "remediator"
MAX_ASKS = 3  # approvals one tool call may wait on: a relay restarted since it asked (it never saw the first) asks again


def remediator_tools(*, environ: dict[str, str] | None = None, transport: httpx.AsyncBaseTransport | None = None) -> list[BaseTool]:
    """`action_search` and `action_invoke` on the Remediator's session in the relay (its policy is the relay's)."""

    async def action_search(query: str, limit: int = 5) -> str:
        try:
            answer = await ask(REMEDIATOR, {"op": "search", "query": query, "limit": limit}, environ=environ, transport=transport)
        except BrokerError as exc:
            return f"Error: {exc}"
        return json.dumps(answer.get("actions") or [])

    async def action_invoke(tool: str, tool_call_id: Annotated[str, InjectedToolCallId], arguments: dict[str, Any] | None = None,
                            rationale: str = "") -> str:
        body = {"op": "invoke", "tool": tool, "arguments": arguments or {}, "rationale": rationale, "call": tool_call_id}
        try:
            answer = await ask(REMEDIATOR, body, environ=environ, transport=transport)
            for _ in range(MAX_ASKS):
                if "approval" not in answer:
                    break
                # The relay holds the approval: the run pauses until it has decided (and run the action, if approved).
                interrupt({"type": "action_approval", **answer["approval"]})
                answer = await ask(REMEDIATOR, body, environ=environ, transport=transport)
        except BrokerError as exc:
            return f"Error: {exc}"
        if "approval" in answer:
            return "Not run: the relay has no decision for this request."
        return str(answer.get("text", ""))

    return [
        StructuredTool.from_function(
            coroutine=action_search, name="action_search",
            description="Find actions from a plain-English description of what to do; returns their names and descriptions."),
        StructuredTool.from_function(
            coroutine=action_invoke, name="action_invoke",
            description=("Call an action by name (from action_search) with its arguments and a rationale of at most 3 sentences. "
                         "Some actions wait for a human's decision; some are refused by policy.")),
    ]
