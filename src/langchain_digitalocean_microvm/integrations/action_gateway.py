"""Action Gateway tools for a supervisor deep agent running in your own process.

A standalone Action Gateway session (`POST /v2/action-gateway/sessions`, data plane
`https://actions.do-ai.run`) serves DigitalOcean and third-party actions to the process that holds
a token with `action_gateway:*` scopes. This module opens one, exposes it as LangChain tools and
maps the gateway's two stops to LangGraph interrupts:

- an `ask` rule answers `requires_approval`: the tool interrupts with
  `{"type": "action_approval", ...}`; the resume value decides (`"approve"`, `"deny"` or
  `{"decision": ..., "reason": ...}`), the decision goes to the gateway with the caller's token and
  the call is repeated once;
- a DigitalOcean action the actor has not connected answers with an OAuth connect link: the tool
  interrupts with `{"type": "connect_required", "url": ...}`; any resume value but `"cancel"`
  retries the call once.

The agent never holds the decision: only the code that resumes the graph does.

    policy = ActionPolicy(allow=["exa_web_search"], ask=["digitalocean_droplet-create"])
    with ActionGateway(policy=policy, tools=["exa_web_search@14", "digitalocean_droplet-create@3"]) as gw:
        agent = create_deep_agent(tools=gw.tools(), checkpointer=checkpointer)

The policy can also be a file of three lists (`ActionPolicy.from_yaml("policy.yaml")`).

Nothing here depends on the sandbox half of the package.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import threading
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any, Protocol

import httpx
import yaml
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config
from langgraph.types import interrupt

from langchain_digitalocean_microvm.config import TRANSPORT_ENV, env_value

API_URL = "https://api.digitalocean.com/v2/action-gateway"
DATA_URL = "https://actions.do-ai.run"
SESSION_PREFIX = "microvm-ag-"
DEFAULT_RATIONALE = "Requested by the agent"
CONNECT_LINK = re.compile(r"https://cloud\.digitalocean\.com/security/connectlinks/confirm[^\s\"'<>)\\]*")
META_TOOLS = ("action_search", "action_invoke")
_POLICY_TIERS = ("allow", "ask", "deny")
_POLICY_KEY_HINTS = {
    "default": "the default is always deny; list the tools to allow, ask or deny",
    "rules": "that is the shape of a session spec's `permissions`, not of an ActionPolicy; use the lists allow, ask and deny",
    "permissions": "that is the shape of a session spec, not of an ActionPolicy; use the lists allow, ask and deny",
}
# An approval that was already used, expired or never existed.
_APPROVAL_GONE = (404, 409, 410)
_LIST_PAGE = 100  # the most sessions the service returns per page


class ActionGatewayError(Exception):
    """A request to the Action Gateway failed, or the gateway is not configured."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ActionPolicy:
    """Which actions the agent may call. Everything not listed is denied.

    Names are tool names without a version (`exa_web_search`, not `exa_web_search@14`); a version
    is dropped. `action_search` and `action_invoke` are always allowed, since the session policy
    needs them for the tools to be reachable at all.
    """

    allow: Sequence[str] = ()
    ask: Sequence[str] = ()
    deny: Sequence[str] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "policy") -> ActionPolicy:
        """Build a policy from `{"allow": [...], "ask": [...], "deny": [...]}`, checking it first.

        Entries are kept as written, versions included. Raises `ValueError` (naming `source`) for an unknown
        key, a value that is not a list of tool names, a tool in more than one tier or twice in one, a name with
        a `do.actions/` prefix, or one of the always-allowed meta tools.
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"{source}: expected a mapping with the lists allow, ask and deny, got {type(data).__name__}")
        for key in data:
            if key not in _POLICY_TIERS:
                hint = _POLICY_KEY_HINTS.get(key, f"the keys are {', '.join(_POLICY_TIERS)}")
                raise ValueError(f"{source}: unknown key {key!r}: {hint}")
        tiers: dict[str, list[str]] = {}
        seen: dict[str, str] = {}
        for tier in _POLICY_TIERS:
            names = data.get(tier)
            if names is None:
                names = []
            if isinstance(names, str) or not isinstance(names, Sequence):
                raise ValueError(f"{source}: {tier!r} must be a list of tool names, got {type(names).__name__}")
            for name in names:
                if not isinstance(name, str) or not name.strip() or name != name.strip() or any(c.isspace() for c in name):
                    raise ValueError(f"{source}: {tier!r} entries are tool names without spaces, got {name!r}")
                bare = _bare(name)
                if not bare:
                    raise ValueError(f"{source}: {tier!r} entry {name!r} has no tool name before the version")
                if bare.startswith("do.actions/"):
                    raise ValueError(
                        f"{source}: {tier!r} entry {name!r}: tool names are plain (`digitalocean_droplet-get`), "
                        "not the `do.actions/` form of a session spec"
                    )
                if bare in META_TOOLS:
                    raise ValueError(f"{source}: {tier!r} entry {name!r}: {bare} is always allowed and cannot be listed")
                if bare in seen:
                    where = f"twice in {tier!r}" if seen[bare] == tier else f"in {seen[bare]!r} and {tier!r}"
                    raise ValueError(f"{source}: tool {bare!r} is listed {where}; a tool belongs to one tier, once")
                seen[bare] = tier
                tiers.setdefault(tier, []).append(name)
        return cls(allow=tiers.get("allow", ()), ask=tiers.get("ask", ()), deny=tiers.get("deny", ()))

    @classmethod
    def from_yaml(cls, source: str | os.PathLike[str] | IO[str]) -> ActionPolicy:
        """Load a policy from a YAML file (a path) or an open text stream, as `from_mapping` checks it.

        A `str` is a path, not YAML text, so a mistyped path raises `FileNotFoundError` and is never parsed as a
        document. Only plain YAML is read (`yaml.safe_load`).

            allow: [github_list_commits@5, digitalocean_droplet-get@3]
            ask: [digitalocean_restore-droplet@3]
            deny: [digitalocean_droplet-delete@3]
        """
        if isinstance(source, (str, os.PathLike)):
            label = os.fspath(source)
            with open(source, encoding="utf-8") as stream:
                text = stream.read()
        else:
            label = str(getattr(source, "name", "<stream>"))
            text = source.read()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"{label}: not valid YAML: {exc}") from exc
        if data is None:
            raise ValueError(f"{label}: empty; expected the lists allow, ask and deny")
        return cls.from_mapping(data, source=label)

    def to_spec(self) -> dict[str, Any]:
        rules = [{"tool": name, "action": "allow"} for name in META_TOOLS]
        for action, names in (("allow", self.allow), ("ask", self.ask), ("deny", self.deny)):
            rules.extend({"tool": _bare(name), "action": action} for name in names)
        return {"defaultAction": "deny", "rules": rules}


class GatewayClient(Protocol):
    """What ActionGateway needs from the service; `ActionGatewayClient` and the local one implement it."""

    def create_session(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        """Return the session's URN."""
        ...

    def delete_session(self, urn: str) -> None: ...

    def search(self, session_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        """The session's tools matching the query: dicts with `name`, `description`, `inputSchema`."""
        ...

    def invoke(self, session_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        """Return the call's `result` object (`status`, `output`, `error`, `_meta`)."""
        ...

    def decide(self, session_id: str, path: str | None, approval_id: str, decision: str) -> None: ...

    def close(self) -> None: ...


class ActionGatewayClient:
    """REST client: sessions on the control plane, tools and approvals on the data plane."""

    def __init__(
        self, token: str, *, api_url: str = API_URL, data_url: str = DATA_URL, timeout: float = 60.0
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"}
        self._api = httpx.Client(base_url=api_url, headers=headers, timeout=timeout)
        self._data = httpx.Client(base_url=data_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self._api.close()
        self._data.close()

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise ActionGatewayError(
                f"Action Gateway error {response.status_code}: {response.text[:300]}", response.status_code
            )
        try:
            return response.json()
        except ValueError:
            return {}

    def create_session(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        body: dict[str, Any] = {"name": name, "tools": list(tools), "policy": dict(policy)}
        if actor_id:
            body["actorId"] = actor_id
        payload = self._json(self._api.post("/sessions", json=body))
        session = payload.get("session", payload)
        urn = session.get("sessionUrn") or session.get("urn")
        if not urn:
            raise ActionGatewayError("Action Gateway created a session without a URN")
        return str(urn)

    def delete_session(self, urn: str) -> None:
        response = self._api.delete(f"/sessions/{urn}")
        if response.status_code != 404:
            self._json(response)

    def list_sessions(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        """The sessions this token can see (every page), as the service describes them (`sessionUrn`, `name`,
        `createdAt`, ...); with `prefix`, only those whose name starts with it (the service has no name filter)."""
        sessions: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._json(self._api.get("/sessions", params={"page": page, "per_page": _LIST_PAGE}))
            found = [s for s in payload.get("sessions") or [] if isinstance(s, dict)]
            sessions.extend(found)
            total = (payload.get("pagination") or {}).get("total")
            if not found or len(found) < _LIST_PAGE or (isinstance(total, int) and len(sessions) >= total):
                break
            page += 1
        if prefix is not None:
            sessions = [s for s in sessions if str(s.get("name") or "").startswith(prefix)]
        return sessions

    def search(self, session_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        payload = self._json(
            self._data.post(
                "/tools/search",
                json={"queries": [{"use_case": query}], "limit": limit},
                headers={"X-Session-Id": session_id},
            )
        )
        if payload.get("status") == "failed":
            raise ActionGatewayError(f"Action Gateway search failed: {str(payload.get('error'))[:300]}")
        groups = (payload.get("output") or {}).get("results") or []
        return [t for group in groups for t in group.get("results", []) if isinstance(t, dict)]

    def invoke(self, session_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        response = self._data.post(
            "/tools/invoke",
            json={"rationale": rationale, "tools": [{"tool": tool, "arguments": dict(arguments)}]},
            headers={"X-Session-Id": session_id},
        )
        payload = self._json(response)
        try:
            result = payload["results"][0]["result"]
        except (KeyError, IndexError, TypeError):
            raise ActionGatewayError(f"Unexpected Action Gateway answer: {str(payload)[:300]}") from None
        return result if isinstance(result, dict) else {"output": result}

    def decide(self, session_id: str, path: str | None, approval_id: str, decision: str) -> None:
        self._json(
            self._data.post(
                path or f"/approvals/session/{session_id}/{approval_id}",
                json={"decision": decision},
                headers={"X-Session-Id": session_id},
            )
        )


def _bare(name: str) -> str:
    return name.split("@", 1)[0]


def _key(tool: str, arguments: Mapping[str, Any]) -> str:
    """One call of one conversation thread (outside a graph: of no thread)."""
    try:
        thread = get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        thread = None
    return json.dumps([thread, tool, arguments], sort_keys=True, default=str)


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _connect_url(result: Mapping[str, Any]) -> str | None:
    """The OAuth connect link in a result, if the actor has no connection for the action."""
    for text in _walk_strings(result):
        match = CONNECT_LINK.search(text)
        if match:
            return match.group(0)
    return None


def _approval_of(result: Mapping[str, Any]) -> dict[str, Any] | None:
    meta = result.get("_meta") or {}
    if "requires_approval" not in (meta.get("status"), result.get("status")):
        return None
    decide = meta.get("decide")
    return {
        "approval_id": meta.get("approval_id") or meta.get("approvalId"),
        "path": decide.get("path") if isinstance(decide, Mapping) else None,
        "expires_at": next((meta[k] for k in ("expires_at", "expiresAt", "expiry", "expires") if k in meta), None),
    }


def _render(result: Mapping[str, Any]) -> str:
    error = result.get("error")
    if error:
        message = error.get("message") if isinstance(error, Mapping) else error
        return f"Error: {message}"
    output = result.get("output")
    if output is None:
        output = {k: v for k, v in result.items() if k != "_meta"}
    return output if isinstance(output, str) else json.dumps(output, default=str)


def _decision(resume: Any) -> tuple[str, str | None]:
    """`approve` or `deny` (anything unrecognised denies), with the optional reason."""
    reason = None
    if isinstance(resume, Mapping):
        reason = resume.get("reason")
        resume = resume.get("decision")
    approved = isinstance(resume, str) and resume.strip().lower() in ("approve", "approved")
    return ("approve" if approved else "deny"), reason


@dataclass(frozen=True)
class PendingApproval:
    """An approval the gateway asked for (`ActionGateway.request`); `ActionGateway.decide` settles it."""

    tool: str
    arguments: Mapping[str, Any]
    rationale: str
    approval_id: str
    path: str | None
    expires_at: Any = None


def _close_client(client: GatewayClient, urn: str | None) -> None:
    try:
        if urn:
            client.delete_session(urn)
    finally:
        client.close()


class ActionGateway:
    """A standalone Action Gateway session as LangChain tools.

    `token` (default `DO_ACTION_GATEWAY_TOKEN`): a DigitalOcean token with `action_gateway:*`
    scopes; it creates and deletes the session, invokes tools and decides approvals. `actor_id`
    (default `DO_ACTION_GATEWAY_ACTOR_ID`): the actor DigitalOcean actions run as: by default the part of your DigitalOcean
    email before the `@` (up to 64 letters, digits, `.`, `_` and `-`), else your user UUID
    (each actor connects its account once, through a link the gateway returns).

    `tools`: the session's tool slugs, with versions (`exa_web_search@14`); default the names in
    `policy.allow` and `policy.ask`. `preload`: slugs to expose as direct tools, each with its own
    input schema, besides the two meta tools (`action_search` finds the session's tools by what they do,
    `action_invoke` calls one by name).

    The session opens on first use (`tools()` opens it) and is deleted by `close()`, the context
    manager or the garbage collector; a killed process leaves it behind. It is named `session_prefix` plus
    8 random hex digits (default `microvm-ag-<hex>`): give each owner its own prefix and
    `ActionGatewayClient.list_sessions(prefix=...)` finds what a killed process left.
    With `DO_MICROVM_TRANSPORT=local` (or `client=`) no network is used.
    """

    def __init__(
        self,
        policy: ActionPolicy,
        *,
        tools: Sequence[str] | None = None,
        preload: Sequence[str] = (),
        token: str | None = None,
        actor_id: str | None = None,
        client: GatewayClient | None = None,
        api_url: str = API_URL,
        data_url: str = DATA_URL,
        session_prefix: str = SESSION_PREFIX,
    ) -> None:
        if not session_prefix:
            raise ValueError("session_prefix must not be empty")
        self.session_prefix = session_prefix
        self.policy = policy
        self.tool_slugs = tuple(tools) if tools is not None else (*policy.allow, *policy.ask)
        if not self.tool_slugs:
            raise ValueError("no tools: pass `tools` or name some in the policy's `allow` or `ask`")
        self.preload = tuple(preload)
        missing = [s for s in self.preload if _bare(s) not in {_bare(t) for t in self.tool_slugs}]
        if missing:
            raise ValueError(f"preload names tools the session does not have: {missing}")
        self.actor_id = actor_id or os.environ.get("DO_ACTION_GATEWAY_ACTOR_ID") or None
        if client is None:
            if env_value(TRANSPORT_ENV) == "local":
                from langchain_digitalocean_microvm.integrations.action_gateway_local import (
                    LocalActionGatewayClient,
                )

                client = LocalActionGatewayClient()
            else:
                token = token or os.environ.get("DO_ACTION_GATEWAY_TOKEN")
                if not token:
                    raise ActionGatewayError(
                        "no token: pass `token` or set DO_ACTION_GATEWAY_TOKEN (a token with action_gateway:* scopes)"
                    )
                client = ActionGatewayClient(token, api_url=api_url, data_url=data_url)
        self._client = client
        self._lock = threading.Lock()
        self._urn: str | None = None
        self._tools: list[BaseTool] | None = None
        # Approvals waiting for a decision, by call: a resumed graph runs the tool again from the
        # top, which must not ask the gateway for a second approval.
        self._pending: dict[str, dict[str, Any]] = {}
        self._finalizer = weakref.finalize(self, _close_client, client, None)

    # -- session -----------------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        return self._urn.rsplit(":", 1)[-1] if self._urn else None

    def open(self) -> ActionGateway:
        with self._lock:
            if self._urn is None:
                name = f"{self.session_prefix}{secrets.token_hex(4)}"
                self._urn = self._client.create_session(name, self.tool_slugs, self.policy.to_spec(), self.actor_id)
                self._finalizer.detach()
                self._finalizer = weakref.finalize(self, _close_client, self._client, self._urn)
        return self

    def close(self) -> None:
        with self._lock:
            self._tools = None
            self._pending.clear()
            self._urn = None
            self._finalizer()

    def __enter__(self) -> ActionGateway:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> ActionGateway:
        return await asyncio.to_thread(self.open)

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self.close)

    # -- calls -------------------------------------------------------------------------------

    def _session(self) -> str:
        self.open()
        assert self.session_id is not None
        return self.session_id

    def _invoke(self, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        return self._client.invoke(self._session(), tool, arguments, rationale)

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None, rationale: str = "") -> str:
        """Call one action; interrupts the graph where the gateway needs a decision or a connection."""
        tool, arguments, rationale = _bare(tool), dict(arguments or {}), rationale or DEFAULT_RATIONALE
        key = _key(tool, arguments)
        pending = self._pending.get(key)
        result = None
        if pending is None:
            result = self._invoke(tool, arguments, rationale)
            pending = _approval_of(result)
        if pending is not None:
            return self._decide_and_repeat(key, tool, arguments, rationale, pending)
        assert result is not None
        url = _connect_url(result)
        if url is None:
            return _render(result)
        # The first call ran nothing (the actor is not connected), so repeating it after a resume is safe.
        if interrupt({"type": "connect_required", "url": url, "tool": tool}) == "cancel":
            return f"Not connected: the user did not open the connect link {url}."
        result = self._invoke(tool, arguments, rationale)
        url = _connect_url(result)
        return _render(result) if url is None else f"Still not connected: the user needs to open {url}."

    def _decide_and_repeat(
        self, key: str, tool: str, arguments: dict[str, Any], rationale: str, pending: dict[str, Any]
    ) -> str:
        self._pending[key] = pending
        resume = interrupt(
            {
                "type": "action_approval",
                "tool": tool,
                "arguments": arguments,
                "rationale": rationale,
                "approval_id": pending["approval_id"],
                "expires_at": pending["expires_at"],
            }
        )
        self._pending.pop(key, None)
        decision, reason = _decision(resume)
        try:
            self._client.decide(self._session(), pending["path"], str(pending["approval_id"]), decision)
        except ActionGatewayError as exc:
            if exc.status_code in _APPROVAL_GONE:
                return "The approval expired or is no longer valid. Call the tool again to request a new one."
            raise
        if decision == "deny":
            return "The user denied this action." + (f" Reason: {reason}" if reason else "")
        result = self._invoke(tool, arguments, rationale)
        if _approval_of(result) is not None:
            return "Not run: the approval did not cover this call."
        # The gateway checks the approval before the connection, so an unconnected actor spends it
        # on a call that does not run. Once connected, the resumed tool asks for a new approval
        # and the decision already given answers it.
        url = _connect_url(result)
        if url is None:
            return _render(result)
        if interrupt({"type": "connect_required", "url": url, "tool": tool}) == "cancel":
            return f"Not connected: the user did not open the connect link {url}."
        return f"Still not connected: the user needs to open {url}."

    # -- calls without interrupts (a supervisor that keeps approvals out of the agent's process) --------------------------

    def request(self, tool: str, arguments: Mapping[str, Any] | None = None, rationale: str = "") -> str | PendingApproval:
        """Call one action without interrupting: its answer as `call` gives it, or the `PendingApproval` an `ask` rule asked
        for, to settle with `decide`. An actor that has not connected the action's account gets the connect link in the
        answer (`Not connected: ...`); nothing ran, so the same request can be sent again once it is connected.

        For code that holds the gateway token outside the agent's process: the agent proposes, this code requests, a human
        decides here and the agent never holds a credential that could approve its own call.
        """
        tool, arguments, rationale = _bare(tool), dict(arguments or {}), rationale or DEFAULT_RATIONALE
        result = self._invoke(tool, arguments, rationale)
        pending = _approval_of(result)
        if pending is not None:
            return PendingApproval(tool=tool, arguments=arguments, rationale=rationale, approval_id=str(pending["approval_id"]),
                                   path=pending["path"], expires_at=pending["expires_at"])
        url = _connect_url(result)
        if url is not None:
            return f"Not connected: the actor needs to open the connect link {url}, then the action can be requested again."
        return _render(result)

    def decide(self, pending: PendingApproval, decision: str, reason: str | None = None) -> str:
        """Send a decision (`approve`; anything else denies) on a `request`'s pending approval and, once approved, repeat
        the call: its answer, or the same texts as `call` for a denial, an expired approval or a missing connection."""
        decision, _ = _decision(decision)
        try:
            self._client.decide(self._session(), pending.path, pending.approval_id, decision)
        except ActionGatewayError as exc:
            if exc.status_code in _APPROVAL_GONE:
                return "The approval expired or is no longer valid. Call the tool again to request a new one."
            raise
        if decision == "deny":
            return "The user denied this action." + (f" Reason: {reason}" if reason else "")
        result = self._invoke(pending.tool, pending.arguments, pending.rationale)
        if _approval_of(result) is not None:
            return "Not run: the approval did not cover this call."
        url = _connect_url(result)
        return _render(result) if url is None else f"Not connected: the actor needs to open the connect link {url}."

    # -- tools -------------------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> str:
        """The session's tools that fit a plain-English description of what to do."""
        found = self._client.search(self._session(), query, max(limit, 1))
        return json.dumps([{"name": _bare(str(t.get("name", ""))), "description": t.get("description", "")} for t in found])

    def tools(self, *, meta: bool = True) -> list[BaseTool]:
        """The LangChain tools: `action_search` and `action_invoke` (unless `meta=False`) and one per `preload` slug."""
        self.open()
        if self._tools is None:
            built: list[BaseTool] = [self._search_tool(), self._invoke_tool()]
            for slug in self.preload:
                name = _bare(slug)
                found = self._client.search(self._session(), name.replace("_", " ").replace("-", " "), 10)
                built.append(self._direct_tool(name, next((t for t in found if _bare(str(t.get("name"))) == name), {})))
            self._tools = built
        return [t for t in self._tools if meta or t.name not in META_TOOLS]

    def _search_tool(self) -> BaseTool:
        def action_search(query: str, limit: int = 5) -> str:
            return self.search(query, limit)

        async def aaction_search(query: str, limit: int = 5) -> str:
            return await asyncio.to_thread(self.search, query, limit)

        return StructuredTool.from_function(
            func=action_search,
            coroutine=aaction_search,
            name="action_search",
            description="Find actions (DigitalOcean and third-party) from a plain-English description of what to do; returns their names and descriptions.",
        )

    def _invoke_tool(self) -> BaseTool:
        def action_invoke(tool: str, arguments: dict[str, Any] | None = None, rationale: str = "") -> str:
            return self.call(tool, arguments, rationale)

        async def aaction_invoke(tool: str, arguments: dict[str, Any] | None = None, rationale: str = "") -> str:
            return await asyncio.to_thread(self.call, tool, arguments, rationale)

        return StructuredTool.from_function(
            func=action_invoke,
            coroutine=aaction_invoke,
            name="action_invoke",
            description=(
                "Call an action by name (from action_search) with its arguments and a one-line rationale. "
                "Some actions wait for the user's approval or for the user to connect an account."
            ),
        )

    def _direct_tool(self, name: str, listed: Mapping[str, Any]) -> BaseTool:
        schema = next(
            (listed[k] for k in ("inputSchema", "input_schema", "parameters") if isinstance(listed.get(k), dict)),
            {"type": "object", "properties": {}, "additionalProperties": True},
        )

        def run(**arguments: Any) -> str:
            return self.call(name, arguments)

        async def arun(**arguments: Any) -> str:
            return await asyncio.to_thread(self.call, name, arguments)

        return StructuredTool(
            name=name,
            description=str(listed.get("description") or f"DigitalOcean Action Gateway action {name}."),
            args_schema={"title": name, **schema},
            func=run,
            coroutine=arun,
        )
