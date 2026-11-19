"""Action Gateway tools for a supervisor deep agent running in your own process.

An Action Gateway endpoint (`POST /v2/action-gateway/endpoints`: a named tool selection and policy, called on
the data plane its `mcp_url` names with `X-Endpoint-Id`) serves DigitalOcean and third-party actions to the
process that holds a token with `action_gateway:*` scopes. This module creates one, exposes it as LangChain
tools and maps the gateway's two stops to LangGraph interrupts:

- an `ask` rule answers `requires_approval`: the tool interrupts with
  `{"type": "action_approval", ...}`; the resume value decides (`"approve"`, `"deny"` or
  `{"decision": ..., "reason": ...}`), the decision goes to the gateway with the caller's token and
  the call is repeated once;
- a DigitalOcean action the actor has not connected answers with an OAuth connect link: the tool
  interrupts with `{"type": "connect_required", "url": ...}`; any resume value but `"cancel"`
  retries the call once.

The agent never holds the decision: only the code that resumes the graph does.

    policy = ActionPolicy(allow=["exa_web_search"], ask=["digitalocean_droplet-create"])
    with ActionGateway(policy=policy) as gw:  # the endpoint's tools: the policy's allow and ask names
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
from urllib.parse import urlsplit

import httpx
import yaml
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config
from langgraph.types import interrupt

from langchain_digitalocean_microvm.config import TRANSPORT_ENV, env_value

API_URL = "https://api.digitalocean.com/v2/action-gateway"
DATA_URL = "https://actions.do-ai.run"
# The data plane is the origin of the endpoint's returned `mcp_url`, accepted only on these domains.
DATA_DOMAINS = ("do-ai.run", "do-ai-test.run")
ENDPOINT_PREFIX = "microvm-ag-"
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
_LIST_PAGE = 100  # the most endpoints the service returns per page


class ActionGatewayError(Exception):
    """A request to the Action Gateway failed, or the gateway is not configured."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ActionPolicy:
    """Which actions the agent may call. Everything not listed is denied.

    Names are tool names without a version (`exa_web_search`, not `exa_web_search@14`); a version
    is dropped. `action_search` and `action_invoke` are always allowed, since the endpoint's policy
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

            allow: [github_list_commits, digitalocean_droplet-get]
            ask: [digitalocean_restore-droplet]
            deny: [digitalocean_droplet-delete]
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

    def create_endpoint(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        """Create an endpoint with one version (these tools and policy); return its id."""
        ...

    def delete_endpoint(self, endpoint_id: str) -> None: ...

    def search(self, endpoint_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        """The endpoint's tools matching the query: dicts with `name`, `description`, `inputSchema`."""
        ...

    def invoke(self, endpoint_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        """Return the call's `result` object (`status`, `output`, `error`, `_meta`)."""
        ...

    def decide(self, endpoint_id: str, path: str | None, approval_id: str, decision: str) -> None: ...

    def close(self) -> None: ...


def _data_origin(mcp_url: str) -> str:
    """The data-plane base URL of an endpoint: the origin of its `mcp_url`, on a DigitalOcean gateway domain only."""
    parts = urlsplit(mcp_url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not any(host == d or host.endswith(f".{d}") for d in DATA_DOMAINS):
        raise ActionGatewayError(f"Action Gateway returned an MCP URL outside {', '.join(DATA_DOMAINS)}: {mcp_url[:200]}")
    return f"https://{parts.netloc}"


class ActionGatewayClient:
    """REST client: endpoints on the control plane, tools and approvals on each endpoint's data plane.

    The data plane of an endpoint is the origin of the `mcp_url` the service returns for it (an `https` host on
    one of `DATA_DOMAINS`). `data_url` (default `DO_ACTION_GATEWAY_DATA_URL`) sends every endpoint to one host instead.
    """

    def __init__(
        self, token: str, *, api_url: str = API_URL, data_url: str | None = None, timeout: float = 60.0
    ) -> None:
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout
        self._api = httpx.Client(base_url=api_url, headers=self._headers, timeout=timeout)
        self._data_url = data_url or os.environ.get("DO_ACTION_GATEWAY_DATA_URL") or None
        self._new_plane = httpx.Client
        self._data: dict[str, httpx.Client] = {}  # by origin
        self._origins: dict[str, str] = {}  # endpoint id -> data-plane origin
        self._lock = threading.Lock()

    def close(self) -> None:
        self._api.close()
        with self._lock:
            for client in self._data.values():
                client.close()
            self._data.clear()

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

    def _remember(self, endpoint: Mapping[str, Any], mcp_url: Any = None) -> None:
        url = mcp_url or endpoint.get("mcp_url") or (endpoint.get("head_version") or {}).get("mcp_url")
        if isinstance(url, str) and url:
            origin = _data_origin(url)
            with self._lock:
                self._origins[str(endpoint["id"])] = origin

    def _plane(self, endpoint_id: str) -> httpx.Client:
        origin = self._data_url
        if origin is None:
            origin = self._origins.get(endpoint_id)
        if origin is None:  # an endpoint this client did not create: ask the control plane where it is served
            found = self._json(self._api.get(f"/endpoints/{endpoint_id}")).get("endpoint") or {}
            self._remember({**found, "id": endpoint_id})
            origin = self._origins.get(endpoint_id, DATA_URL)
        with self._lock:
            client = self._data.get(origin)
            if client is None:
                client = self._data[origin] = self._new_plane(base_url=origin, headers=self._headers, timeout=self._timeout)
            return client

    def create_endpoint(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        version: dict[str, Any] = {"tools": list(tools), "policy": dict(policy)}
        if actor_id:
            version["actor_id"] = actor_id
        payload = self._json(self._api.post("/endpoints", json={"name": name, "version": version}))
        endpoint = payload.get("endpoint") or {}
        if not endpoint.get("id"):
            raise ActionGatewayError("Action Gateway created an endpoint without an id")
        try:
            self._remember(endpoint, payload.get("mcp_url"))
        except ActionGatewayError:
            self.delete_endpoint(str(endpoint["id"]))
            raise
        return str(endpoint["id"])

    def delete_endpoint(self, endpoint_id: str) -> None:
        response = self._api.delete(f"/endpoints/{endpoint_id}")
        if response.status_code != 404:
            self._json(response)
        with self._lock:
            self._origins.pop(endpoint_id, None)

    def list_endpoints(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        """The endpoints this token can see (every page), as the service describes them (`id`, `name`, `actor_id`,
        `created_at`, ...); with `prefix`, only those whose name starts with it (the client reads every page and filters)."""
        endpoints: list[dict[str, Any]] = []
        token = ""
        while True:
            params: dict[str, Any] = {"page_size": _LIST_PAGE}
            if token:
                params["page_token"] = token
            payload = self._json(self._api.get("/endpoints", params=params))
            endpoints.extend(e for e in payload.get("endpoints") or [] if isinstance(e, dict))
            token = str(payload.get("next_page_token") or "")
            if not token:
                break
        if prefix is not None:
            endpoints = [e for e in endpoints if str(e.get("name") or "").startswith(prefix)]
        return endpoints

    def search(self, endpoint_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        payload = self._json(
            self._plane(endpoint_id).post(
                "/tools/search",
                json={"queries": [{"use_case": query}], "limit": limit},
                headers={"X-Endpoint-Id": endpoint_id},
            )
        )
        if payload.get("status") == "failed":
            raise ActionGatewayError(f"Action Gateway search failed: {str(payload.get('error'))[:300]}")
        groups = (payload.get("output") or {}).get("results") or []
        return [t for group in groups for t in group.get("results", []) if isinstance(t, dict)]

    def invoke(self, endpoint_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        response = self._plane(endpoint_id).post(
            "/tools/invoke",
            json={"rationale": rationale, "tools": [{"tool": tool, "arguments": dict(arguments)}]},
            headers={"X-Endpoint-Id": endpoint_id},
        )
        payload = self._json(response)
        try:
            result = payload["results"][0]["result"]
        except (KeyError, IndexError, TypeError):
            raise ActionGatewayError(f"Unexpected Action Gateway answer: {str(payload)[:300]}") from None
        return result if isinstance(result, dict) else {"output": result}

    def decide(self, endpoint_id: str, path: str | None, approval_id: str, decision: str) -> None:
        self._json(
            self._plane(endpoint_id).post(
                path or f"/approvals/{approval_id}",
                json={"decision": decision},
                headers={"X-Endpoint-Id": endpoint_id},
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


def _close_client(client: GatewayClient, endpoint_id: str | None) -> None:
    try:
        if endpoint_id:
            client.delete_endpoint(endpoint_id)
    finally:
        client.close()


class ActionGateway:
    """An Action Gateway endpoint of its own, as LangChain tools.

    `token` (default `DO_ACTION_GATEWAY_TOKEN`): a DigitalOcean token with `action_gateway:*`
    scopes; it creates and deletes the endpoint, invokes tools and decides approvals. `actor_id`
    (default `DO_ACTION_GATEWAY_ACTOR_ID`): the actor DigitalOcean actions run as: by default the part of your DigitalOcean
    email before the `@` (up to 64 letters, digits, `.`, `_` and `-`), else your user UUID
    (each actor connects its account once, through a link the gateway returns).

    `tools`: the endpoint's tool slugs (`exa_web_search`, sent as it is; `exa_web_search@14` pins one); default the
    names in `policy.allow` and `policy.ask`, as written. `preload`: slugs to expose as direct tools, each with its own
    input schema, besides the two meta tools (`action_search` finds the endpoint's tools by what they do,
    `action_invoke` calls one by name).

    The endpoint is created on first use (`tools()` creates it) and deleted by `close()`, the context
    manager or the garbage collector; a killed process leaves it behind. It is named `endpoint_prefix` plus
    8 random hex digits (default `microvm-ag-<hex>`): give each owner its own prefix and
    `ActionGatewayClient.list_endpoints(prefix=...)` finds what a killed process left.
    `data_url` (default `DO_ACTION_GATEWAY_DATA_URL`; otherwise the host of the endpoint's returned MCP URL) is the
    data plane.
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
        data_url: str | None = None,
        endpoint_prefix: str = ENDPOINT_PREFIX,
    ) -> None:
        if not endpoint_prefix:
            raise ValueError("endpoint_prefix must not be empty")
        self.endpoint_prefix = endpoint_prefix
        self.policy = policy
        self.tool_slugs = tuple(tools) if tools is not None else (*policy.allow, *policy.ask)
        if not self.tool_slugs:
            raise ValueError("no tools: pass `tools` or name some in the policy's `allow` or `ask`")
        self.preload = tuple(preload)
        missing = [s for s in self.preload if _bare(s) not in {_bare(t) for t in self.tool_slugs}]
        if missing:
            raise ValueError(f"preload names tools the endpoint does not have: {missing}")
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
        self._endpoint_id: str | None = None
        self._tools: list[BaseTool] | None = None
        # Approvals waiting for a decision, by call: a resumed graph runs the tool again from the
        # top, which must not ask the gateway for a second approval.
        self._pending: dict[str, dict[str, Any]] = {}
        self._finalizer = weakref.finalize(self, _close_client, client, None)

    # -- endpoint ----------------------------------------------------------------------------

    @property
    def endpoint_id(self) -> str | None:
        return self._endpoint_id

    def open(self) -> ActionGateway:
        with self._lock:
            if self._endpoint_id is None:
                name = f"{self.endpoint_prefix}{secrets.token_hex(4)}"
                self._endpoint_id = self._client.create_endpoint(name, self.tool_slugs, self.policy.to_spec(), self.actor_id)
                self._finalizer.detach()
                self._finalizer = weakref.finalize(self, _close_client, self._client, self._endpoint_id)
        return self

    def close(self) -> None:
        with self._lock:
            self._tools = None
            self._pending.clear()
            self._endpoint_id = None
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

    def _endpoint(self) -> str:
        self.open()
        assert self._endpoint_id is not None
        return self._endpoint_id

    def _invoke(self, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        return self._client.invoke(self._endpoint(), tool, arguments, rationale)

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
            self._client.decide(self._endpoint(), pending["path"], str(pending["approval_id"]), decision)
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
            self._client.decide(self._endpoint(), pending.path, pending.approval_id, decision)
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
        """The endpoint's tools that fit a plain-English description of what to do."""
        found = self._client.search(self._endpoint(), query, max(limit, 1))
        return json.dumps([{"name": _bare(str(t.get("name", ""))), "description": t.get("description", "")} for t in found])

    def tools(self, *, meta: bool = True) -> list[BaseTool]:
        """The LangChain tools: `action_search` and `action_invoke` (unless `meta=False`) and one per `preload` slug."""
        self.open()
        if self._tools is None:
            built: list[BaseTool] = [self._search_tool(), self._invoke_tool()]
            for slug in self.preload:
                name = _bare(slug)
                found = self._client.search(self._endpoint(), name.replace("_", " ").replace("-", " "), 10)
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
