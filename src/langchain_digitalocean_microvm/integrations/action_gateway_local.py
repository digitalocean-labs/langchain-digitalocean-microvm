"""In-memory Action Gateway for local mode and tests: no credentials, no network.

Follows the gateway's rules: the endpoint's policy decides `allow`, `ask` or `deny`; an
`ask` call answers `requires_approval` with an approval id and a decide path; an approval covers
one repeat of the same arguments, once; a denied call fails; an action that needs a connection
answers with a connect link until `connect()` is called.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGatewayError,
)

CONNECT_URL = "https://cloud.digitalocean.com/security/connectlinks/confirm?token=local"


@dataclass
class LocalAction:
    description: str = ""
    run: Callable[[Mapping[str, Any]], Any] = lambda arguments: {"echo": dict(arguments)}
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    needs_connection: bool = False


DEFAULT_ACTIONS = {
    "digitalocean_region-list": LocalAction(
        "List the regions.", lambda arguments: {"regions": ["nyc3", "sfo3", "ams3"]}, needs_connection=True
    ),
    "exa_web_search": LocalAction(
        "Search the web.",
        lambda arguments: {"results": [f"result for {arguments.get('query', '')}"]},
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
    "digitalocean_droplet-create": LocalAction(
        "Create a Droplet.",
        lambda arguments: {"created": dict(arguments)},
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    ),
}


class LocalActionGatewayClient:
    """Implements `GatewayClient`. `calls` records what ran; `expire(approval_id)` ends an approval."""

    def __init__(self, actions: Mapping[str, LocalAction] | None = None) -> None:
        self.actions = dict(DEFAULT_ACTIONS if actions is None else actions)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.invokes: list[tuple[str, dict[str, Any]]] = []
        self.endpoints: dict[str, dict[str, Any]] = {}
        self.decisions: list[tuple[str, str]] = []
        self._connected: set[str] = set()
        self._approvals: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.closed = False

    # test controls
    def connect(self, tool: str) -> None:
        self._connected.add(tool)

    def expire(self, approval_id: str) -> None:
        self._approvals[approval_id]["state"] = "expired"

    # GatewayClient
    def create_endpoint(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        endpoint_id = f"local-{next(self._ids)}"
        self.endpoints[endpoint_id] = {"name": name, "tools": [t.split("@")[0] for t in tools], "policy": dict(policy),
                                       "actor_id": actor_id, "created_at": time.time()}
        return endpoint_id

    def list_endpoints(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        """The endpoints as `ActionGatewayClient.list_endpoints` returns them (`id`, `name`, `actor_id`, `created_at`)."""
        return [{"id": endpoint_id, "name": e["name"], "actor_id": e["actor_id"],
                 "created_at": datetime.fromtimestamp(e["created_at"], UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
                for endpoint_id, e in list(self.endpoints.items()) if prefix is None or e["name"].startswith(prefix)]

    def delete_endpoint(self, endpoint_id: str) -> None:
        self.endpoints.pop(endpoint_id, None)

    def close(self) -> None:
        self.closed = True

    def _endpoint(self, endpoint_id: str) -> dict[str, Any]:
        endpoint = self.endpoints.get(endpoint_id)
        if endpoint is None:
            raise ActionGatewayError("endpoint not found", 404)
        return endpoint

    def search(self, endpoint_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        endpoint = self._endpoint(endpoint_id)
        words = query.lower().split()
        found = [
            {"name": name, "description": self.actions[name].description, "inputSchema": self.actions[name].input_schema}
            for name in endpoint["tools"]
            if name in self.actions and all(w in f"{name} {self.actions[name].description}".lower() for w in words)
        ]
        return found[:limit]

    @staticmethod
    def _rule(policy: Mapping[str, Any], tool: str) -> str:
        for rule in policy.get("rules", []):
            if rule["tool"] == tool:
                return str(rule["action"])
        return str(policy.get("defaultAction", "deny"))

    def invoke(self, endpoint_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        endpoint = self._endpoint(endpoint_id)
        arguments = dict(arguments)
        with self._lock:
            self.invokes.append((tool, arguments))
            if tool not in endpoint["tools"] or tool not in self.actions:
                return {"status": "error", "error": {"message": f"unknown tool {tool}"}}
            action = self._rule(endpoint["policy"], tool)
            if action == "deny":
                return {"status": "error", "error": {"message": f"{tool} is denied by the endpoint's policy"}}
            if action == "ask":
                key = json.dumps([tool, arguments], sort_keys=True)
                approval = next((a for a in self._approvals.values() if a["key"] == key and a["state"] != "used"), None)
                if approval is None or approval["state"] == "expired":
                    approval_id = f"apr-{next(self._ids)}"
                    approval = self._approvals[approval_id] = {"key": key, "state": "pending", "id": approval_id}
                if approval["state"] == "pending":
                    return {
                        "status": "requires_approval",
                        "_meta": {
                            "status": "requires_approval",
                            "approval_id": approval["id"],
                            "expires_at": "in 30 minutes",
                            "decide": {"path": f"/approvals/endpoint/{endpoint_id}/{approval['id']}"},
                        },
                    }
                if approval["state"] == "denied":
                    approval["state"] = "used"
                    return {"status": "error", "error": {"message": "denied by the user"}}
                approval["state"] = "used"
            spec = self.actions[tool]
            if spec.needs_connection and tool not in self._connected:
                return {"status": "requires_authorization", "output": f"Connect your account to use {tool}: {CONNECT_URL}"}
            self.calls.append((tool, arguments))
            return {"status": "success", "output": spec.run(arguments)}

    def decide(self, endpoint_id: str, path: str | None, approval_id: str, decision: str) -> None:
        self._endpoint(endpoint_id)
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval["state"] != "pending":
                raise ActionGatewayError("approval expired or unknown", 410)
            approval["state"] = "approved" if decision == "approve" else "denied"
            self.decisions.append((approval_id, decision))
