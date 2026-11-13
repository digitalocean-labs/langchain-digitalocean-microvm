"""The Action Gateway session of the incident commander: which actions it may call, and the audit of every call.

One policy, three tiers (local mode and real mode alike):

    allow  reads: the repository's commits, the droplet, its snapshots
    ask    restoring the droplet from a snapshot
    deny   deleting the droplet

The tiers live in policy.yaml. `ask` ends the run with an `action_approval` interrupt that the console answers (run_script.py); the decision is posted
with this process's own token, never by an agent. `AuditedClient` writes every call and decision to the event log,
where the dashboard lists them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from incident_common.events import log_event

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGateway,
    ActionGatewayClient,
    ActionPolicy,
    GatewayClient,
)
from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    LocalActionGatewayClient,
)

# policy.yaml holds the three tiers, versions included, so the same lists name the session's tools.
POLICY = ActionPolicy.from_yaml(Path(__file__).resolve().parent / "policy.yaml")
READS, ASKED, DENIED = list(POLICY.allow), list(POLICY.ask), list(POLICY.deny)


def _bare(name: str) -> str:
    return name.split("@", 1)[0]


class AuditedClient:
    """A `GatewayClient` that logs each call (allow / ask / deny / approved / error) and each decision."""

    def __init__(self, inner: GatewayClient) -> None:
        self.inner = inner
        self._approval_tool: dict[str, str] = {}
        self._approved: set[str] = set()

    def create_session(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        return self.inner.create_session(name, tools, policy, actor_id)

    def delete_session(self, urn: str) -> None:
        self.inner.delete_session(urn)

    def search(self, session_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        return self.inner.search(session_id, query, limit)

    def close(self) -> None:
        self.inner.close()

    def invoke(self, session_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        result = self.inner.invoke(session_id, tool, arguments, rationale)
        meta = result.get("_meta") or {}
        message = str((result.get("error") or {}).get("message", "") if isinstance(result.get("error"), Mapping) else result.get("error", ""))
        if "requires_approval" in (result.get("status"), meta.get("status")):
            status = "ask"
            self._approval_tool[str(meta.get("approval_id") or meta.get("approvalId"))] = tool
        elif "denied by the user" in message:
            status = "denied"
        elif message and ("polic" in message.lower() or "denied" in message.lower()):
            status = "deny"
        elif message:
            status = "error"
        elif tool in self._approved:
            status = "approved"
            self._approved.discard(tool)
        else:
            status = "allow"
        log_event(event="gateway", tool=_bare(tool), status=status, arguments=dict(arguments), rationale=rationale[:300],
                  error=message[:300] if status in ("error", "deny", "denied") else "")
        return result

    def decide(self, session_id: str, path: str | None, approval_id: str, decision: str) -> None:
        self.inner.decide(session_id, path, approval_id, decision)
        tool = self._approval_tool.get(approval_id, "?")
        if decision == "approve":
            self._approved.add(tool)
        log_event(event="decision", tool=_bare(tool), decision=decision)


def make_gateway(world: Any, *, local: bool) -> ActionGateway:
    """The incident commander's gateway session (opened on first use, deleted when the process ends)."""
    if local:
        client: GatewayClient = LocalActionGatewayClient(actions=world.actions())
    else:
        token = os.environ.get("DO_ACTION_GATEWAY_TOKEN")
        if not token:
            raise RuntimeError("DO_ACTION_GATEWAY_TOKEN is not set (a token with action_gateway:* scopes), or DO_MICROVM_TRANSPORT=local")
        client = ActionGatewayClient(token)
    return ActionGateway(POLICY, tools=[*READS, *ASKED, *DENIED], preload=READS, client=AuditedClient(client))
