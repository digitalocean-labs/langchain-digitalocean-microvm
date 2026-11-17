"""The Action Gateway endpoint of the erasure coordinator: which actions it may call, and the audit of every call.

One policy, three tiers, written out by hand (the catalog's own hints are not used for tiers: it marks
`intercom_search_contacts` as not read-only though it only reads, and `mailchimp_forget_contact` as not destructive
though it cannot be undone):

    allow  reads: search and get, in every system
    ask    each erasure (CRM, support, marketing, analytics) and the billing redaction: one human decision per record
    deny   deleting the billing customer (its invoices must stay), GDPR-deleting CRM files

The tiers live in policy.yaml. `ask` ends the run with an `action_approval` interrupt that the console answers (run_script.py); the decision is posted
with this process's own token, never by an agent. `AuditedClient` writes every call and decision to the event log and
keeps them for the evidence report.

The five systems are mocks (world.py) and the gateway is the package's local gateway: this example has no real mode for them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from events import log_event

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGateway,
    ActionPolicy,
    GatewayClient,
)
from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    LocalActionGatewayClient,
)

# policy.yaml holds the three tiers; the same lists name the endpoint's tools.
POLICY = ActionPolicy.from_yaml(Path(__file__).resolve().parent / "policy.yaml")
READS, ASKED, DENIED = list(POLICY.allow), list(POLICY.ask), list(POLICY.deny)

# What each changing action acts on: (system, kind, how to read the record id from its arguments).
TARGETS: dict[str, tuple[str, str, Any]] = {
    "hubspot_crm_contacts_gdpr_delete_create": ("crm", "contact", lambda a: a.get("object_id")),
    "intercom_delete_a_contact": ("support", "contact", lambda a: a.get("contact_id")),
    "stripe_post_customers_customer": ("billing", "customer", lambda a: a.get("customer")),
    "stripe_delete_customers_customer": ("billing", "customer", lambda a: a.get("customer")),
    "mailchimp_forget_contact": ("marketing", "member", lambda a: a.get("contact_id")),
    "mixpanel_delete_profile": ("analytics", "profile", lambda a: next(iter(a.get("updates") or [{}]), {}).get("$distinct_id")),
    "hubspot_files_gdpr_delete_by_fileid": ("crm", "file", lambda a: a.get("file_id")),
}


def _bare(name: str) -> str:
    return name.split("@", 1)[0]


def target_of(tool: str, arguments: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """(system, kind, record id) of a changing action's call; None for a read."""
    found = TARGETS.get(_bare(tool))
    if found is None:
        return None
    system, kind, read = found
    try:
        return system, kind, str(read(arguments))
    except (AttributeError, TypeError):
        return system, kind, "?"


class AuditedClient:
    """A `GatewayClient` that logs each call (allow / ask / deny / denied / approved / error) and each decision.

    `audit` keeps them in memory too, for the evidence report: tool, status and the record a changing call targets;
    no arguments and no rationale (they could carry what the report must not).
    """

    def __init__(self, inner: GatewayClient) -> None:
        self.inner = inner
        self.audit: list[dict[str, Any]] = []
        self._approval_tool: dict[str, tuple[str, Any]] = {}
        self._approved: set[str] = set()

    def create_endpoint(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        return self.inner.create_endpoint(name, tools, policy, actor_id)

    def delete_endpoint(self, endpoint_id: str) -> None:
        self.inner.delete_endpoint(endpoint_id)

    def search(self, endpoint_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        return self.inner.search(endpoint_id, query, limit)

    def close(self) -> None:
        self.inner.close()

    def invoke(self, endpoint_id: str, tool: str, arguments: Mapping[str, Any], rationale: str) -> dict[str, Any]:
        result = self.inner.invoke(endpoint_id, tool, arguments, rationale)
        meta = result.get("_meta") or {}
        message = str((result.get("error") or {}).get("message", "") if isinstance(result.get("error"), Mapping) else result.get("error", ""))
        target = target_of(tool, arguments)
        if "requires_approval" in (result.get("status"), meta.get("status")):
            status = "ask"
            self._approval_tool[str(meta.get("approval_id") or meta.get("approvalId"))] = (tool, target)
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
                  error=message[:300] if status in ("error", "deny", "denied") else "", target=target)
        self.audit.append({"event": "gateway", "tool": _bare(tool), "status": status, "target": target})
        return result

    def decide(self, endpoint_id: str, path: str | None, approval_id: str, decision: str) -> None:
        self.inner.decide(endpoint_id, path, approval_id, decision)
        tool, target = self._approval_tool.get(approval_id, ("?", None))
        if decision == "approve":
            self._approved.add(tool)
        log_event(event="decision", tool=_bare(tool), decision=decision, target=target)
        self.audit.append({"event": "decision", "tool": _bare(tool), "decision": decision, "target": target})


def make_gateway(systems: Any) -> tuple[ActionGateway, AuditedClient]:
    """The coordinator's gateway endpoint over the mock systems (created on first use), and its audited client."""
    client = AuditedClient(LocalActionGatewayClient(actions=systems.actions()))
    return ActionGateway(POLICY, tools=[*READS, *ASKED, *DENIED], client=client), client
