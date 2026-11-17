"""The incident team: one row per specialist, read by the commander (its async sub-agents), the specialists' host app (each
role's graph) and the relay (the broker's allow lists, the page's tiles). Each specialist runs in a MicroVM agent host of its
own, with its own model and its own Action Gateway actions.

Every specialist reaches the Action Gateway through the relay's broker (relay.py, `POST /incidents/{id}/actions/{role}`). The
relay holds the gateway token and every session; the specialist's host holds only a token for its own role in its own incident.
No gateway credential enters a specialist's MicroVM, and the policy is set by the relay, outside the agent's reach.
- `actions`: the read-only roles, one allow-only session per role.
- the Remediator: one session per incident whose policy asks a human before the restore and refuses a delete. Its broker token
  can request an action but never approve one: the relay asks the human and, once approved, runs the action itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SONNET, HAIKU = "anthropic-claude-4.6-sonnet", "anthropic-claude-haiku-4.5"


def inference_model(default: str) -> str:
    """The Serverless Inference model for a role whose default is `default` (SONNET: the strong slot, else the common one)."""
    variable = "DO_INFERENCE_STRONG_MODEL" if default == SONNET else "DO_INFERENCE_MODEL"
    return os.environ.get(variable) or default


REMEDIATOR_ASK = ("digitalocean_restore-droplet@3",)
REMEDIATOR_DENY = ("digitalocean_droplet-delete@3",)


@dataclass(frozen=True)
class Role:
    name: str  # the async sub-agent's name and its graph's id in specialists/langgraph.json
    title: str  # what the page and the README call it
    model: str  # default; DO_INCIDENT_MODEL_<NAME>, then DO_INFERENCE_STRONG_MODEL / DO_INFERENCE_MODEL, pick another
    description: str  # what the commander reads about it
    actions: tuple[str, ...] = field(default=())  # Action Gateway actions through the relay's broker (allow list)

    @property
    def model_id(self) -> str:
        return os.environ.get(f"DO_INCIDENT_MODEL_{self.name.upper()}") or inference_model(self.model)


ROLES: tuple[Role, ...] = (
    Role("metrics", "Metrics Analyst", HAIKU,
         "Reads the service's metrics in its own agent host. Ask when the errors started and how bad they got; it answers with "
         "the onset time, the peak, and which version came in when."),
    Role("changes", "Change Analyst", SONNET,
         "Reads the repository's commits in its own agent host (through its own Action Gateway actions). Give it the "
         "repository (owner/name) and, if known, the error onset; it answers with the change timeline, the commit just before "
         "the onset and what that commit changed.",
         ("github_list_commits@5", "github_compare_commits@4")),
    Role("infra", "Infra Analyst", HAIKU,
         "Reads the droplet and its snapshots in its own agent host (through its own Action Gateway actions). Give it the "
         "droplet id; it answers with the droplet's state and its snapshots (id, name, age).",
         ("digitalocean_droplet-get@3", "digitalocean_image-list@2")),
    Role("remediator", "Remediator", HAIKU,
         "The only agent that can change anything, in its own agent host. Give it the exact action and its arguments (for the "
         "restore: digitalocean_restore-droplet with ID and ImageID) and the reason, with the evidence in at most 3 sentences. "
         "The Action Gateway asks a human before a restore and refuses a delete; it answers with what the gateway answered."),
    Role("verifier", "Recovery Validator", HAIKU,
         "Checks, in its own agent host, whether the service is healthy again; it did not do the fix and only reads. Start it "
         "after the restore ran, with the droplet id: it samples the service's health and answers with a verdict, healthy or "
         "not_healthy, and the evidence.",
         ("digitalocean_droplet-get@3",)),
)
BY_NAME = {role.name: role for role in ROLES}
NAMES = tuple(role.name for role in ROLES)
REMEDIATOR = "remediator"
REMEDIATOR_TOOLS = (*REMEDIATOR_ASK, *REMEDIATOR_DENY)
BROKERED = (*(role.name for role in ROLES if role.actions), REMEDIATOR)  # the roles the relay's broker serves


def broker_token_name(role: str) -> str:
    """The commander host's variable holding `role`'s broker token (the relay sets one per role with actions)."""
    return f"DO_INCIDENT_BROKER_TOKEN_{role.upper()}"


def remediator_session_prefix(incident: str) -> str:
    """The name prefix of the Remediator's gateway session in one incident (the relay's broker opens it on the first request);
    closing the incident deletes it, and the relay sweeps by this prefix what a killed relay left."""
    return f"microvm-ag-{incident}-remediator-"


def bare(action: str) -> str:
    """An action's name without its version: what the specialist's tool is called and what the gateway's catalog lists."""
    return action.split("@", 1)[0]
