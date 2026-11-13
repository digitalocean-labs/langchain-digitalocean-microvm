"""Data shapes for agent hosts: MicroVMs that run a LangGraph Agent Server.

An *agent host* is the "agent in sandbox" counterpart of MicroVMSandbox:
instead of the supervisor running a subagent's LLM loop locally and shipping
shell commands into the VM, the whole subagent (Agent Server + graph +
filesystem) lives inside the MicroVM and the supervisor reaches it over the
Agent Protocol. See docs/design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from langchain_digitalocean_microvm.status import MicroVMStatus

# Every agent host carries HOST_MANAGED_TAG (bulk scans, GC sweeps), one
# host_id_tag (exact lookup) and one host_graph_tag (which graph it serves).
# MicroVMClient.create additionally always adds client.MANAGED_TAG.
HOST_MANAGED_TAG = "microvm-agenthost"
_HOST_ID_TAG_PREFIX = "microvm-agenthost-id:"
_HOST_GRAPH_TAG_PREFIX = "microvm-agenthost-graph:"


def host_id_tag(host_id: str) -> str:
    """Tag used to look up the exact MicroVM backing one host_id."""
    return f"{_HOST_ID_TAG_PREFIX}{host_id}"


def host_graph_tag(graph_id: str) -> str:
    """Tag recording which graph (assistant id on the host) a host serves."""
    return f"{_HOST_GRAPH_TAG_PREFIX}{graph_id}"


def host_id_from_tags(tags: tuple[str, ...]) -> str | None:
    return _value_from_tags(tags, _HOST_ID_TAG_PREFIX)


def graph_id_from_tags(tags: tuple[str, ...]) -> str | None:
    return _value_from_tags(tags, _HOST_GRAPH_TAG_PREFIX)


def _value_from_tags(tags: tuple[str, ...], prefix: str) -> str | None:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


class HostStatus(StrEnum):
    """Agent-host status, layered over MicroVMStatus.

    The MicroVM may be RUNNING while the host is BOOTING (server not up yet)
    or BUSY (a run is executing); only the gateway knows BUSY, so a host
    manager reports READY for any RUNNING MicroVM whose /ok answers.
    RETIRING exists only in the gateway store: a process has decided to delete
    the host (GatewayStore.begin_retire_host) and nothing may resume it.
    """

    PROVISIONING = "provisioning"
    BOOTING = "booting"
    READY = "ready"
    BUSY = "busy"
    PAUSED = "paused"
    RESUMING = "resuming"
    RETIRING = "retiring"
    DESTROYED = "destroyed"
    FAILED = "failed"
    UNKNOWN = "unknown"


def host_status_from_microvm(status: MicroVMStatus) -> HostStatus:
    """Best-effort mapping for inventory listings (no /ok probe involved)."""
    mapping = {
        MicroVMStatus.CREATING: HostStatus.PROVISIONING,
        MicroVMStatus.RUNNING: HostStatus.READY,
        MicroVMStatus.PAUSING: HostStatus.PAUSED,
        MicroVMStatus.PAUSED: HostStatus.PAUSED,
        MicroVMStatus.RESUMING: HostStatus.RESUMING,
        MicroVMStatus.TERMINATING: HostStatus.DESTROYED,
        MicroVMStatus.TERMINATED: HostStatus.DESTROYED,
        MicroVMStatus.FAILED: HostStatus.FAILED,
    }
    return mapping.get(status, HostStatus.UNKNOWN)


@dataclass(frozen=True, slots=True)
class HostSpec:
    """What kind of agent host to run for one subagent type.

    `graph_id` is the graph name registered in the in-MicroVM langgraph.json
    (the `assistant_id` the gateway forwards). `image` and `size` (e.g.
    "2vcpu-4gb") override config.agent_image / config.agent_size. `env` is
    the host's environment on top of any config.subagent_env_passthrough
    variables; the host token and N_JOBS_PER_WORKER are always set last.
    `reuse`: one host per task, or one shared ("pooled") host per graph.
    """

    graph_id: str
    image: str | None = None
    size: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    reuse: Literal["per-task", "pooled"] = "per-task"
    n_jobs_per_worker: int = 2


@dataclass(frozen=True, slots=True)
class HostHandle:
    """A ready (or being-readied) agent host, as seen by the gateway.

    `token` is the per-host secret the Agent Server's auth handler expects
    in `x-api-key`; it never leaves the gateway. `booted_at` marks the last
    (re)start of the server process so run rows older than it can be
    rejected as orphans. An empty `token`/`booted_at` means this
    process has not bootstrapped or attached to the host yet.
    """

    host_id: str
    microvm_id: str
    graph_id: str
    url: str
    token: str
    status: HostStatus
    booted_at: str
