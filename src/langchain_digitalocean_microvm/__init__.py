"""DigitalOcean MicroVM sandbox backend and agent hosts for deepagents.

langchain_digitalocean_microvm.client follows
the MicroVMs documentation as corrected by probing the real API
(scripts/probe_api.py); every DigitalOcean-specific detail stays in that one
module. See README.md for usage and environment variables, and
../../docs/design.md for the agent-host design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import (
    HostCapacityError,
    HostNotReadyError,
    MicroVMAlreadyExistsError,
    MicroVMAPIError,
    MicroVMClientError,
    MicroVMForbiddenError,
    MicroVMNotFoundError,
    MicroVMRateLimitError,
    SandboxError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.models import (
    Checkpoint,
    ExecResult,
    MicroVMSize,
    MicroVMState,
)
from langchain_digitalocean_microvm.provider import MicroVMLifecycle, SandboxTemplate
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.template_cache import TemplateCache
from langchain_digitalocean_microvm.threads import ThreadSandboxes

if TYPE_CHECKING:
    from langchain_digitalocean_microvm.agent_host import (
        HostHandle,
        HostSpec,
        HostStatus,
        MicroVMAgentHost,
    )

# Agent-host names load on first access: they need the `agent-host` extra.
_AGENT_HOST_NAMES = frozenset({"HostHandle", "HostSpec", "HostStatus", "MicroVMAgentHost"})


def __getattr__(name: str) -> Any:
    if name in _AGENT_HOST_NAMES:
        from langchain_digitalocean_microvm import agent_host

        return getattr(agent_host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MicroVMSandbox",
    "MicroVMConfig",
    "MicroVMLifecycle",
    "SandboxTemplate",
    "TemplateCache",
    "ThreadSandboxes",
    "MicroVMAgentHost",
    "HostHandle",
    "HostSpec",
    "HostStatus",
    "MicroVMState",
    "MicroVMSize",
    "MicroVMClient",
    "Checkpoint",
    "ExecResult",
    "MicroVMStatus",
    "MicroVMClientError",
    "MicroVMForbiddenError",
    "MicroVMRateLimitError",
    "MicroVMNotFoundError",
    "MicroVMAlreadyExistsError",
    "MicroVMAPIError",
    "SandboxError",
    "SandboxNotFoundError",
    "HostCapacityError",
    "HostNotReadyError",
]
