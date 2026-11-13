"""Agent hosts: MicroVMs that run a LangGraph Agent Server for one subagent graph.

See docs/design.md. The package contains the real-MicroVM host manager (env at
create time, exec, HTTPS endpoints); the local mode LocalDevServerHost (a local
`langgraph dev` per host); the SubAgentGateway that gives a supervisor one
stable Agent Protocol URL in front of many hosts; the LazyGateway helper a
supervisor embeds (start on first use, teardown at exit); artifacts (per-task
outputs directories, gateway routes, fetch_artifacts_tool); and lifecycle
management (TTL deletion in the gateway's tick, the sweeper behind
scripts/gc_microvms.py).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Names resolve on first access, so importing a light submodule (`agent_host.store`,
# `agent_host.models`) does not load the gateway and its web-server dependencies,
# which come with the `agent-host` extra.
_LAZY = {
    "ArtifactError": "artifacts",
    "ArtifactInfo": "artifacts",
    "ArtifactListing": "artifacts",
    "GatewaySettings": "supervisor",
    "GatewayStartMiddleware": "supervisor",
    "HOST_MANAGED_TAG": "models",
    "HostHandle": "models",
    "HostSpec": "models",
    "HostStatus": "models",
    "LazyGateway": "supervisor",
    "LocalDevServerHost": "local",
    "MicroVMAgentHost": "host",
    "SubAgentGateway": "gateway",
    "SweepPlan": "sweeper",
    "apply_sweep": "sweeper",
    "async_subagent_spec": "spec",
    "build_host_env": "env",
    "default_probe": "host",
    "export_sdk_api_key": "supervisor",
    "fetch_artifacts_tool": "supervisor",
    "host_graph_tag": "models",
    "host_id_tag": "models",
    "plan_sweep": "sweeper",
    "render_env_file": "env",
}

if TYPE_CHECKING:
    from langchain_digitalocean_microvm.agent_host.artifacts import (
        ArtifactError,
        ArtifactInfo,
        ArtifactListing,
    )
    from langchain_digitalocean_microvm.agent_host.env import (
        build_host_env,
        render_env_file,
    )
    from langchain_digitalocean_microvm.agent_host.gateway import SubAgentGateway
    from langchain_digitalocean_microvm.agent_host.host import (
        MicroVMAgentHost,
        default_probe,
    )
    from langchain_digitalocean_microvm.agent_host.local import LocalDevServerHost
    from langchain_digitalocean_microvm.agent_host.models import (
        HOST_MANAGED_TAG,
        HostHandle,
        HostSpec,
        HostStatus,
        host_graph_tag,
        host_id_tag,
    )
    from langchain_digitalocean_microvm.agent_host.spec import async_subagent_spec
    from langchain_digitalocean_microvm.agent_host.supervisor import (
        GatewaySettings,
        GatewayStartMiddleware,
        LazyGateway,
        export_sdk_api_key,
        fetch_artifacts_tool,
    )
    from langchain_digitalocean_microvm.agent_host.sweeper import (
        SweepPlan,
        apply_sweep,
        plan_sweep,
    )


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


__all__ = [
    "HOST_MANAGED_TAG",
    "ArtifactError",
    "ArtifactInfo",
    "ArtifactListing",
    "GatewaySettings",
    "GatewayStartMiddleware",
    "HostHandle",
    "HostSpec",
    "HostStatus",
    "LazyGateway",
    "LocalDevServerHost",
    "MicroVMAgentHost",
    "SubAgentGateway",
    "SweepPlan",
    "apply_sweep",
    "async_subagent_spec",
    "build_host_env",
    "default_probe",
    "export_sdk_api_key",
    "fetch_artifacts_tool",
    "host_graph_tag",
    "host_id_tag",
    "plan_sweep",
    "render_env_file",
]
