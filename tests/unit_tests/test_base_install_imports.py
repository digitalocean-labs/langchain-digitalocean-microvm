"""The base install (no `agent-host` extra) must import the core package.

Run in a subprocess whose import system refuses the extra's dependencies, so the
check holds even though the dev environment has every extra installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCKED = ("starlette", "uvicorn", "langgraph_api", "langgraph_runtime_inmem", "langgraph_cli")

SCRIPT = textwrap.dedent(
    """
    import importlib, importlib.abc, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {blocked!r}:
                raise ModuleNotFoundError(f"blocked: {{name}}", name=name)

    sys.meta_path.insert(0, Block())

    import langchain_digitalocean_microvm as pkg
    for mod in ("backend", "provider", "subagents", "threads", "template_cache", "transport",
                "client", "config", "sandbox_server", "reconcile", "agent_host.store",
                "agent_host.models", "agent_host.env", "integrations.action_gateway",
                "integrations.action_gateway_local"):
        importlib.import_module("langchain_digitalocean_microvm." + mod)
    assert pkg.MicroVMSandbox and pkg.ThreadSandboxes and pkg.MicroVMLifecycle
    assert pkg.HostSpec  # light agent-host names resolve without the extra
    from langchain_digitalocean_microvm.agent_host.store import GatewayStore, default_db_path
    assert GatewayStore and default_db_path("microvm").name == "gateway.sqlite"
    loaded = sorted(n for n in sys.modules if n.split(".")[0] in {blocked!r})
    assert not loaded, loaded
    """
).format(blocked=BLOCKED)


def test_core_package_imports_without_agent_host_extra() -> None:
    result = subprocess.run([sys.executable, "-I", "-c", SCRIPT], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_agent_host_names_still_resolve_lazily() -> None:
    import langchain_digitalocean_microvm as pkg
    from langchain_digitalocean_microvm import agent_host

    assert pkg.MicroVMAgentHost is agent_host.MicroVMAgentHost
    assert agent_host.LazyGateway.__name__ == "LazyGateway"
    assert "MicroVMAgentHost" in pkg.__all__


def test_default_db_path_is_shared_by_store_and_supervisor() -> None:
    from langchain_digitalocean_microvm.agent_host import store, supervisor

    assert supervisor.default_db_path is store.default_db_path
    assert store.default_db_path("local").name == "gateway-local.sqlite"
