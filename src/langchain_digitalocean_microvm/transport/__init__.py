"""Pluggable transports for MicroVMSandbox: endpoint (default) and exec (real MicroVMs), local."""

from __future__ import annotations

from langchain_digitalocean_microvm.transport.base import (
    SandboxTransport,
    TransportCredentials,
    TransportExecResult,
)
from langchain_digitalocean_microvm.transport.errors import (
    TransportError,
    TransportFileNotFoundError,
    TransportPermissionError,
)
from langchain_digitalocean_microvm.transport.registry import (
    available_transports,
    create_transport,
    register_transport,
)


def _register_builtins() -> None:
    from langchain_digitalocean_microvm.transport.endpoint import EndpointTransport
    from langchain_digitalocean_microvm.transport.exec import ExecTransport
    from langchain_digitalocean_microvm.transport.local import LocalTransport

    register_transport("exec", ExecTransport, overwrite=True)
    register_transport("endpoint", EndpointTransport, overwrite=True)
    register_transport("local", LocalTransport, overwrite=True)


_register_builtins()

__all__ = [
    "SandboxTransport",
    "TransportCredentials",
    "TransportExecResult",
    "TransportError",
    "TransportFileNotFoundError",
    "TransportPermissionError",
    "available_transports",
    "create_transport",
    "register_transport",
]
