"""Factory/registry for SandboxTransport implementations.

A simple name -> factory dict. No importlib.metadata entry-point discovery
yet — there are zero external transport consumers today, so that machinery
would be speculative. Adding entry-point discovery later (mirroring
deepagents_code's `deepagents_code.sandbox_providers` group) is a small,
additive change to create_transport() if a third-party transport package
ever appears; nothing else in this package would need to change.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_digitalocean_microvm.transport.base import SandboxTransport

TransportFactory = Callable[[], SandboxTransport]

_REGISTRY: dict[str, TransportFactory] = {}


def register_transport(name: str, factory: TransportFactory, *, overwrite: bool = False) -> None:
    if not overwrite and name in _REGISTRY:
        raise ValueError(f"Transport {name!r} is already registered")
    _REGISTRY[name] = factory


def create_transport(name: str) -> SandboxTransport:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ValueError(f"Unknown transport {name!r}; available: {available}") from None
    return factory()


def available_transports() -> list[str]:
    return sorted(_REGISTRY)
