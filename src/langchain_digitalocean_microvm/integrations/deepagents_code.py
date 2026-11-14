"""`MicroVMSandboxProvider`: MicroVMs as a deepagents_code (`dcode`) sandbox provider.

Registered under the `deepagents_code.sandbox_providers` entry-point group as
`microvm` (pyproject.toml), so with this package installed next to
deepagents-code, `dcode --sandbox microvm` works. Only dcode imports this
module; the rest of langchain_digitalocean_microvm never needs deepagents-code (see docs/design.md).

The provider wraps `MicroVMLifecycle` (DO_MICROVM_* settings, built on first
use so the registry can read `metadata` without credentials):

- `get_or_create(sandbox_id=None)` creates a sandbox under a fresh id
  (`dcode-<hex>`); with an id it reuses, resumes or creates that sandbox.
  dcode reports a sandbox by `backend.id`, which is the MicroVM's id, and
  later deletes it by that id, so either id is accepted everywhere.
- Extra keyword arguments: `seed_files` (uploaded on create only) and `tags`.
  Anything else is refused rather than silently ignored.
"""

from __future__ import annotations

import secrets
from typing import Any

try:
    from deepagents_code.integrations.sandbox_provider import (  # pyright: ignore[reportMissingImports]
        SandboxError,
        SandboxNotFoundError,
        SandboxProvider,
        SandboxProviderMetadata,
    )
except ImportError as exc:  # pragma: no cover - exercised only without deepagents-code
    raise ImportError(
        "langchain_digitalocean_microvm.integrations.deepagents_code needs the deepagents-code package"
    ) from exc

from langchain_digitalocean_microvm import exceptions as _mvx
from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.provider import MicroVMLifecycle

PROVIDER_NAME = "microvm"
_KNOWN_KWARGS = frozenset({"seed_files", "tags"})


class MicroVMSandboxProvider(SandboxProvider):
    """dcode sandbox provider backed by DigitalOcean MicroVMs."""

    def __init__(self, lifecycle: MicroVMLifecycle | None = None, *, id_prefix: str = "dcode") -> None:
        self._lifecycle = lifecycle
        self._id_prefix = id_prefix
        self._by_backend_id: dict[str, str] = {}  # MicroVMSandbox.id -> sandbox_id, this process

    @property
    def metadata(self) -> SandboxProviderMetadata:
        return SandboxProviderMetadata(
            name=PROVIDER_NAME,
            working_dir="/",
            supports_sandbox_id=True,
            supports_snapshot_name=False,
            backend_module="langchain_digitalocean_microvm",
        )

    @property
    def lifecycle(self) -> MicroVMLifecycle:
        if self._lifecycle is None:
            try:
                self._lifecycle = MicroVMLifecycle()
            except _mvx.SandboxError as exc:
                raise SandboxError(str(exc)) from exc
        return self._lifecycle

    def get_or_create(self, *, sandbox_id: str | None = None, **kwargs: Any) -> MicroVMSandbox:
        unknown = sorted(set(kwargs) - _KNOWN_KWARGS)
        if unknown:
            raise ValueError(f"MicroVM sandbox provider does not support {unknown}; known: {sorted(_KNOWN_KWARGS)}")
        resolved = self._resolve(sandbox_id) if sandbox_id else f"{self._id_prefix}-{secrets.token_hex(6)}"
        try:
            backend = self.lifecycle.get_or_create(
                sandbox_id=resolved, seed_files=kwargs.get("seed_files"), tags=tuple(kwargs.get("tags") or ()),
            )
        except _mvx.SandboxNotFoundError as exc:
            raise SandboxNotFoundError(str(exc)) from exc
        except (_mvx.SandboxError, _mvx.MicroVMClientError) as exc:
            raise SandboxError(str(exc)) from exc
        self._by_backend_id[backend.id] = resolved
        return backend

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        if kwargs:
            raise ValueError(f"MicroVM sandbox provider delete() takes no options, got {sorted(kwargs)}")
        resolved = self._resolve(sandbox_id)
        try:
            self.lifecycle.delete(sandbox_id=resolved)
        except _mvx.SandboxNotFoundError as exc:
            raise SandboxNotFoundError(sandbox_id) from exc
        except (_mvx.SandboxError, _mvx.MicroVMClientError) as exc:
            raise SandboxError(str(exc)) from exc
        for backend_id, sid in list(self._by_backend_id.items()):
            if sid == resolved:
                del self._by_backend_id[backend_id]

    def _resolve(self, sandbox_id: str) -> str:
        """A sandbox_id, given either one or a MicroVM id (what dcode prints)."""
        if sandbox_id in self._by_backend_id:
            return self._by_backend_id[sandbox_id]
        try:
            return self.lifecycle.sandbox_id_for(sandbox_id) or sandbox_id
        except _mvx.MicroVMClientError as exc:
            raise SandboxError(str(exc)) from exc


__all__ = ["PROVIDER_NAME", "MicroVMSandboxProvider"]
