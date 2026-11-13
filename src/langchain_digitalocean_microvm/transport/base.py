"""Pluggable command/file-exchange channel to one attached MicroVM instance.

MicroVMSandbox (backend.py) and MicroVMLifecycle (provider.py)
depend only on SandboxTransport, never on a concrete transport (exec for
real MicroVMs, local for runs on this machine), so adding one never touches those two
files. See transport/registry.py for how a concrete transport is
selected at runtime.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransportCredentials:
    """Secrets needed to authenticate a transport connection.

    ExecTransport reads `api_token` (needs the `microvm:update` scope, which
    covers exec). EndpointTransport reads both: `api_token` for exec and the
    sandbox token, `access_token` (scope `microvm:access` only) for the
    MicroVM's endpoint. LocalTransport needs nothing.
    """

    api_token: str | None = None
    access_token: str | None = None


@dataclass(frozen=True, slots=True)
class TransportExecResult:
    """Result of running one command over a transport.

    Deliberately NOT deepagents.backends.protocol.ExecuteResponse — a
    transport is a generic command/file channel, not a deepagents-specific
    type. backend.py does the (one-line) translation between the two.
    """

    stdout: str
    stderr: str
    exit_code: int | None


class SandboxTransport(ABC):
    """Minimal channel for talking to a running MicroVM."""

    # Whether a command may write to an absolute path with a plain shell
    # redirect and have it land in the sandbox. deepagents' execute wrapper
    # (capture-at-source offload) does exactly that with /large_tool_results/.
    # True for a real MicroVM, where `/` is the VM's; see LocalTransport.
    supports_capture_offload: bool = True

    @abstractmethod
    def connect(
        self,
        *,
        host: str,
        port: int,
        credentials: TransportCredentials,
        timeout: float | None = None,
    ) -> None:
        """Establish the connection. Safe to call again after close().

        Raises:
            transport.errors.TransportError: if the connection cannot be
                established within `timeout` seconds.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any held connection/session. Safe to call when not connected."""

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def run(self, command: str, *, timeout: float | None = None) -> TransportExecResult:
        """Run `command` and return its captured output.

        A failing *command* (non-zero exit code) is a normal result, not an
        exception — only raise transport.errors.TransportError for
        connectivity/protocol failures.
        """

    @abstractmethod
    def put_file(self, path: str, content: bytes) -> None:
        """Write `content` to `path` on the remote host.

        Raises:
            transport.errors.TransportFileNotFoundError: parent directory
                does not exist and could not be created.
            transport.errors.TransportPermissionError: access denied.
            transport.errors.TransportError: other transport-level failure.
        """

    @abstractmethod
    def get_file(self, path: str) -> bytes:
        """Read `path` from the remote host.

        Raises:
            transport.errors.TransportFileNotFoundError: path does not exist.
            transport.errors.TransportPermissionError: access denied.
            transport.errors.TransportError: other transport-level failure.
        """

    async def arun(self, command: str, *, timeout: float | None = None) -> TransportExecResult:
        return await asyncio.to_thread(self.run, command, timeout=timeout)

    async def aput_file(self, path: str, content: bytes) -> None:
        return await asyncio.to_thread(self.put_file, path, content)

    async def aget_file(self, path: str) -> bytes:
        return await asyncio.to_thread(self.get_file, path)
