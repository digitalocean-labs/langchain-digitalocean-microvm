"""Exception hierarchy for SandboxTransport implementations."""

from __future__ import annotations


class TransportError(Exception):
    """Base class for transport-level failures (network, auth, protocol)."""


class TransportFileNotFoundError(TransportError):
    """Remote path doesn't exist (download) or its parent doesn't (upload)."""


class TransportPermissionError(TransportError):
    """Access denied by the remote side."""


class TransportIsADirectoryError(TransportError):
    """The remote path is a directory where a file was expected."""
