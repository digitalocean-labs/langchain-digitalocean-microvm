"""Exception hierarchy for the MicroVM client, lifecycle manager, and agent hosts.

Two independent hierarchies, matching the two layers of this package:

- MicroVMClientError: raised by MicroVMClient for REST-level
  failures (client.py).
- SandboxError / SandboxNotFoundError: raised by MicroVMLifecycle
  (provider.py) and MicroVMAgentHost (agent_host/host.py), shaped to
  structurally match deepagents_code.integrations.sandbox_provider's
  SandboxError / SandboxNotFoundError (same `.original_exc` property, no
  import of that package) so wrapping MicroVMLifecycle as a real
  SandboxProvider later is a thin adapter, not a rewrite.
"""

from __future__ import annotations


class MicroVMClientError(Exception):
    """Base class for MicroVMClient failures."""


class MicroVMNotFoundError(MicroVMClientError):
    """No MicroVM or checkpoint found for the given id (server returned 404)."""


class MicroVMAlreadyExistsError(MicroVMClientError):
    """Create collided with an existing resource (server returned 409)."""


class MicroVMAPIError(MicroVMClientError):
    """Any other non-2xx response from the MicroVM API."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"MicroVM API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class MicroVMForbiddenError(MicroVMAPIError):
    """HTTP 403: the token lacks a `microvm:*` scope, or (on create) the team reached its MicroVM limit.

    The limit counts running AND paused MicroVMs; GET /v2/microvms/options
    reports it as `account_limits.max_total_count`.
    """


class MicroVMRateLimitError(MicroVMAPIError):
    """HTTP 429 that persisted through the client's retries (e.g. the team's exec limit: 60/min, 32 concurrent)."""


class SandboxError(Exception):
    """Base class for MicroVMLifecycle / MicroVMAgentHost failures."""

    @property
    def original_exc(self) -> BaseException | None:
        return self.__cause__


class SandboxNotFoundError(SandboxError):
    """No MicroVM is registered under the given sandbox_id / host_id."""


class HostNotReadyError(SandboxError):
    """The MicroVM is RUNNING but its Agent Server never answered GET /ok in time."""


class HostCapacityError(SandboxError):
    """A new agent host cannot be created because the team is at its MicroVM limit.

    The limit counts running and paused MicroVMs (GET /v2/microvms/options,
    `account_limits.max_total_count` / `max_memory_bytes`).
    """
