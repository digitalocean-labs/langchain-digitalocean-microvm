"""Lifecycle reconciliation shared by the sandbox and agent-host managers.

Both MicroVMLifecycle (provider.py, "sandbox as a tool") and
MicroVMAgentHost (agent_host/host.py, "agent in sandbox") need the same two
things from a MicroVM: drive it to RUNNING (resume PAUSED, wait out
creating/pausing/resuming, recreate FAILED) and open an exec transport to
it. They live here so the state machine has exactly one implementation and
the two managers cannot drift apart.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import (
    MicroVMClientError,
    MicroVMNotFoundError,
    SandboxError,
)
from langchain_digitalocean_microvm.models import Checkpoint, MicroVMState
from langchain_digitalocean_microvm.status import (
    RESUMABLE_STATUSES,
    REUSABLE_STATUSES,
    TERMINAL_FAILURE_STATUSES,
    TRANSITIONAL_STATUSES,
    MicroVMStatus,
)
from langchain_digitalocean_microvm.transport.base import (
    SandboxTransport,
    TransportCredentials,
)

logger = logging.getLogger(__name__)

# Safety net against a pathological loop (e.g. a MicroVM that keeps landing
# back in FAILED immediately after every recreate); ordinary flows settle in
# 1-3 iterations.
MAX_SETTLE_ITERATIONS = 6

# A zero-arg callback that creates a replacement MicroVM and reports whether
# THIS caller created it (see MicroVMLifecycle._create_real for the
# create-race semantics behind the bool).
Recreate = Callable[[], tuple[MicroVMState, bool]]


def checkpoint_restorable(client: MicroVMClient, checkpoint: Checkpoint) -> bool:
    """False for a checkpoint with outbound credentials whose origin MicroVM is gone.

    The service deletes a MicroVM's stored credentials with the MicroVM, and its checkpoints
    cannot be restored afterwards.
    """
    if not checkpoint.has_credentials:
        return True
    if checkpoint.microvm_id is None:
        return False
    try:
        origin = client.get(checkpoint.microvm_id)
    except MicroVMNotFoundError:
        return False
    return origin.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


def settle_to_running(
    client: MicroVMClient,
    state: MicroVMState,
    *,
    label: str,
    ready_timeout: float,
    recreate: Recreate | None,
    freshly_created: bool = False,
) -> tuple[MicroVMState, bool]:
    """Resolve `state` to RUNNING, or raise SandboxError.

    Reuses RUNNING as-is, resumes PAUSED, waits out a transitional status,
    and gives UNKNOWN exactly one retry (a fresh GET) before refusing to
    guess that an unrecognized status means "ready".

    On FAILED/TERMINATING/TERMINATED: if `recreate` is given, best-effort delete then
    call it and keep settling the replacement; if `recreate` is None, raise
    -- callers that must not silently swap in a blank MicroVM (an agent host
    being *resumed*, whose in-VM state is the whole point) pass None.

    `label` only decorates error messages (e.g. "sandbox_id='x'").
    """
    unknown_retried = False

    for _ in range(MAX_SETTLE_ITERATIONS):
        if state.status in REUSABLE_STATUSES:
            return state, freshly_created

        if state.status in RESUMABLE_STATUSES:
            state = client.resume(state.id, timeout=ready_timeout)
            continue

        if state.status in TRANSITIONAL_STATUSES:
            state = client.wait_for_status(state.id, timeout=ready_timeout)
            continue

        if state.status in TERMINAL_FAILURE_STATUSES:
            if recreate is None:
                raise SandboxError(
                    f"MicroVM {state.id} for {label} is {state.status}; refusing to "
                    "recreate it because its in-VM state would be lost"
                )
            try:
                client.delete(state.id)
            except MicroVMClientError:
                logger.warning(
                    "Best-effort delete of a %s MicroVM before recreating it failed",
                    state.status,
                    exc_info=True,
                )
            state, freshly_created = recreate()
            continue

        # MicroVMStatus.UNKNOWN: refetch once before giving up.
        if unknown_retried:
            raise SandboxError(
                f"MicroVM {state.id} for {label} reports an unrecognized status even "
                "after a retry; refusing to guess whether it is ready."
            )
        unknown_retried = True
        state = client.get(state.id)

    raise SandboxError(
        f"MicroVM for {label} did not settle to RUNNING after "
        f"{MAX_SETTLE_ITERATIONS} lifecycle transitions"
    )


def connect_transport(config: MicroVMConfig, state: MicroVMState, transport: SandboxTransport) -> SandboxTransport:
    """Connect `transport` (an ExecTransport or EndpointTransport) to the MicroVM in `state` and return it."""
    if config.api_token is None:
        raise SandboxError("MicroVMConfig.api_token is required to run commands in a MicroVM")
    transport.connect(
        host=state.id,
        port=config.http_port,
        credentials=TransportCredentials(api_token=config.api_token, access_token=config.access_token),
        timeout=config.ready_timeout,
    )
    return transport
