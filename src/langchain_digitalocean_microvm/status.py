"""Typed lifecycle status for a MicroVM.

Values are the `state` strings of the DigitalOcean MicroVMs API (see
"How to Monitor a MicroVM"): creating, running, pausing,
paused, resuming, terminating, terminated, failed, unknown. Every raw server
string is routed through parse_status() so a value added later degrades to
UNKNOWN instead of crashing the lifecycle state machine in reconcile.py.
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class MicroVMStatus(StrEnum):
    CREATING = "creating"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"
    UNKNOWN = "unknown"


REUSABLE_STATUSES: frozenset[MicroVMStatus] = frozenset({MicroVMStatus.RUNNING})
RESUMABLE_STATUSES: frozenset[MicroVMStatus] = frozenset({MicroVMStatus.PAUSED})
TRANSITIONAL_STATUSES: frozenset[MicroVMStatus] = frozenset(
    {
        MicroVMStatus.CREATING,
        MicroVMStatus.PAUSING,
        MicroVMStatus.RESUMING,
    }
)
# TERMINATING is final for our purposes: DELETE returns 204 and the MicroVM
# then reports `terminating` for a while before it disappears. Nothing can bring it back, so it is never waited out.
TERMINAL_FAILURE_STATUSES: frozenset[MicroVMStatus] = frozenset(
    {MicroVMStatus.FAILED, MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED}
)


def parse_status(raw: str) -> MicroVMStatus:
    """Parse a raw server status string, falling back to UNKNOWN instead of raising."""
    try:
        return MicroVMStatus(raw)
    except ValueError:
        logger.warning("Unrecognized MicroVM status %r; treating as UNKNOWN", raw)
        return MicroVMStatus.UNKNOWN
