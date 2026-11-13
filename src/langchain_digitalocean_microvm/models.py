"""Data shapes shared across the client, transport, and lifecycle layers.

Field names follow the DigitalOcean MicroVMs API objects (see
the "API Reference" docs); `raw` keeps the untouched payload so fields we have
not modeled stay visible for debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_digitalocean_microvm.status import MicroVMStatus

# The only cpu/memory pairs the API accepts (docs, "Sizes"). Which of them a
# team may use is reported by GET /v2/microvms/options (`sizes[].available`).
VALID_SIZES: frozenset[tuple[int, int]] = frozenset({(1, 2048), (2, 4096), (8, 16384), (16, 32768)})
_SIZE_SLUG = re.compile(r"^\s*(\d+)\s*vcpu\s*-\s*(\d+)\s*gb\s*$", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(h|m|s)")


def parse_duration(text: str) -> float:
    """'10m' -> 600.0; '1h30m' -> 5400.0; '45s' / '5m0s' (the API's response form) also accepted."""
    raw = text.strip().lower()
    parts = _DURATION_PART.findall(raw)
    if not parts or "".join(n + u for n, u in parts) != raw:
        raise ValueError(f"Duration must look like '30s', '10m' or '1h', got {text!r}")
    factor = {"h": 3600.0, "m": 60.0, "s": 1.0}
    return sum(float(n) * factor[u] for n, u in parts)


@dataclass(frozen=True, slots=True)
class MicroVMSize:
    """vCPUs and memory in MiB, as the API's `size` object. `disk` (GB) is set by the API."""

    cpu: int
    memory: int
    disk: int | None = None

    def __post_init__(self) -> None:
        if (self.cpu, self.memory) not in VALID_SIZES:
            valid = ", ".join(f"{c}vcpu-{m // 1024}gb" for c, m in sorted(VALID_SIZES))
            raise ValueError(f"Unsupported MicroVM size {self.cpu} vCPU / {self.memory} MiB; valid: {valid}")

    @classmethod
    def parse(cls, slug: str) -> MicroVMSize:
        """'1vcpu-2gb' -> MicroVMSize(cpu=1, memory=2048)."""
        match = _SIZE_SLUG.match(slug)
        if match is None:
            raise ValueError(f"MicroVM size must look like '1vcpu-2gb', got {slug!r}")
        return cls(cpu=int(match.group(1)), memory=int(match.group(2)) * 1024)

    @property
    def slug(self) -> str:
        return f"{self.cpu}vcpu-{self.memory // 1024}gb"

    def to_api(self) -> dict[str, int]:
        return {"cpu": self.cpu, "memory": self.memory}


@dataclass(frozen=True)
class MicroVMState:
    """A snapshot of one MicroVM, as returned by MicroVMClient.

    `endpoint` is the hostname of the default HTTPS endpoint
    (`https://<id>.<region>.microvm.ondigitalocean.com`), or None until the
    API reports that endpoint ACTIVE. Requests to it need a DigitalOcean API
    token with the `microvm:access` scope.
    """

    id: str
    name: str
    status: MicroVMStatus
    size: MicroVMSize | None
    region: str | None
    endpoint: str | None
    http_port: int | None
    created_at: datetime | None
    tags: tuple[str, ...] = field(default_factory=tuple)
    failure_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str | None:
        return f"https://{self.endpoint}" if self.endpoint else None


@dataclass(frozen=True)
class Checkpoint:
    """A captured copy of a MicroVM's memory and disk (docs, "Checkpoint Object")."""

    id: str
    name: str
    microvm_id: str | None
    region: str | None
    size: MicroVMSize | None
    status: str  # CHECKPOINT_CREATING / _AVAILABLE / _FAILED / _DELETING / _DELETED
    created_at: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "CHECKPOINT_AVAILABLE"


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of POST /v2/microvms/{id}/exec. A non-zero exit code is a normal result."""

    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False
