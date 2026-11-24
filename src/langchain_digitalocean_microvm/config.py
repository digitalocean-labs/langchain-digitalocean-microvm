"""Configuration for the MicroVM client, transports, sandbox lifecycle, and agent hosts.

This module is the one place that reads `DO_MICROVM_*` environment variables (a test enforces it):
`MicroVMConfig.from_env()` reads the sandbox, transport and agent-host ones, and the readers below
(`env_value`, `env_flag`, `env_int`, `env_csv`) serve the settings that live elsewhere
(`GatewaySettings.from_env`, the local agent host). The variables:

| Variable | Read by | Meaning |
| --- | --- | --- |
| `DO_MICROVM_API_TOKEN`, `DO_MICROVM_ACCESS_TOKEN` | `MicroVMConfig` | control-plane and endpoint tokens |
| `DO_MICROVM_TRANSPORT` | `MicroVMConfig` | `endpoint` (default), `exec` or `local` |
| `DO_MICROVM_IMAGE`, `_HTTP_PORT`, `_SIZE`, `_REGION`, `_IDLE_TIMEOUT` | `MicroVMConfig` | sandbox image and shape |
| `DO_MICROVM_NAMESPACE` | `MicroVMConfig`, `GatewaySettings` | owner or deployment on a shared team |
| `DO_MICROVM_MAX_CONCURRENT_CREATES` | `MicroVMConfig` | bound on creates in flight |
| `DO_MICROVM_EGRESS`, `_EGRESS_CREDENTIALS` | `MicroVMConfig` | outbound policy (egress.py) |
| `DO_MICROVM_AGENT_IMAGE`, `_AGENT_SIZE`, `_AGENT_PORT` | `MicroVMConfig` | agent hosts |
| `DO_MICROVM_SUBAGENT_ENV_PASSTHROUGH`, `_WARM_START` | `MicroVMConfig`, local agent host | agent hosts |
| `DO_MICROVM_HOST_MODE`, `DO_MICROVM_GATEWAY_TOKEN` | `GatewaySettings` | where hosts run, gateway token |
| `DO_MICROVM_SUBAGENT_TEMPLATE` | local agent host | directory served by the local host |
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from langchain_digitalocean_microvm.egress import EgressPolicy
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.namespace import validate_namespace

DEFAULT_BASE_URL = "https://api.digitalocean.com/v2"
DEFAULT_REGION = "mkc1"  # the only MicroVMs region currently available
DEFAULT_SIZE = "1vcpu-2gb"
DEFAULT_READY_TIMEOUT = 300.0
# A running MicroVM normally lists its HTTPS endpoint within a second; past this, waiting longer
# does not help (the endpoint is not being provisioned), so callers fail with a clear error.
DEFAULT_ENDPOINT_WAIT = 60.0
# MicroVMs pause after this long without an INBOUND request -- CPU work and
# outbound traffic do not count. Anything that must keep
# running has to be polled more often than this. API maximum: 4 h per the
# docs; GET /v2/microvms/options can report a lower limit for a team.
DEFAULT_IDLE_TIMEOUT = "10m"
DEFAULT_HTTP_PORT = 8080

# Agent hosts (agent_host/): the Agent Server's port inside the container and
# the host size. 1 vCPU / 2 GB is the default because a team's
# total MicroVM memory can be limited (see docs/design.md), so two hosts fit.
DEFAULT_AGENT_PORT = 2024
DEFAULT_AGENT_SIZE = "1vcpu-2gb"

TransportName = Literal["exec", "endpoint", "local"]
_VALID_TRANSPORTS = ("exec", "endpoint", "local")

# Names of the variables read outside `MicroVMConfig.from_env`.
TRANSPORT_ENV = "DO_MICROVM_TRANSPORT"
NAMESPACE_ENV = "DO_MICROVM_NAMESPACE"
PASSTHROUGH_ENV = "DO_MICROVM_SUBAGENT_ENV_PASSTHROUGH"
TEMPLATE_DIR_ENV = "DO_MICROVM_SUBAGENT_TEMPLATE"
HOST_MODE_ENV = "DO_MICROVM_HOST_MODE"
GATEWAY_TOKEN_ENV = "DO_MICROVM_GATEWAY_TOKEN"
EGRESS_ENV = "DO_MICROVM_EGRESS"
EGRESS_CREDENTIALS_ENV = "DO_MICROVM_EGRESS_CREDENTIALS"


def env_value(name: str) -> str | None:
    """The variable's value, or None when it is unset or empty."""
    return os.environ.get(name) or None


def env_flag(name: str) -> bool:
    """True for `1`, `true`, `yes` or `on` (any case)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    """The variable as an int, `default` when it is unset or empty."""
    return int(os.environ.get(name) or default)


def env_csv(name: str) -> tuple[str, ...]:
    """The variable's comma-separated items, stripped, without empty ones."""
    return _split_csv(os.environ.get(name, ""))


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def egress_from_env() -> EgressPolicy | None:
    """The outbound policy `DO_MICROVM_EGRESS` and `DO_MICROVM_EGRESS_CREDENTIALS` describe, or None.

    `DO_MICROVM_EGRESS`: `none` denies every destination, `*` allows every one, anything else is a
    comma-separated list of allowed destinations. `DO_MICROVM_EGRESS_CREDENTIALS`: comma-separated
    `https-url=VARIABLE` items; each secret is read from the variable named after `=`, so no secret
    appears in this value. Credentials without `DO_MICROVM_EGRESS` allow every destination.
    """
    policy = EgressPolicy.parse(os.environ.get(EGRESS_ENV, ""))
    items = env_csv(EGRESS_CREDENTIALS_ENV)
    if items and policy is None:
        policy = EgressPolicy.allow_all()
    for item in items:
        url, sep, variable = item.rpartition("=")
        if not sep or not url or not variable:
            raise ValueError(f"{EGRESS_CREDENTIALS_ENV} items are https-url=VARIABLE, got {item!r}")
        secret = os.environ.get(variable.strip())
        if not secret:
            raise ValueError(f"{EGRESS_CREDENTIALS_ENV} names {variable.strip()!r}, which is unset or empty")
        assert policy is not None
        policy = policy.with_credential(url.strip(), secret=secret)
    return policy


@dataclass(frozen=True)
class MicroVMConfig:
    """Runtime configuration, normally built via `from_env()`.

    `api_token` drives the control plane (create/read/pause/resume/exec/
    delete). `access_token` is what requests to a MicroVM's HTTPS endpoint
    carry: a separate token scoped to `microvm:access` only, required by
    agent hosts and the endpoint transport (the control token is never sent
    to an endpoint).

    `transport`: "endpoint" (default: through the sandbox server in the
    MicroVM, over its HTTPS endpoint; needs `access_token`), "exec" (commands
    and files through the exec API) or "local" (local, no credentials).

    `base_url`/`ready_timeout` are constructor-only overrides. The `agent_*`
    / `subagent_env_passthrough` fields only matter to agent hosts.
    """

    api_token: str | None = None
    access_token: str | None = None
    base_url: str = DEFAULT_BASE_URL
    transport: TransportName = "endpoint"
    size: MicroVMSize = field(default_factory=lambda: MicroVMSize.parse(DEFAULT_SIZE))
    image: str | None = None
    region: str = DEFAULT_REGION
    http_port: int = DEFAULT_HTTP_PORT
    idle_timeout: str = DEFAULT_IDLE_TIMEOUT
    ready_timeout: float = DEFAULT_READY_TIMEOUT
    # Separates this owner or deployment from others on a shared team (see namespace.py):
    # 1-16 lowercase letters or digits; None is the default namespace.
    namespace: str | None = None
    # At most this many sandboxes are created or resumed at once by one MicroVMLifecycle (all callers
    # share it: threads, subagents, per-task sandboxes); None: no limit. Per process, not across processes.
    max_concurrent_creates: int | None = None
    # Outbound policy of the sandboxes and templates this config creates from the image (egress.py);
    # None sends no policy. A sandbox created from a template has the template's policy.
    egress: EgressPolicy | None = None

    # --- agent hosts -------------------------------------------------
    agent_port: int = DEFAULT_AGENT_PORT
    agent_image: str | None = None
    agent_size: MicroVMSize = field(default_factory=lambda: MicroVMSize.parse(DEFAULT_AGENT_SIZE))
    # Names of supervisor-side environment variables copied verbatim into
    # each host's environment (e.g. ("DO_MODEL_ACCESS_KEY",)). Empty by
    # default: every secret that enters a VM should be an explicit decision.
    subagent_env_passthrough: tuple[str, ...] = ()
    # Start agent hosts from a checkpoint of an already-booted host (about 6 s
    # instead of ~14 s once the checkpoint exists). See MicroVMAgentHost.
    warm_start: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", validate_namespace(self.namespace))
        if self.max_concurrent_creates is not None and self.max_concurrent_creates < 1:
            raise ValueError(f"max_concurrent_creates must be at least 1 or None, got {self.max_concurrent_creates}")

    @classmethod
    def from_env(cls) -> MicroVMConfig:
        """Build a config from DO_MICROVM_* environment variables.

        Does not load a .env file itself -- callers load one first (the
        examples use python-dotenv; `langgraph dev` reads langgraph.json's `env`).
        """
        transport = env_value(TRANSPORT_ENV) or "endpoint"
        if transport not in _VALID_TRANSPORTS:
            raise ValueError(f"DO_MICROVM_TRANSPORT must be one of {_VALID_TRANSPORTS}, got {transport!r}")
        return cls(
            api_token=env_value("DO_MICROVM_API_TOKEN"),
            access_token=env_value("DO_MICROVM_ACCESS_TOKEN"),
            transport=transport,  # type: ignore[arg-type]
            size=MicroVMSize.parse(env_value("DO_MICROVM_SIZE") or DEFAULT_SIZE),
            image=env_value("DO_MICROVM_IMAGE"),
            region=env_value("DO_MICROVM_REGION") or DEFAULT_REGION,
            namespace=env_value(NAMESPACE_ENV),
            max_concurrent_creates=env_int("DO_MICROVM_MAX_CONCURRENT_CREATES", 0) or None,
            egress=egress_from_env(),
            http_port=env_int("DO_MICROVM_HTTP_PORT", DEFAULT_HTTP_PORT),
            idle_timeout=env_value("DO_MICROVM_IDLE_TIMEOUT") or DEFAULT_IDLE_TIMEOUT,
            agent_port=env_int("DO_MICROVM_AGENT_PORT", DEFAULT_AGENT_PORT),
            agent_image=env_value("DO_MICROVM_AGENT_IMAGE"),
            agent_size=MicroVMSize.parse(env_value("DO_MICROVM_AGENT_SIZE") or DEFAULT_AGENT_SIZE),
            subagent_env_passthrough=env_csv(PASSTHROUGH_ENV),
            warm_start=env_flag("DO_MICROVM_WARM_START"),
        )
