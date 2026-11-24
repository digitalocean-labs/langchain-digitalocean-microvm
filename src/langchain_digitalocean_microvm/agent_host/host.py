"""Turn a MicroVM into a ready LangGraph Agent Server: MicroVMAgentHost.

This is the "agent in sandbox" sibling of provider.MicroVMLifecycle.
Where the lifecycle manager hands back a MicroVMSandbox (a backend the
*local* agent runs `execute` through), the agent host hands back a
HostHandle (an HTTPS URL + token the *gateway* sends Agent Protocol calls
to). Both sit on the same MicroVMClient and reconcile.settle_to_running.

get_or_create(host_id, spec) does, in order:

1. look the MicroVM up by its deterministic name (microvm_name); if missing,
   create it from the agent image with the host's whole environment --
   including a NEW random SUBAGENT_HOST_TOKEN -- in the create call (the
   MicroVMs API takes env at create time and never changes or returns it);
2. settle it to RUNNING (resume PAUSED, wait out creating/resuming, recreate
   FAILED) and wait for its HTTPS endpoint to become ACTIVE;
3. probe GET /ok through that endpoint until it answers 200 or
   ready_timeout elapses (HostNotReadyError).

Requests to a MicroVM endpoint need TWO credentials: `Authorization: Bearer
<DigitalOcean token with microvm:access>` (checked by DigitalOcean's edge)
and `x-api-key: <host token>` (checked by the Agent Server's auth.py). The
endpoint token should be a separate access-only one (config.access_token).

Warm start (config.warm_start, DO_MICROVM_WARM_START=1) lives in warm.py:
new hosts are clones of a golden checkpoint.

Because the environment is immutable, a host whose token this process does
not know (e.g. after a supervisor restart) cannot be re-keyed. Callers that
persisted the token (the gateway's store) hand it back via remember_token();
otherwise get_or_create replaces the host and resume refuses.

local hosts (DO_MICROVM_HOST_MODE=local) are LocalDevServerHost; DO_MICROVM_TRANSPORT,
a sandbox-backend setting, does not affect this class.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx

from langchain_digitalocean_microvm.agent_host.env import build_host_env
from langchain_digitalocean_microvm.agent_host.models import (
    HOST_MANAGED_TAG,
    HostHandle,
    HostSpec,
    HostStatus,
    graph_id_from_tags,
    host_graph_tag,
    host_id_from_tags,
    host_id_tag,
    host_status_from_microvm,
)
from langchain_digitalocean_microvm.agent_host.warm import (  # the GOLDEN_* names stay importable from here
    GOLDEN_PREFIX as GOLDEN_PREFIX,
)
from langchain_digitalocean_microvm.agent_host.warm import (
    GOLDEN_TAG as GOLDEN_TAG,
)
from langchain_digitalocean_microvm.agent_host.warm import (
    WarmStart,
)
from langchain_digitalocean_microvm.client import MicroVMClient, microvm_name
from langchain_digitalocean_microvm.config import DEFAULT_ENDPOINT_WAIT, MicroVMConfig
from langchain_digitalocean_microvm.egress import EgressPolicy
from langchain_digitalocean_microvm.exceptions import (
    HostCapacityError,
    HostNotReadyError,
    MicroVMAlreadyExistsError,
    MicroVMAPIError,
    MicroVMClientError,
    MicroVMForbiddenError,
    MicroVMNotFoundError,
    SandboxError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.models import (
    MicroVMSize,
    MicroVMState,
    parse_duration,
)
from langchain_digitalocean_microvm.namespace import (
    namespace_from_tags,
    namespace_tags,
    namespaced_prefix,
)
from langchain_digitalocean_microvm.reconcile import settle_to_running
from langchain_digitalocean_microvm.status import TRANSITIONAL_STATUSES, MicroVMStatus

logger = logging.getLogger(__name__)

HOST_NAME_PREFIX = "microvm-agenthost"
DEFAULT_TOKEN_FILE = "/work/.microvm-agent/token"  # the image's HOME; see images/agent-host/auth.py

# (url, token, per-attempt timeout seconds) -> did GET /ok answer 200?
Probe = Callable[[str, str, float], bool]
# (url, token, timeout seconds) -> parsed body of GET /metrics?format=json
MetricsFetcher = Callable[[str, str, float], dict[str, Any]]

_TOKEN_BYTES = 32


def default_probe(url: str, token: str, timeout: float, *, headers: Mapping[str, str] | None = None) -> bool:
    """GET {url}/ok with the host token (plus any edge `headers`); True only on HTTP 200."""
    try:
        response = httpx.get(f"{url}/ok", headers={**(headers or {}), "x-api-key": token}, timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def default_metrics_fetcher(
    url: str, token: str, timeout: float, *, headers: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """GET {url}/metrics?format=json with the host token; raises on non-2xx."""
    response = httpx.get(
        f"{url}/metrics",
        params={"format": "json"},
        headers={**(headers or {}), "x-api-key": token},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def queue_is_idle(body: dict[str, Any]) -> bool:
    """True when a GET /metrics?format=json body reports no pending and no running runs.

    Keys follow langgraph_runtime_inmem's Runs.stats: `queue.n_pending`,
    `queue.n_running`. Anything unparseable is NOT idle.
    """
    queue = body.get("queue")
    if not isinstance(queue, dict):
        return False
    try:
        pending = int(queue.get("n_pending") or 0)
        running = int(queue.get("n_running") or 0)
    except (TypeError, ValueError):
        return False
    return pending == 0 and running == 0


def _iso(moment: datetime | None) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if moment else ""


class MicroVMAgentHost:
    """Create-or-reuse manager for agent hosts, one MicroVM per host_id."""

    def __init__(
        self,
        config: MicroVMConfig | None = None,
        *,
        client: MicroVMClient | None = None,
        probe: Probe | None = None,
        metrics_fetcher: MetricsFetcher | None = None,
        token_factory: Callable[[], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        probe_interval: float = 1.0,
        probe_attempt_timeout: float = 5.0,
        token_file: str = DEFAULT_TOKEN_FILE,
        endpoint_wait: float = DEFAULT_ENDPOINT_WAIT,
    ) -> None:
        """Build a host manager.

        `client`, `probe`, `metrics_fetcher`,
        `token_factory`, `sleep` and `monotonic` are test seams; production
        callers pass `config` (or nothing: DO_MICROVM_* from the environment).
        `endpoint_wait` is how long a host that is already running may take to
        list its HTTPS endpoint (booting to running gets `config.ready_timeout`).
        """
        # config.transport is a sandbox-backend setting (MicroVMLifecycle); agent hosts
        # are always real MicroVMs reached over exec/HTTPS. local hosts are LocalDevServerHost,
        # chosen by DO_MICROVM_HOST_MODE, not by DO_MICROVM_TRANSPORT.
        self.config = config or MicroVMConfig.from_env()
        self._client: MicroVMClient | None = client
        if self._client is None:
            if not self.config.api_token:
                raise SandboxError("MicroVMConfig.api_token (DO_MICROVM_API_TOKEN) is required for agent hosts")
            self._client = MicroVMClient(self.config.api_token, base_url=self.config.base_url)
        if not self.config.access_token:
            raise SandboxError(
                "MicroVMConfig.access_token (DO_MICROVM_ACCESS_TOKEN) is required for agent hosts: a token "
                "scoped to microvm:access only, for requests to the hosts' endpoints"
            )

        edge = self.edge_headers()
        self._probe: Probe = probe if probe is not None else (lambda u, t, s: default_probe(u, t, s, headers=edge))
        self._metrics_fetcher: MetricsFetcher = (
            metrics_fetcher if metrics_fetcher is not None else (lambda u, t, s: default_metrics_fetcher(u, t, s, headers=edge))
        )
        self._token_factory = token_factory if token_factory is not None else lambda: secrets.token_urlsafe(_TOKEN_BYTES)
        self._sleep = sleep
        self._monotonic = monotonic
        self._probe_interval = probe_interval
        self._probe_attempt_timeout = probe_attempt_timeout

        # host_id -> the token its Agent Server was created with. Only this
        # process (or whoever calls remember_token) knows it.
        self._tokens: dict[str, str] = {}
        # Warm start: where a clone's own token goes, and the golden checkpoints (warm.py).
        self.token_file = token_file
        self._warm = WarmStart(self)
        self._endpoint_wait = endpoint_wait

    # ------------------------------------------------------------------
    # public sync API
    # ------------------------------------------------------------------

    @property
    def idle_timeout_seconds(self) -> float:
        """Hosts pause after this long without inbound requests (the gateway polls faster than this)."""
        return parse_duration(self.config.idle_timeout)

    def edge_headers(self) -> dict[str, str]:
        """Headers DigitalOcean's edge needs on every request to a MicroVM endpoint."""
        return {"Authorization": f"Bearer {self.config.access_token}"}

    def remember_token(self, host_id: str, token: str) -> None:
        """Teach this manager the token of a host created earlier (e.g. by a previous process)."""
        self._tokens[host_id] = token

    def get_or_create(self, *, host_id: str, spec: HostSpec) -> HostHandle:
        """Return a READY HostHandle for `host_id`, creating/resuming as needed."""
        client = self._client_or_raise()
        state = client.get_by_name(self._name(host_id))
        freshly_created = False
        if state is not None and host_id not in self._tokens:
            logger.warning(
                "Agent host %s exists but its token is unknown and its environment is immutable; replacing it",
                host_id,
            )
            self._delete_quietly(state.id)
            state = None
        if state is None:
            state, freshly_created = self._create(host_id=host_id, spec=spec)

        state, _ = settle_to_running(
            client,
            state,
            label=f"host_id={host_id!r}",
            ready_timeout=self.config.ready_timeout,
            recreate=lambda: self._create(host_id=host_id, spec=spec),
            freshly_created=freshly_created,
        )
        state = self._wait_endpoint(state, host_id)
        handle = self.wait_ready(self._handle(state, host_id=host_id, graph_id=spec.graph_id))
        if self._warm.is_token_pending(host_id):
            # A clone: GET /ok needs no token, so probe first. An exec sent to a fresh clone before its
            # workload answers slows the clone's start by more than the exec itself takes.
            self._warm.install_token(state, host_id)
        return handle

    def wait_ready(self, handle: HostHandle, *, timeout: float | None = None) -> HostHandle:
        """Probe GET /ok until it answers 200; return the handle marked READY."""
        budget = self.config.ready_timeout if timeout is None else timeout
        deadline = self._monotonic() + budget
        while True:
            if self._probe(handle.url, handle.token, self._probe_attempt_timeout):
                return replace(handle, status=HostStatus.READY)
            if self._monotonic() >= deadline:
                raise HostNotReadyError(
                    f"Agent Server on MicroVM {handle.microvm_id} (host_id={handle.host_id!r}) "
                    f"did not answer GET {handle.url}/ok within {budget}s"
                )
            self._sleep(self._probe_interval)

    def pause(self, *, host_id: str) -> None:
        """Pause the host's MicroVM. Idempotent on an already-PAUSED host.

        MicroVMs also pause by themselves after `config.idle_timeout` without
        inbound requests. The never-pause-BUSY rule is the
        gateway's to enforce.
        """
        client = self._client_or_raise()
        state = self._lookup(host_id)
        if state.status == MicroVMStatus.PAUSED:
            return
        if state.status in TRANSITIONAL_STATUSES:
            raise SandboxError(f"MicroVM {state.id} for host_id={host_id!r} is {state.status}; retry once it settles")
        if state.status != MicroVMStatus.RUNNING:
            raise SandboxError(f"Cannot pause MicroVM {state.id} for host_id={host_id!r} in status {state.status}")
        client.pause(state.id, timeout=self.config.ready_timeout)

    def resume(self, *, host_id: str, spec: HostSpec | None = None) -> HostHandle:
        """Resume the host's MicroVM and return a READY handle.

        Never recreates a FAILED MicroVM: a resumed host's value is its
        in-VM state, so the caller (gateway) must treat that as an error.
        """
        client = self._client_or_raise()
        if host_id not in self._tokens:
            raise SandboxError(
                f"host_id={host_id!r}: token unknown in this process and MicroVM environments are immutable; "
                "call remember_token() with the stored token first"
            )
        state = self._lookup(host_id)
        state, _ = settle_to_running(
            client, state, label=f"host_id={host_id!r}", ready_timeout=self.config.ready_timeout, recreate=None
        )
        state = self._wait_endpoint(state, host_id)
        graph_id = spec.graph_id if spec else (graph_id_from_tags(state.tags) or "")
        return self.wait_ready(self._handle(state, host_id=host_id, graph_id=graph_id))

    def delete(self, *, host_id: str) -> None:
        """Permanently destroy the host's MicroVM and forget its token."""
        client = self._client_or_raise()
        state = self._lookup(host_id)
        client.delete(state.id)
        self._tokens.pop(host_id, None)

    def list_hosts(self, *, graph_id: str | None = None) -> list[HostHandle]:
        """Inventory every agent host (optionally one graph's), by MicroVM status.

        Statuses come from the API, not from probing /ok, so a RUNNING
        MicroVM is reported READY even if its server is down. Hosts whose
        token this process does not know carry an empty token.
        """
        client = self._client_or_raise()
        handles: list[HostHandle] = []
        for state in client.list(tag=HOST_MANAGED_TAG):
            if namespace_from_tags(state.tags) != self.config.namespace:  # another namespace's host
                continue
            host_id = host_id_from_tags(state.tags)
            if host_id is None:
                logger.warning("A MicroVM carries %s but no host id tag; skipping it", HOST_MANAGED_TAG)
                continue
            state_graph = graph_id_from_tags(state.tags) or ""
            if graph_id is not None and state_graph != graph_id:
                continue
            handles.append(
                HostHandle(
                    host_id=host_id,
                    microvm_id=state.id,
                    graph_id=state_graph,
                    url=state.url or "",
                    token=self._tokens.get(host_id, ""),
                    status=host_status_from_microvm(state.status),
                    booted_at=_iso(state.created_at),
                )
            )
        return handles

    def is_idle(self, handle: HostHandle) -> bool:
        """True when the server reports no pending and no running runs (GET /metrics).

        Any failure to fetch or parse is reported as NOT idle: pausing a host
        we cannot inspect is the one mistake this method must never make.
        """
        try:
            body = self._metrics_fetcher(handle.url, handle.token, self._probe_attempt_timeout)
        except Exception:  # noqa: BLE001 - any failure means "unknown", i.e. not idle
            logger.warning("Could not fetch metrics from host %s; treating as busy", handle.host_id, exc_info=True)
            return False
        return queue_is_idle(body)

    # ------------------------------------------------------------------
    # public async API -- thin asyncio.to_thread wrappers.
    # ------------------------------------------------------------------

    async def aget_or_create(self, **kwargs: Any) -> HostHandle:
        return await asyncio.to_thread(self.get_or_create, **kwargs)

    async def await_ready(self, handle: HostHandle, **kwargs: Any) -> HostHandle:
        return await asyncio.to_thread(self.wait_ready, handle, **kwargs)

    async def apause(self, **kwargs: Any) -> None:
        return await asyncio.to_thread(self.pause, **kwargs)

    async def aresume(self, **kwargs: Any) -> HostHandle:
        return await asyncio.to_thread(self.resume, **kwargs)

    async def adelete(self, **kwargs: Any) -> None:
        return await asyncio.to_thread(self.delete, **kwargs)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _client_or_raise(self) -> MicroVMClient:
        assert self._client is not None
        return self._client

    def _name(self, host_id: str) -> str:
        return microvm_name(namespaced_prefix(HOST_NAME_PREFIX, self.config.namespace), host_id)

    def _lookup(self, host_id: str) -> MicroVMState:
        name = self._name(host_id)
        state = self._client_or_raise().get_by_name(name)
        if state is None:
            raise SandboxNotFoundError(f"No MicroVM registered for host_id={host_id!r}") from MicroVMNotFoundError(
                f"No MicroVM named {name!r}"
            )
        return state

    def _create(self, *, host_id: str, spec: HostSpec) -> tuple[MicroVMState, bool]:
        """Create a fresh agent-host MicroVM with a new token baked into its environment.

        (state, True) on a normal create. If a concurrent create won the
        name, that MicroVM's token is unknown to us, so it is replaced.
        """
        client = self._client_or_raise()
        image = spec.image or self.config.agent_image
        if image is None:
            raise SandboxError(
                "An agent-host image is required: set HostSpec.image or DO_MICROVM_AGENT_IMAGE "
                "(a linux/amd64 image serving the Agent Server on DO_MICROVM_AGENT_PORT)"
            )
        size = self.config.agent_size if spec.size is None else MicroVMSize.parse(spec.size)
        name = self._name(host_id)
        token = self._token_factory()
        if self.config.warm_start:
            checkpoint = self._warm.golden_checkpoint(spec, image, size)
            if checkpoint is not None:
                warm = self._warm.create_from_checkpoint(name, checkpoint, host_id, spec)
                if warm is not None:
                    self._tokens[host_id] = token
                    self._warm.mark_token_pending(host_id)
                    return warm, True
        env = build_host_env(spec, token, passthrough=self.config.subagent_env_passthrough)
        policy = self.egress_for(spec)
        kwargs: dict[str, Any] = {
            "name": name,
            "oci_ref": image,
            "region": self.config.region,
            "size": size,
            "http_port": self.config.agent_port,
            "http_protocol": "http",
            "environment": env,
            "idle_timeout": self.config.idle_timeout,
            "tags": [HOST_MANAGED_TAG, host_id_tag(host_id), host_graph_tag(spec.graph_id), *namespace_tags(self.config.namespace)],
            "egress_policy": policy.for_image(image) if policy is not None else None,
        }
        try:
            state = client.create(**kwargs)
        except MicroVMForbiddenError as exc:
            raise self._explain_forbidden(exc) from exc
        except (MicroVMAlreadyExistsError, MicroVMAPIError) as exc:
            if isinstance(exc, MicroVMAPIError) and exc.status_code not in (400, 409, 422):
                raise
            existing = client.get_by_name(name)
            if existing is None:
                raise
            logger.warning("Lost a create race for host %s; replacing its MicroVM", host_id)
            self._delete_quietly(existing.id)
            state = client.create(**kwargs)
        self._tokens[host_id] = token
        return state, True

    def egress_for(self, spec: HostSpec) -> EgressPolicy | None:
        """The outbound policy a host for `spec` is created with: `spec.egress`, else `config.agent_egress`."""
        return spec.egress or self.config.agent_egress

    def _golden_name(self, spec: HostSpec, image: str, size: MicroVMSize) -> str:
        return self._warm.golden_name(spec, image, size)

    def _explain_forbidden(self, exc: MicroVMForbiddenError) -> SandboxError:
        """A 403 on create is either the team's MicroVM limit or a missing scope; tell them apart."""
        client = self._client_or_raise()
        try:
            limits = client.options().get("account_limits") or {}
            live = [s for s in client.list() if s.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)]
        except MicroVMClientError:
            return SandboxError(f"Creating an agent host was refused: {exc}")
        max_count = limits.get("max_total_count")
        max_memory = limits.get("max_memory_bytes")
        used_memory = sum((s.size.memory if s.size else 0) * 1024 * 1024 for s in live)
        wanted = self.config.agent_size.memory * 1024 * 1024
        if (isinstance(max_count, int) and len(live) >= max_count) or (
            isinstance(max_memory, int) and used_memory + wanted > max_memory
        ):
            return HostCapacityError(
                f"The team is at its MicroVM limit ({len(live)} of {max_count} MicroVMs, "
                f"{used_memory // 2**20} of {max_memory // 2**20 if isinstance(max_memory, int) else '?'} MiB; "
                "paused ones count)."
            )
        return SandboxError(f"Creating an agent host was refused (check the token's microvm:create scope): {exc}")

    def _wait_endpoint(self, state: MicroVMState, host_id: str) -> MicroVMState:
        if state.status == MicroVMStatus.RUNNING and state.endpoint:
            return state
        try:
            return self._wait_for_url(state.id)
        except (TimeoutError, MicroVMClientError) as exc:
            raise HostNotReadyError(f"MicroVM {state.id} for host_id={host_id!r}: {exc}") from exc

    def _wait_for_url(self, microvm_id: str) -> MicroVMState:
        """Wait (up to `config.ready_timeout`) for the MicroVM to boot, then (up to `endpoint_wait`)
        for its HTTPS endpoint, which a running MicroVM normally lists within a second."""
        client = self._client_or_raise()
        client.wait_for_status(microvm_id, timeout=self.config.ready_timeout)
        try:
            return client.wait_for_endpoint(microvm_id, timeout=self._endpoint_wait)
        except TimeoutError as exc:
            raise TimeoutError(f"{exc}: the MicroVMs API lists no HTTPS endpoint for it, which an agent host "
                               "needs") from exc

    def _delete_quietly(self, microvm_id: str) -> None:
        try:
            self._client_or_raise().delete(microvm_id)
        except MicroVMClientError:
            logger.warning("Best-effort delete of a MicroVM failed", exc_info=True)

    def _handle(self, state: MicroVMState, *, host_id: str, graph_id: str) -> HostHandle:
        if state.url is None:
            raise SandboxError(f"MicroVM {state.id} for host_id={host_id!r} has no ACTIVE endpoint")
        return HostHandle(
            host_id=host_id,
            microvm_id=state.id,
            graph_id=graph_id,
            url=state.url,
            token=self._tokens[host_id],
            status=HostStatus.BOOTING,
            booted_at=_iso(state.created_at),
        )
