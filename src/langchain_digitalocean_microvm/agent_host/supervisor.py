"""Supervisor-side glue: an embedded gateway that starts on first use and cleans up at exit.

A supervisor deep agent only needs two things from this package: the URL its
`AsyncSubAgent` specs point at, and the token that URL expects. `LazyGateway`
owns both and defers everything else -- picking a host manager (local or real
MicroVMs), opening the SQLite store, binding the loopback port -- until the
agent actually runs:

    gateway = LazyGateway({"echo": HostSpec(graph_id="echo")}, poll_interval=2.0)
    graph = create_deep_agent(
        model=model,
        subagents=[gateway.subagent(name="echo", description="...")],
        middleware=[gateway.middleware()],   # starts the gateway on the first turn
    )

Why lazy rather than at import: `langgraph dev` imports the graph module on
every reload, and a graph that is imported but never invoked should not spawn
`langgraph dev` processes or open ports. The URL is still known at import
(the port is reserved up front) so the specs can be built eagerly. `close()`
is registered with `atexit` on first start: in local mode it kills every local
host and records them as destroyed so a later process reports their tasks as
errors instead of proxying to a port nobody listens on; in microvm mode the
MicroVMs are left alone (their state is the point; the gateway's `tick()`
pauses idle ones and deletes them after their TTL, and scripts/gc_microvms.py
reclaims whatever outlived the supervisor).

Under `langgraph dev` that hook never fires: uvicorn ends its reload worker
with a re-raised SIGTERM after the graceful shutdown, so the interpreter dies
by signal. local hosts therefore do not depend on it -- LocalDevServerHost runs
each one under `_guard.py`, which terminates the server when its parent
process disappears -- and the gateway reconciles the stale store rows on the
next start (`tick()` marks hosts the manager no longer knows as FAILED, and a
`check` on their tasks answers `status: error`).

Files a subagent saved under its /outputs/ stay in its host. `artifacts_tool()`
gives the supervisor a `fetch_subagent_artifacts` tool that copies them through
the gateway into the supervisor's own backend (default: the deep agent's
StateBackend), under /subagents/<task_id>/, where `read_file` can see them:

    create_deep_agent(..., tools=[gateway.artifacts_tool()])

Configuration comes from the environment: DO_MICROVM_HOST_MODE
(`local`, the default, or `microvm`) and DO_MICROVM_GATEWAY_TOKEN (random per
process when unset). On start the token is
also exported as LANGGRAPH_API_KEY unless that is already set, because the
LangGraph SDK always sends `x-api-key` from LANGGRAPH_API_KEY / LANGSMITH_API_KEY
and the gateway is the only place that header should carry a real secret to
(see docs/design.md).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
from deepagents import AsyncSubAgent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool, StructuredTool

from langchain_digitalocean_microvm.agent_host.artifacts import (
    OUTPUTS_DIR,
    ArtifactError,
    safe_relative_path,
)
from langchain_digitalocean_microvm.agent_host.gateway import SubAgentGateway
from langchain_digitalocean_microvm.agent_host.host import MicroVMAgentHost
from langchain_digitalocean_microvm.agent_host.lifecycle import HostManager
from langchain_digitalocean_microvm.agent_host.local import (
    LOOPBACK,
    LocalDevServerHost,
    free_port,
)
from langchain_digitalocean_microvm.agent_host.models import HostSpec, HostStatus
from langchain_digitalocean_microvm.agent_host.spec import async_subagent_spec
from langchain_digitalocean_microvm.agent_host.store import default_db_path
from langchain_digitalocean_microvm.config import (
    GATEWAY_TOKEN_ENV,
    HOST_MODE_ENV,
    NAMESPACE_ENV,
    env_value,
)
from langchain_digitalocean_microvm.namespace import validate_namespace

logger = logging.getLogger(__name__)

HostMode = Literal["local", "microvm"]
SDK_API_KEY_ENV = "LANGGRAPH_API_KEY"
_VALID_HOST_MODES: tuple[HostMode, ...] = ("local", "microvm")
_TOKEN_BYTES = 32
_RETIRED_HOST_STATUSES = {HostStatus.FAILED.value, HostStatus.DESTROYED.value}
ARTIFACTS_TOOL_NAME = "fetch_subagent_artifacts"
# A middleware name is unique per agent: this package's own, apart from other gateways' start middleware.
GATEWAY_START_MIDDLEWARE_NAME = "microvm-gateway-start"
DEFAULT_ARTIFACTS_DEST = "/subagents"
DEFAULT_FETCH_MAX_BYTES = 8 * 1024 * 1024  # per tool call: everything lands in the supervisor's state

HostsFactory = Callable[[HostMode], HostManager]


def default_hosts_factory(host_mode: HostMode) -> HostManager:
    """Local mode: one local `langgraph dev` per host. MicroVM mode: real MicroVMs from DO_MICROVM_* config."""
    if host_mode == "local":
        return LocalDevServerHost()
    return MicroVMAgentHost()


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """Where subagent hosts run and how the supervisor authenticates to its gateway.

    Normally built with `from_env()`. `db_path` is constructor-only (the two
    modes get separate default files so local mode leftovers never shadow real
    MicroVM rows).
    """

    host_mode: HostMode = "local"
    token: str = ""
    db_path: Path | None = None
    namespace: str | None = None  # DO_MICROVM_NAMESPACE: selects the default store file

    def __post_init__(self) -> None:
        if self.host_mode not in _VALID_HOST_MODES:
            raise ValueError(f"{HOST_MODE_ENV} must be one of {_VALID_HOST_MODES}, got {self.host_mode!r}")
        if not self.token:
            raise ValueError("GatewaySettings.token must not be empty")
        object.__setattr__(self, "namespace", validate_namespace(self.namespace))

    @classmethod
    def from_env(cls, *, token_factory: Callable[[], str] | None = None) -> GatewaySettings:
        """Read DO_MICROVM_HOST_MODE and DO_MICROVM_GATEWAY_TOKEN (random when unset).

        Does not load a .env file itself; callers load one first (the examples do,
        and `langgraph dev` honours the `env` key of langgraph.json).
        """
        host_mode = (env_value(HOST_MODE_ENV) or "local").strip().lower() or "local"
        if host_mode not in _VALID_HOST_MODES:
            raise ValueError(f"{HOST_MODE_ENV} must be one of {_VALID_HOST_MODES}, got {host_mode!r}")
        # The embedded gateway listens on 127.0.0.1 inside this process, so a
        # random per-process token protects it as well as a configured one.
        # Host tokens live in the gateway's store and do not depend on it.
        token = (env_value(GATEWAY_TOKEN_ENV) or "").strip() or (token_factory or _random_token)()
        return cls(host_mode=host_mode, token=token, namespace=env_value(NAMESPACE_ENV))  # type: ignore[arg-type]

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path if self.db_path is not None else default_db_path(self.host_mode, self.namespace)


def _random_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


class LazyGateway:
    """A SubAgentGateway (plus its host manager) that exists only once the supervisor runs.

    Construct it at import time next to the deep agent; call `subagent()` for
    each spec and put `middleware()` on the agent. `ensure_started()` is
    idempotent and thread-safe; `close()` tears everything down and is also
    registered with `atexit` on first start.
    """

    def __init__(
        self,
        graphs: Mapping[str, HostSpec],
        *,
        settings: GatewaySettings | None = None,
        port: int | None = None,
        poll_interval: float = 15.0,
        pause_grace: float = 45.0,
        idle_backstop: float = 300.0,
        hosts_factory: HostsFactory | None = None,
        register_atexit: bool = True,
        export_sdk_api_key: bool = True,
        **gateway_kwargs: Any,
    ) -> None:
        """`graphs` maps the name a supervisor uses (`AsyncSubAgent.graph_id`) to the HostSpec serving it.

        `poll_interval`/`pause_grace`/`idle_backstop` and `gateway_kwargs` are
        forwarded to SubAgentGateway; the embedded gateway's own poll loop is
        what turns a finished run into a cached result, so labs that want snappy
        `check` answers pass a small `poll_interval`. `hosts_factory` is the
        test seam for the host manager (default: LocalDevServerHost in local
        mode, MicroVMAgentHost in microvm mode).
        """
        if not graphs:
            raise ValueError("LazyGateway needs at least one graph")
        self.graphs: dict[str, HostSpec] = dict(graphs)
        self.settings = settings if settings is not None else GatewaySettings.from_env()
        self._port = port if port is not None else free_port(LOOPBACK)
        self.url = f"http://{LOOPBACK}:{self._port}"
        self._gateway_kwargs: dict[str, Any] = {
            "poll_interval": poll_interval,
            "pause_grace": pause_grace,
            "idle_backstop": idle_backstop,
            **gateway_kwargs,
        }
        self._hosts_factory: HostsFactory = hosts_factory if hosts_factory is not None else default_hosts_factory
        self._register_atexit = register_atexit
        self._export_sdk_api_key = export_sdk_api_key
        self._lock = threading.Lock()
        self._atexit_registered = False
        self.hosts: HostManager | None = None
        self.gateway: SubAgentGateway | None = None

    # ------------------------------------------------------------------
    # what the supervisor needs at import time
    # ------------------------------------------------------------------

    @property
    def token(self) -> str:
        return self.settings.token

    @property
    def host_mode(self) -> HostMode:
        return self.settings.host_mode

    @property
    def started(self) -> bool:
        return self.gateway is not None

    def subagent(self, *, name: str, description: str, graph_id: str | None = None) -> AsyncSubAgent:
        """An AsyncSubAgent spec for `graph_id` (default: `name`) pointing at this gateway."""
        graph = graph_id if graph_id is not None else name
        if graph not in self.graphs:
            raise ValueError(f"Unknown graph {graph!r}; this gateway serves {sorted(self.graphs)}")
        return async_subagent_spec(
            name=name,
            description=description,
            graph_id=graph,
            gateway_url=self.url,
            gateway_token=self.token,
        )

    def middleware(self, *, name: str = GATEWAY_START_MIDDLEWARE_NAME) -> AgentMiddleware:
        """Middleware that starts the gateway before the agent's first step (sync and async).

        An agent refuses two middleware with one name: give each gateway's its own `name` when
        one agent uses several.
        """
        return GatewayStartMiddleware(self, name=name)

    def artifacts_tool(self, **kwargs: Any) -> BaseTool:
        """The `fetch_subagent_artifacts` tool bound to this gateway; kwargs as for `fetch_artifacts_tool`."""
        return fetch_artifacts_tool(gateway_url=self.url, gateway_token=self.token, **kwargs)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ensure_started(self) -> str:
        """Start the host manager and the embedded gateway if needed; return the gateway URL."""
        with self._lock:
            if self.gateway is None:
                hosts = self._hosts_factory(self.settings.host_mode)
                gateway = SubAgentGateway(
                    hosts,
                    self.graphs,
                    db_path=self.settings.resolved_db_path,
                    token=self.token,
                    **self._gateway_kwargs,
                )
                try:
                    url = gateway.start_embedded(host=LOOPBACK, port=self._port)
                except Exception:
                    gateway.stop()
                    _close_if_possible(hosts)
                    raise
                self.hosts, self.gateway = hosts, gateway
                if self._export_sdk_api_key:
                    export_sdk_api_key(self.token)
                if self._register_atexit and not self._atexit_registered:
                    atexit.register(self.close)
                    self._atexit_registered = True
                logger.info(
                    "Subagent gateway up at %s (%s hosts, graphs %s, store %s)",
                    url,
                    self.settings.host_mode,
                    sorted(self.graphs),
                    self.settings.resolved_db_path,
                )
            return self.url

    def close(self) -> None:
        """Stop the gateway; in local mode also destroy every local host and say so in the store."""
        with self._lock:
            gateway, hosts = self.gateway, self.hosts
            self.gateway = self.hosts = None
        if gateway is None:
            return
        logger.info("Subagent gateway at %s shutting down (%s hosts)", self.url, self.settings.host_mode)
        try:
            if self.settings.host_mode == "local" and hosts is not None:
                _close_if_possible(hosts)
                for row in gateway.store.list_hosts():
                    if row.status not in _RETIRED_HOST_STATUSES:
                        gateway.store.retire_host(row.host_id, "closed")
        finally:
            gateway.stop()

    def __enter__(self) -> LazyGateway:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class GatewayStartMiddleware(AgentMiddleware):
    """Calls `LazyGateway.ensure_started()` before the agent's first step.

    Registered through `create_deep_agent(middleware=[...])`. The async hook
    runs the start in a worker thread: the embedded server's startup waits
    with `time.sleep` and opens SQLite, both of which `langgraph dev`'s
    blocking-call detector would otherwise flag on the event loop.
    """

    def __init__(self, lazy: LazyGateway, *, name: str = GATEWAY_START_MIDDLEWARE_NAME) -> None:
        super().__init__()
        self._lazy = lazy
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._lazy.ensure_started()
        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        await asyncio.to_thread(self._lazy.ensure_started)
        return None


_ARTIFACTS_TOOL_DESCRIPTION = f"""Copy files an async subagent saved under its /{OUTPUTS_DIR}/ directory into your own filesystem.

Files land at {{dest}}/<task_id>/<path> (e.g. {{dest}}/<task_id>/report.md), where you can read_file them.
Omit `paths` to copy every file the task produced, or pass the paths the subagent reported
("/{OUTPUTS_DIR}/report.md" or "report.md"). Works while the task runs and after it finished,
as long as its host still exists. Idle hosts are deleted some time after their last task finished
(sooner when the slot is needed for a new task), so copy the files you need soon after a task
completes; once the host is gone the tool says the files are gone."""


def fetch_artifacts_tool(
    *,
    gateway_url: str,
    gateway_token: str,
    backend: BackendProtocol | None = None,
    dest_dir: str = DEFAULT_ARTIFACTS_DEST,
    max_total_bytes: int = DEFAULT_FETCH_MAX_BYTES,
    timeout: float = 300.0,
    transport: httpx.BaseTransport | None = None,
    name: str = ARTIFACTS_TOOL_NAME,
) -> BaseTool:
    """Build `fetch_subagent_artifacts(task_id, paths=None)` for a supervisor deep agent.

    `backend` is where files are written (default: StateBackend(), which is
    what create_deep_agent uses when given no backend; pass the agent's own
    backend otherwise). Non-UTF-8 files are stored base64-encoded by
    StateBackend. `max_total_bytes` caps one call; larger artifacts are
    listed as skipped. `transport` is a test seam for the HTTP client.
    `name` is the tool's name: an agent with several gateways (and so several of
    these tools) needs one name per gateway, or one tool silently replaces the other.
    """
    target = backend if backend is not None else StateBackend()
    dest = "/" + dest_dir.strip("/")

    def fetch(task_id: str, paths: list[str] | None = None) -> str:
        with httpx.Client(
            base_url=gateway_url,
            headers={"Authorization": f"Bearer {gateway_token}"},
            timeout=timeout,
            transport=transport,
        ) as http:
            return _fetch_artifacts(http, target, dest, task_id, paths, max_total_bytes)

    async def afetch(task_id: str, paths: list[str] | None = None) -> str:
        # A worker thread keeps the blocking HTTP calls off the event loop (langgraph dev's
        # blocking-call detector); StateBackend still finds the run's config there.
        return await asyncio.to_thread(fetch, task_id, paths)

    return StructuredTool.from_function(
        func=fetch,
        coroutine=afetch,
        name=name,
        description=_ARTIFACTS_TOOL_DESCRIPTION.format(dest=dest),
    )


_UNKNOWN_TASK_HINT = (
    "If the agent uses several gateways, give each gateway's artifacts tool its own name "
    "(artifacts_tool(name=...)): of two tools with one name, the agent keeps only one."
)


def _fetch_artifacts(
    http: httpx.Client,
    backend: BackendProtocol,
    dest: str,
    task_id: str,
    paths: list[str] | None,
    max_total_bytes: int,
) -> str:
    task_id = task_id.strip()
    listing = http.get(f"/artifacts/{quote(task_id, safe='')}")
    if listing.status_code == 404 and _detail(listing).startswith("Unknown task"):
        # An agent keeps one tool per name: another gateway's tool of the same name may have
        # replaced the one that knows this task.
        logger.warning("Artifacts asked of a gateway that has no task %s: %s", task_id, _UNKNOWN_TASK_HINT)
        return f"Could not list the files of task {task_id}: this gateway has no such task. {_UNKNOWN_TASK_HINT}"
    if listing.status_code != 200:
        return f"Could not list the files of task {task_id}: {_detail(listing)}"
    body = listing.json()
    available = {str(f["path"]): int(f["size"]) for f in body.get("files", [])}
    if paths:
        wanted: list[str] = []
        bad: list[str] = []
        for path in paths:
            try:
                wanted.append(safe_relative_path(path))
            except ArtifactError:
                bad.append(path)
    else:
        wanted, bad = sorted(available), []
    if not wanted and not bad:
        return f"Task {task_id} has no files under /{OUTPUTS_DIR}/."

    uploads: list[tuple[str, bytes]] = []
    notes: list[str] = [f"{p}: not a path inside /{OUTPUTS_DIR}/" for p in bad]
    total = 0
    for rel in wanted:
        if rel not in available:
            notes.append(f"{rel}: not found")
            continue
        if total + available[rel] > max_total_bytes:
            notes.append(f"{rel}: skipped, {available[rel]} bytes would exceed this call's {max_total_bytes}-byte limit")
            continue
        response = http.get(f"/artifacts/{quote(task_id, safe='')}/files/{quote(rel)}")
        if response.status_code != 200:
            notes.append(f"{rel}: {_detail(response)}")
            continue
        total += len(response.content)
        uploads.append((f"{dest}/{task_id}/{rel}", response.content))

    lines: list[str] = []
    if uploads:
        lines.append(f"Copied {len(uploads)} file(s) from task {task_id}:")
        for (path, content), result in zip(uploads, backend.upload_files(uploads), strict=True):
            if result.error:
                lines.append(f"- {path}: could not write ({result.error})")
                continue
            try:
                content.decode("utf-8")
                kind = ""
            except UnicodeDecodeError:
                kind = ", binary: stored base64-encoded"
            lines.append(f"- {path} ({len(content)} bytes{kind})")
    if body.get("truncated"):
        lines.append(f"The listing was truncated; only the first {len(available)} files were considered.")
    if notes:
        lines.append("Not copied:")
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return f"{detail or response.text[:200]} (HTTP {response.status_code})"


def export_sdk_api_key(token: str) -> None:
    """Make the LangGraph SDK's mandatory `x-api-key` header carry the gateway token.

    The SDK reads LANGGRAPH_API_KEY, then LANGSMITH_API_KEY, then
    LANGCHAIN_API_KEY when a client is built. Setting the first to the gateway
    token stops the LangSmith key from being sent to the gateway.
    An existing different value (another gateway's token, a LangSmith key) is
    left alone: the gateway checks the `Authorization: Bearer` header every
    spec carries first, so requests authenticate either way.
    """
    current = os.environ.get(SDK_API_KEY_ENV)
    if current is None or current == "":
        os.environ[SDK_API_KEY_ENV] = token
        return
    if current != token:
        logger.debug("%s already holds another value; this gateway authenticates by its Bearer header", SDK_API_KEY_ENV)


def _close_if_possible(hosts: HostManager) -> None:
    close = getattr(hosts, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.warning("Closing the host manager failed", exc_info=True)
