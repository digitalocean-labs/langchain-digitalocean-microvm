"""SubAgentGateway: one stable Agent Protocol URL in front of many agent hosts.

The deepagents async-subagent middleware wants a fixed `url` per spec, caches
one client per URL, and makes six calls: create thread, create run, get run,
get thread, list runs, cancel run. It cannot notice a host that died (a failing
`runs.get` leaves the task `running` forever) and it never resumes anything.
The gateway sits between the supervisor and the MicroVMs to fix exactly
that (see docs/design.md):

- `POST /threads` is answered locally: mint a thread id, remember nothing
  else yet (`assistant_id` only arrives with the first run).
- `POST /threads/{tid}/runs` allocates a host for the spec (`per-task`: one
  host per thread; `pooled`: one shared host per graph), creates the thread
  ON the host with the SAME id, injects a completion webhook and forwards the
  run. The supervisor's task id therefore equals the host's thread id.
- `GET .../runs/{rid}` and `GET /threads/{tid}` are served from the cache
  once a run is terminal, so a `check` never wakes a paused host. Before
  that they are proxied; an unreachable, destroyed or restarted host turns
  into a synthetic `{"status": "error"}` that the middleware understands.
  A run that paused for a human (an `interrupt()`) ends `success`; the
  cached thread keeps its `interrupted` status and its interrupts, which is
  how the middleware tells such a task (`waiting`) from a finished one.
- `GET /threads/{tid}/runs` lists the latest run only (the gateway tracks no
  other): the middleware asks for it when a run another client started on
  the task's thread (an approver resuming it) replaced the one it tracks.
- `cancel` is answered locally for terminal runs (the Agent Server returns
  404 for those) and proxied otherwise.
- `tick()` is the periodic maintenance step: poll active runs whose webhook
  never arrived, pause hosts that have been idle for `pause_grace` seconds
  (never a BUSY one), and recover anomalies. On real MicroVMs that
  poll is also what keeps a busy host awake: they auto-pause after their
  idle timeout without INBOUND requests, whatever the workload is doing, so
  the gateway refuses a `poll_interval` that is not well below it.
- `tick()` also collects garbage: a FAILED host is deleted on the next
  tick, and a READY/PAUSED host with no active run is deleted once it has
  been unused for its time-to-live (`release_after` for per-task hosts,
  `paused_ttl` for every host). The store keeps the row as `destroyed` with
  a `retired_reason`, so `check` on its tasks is still answered from the
  cache and an artifact fetch gets a 410 that says why the files are gone.
  The TTL is also the artifact retention window: it counts from the host's
  last activity (run finished, file fetched, update), so files stay
  fetchable at least that long unless the slot is needed (below).
- `GET /artifacts/{tid}` lists the files a task left in its host's per-task
  outputs directory and `GET /artifacts/{tid}/files/{path}` returns one, read
  from the host's Agent Server over its endpoint (artifacts.py; the routes
  are in images/agent-host/artifacts_app.py). A paused host is resumed for
  it (and paused again by `tick()`); a host that was deleted, evicted or
  replaced since the task ran answers 410: its files are gone, they only
  ever lived inside that host.
- Real MicroVMs also need DigitalOcean's edge credential on every request
  (`hosts.edge_headers()`, an access-only token) next to the host token, and
  a team's MicroVM limit can be reached: when creating a host fails because
  the team is at its limit (HostCapacityError), the gateway deletes one idle
  host whose task results are already cached and tries once more. A host is
  only evicted once it has been unused for `eviction_grace` seconds, so a
  supervisor that just saw a task finish can still fetch its files; inside
  that window the launch fails with a message that says when to retry.

This module holds the routes and the calls to the hosts' Agent Servers; host
allocation, claims, pause, garbage collection, eviction and startup
reconciliation are in lifecycle.py (`HostLifecycle`, a base class of the gateway).

Runs embedded (uvicorn on a loopback port inside the supervisor process, see
`start_embedded`) or as a standalone ASGI app (`gateway.app`).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from langchain_digitalocean_microvm.agent_host.artifacts import (
    DEFAULT_MAX_FILES,
    MARKER_HEADER,
    OUTPUTS_DIR,
    ArtifactError,
    ArtifactListing,
    ArtifactUpstreamError,
    HostGet,
    list_artifacts,
    read_artifact,
)
from langchain_digitalocean_microvm.agent_host.host import utc_now_iso
from langchain_digitalocean_microvm.agent_host.lifecycle import (
    _DEAD_HOST_STATUSES,
    ACTIVE_RUN_STATUSES,
    HostLifecycle,
    HostManager,
    _HostRetired,
    _retired_text,
)
from langchain_digitalocean_microvm.agent_host.models import HostSpec, HostStatus
from langchain_digitalocean_microvm.agent_host.store import (
    TERMINAL_RUN_STATUSES,
    GatewayStore,
    HostRow,
    ThreadRow,
)
from langchain_digitalocean_microvm.exceptions import (
    SandboxError,
    SandboxNotFoundError,
)

logger = logging.getLogger(__name__)

HOOK_PATH = "/hooks/run-done"
_UNKNOWN = object()  # _note_unreachable: inventory status not looked up yet
DEFAULT_ARTIFACT_MAX_BYTES = 10 * 1024 * 1024  # per file: three Range pages at most
DEFAULT_ARTIFACT_PAGE_BYTES = 4 * 1024 * 1024  # well inside the endpoint's ~15 s per response
# Upstream answers retried before a host counts as unreachable: a host that paused itself at its
# idle timeout can answer the first request with a 503 while it resumes.
_RETRY_STATUSES = frozenset({502, 503, 504})
# ArtifactError.reason -> HTTP status of the artifact routes
_ARTIFACT_ERROR_STATUS = {"not_found": 404, "too_large": 413, "bad_path": 400, "bad_range": 502}


def webhook_policy_env(*, allow_loopback: bool = False, allowed_domains: list[str] | None = None) -> dict[str, str]:
    """Env for a host that must accept the gateway's plain-http webhook URL.

    The Agent Server's webhook policy (langgraph_api.webhook) requires https
    and denies loopback targets by default; private IPs are allowed. Put the
    result into `HostSpec.env` when the gateway runs with a `webhook_base_url`.
    `allow_loopback=True` is for local mode (gateway and hosts on 127.0.0.1).
    """
    policy: dict[str, Any] = {"require_https": False}
    if allow_loopback:
        policy["disable_loopback"] = False
    if allowed_domains:
        policy["allowed_domains"] = list(allowed_domains)
    return {"LANGGRAPH_WEBHOOKS": json.dumps({"url": policy})}



class _UpstreamDown(Exception):
    """The host did not answer (connection error, timeout, 5xx)."""


@dataclass
class GatewayStatus:
    url: str | None
    hosts: list[dict[str, Any]]
    threads: list[dict[str, Any]]


def _paused_state(thread: Any) -> dict[str, Any] | None:
    """A finished run's thread state worth keeping: its status and interrupts when it waits for a human, else None."""
    if not isinstance(thread, dict) or thread.get("status") != "interrupted":
        return None
    interrupts = thread.get("interrupts")
    return {"status": "interrupted", "interrupts": interrupts if isinstance(interrupts, dict) else {}}


class SubAgentGateway(HostLifecycle):
    def __init__(
        self,
        hosts: HostManager,
        graphs: Mapping[str, HostSpec],
        *,
        db_path: Path | str,
        token: str,
        webhook_base_url: str | None = None,
        pause_grace: float = 45.0,
        idle_backstop: float = 300.0,
        poll_interval: float = 15.0,
        release_after: float | None = 3600.0,
        paused_ttl: float | None = 86400.0,
        eviction_grace: float = 60.0,
        retire_wait: float = 60.0,
        unreachable_threshold: int = 3,
        upstream_timeout: float = 30.0,
        upstream_retries: int = 2,
        upstream_retry_delay: float = 1.0,
        artifact_max_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES,
        artifact_max_files: int = DEFAULT_MAX_FILES,
        artifact_page_bytes: int = DEFAULT_ARTIFACT_PAGE_BYTES,
        upstream_transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        iso_now: Callable[[], str] = utc_now_iso,
    ) -> None:
        """Build a gateway.

        `graphs` maps the `assistant_id` a supervisor uses (the AsyncSubAgent
        `graph_id`) to the HostSpec that serves it. `webhook_base_url` is the
        URL hosts can reach the gateway on for completion callbacks. None
        (default) disables webhooks and relies on `tick()` polling, which is
        mandatory anyway; "embedded" means the embedded server's own URL.
        The Agent Server validates webhook URLs (https required and loopback
        denied by default) so only enable this for hosts whose env carries a
        matching LANGGRAPH_WEBHOOKS policy: see `webhook_policy_env()`.
        Webhooks only make sense for a gateway the hosts can reach: a
        standalone deployment with a public (or VPC) https URL, or local mode.
        An embedded gateway on a laptop is unreachable from real MicroVMs, so
        it relies on polling, which also keeps busy MicroVMs from auto-pausing.
        `release_after` (per-task hosts) and `paused_ttl` (every host) are
        the seconds a host with no active run may stay unused before `tick()`
        deletes it (None: never); `eviction_grace` is how long an unused host
        is safe from eviction at the team's MicroVM limit. Both windows are
        how long a finished task's files stay fetchable. `retire_wait` is how
        long a launch waits for another process (scripts/gc_microvms.py
        sharing the store) to finish deleting a host it needs before it
        deletes that host itself.
        `artifact_max_bytes` caps one artifact download (413 above it),
        `artifact_max_files` one listing; files are read in `artifact_page_bytes`
        Range pages (halved when one does not arrive in time). A 502/503/504 from a host is
        retried `upstream_retries` times, `upstream_retry_delay` seconds apart, before the
        host counts as unreachable; a run creation only when the answer did not come from
        the Agent Server (which answers JSON), so a run is never started twice. `upstream_transport`, `clock`,
        `iso_now` are test seams.
        """
        self.hosts = hosts
        self.graphs: dict[str, HostSpec] = dict(graphs)
        self.store = GatewayStore(db_path)
        self.token = token
        self.clock = clock
        self._init_lifecycle(
            pause_grace=pause_grace,
            idle_backstop=idle_backstop,
            release_after=release_after,
            paused_ttl=paused_ttl,
            eviction_grace=eviction_grace,
            retire_wait=retire_wait,
        )
        self.poll_interval = poll_interval
        self.unreachable_threshold = unreachable_threshold
        self._upstream_timeout = upstream_timeout
        self.upstream_retries = upstream_retries
        self.upstream_retry_delay = upstream_retry_delay
        self.artifact_max_bytes = artifact_max_bytes
        self.artifact_max_files = artifact_max_files
        self.artifact_page_bytes = artifact_page_bytes
        self._upstream_transport = upstream_transport
        self.iso_now = iso_now

        self._webhook_base: str | None = webhook_base_url
        self._hook_secret = secrets.token_urlsafe(24)
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

        self._edge_headers: dict[str, str] = dict(getattr(hosts, "edge_headers", dict)())
        idle = getattr(hosts, "idle_timeout_seconds", None)
        if isinstance(idle, (int, float)) and poll_interval * 3 > idle:
            raise ValueError(
                f"poll_interval={poll_interval}s is too long for hosts that pause after {idle:g}s without "
                "inbound requests: active runs would be frozen between polls. Use at most a third of it."
            )
        self._remember_tokens()
        self._reconcile_with_inventory()

        self.app = Starlette(
            routes=[
                Route("/ok", self._ok, methods=["GET"]),
                Route("/threads", self._create_thread, methods=["POST"]),
                Route("/threads/{tid}/runs", self._create_run, methods=["POST"]),
                Route("/threads/{tid}/runs", self._list_runs, methods=["GET"]),
                Route("/threads/{tid}/runs/{rid}", self._get_run, methods=["GET"]),
                Route("/threads/{tid}/runs/{rid}/cancel", self._cancel_run, methods=["POST"]),
                Route("/threads/{tid}", self._get_thread, methods=["GET"]),
                Route("/artifacts/{tid}", self._list_artifacts, methods=["GET"]),
                Route("/artifacts/{tid}/files/{path:path}", self._get_artifact, methods=["GET"]),
                Route(HOOK_PATH, self._run_done_hook, methods=["POST"]),
                Route("/gateway/status", self._status, methods=["GET"]),
            ],
            lifespan=self._lifespan,
        )

    # ------------------------------------------------------------------
    # embedded runner
    # ------------------------------------------------------------------

    def start_embedded(self, *, host: str = "127.0.0.1", port: int | None = None, startup_timeout: float = 30.0) -> str:
        """Serve `self.app` with uvicorn on a background thread; return its URL."""
        import uvicorn  # noqa: PLC0415 - only needed for embedded mode

        if self._server is not None:
            assert self.url is not None
            return self.url
        if port is None:
            from langchain_digitalocean_microvm.agent_host.local import (
                free_port,  # noqa: PLC0415
            )

            port = free_port(host)
        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning", lifespan="on")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._serve_embedded, name="subagent-gateway", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + startup_timeout
        while not self._server.started:
            if not self._thread.is_alive():
                raise SandboxError("Embedded gateway server exited during startup")
            if time.monotonic() > deadline:
                raise SandboxError("Embedded gateway server did not start in time")
            time.sleep(0.02)
        self.url = f"http://{host}:{port}"
        if self._webhook_base == "embedded":
            self._webhook_base = self.url
        return self.url

    def _serve_embedded(self) -> None:
        """Thread target: run uvicorn, exempt from the host process's blocking-call detector.

        `langgraph dev` activates blockbuster, which raises on sqlite3 (and
        other blocking) calls made from ANY thread that has a running event
        loop -- including this one. The gateway's SQLite bookkeeping is
        synchronous by design (store.py), so opt this thread out via
        blockbuster's own skip flag: a context variable up to 1.5.28 (tasks
        created on this loop inherit it), a thread-local flag from 1.5.29 (the
        loop runs on this thread).
        """
        try:
            from blockbuster import (
                blockbuster as bb,  # noqa: PLC0415 - optional dev dependency
            )
        except ImportError:
            pass
        else:
            context_flag = getattr(bb, "blockbuster_skip", None)
            thread_flag = getattr(bb, "_skip", None)
            if context_flag is not None:
                context_flag.set(True)
            elif thread_flag is not None:
                thread_flag.value = True
        assert self._server is not None
        self._server.run()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=15)
            self._server = None
            self._thread = None
        self.store.close()

    def tick_sync(self, *, timeout: float = 60.0) -> None:
        """Run one maintenance tick on the gateway's own event loop (tests, cron)."""
        if self._loop is None:
            raise SandboxError("Gateway is not running; call start_embedded() first")
        future = asyncio.run_coroutine_threadsafe(self.tick(), self._loop)
        future.result(timeout)

    def status(self) -> GatewayStatus:
        return GatewayStatus(
            url=self.url,
            hosts=[{k: v for k, v in vars(h).items() if k != "token"} for h in self.store.list_hosts()],  # never expose host tokens
            threads=[
                {k: v for k, v in vars(t).items() if k not in ("cached_run", "cached_values")}
                for t in self.store.list_threads()
            ],
        )

    # ------------------------------------------------------------------
    # lifespan / background polling
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _lifespan(self, app: Starlette):
        self._loop = asyncio.get_running_loop()
        self._client = httpx.AsyncClient(transport=self._upstream_transport, timeout=self._upstream_timeout)
        self._poll_task = asyncio.create_task(self._poll_loop())
        try:
            yield
        finally:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutting down
                pass
            await self._client.aclose()
            self._client = None
            self._loop = None

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 - the loop must survive one bad tick
                logger.exception("Gateway tick failed")

    async def tick(self) -> None:
        """Delete failed and expired hosts, poll active runs (webhook fallback), pause hosts idle past the grace."""
        await self._collect_garbage()
        for thread in self.store.list_threads():
            if thread.run_status in ACTIVE_RUN_STATUSES and thread.host_id:
                host = self.store.get_host(thread.host_id)
                if host is None:
                    continue
                if host.status == HostStatus.PAUSED.value:
                    # Anomaly: a paused host with an active run. Wake it so the
                    # run can finish or be reported.
                    try:
                        await self._resume_host(host, self._spec_for(thread))
                    except SandboxError:
                        logger.warning("Could not resume paused host %s with active run", host.host_id, exc_info=True)
                        continue
                    host = self.store.get_host(thread.host_id) or host
                if host.status in (HostStatus.READY.value, HostStatus.BUSY.value):
                    await self._refresh_run(thread, host)


        await self._pause_idle_hosts()

    # ------------------------------------------------------------------
    # route handlers
    # ------------------------------------------------------------------

    async def _ok(self, request: Request) -> Response:
        return JSONResponse({"ok": True})

    async def _status(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        return JSONResponse(vars(self.status()))

    async def _create_thread(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        body = await _json_body(request)
        thread_id = str(body.get("thread_id") or uuid.uuid4())
        if self.store.get_thread(thread_id) is not None:
            if body.get("if_exists") == "do_nothing":
                return JSONResponse(self._thread_payload(thread_id))
            return JSONResponse({"detail": "Thread already exists"}, status_code=409)
        self.store.create_thread(thread_id, now=self.clock())
        return JSONResponse(self._thread_payload(thread_id))

    async def _create_run(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        thread_id = request.path_params["tid"]
        # One run creation per thread at a time. With multitask_strategy="interrupt"
        # the run created LAST wins on the host, so the store must record runs in
        # creation order; two concurrent updates (parallel tool calls) whose
        # responses arrive out of order would otherwise leave the gateway tracking
        # the run that was interrupted.
        async with self._thread_lock_for(thread_id):
            return await self._create_run_locked(request, thread_id)

    async def _create_run_locked(self, request: Request, thread_id: str) -> Response:
        thread = self.store.get_thread(thread_id)
        if thread is None:
            return JSONResponse({"detail": "Thread not found"}, status_code=404)
        body = await _json_body(request)
        assistant = str(body.get("assistant_id") or "")
        spec = self.graphs.get(assistant)
        if spec is None:
            return JSONResponse(
                {"detail": f"Unknown assistant_id {assistant!r}; known: {sorted(self.graphs)}"},
                status_code=404,
            )
        if thread.graph_id and thread.graph_id != assistant:
            return JSONResponse(
                {"detail": f"Thread {thread_id} belongs to assistant {thread.graph_id!r}, not {assistant!r}"},
                status_code=409,
            )
        try:
            host = await self._ensure_host_for_thread(thread, assistant, spec)
        except SandboxError as exc:
            return JSONResponse({"detail": f"Could not provision agent host: {exc}"}, status_code=503)

        forwarded = dict(body)
        forwarded["assistant_id"] = spec.graph_id
        if self._webhook_base:
            forwarded["webhook"] = f"{self._webhook_base}{HOOK_PATH}?secret={self._hook_secret}"
        try:
            if thread.host_id is None:
                await self._upstream_create_thread(host, thread_id)
            response = await self._upstream(host, "POST", f"/threads/{thread_id}/runs", json=forwarded)
        except _UpstreamDown as exc:
            await self._note_unreachable(thread, host, reason=str(exc))
            return JSONResponse({"detail": f"Agent host unreachable: {exc}"}, status_code=502)
        if response.status_code >= 400:
            return Response(response.content, status_code=response.status_code, media_type="application/json")
        run = response.json()
        now = self.clock()
        self.store.update_thread(
            thread_id,
            graph_id=assistant,
            host_id=host.host_id,
            run_id=str(run.get("run_id", "")),
            run_status=str(run.get("status", "pending")),
            run_started_at=self.iso_now(),
            host_microvm_id=host.microvm_id,
            cached_run=None,
            cached_values=None,
            cached_thread=None,
            last_active_at=now,
        )
        self.store.update_host(host.host_id, status=HostStatus.BUSY.value, idle_since=None, last_active_at=now, failures=0)
        return JSONResponse(run)

    async def _get_run(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        thread_id, run_id = request.path_params["tid"], request.path_params["rid"]
        thread = self.store.get_thread(thread_id)
        if thread is None or thread.run_id is None:
            return JSONResponse({"detail": "Run not found"}, status_code=404)
        if thread.run_id != run_id:
            # An older run on this thread (before an update). We only track the
            # latest; report it as superseded rather than pretend it is running.
            return JSONResponse(self._synthetic_run(thread, run_id=run_id, status="interrupted"))
        return JSONResponse(await self._latest_run(thread))

    async def _list_runs(self, request: Request) -> Response:
        """The thread's runs as far as the gateway knows them: only the latest (newest first, as the Agent Server lists them).

        The async-subagent middleware asks for it when its tracked run was replaced by a run another client started on the
        task's thread (an approver resuming it), to follow the newest one.
        """
        if not self._authorized(request):
            return _unauthorized()
        thread = self.store.get_thread(request.path_params["tid"])
        if thread is None:
            return JSONResponse({"detail": "Thread not found"}, status_code=404)
        if thread.run_id is None:
            return JSONResponse([])
        run = await self._latest_run(thread)
        wanted = request.query_params.get("status")
        return JSONResponse([] if wanted and run.get("status") != wanted else [run])

    async def _latest_run(self, thread: ThreadRow) -> dict[str, Any]:
        """The thread's latest run: from the cache once it is over, else read from its host."""
        if thread.terminal and thread.cached_run is not None and thread.cached_run.get("run_id") == thread.run_id:
            return thread.cached_run
        host = self.store.get_host(thread.host_id or "")
        if host is None:
            return self._fail_thread(thread, "agent host record missing")
        if host.status in _DEAD_HOST_STATUSES:
            return self._fail_thread(thread, f"agent host {host.host_id} is {host.status}")
        if host.status == HostStatus.PAUSED.value:
            # Should not happen (never pause BUSY); report last known status
            # honestly and let tick() wake the host.
            return self._synthetic_run(thread, status=thread.run_status or "running")
        self.store.update_thread(thread.thread_id, last_active_at=self.clock())
        return await self._refresh_run(thread, host)

    async def _get_thread(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        thread_id = request.path_params["tid"]
        thread = self.store.get_thread(thread_id)
        if thread is None:
            return JSONResponse({"detail": "Thread not found"}, status_code=404)
        if thread.cached_values is not None:
            return JSONResponse(self._thread_payload(thread_id, values=thread.cached_values, state=thread.cached_thread))
        if thread.host_id is None:
            return JSONResponse(self._thread_payload(thread_id, values={}))
        host = self.store.get_host(thread.host_id)
        if host is None or host.status in _DEAD_HOST_STATUSES:
            return JSONResponse({"detail": "Agent host is gone"}, status_code=404)
        if host.status == HostStatus.PAUSED.value:
            try:
                host = await self._resume_host(host, self._spec_for(thread))
            except SandboxError as exc:
                return JSONResponse({"detail": f"Could not resume agent host: {exc}"}, status_code=503)
        try:
            response = await self._upstream(host, "GET", f"/threads/{thread_id}")
        except _UpstreamDown as exc:
            return JSONResponse({"detail": f"Agent host unreachable: {exc}"}, status_code=502)
        return Response(response.content, status_code=response.status_code, media_type="application/json")

    async def _cancel_run(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        thread_id, run_id = request.path_params["tid"], request.path_params["rid"]
        thread = self.store.get_thread(thread_id)
        if thread is None or thread.run_id is None:
            return JSONResponse({"detail": "Run not found"}, status_code=404)
        if thread.run_id != run_id or thread.terminal:
            return JSONResponse({})  # nothing running under that id: cancelled as far as anyone can tell
        host = self.store.get_host(thread.host_id or "")
        if host is None or host.status != HostStatus.READY.value and host.status != HostStatus.BUSY.value:
            self._mark_terminal(thread, self._synthetic_run(thread, status="interrupted"), values=None)
            return JSONResponse({})
        try:
            response = await self._upstream(host, "POST", f"/threads/{thread_id}/runs/{run_id}/cancel")
        except _UpstreamDown as exc:
            await self._note_unreachable(thread, host, reason=str(exc))
            return JSONResponse({})
        if response.status_code == 404 or response.status_code < 400:
            self._mark_terminal(thread, self._synthetic_run(thread, status="interrupted"), values=None)
            return JSONResponse({})
        return Response(response.content, status_code=response.status_code, media_type="application/json")

    async def _run_done_hook(self, request: Request) -> Response:
        secret = request.query_params.get("secret", "")
        if not hmac.compare_digest(secret, self._hook_secret):
            return _unauthorized()
        payload = await _json_body(request)
        thread_id = str(payload.get("thread_id") or "")
        run_id = str(payload.get("run_id") or "")
        thread = self.store.get_thread(thread_id)
        if thread is None or thread.run_id != run_id:
            return JSONResponse({"ignored": True})
        status = str(payload.get("status") or "")
        if status not in TERMINAL_RUN_STATUSES:
            return JSONResponse({"ignored": True})
        run = {k: v for k, v in payload.items() if k not in ("values", "webhook_sent_at")}
        values, state = payload.get("values"), None
        if thread.host_id:  # the thread itself: a run that paused for a human ends `success`, only its thread says so
            host = self.store.get_host(thread.host_id)
            if host is not None:
                try:
                    response = await self._upstream(host, "GET", f"/threads/{thread_id}")
                    if response.status_code < 400:
                        body = response.json()
                        values = body.get("values") if values is None else values
                        state = _paused_state(body)
                except _UpstreamDown:
                    pass
        self._mark_terminal(thread, run, values=values if isinstance(values, dict) else None, state=state)
        return JSONResponse({"ok": True})

    async def _list_artifacts(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        thread_id = request.path_params["tid"]
        max_files = self.artifact_max_files

        async def op(get: HostGet) -> ArtifactListing:
            return await list_artifacts(get, thread_id, max_files=max_files)

        result = await self._with_task_host(thread_id, op)
        if isinstance(result, Response):
            return result
        return JSONResponse(
            {
                "task_id": thread_id,
                "directory": f"/{OUTPUTS_DIR}",
                "exists": result.exists,
                "files": [{"path": f.path, "size": f.size} for f in result.files],
                "truncated": result.truncated,
                "max_file_bytes": self.artifact_max_bytes,
            }
        )

    async def _get_artifact(self, request: Request) -> Response:
        if not self._authorized(request):
            return _unauthorized()
        thread_id, path = request.path_params["tid"], request.path_params["path"]
        max_bytes, page_bytes = self.artifact_max_bytes, self.artifact_page_bytes

        async def op(get: HostGet) -> bytes:
            return await read_artifact(get, thread_id, path, max_bytes=max_bytes, page_bytes=page_bytes)

        result = await self._with_task_host(thread_id, op)
        if isinstance(result, Response):
            return result
        return Response(result, media_type="application/octet-stream")

    async def _with_task_host(self, thread_id: str, op: Callable[[HostGet], Any]) -> Any:
        """Run `op(get)` against the host that ran `thread_id` (`get` reads from its endpoint); errors become Responses.

        Holds the host's lock (so `tick()` cannot pause it mid-read) and
        resumes a paused host first; `tick()` pauses it again after the grace.
        """
        thread = self.store.get_thread(thread_id)
        if thread is None:
            return JSONResponse({"detail": f"Unknown task {thread_id}"}, status_code=404)
        if thread.host_id is None:
            return JSONResponse(
                {"detail": f"Task {thread_id} has not started a run, so it has no artifacts"}, status_code=404
            )
        async with self._lock_for(thread.host_id):
            host = self.store.get_host(thread.host_id)
            gone = self._artifacts_gone(thread, host)
            if gone is not None:
                return JSONResponse({"detail": gone, "reason": "host_gone"}, status_code=410)
            assert host is not None
            if host.status == HostStatus.PAUSED.value:
                try:
                    host = await self._resume_host_locked(host, self._spec_for(thread))
                except _HostRetired as exc:
                    return JSONResponse({"detail": self._artifacts_gone(thread, exc.row), "reason": "host_gone"},
                                        status_code=410)
                except SandboxNotFoundError:
                    return self._artifacts_host_vanished(thread, host)
                except SandboxError as exc:
                    return JSONResponse({"detail": f"Could not resume agent host: {exc}"}, status_code=503)
            reading = host

            async def get(path: str, headers: Mapping[str, str]) -> httpx.Response:
                assert self._client is not None, "gateway not started"
                attempt = 0
                while True:
                    try:
                        response = await self._client.get(
                            f"{reading.url}{path}", headers={**self._edge_headers, "x-api-key": reading.token, **headers}
                        )
                    except httpx.TransportError as exc:
                        raise ArtifactUpstreamError(f"GET {path}: {exc.__class__.__name__}") from exc
                    if (response.status_code not in _RETRY_STATUSES or MARKER_HEADER in response.headers
                            or attempt >= self.upstream_retries):
                        break
                    attempt += 1
                    await asyncio.sleep(self.upstream_retry_delay)
                if response.status_code >= 500 and MARKER_HEADER not in response.headers:
                    raise ArtifactUpstreamError(f"GET {path}: HTTP {response.status_code}")
                return response

            try:
                result = await op(get)
            except ArtifactError as exc:
                if exc.reason == "not_served":
                    return await self._artifacts_not_served(thread, host, exc)
                return JSONResponse(
                    {"detail": str(exc), "reason": exc.reason}, status_code=_ARTIFACT_ERROR_STATUS.get(exc.reason, 403)
                )
            except ArtifactUpstreamError as exc:
                live = await self._live_status(host.host_id)
                if live is None or live in _DEAD_HOST_STATUSES:
                    return self._artifacts_host_vanished(thread, host)
                return JSONResponse({"detail": f"Could not read files from agent host: {exc}"}, status_code=502)
            self.store.update_host(host.host_id, last_active_at=self.clock())
            return result

    async def _artifacts_not_served(self, thread: ThreadRow, host: HostRow, exc: ArtifactError) -> Response:
        """An answer without the artifact routes' marker: the host is gone (edge 404) or its image lacks the routes."""
        live = await self._live_status(host.host_id)
        if live is None or live in _DEAD_HOST_STATUSES:
            return self._artifacts_host_vanished(thread, host)
        if exc.status == 404:
            detail = ("The agent host does not serve artifacts: its image predates the artifact routes "
                      "(rebuild it with images/agent-host/build_image.py)")
        elif exc.status in (401, 403):
            detail = f"The agent host rejected the gateway's credentials (HTTP {exc.status})"
        else:
            detail = f"Unexpected answer from the agent host: HTTP {exc.status}"
        return JSONResponse({"detail": detail, "reason": "not_served"}, status_code=502)

    def _artifacts_host_vanished(self, thread: ThreadRow, host: HostRow) -> Response:
        """The host manager no longer knows the host (deleted behind the gateway's back): record it, answer 410."""
        self.store.retire_host(host.host_id, "vanished")
        return JSONResponse({"detail": self._artifacts_gone(thread, None), "reason": "host_gone"}, status_code=410)

    @staticmethod
    def _artifacts_gone(thread: ThreadRow, host: HostRow | None) -> str | None:
        """Why the files of `thread` no longer exist, or None if its host is still the one that ran it."""
        prefix = f"The artifacts of task {thread.thread_id} are gone"
        if host is None:
            return f"{prefix}: its agent host {thread.host_id} no longer exists. Files live only inside the host."
        if host.status in _DEAD_HOST_STATUSES:
            return f"{prefix}: its agent host {host.host_id} {_retired_text(host)}. Files live only inside the host."
        if thread.host_microvm_id and host.microvm_id and host.microvm_id != thread.host_microvm_id:
            return (
                f"{prefix}: its agent host {host.host_id} was replaced by a new MicroVM after the task ran. "
                "Files live only inside the host."
            )
        return None

    # ------------------------------------------------------------------
    # run-creation locks
    # ------------------------------------------------------------------

    def _thread_lock_for(self, thread_id: str) -> asyncio.Lock:
        """Per-thread lock for run creation; separate from host locks (a per-task host id IS the thread id)."""
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = self._thread_locks[thread_id] = asyncio.Lock()
        return lock

    # ------------------------------------------------------------------
    # upstream (host) calls
    # ------------------------------------------------------------------

    async def _upstream(self, host: HostRow, method: str, path: str, *, json: Any | None = None) -> httpx.Response:
        assert self._client is not None, "gateway not started"
        creates_run = method == "POST" and path.endswith("/runs")
        attempt = 0
        while True:
            try:
                response = await self._client.request(
                    method, f"{host.url}{path}", json=json, headers={**self._edge_headers, "x-api-key": host.token}
                )
            except httpx.TransportError as exc:
                raise _UpstreamDown(f"{method} {path}: {exc.__class__.__name__}") from exc
            if response.status_code not in _RETRY_STATUSES or attempt >= self.upstream_retries:
                break
            if creates_run and _from_agent_server(response):
                break  # the server saw the run: retrying could start it twice
            attempt += 1
            logger.info("Host %s answered %s %s with HTTP %d; retrying in %gs", host.host_id, method, path,
                        response.status_code, self.upstream_retry_delay)
            await asyncio.sleep(self.upstream_retry_delay)
        if response.status_code >= 500:
            raise _UpstreamDown(f"{method} {path}: HTTP {response.status_code}")
        if response.status_code in (401, 403):
            # Our credentials are not accepted: the host's token is wrong, or the host is gone
            # (a deleted MicroVM's endpoint may answer 403 or 404; the 404 case is handled in
            # _refresh_run). Both make the host unusable; _note_unreachable decides
            # via the inventory.
            raise _UpstreamDown(f"{method} {path}: HTTP {response.status_code} {response.text[:80]!r}")
        return response

    async def _upstream_create_thread(self, host: HostRow, thread_id: str) -> None:
        response = await self._upstream(host, "POST", "/threads", json={"thread_id": thread_id, "if_exists": "do_nothing"})
        if response.status_code >= 400 and response.status_code != 409:
            raise _UpstreamDown(f"POST /threads: HTTP {response.status_code} {response.text[:200]}")

    async def _refresh_run(self, thread: ThreadRow, host: HostRow) -> dict[str, Any]:
        """Fetch the live run from the host; cache it if terminal; convert host death into an error."""
        if thread.run_started_at and host.booted_at and host.booted_at > thread.run_started_at:
            return self._fail_thread(thread, f"agent host {host.host_id} restarted after the run began; the run was orphaned")
        try:
            response = await self._upstream(host, "GET", f"/threads/{thread.thread_id}/runs/{thread.run_id}")
        except _UpstreamDown as exc:
            return await self._note_unreachable(thread, host, reason=str(exc))
        if (current := self._superseding(thread)) is not None:
            # `update` started a new run while this request was in flight (the
            # poll loop and the routes interleave at every await): what came
            # back describes the old run and must not land on the new one.
            return self._synthetic_run(current, status=current.run_status or "running")
        if response.status_code == 404:
            # A MicroVM deleted outside this gateway answers 404 at DigitalOcean's edge, which
            # looks like the Agent Server's "run not found": the inventory tells them apart.
            live = await self._live_status(host.host_id)
            if (current := self._superseding(thread)) is not None:  # same race as above, across this await
                return self._synthetic_run(current, status=current.run_status or "running")
            if live is None or live in _DEAD_HOST_STATUSES:
                return await self._note_unreachable(thread, host, reason="HTTP 404", live=live)
            return self._fail_thread(thread, f"run {thread.run_id} no longer exists on agent host {host.host_id}")
        if response.status_code >= 400:
            return self._synthetic_run(thread, status=thread.run_status or "running")
        run = response.json()
        if host.failures:
            self.store.update_host(host.host_id, failures=0)
        status = str(run.get("status") or "")
        if status in TERMINAL_RUN_STATUSES:
            values: dict[str, Any] | None = None
            state: dict[str, Any] | None = None
            if status == "success":
                try:
                    tr = await self._upstream(host, "GET", f"/threads/{thread.thread_id}")
                    if tr.status_code < 400:
                        body = tr.json()
                        raw_values = body.get("values")
                        values = raw_values if isinstance(raw_values, dict) else None
                        state = _paused_state(body)
                except _UpstreamDown:
                    values = None
                if (current := self._superseding(thread)) is not None:
                    return self._synthetic_run(current, status=current.run_status or "running")
            self._mark_terminal(thread, run, values=values, state=state)
        elif status != thread.run_status:
            self.store.update_thread(thread.thread_id, run_status=status)
        return run

    def _superseding(self, thread: ThreadRow) -> ThreadRow | None:
        """The thread's current row if a newer run has replaced `thread.run_id`, else None."""
        current = self.store.get_thread(thread.thread_id)
        if current is not None and current.run_id != thread.run_id:
            return current
        return None

    async def _note_unreachable(
        self, thread: ThreadRow, host: HostRow, *, reason: str, live: str | None | object = _UNKNOWN
    ) -> dict[str, Any]:
        """Decide between 'transient blip' and 'host is gone' for an unreachable host.

        `live` is the host's inventory status when the caller has just looked it up.
        """
        failures = host.failures + 1
        if live is _UNKNOWN:
            live = await self._live_status(host.host_id)
        dead = live is None or live in _DEAD_HOST_STATUSES or failures >= self.unreachable_threshold
        if dead:
            self.store.update_host(host.host_id, status=HostStatus.FAILED.value, failures=failures)
            for other in self.store.list_threads(host_id=host.host_id):
                if other.run_status in ACTIVE_RUN_STATUSES and other.thread_id != thread.thread_id:
                    self._fail_thread(other, f"agent host {host.host_id} unreachable: {reason}")
            return self._fail_thread(thread, f"agent host {host.host_id} unreachable ({live or 'gone'}): {reason}")
        self.store.update_host(host.host_id, failures=failures)
        logger.warning("Host %s unreachable (%d/%d): %s", host.host_id, failures, self.unreachable_threshold, reason)
        return self._synthetic_run(thread, status=thread.run_status or "running")


    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------

    def _mark_terminal(
        self, thread: ThreadRow, run: dict[str, Any], *, values: dict[str, Any] | None, state: dict[str, Any] | None = None
    ) -> None:
        """Cache a finished run, its thread's values and, for a run that paused for a human, the thread's status and interrupts."""
        now = self.clock()
        self.store.update_thread(
            thread.thread_id,
            run_status=str(run.get("status") or "error"),
            cached_run=run,
            cached_values=values,
            cached_thread=state,
            last_active_at=now,
        )
        if thread.host_id:
            still_active = any(
                t.run_status in ACTIVE_RUN_STATUSES and t.thread_id != thread.thread_id
                for t in self.store.list_threads(host_id=thread.host_id)
            )
            host = self.store.get_host(thread.host_id)
            if host is not None and host.status not in _DEAD_HOST_STATUSES and host.status != HostStatus.PAUSED.value:
                self.store.update_host(
                    thread.host_id,
                    status=HostStatus.BUSY.value if still_active else HostStatus.READY.value,
                    idle_since=None if still_active else now,
                    last_active_at=now,
                )

    def _fail_thread(self, thread: ThreadRow, message: str) -> dict[str, Any]:
        run = self._synthetic_run(thread, status="error", error=message)
        self._mark_terminal(thread, run, values=None)
        logger.warning("Task %s failed: %s", thread.thread_id, message)
        return run

    def _synthetic_run(
        self, thread: ThreadRow, *, status: str, run_id: str | None = None, error: str | None = None
    ) -> dict[str, Any]:
        now = self.iso_now()
        spec = self._spec_for(thread)
        run: dict[str, Any] = {
            "run_id": run_id or thread.run_id or "",
            "thread_id": thread.thread_id,
            "assistant_id": (spec.graph_id if spec else thread.graph_id) or "",
            "status": status,
            "created_at": thread.run_started_at or now,
            "updated_at": now,
            "metadata": {"synthesized_by": "subagent-gateway"},
            "kwargs": {},
            "multitask_strategy": "reject",
        }
        if error is not None:
            run["error"] = error
        return run

    def _thread_payload(
        self, thread_id: str, *, values: dict[str, Any] | None = None, state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """A thread as the Agent Server answers it; `state` (cached_thread) keeps a paused run's status and interrupts."""
        now = self.iso_now()
        return {
            "thread_id": thread_id,
            "created_at": now,
            "updated_at": now,
            "metadata": {},
            "status": (state or {}).get("status", "idle"),
            "config": {},
            "values": values,
            "interrupts": (state or {}).get("interrupts") or {},
        }

    def _authorized(self, request: Request) -> bool:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            if hmac.compare_digest(authorization[7:].strip(), self.token):
                return True
        api_key = request.headers.get("x-api-key", "")
        return bool(api_key) and hmac.compare_digest(api_key, self.token)



def _from_agent_server(response: httpx.Response) -> bool:
    """True for an answer of the host's server (JSON, or the artifact routes' marker), not of the edge in front of it."""
    content_type = response.headers.get("content-type", "")
    return MARKER_HEADER in response.headers or content_type.split(";")[0].strip().endswith("json")


def _unauthorized() -> Response:
    return JSONResponse({"detail": "invalid or missing gateway token"}, status_code=401)


async def _json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
