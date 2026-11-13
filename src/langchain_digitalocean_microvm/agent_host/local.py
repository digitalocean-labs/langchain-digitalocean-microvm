"""Local mode agent host: one local `langgraph dev` process per host_id.

LocalDevServerHost has the same public surface as MicroVMAgentHost but
needs no DigitalOcean credentials: each host is an in-memory LangGraph Agent
Server spawned from the agent-host template on a free loopback port.

How the MicroVM lifecycle is emulated:

- create   -> a fresh working directory under `root_dir/<host_id>` holding a
              generated langgraph.json (absolute paths into the template),
              a ./.env file (same variables a real host gets at create time)
              and server.log;
              then `python -m langgraph_cli dev ...` is spawned with that cwd,
              so `.langgraph_api/` pickles land per host exactly as they land
              on a MicroVM's disk.
- pause    -> SIGSTOP the process group. In-flight HTTP hangs, exactly like a
              checkpointed VM, which is what the gateway tests need.
- resume   -> SIGCONT.
- destroy  -> SIGTERM (SIGKILL after a grace period) and rmtree.
- FAILED   -> a server that exited is respawned in the same cwd on the next
              get_or_create/resume, the way a crashed in-VM service restarts.
- parent   -> every server runs under `_guard.py`, a stdlib-only wrapper in
  death       the same process group that terminates it when THIS process
              disappears. Needed because a `langgraph dev` reload worker ends
              by a re-raised SIGTERM (uvicorn restores the default handler
              first), so no atexit/threading hook in the supervisor ever runs.

The subagent's filesystem (SUBAGENT_WORKDIR) is LocalTransport's root for
`agenthost-<host_id>` (`workdir_for()`). The server serves the template's
artifact routes (images/agent-host/artifacts_app.py) like a real host, so the
gateway reads artifacts the same way in both modes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from langchain_digitalocean_microvm.agent_host.env import (
    build_host_env,
    render_env_file,
)
from langchain_digitalocean_microvm.agent_host.host import (
    MetricsFetcher,
    Probe,
    default_metrics_fetcher,
    default_probe,
    queue_is_idle,
    utc_now_iso,
)
from langchain_digitalocean_microvm.agent_host.models import (
    HostHandle,
    HostSpec,
    HostStatus,
)
from langchain_digitalocean_microvm.config import (
    PASSTHROUGH_ENV,
    TEMPLATE_DIR_ENV,
    env_csv,
    env_value,
)
from langchain_digitalocean_microvm.exceptions import (
    HostNotReadyError,
    SandboxError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.transport.local import root_for_host

logger = logging.getLogger(__name__)

LOOPBACK = "127.0.0.1"
_LOG_TAIL_LINES = 40
_TOKEN_BYTES = 32
_TERMINATE_GRACE = 10.0
GUARD_PATH = Path(__file__).with_name("_guard.py")


class ProcessLike(Protocol):
    """The subset of subprocess.Popen the host relies on (so tests can fake it)."""

    pid: int

    def poll(self) -> int | None: ...
    def send_signal(self, sig: int) -> None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


# (argv, cwd, env, log path) -> running process
ProcessFactory = Callable[[list[str], Path, dict[str, str], Path], ProcessLike]
SignalSender = Callable[[ProcessLike, int], None]


PACKAGED_TEMPLATE_DIR = Path(__file__).with_name("_template")


def default_template_dir() -> Path:
    """The agent-host template a local host serves, first match wins.

    1. `$DO_MICROVM_SUBAGENT_TEMPLATE`.
    2. `images/agent-host/` of a source checkout, when it holds a langgraph.json
       (an editable install: edits there apply without a rebuild).
    3. The copy the wheel ships in `agent_host/_template/` (agent.py, auth.py,
       artifacts_app.py, langgraph.json; built from `images/agent-host/`).
    """
    override = env_value(TEMPLATE_DIR_ENV)
    if override:
        return Path(override)
    # <repo>/src/langchain_digitalocean_microvm/agent_host/local.py -> <repo>/images/agent-host
    checkout = Path(__file__).resolve().parents[3] / "images" / "agent-host"
    if (checkout / "langgraph.json").is_file():
        return checkout
    return PACKAGED_TEMPLATE_DIR


def default_root_dir() -> Path:
    return Path(tempfile.gettempdir()) / "microvm_agent_local"


def default_command() -> list[str]:
    """Run the CLI from THIS interpreter, so whatever venv the caller is in is what serves."""
    return [sys.executable, "-m", "langgraph_cli", "dev"]


def default_process_factory(args: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> ProcessLike:
    with log_path.open("ab") as log:
        return subprocess.Popen(  # noqa: S603 - argv built by us, not from user input
            args,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group: pause/stop the whole tree
        )


def default_signal_sender(process: ProcessLike, sig: int) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        process.send_signal(sig)


def free_port(host: str = LOOPBACK) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _safe_dirname(host_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", host_id) or "host"


def _absolutize_spec(spec: str, base: Path) -> str:
    """'./agent.py:graph' -> '/abs/agent.py:graph'; module specs ('pkg.mod:graph') untouched."""
    if "/" not in spec:
        return spec
    path, sep, attr = spec.rpartition(":")
    if not sep:
        return str((base / spec).resolve())
    return f"{(base / path).resolve()}{sep}{attr}"


@dataclass
class _HostRecord:
    host_id: str
    spec: HostSpec
    port: int
    token: str
    cwd: Path
    log_path: Path
    process: ProcessLike | None = None
    status: HostStatus = HostStatus.BOOTING
    booted_at: str = ""


class LocalDevServerHost:
    """Create-or-reuse manager for local `langgraph dev` hosts, one per host_id."""

    def __init__(
        self,
        *,
        template_dir: Path | str | None = None,
        root_dir: Path | str | None = None,
        command: Sequence[str] | None = None,
        extra_args: Sequence[str] = (),
        probe: Probe | None = None,
        metrics_fetcher: MetricsFetcher | None = None,
        token_factory: Callable[[], str] | None = None,
        process_factory: ProcessFactory | None = None,
        signal_sender: SignalSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], str] = utc_now_iso,
        ready_timeout: float = 90.0,
        probe_interval: float = 0.5,
        probe_attempt_timeout: float = 2.0,
        env_passthrough: Sequence[str] | None = None,
        guard: bool = True,
    ) -> None:
        """`guard=False` runs `command` directly instead of under `_guard.py` (scripted unit tests)."""
        self.template_dir = Path(template_dir) if template_dir is not None else default_template_dir()
        self.template_dir = self.template_dir.resolve()
        self._template_config = self.template_dir / "langgraph.json"
        if not self._template_config.is_file():
            raise SandboxError(
                f"No subagent template at {self.template_dir} (expected langgraph.json); "
                f"set {TEMPLATE_DIR_ENV} or pass template_dir="
            )
        self.root_dir = Path(root_dir) if root_dir is not None else default_root_dir()
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.command = list(command) if command is not None else default_command()
        self.extra_args = list(extra_args)
        self.guard = guard
        self._probe = probe if probe is not None else default_probe
        self._metrics_fetcher = metrics_fetcher if metrics_fetcher is not None else default_metrics_fetcher
        self._token_factory = (
            token_factory if token_factory is not None else lambda: secrets.token_urlsafe(_TOKEN_BYTES)
        )
        self._process_factory = process_factory if process_factory is not None else default_process_factory
        self._signal = signal_sender if signal_sender is not None else default_signal_sender
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self.ready_timeout = ready_timeout
        self._probe_interval = probe_interval
        self._probe_attempt_timeout = probe_attempt_timeout
        if env_passthrough is None:
            env_passthrough = env_csv(PASSTHROUGH_ENV)
        self.env_passthrough = tuple(env_passthrough)

        self._hosts: dict[str, _HostRecord] = {}

    # ------------------------------------------------------------------
    # public sync API (mirrors MicroVMAgentHost)
    # ------------------------------------------------------------------

    def get_or_create(self, *, host_id: str, spec: HostSpec) -> HostHandle:
        record = self._hosts.get(host_id)
        if record is None:
            record = self._provision(host_id, spec)
        else:
            record.spec = spec
            if record.status == HostStatus.PAUSED:
                self._signal_record(record, signal.SIGCONT)
                record.status = HostStatus.BOOTING
            elif self._exited(record):
                logger.warning("local host %s exited; respawning in %s", host_id, record.cwd)
                self._spawn(record)
        return self.wait_ready(self._handle(record))

    def wait_ready(self, handle: HostHandle, *, timeout: float | None = None) -> HostHandle:
        record = self._record(handle.host_id)
        budget = self.ready_timeout if timeout is None else timeout
        deadline = self._monotonic() + budget
        while True:
            if self._probe(handle.url, record.token, self._probe_attempt_timeout):
                record.status = HostStatus.READY
                return replace(handle, status=HostStatus.READY, booted_at=record.booted_at)
            if self._exited(record):
                record.status = HostStatus.FAILED
                code = record.process.poll() if record.process is not None else None
                raise HostNotReadyError(
                    f"local host {handle.host_id!r} exited with code {code} before answering "
                    f"GET {handle.url}/ok.\n--- {record.log_path} (tail) ---\n{self._log_tail(record)}"
                )
            if self._monotonic() >= deadline:
                raise HostNotReadyError(
                    f"local host {handle.host_id!r} did not answer GET {handle.url}/ok within "
                    f"{budget}s.\n--- {record.log_path} (tail) ---\n{self._log_tail(record)}"
                )
            self._sleep(self._probe_interval)

    def pause(self, *, host_id: str) -> None:
        record = self._record(host_id)
        if record.status == HostStatus.PAUSED:
            return
        if self._exited(record):
            raise SandboxError(f"local host {host_id!r} is not running; nothing to pause")
        self._signal_record(record, signal.SIGSTOP)
        record.status = HostStatus.PAUSED

    def resume(self, *, host_id: str, spec: HostSpec | None = None) -> HostHandle:
        record = self._record(host_id)
        if spec is not None:
            record.spec = spec
        if record.status == HostStatus.PAUSED:
            self._signal_record(record, signal.SIGCONT)
            record.status = HostStatus.BOOTING
        elif self._exited(record):
            logger.warning("local host %s exited; respawning in %s", host_id, record.cwd)
            self._spawn(record)
        return self.wait_ready(self._handle(record))

    def delete(self, *, host_id: str) -> None:
        record = self._hosts.pop(host_id, None)
        if record is None:
            raise SandboxNotFoundError(f"No local host registered for host_id={host_id!r}")
        self._terminate(record)
        shutil.rmtree(record.cwd, ignore_errors=True)
        shutil.rmtree(self.workdir_for(host_id), ignore_errors=True)

    def list_hosts(self, *, graph_id: str | None = None) -> list[HostHandle]:
        handles: list[HostHandle] = []
        for record in self._hosts.values():
            if graph_id is not None and record.spec.graph_id != graph_id:
                continue
            status = record.status
            if status != HostStatus.PAUSED and self._exited(record):
                status = HostStatus.FAILED
            handles.append(replace(self._handle(record), status=status))
        return handles

    def is_idle(self, handle: HostHandle) -> bool:
        record = self._record(handle.host_id)
        try:
            body = self._metrics_fetcher(handle.url, record.token, self._probe_attempt_timeout)
        except Exception:  # noqa: BLE001 - unknown means not idle
            logger.warning("Could not fetch metrics from local host %s; treating as busy", handle.host_id, exc_info=True)
            return False
        return queue_is_idle(body)

    def workdir_for(self, host_id: str) -> Path:
        """Where the subagent's `/` lives on this machine (its SUBAGENT_WORKDIR)."""
        return root_for_host(self._local_transport_host(host_id))

    def close(self) -> None:
        """Destroy every host this manager started (tests, atexit)."""
        for host_id in list(self._hosts):
            try:
                self.delete(host_id=host_id)
            except Exception:  # noqa: BLE001 - best effort teardown
                logger.warning("Failed to tear down local host %s", host_id, exc_info=True)

    def __enter__(self) -> LocalDevServerHost:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # public async API
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

    def _record(self, host_id: str) -> _HostRecord:
        record = self._hosts.get(host_id)
        if record is None:
            raise SandboxNotFoundError(f"No local host registered for host_id={host_id!r}")
        return record

    @staticmethod
    def _local_transport_host(host_id: str) -> str:
        return f"agenthost-{host_id}"

    def _provision(self, host_id: str, spec: HostSpec) -> _HostRecord:
        cwd = self.root_dir / _safe_dirname(host_id)
        if cwd.exists():
            shutil.rmtree(cwd)  # a fresh host never inherits a dead predecessor's pickles
        cwd.mkdir(parents=True)
        self.workdir_for(host_id).mkdir(parents=True, exist_ok=True)
        record = _HostRecord(
            host_id=host_id,
            spec=spec,
            port=free_port(),
            token=self._token_factory(),
            cwd=cwd,
            log_path=cwd / "server.log",
        )
        self._write_env(record)
        self._write_config(record)
        self._spawn(record)
        self._hosts[host_id] = record
        return record

    def _build_env(self, record: _HostRecord) -> dict[str, str]:
        env = build_host_env(record.spec, record.token, passthrough=self.env_passthrough)
        env["SUBAGENT_WORKDIR"] = str(self.workdir_for(record.host_id))
        env["SUBAGENT_HOST_TOKEN_FILE"] = str(record.cwd / "host-token")  # never a shared path on this machine
        return env

    def _write_env(self, record: _HostRecord) -> None:
        env_path = record.cwd / ".env"
        env_path.write_text(render_env_file(self._build_env(record)))
        env_path.chmod(0o600)

    def _write_config(self, record: _HostRecord) -> None:
        """Per-host langgraph.json: template paths made absolute, env pointed at ./.env."""
        config = json.loads(self._template_config.read_text())
        deps = config.get("dependencies", ["."])
        config["dependencies"] = [
            str((self.template_dir / dep).resolve()) if ("/" in dep or dep.startswith(".")) else dep
            for dep in deps
        ]
        config["graphs"] = {
            name: _absolutize_spec(spec, self.template_dir) for name, spec in config.get("graphs", {}).items()
        }
        config["env"] = "./.env"
        auth = config.get("auth")
        if isinstance(auth, dict) and isinstance(auth.get("path"), str):
            auth["path"] = _absolutize_spec(auth["path"], self.template_dir)
        http = config.get("http")
        if isinstance(http, dict) and isinstance(http.get("app"), str):
            http["app"] = _absolutize_spec(http["app"], self.template_dir)
        (record.cwd / "langgraph.json").write_text(json.dumps(config, indent=2) + "\n")

    def _spawn(self, record: _HostRecord) -> None:
        args = [
            *self._guard_prefix(),
            *self.command,
            "--config",
            str(record.cwd / "langgraph.json"),
            "--host",
            LOOPBACK,
            "--port",
            str(record.port),
            "--no-browser",
            "--no-reload",
            "--n-jobs-per-worker",
            str(record.spec.n_jobs_per_worker),
            "--allow-blocking",
            *self.extra_args,
        ]
        record.process = self._process_factory(args, record.cwd, dict(os.environ), record.log_path)
        record.status = HostStatus.BOOTING
        record.booted_at = self._now()

    def _guard_prefix(self) -> list[str]:
        """Wrap the server in `_guard.py` so it dies with this process (see module docstring)."""
        if not self.guard:
            return []
        return [sys.executable, str(GUARD_PATH), "--parent", str(os.getpid()), "--"]

    def _exited(self, record: _HostRecord) -> bool:
        return record.process is None or record.process.poll() is not None

    def _signal_record(self, record: _HostRecord, sig: int) -> None:
        if record.process is None:
            raise SandboxError(f"local host {record.host_id!r} has no process")
        self._signal(record.process, sig)

    def _terminate(self, record: _HostRecord) -> None:
        process = record.process
        if process is None or process.poll() is not None:
            return
        if record.status == HostStatus.PAUSED:
            self._signal(process, signal.SIGCONT)  # a stopped process ignores SIGTERM
        self._signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=_TERMINATE_GRACE)
        except subprocess.TimeoutExpired:
            self._signal(process, signal.SIGKILL)
            process.wait(timeout=_TERMINATE_GRACE)

    def _log_tail(self, record: _HostRecord) -> str:
        try:
            lines = record.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return "(no log)"
        return "\n".join(lines[-_LOG_TAIL_LINES:]) or "(empty log)"

    def _handle(self, record: _HostRecord) -> HostHandle:
        return HostHandle(
            host_id=record.host_id,
            microvm_id=f"local-{record.host_id}",
            graph_id=record.spec.graph_id,
            url=f"http://{LOOPBACK}:{record.port}",
            token=record.token,
            status=record.status,
            booted_at=record.booted_at,
        )
