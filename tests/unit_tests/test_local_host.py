"""Unit tests for agent_host.local.LocalDevServerHost with a scripted process.

No `langgraph dev` is launched here: `process_factory` returns a FakeProcess
and `signal_sender` records signals, so these tests pin the host's state
machine (provision -> spawn -> probe, SIGSTOP/SIGCONT, respawn after exit,
terminate + rmtree) and the files it writes. The real server is exercised by
test_local_host_e2e.py (opt-in).
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    HostStatus,
    LocalDevServerHost,
)
from langchain_digitalocean_microvm.agent_host import local as local_module
from langchain_digitalocean_microvm.agent_host.local import (
    GUARD_PATH,
    default_template_dir,
)
from langchain_digitalocean_microvm.config import TEMPLATE_DIR_ENV
from langchain_digitalocean_microvm.exceptions import (
    HostNotReadyError,
    SandboxError,
    SandboxNotFoundError,
)

_PIDS = itertools.count(4000)


class FakeProcess:
    def __init__(self) -> None:
        self.pid = next(_PIDS)
        self.returncode: int | None = None
        self.signals: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.returncode = -signal.SIGKILL

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="local", timeout=timeout or 0)
        return self.returncode


class Spawns:
    """process_factory double: records every spawn and hands out FakeProcesses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, args: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> FakeProcess:
        process = FakeProcess()
        self.calls.append({"args": args, "cwd": cwd, "env": env, "log_path": log_path})
        self.processes.append(process)
        return process


class Signals:
    def __init__(self) -> None:
        self.sent: list[tuple[int, int]] = []

    def __call__(self, process: FakeProcess, sig: int) -> None:
        self.sent.append((process.pid, sig))
        if sig in (signal.SIGTERM, signal.SIGKILL):
            process.returncode = -sig


class ScriptedProbe:
    def __init__(self, results: list[bool] | None = None, *, default: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self._results = list(results or [])
        self._default = default

    def __call__(self, url: str, token: str, timeout: float) -> bool:
        self.calls.append((url, token))
        return self._results.pop(0) if self._results else self._default


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _make(tmp_path: Path, *, probe: ScriptedProbe | None = None, clock: FakeClock | None = None):
    spawns, signals = Spawns(), Signals()
    probe = probe or ScriptedProbe()
    clock = clock or FakeClock()
    tokens = (f"tok-{n}" for n in itertools.count(1))
    host = LocalDevServerHost(
        root_dir=tmp_path / "hosts",
        command=["fake-langgraph", "dev"],
        probe=probe,
        token_factory=tokens.__next__,
        process_factory=spawns,
        signal_sender=signals,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=lambda: "2026-09-03T00:00:00Z",
        ready_timeout=10.0,
        probe_interval=1.0,
        env_passthrough=(),
    )
    return host, spawns, signals, probe, clock


SPEC = HostSpec(graph_id="echo", env={"DO_MODEL_ACCESS_KEY": "model-key"}, n_jobs_per_worker=2)


def test_provision_writes_config_env_and_spawns_langgraph_dev(tmp_path: Path) -> None:
    host, spawns, _signals, probe, _ = _make(tmp_path)

    handle = host.get_or_create(host_id="h1", spec=SPEC)

    assert handle.status is HostStatus.READY
    assert handle.token == "tok-1"
    assert handle.microvm_id == "local-h1"
    assert handle.url.startswith("http://127.0.0.1:")
    assert probe.calls == [(handle.url, "tok-1")]

    [spawn] = spawns.calls
    cwd = spawn["cwd"]
    assert cwd == tmp_path / "hosts" / "h1"
    port = handle.url.rsplit(":", 1)[1]
    args = spawn["args"]
    # The server runs under _guard.py so it dies with this process (see fake.py docstring) ...
    assert args[:5] == [sys.executable, str(GUARD_PATH), "--parent", str(os.getpid()), "--"]
    args = args[5:]
    # ... and the real command follows.
    assert args[:2] == ["fake-langgraph", "dev"]
    for flag, value in (("--config", str(cwd / "langgraph.json")), ("--port", port), ("--n-jobs-per-worker", "2")):
        assert args[args.index(flag) + 1] == value
    assert {"--host", "--no-browser", "--no-reload", "--allow-blocking"} <= set(args)

    template = default_template_dir()
    config = json.loads((cwd / "langgraph.json").read_text())
    assert config["env"] == "./.env"
    assert config["dependencies"] == [str(template)]
    assert config["graphs"]["echo"] == f"{template / 'agent.py'}:echo"
    assert config["graphs"]["researcher"] == f"{template / 'agent.py'}:make_graph"
    assert config["auth"]["path"] == f"{template / 'auth.py'}:auth"
    assert config["http"] == {"app": f"{template / 'artifacts_app.py'}:app", "enable_custom_route_auth": True}

    env_file = (cwd / ".env").read_text()
    assert 'SUBAGENT_HOST_TOKEN="tok-1"' in env_file
    assert 'N_JOBS_PER_WORKER="2"' in env_file
    assert 'DO_MODEL_ACCESS_KEY="model-key"' in env_file
    assert f'SUBAGENT_WORKDIR="{host.workdir_for("h1")}"' in env_file
    assert host.workdir_for("h1").is_dir()


def test_pause_and_resume_send_sigstop_and_sigcont(tmp_path: Path) -> None:
    host, spawns, signals, probe, _ = _make(tmp_path)
    handle = host.get_or_create(host_id="h2", spec=SPEC)
    pid = spawns.processes[0].pid

    host.pause(host_id="h2")
    host.pause(host_id="h2")  # idempotent
    assert signals.sent == [(pid, signal.SIGSTOP)]
    assert host.list_hosts()[0].status is HostStatus.PAUSED

    resumed = host.resume(host_id="h2")
    assert signals.sent == [(pid, signal.SIGSTOP), (pid, signal.SIGCONT)]
    assert resumed.status is HostStatus.READY
    assert resumed.url == handle.url
    assert len(probe.calls) == 2
    assert len(spawns.calls) == 1  # no respawn on resume

    host.pause(host_id="h2")
    again = host.get_or_create(host_id="h2", spec=SPEC)  # get_or_create also resumes
    assert signals.sent[-1] == (pid, signal.SIGCONT)
    assert again.status is HostStatus.READY


def test_delete_terminates_process_and_removes_directories(tmp_path: Path) -> None:
    host, spawns, signals, _, _ = _make(tmp_path)
    host.get_or_create(host_id="h3", spec=SPEC)
    cwd = spawns.calls[0]["cwd"]
    workdir = host.workdir_for("h3")
    (workdir / "outputs").mkdir(parents=True, exist_ok=True)
    assert cwd.is_dir()

    host.pause(host_id="h3")
    host.delete(host_id="h3")

    pid = spawns.processes[0].pid
    # A stopped process must be continued before SIGTERM can take effect.
    assert signals.sent[-2:] == [(pid, signal.SIGCONT), (pid, signal.SIGTERM)]
    assert spawns.processes[0].returncode == -signal.SIGTERM
    assert not cwd.exists()
    assert not workdir.exists()
    with pytest.raises(SandboxNotFoundError):
        host.delete(host_id="h3")
    with pytest.raises(SandboxNotFoundError):
        host.pause(host_id="h3")


def test_process_exit_before_ready_raises_with_log_tail(tmp_path: Path) -> None:
    probe = ScriptedProbe(default=False)
    host, spawns, _, _, _ = _make(tmp_path, probe=probe)

    def crashing(args, cwd, env, log_path):
        process = spawns(args, cwd, env, log_path)
        log_path.write_text("Traceback...\nModuleNotFoundError: No module named 'nope'\n")
        process.returncode = 1
        return process

    host._process_factory = crashing  # type: ignore[attr-defined]

    with pytest.raises(HostNotReadyError) as excinfo:
        host.get_or_create(host_id="h4", spec=SPEC)

    message = str(excinfo.value)
    assert "exited with code 1" in message
    assert "ModuleNotFoundError" in message
    assert host.list_hosts()[0].status is HostStatus.FAILED


def test_probe_timeout_raises_without_real_sleep(tmp_path: Path) -> None:
    probe = ScriptedProbe(default=False)
    clock = FakeClock()
    host, _, _, _, _ = _make(tmp_path, probe=probe, clock=clock)

    with pytest.raises(HostNotReadyError, match="did not answer"):
        host.get_or_create(host_id="h5", spec=SPEC)

    assert len(probe.calls) >= 10
    assert clock.sleeps and all(s == 1.0 for s in clock.sleeps)


def test_exited_server_is_respawned_in_the_same_directory(tmp_path: Path) -> None:
    host, spawns, _, probe, _ = _make(tmp_path)
    first = host.get_or_create(host_id="h6", spec=SPEC)
    spawns.processes[0].returncode = 137  # the server died

    assert host.list_hosts()[0].status is HostStatus.FAILED

    second = host.get_or_create(host_id="h6", spec=SPEC)

    assert len(spawns.calls) == 2
    assert spawns.calls[1]["cwd"] == spawns.calls[0]["cwd"]  # pickles survive, like a service restart
    assert second.url == first.url
    assert second.token == first.token
    assert second.status is HostStatus.READY


def test_list_hosts_filters_by_graph(tmp_path: Path) -> None:
    host, _, _, _, _ = _make(tmp_path)
    host.get_or_create(host_id="e1", spec=HostSpec(graph_id="echo"))
    host.get_or_create(host_id="r1", spec=HostSpec(graph_id="researcher"))

    assert sorted(h.host_id for h in host.list_hosts()) == ["e1", "r1"]
    assert [h.host_id for h in host.list_hosts(graph_id="researcher")] == ["r1"]
    assert host.list_hosts(graph_id="nope") == []


def test_the_subagent_workdir_is_the_hosts_own_directory(tmp_path: Path) -> None:
    host, spawns, _, _, _ = _make(tmp_path)
    host.get_or_create(host_id="h7", spec=SPEC)
    host.get_or_create(host_id="h8", spec=SPEC)
    env_file = (spawns.calls[0]["cwd"] / ".env").read_text()
    assert f'SUBAGENT_WORKDIR="{host.workdir_for("h7")}"' in env_file
    assert host.workdir_for("h7").is_dir() and host.workdir_for("h7") != host.workdir_for("h8")


def test_is_idle_uses_queue_metrics_and_fails_closed(tmp_path: Path) -> None:
    host, _, _, _, _ = _make(tmp_path)
    responses: list[object] = [
        {"queue": {"n_pending": 0, "n_running": 0}},
        {"queue": {"n_pending": 1, "n_running": 0}},
        RuntimeError("down"),
    ]

    def metrics(url: str, token: str, timeout: float) -> dict:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    host._metrics_fetcher = metrics  # type: ignore[attr-defined]
    handle = host.get_or_create(host_id="h8", spec=SPEC)

    assert host.is_idle(handle) is True
    assert host.is_idle(handle) is False
    assert host.is_idle(handle) is False


def test_close_tears_down_every_host(tmp_path: Path) -> None:
    host, spawns, _, _, _ = _make(tmp_path)
    host.get_or_create(host_id="a", spec=SPEC)
    host.get_or_create(host_id="b", spec=SPEC)

    with host:
        pass

    assert host.list_hosts() == []
    assert all(p.returncode == -signal.SIGTERM for p in spawns.processes)


def _fake_install(root: Path, *, checkout_config: bool) -> Path:
    """A <root>/src/langchain_digitalocean_microvm/agent_host/local.py layout; returns that fake.py path."""
    module = root / "src" / "langchain_digitalocean_microvm" / "agent_host" / "fake.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (root / "images" / "agent-host").mkdir(parents=True)
    if checkout_config:
        (root / "images" / "agent-host" / "langgraph.json").write_text("{}")
    return module


def test_template_dir_prefers_the_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_module, "__file__", str(_fake_install(tmp_path, checkout_config=True)))
    monkeypatch.setenv(TEMPLATE_DIR_ENV, str(tmp_path / "mine"))
    assert default_template_dir() == tmp_path / "mine"


def test_template_dir_uses_a_source_checkout_with_a_langgraph_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_module, "__file__", str(_fake_install(tmp_path, checkout_config=True)))
    monkeypatch.delenv(TEMPLATE_DIR_ENV, raising=False)
    assert default_template_dir() == (tmp_path / "images" / "agent-host").resolve()


@pytest.mark.parametrize("checkout", ["none", "without langgraph.json"])
def test_template_dir_falls_back_to_the_packaged_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkout: str
) -> None:
    if checkout == "none":
        module = tmp_path / "site-packages" / "langchain_digitalocean_microvm" / "agent_host" / "fake.py"
        module.parent.mkdir(parents=True)
        module.touch()
    else:
        module = _fake_install(tmp_path, checkout_config=False)
    monkeypatch.setattr(local_module, "__file__", str(module))
    monkeypatch.setenv(TEMPLATE_DIR_ENV, "")  # empty counts as unset
    assert default_template_dir() == local_module.PACKAGED_TEMPLATE_DIR


def test_packaged_template_sits_in_the_agent_host_package() -> None:
    # The wheel's force-include (pyproject.toml) must target this directory.
    assert local_module.PACKAGED_TEMPLATE_DIR == GUARD_PATH.parent / "_template"
    assert local_module.PACKAGED_TEMPLATE_DIR.parent.name == "agent_host"


def test_missing_template_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="langgraph.json"):
        LocalDevServerHost(template_dir=tmp_path / "nowhere", root_dir=tmp_path / "hosts")
