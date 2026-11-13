"""Warm start: agent hosts created from a golden checkpoint.

FakeMicroVMClient implements checkpoints (instantly AVAILABLE) and clones
(come back running), and its exec() runs commands locally, so the token
install really writes the token file -- pointed at tmp_path here.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient
from test_agent_host import FakeClock, ScriptedProbe, _config, _tokens

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    HostStatus,
    MicroVMAgentHost,
)
from langchain_digitalocean_microvm.agent_host.host import GOLDEN_PREFIX, GOLDEN_TAG
from langchain_digitalocean_microvm.exceptions import HostNotReadyError
from langchain_digitalocean_microvm.models import ExecResult
from langchain_digitalocean_microvm.status import MicroVMStatus

SPEC = HostSpec(graph_id="echo", env={"DO_MODEL_ACCESS_KEY": "k"})


def make(client: FakeMicroVMClient, tmp_path: Path, *, probe: ScriptedProbe | None = None) -> tuple[MicroVMAgentHost, ScriptedProbe]:
    probe = probe or ScriptedProbe()
    clock = FakeClock()
    host = MicroVMAgentHost(
        _config(warm_start=True),
        client=client,  # type: ignore[arg-type]
        probe=probe,
        token_factory=_tokens().__next__,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        token_file=str(tmp_path / "agent" / "token"),
    )
    return host, probe


def test_first_host_builds_the_golden_checkpoint_then_runs_as_a_clone_with_its_own_token(tmp_path) -> None:
    client = FakeMicroVMClient()
    host, probe = make(client, tmp_path)

    handle = host.get_or_create(host_id="h1", spec=SPEC)

    golden, clone = client.created
    assert golden["name"].startswith(GOLDEN_PREFIX) and golden["oci_ref"] and golden["tags"] == [GOLDEN_TAG]
    assert golden["environment"]["SUBAGENT_HOST_TOKEN"] == "tok-2"  # the golden VM's throwaway token
    assert clone["checkpoint_id"] and clone["environment"] is None and clone["oci_ref"] is None
    golden_vm = next(s for s in client._microvms.values() if s.name == golden["name"])
    assert golden_vm.status is MicroVMStatus.TERMINATING  # deleted once checkpointed: frees the slot

    assert handle.status is HostStatus.READY and handle.token == "tok-1"
    assert (tmp_path / "agent" / "token").read_text() == "tok-1"  # installed via exec
    assert (tmp_path / "agent" / "token").stat().st_mode & 0o077 == 0
    assert probe.calls[-1] == (handle.url, "tok-1")  # probed with the clone's token, not the inherited one


def test_later_hosts_reuse_the_checkpoint(tmp_path) -> None:
    client = FakeMicroVMClient()
    host, _ = make(client, tmp_path)
    host.get_or_create(host_id="h1", spec=SPEC)

    second = host.get_or_create(host_id="h2", spec=SPEC)

    assert [c["name"].startswith(GOLDEN_PREFIX) for c in client.created] == [True, False, False]
    assert len(client._checkpoints) == 1
    assert second.token == "tok-3"


def test_a_different_environment_gets_its_own_checkpoint(tmp_path) -> None:
    client = FakeMicroVMClient()
    host, _ = make(client, tmp_path)
    host.get_or_create(host_id="h1", spec=SPEC)
    host.get_or_create(host_id="h2", spec=HostSpec(graph_id="echo", env={"DO_MODEL_ACCESS_KEY": "rotated"}))
    assert len(client._checkpoints) == 2
    assert len({c.name for c in client._checkpoints.values()}) == 2


def test_a_golden_that_never_becomes_ready_falls_back_to_a_cold_create(tmp_path) -> None:
    client = FakeMicroVMClient()
    probe = ScriptedProbe()
    probe_calls = probe.__call__

    def golden_never_ready(url: str, token: str, timeout: float) -> bool:
        return False if token == "tok-2" else probe_calls(url, token, timeout)

    host, _ = make(client, tmp_path, probe=golden_never_ready)  # type: ignore[arg-type]
    handle = host.get_or_create(host_id="h1", spec=SPEC)

    golden, cold = client.created
    assert golden["name"].startswith(GOLDEN_PREFIX)
    assert cold["oci_ref"] and cold["environment"]["SUBAGENT_HOST_TOKEN"] == "tok-1"
    assert not client._checkpoints and handle.token == "tok-1"
    assert not (tmp_path / "agent" / "token").exists()  # cold hosts use the env token


def test_a_failed_token_install_is_reported(tmp_path, monkeypatch) -> None:
    client = FakeMicroVMClient()
    host, _ = make(client, tmp_path)
    original = client.exec

    def broken_exec(microvm_id, argv, *, cwd=None):
        if host.token_file in argv:
            return ExecResult(stdout="", stderr="read-only file system", exit_code=1)
        return original(microvm_id, argv, cwd=cwd)

    monkeypatch.setattr(client, "exec", broken_exec)
    with pytest.raises(HostNotReadyError, match="read-only"):
        host.get_or_create(host_id="h1", spec=SPEC)


def test_a_clone_installs_its_token_after_its_server_answers(tmp_path, monkeypatch) -> None:
    """GET /ok (no token needed) comes first and the token install (one exec) after it: an exec sent
    to a fresh clone before its workload answers slows the clone's start."""
    client = FakeMicroVMClient()
    clone_probed = threading.Event()
    original = client.exec

    def exec_after_probe(microvm_id, argv, *, cwd=None):
        if not clone_probed.wait(timeout=5):
            raise AssertionError("the token install ran before the clone was probed")
        return original(microvm_id, argv, cwd=cwd)

    def probe(url: str, token: str, timeout: float) -> bool:
        if token != "tok-2":  # not the golden VM: the clone
            clone_probed.set()
        return True

    host, _ = make(client, tmp_path, probe=probe)  # type: ignore[arg-type]
    monkeypatch.setattr(client, "exec", exec_after_probe)
    handle = host.get_or_create(host_id="h1", spec=SPEC)
    assert handle.status is HostStatus.READY
    assert (tmp_path / "agent" / "token").read_text() == "tok-1"  # written before the handle came back
    assert sum(host.token_file in argv for _id, argv in client.exec_calls) == 1


def test_a_failed_token_install_is_reported_even_when_the_server_is_up(tmp_path, monkeypatch) -> None:
    """/ok needs no token, so a READY probe says nothing about the install: its failure still counts."""
    client = FakeMicroVMClient()
    host, probe = make(client, tmp_path)

    def broken_exec(microvm_id, argv, *, cwd=None):
        return ExecResult(stdout="", stderr="read-only file system", exit_code=1)

    monkeypatch.setattr(client, "exec", broken_exec)
    with pytest.raises(HostNotReadyError, match="read-only"):
        host.get_or_create(host_id="h1", spec=SPEC)
    assert probe.calls  # the probe ran meanwhile
