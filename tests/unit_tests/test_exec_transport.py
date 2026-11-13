"""Tests for transport.exec.ExecTransport.

The MicroVM is FakeMicroVMClient, whose exec() runs argv as a real local
subprocess and enforces the API's limits (128 KiB per argument, 1 MiB body,
4 MiB output). So these tests run the transport's actual shell wrapper:
inline completion, detach-and-poll, timeout kill, chunked file transfer.
Paths live under tmp_path; the job directory too (`job_root`).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.transport import exec as exec_module
from langchain_digitalocean_microvm.transport.base import TransportCredentials
from langchain_digitalocean_microvm.transport.errors import (
    TransportError,
    TransportFileNotFoundError,
    TransportIsADirectoryError,
    TransportPermissionError,
)
from langchain_digitalocean_microvm.transport.exec import (
    TIMEOUT_EXIT_CODE,
    ExecTransport,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")


@pytest.fixture
def vm() -> tuple[FakeMicroVMClient, str]:
    client = FakeMicroVMClient()
    state = client.create(name="t", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048))
    client.get(state.id)  # settle to running
    return client, state.id


def connect(client: FakeMicroVMClient, microvm_id: str, tmp_path: Path, **kwargs) -> ExecTransport:
    transport = ExecTransport(
        client,  # type: ignore[arg-type]  # duck-typed double
        job_root=str(tmp_path / "jobs"),
        sleep=lambda s: time.sleep(min(s, 0.2)),
        **kwargs,
    )
    transport.connect(host=microvm_id, credentials=TransportCredentials(api_token="t"))
    return transport


def _gnu_grep() -> bool:
    import subprocess

    out = subprocess.run(["grep", "--version"], capture_output=True, text=True, check=False).stdout
    return out.startswith("grep (GNU grep)")


def jobs_left(tmp_path: Path) -> list[Path]:
    root = tmp_path / "jobs"
    return list(root.iterdir()) if root.exists() else []


def test_short_command_costs_one_exec_and_keeps_streams_apart(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path)
    result = t.run("echo out; echo err >&2; exit 3")
    assert (result.stdout, result.stderr, result.exit_code) == ("out\n", "err\n", 3)
    assert len(client.exec_calls) == 1
    assert jobs_left(tmp_path) == []  # job directory cleaned up


def test_quotes_heredocs_and_dollars_survive_unchanged(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path)
    command = "x='it'\"'\"'s'; cat <<'END'\n$HOME \"quoted\" `tick` $x\nEND\necho \"$x\""
    result = t.run(command)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "$HOME \"quoted\" `tick` $x\nit's\n"


def test_slow_command_detaches_and_is_polled_to_completion(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path, inline_wait=1, poll_interval=0.2)
    result = t.run("sleep 2; echo done; echo late >&2", timeout=30)
    assert (result.stdout, result.stderr, result.exit_code) == ("done\n", "late\n", 0)
    assert len(client.exec_calls) >= 2  # start + at least one poll
    assert jobs_left(tmp_path) == []


def test_timeout_kills_the_job_and_returns_124(vm, tmp_path) -> None:
    client, vid = vm
    marker = tmp_path / "should-not-exist"
    t = connect(client, vid, tmp_path, inline_wait=1, poll_interval=0.2)
    result = t.run(f"echo started; sleep 5; touch {marker}", timeout=2)
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert "started" in result.stdout and "timed out after 2s" in result.stderr
    time.sleep(4)
    assert not marker.exists()  # the job really was killed
    assert jobs_left(tmp_path) == []


def test_payloads_are_split_into_args_under_the_kernel_limit(vm, tmp_path, monkeypatch) -> None:
    client, vid = vm
    monkeypatch.setattr(exec_module, "_ARG_CHARS", 1000)
    t = connect(client, vid, tmp_path)
    blob = "x" * 5000
    result = t.run(f"printf '%s' '{blob}' | wc -c")
    assert result.stdout.strip() == "5000"
    argv = client.exec_calls[-1][1]
    assert len(argv) > 4 and all(len(a) <= 1000 for a in argv[3:])


def test_command_too_large_for_one_request_is_uploaded_first(vm, tmp_path, monkeypatch) -> None:
    client, vid = vm
    monkeypatch.setattr(exec_module, "_PAYLOAD_CHARS", 100)
    t = connect(client, vid, tmp_path)
    result = t.run("echo " + "y" * 300 + " | wc -c")
    assert result.stdout.strip() == "301"
    assert len(client.exec_calls) == 2  # upload + run


def test_files_round_trip_in_chunks_without_breaking_api_limits(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path, write_chunk=4096, read_chunk=6000)
    data = os.urandom(20_000)
    target = tmp_path / "sub" / "dir" / "blob.bin"
    t.put_file(str(target), data)
    assert target.read_bytes() == data  # parent directories created
    assert t.get_file(str(target)) == data
    t.put_file(str(tmp_path / "empty"), b"")
    assert t.get_file(str(tmp_path / "empty")) == b""


def test_default_chunks_handle_a_large_file_in_few_requests(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path)
    data = os.urandom(1_500_000)
    before = len(client.exec_calls)
    t.put_file(str(tmp_path / "big"), data)  # every arg <= 128 KiB, body <= 1 MiB (enforced by the fake)
    assert t.get_file(str(tmp_path / "big")) == data
    assert len(client.exec_calls) - before <= 4


def test_file_errors_map_to_transport_errors(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path)
    with pytest.raises(TransportFileNotFoundError):
        t.get_file(str(tmp_path / "missing"))
    with pytest.raises(TransportIsADirectoryError, match="directory"):
        t.get_file(str(tmp_path))
    if os.geteuid() != 0:
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "f").write_text("x")
        (locked / "f").chmod(0o400)
        with pytest.raises(TransportPermissionError):
            t.put_file(str(locked / "f"), b"y")


def test_truncated_output_is_flagged(vm, tmp_path) -> None:
    client, vid = vm
    t = connect(client, vid, tmp_path)
    result = t.run("head -c 5000000 /dev/zero | tr '\\0' a")
    assert "truncated" in result.stderr


def test_connect_to_a_missing_microvm_fails_clearly() -> None:
    with pytest.raises(TransportError, match="does not exist"):
        ExecTransport(FakeMicroVMClient()).connect(host="nope", credentials=TransportCredentials(api_token="t"))  # type: ignore[arg-type]


def test_base_sandbox_operations_work_over_exec(vm, tmp_path) -> None:
    """deepagents' generated ls/read/write/edit/grep scripts, run through the wrapper."""
    client, vid = vm
    sandbox = MicroVMSandbox(microvm_id=vid, transport=connect(client, vid, tmp_path))
    path = str(tmp_path / "notes.md")
    assert sandbox.write(path, "alpha\nbeta\n").error is None
    assert "beta" in sandbox.read(path).file_data["content"]
    assert sandbox.edit(path, "beta", "gamma").occurrences == 1
    assert any(entry["path"] == path for entry in sandbox.ls(str(tmp_path)).entries)
    if _gnu_grep():  # deepagents passes -Z (GNU: NUL separator; BSD/macOS: decompress)
        matches = sandbox.grep("gamma", path=str(tmp_path)).matches
        assert matches and matches[0]["path"] == path
    response = sandbox.execute(f"wc -l < {path}")
    assert response.exit_code == 0 and response.output.strip() == "2"
