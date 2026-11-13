"""agent_host/_guard.py: a local mode host must not outlive the process that spawned it.

Real processes, no `langgraph dev`: the "host" is a sleeping Python child
tagged with a unique marker so it can be found (or, better, not found) by
`pgrep -f`. The "supervisor" is a throwaway interpreter that spawns the guard
and then dies abruptly with os._exit, the way a uvicorn reload worker dies by
a re-raised SIGTERM.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid

import pytest

from langchain_digitalocean_microvm.agent_host.local import GUARD_PATH

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and pgrep")


def _pgrep(marker: str) -> list[int]:
    result = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True, check=False)
    return [int(line) for line in result.stdout.split()]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(marker: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while _pgrep(marker):
        if time.monotonic() > deadline:
            raise AssertionError(f"child tagged {marker} still alive after {timeout}s: {_pgrep(marker)}")
        time.sleep(0.2)


def _child_command(marker: str) -> list[str]:
    return [sys.executable, "-c", f"import time\n# {marker}\ntime.sleep(60)"]


def test_guard_terminates_the_child_when_its_parent_dies() -> None:
    marker = f"guard-test-{uuid.uuid4().hex}"
    supervisor = f"""
import os, subprocess, sys, time
guard = subprocess.Popen(
    [sys.executable, {str(GUARD_PATH)!r}, "--parent", str(os.getpid()), "--interval", "0.1", "--"] + {_child_command(marker)!r},
    start_new_session=True,
)
child_seen = False
for _ in range(50):  # wait until the child exists
    if subprocess.run(["pgrep", "-f", {marker!r}], capture_output=True).returncode == 0:
        child_seen = True
        break
    time.sleep(0.1)
print(guard.pid, int(child_seen), flush=True)
os._exit(0)  # no atexit, no cleanup: exactly how a uvicorn reload worker ends
"""
    out = subprocess.run([sys.executable, "-c", supervisor], capture_output=True, text=True, check=True, timeout=30)
    guard_pid, child_seen = (int(part) for part in out.stdout.split())
    assert child_seen, "child never started while the supervisor was alive"

    _wait_gone(marker, timeout=10.0)
    deadline = time.monotonic() + 10.0
    while _alive(guard_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(guard_pid)


def test_guard_exits_with_the_childs_code_and_forwards_signals() -> None:
    marker = f"guard-test-{uuid.uuid4().hex}"
    # 1. child exits on its own -> guard returns its exit code
    guard = subprocess.Popen(
        [sys.executable, str(GUARD_PATH), "--parent", str(os.getpid()), "--interval", "0.1", "--", sys.executable, "-c", "raise SystemExit(7)"],
    )
    assert guard.wait(timeout=20) == 7

    # 2. SIGTERM to the guard alone is forwarded to the child
    guard = subprocess.Popen(
        [sys.executable, str(GUARD_PATH), "--parent", str(os.getpid()), "--interval", "0.1", "--", *_child_command(marker)],
        start_new_session=True,
    )
    deadline = time.monotonic() + 10.0
    while not _pgrep(marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _pgrep(marker), "child never started"
    guard.send_signal(signal.SIGTERM)
    _wait_gone(marker, timeout=10.0)
    assert guard.wait(timeout=10) == 128 + signal.SIGTERM  # the child died by the forwarded SIGTERM
