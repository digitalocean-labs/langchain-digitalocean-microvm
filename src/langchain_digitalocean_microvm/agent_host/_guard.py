"""Run a local mode host and make sure it dies with the process that spawned it.

LocalDevServerHost starts every local `langgraph dev` through this script:

    python _guard.py --parent <supervisor pid> -- python -m langgraph_cli dev ...

Why a guard instead of an exit hook: the supervisor may itself be a
`langgraph dev` reload worker, and uvicorn ends those by restoring the default
signal handlers after its graceful shutdown and re-raising the captured
SIGTERM. The interpreter then dies by signal: no `atexit`, no
`threading._shutdown`, no finaliser ever runs, and every local host would be
orphaned on each reload or Ctrl-C. The guard polls `os.getppid()` instead;
once the parent is gone it terminates the child, waits `--grace` seconds,
kills it, and exits.

The guard shares the host's process group, so LocalDevServerHost's
SIGSTOP/SIGCONT/SIGTERM via `killpg` reach both (a stopped guard cannot
notice its parent dying, which is why a hard-killed supervisor can leave a
paused host behind: `pkill -CONT -f microvm_agent_local; pkill -f
microvm_agent_local`). SIGTERM/SIGINT/SIGHUP delivered to the guard alone are
forwarded to the child. It exits with the child's exit code (128 + signal
number when the child died by a signal, the shell convention).

Stdlib only, run by path: importing `langchain_digitalocean_microvm` here would drag
deepagents and starlette into every host for nothing.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

DEFAULT_GRACE = 10.0
DEFAULT_INTERVAL = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "Run a command; stop it when the parent exits.").splitlines()[0])
    parser.add_argument("--parent", type=int, required=True, help="pid of the process this host must not outlive")
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE, help="seconds between SIGTERM and SIGKILL")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="seconds between parent checks")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command to run")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after --")

    child = subprocess.Popen(command)  # noqa: S603 - argv comes from LocalDevServerHost

    def forward(sig: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(sig)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)

    while True:
        if child.poll() is not None:
            return _exit_code(child)
        if os.getppid() != args.parent:
            _stop(child, args.grace)
            return _exit_code(child)
        time.sleep(args.interval)


def _exit_code(child: subprocess.Popen) -> int:
    code = child.returncode if child.returncode is not None else 0
    return code if code >= 0 else 128 - code  # died by signal N -> 128 + N


def _stop(child: subprocess.Popen, grace: float) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


if __name__ == "__main__":
    sys.exit(main())
