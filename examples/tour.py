"""A guided tour of the package in local mode: examples in a row, no DigitalOcean account.

The default tour runs example 11 with its live dashboard, about 2 minutes: a security lead, one sandbox per
analysed repository, a row per repository and every approval and refusal. On macOS the dashboard opens in a
window of its own (Terminal, or iTerm when the tour runs in iTerm) and stays open after the run; elsewhere, or
with `--same-terminal`, it runs full screen in this terminal and closes a few seconds after the agent ends.
The agent's output then ends with the gateway audit and the report's path. `--full` adds examples 01 (a sandbox
that pauses and resumes), 03 (one sandbox per subagent), 06 (templates and warm clones) and 10 (approvals and a
data-erasure workflow over simulated systems), about 10 minutes in all. A step without a dashboard, or any step
with `--verbose` or when the output is not a terminal, shows a line on what it shows and the tail of its output.
Sandboxes are local directories under a temporary directory, and so is the store; both are removed at the end.

Needs a model: `DO_MODEL_ACCESS_KEY`, or `EXAMPLES_MODEL` with its provider's key (see
`examples/README.md`); `uv run python scripts/check_setup.py --local` checks it. A local mode agent runs
its shell commands on this machine, in the temporary directory. Examples 10 and 11 also write their reports
and event logs under their `outputs/` directories (ignored by git), which the tour leaves in place.

    uv run examples/tour.py                   # example 11 with its dashboard
    uv run examples/tour.py --same-terminal   # the dashboard in this terminal, not in a new window
    uv run examples/tour.py --full            # all five steps
    uv run examples/tour.py --only 2,3        # chosen steps (numbers below)
    uv run examples/tour.py --verbose         # stream every line, no dashboard
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_setup import merged_env, model_variables  # noqa: E402

THREAD = "tour"  # the thread id the agent and its dashboard share


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    shows: str
    script: str
    args: tuple[str, ...] = ()
    quick: bool = False  # in the default tour
    dashboard: str | None = None  # a live view of the step, started with `--thread tour` before the script


STEPS = (
    Step(1, "Quickstart (example 01)",
         "A deep agent whose shell and files are a sandbox. It writes a file, the sandbox is paused and resumed, "
         "and the file is still there.",
         "01_quickstart/quickstart.py"),
    Step(2, "One sandbox per subagent (example 03)",
         "Synchronous subagents each get a sandbox of their own, created in parallel and paused between turns.",
         "03_subagent_sandboxes/run_script.py", ("--delete",)),
    Step(3, "Templates and warm clones (example 06)",
         "A sandbox is prepared once, checkpointed, and new sandboxes are cloned from it already set up.",
         "06_warm_start/warm_start.py"),
    Step(4, "Approvals over simulated systems (example 10)",
         "A data-erasure request across five simulated systems: locators work in their own sandboxes, each "
         "erasure waits for an approval (here: all approved but one denied), and a verified report is written.",
         "10_erasure/run_script.py", ("--approve", "--deny-system", "marketing", "--delete")),
    Step(5, "A team of agents with approvals (example 11)",
         "A security advisory against a library: a security lead starts one analyst per repository, each in a "
         "sandbox of its own, to find out whether the repository is affected, whether outside input reaches the "
         "vulnerable call and whether its tests pass with the fix. Every issue and the chat summary wait for an "
         "approval (here: all approved but one repository's issue denied), and the policy refuses closing an issue.",
         "11_vulnerability_response/run_script.py", ("--approve", "--deny-repo", "report-worker", "--delete"),
         quick=True, dashboard="11_vulnerability_response/dashboard.py"),
)

# What tour_env sets: the only variables a dashboard in a new window gets on top of the login shell's.
TOUR_VARIABLES = (
    "DO_MICROVM_TRANSPORT", "DO_MICROVM_HOST_MODE", "DO_MICROVM_API_TOKEN", "DO_MICROVM_ACCESS_TOKEN",
    "DO_ACTION_GATEWAY_TOKEN", "DO_MICROVM_NAMESPACE", "TMPDIR", "HOME", "XDG_STATE_HOME", "LOCALAPPDATA",
    "PYTHONUNBUFFERED",
)


def select(only: str | None, steps: Sequence[Step] = STEPS, *, full: bool = False) -> list[Step]:
    if not only:
        return [s for s in steps if full or s.quick]
    wanted = {int(n) for n in only.split(",") if n.strip()}
    unknown = wanted - {s.number for s in steps}
    if unknown:
        raise SystemExit(f"No such step: {sorted(unknown)} (1 to {len(steps)})")
    return [s for s in steps if s.number in wanted]


def tour_env(tmp: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Local mode, every file the run makes under `tmp`, and no real credential for a step to pick up from `.env`."""
    state = tmp / "state"
    env = dict(os.environ if base is None else base)
    values = {  # keep TOUR_VARIABLES in step
        "DO_MICROVM_TRANSPORT": "local",
        "DO_MICROVM_HOST_MODE": "local",
        # an empty value wins over the repository's .env (the examples load it without overriding the shell)
        "DO_MICROVM_API_TOKEN": "",
        "DO_MICROVM_ACCESS_TOKEN": "",
        "DO_ACTION_GATEWAY_TOKEN": "",
        "DO_MICROVM_NAMESPACE": "",
        "TMPDIR": str(tmp),  # the local sandboxes are directories under the temporary directory
        "HOME": str(state),  # the gateway store and the owner id live in the user's state directory
        "XDG_STATE_HOME": str(state),
        "LOCALAPPDATA": str(state),
        "PYTHONUNBUFFERED": "1",
    }
    env.update(values)
    return env


def run_step(step: Step, env: dict[str, str], *, lines: int, verbose: bool,
             python: str = sys.executable, base: Path = HERE) -> tuple[bool, float]:
    print(f"\n=== {step.number}. {step.title}\n{step.shows}")
    command = [python, str(base / step.script), *step.args]
    print(f"$ uv run examples/{step.script} {' '.join(step.args)}".rstrip())
    started = time.monotonic()
    if verbose:
        done = subprocess.run(command, env=env, cwd=ROOT, check=False)
        return done.returncode == 0, time.monotonic() - started
    done = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, check=False)
    output = (done.stdout + done.stderr).rstrip().splitlines()
    shown = output[-lines:]
    if len(output) > len(shown):
        print(f"... ({len(output) - len(shown)} earlier lines; --verbose shows all)")
    print("\n".join(shown))
    return done.returncode == 0, time.monotonic() - started


def window_script(command: str, app: str) -> str:
    """AppleScript that opens a new Terminal or iTerm window running `command` in the login shell."""
    quoted = command.replace("\\", "\\\\").replace('"', '\\"')
    if app == "iTerm":
        return ('tell application "iTerm"\n  activate\n  set w to (create window with default profile)\n'
                f'  tell current session of w to write text "{quoted}"\nend tell')
    return f'tell application "Terminal"\n  activate\n  do script "{quoted}"\nend tell'


def open_window(argv: Sequence[str], env: dict[str, str], pidfile: Path, *, timeout: float = 20.0) -> bool:
    """Run `argv` in a new macOS terminal window with the tour's variables; True once the window's shell has started."""
    if sys.platform != "darwin" or shutil.which("osascript") is None:
        return False
    pidfile.unlink(missing_ok=True)
    variables = [f"{k}={env[k]}" for k in TOUR_VARIABLES if k in env]
    command = (f"cd {shlex.quote(str(ROOT))} && echo $$ > {shlex.quote(str(pidfile))} && clear && "
               + shlex.join(["env", *variables, *argv]))
    app = "iTerm" if os.environ.get("TERM_PROGRAM") == "iTerm.app" else "Terminal"
    done = subprocess.run(["osascript", "-e", window_script(command, app)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pidfile.is_file() and pidfile.read_text().strip():
            return True
        time.sleep(0.2)
    return False


def closing_lines(output: list[str], lines: int) -> list[str]:
    """The output from the gateway audit on, without the `deleted ...` lines; the last `lines` lines without an audit."""
    start = next((i for i, line in enumerate(output) if "gateway audit" in line), None)
    if start is None:
        return output[-lines:]
    return [line for line in output[start:] if not line.startswith("deleted ")]


def stop(process: subprocess.Popen[bytes], *, wait: float = 10.0) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_watched(step: Step, env: dict[str, str], *, tmp: Path, same_terminal: bool = False, lines: int = 20,
                python: str = sys.executable, base: Path = HERE, settle: float = 3.0,
                linger: float = 6.0) -> tuple[bool, float]:
    """Run a step with its dashboard: in a new window on macOS, otherwise full screen in this terminal."""
    assert step.dashboard is not None
    print(f"\n=== {step.number}. {step.title}\n{step.shows}")
    dashboard = [python, str(base / step.dashboard), "--thread", THREAD]
    command = [python, str(base / step.script), "--thread", THREAD, *step.args]
    print(f"$ uv run examples/{step.dashboard} --thread {THREAD}")
    print(f"$ uv run examples/{step.script} --thread {THREAD} {' '.join(step.args)}".rstrip())
    viewer: subprocess.Popen[bytes] | None = None
    if not same_terminal and open_window(dashboard, env, tmp / "dashboard.pid"):
        print("The dashboard is in a new window and stays open after the run, so its last frame stays readable "
              "(Ctrl-C there closes it). The first time, macOS asks to allow the terminal app to be automated.")
    else:
        viewer = subprocess.Popen(dashboard, env=env, cwd=ROOT)  # follows the event log from now on
    log = tmp / f"step-{step.number}.log"
    started = time.monotonic()
    try:
        time.sleep(settle)
        with log.open("w") as out:
            done = subprocess.run(command, env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=False)
        seconds = time.monotonic() - started
        if viewer is not None:
            time.sleep(linger)
    finally:
        if viewer is not None:
            stop(viewer)
    print("\n".join(closing_lines(log.read_text().rstrip().splitlines(), lines)))
    return done.returncode == 0, seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", help="comma-separated step numbers, e.g. 2,3 (default: 5, example 11 with its dashboard)")
    parser.add_argument("--full", action="store_true", help="all five steps, about 10 minutes")
    parser.add_argument("--lines", type=int, default=20, help="lines of output shown per step (default: 20)")
    parser.add_argument("--verbose", action="store_true", help="stream every line of every step, no dashboard")
    parser.add_argument("--same-terminal", action="store_true",
                        help="run a step's dashboard full screen in this terminal, not in a new window (macOS)")
    args = parser.parse_args(argv)

    steps = select(args.only, full=args.full)
    ok, missing = model_variables(merged_env(ROOT / ".env"))
    if not ok:
        print(f"The tour needs a model: set {missing}.\nSee examples/README.md, then "
              "`uv run python scripts/check_setup.py --local`.")
        return 2
    print("Local mode: no DigitalOcean account. The agents' commands run on this machine, in a temporary directory.")

    watch = sys.stdout.isatty() and not args.verbose
    results: list[tuple[Step, bool, float]] = []
    with tempfile.TemporaryDirectory(prefix="microvm-tour-") as tmp:
        env = tour_env(Path(tmp))
        for step in steps:
            if step.dashboard and watch:
                passed, seconds = run_watched(step, env, tmp=Path(tmp), same_terminal=args.same_terminal, lines=args.lines)
            else:
                passed, seconds = run_step(step, env, lines=args.lines, verbose=args.verbose)
            results.append((step, passed, seconds))
            if not passed:
                print(f"\nStep {step.number} failed; the rest are skipped. Re-run it alone with --only {step.number}"
                      " --verbose.")
                break
    print("\n=== Summary")
    for step, passed, seconds in results:
        print(f"{'ok  ' if passed else 'FAIL'}  {step.number}. {step.title}  ({seconds:.0f} s)")
    if not args.full and not args.only:
        print("\n`--full` adds examples 01 (quickstart), 03 (subagent sandboxes), 06 (templates and warm clones) "
              "and 10 (data erasure).")
    print("\nNext: run them on real MicroVMs (`uv run python scripts/check_setup.py`), then examples/README.md.")
    return 0 if results and all(p for _, p, _ in results) and len(results) == len(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
