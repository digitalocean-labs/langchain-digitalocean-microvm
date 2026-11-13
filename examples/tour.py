"""A guided tour of the package in local mode: examples in a row, no DigitalOcean account.

Runs example 01 (a sandbox that pauses and resumes) and 10 (approvals and a data-erasure workflow over
simulated systems), about 1 to 2 minutes; `--full` adds 03 (one sandbox per subagent) and 06 (templates and
warm clones), longer. Each step has a line on what it shows and the tail of its output. Sandboxes are
local directories under a temporary directory, and so is the store; both are removed at the end.

Needs a model: `DO_MODEL_ACCESS_KEY`, or `EXAMPLES_MODEL` with its provider's key (see
`examples/README.md`); `uv run python scripts/check_setup.py --local` checks it. A local mode agent runs
its shell commands on this machine, in the temporary directory. Example 10 also writes its report under
`examples/10_erasure/outputs/` (ignored by git), which the tour leaves in place.

    uv run examples/tour.py                 # 01 and 10, the last 20 lines of each
    uv run examples/tour.py --full          # all four steps
    uv run examples/tour.py --only 2,3      # chosen steps (numbers below)
    uv run examples/tour.py --verbose       # stream every line
"""

from __future__ import annotations

import argparse
import os
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


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    shows: str
    script: str
    args: tuple[str, ...] = ()
    quick: bool = False  # in the default tour


STEPS = (
    Step(1, "Quickstart (example 01)",
         "A deep agent whose shell and files are a sandbox. It writes a file, the sandbox is paused and resumed, "
         "and the file is still there.",
         "01_quickstart/quickstart.py", quick=True),
    Step(2, "One sandbox per subagent (example 03)",
         "Synchronous subagents each get a sandbox of their own, created in parallel and paused between turns.",
         "03_subagent_sandboxes/run_script.py", ("--delete",)),
    Step(3, "Templates and warm clones (example 06)",
         "A sandbox is prepared once, checkpointed, and new sandboxes are cloned from it already set up.",
         "06_warm_start/warm_start.py"),
    Step(4, "Approvals over simulated systems (example 10)",
         "A data-erasure request across five simulated systems: locators work in their own sandboxes, each "
         "erasure waits for an approval (here: all approved but one denied), and a verified report is written.",
         "10_erasure/run_script.py", ("--approve", "--deny-system", "marketing", "--delete"), quick=True),
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
    env.update({
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
    })
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", help="comma-separated step numbers, e.g. 2,3 (default: the quick ones, 1 and 4)")
    parser.add_argument("--full", action="store_true", help="all four steps")
    parser.add_argument("--lines", type=int, default=20, help="lines of output shown per step (default: 20)")
    parser.add_argument("--verbose", action="store_true", help="stream every line of every step")
    args = parser.parse_args(argv)

    steps = select(args.only, full=args.full)
    ok, missing = model_variables(merged_env(ROOT / ".env"))
    if not ok:
        print(f"The tour needs a model: set {missing}.\nSee examples/README.md, then "
              "`uv run python scripts/check_setup.py --local`.")
        return 2
    print("Local mode: no DigitalOcean account. The agents' commands run on this machine, in a temporary directory.")

    results: list[tuple[Step, bool, float]] = []
    with tempfile.TemporaryDirectory(prefix="microvm-tour-") as tmp:
        env = tour_env(Path(tmp))
        for step in steps:
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
        print("\n`--full` adds examples 03 (subagent sandboxes) and 06 (templates and warm clones).")
    print("\nNext: run them on real MicroVMs (`uv run python scripts/check_setup.py`), then examples/README.md.")
    return 0 if results and all(p for _, p, _ in results) and len(results) == len(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
