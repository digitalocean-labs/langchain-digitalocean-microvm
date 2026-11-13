"""The isolation moment: wipe one specialist's MicroVM with `rm -rf /`, show nothing else noticed, heal it.

1. picks the newest VM of `--role` (default analyst-asia-pacific) in the team,
2. runs `rm -rf --no-preserve-root /` in it, through the MicroVMs exec API,
3. shows that VM can no longer even run `ls`, while a sibling specialist's VM and this machine are
   untouched,
4. with `--heal`, deletes the wrecked VM: the next request that needs that specialist clones a
   fresh one from the template in seconds.

REAL MicroVMs ONLY. In local mode a "sandbox" is a directory on this machine and the command would
run here, so the script refuses.

RUN (from the repository root)
    uv run examples/08_board_pack/chaos.py                    # wreck the APAC analyst's VM (asks first)
    uv run examples/08_board_pack/chaos.py --heal             # then delete it
    uv run examples/08_board_pack/chaos.py --role auditor --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")

from dashboard import ROLES, split
from events import log_event

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle
from langchain_digitalocean_microvm.client import (
    SUBAGENT_SANDBOX_TAG,
    sandbox_id_from_tags,
)
from langchain_digitalocean_microvm.status import MicroVMStatus

RED, GREEN, DIM, BOLD, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def team(client) -> list[tuple[str, str, str, object]]:
    """(thread, role, sandbox_id, state) of every live specialist VM, newest first."""
    out = []
    for state in client.list(tag=SUBAGENT_SANDBOX_TAG):
        if state.status in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED):
            continue
        sandbox_id = sandbox_id_from_tags(state.tags) or ""
        parts = split(sandbox_id)
        if parts:
            out.append((parts[0], parts[1], sandbox_id, state))
    return sorted(out, key=lambda item: item[3].created_at.timestamp() if item[3].created_at else 0.0, reverse=True)


def run(client, microvm_id: str, script: str) -> tuple[int, str]:
    """Run `script` in the VM through the exec API; returns (exit code, output), or (-1, reason) if it cannot run."""
    try:
        result = client.exec(microvm_id, ["sh", "-c", script])
    except Exception as exc:  # noqa: BLE001 - a wrecked VM may not run anything at all
        status = getattr(exc, "status_code", None)
        return -1, f"the VM cannot run commands any more (exec API answered {status or type(exc).__name__})"
    return result.exit_code, (result.stdout + result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", default="analyst-asia-pacific", choices=ROLES)
    parser.add_argument("--heal", action="store_true", help="delete the role's newest VM (after wrecking it)")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    config = MicroVMConfig.from_env()
    if config.transport == "local":
        sys.exit("Refusing: in local mode the sandbox is this machine. Run with real MicroVMs.")
    lifecycle = MicroVMLifecycle(config)
    client = lifecycle._client_or_raise()
    members = team(client)
    target = next((m for m in members if m[1] == args.role), None)
    if target is None:
        sys.exit(f"No live VM for {args.role}. Ask the team something that needs it first.")
    thread, role, sandbox_id, state = target

    if args.heal:
        lifecycle.delete(sandbox_id=sandbox_id)
        log_event(event="delete", sandbox_id=sandbox_id)
        print(f"{GREEN}Deleted {role}'s VM {state.id}.{RESET} The next request that needs {role} clones a fresh one "
              "from the template.")
        return

    # A sibling in the same thread proves the blast radius: prefer another analyst, else any specialist.
    siblings = [m for m in members if m[0] == thread and m[1] != role]
    sibling = next((m for m in siblings if m[1].startswith("analyst-")), siblings[0] if siblings else None)
    print(f"{BOLD}Target:{RESET} {role} · VM {state.id} · thread {thread} · {state.status.value}")
    code, out = run(client, state.id, "ls / && echo && ls /data /tools")
    print(f"{DIM}$ ls /   (inside {role}'s VM){RESET}\n{out}\n")
    if not args.yes and input(f"Run {RED}rm -rf --no-preserve-root /{RESET} inside {role}'s VM? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")

    print(f"{DIM}$ rm -rf --no-preserve-root /   (inside {role}'s VM){RESET}")
    run(client, state.id, "rm -rf --no-preserve-root / 2>/dev/null; true")
    log_event(event="chaos", sandbox_id=sandbox_id, microvm_id=state.id)
    code, out = run(client, state.id, "ls /")
    print(f"{RED}$ ls /   (inside {role}'s VM, after):{RESET} {out[:200] or '(nothing)'}\n")

    if sibling:
        code, out = run(client, sibling[3].id, "ls /data /tools")
        mark = GREEN if code == 0 else RED
        print(f"{mark}{sibling[1]}'s VM (same thread, same template):{RESET} exit {code}\n{out}\n")
    print(f"{GREEN}This machine:{RESET} untouched ({Path.home()} still has {sum(1 for _ in Path.home().iterdir())} entries).")
    print(f"\nHeal it: uv run examples/08_board_pack/chaos.py --role {role} --heal")


if __name__ == "__main__":
    main()
