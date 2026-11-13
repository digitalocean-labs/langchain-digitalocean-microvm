"""Demo-day pre-flight: check the configuration, warm the template and the models, clear old team MicroVMs.

    uv run examples/10_erasure/prepare.py            # check + build/reuse the template + ping the models
    uv run examples/10_erasure/prepare.py --clean    # also delete every locator MicroVM (keeps the template), the old report and the dev server's saved runs

--clean also deletes .langgraph_api/ (`langgraph dev` keeps unfinished runs there and re-queues them on start, which
recreates MicroVMs and blocks new requests). Stop `langgraph dev` first: with the server running, that step is skipped.

Run it 10-15 minutes before going on stage. It ends with a READY / NOT READY line.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from agent import SUBAGENTS, lifecycle, templates
from gateway import POLICY
from models import model, strong_model
from tools import REPORT

from langchain_digitalocean_microvm.client import (
    SUBAGENT_SANDBOX_TAG,
    sandbox_id_from_tags,
)
from langchain_digitalocean_microvm.owner import is_mine
from langchain_digitalocean_microvm.status import MicroVMStatus

DEV_STATE = Path(__file__).resolve().parent / ".langgraph_api"  # langgraph dev's saved threads and runs
ROLES = [spec["name"] for spec in SUBAGENTS]


def is_locator(sandbox_id: str) -> bool:
    """A `<thread>-<role>` sandbox of this example's roles."""
    return any(sandbox_id.endswith(f"-{role}") for role in ROLES)


def dev_server_running() -> bool:
    """True if a `langgraph dev` process is running (it would rewrite the state we delete)."""
    try:
        return subprocess.run(["pgrep", "-f", "langgraph dev"], capture_output=True, check=False).returncode == 0
    except OSError:  # no pgrep (Windows): assume it is not
        return False


OK, BAD, WARN, RESET = "\033[32m✔\033[0m", "\033[31m✘\033[0m", "\033[33m!\033[0m", "\033[0m"


def check(label: str, ok: bool | None, detail: str = "") -> bool:
    """Print one pre-flight line. ok: True = pass, False = fail, None = warning only. Returns False only on a fail."""
    print(f" {OK if ok else (WARN if ok is None else BAD)} {label}{f'  ({detail})' if detail else ''}")
    return ok is not False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true", help="delete every locator MicroVM and the old report first")
    args = parser.parse_args()
    ready = True
    local = lifecycle.config.transport == "local"

    print("Configuration")
    ready &= check("MicroVM mode", True, "local (directories on this machine)" if local else f"real · size {lifecycle.config.size}")
    if not local:
        ready &= check("DO_MICROVM_API_TOKEN", bool(os.environ.get("DO_MICROVM_API_TOKEN")))
    ready &= check("model access", bool(os.environ.get("DO_MODEL_ACCESS_KEY") or os.environ.get("EXAMPLES_MODEL")),
                   "DO Serverless Inference" if os.environ.get("DO_MODEL_ACCESS_KEY") else os.environ.get("EXAMPLES_MODEL", ""))
    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))
    check("LangSmith tracing", tracing or None, f"project {os.environ.get('LANGSMITH_PROJECT', 'default')}" if tracing
          else "off: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to show traces")
    check("systems", True, "five mocks in memory (world.py) behind the local Action Gateway")

    if args.clean:
        print("\nLocal state")
        if REPORT.exists():
            REPORT.unlink()
            check("old report deleted", True, str(REPORT.name))
        if DEV_STATE.exists():
            if dev_server_running():
                check("langgraph dev state (.langgraph_api/)", None, "not deleted: stop `langgraph dev`, then run --clean again")
            else:
                shutil.rmtree(DEV_STATE)
                check("langgraph dev state (.langgraph_api/) deleted", True, "saved threads and unfinished runs cleared")

    if not local:
        print("\nMicroVMs")
        client = lifecycle._client_or_raise()
        team = [s for s in client.list(tag=SUBAGENT_SANDBOX_TAG)
                if s.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)
                and is_mine(s.tags)
                and is_locator(sandbox_id_from_tags(s.tags) or "")]
        if args.clean and team:
            for state in team:
                client.delete(state.id)
            check("old locator MicroVMs deleted", True, str(len(team)))
            team = []
        check("locator MicroVMs left from earlier runs", not team or None,
              f"{len(team)}: run with --clean to delete them" if team else "none")

    print("\nSandbox template (the classifier)")
    started = time.monotonic()
    try:
        template = templates.get()
        ready &= check("template", True, f"{template.checkpoint_name} {'built' if template.built else 'reused'} in "
                       f"{time.monotonic() - started:.1f} s")
    except Exception as exc:  # noqa: BLE001
        ready &= check("template", False, str(exc)[:200])

    print("\nModels (first calls are slower; this warms them)")
    for label, chat in (("coordinator", strong_model), ("locators", model)):
        started = time.monotonic()
        try:
            chat.invoke("Reply with OK.")
            ready &= check(label, True, f"{getattr(chat, 'model', '')} answered in {time.monotonic() - started:.1f} s")
        except Exception as exc:  # noqa: BLE001
            ready &= check(label, False, str(exc)[:200])

    print("\nAction Gateway (local)")
    check("policy", True, f"ask: {len(POLICY.ask)} erasures; deny: {', '.join(n.split('@')[0] for n in POLICY.deny)}")
    print(f"\nTeam: {', '.join(ROLES)}")
    print("\n\033[1m" + ("READY" if ready else "NOT READY") + RESET)


if __name__ == "__main__":
    main()
