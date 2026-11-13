"""Pause every running MicroVM you created with this package (tag `microvm-sandbox`, your owner tag, your
namespace); other users' MicroVMs on the team and everything else are not touched.

Port of the course's python/m5/sales_assistant_sandbox/stop_sandboxes.py. A
paused MicroVM keeps its memory and files, which is the point of the
quickstart; to delete everything instead, use scripts/delete_all.py.

    uv run examples/01_quickstart/pause_all.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_digitalocean_microvm import MicroVMStatus
from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.namespace import list_owned, validate_namespace
from langchain_digitalocean_microvm.owner import is_mine

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def main() -> None:
    token = os.environ.get("DO_MICROVM_API_TOKEN")
    if not token:
        print("DO_MICROVM_API_TOKEN is not set: nothing to pause (local mode creates no MicroVMs).")
        return
    client = MicroVMClient(token)
    namespace = validate_namespace(os.environ.get("DO_MICROVM_NAMESPACE"))
    running = [state for state in list_owned(client, namespace=namespace)
               if state.status == MicroVMStatus.RUNNING and is_mine(state.tags)]
    if not running:
        print("no running MicroVMs created by you in this namespace")
    for state in running:
        try:
            client.pause(state.id)
            print(f"paused {state.name} ({state.id})")
        except Exception as exc:  # keep going: one failure should not leave the rest running
            print(f"could not pause {state.name} ({state.id}): {exc}")


if __name__ == "__main__":
    main()
