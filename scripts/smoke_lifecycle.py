"""Live smoke test for the host lifecycle on real MicroVMs (at most one MicroVM at a time).

1. TTL in the gateway: an embedded SubAgentGateway on MicroVMAgentHost with a
   short `release_after` runs one `echo` task that saves a file, lets its own
   poll loop pause the host and then delete it, and checks that the task is
   still answered from the cache while its artifacts answer 410 (reason: TTL).
2. The sweeper (scripts/gc_microvms.py): an agent host created behind the
   gateway's back (no store row) is planned as an orphan and deleted; only
   items this script created are applied, so nothing else in the team is
   touched.

Refuses to start while MicroVMs of its namespace (DO_MICROVM_NAMESPACE) exist.
Deletes everything it created in a `finally` block. Tokens come from
.env; results go to ~/.langchain_digitalocean_microvm/probe/<date>/smoke_lifecycle.json.

    uv run python \\
        scripts/smoke_lifecycle.py [--image <ref>]
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from _env import (  # scripts/_env.py
    add_env_file_argument,
    env_namespace,
    load_env,
    refuse_if_owned,
)
from langgraph_sdk import get_sync_client

from langchain_digitalocean_microvm import MicroVMConfig
from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    MicroVMAgentHost,
    SubAgentGateway,
)
from langchain_digitalocean_microvm.agent_host.host import HOST_NAME_PREFIX
from langchain_digitalocean_microvm.agent_host.sweeper import apply_sweep, plan_sweep
from langchain_digitalocean_microvm.client import MicroVMClient, microvm_name
from langchain_digitalocean_microvm.namespace import namespaced_prefix
from langchain_digitalocean_microvm.status import MicroVMStatus

TERMINAL = {"success", "error", "timeout", "interrupted", "cancelled"}
GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=None)
    parser.add_argument("--release-after", type=float, default=30.0, help="per-task host TTL in seconds (default 30)")
    add_env_file_argument(parser)
    args = parser.parse_args()
    env = load_env(args.env_file)
    image = args.image or env.get("DO_MICROVM_AGENT_IMAGE")
    if not (env.get("DO_MICROVM_API_TOKEN") and env.get("DO_MICROVM_ACCESS_TOKEN") and image):
        print("need DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN and --image / DO_MICROVM_AGENT_IMAGE", file=sys.stderr)
        return 2

    namespace = env_namespace(env)
    config = MicroVMConfig(transport="exec", api_token=env["DO_MICROVM_API_TOKEN"], access_token=env["DO_MICROVM_ACCESS_TOKEN"],
                           agent_image=image, idle_timeout="5m", ready_timeout=420.0, namespace=namespace)
    hosts = MicroVMAgentHost(config)
    control = MicroVMClient(env["DO_MICROVM_API_TOKEN"])
    if refuse_if_owned(control, namespace):
        return 3
    gw_token = secrets.token_urlsafe(16)
    results: dict[str, Any] = {"image": image, "release_after": args.release_after}
    created_names: set[str] = set()
    t0 = time.monotonic()

    def step(name: str, value: Any) -> None:
        results[name] = value
        print(f"[{time.monotonic() - t0:6.1f}s] {name}: {json.dumps(value, default=str)[:300]}", flush=True)

    tmp = tempfile.TemporaryDirectory(prefix="smoke-lifecycle-")
    gateway = SubAgentGateway(hosts, {"echo": HostSpec(graph_id="echo")}, db_path=Path(tmp.name) / "gw.sqlite",
                              token=gw_token, poll_interval=5.0, pause_grace=5.0, release_after=args.release_after,
                              upstream_timeout=60.0)
    url = gateway.start_embedded()
    client = get_sync_client(url=url, api_key=gw_token, timeout=httpx.Timeout(600.0))
    http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {gw_token}"}, timeout=120.0)
    code = 0
    try:
        # ---- 1. TTL deletion by the gateway's own poll loop
        tid = client.threads.create()["thread_id"]
        created_names.add(microvm_name(namespaced_prefix(HOST_NAME_PREFIX, namespace), tid))
        run = client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "save a.txt: kept"}]})
        step("run_created", {"task": tid, "microvm": gateway.store.get_host(tid).microvm_id})
        deadline = time.monotonic() + 180
        while (status := client.runs.get(tid, run["run_id"])["status"]) not in TERMINAL:
            if time.monotonic() > deadline:
                raise AssertionError(f"run still {status}")
            time.sleep(2)
        step("run_status", status)
        step("artifacts_listed", http.get(f"/artifacts/{tid}").json().get("files"))
        done_at = time.monotonic()

        seen: list[str] = []
        deadline = time.monotonic() + args.release_after + 120
        while (row := gateway.store.get_host(tid)).status != "destroyed":
            if not seen or seen[-1] != row.status:
                seen.append(row.status)
                step("host_status", {"status": row.status, "after_s": round(time.monotonic() - done_at, 1)})
            if time.monotonic() > deadline:
                raise AssertionError(f"host still {row.status} after {args.release_after + 120}s")
            time.sleep(1)
        step("host_deleted", {"after_s": round(time.monotonic() - done_at, 1), "reason": row.retired_reason, "states": seen})
        state = control.get_by_name(microvm_name(namespaced_prefix(HOST_NAME_PREFIX, namespace), tid))
        step("microvm_after_ttl", state.status.value if state else "absent")
        step("check_after_ttl", client.runs.get(tid, run["run_id"])["status"])
        gone = http.get(f"/artifacts/{tid}")
        step("artifacts_after_ttl", {"status": gone.status_code, "detail": gone.json().get("detail")})
        if not (status == "success" and row.retired_reason == "ttl" and (state is None or state.status in GONE)
                and gone.status_code == 410 and results["check_after_ttl"] == "success"):
            code = 1

        # ---- 2. the sweeper finds a host created behind the gateway's back
        orphan_id = f"smoke-gc-{secrets.token_hex(3)}"
        created_names.add(microvm_name(namespaced_prefix(HOST_NAME_PREFIX, namespace), orphan_id))
        handle = hosts.get_or_create(host_id=orphan_id, spec=HostSpec(graph_id="echo"))
        step("orphan_created", handle.microvm_id)
        plan = plan_sweep(control, store=gateway.store, host_ttl=3600.0, orphan_ttl=0.0, namespace=namespace)
        ours = replace(
            plan,
            delete_microvms=tuple(i for i in plan.delete_microvms if i.name in created_names),
            delete_checkpoints=(),
            retire_rows=tuple(i for i in plan.retire_rows if i.id in (tid, orphan_id)),
        )
        step("sweep_plan", [(i.kind, i.name, i.reason) for i in ours.delete_microvms])
        swept = apply_sweep(ours, control, store=gateway.store)
        state = control.get_by_name(microvm_name(namespaced_prefix(HOST_NAME_PREFIX, namespace), orphan_id))
        step("orphan_after_sweep", {"deleted": [i.id for i in swept.deleted_microvms], "errors": swept.errors,
                                    "state": state.status.value if state else "absent"})
        if handle.microvm_id not in [i.id for i in swept.deleted_microvms] or swept.errors:
            code = 1
    except Exception as exc:  # noqa: BLE001 - report and clean up
        step("error", repr(exc))
        code = 1
    finally:
        http.close()
        gateway.stop()
        for name in created_names:
            state = control.get_by_name(name)
            if state is not None and state.status not in GONE:
                control.delete(state.id)
                step("cleanup_deleted", name)
        tmp.cleanup()
        out = Path.home() / ".langchain_digitalocean_microvm" / "probe" / date.today().isoformat()
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_lifecycle.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"results: {out / 'smoke_lifecycle.json'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
