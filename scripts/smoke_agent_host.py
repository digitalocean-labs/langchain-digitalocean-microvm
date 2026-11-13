"""Live smoke test: one real agent host behind an embedded SubAgentGateway.

Creates ONE MicroVM from DO_MICROVM_AGENT_IMAGE (the images/agent-host image)
through MicroVMAgentHost, then drives it the way the deepagents middleware
does, via the LangGraph SDK against the gateway:

  launch an `echo` run -> check while running -> result -> update (new run,
  interrupt) -> cancel a slow run -> the host's endpoint refuses requests that
  lack either credential (DO access token / host token) -> artifacts:
  `save NAME:` writes a file, a 9 MB file is written in the VM over exec,
  both are listed and fetched through the gateway (sha256 compared), again
  after pausing the host, and the gateway answers 410 once the host is
  deleted -> delete.

Refuses to start while MicroVMs of its namespace (DO_MICROVM_NAMESPACE) exist.
Deletes everything it created in a `finally` block. Tokens come from
.env (DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN); results go to
~/.langchain_digitalocean_microvm/probe/<date>/smoke_agent_host.json.

    uv run python \\
        scripts/smoke_agent_host.py --image <ref>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import tempfile
import time
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
from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.status import MicroVMStatus

TERMINAL = {"success", "error", "timeout", "interrupted", "cancelled"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=None)
    parser.add_argument("--warm", action="store_true",
                        help="warm start: build a golden checkpoint, then start a second host from it and time both")
    add_env_file_argument(parser)
    args = parser.parse_args()
    env = load_env(args.env_file)
    image = args.image or env.get("DO_MICROVM_AGENT_IMAGE")
    if not (env.get("DO_MICROVM_API_TOKEN") and env.get("DO_MICROVM_ACCESS_TOKEN") and image):
        print("need DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN and --image / DO_MICROVM_AGENT_IMAGE", file=sys.stderr)
        return 2

    namespace = env_namespace(env)
    config = MicroVMConfig(transport="exec", api_token=env["DO_MICROVM_API_TOKEN"], access_token=env["DO_MICROVM_ACCESS_TOKEN"],
                           agent_image=image, idle_timeout="5m", ready_timeout=420.0, warm_start=args.warm, namespace=namespace)
    hosts = MicroVMAgentHost(config)
    control = MicroVMClient(env["DO_MICROVM_API_TOKEN"])
    if refuse_if_owned(control, namespace):
        return 3
    gw_token = secrets.token_urlsafe(16)
    results: dict[str, Any] = {"image": image}
    t0 = time.monotonic()

    def step(name: str, value: Any) -> None:
        results[name] = value
        print(f"[{time.monotonic() - t0:6.1f}s] {name}: {json.dumps(value, default=str)[:300]}", flush=True)

    tmp = tempfile.TemporaryDirectory(prefix="smoke-agent-host-")
    gateway = SubAgentGateway(hosts, {"echo": HostSpec(graph_id="echo")}, db_path=Path(tmp.name) / "gw.sqlite",  # per-task hosts
                              token=gw_token, poll_interval=5.0, pause_grace=3600.0, upstream_timeout=60.0)
    url = gateway.start_embedded()
    client = get_sync_client(url=url, api_key=gw_token, timeout=httpx.Timeout(600.0))
    code = 0

    def wait(tid: str, rid: str, timeout: float = 120.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            run = client.runs.get(tid, rid)
            if run["status"] in TERMINAL or time.monotonic() > deadline:
                return run
            time.sleep(2)

    def say(tid: str, text: str, **kwargs: Any) -> dict[str, Any]:
        return client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": text}]}, **kwargs)

    try:
        tid = client.threads.create()["thread_id"]
        started = time.monotonic()
        run = say(tid, "sleep 6: ping")  # the gateway creates the MicroVM here
        step("first_run_created_seconds", round(time.monotonic() - started, 1))
        host_row = gateway.store.get_host(tid)
        step("host", {"url": host_row.url, "status": host_row.status})
        step("check_while_running", client.runs.get(tid, run["run_id"])["status"])
        finished = wait(tid, run["run_id"])
        step("result", {"status": finished["status"], "last": client.threads.get(tid)["values"]["messages"][-1]["content"]})

        run2 = say(tid, "pong", multitask_strategy="interrupt")
        step("update_result", {"status": wait(tid, run2["run_id"])["status"],
                               "last": client.threads.get(tid)["values"]["messages"][-1]["content"]})

        run3 = say(tid, "sleep 60: never")
        time.sleep(3)
        client.runs.cancel(tid, run3["run_id"])
        step("cancelled", wait(tid, run3["run_id"], timeout=30)["status"])

        edge = hosts.edge_headers()
        no_host_token = httpx.post(f"{host_row.url}/threads/search", json={}, headers=edge, timeout=60)
        no_edge_token = httpx.post(f"{host_row.url}/threads/search", json={}, headers={"x-api-key": host_row.token}, timeout=60)
        both = httpx.post(f"{host_row.url}/threads/search", json={}, headers={**edge, "x-api-key": host_row.token}, timeout=60)
        step("auth", {"edge_only": no_host_token.status_code, "host_token_only": no_edge_token.status_code, "both": both.status_code})
        step("is_idle_after_runs", hosts.is_idle(next(h for h in hosts.list_hosts() if h.host_id == host_row.host_id)))

        # Artifacts, through the gateway's routes (what fetch_subagent_artifacts calls).
        saved = wait(tid, say(tid, "save notes/hello.md: hello from a MicroVM")["run_id"])
        step("save_result", {"status": saved["status"], "last": client.threads.get(tid)["values"]["messages"][-1]["content"]})
        root = f"/work/tasks/{tid}/outputs"
        made = hosts._client_or_raise().exec(  # test setup only: write a file larger than one Range page
            host_row.microvm_id, ["sh", "-c", f"head -c 9000000 /dev/urandom > {root}/big.bin && sha256sum {root}/big.bin"])
        step("big_written_in_vm", {"exit_code": made.exit_code})
        api = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {gw_token}"}, timeout=300)
        started = time.monotonic()
        listing = api.get(f"/artifacts/{tid}")
        step("artifacts_list", {"status": listing.status_code, "files": listing.json().get("files"),
                                "seconds": round(time.monotonic() - started, 1)})
        hello = api.get(f"/artifacts/{tid}/files/notes/hello.md")
        step("artifact_small", {"status": hello.status_code, "content": hello.text})
        started = time.monotonic()
        big = api.get(f"/artifacts/{tid}/files/big.bin")
        step("artifact_big", {"status": big.status_code, "bytes": len(big.content),
                              "sha256_matches": hashlib.sha256(big.content).hexdigest() == made.stdout.split()[0],
                              "seconds": round(time.monotonic() - started, 1)})
        hosts.pause(host_id=host_row.host_id)
        gateway.store.update_host(host_row.host_id, status="paused")  # as if tick() had paused it
        started = time.monotonic()
        again = api.get(f"/artifacts/{tid}/files/notes/hello.md")
        step("artifact_after_pause", {"status": again.status_code, "content": again.text,
                                      "seconds": round(time.monotonic() - started, 1),
                                      "host_status_after": gateway.store.get_host(host_row.host_id).status})
        hosts.delete(host_id=host_row.host_id)  # behind the gateway's back: it learns from the endpoint
        started = time.monotonic()
        gone = api.get(f"/artifacts/{tid}")
        while gone.status_code == 200 and time.monotonic() - started < 120:  # a terminating VM may still answer
            time.sleep(2)
            gone = api.get(f"/artifacts/{tid}")
        step("artifacts_after_delete", {"status": gone.status_code, "detail": gone.json().get("detail"),
                                        "seconds": round(time.monotonic() - started, 1)})
        api.close()

        if args.warm:  # a second per-task host: created from the golden checkpoint
            tid2 = client.threads.create()["thread_id"]
            started = time.monotonic()
            run = say(tid2, "warm ping")
            step("second_host_first_run_created_seconds", round(time.monotonic() - started, 1))
            step("second_host_result", {"status": wait(tid2, run["run_id"])["status"],
                                        "last": client.threads.get(tid2)["values"]["messages"][-1]["content"]})
            row2 = gateway.store.get_host(tid2)
            stale = httpx.post(f"{row2.url}/threads/search", json={}, headers={**edge, "x-api-key": host_row.token}, timeout=60)
            step("second_host_rejects_first_hosts_token", stale.status_code)
            step("checkpoints", [(c.name, c.status) for c in control.list_checkpoints()])
    except Exception as exc:  # noqa: BLE001 - report and still clean up
        step("error", f"{type(exc).__name__}: {exc}")
        for state in control.list(tag="microvm-agenthost"):
            if state.status == MicroVMStatus.FAILED:
                step("failure_reason", state.failure_reason)
        code = 1
    finally:
        gateway.stop()
        tmp.cleanup()
        leftovers = [s for s in control.list(tag="microvm-agenthost")
                     if s.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)]
        leftovers += [s for s in control.list(tag="microvm-golden") if s.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)]
        for state in leftovers:
            control.delete(state.id)
        step("deleted", [s.id for s in leftovers])
        goldens = [c for c in control.list_checkpoints() if c.name.startswith("microvm-golden-")]
        for checkpoint in goldens:
            for _ in range(24):
                try:
                    control.delete_checkpoint(checkpoint.id)
                    break
                except Exception:  # noqa: BLE001 - 400 while still capturing
                    time.sleep(5)
        step("deleted_checkpoints", [c.name for c in goldens])
        out = Path.home() / ".langchain_digitalocean_microvm" / "probe" / date.today().isoformat()
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_agent_host.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
