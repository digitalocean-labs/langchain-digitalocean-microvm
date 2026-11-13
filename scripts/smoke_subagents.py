"""Live smoke test for subagent sandboxes: two subagents, each with its own real MicroVM (subagents).

Refuses to start while MicroVMs of its namespace (DO_MICROVM_NAMESPACE) exist. Then, with a scripted model (no model key):

  turn 1: provision both sandboxes in parallel (create) -> a real deepagents
          supervisor delegates to `analyst` (writes /outputs/a.md and runs
          python3 in its VM), reads the file at /subagents/analyst/outputs/a.md,
          copies it to /subagents/reviewer/inputs/a.md, delegates to `reviewer`
          (reads it in ITS VM) -> pause both;
  turn 2: provision again (resume) -> the copy is still there -> pause both;
  sweeper: both kept as fresh, both planned for deletion two days later, and
          deleted by apply_sweep (store rows marked destroyed, reason gc).

Everything it creates is deleted in a `finally` block; the store is a
temporary file, not the default gateway store. Results go to
~/.langchain_digitalocean_microvm/probe/<date>/smoke_subagents.json.

    uv run python \\
        scripts/smoke_subagents.py [--image IMAGE --http-port PORT] [--transport endpoint]

The image defaults to DO_MICROVM_IMAGE / DO_MICROVM_HTTP_PORT (the sandbox
image from scripts/build_sandbox_image.py, port 8080), else httpbin on port 80.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

from _env import (  # scripts/_env.py
    add_env_file_argument,
    env_namespace,
    load_env,
    refuse_if_owned,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "unit_tests"))  # scripted_model
from deepagents import create_deep_agent  # noqa: E402
from deepagents.backends import StateBackend  # noqa: E402
from scripted_model import ScriptedChatModel, say, tool_call  # noqa: E402

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle  # noqa: E402
from langchain_digitalocean_microvm.agent_host.store import GatewayStore  # noqa: E402
from langchain_digitalocean_microvm.agent_host.sweeper import (  # noqa: E402
    apply_sweep,
    plan_sweep,
)
from langchain_digitalocean_microvm.client import (  # noqa: E402
    MANAGED_TAG,
    SUBAGENT_SANDBOX_TAG,
    MicroVMClient,
)
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError  # noqa: E402
from langchain_digitalocean_microvm.status import MicroVMStatus  # noqa: E402
from langchain_digitalocean_microvm.subagents import (  # noqa: E402
    pause_subagent_sandboxes,
    provision_subagent_sandboxes,
    subagent_sandbox_id,
)

FALLBACK_IMAGE, FALLBACK_PORT = "docker.io/kennethreitz/httpbin:latest", 80  # root, sh, coreutils, python3
GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)
SPECS = [
    {"name": "analyst", "description": "Analyses.", "system_prompt": "You analyse."},
    {"name": "reviewer", "description": "Reviews.", "system_prompt": "You review."},
]
REPORT = "# Analysis\nanswer: 42\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="default: DO_MICROVM_IMAGE, else httpbin")
    parser.add_argument("--http-port", type=int, help="default: DO_MICROVM_HTTP_PORT (8080), or 80 for httpbin")
    parser.add_argument("--transport", choices=("endpoint", "exec"), default="endpoint",
                        help="endpoint needs DO_MICROVM_ACCESS_TOKEN and the sandbox image")
    add_env_file_argument(parser)
    args = parser.parse_args()
    env = load_env(args.env_file)
    if args.image is None:
        args.image = env.get("DO_MICROVM_IMAGE") or FALLBACK_IMAGE
    if args.http_port is None:
        configured = env.get("DO_MICROVM_HTTP_PORT")
        args.http_port = int(configured) if configured else (FALLBACK_PORT if args.image == FALLBACK_IMAGE else 8080)
    token = env.get("DO_MICROVM_API_TOKEN")
    if not token:
        print("DO_MICROVM_API_TOKEN not set", file=sys.stderr)
        return 2

    client = MicroVMClient(token)
    namespace = env_namespace(env)
    if refuse_if_owned(client, namespace):
        return 2

    config = MicroVMConfig(transport=args.transport, api_token=token, access_token=env.get("DO_MICROVM_ACCESS_TOKEN"),
                           image=args.image, http_port=args.http_port, idle_timeout="5m", ready_timeout=300.0,
                           namespace=namespace)
    lifecycle = MicroVMLifecycle(config)
    thread_id = f"smoke-{secrets.token_hex(3)}"
    sandbox_ids = [subagent_sandbox_id(thread_id, spec["name"]) for spec in SPECS]
    store_dir = tempfile.TemporaryDirectory()
    store = GatewayStore(Path(store_dir.name) / "store.sqlite")
    results: dict[str, Any] = {"thread_id": thread_id, "image": args.image, "transport": args.transport}
    t0 = time.monotonic()

    def step(name: str, value: Any) -> None:
        results[name] = value
        print(f"[{time.monotonic() - t0:6.1f}s] {name}: {json.dumps(value, default=str)[:400]}", flush=True)

    def states(sb) -> dict[str, str]:  # type: ignore[no-untyped-def]
        return {name: client.get(b.id).status.value for name, b in sb.backends.items()}

    code = 0
    model = ScriptedChatModel()
    try:
        started = time.monotonic()
        sb = provision_subagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model, store=store)
        step("turn1_provision_seconds", round(time.monotonic() - started, 1))
        step("microvms", {name: b.id for name, b in sb.backends.items()})
        tags = {name: list(client.get(b.id).tags) for name, b in sb.backends.items()}
        step("tagged", all(SUBAGENT_SANDBOX_TAG in t and MANAGED_TAG in t for t in tags.values()))

        agent = create_deep_agent(model=model, subagents=list(sb.subagents), backend=sb.mount(StateBackend()))
        model.script(
            tool_call("task", subagent_type="analyst", description="write the analysis"),
            tool_call("write_file", file_path="/outputs/a.md", content=REPORT),
            tool_call("execute", command="python3 -c 'print(6 * 7)'; hostname"),
            say("wrote /outputs/a.md"),
            tool_call("read_file", file_path="/subagents/analyst/outputs/a.md"),
            tool_call("write_file", file_path="/subagents/reviewer/inputs/a.md", content=REPORT),
            tool_call("task", subagent_type="reviewer", description="review /inputs/a.md"),
            tool_call("execute", command="cat /inputs/a.md; ls /outputs 2>&1; hostname"),
            say("reviewed"),
            say("done"),
        )
        started = time.monotonic()
        out = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        step("turn1_agent_seconds", round(time.monotonic() - started, 1))
        tools = [m for m in out["messages"] if m.type == "tool"]
        step("parent_read_via_mount", "answer: 42" in str(next(m for m in tools if m.name == "read_file").content))
        sub_outputs = [str(m.content) for p in model.prompts for m in p if m.type == "tool" and m.name == "execute"]
        step("analyst_execute", next((o for o in sub_outputs if "42" in o and "answer" not in o), sub_outputs[:1]))
        step("reviewer_execute", next((o for o in sub_outputs if "answer: 42" in o), None))
        step("isolated", "No such file" in (results["reviewer_execute"] or "") or "cannot access" in (results["reviewer_execute"] or ""))

        started = time.monotonic()
        errors = pause_subagent_sandboxes(lifecycle, sb)
        step("turn1_pause", {"errors": [str(e) for e in errors], "seconds": round(time.monotonic() - started, 1),
                             "states": states(sb)})

        started = time.monotonic()
        sb2 = provision_subagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model, store=store)
        step("turn2_provision_seconds", round(time.monotonic() - started, 1))
        step("turn2_same_vms", {n: b.id for n, b in sb2.backends.items()} == results["microvms"])
        copy = sb2.mount(StateBackend()).read("/subagents/reviewer/inputs/a.md")
        step("turn2_copy_survived", (copy.file_data or {}).get("content") == REPORT.rstrip("\n"))  # read() drops the final newline
        errors = pause_subagent_sandboxes(lifecycle, sb2)
        step("turn2_pause", {"errors": [str(e) for e in errors], "states": states(sb2)})
        step("store_rows", [(r.subagent, r.status) for r in store.list_sandboxes()])

        now = time.time()
        fresh = plan_sweep(client, store=store, now=now, sandbox_ttl=24 * 3600.0, namespace=namespace)
        step("sweep_now", {"delete": len(fresh.delete_microvms), "kept": [i.reason for i in fresh.kept if i.kind == "sandbox"]})
        later = plan_sweep(client, store=store, now=now + 2 * 24 * 3600.0, sandbox_ttl=24 * 3600.0, orphan_ttl=1e9, namespace=namespace)
        ours = [i for i in later.delete_microvms if i.kind == "sandbox" and i.sandbox_id in sandbox_ids]
        step("sweep_in_two_days", [i.reason for i in ours])
        swept = apply_sweep(type(later)(delete_microvms=tuple(ours)), client, store=store)
        step("swept", {"deleted": [i.name for i in swept.deleted_microvms], "errors": swept.errors,
                       "rows": [(r.subagent, r.status, r.retired_reason) for r in store.list_sandboxes()]})
    except Exception as exc:  # noqa: BLE001 - report and still clean up
        step("error", f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        for sandbox_id in sandbox_ids:
            try:
                lifecycle.delete(sandbox_id=sandbox_id)
                step(f"deleted {sandbox_id}", True)
            except SandboxNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                step(f"delete_error {sandbox_id}", f"{type(exc).__name__}: {exc}")
                code = 1
        store.close()
        store_dir.cleanup()
        left = [s.name for s in client.list(tag=MANAGED_TAG) if s.status not in GONE]
        step("remaining_live_microvms", left)
        if left:
            code = 1
        out_dir = Path.home() / ".langchain_digitalocean_microvm" / "probe" / date.today().isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smoke_subagents.json").write_text(json.dumps(results, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
