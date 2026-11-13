"""Live smoke test for per-task subagent sandboxes: one real MicroVM per `task` call (subagents.PerTask).

Refuses to start while MicroVMs of its namespace (DO_MICROVM_NAMESPACE) exist. Then, with a scripted
model (no model key), at most two MicroVMs at a time:

  1. `when_done="delete"`: one `task` call writes /work/delta.md in its own MicroVM; the MicroVM is
     deleted when the task returns (and its store row marked destroyed);
  2. `when_done="pause"`, `max_sessions=2`, key = first word of the description: two parallel tasks
     (alpha, beta) each see an empty /work, write their own file and see only it (two MicroVMs, tagged
     `microvm-per-task-sandbox`); a third key (gamma) is refused with a tool message; both MicroVMs are
     paused when their tasks return;
  3. a second turn: key alpha again resumes the same MicroVM and finds its file.

Everything it creates is deleted in a `finally` block; the store is a temporary file. Results go to
~/.langchain_digitalocean_microvm/probe/<date>/smoke_per_task.json.

    uv run python \\
        scripts/smoke_per_task.py [--image IMAGE --http-port PORT] [--transport endpoint]

The image defaults to DO_MICROVM_IMAGE / DO_MICROVM_HTTP_PORT (the sandbox image from
scripts/build_sandbox_image.py, port 8080), else httpbin on port 80.
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
from scripted_model import ScriptedChatModel, say, tool_call, tool_calls  # noqa: E402

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle  # noqa: E402
from langchain_digitalocean_microvm.agent_host.store import GatewayStore  # noqa: E402
from langchain_digitalocean_microvm.client import (  # noqa: E402
    MANAGED_TAG,
    PER_TASK_SANDBOX_TAG,
    SUBAGENT_SANDBOX_TAG,
    MicroVMClient,
)
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError  # noqa: E402
from langchain_digitalocean_microvm.status import MicroVMStatus  # noqa: E402
from langchain_digitalocean_microvm.subagents import (  # noqa: E402
    PerTask,
    provision_subagent_sandboxes,
    subagent_sandbox_id,
    subagent_sandboxes,
)

FALLBACK_IMAGE, FALLBACK_PORT = "docker.io/kennethreitz/httpbin:latest", 80  # root, sh, coreutils, python3
GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)
SPECS = [{"name": "analyst", "description": "Analyses one item.", "system_prompt": "You analyse.", "permissions": []}]


def _out(message: Any) -> str:
    return str(message.content).split("\n[Command")[0].strip()


def worker(messages: list[Any]) -> Any:
    """A subagent's step: list /work, write /work/<first word>.md, list again, answer with both listings."""
    done = [m for m in messages if m.type == "tool"]
    word = next(str(m.content) for m in messages if m.type == "human").split()[0]
    if not done:
        return tool_call("execute", command="ls /work 2>/dev/null; true")
    if len(done) == 1:
        return tool_call("write_file", file_path=f"/work/{word}.md", content=word)
    if len(done) == 2:
        return tool_call("execute", command="ls /work")
    return say(f"before: [{_out(done[0])}] after: [{_out(done[2])}]")


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
    run = secrets.token_hex(3)
    thread_id = f"smoke-{run}"
    sandbox_ids = [subagent_sandbox_id(thread_id, f"analyst-{key}") for key in ("delta", "alpha", "beta", "gamma")]
    store_dir = tempfile.TemporaryDirectory()
    store = GatewayStore(Path(store_dir.name) / "store.sqlite")
    results: dict[str, Any] = {"thread_id": thread_id, "image": args.image, "transport": args.transport}
    t0 = time.monotonic()

    def step(name: str, value: Any) -> None:
        results[name] = value
        print(f"[{time.monotonic() - t0:6.1f}s] {name}: {json.dumps(value, default=str)[:400]}", flush=True)

    def agent_for(model: ScriptedChatModel, sb: Any) -> Any:
        return create_deep_agent(model=model, subagents=list(sb.subagents), middleware=[*sb.middleware],
                                 backend=sb.mount(StateBackend()))

    def answers(out: dict[str, Any]) -> list[str]:
        return [str(m.content) for m in out["messages"] if m.type == "tool" and m.name == "task"]

    def live_names() -> list[str]:
        return sorted(s.name for s in client.list(tag=PER_TASK_SANDBOX_TAG) if s.status not in GONE)

    code = 0
    try:
        # 1. delete when the task returns
        model = ScriptedChatModel()
        with subagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model, on_demand=True, store=store,
                                per_task={"analyst": PerTask(key=lambda c: c.description.split()[0])}) as sb:
            model.script(tool_call("task", subagent_type="analyst", description="delta item"), *[worker] * 4, say("done"))
            out = agent_for(model, sb).invoke({"messages": [{"role": "user", "content": "go"}]})
            step("delete_mode_answer", answers(out))
            step("delete_mode_live_after_task", live_names())
        row = next(r for r in store.list_sandboxes() if r.sandbox_id == sandbox_ids[0])
        step("delete_mode_row", (row.status, row.retired_reason))

        # 2. pause mode, parallel tasks, the cap
        rules = PerTask(key=lambda c: c.description.split()[0], when_done="pause", max_sessions=2, max_concurrent=2)
        model = ScriptedChatModel()
        sb = provision_subagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model, on_demand=True,
                                          store=store, per_task={"analyst": rules})
        model.script(
            tool_calls(("task", {"subagent_type": "analyst", "description": "alpha item"}),
                       ("task", {"subagent_type": "analyst", "description": "beta item"})),
            *[worker] * 8,
            tool_call("task", subagent_type="analyst", description="gamma item"),
            say("done"),
        )
        started = time.monotonic()
        out = agent_for(model, sb).invoke({"messages": [{"role": "user", "content": "go"}]})
        step("turn1_agent_seconds", round(time.monotonic() - started, 1))
        got = answers(out)
        step("parallel_answers", got)
        step("isolated", any("alpha.md" in a and "beta.md" not in a for a in got)
             and any("beta.md" in a and "alpha.md" not in a for a in got))
        step("third_key_refused", any("task refused" in a and "limit is 2" in a for a in got))
        states = {s.name: (s.status.value, [t for t in s.tags if t in (SUBAGENT_SANDBOX_TAG, PER_TASK_SANDBOX_TAG, MANAGED_TAG)])
                  for s in client.list(tag=PER_TASK_SANDBOX_TAG) if s.status not in GONE}
        step("microvms", states)
        ids_turn1 = {s.name: s.id for s in client.list(tag=PER_TASK_SANDBOX_TAG) if s.status not in GONE}
        step("two_microvms_paused_tagged", len(states) == 2 and all(v[0] == "paused" and PER_TASK_SANDBOX_TAG in v[1]
                                                                    for v in states.values()))
        step("store_rows", sorted((r.sandbox_id.rsplit("-", 1)[-1], r.status) for r in store.list_sandboxes()))

        # 3. resume by key in a later turn
        model = ScriptedChatModel()
        sb = provision_subagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model, on_demand=True,
                                          store=store, per_task={"analyst": rules})
        model.script(tool_call("task", subagent_type="analyst", description="alpha again"), *[worker] * 4, say("done"))
        out = agent_for(model, sb).invoke({"messages": [{"role": "user", "content": "go"}]})
        again = answers(out)
        step("turn2_answer", again)
        step("turn2_found_its_file", any("before: [alpha.md]" in a for a in again))
        step("turn2_same_microvm", {s.name: s.id for s in client.list(tag=PER_TASK_SANDBOX_TAG)
                                    if s.status not in GONE} == ids_turn1)
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
        (out_dir / "smoke_per_task.json").write_text(json.dumps(results, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
