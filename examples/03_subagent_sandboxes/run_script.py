"""Drive the subagent-sandboxes supervisor from a script, without `langgraph dev`.

Same supervisor as agent.py, with an in-memory checkpointer so the
conversation carries across turns. Each turn does what the graph factory does
under `langgraph dev`: get the thread's two sandboxes (created on the first
turn, resumed afterwards), run, pause them. Between turns it prints what is
in each subagent's filesystem, read straight from the sandboxes, and how long
each phase took with how many MicroVMs API requests it sent (exec requests are
what the team's 60/min budget counts).

RUN (from the repository root; DO_MICROVM_TRANSPORT=local needs no DigitalOcean account):
    uv run examples/03_subagent_sandboxes/run_script.py
    uv run examples/03_subagent_sandboxes/run_script.py --delete   # also delete the sandboxes at the end
    uv run examples/03_subagent_sandboxes/run_script.py --delete --save outputs
        # copy each subagent's /outputs/ to outputs/<thread>/<subagent>/outputs/ first
Needs a model: DO_MODEL_ACCESS_KEY, or EXAMPLES_MODEL with its provider's key (agent.py).
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # before agent.py reads the environment

from agent import (  # noqa: E402
    SUBAGENTS,
    build_supervisor,
    lifecycle,
    sandboxes_for,
    store,
)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from langchain_digitalocean_microvm.client import api_calls  # noqa: E402
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError  # noqa: E402
from langchain_digitalocean_microvm.subagents import subagent_sandbox_id  # noqa: E402

parser = argparse.ArgumentParser(description="Drive the subagent-sandboxes supervisor without langgraph dev.")
parser.add_argument("--thread", default=None, help="thread id (default: a new one)")
parser.add_argument("--delete", action="store_true", help="delete the thread's sandboxes at the end")
parser.add_argument("--save", type=Path, metavar="DIR",
                    help="after each turn, copy each subagent's /outputs/ to DIR/<thread>/<subagent>/outputs/")
args = parser.parse_args()

thread_id = args.thread or f"lab-{uuid.uuid4().hex[:8]}"
config = {"configurable": {"thread_id": thread_id}}
saver = InMemorySaver()


def _text(message) -> str:
    """A message's text; content is either a string or a list of blocks (keep only the text ones)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def _files(backend) -> list[str]:
    """Files under /data, /inputs and /outputs, via BaseSandbox's glob (works in local mode too)."""
    found = []
    for root in ("/data", "/inputs", "/outputs"):
        result = backend.glob("**", path=root)
        # deepagents >= 0.7.21 returns absolute paths; earlier 0.7.x, paths relative to `root`
        found += [p if p.startswith("/") else f"{root}/{p}" for m in result.matches or [] if not m.get("is_dir") for p in [m["path"]]]
    return found


def _calls_since(before: dict[str, int]) -> str:
    """Per-endpoint MicroVMs API request counts since the `before` snapshot, as a short string."""
    now = api_calls()
    delta = {k: now[k] - before.get(k, 0) for k in sorted(now) if now[k] != before.get(k, 0)}
    return ", ".join(f"{k} {v}" for k, v in delta.items()) or "none"


def _save_outputs(name: str, backend, files: list[str]) -> None:
    """Copy `files` under /outputs/ from one sandbox to args.save/<thread>/<name>/ (same relative paths)."""
    dest = (args.save / thread_id / name).resolve()
    wanted = [f for f in files if f.startswith("/outputs/")]
    for response in backend.download_files(wanted) if wanted else []:
        target = (dest / response.path.lstrip("/")).resolve()
        if response.error or response.content is None or not target.is_relative_to(dest):
            print(f"[{name}] not saved: {response.path} ({response.error or 'outside the save directory'})")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        print(f"[{name}] saved {response.path} -> {target}")


async def turn(prompt: str) -> None:
    print(f"\n>>> {prompt}")
    started, calls = time.monotonic(), api_calls()
    # Entering the context creates or resumes both sandboxes; leaving it pauses them.
    async with sandboxes_for(thread_id) as sandboxes:
        ready, ready_calls = time.monotonic() - started, _calls_since(calls)
        graph = build_supervisor(sandboxes, checkpointer=saver)
        run_started, calls = time.monotonic(), api_calls()
        state = await graph.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
        ran, run_calls = time.monotonic() - run_started, _calls_since(calls)
        print(_text(state["messages"][-1]))
        for name, backend in sandboxes.backends.items():
            files = _files(backend)
            print(f"[{name} VM {backend.id}] {', '.join(files) or '(no files)'}")
            if args.save:
                _save_outputs(name, backend, files)  # while it runs: once paused or deleted it would need a resume
        pause_started, calls = time.monotonic(), api_calls()
    paused, pause_calls = time.monotonic() - pause_started, _calls_since(calls)
    # Count only this turn's tool calls: the checkpointer replays earlier turns in state["messages"].
    last_prompt = max(i for i, m in enumerate(state["messages"]) if m.type == "human")
    tool_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in state["messages"][last_prompt:])
    print(f"[sandboxes ready in {ready:.1f} s (API: {ready_calls})]")
    print(f"[agent run {ran:.1f} s, {tool_calls} supervisor tool calls (API: {run_calls})]")
    print(f"[paused in {paused:.1f} s (API: {pause_calls})]")


async def main() -> None:
    print(f"thread {thread_id}; sandboxes: {lifecycle.config.transport}")
    try:
        await turn(
            "Ask the analyst for total revenue per region and the month-over-month revenue growth, "
            "saved as /outputs/summary.md (with the script that computed it)."
        )
        await turn(
            "Give the analyst's summary to the reviewer (copy it into the reviewer's inputs) and have "
            "it review the numbers. Then show me the review."
        )
    finally:
        if args.delete:  # also after a failed turn: paused MicroVMs count against the team limit
            for spec in SUBAGENTS:
                sandbox_id = subagent_sandbox_id(thread_id, spec["name"])
                try:
                    lifecycle.delete(sandbox_id=sandbox_id)
                except SandboxNotFoundError:
                    print(f"no sandbox {sandbox_id} to delete")
                    continue
                if store is not None:
                    store.retire_sandbox(sandbox_id, "deleted")
                print(f"deleted sandbox {sandbox_id}")
        print(f"\nMicroVMs API requests in total: {_calls_since({})}")


asyncio.run(main())
