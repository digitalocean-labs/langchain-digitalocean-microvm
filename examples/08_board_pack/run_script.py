"""Drive the board pack team from a terminal, without `langgraph dev`: rehearsals, recordings, fallback.

Same team as agent.py, with an in-memory checkpointer so a conversation carries across turns.
Every turn does what the graph factory does under `langgraph dev` (specialist sandboxes started
on first use, paused at the end) and streams each tool call as it happens, labelled with the
agent that made it, so the parallel fan-out is visible in the console.

RUN (from the repository root)
    uv run examples/08_board_pack/run_script.py                     # the board pack, then a follow-up
    uv run examples/08_board_pack/run_script.py --ask "..."         # your own prompt(s), one turn each
    uv run examples/08_board_pack/run_script.py --delete            # delete the thread's VMs at the end
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from agent import SUBAGENTS, build_orchestrator, lifecycle, team_sandboxes, templates
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_digitalocean_microvm.client import api_calls
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError
from langchain_digitalocean_microvm.subagents import subagent_sandbox_id

PROMPTS = [
    "Prepare the FY2025 board pack.",
    "Asia-Pacific looks weak. Which country drove the decline, and what did customers there stop buying?",
]
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
COLORS = ["\033[38;5;33m", "\033[38;5;43m", "\033[38;5;214m", "\033[38;5;204m", "\033[38;5;141m", "\033[38;5;79m", "\033[38;5;250m"]


def text(message: BaseMessage) -> str:
    """A message's text; content is either a string or a list of blocks (keep only the text ones)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def calls_since(before: dict[str, int]) -> str:
    """Per-endpoint MicroVMs API request counts since the `before` snapshot, as a short string."""
    now = api_calls()
    delta = {k: now[k] - before.get(k, 0) for k in sorted(now) if now[k] != before.get(k, 0)}
    return ", ".join(f"{k} {v}" for k, v in delta.items()) or "none"


def short(args: object, limit: int = 90) -> str:
    """A tool call's arguments as one line of JSON, cut to `limit` characters."""
    raw = json.dumps(args, default=str)
    return raw if len(raw) <= limit else raw[: limit - 1] + "…"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", default=None, help="thread id (default: a new one)")
    parser.add_argument("--ask", action="append", help="a prompt, one turn each (default: the board pack, then a follow-up)")
    parser.add_argument("--delete", action="store_true", help="delete the thread's specialist VMs at the end")
    args = parser.parse_args()

    thread_id = args.thread or f"demo-{uuid.uuid4().hex[:8]}"
    # High recursion limit: the fan-out, handoffs and six specialists take many graph steps per turn.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    saver = InMemorySaver()
    colors = {spec["name"]: COLORS[i % len(COLORS)] for i, spec in enumerate(SUBAGENTS)}
    print(f"thread {thread_id} ({lifecycle.config.transport})")

    try:
        started = time.monotonic()
        template = await asyncio.to_thread(templates.get)
        print(f"template {template.checkpoint_name} {'built' if template.built else 'ready'} in {time.monotonic() - started:.1f} s")
        for prompt in args.ask or PROMPTS:
            print(f"\n{BOLD}>>> {prompt}{RESET}")
            turn_started, calls = time.monotonic(), api_calls()
            async with team_sandboxes(thread_id) as sandboxes:
                orchestrator = build_orchestrator(sandboxes, thread_id, checkpointer=saver)
                async for event in orchestrator.astream_events(
                    {"messages": [{"role": "user", "content": prompt}]}, config=config, version="v2"
                ):
                    if event["event"] != "on_tool_start":
                        continue
                    # lc_agent_name tells which (sub)agent made the call; subagent events bubble up here.
                    agent = event.get("metadata", {}).get("lc_agent_name") or "board-pack-orchestrator"
                    color = colors.get(agent, BOLD)
                    elapsed = time.monotonic() - turn_started
                    print(f"{DIM}{elapsed:6.1f}s{RESET} {color}{agent:>22}{RESET} → {event['name']}{DIM}({short(event['data'].get('input'))}){RESET}")
                used = sorted(sandboxes.started())
                ran = time.monotonic() - turn_started
                pause_started = time.monotonic()
            state = await orchestrator.aget_state(config)
            print(f"\n{text(state.values['messages'][-1])}\n")
            print(f"{DIM}[turn {ran:.0f} s; VMs used: {', '.join(used) or 'none'}; paused in "
                  f"{time.monotonic() - pause_started:.1f} s; MicroVMs API: {calls_since(calls)}]{RESET}")
    finally:
        if args.delete:
            for spec in SUBAGENTS:
                sandbox_id = subagent_sandbox_id(thread_id, spec["name"])
                try:
                    lifecycle.delete(sandbox_id=sandbox_id)
                    print(f"deleted {sandbox_id}")
                except SandboxNotFoundError:
                    pass
        else:
            print(f"\nThe thread's VMs are paused; continue with --thread {thread_id}, delete with --delete "
                  "or scripts/delete_all.py")
        print(f"MicroVMs API requests in total: {calls_since({})}")


if __name__ == "__main__":
    asyncio.run(main())
