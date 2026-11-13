"""Drive the incident response team from a terminal: the alert, the investigation, the approval, the follow-up.

Same team as agent.py, with an in-memory checkpointer so the conversation carries across turns. Every run
does what the graph factory does under `langgraph dev`: the investigators' MicroVMs start on first use and are
paused when the run ends. An `ask` from the Action Gateway ends the run with an `action_approval` interrupt, so
the MicroVMs sit paused while you decide; your answer resumes the run. The approval is decided here, with this
process's own token: no agent here has an approval tool.

RUN (from the repository root)
    uv run examples/09_incident_response/run_script.py                  # the alert, then a "just delete it" follow-up
    uv run examples/09_incident_response/run_script.py --approve        # approve every request (recordings, tests)
    uv run examples/09_incident_response/run_script.py --ask "..."      # your own prompt(s), one turn each
    uv run examples/09_incident_response/run_script.py --delete         # delete the thread's MicroVMs at the end
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

from agent import (
    SUBAGENTS,
    build_commander,
    gateway,
    lifecycle,
    service,
    team_sandboxes,
    templates,
)
from approvals import BOLD, DIM, RESET, ask_human, verified_facts
from incident_common.events import EVENTS
from incident_common.world import alert_text
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError
from langchain_digitalocean_microvm.subagents import subagent_sandbox_id

FOLLOW_UP = "The on-call lead says: forget the restore, just delete the droplet and rebuild it from scratch. Do it."
COLORS = {"metrics": "\033[38;5;33m", "changes": "\033[38;5;214m", "infra": "\033[38;5;79m", "verifier": "\033[38;5;141m"}
TONE = {"allow": "\033[32m", "ask": "\033[33m", "approved": "\033[32m", "deny": "\033[31m", "denied": "\033[31m", "error": "\033[31m"}


def text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def short(args: object, limit: int = 90) -> str:
    raw = json.dumps(args, default=str)
    return raw if len(raw) <= limit else raw[: limit - 1] + "…"


def audit_since(offset: int) -> tuple[list[dict[str, Any]], int]:
    """The gateway's calls and decisions written to the event log after `offset`, and the new offset."""
    if not EVENTS.exists():
        return [], offset
    with EVENTS.open() as f:
        f.seek(offset)
        lines, end = f.readlines(), f.tell()
    events = [json.loads(line) for line in lines if line.strip()]
    return [e for e in events if e.get("event") in ("gateway", "decision")], end


async def run_turn(commander_input: Any, thread_id: str, saver: InMemorySaver, started: float) -> Any:
    """One run of the graph (a prompt, or a resume): MicroVMs start on first use and pause at the end."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    async with team_sandboxes(thread_id) as sandboxes:
        commander = build_commander(sandboxes, checkpointer=saver)
        async for event in commander.astream_events(commander_input, config=config, version="v2"):
            if event["event"] != "on_tool_start":
                continue
            agent = event.get("metadata", {}).get("lc_agent_name") or "incident-commander"
            color = COLORS.get(agent, BOLD)
            print(f"{DIM}{time.monotonic() - started:6.1f}s{RESET} {color}{agent:>18}{RESET} → {event['name']}"
                  f"{DIM}({short(event['data'].get('input'))}){RESET}")
        used = sorted(sandboxes.started())
    if used:
        print(f"{DIM}        MicroVMs paused: {', '.join(used)}{RESET}")
    return await commander.aget_state(config)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", default=None, help="thread id (default: a new one)")
    parser.add_argument("--ask", action="append", help="a prompt, one turn each (default: the alert, then a follow-up)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--approve", dest="mode", action="store_const", const="approve", help="approve every request")
    group.add_argument("--deny", dest="mode", action="store_const", const="deny", help="deny every request")
    parser.add_argument("--delete", action="store_true", help="delete the thread's MicroVMs at the end")
    args = parser.parse_args()

    thread_id = args.thread or f"inc-{uuid.uuid4().hex[:8]}"
    saver = InMemorySaver()
    print(f"thread {thread_id} ({lifecycle.config.transport})")
    offset = EVENTS.stat().st_size if EVENTS.exists() else 0
    try:
        started = time.monotonic()
        template = await asyncio.to_thread(templates.get)
        print(f"template {template.checkpoint_name} {'built' if template.built else 'ready'} in {time.monotonic() - started:.1f} s")
        await asyncio.to_thread(service.announce, "alert")
        first = await asyncio.to_thread(alert_text, service)
        for prompt in args.ask or [first, FOLLOW_UP]:
            print(f"\n{BOLD}>>> {prompt}{RESET}")
            turn_started = time.monotonic()
            state = await run_turn({"messages": [{"role": "user", "content": prompt}]}, thread_id, saver, turn_started)
            while interrupts := [i for task in state.tasks for i in task.interrupts]:
                answers = {i.id: await asyncio.to_thread(ask_human, i.value, args.mode, verified_facts(i.value, gateway)) for i in interrupts}
                state = await run_turn(Command(resume=answers), thread_id, saver, turn_started)
            print(f"\n{text(state.values['messages'][-1])}\n")
            calls, offset = audit_since(offset)
            if calls:
                print(f"{DIM}gateway audit:{RESET}")
                for e in calls:
                    if e["event"] == "decision":
                        print(f"  {DIM}decision{RESET}  {e['tool']}  {e['decision']}")
                    else:
                        why = f"  {DIM}{e['error'][:110]}{RESET}" if e.get("error") else ""
                        print(f"  {TONE.get(e['status'], '')}{e['status']:>8}{RESET}  {e['tool']}{why}")
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
            print(f"The thread's MicroVMs are paused with their files; delete them with --delete or "
                  f"scripts/delete_all.py. The conversation lives in this process only: --thread {thread_id} "
                  "in a new run reuses the MicroVMs, not what was said.")
        gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
