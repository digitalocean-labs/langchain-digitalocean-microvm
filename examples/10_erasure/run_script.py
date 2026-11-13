"""Drive the erasure team from a terminal: the request, the locators, one approval per erasure, the follow-up, the report.

Same team as agent.py, with an in-memory checkpointer so the conversation carries across turns. Every run does what
the graph factory does under `langgraph dev`: the locators' MicroVMs start on first use and are paused when the run
ends. An `ask` from the Action Gateway ends the run with an `action_approval` interrupt, so the MicroVMs sit paused
while you decide; your answer resumes the run. The approval is decided here, with this process's own token: no agent
here has an approval tool.

RUN (from the repository root)
    uv run examples/10_erasure/run_script.py                          # the request, then a "delete the billing customer" follow-up
    uv run examples/10_erasure/run_script.py --approve                # approve every request (recordings, tests)
    uv run examples/10_erasure/run_script.py --approve --deny-system marketing   # approve all but the email-marketing erasure
    uv run examples/10_erasure/run_script.py --ask "..."              # your own prompt(s), one turn each
    uv run examples/10_erasure/run_script.py --delete                 # delete the thread's MicroVMs at the end
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
    audited,
    build_coordinator,
    gateway,
    lifecycle,
    team_sandboxes,
    templates,
)
from gateway import target_of
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tools import REPORT, SYSTEMS
from world import MIXPANEL_PROJECT, REQUEST_EMAIL, REQUEST_NAME

from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError
from langchain_digitalocean_microvm.subagents import subagent_sandbox_id

REQUEST = (f"Erasure request from {REQUEST_NAME} <{REQUEST_EMAIL}>, identity verified by the privacy team: erase her personal "
           "data from every system that holds it, keep only what the law requires us to keep, and report.")
FOLLOW_UP = "The billing lead says: delete her billing customer entirely as well, so nothing is left. Do it."
DIM, BOLD, RESET, AMBER, GREEN, RED = "\033[2m", "\033[1m", "\033[0m", "\033[33m", "\033[32m", "\033[31m"
COLORS = {"crm": "\033[38;5;33m", "support": "\033[38;5;214m", "billing": "\033[38;5;79m", "marketing": "\033[38;5;170m"}
TONE = {"allow": GREEN, "ask": AMBER, "approved": GREEN, "deny": RED, "denied": RED, "error": RED}


def text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def short(args: object, limit: int = 90) -> str:
    raw = json.dumps(args, default=str)
    return raw if len(raw) <= limit else raw[: limit - 1] + "…"


def _read(tool: str, arguments: dict[str, Any]) -> Any:
    answer = gateway.call(tool, arguments)
    try:
        return json.loads(answer)
    except ValueError:
        return {}


def _identity(email: object) -> str:
    """Whether a record's email address is the request's, shown without the address itself."""
    if not email:
        return f"{RED}no email address on the record{RESET}"
    if str(email).strip().lower() == REQUEST_EMAIL:
        return f"{GREEN}email address matches the request ✔{RESET}"
    return f"{RED}a DIFFERENT email address (@{str(email).rsplit('@', 1)[-1]}): not the requester ✘{RESET}"


def verified_facts(stop: dict[str, Any]) -> list[str]:
    """What the arguments point at, read by this console through the gateway, not taken from the coordinator's words."""
    target = target_of(stop.get("tool", ""), stop.get("arguments", {}))
    if target is None:
        return []
    system, kind, rid = target
    arguments = stop.get("arguments", {})
    lines = [f"record     {system} ({SYSTEMS.get(system, '?')}, mock) {kind} {rid}"]
    if system == "crm" and kind == "contact":
        found = _read("hubspot_crm_contacts_get_by_contactid", {"contact_id": rid, "properties": ["email"], "associations": ["notes"]})
        if found.get("id") != rid:
            return [*lines, f"{RED}NOT FOUND in the CRM{RESET}"]
        notes = len(found.get("associations", {}).get("notes", {}).get("results", []))
        return [*lines, f"created    {found['properties'].get('createdate', '?')[:10]}; {notes} notes go with it",
                f"identity   {_identity(found['properties'].get('email'))}"]
    if system == "support":
        found = _read("intercom_get_a_contact", {"contact_id": rid})
        if found.get("id") != rid:
            return [*lines, f"{RED}NOT FOUND in support{RESET}"]
        created = time.strftime("%Y-%m-%d", time.gmtime(found.get("created_at", 0)))
        return [*lines, f"created    {created}", f"identity   {_identity(found.get('email'))}"]
    if system == "billing":
        found = _read("stripe_get_customers_customer", {"customer": rid})
        if found.get("id") != rid:
            return [*lines, f"{RED}NOT FOUND in billing{RESET}"]
        invoices = _read("stripe_get_invoices", {"customer": rid, "limit": 100}).get("data", [])
        out = [*lines, f"identity   {_identity(found.get('email'))}", f"kept       {len(invoices)} invoices (not touched by this call)"]
        if kind == "customer" and stop.get("tool") == "stripe_post_customers_customer":
            values = {k: v for k, v in arguments.items() if k != "customer"}
            sets = [k for k, v in values.items() if (v if not isinstance(v, dict) else any(v.values()))]
            out.append(f"change     {RED}SETS {', '.join(sets)}: not a redaction ✘{RESET}" if sets
                       else f"change     {GREEN}clears {', '.join(values) or 'nothing'} ✔{RESET}")
        return out
    if system == "marketing":
        found = _read("mailchimp_search_members", {"query": REQUEST_EMAIL})
        members = found.get("exact_matches", {}).get("members", [])
        member = next((m for m in members if m.get("contact_id") == rid), None)
        if member is None:
            return [*lines, f"identity   {RED}NOT among the members found under the request's email address ✘{RESET}"]
        return [*lines, f"signed up  {member.get('timestamp_signup', '?')[:10]} (audience {member.get('list_id')})",
                f"identity   {_identity(member.get('email_address'))}"]
    if system == "analytics":
        found = _read("mixpanel_engage_query", {"project_id": MIXPANEL_PROJECT, "where": f'properties["$email"] == "{REQUEST_EMAIL}"'})
        profile = next((p for p in found.get("results", []) if p.get("$distinct_id") == rid), None)
        if profile is None:
            return [*lines, f"identity   {RED}NOT among the profiles found under the request's email address ✘{RESET}"]
        return [*lines, f"created    {profile['$properties'].get('$created', '?')[:10]}",
                f"identity   {_identity(profile['$properties'].get('$email'))}"]
    return lines


def ask_human(stop: dict[str, Any], mode: str | None, deny_systems: set[str]) -> str:
    """Show an `action_approval` (or connect) interrupt and return the resume value."""
    if stop["type"] == "connect_required":
        print(f"\n{AMBER}CONNECT REQUIRED{RESET} {stop['tool']}: open {stop['url']}")
        if mode is None:
            input("Connect the account, then press Enter: ")
        return "connected"
    print(f"\n{AMBER}{BOLD}APPROVAL REQUIRED{RESET}  {BOLD}{stop['tool']}{RESET}  {DIM}(cannot be undone){RESET}")
    print(f"  arguments  {short(stop['arguments'], 160)}")
    print(f"  rationale  {stop.get('rationale', '')}   {DIM}(the coordinator's words){RESET}")
    for line in verified_facts(stop):
        print(f"  {line}   {DIM}(read by this console){RESET}")
    print(f"  expires    {stop.get('expires_at')}")
    target = target_of(stop["tool"], stop["arguments"])
    if target and target[0] in deny_systems:
        print(f"  {DIM}--deny-system {target[0]}: answering deny{RESET}")
        return "deny"
    if mode:
        print(f"  {DIM}--{mode}: answering {mode}{RESET}")
        return mode
    while True:
        answer = input("  [a]pprove / [d]eny: ").strip().lower()
        if answer in ("a", "approve"):
            return "approve"
        if answer in ("d", "deny"):
            return "deny"


async def run_turn(coordinator_input: Any, thread_id: str, saver: InMemorySaver, started: float) -> Any:
    """One run of the graph (a prompt, or a resume): MicroVMs start on first use and pause at the end."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    async with team_sandboxes(thread_id) as sandboxes:
        coordinator = build_coordinator(sandboxes, checkpointer=saver)
        async for event in coordinator.astream_events(coordinator_input, config=config, version="v2"):
            if event["event"] != "on_tool_start":
                continue
            agent = event.get("metadata", {}).get("lc_agent_name") or "erasure-coordinator"
            color = COLORS.get(agent, BOLD)
            print(f"{DIM}{time.monotonic() - started:6.1f}s{RESET} {color}{agent:>19}{RESET} → {event['name']}"
                  f"{DIM}({short(event['data'].get('input'))}){RESET}")
        used = sorted(sandboxes.started())
    if used:
        print(f"{DIM}        MicroVMs paused: {', '.join(used)}{RESET}")
    return await coordinator.aget_state(config)


def print_audit(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    print(f"{DIM}gateway audit:{RESET}")
    for e in entries:
        where = f"  {DIM}{' '.join(e['target'])}{RESET}" if e.get("target") else ""
        if e["event"] == "decision":
            print(f"  {DIM}decision{RESET}  {e['tool']}  {e['decision']}{where}")
        else:
            print(f"  {TONE.get(e['status'], '')}{e['status']:>8}{RESET}  {e['tool']}{where}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", default=None, help="thread id (default: a new one)")
    parser.add_argument("--ask", action="append", help="a prompt, one turn each (default: the request, then a follow-up)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--approve", dest="mode", action="store_const", const="approve", help="approve every request")
    group.add_argument("--deny", dest="mode", action="store_const", const="deny", help="deny every request")
    parser.add_argument("--deny-system", action="append", default=[], choices=sorted(SYSTEMS),
                        help="deny the erasure in this system (repeatable); the others follow --approve/--deny or ask")
    parser.add_argument("--delete", action="store_true", help="delete the thread's MicroVMs at the end")
    args = parser.parse_args()

    thread_id = args.thread or f"era-{uuid.uuid4().hex[:8]}"
    saver = InMemorySaver()
    print(f"thread {thread_id} ({lifecycle.config.transport}; the five systems are mocks)")
    seen = 0
    try:
        started = time.monotonic()
        template = await asyncio.to_thread(templates.get)
        print(f"template {template.checkpoint_name} {'built' if template.built else 'ready'} in {time.monotonic() - started:.1f} s")
        for prompt in args.ask or [REQUEST, FOLLOW_UP]:
            print(f"\n{BOLD}>>> {prompt}{RESET}")
            turn_started = time.monotonic()
            state = await run_turn({"messages": [{"role": "user", "content": prompt}]}, thread_id, saver, turn_started)
            while interrupts := [i for task in state.tasks for i in task.interrupts]:
                answers = {i.id: await asyncio.to_thread(ask_human, i.value, args.mode, set(args.deny_system)) for i in interrupts}
                state = await run_turn(Command(resume=answers), thread_id, saver, turn_started)
            print(f"\n{text(state.values['messages'][-1])}\n")
            changes = [e for e in audited.audit[seen:] if e["event"] == "decision" or e["status"] != "allow"]
            print_audit(changes)  # reads are many and all `allow`: only what changed or was stopped
            seen = len(audited.audit)
        if REPORT.exists():
            print(f"\nreport: {REPORT}")
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
            print(f"The thread's MicroVMs are paused (the raw records were deleted in them after classifying); delete them with --delete "
                  f"or scripts/delete_all.py. The conversation lives in this process only: --thread {thread_id} "
                  "in a new run reuses the MicroVMs, not what was said.")
        gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
