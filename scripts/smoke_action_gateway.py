"""Live smoke test: Action Gateway tools for a supervisor, with the approval and connect interrupts.

Creates one Action Gateway endpoint (`microvm-ag-*`) and drives its `action_invoke` tool from
a one-node LangGraph graph, answering the interrupts at the terminal:

  1. `action_search` lists the endpoint's tools;
  2. an allowed read-only action (`digitalocean_size-list`) runs; if the actor has no OAuth
     connection yet, the connect link is printed, you open it, press Enter, and the call retries;
  3. an `ask` action (`digitalocean_region-list`) interrupts with `action_approval`; you type
     approve or deny, the decision goes to the gateway and an approved call runs once.

Needs DO_ACTION_GATEWAY_TOKEN (scopes `action_gateway:*`) and DO_ACTION_GATEWAY_ACTOR_ID (the actor: by default
the part of your email before the @) in .env or the environment, and a terminal. The endpoint is deleted in a `finally` block.
Results go to outputs/live/smoke_action_gateway-<time>.json.

    uv run python scripts/smoke_action_gateway.py [--slug-version N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from _env import add_env_file_argument
from dotenv import dotenv_values
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGateway,
    ActionPolicy,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "live"
ALLOWED, ASKED = "digitalocean_size-list", "digitalocean_region-list"


class State(TypedDict, total=False):
    answer: str


def drive(gateway: ActionGateway, tool: str, record: list[dict[str, Any]]) -> str:
    """Run `tool` through `action_invoke` in a graph and answer each interrupt at the terminal."""
    invoke = next(t for t in gateway.tools() if t.name == "action_invoke")

    def node(state: State) -> State:
        return {"answer": invoke.invoke({"tool": tool, "arguments": {}, "rationale": "smoke test"})}

    graph = StateGraph(State)
    graph.add_node("act", node)
    graph.add_edge(START, "act")
    graph.add_edge("act", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"smoke-{tool}"}}
    out = app.invoke({}, config)
    while "__interrupt__" in out:
        stop = out["__interrupt__"][0].value
        record.append({"interrupt": stop})
        print(json.dumps(stop, indent=2))
        if stop["type"] == "connect_required":
            input("Open the link, connect, then press Enter (Ctrl-C to cancel): ")
            resume = "connected"
        else:
            resume = input("approve or deny: ").strip() or "deny"
        out = app.invoke(Command(resume=resume), config)
    return out["answer"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug-version", default=None, help="pin the two slugs to this version (default: no pin)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="default: %(default)s")
    add_env_file_argument(parser)
    args = parser.parse_args()
    for key, value in dotenv_values(args.env_file).items():
        if key.startswith("DO_ACTION_GATEWAY_") and value:
            os.environ.setdefault(key, value)
    if not (os.environ.get("DO_ACTION_GATEWAY_TOKEN") and os.environ.get("DO_ACTION_GATEWAY_ACTOR_ID")):
        print("set DO_ACTION_GATEWAY_TOKEN and DO_ACTION_GATEWAY_ACTOR_ID", file=sys.stderr)
        return 2

    if not sys.stdin.isatty():
        print("this script asks for input (connect, approve or deny): run it in a terminal", file=sys.stderr)
        return 2

    record: list[dict[str, Any]] = []
    started = time.monotonic()
    policy = ActionPolicy(allow=[ALLOWED], ask=[ASKED])
    slugs = [f"{s}@{args.slug_version}" if args.slug_version else s for s in (ALLOWED, ASKED)]
    gateway = ActionGateway(policy, tools=slugs)
    try:
        with gateway:
            search = next(t for t in gateway.tools() if t.name == "action_search")
            record.append({"search": json.loads(search.invoke({"query": "list"}))})
            for step, tool in (("allowed", ALLOWED), ("ask", ASKED)):
                answer = drive(gateway, tool, record)
                record.append({step: answer[:600]})
                print(f"[{step}] {answer[:300]}")
    finally:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"smoke_action_gateway-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps({"seconds": round(time.monotonic() - started, 1), "steps": record}, indent=2))
        print(f"results: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
