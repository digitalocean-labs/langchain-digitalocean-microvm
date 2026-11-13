"""Drive the sales assistant from a script, without `langgraph dev`.

Same assistant as agent.py, with an in-memory checkpointer so the conversation carries across
turns. Every run does what the graph factory does under `langgraph dev`: get the thread's
sandbox (cloned from the template on the first run, resumed afterwards), run, pause. Human
approvals (a new customer, a saved draft) end a run and the answer starts the next one, so
they are visible as extra resumes; the script approves them and prints what it approved.

Starts the mock mail server (and resets the mailbox) unless something already listens on its port.

RUN (from the repository root; DO_MICROVM_TRANSPORT=local needs no DigitalOcean account)
    uv run examples/07_sales_assistant/run_script.py                   # territory report, then the RFQ
    uv run examples/07_sales_assistant/run_script.py --ask "..."       # your own prompt(s)
    uv run examples/07_sales_assistant/run_script.py --delete --save outputs
        # copy /outputs/ to outputs/<thread>/ after each turn and delete the sandbox at the end
Needs a model: DO_MODEL_ACCESS_KEY, or EXAMPLES_MODEL with its provider's key (models.py).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from agent import (
    HERE,
    MAIL_URL,
    build_agent,
    lifecycle,
    templates,
    threads,
)
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_digitalocean_microvm.client import api_calls
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError

PROMPTS = [
    "How's my book of business looking? Give me a territory report.",
    "Check the inbox for any quote requests and process them.",
]


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


@contextlib.contextmanager
def mail_server():
    """Make sure the mock mail MCP server answers: start it (with a fresh mailbox) if nothing listens."""
    url = urlparse(MAIL_URL)
    with socket.socket() as probe:
        if probe.connect_ex((url.hostname, url.port)) == 0:
            yield  # somebody else's server: leave its mailbox alone
            return
    # Seed a fresh inbox, then start the server and wait (up to ~10 s) for its port to open.
    subprocess.run([sys.executable, str(HERE / "mail_server" / "send_to_inbox.py"), "--reset"], check=True, stdout=subprocess.DEVNULL)
    server = subprocess.Popen([sys.executable, str(HERE / "mail_server" / "mock_mail_server.py")], stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            with socket.socket() as probe:
                if probe.connect_ex((url.hostname, url.port)) == 0:
                    break
            time.sleep(0.2)
        yield
    finally:
        server.terminate()
        server.wait()


def save_outputs(backend, dest: Path) -> None:
    """Copy every file under the sandbox's /outputs/ to dest, same relative paths."""
    if backend.execute("test -d /outputs").exit_code != 0:
        return
    found = backend.glob("**", path="/outputs").matches or []
    # deepagents >= 0.7.21 returns absolute paths; earlier 0.7.x, paths relative to `path`
    paths = [m["path"] if m["path"].startswith("/") else f"/outputs/{m['path']}" for m in found if not m.get("is_dir")]
    for response in backend.download_files(paths) if paths else []:
        if response.content is None:
            continue
        target = dest / response.path.removeprefix("/outputs/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        print(f"saved {response.path} -> {target}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thread", default=None, help="thread id (default: a new one); the sandbox is sales-<thread>")
    parser.add_argument("--ask", action="append", help="a prompt, one turn each (default: the territory report, then the RFQ)")
    parser.add_argument("--delete", action="store_true", help="delete the thread's sandbox at the end")
    parser.add_argument("--save", type=Path, metavar="DIR", help="after each turn, copy the sandbox's /outputs/ to DIR/<thread>/")
    args = parser.parse_args()

    thread_id = args.thread or f"lab-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    saver = InMemorySaver()
    print(f"thread {thread_id}; sandbox {threads.sandbox_id(thread_id)} ({lifecycle.config.transport})")

    async def run(payload) -> dict:
        """One run, as the server would do it: sandbox ready, agent, pause."""
        started, calls = time.monotonic(), api_calls()
        async with threads.asandbox(thread_id) as backend:
            ready, ready_calls = time.monotonic() - started, calls_since(calls)
            agent = await build_agent(backend, checkpointer=saver)
            run_started, calls = time.monotonic(), api_calls()
            state = await agent.ainvoke(payload, config=config)
            ran, run_calls = time.monotonic() - run_started, calls_since(calls)
            if args.save and not state.get("__interrupt__"):
                save_outputs(backend, args.save / thread_id)  # while it runs: paused, it would need a resume
            pause_started, calls = time.monotonic(), api_calls()
        print(f"[sandbox {backend.id} ready in {ready:.1f} s (API: {ready_calls}); agent {ran:.1f} s (API: {run_calls}); "
              f"paused in {time.monotonic() - pause_started:.1f} s (API: {calls_since(calls)})]")
        return state

    try:
        with mail_server():
            started = time.monotonic()  # a run would do this too, on the thread's first sandbox
            template = await asyncio.to_thread(templates.get)
            print(f"template {template.checkpoint_name} {'built' if template.built else 'reused'} in {time.monotonic() - started:.1f} s")
            for prompt in args.ask or PROMPTS:
                print(f"\n>>> {prompt}")
                turn_started = time.monotonic()
                state = await run({"messages": [{"role": "user", "content": prompt}]})
                # A gated write (add_customer, mail_create_draft) ends the run with an interrupt;
                # approve it and resume with Command(resume=...) until the turn completes.
                while interrupts := state.get("__interrupt__"):
                    decisions: dict[str, dict] = {}
                    for interrupt in interrupts:
                        requests = interrupt.value["action_requests"]
                        for request in requests:
                            print(f"approval requested: {request['name']}({request['args']}) -> approve")
                        decisions[interrupt.id] = {"decisions": [{"type": "approve"} for _ in requests]}
                    state = await run(Command(resume=next(iter(decisions.values())) if len(decisions) == 1 else decisions))
                print(f"\n{text(state['messages'][-1])}\n[turn {time.monotonic() - turn_started:.0f} s]")
    finally:
        if args.delete:  # also after a failed turn: paused MicroVMs count against the team limit
            try:
                lifecycle.delete(sandbox_id=threads.sandbox_id(thread_id))
                print(f"\ndeleted sandbox {threads.sandbox_id(thread_id)}")
            except SandboxNotFoundError:
                print(f"\nno sandbox {threads.sandbox_id(thread_id)} to delete")
        else:
            print(f"\nsandbox {threads.sandbox_id(thread_id)} is paused; continue with --thread {thread_id}, delete with --delete or scripts/delete_all.py")
        print(f"MicroVMs API requests in total: {calls_since({})}")


if __name__ == "__main__":
    asyncio.run(main())
