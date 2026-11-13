"""Drive the async-subagents supervisor from a script, without `langgraph dev`.

Same supervisor as agent.py (same model, specs, and embedded
gateway), given an in-memory checkpointer so the conversation -- and with it
the `async_tasks` state -- carries across turns. It walks the whole task
lifecycle: two echo tasks at once, check one while it runs, update it, cancel the
other, list, then an echo task that saves a file the supervisor fetches
and reads, then delete a task's host through the host manager and watch the
task report an error rather than hang.

RUN (from the repository root; local mode by default -- no DigitalOcean account):
    uv run examples/04_async_subagents/run_script.py
    DO_MICROVM_HOST_MODE=microvm uv run examples/04_async_subagents/run_script.py --with-researcher --delete
Needs a model for the supervisor: DO_MODEL_ACCESS_KEY, or EXAMPLES_MODEL with its
provider's key (agent.py). In
microvm mode the tasks run on real MicroVMs: one shared host per subagent
type, so deleting "the third task's host" also removes the earlier echo
tasks' host -- their results are cached, so they can still be checked.

On real MicroVMs the hosts are left paused at the end, for the next run to reuse (and the
team limit counts them): --delete removes every live agent host in the gateway's store at the
end, also after a failed run; or use scripts/delete_all.py / scripts/gc_microvms.py later.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # before agent.py reads the environment

from agent import build_graph, gateway  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

parser = argparse.ArgumentParser(description="Drive the async-subagents supervisor without langgraph dev.")
parser.add_argument("--with-researcher", action="store_true", help="also run one researcher task (uses the model key)")
parser.add_argument("--delete", action="store_true", help="microvm mode: delete the agent hosts at the end")
args = parser.parse_args()

graph = build_graph(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "async-subagents-example"}}


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def turn(prompt: str) -> dict:
    print(f"\n>>> {prompt}")
    state = graph.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    print(_text(state["messages"][-1]))
    return state


def delete_hosts() -> None:
    """Delete every live agent host in the gateway's store (real MicroVMs) and record it there."""
    if gateway.host_mode == "local" or gateway.gateway is None or gateway.hosts is None:
        return  # local hosts are destroyed by gateway.close()
    store = gateway.gateway.store
    for row in store.list_hosts():
        if row.status in ("failed", "destroyed"):
            continue
        try:
            gateway.hosts.delete(host_id=row.host_id)
            print(f"deleted host {row.host_id} ({row.graph_id}, MicroVM {row.microvm_id})")
        except Exception as exc:  # keep going: one failure must not leave the rest behind
            print(f"could not delete host {row.host_id}: {exc}")
            continue
        store.retire_host(row.host_id, "deleted")


try:
    turn(
        "Start two echo tasks at once, each taking about 45 seconds: one saying 'ping one' "
        "and one saying 'ping two'. Report both task ids and stop."
    )
    print(f"\n[gateway] {gateway.host_mode} hosts behind {gateway.url}; store: {gateway.settings.resolved_db_path}")
    turn("Check on the 'ping one' task.")
    turn("Update the 'ping one' task: tell it 'pong', with no delay this time.")
    turn("Cancel the 'ping two' task.")
    time.sleep(3)  # let the gateway's poll loop record the update's result
    turn("List all tasks with their current status. Then check the 'ping one' task and show its result.")

    turn(
        "Start an echo task with the message 'save hello.md: # Hello from a MicroVM'. "
        "Report its task id and stop."
    )
    time.sleep(3)  # it finishes at once; let the poll loop record it
    turn(
        "Check that echo task. When it has finished, fetch its files with fetch_subagent_artifacts "
        "and read the file you fetched."
    )

    before = set(graph.get_state(config).values.get("async_tasks", {}))
    turn("Start a third echo task that takes 90 seconds and says 'gone'. Report its task id and stop.")
    third = next(tid for tid in graph.get_state(config).values["async_tasks"] if tid not in before)

    assert gateway.hosts is not None and gateway.gateway is not None
    host_id = gateway.gateway.store.get_thread(third).host_id  # == task id per task; "pool-echo" when pooled
    print(f"\n[host manager] deleting host {host_id} (under task {third}) while its run is in flight ...")
    gateway.hosts.delete(host_id=host_id)  # the VM is gone
    time.sleep(5)  # give the gateway's poll loop a moment to notice, as a person would
    turn("Check on the third task.")

    if args.with_researcher:
        before = set(graph.get_state(config).values.get("async_tasks", {}))
        turn(
            "Launch a researcher task: write a Python script /outputs/fib.py that prints the first 15 "
            "Fibonacci numbers, run it, and report the output. Report the task id and stop."
        )
        research = next(tid for tid in graph.get_state(config).values["async_tasks"] if tid not in before)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            time.sleep(15)  # the gateway polls the host meanwhile, which also keeps a MicroVM awake
            turn(f"Check on task {research}. If it is still running just say 'still running'.")
            status = graph.get_state(config).values["async_tasks"][research]["status"]
            if status != "running":
                break
        turn(f"Fetch the files of task {research} and show me the script it wrote.")

finally:
    if args.delete:  # also after a failed run: paused MicroVMs count against the team limit
        delete_hosts()
    elif gateway.host_mode != "local":
        print("\nagent hosts are left paused for the next run; delete them with --delete or scripts/delete_all.py --yes")
    gateway.close()  # atexit would do this too; being explicit keeps the output tidy
