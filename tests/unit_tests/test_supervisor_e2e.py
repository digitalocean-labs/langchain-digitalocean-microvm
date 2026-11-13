"""End-to-end lab flow: scripted supervisor -> deepagents async tools -> LazyGateway -> `langgraph dev` hosts.

Marked `e2e` (`uv run pytest -m e2e`). The automated form of the manual acceptance flow:
launch two echo tasks at once, check one while it is still running, update it,
cancel the other, then delete the host under a running task and watch that task
report an error instead of hanging. The supervisor's model is a
ScriptedChatModel (no model key); the hosts are real local Agent Servers
serving the template's `echo` graph, whose `sleep N:` prefix keeps runs alive
long enough to observe.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from scripted_model import (
    ScriptedChatModel,
    check_result,
    last_tool_message,
    say,
    tool_call,
    tool_calls,
)

from langchain_digitalocean_microvm.agent_host import (
    GatewaySettings,
    HostSpec,
    LazyGateway,
    LocalDevServerHost,
)
from langchain_digitalocean_microvm.agent_host.store import GatewayStore

pytestmark = pytest.mark.e2e

pytest.importorskip("langgraph_sdk")
pytest.importorskip("langgraph_cli")

TOKEN = "gw-e2e-lab-token"
CONFIG = {"configurable": {"thread_id": "supervisor"}}
TERMINAL = {"success", "error", "timeout", "interrupted", "cancelled"}


def _turn(agent, model: ScriptedChatModel, prompt: str, *steps) -> dict[str, Any]:
    model.script(*steps)
    return agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=CONFIG)


def _check(agent, model: ScriptedChatModel, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _turn(agent, model, f"check {task_id}", tool_call("check_async_task", task_id=task_id), say("checked"))
    return check_result(last_tool_message(state["messages"])), state


def _check_until(
    agent, model: ScriptedChatModel, task_id: str, done: Callable[[dict[str, Any]], bool], *, timeout: float = 60.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        result, state = _check(agent, model, task_id)
        if done(result):
            return result, state
        if time.monotonic() > deadline:
            raise AssertionError(f"task {task_id} still {result} after {timeout}s")
        time.sleep(0.5)


def test_lab_flow_against_real_local_agent_servers(tmp_path) -> None:
    hosts = LocalDevServerHost(root_dir=tmp_path / "hosts", ready_timeout=120.0)
    lazy = LazyGateway(
        {"echo": HostSpec(graph_id="echo", n_jobs_per_worker=2)},
        settings=GatewaySettings(host_mode="local", token=TOKEN, db_path=tmp_path / "gateway.sqlite"),
        hosts_factory=lambda mode: hosts,
        register_atexit=False,
        export_sdk_api_key=False,
        poll_interval=1.0,  # the lab's snappy setting
        pause_grace=3600.0,  # keep hosts awake; pause/cache behaviour is covered by test_gateway_e2e.py
    )
    model = ScriptedChatModel()
    agent = create_deep_agent(
        model=model,
        subagents=[lazy.subagent(name="echo", description="Replies 'echo: <message>'; 'sleep N:' delays it")],
        middleware=[lazy.middleware()],
        checkpointer=InMemorySaver(),
    )
    assert not lazy.started
    try:
        # 1. Two tasks at once, each busy for a while.
        state = _turn(
            agent,
            model,
            "launch two",
            tool_calls(
                ("start_async_task", {"description": "sleep 20: ping one", "subagent_type": "echo"}),
                ("start_async_task", {"description": "sleep 20: ping two", "subagent_type": "echo"}),
            ),
            say("Launched both."),
        )
        tasks = state["async_tasks"]
        assert len(tasks) == 2 and all(t["status"] == "running" for t in tasks.values())
        assert sorted(h.host_id for h in hosts.list_hosts()) == sorted(tasks)  # one real `langgraph dev` per task
        t1, t2 = tasks  # tool-call order: "ping one" first

        # 2. Check while running.
        result, _ = _check(agent, model, t1)
        assert result["status"] in {"pending", "running"}, result

        # 3. Update it: the sleeping run is interrupted, "pong" is answered right away.
        state = _turn(agent, model, "update", tool_call("update_async_task", task_id=t1, message="pong"), say("Updated."))
        assert last_tool_message(state["messages"]).content == f"Updated async subagent. task_id: {t1}"
        result, state = _check_until(agent, model, t1, lambda r: r["status"] in TERMINAL, timeout=45.0)
        assert result == {"status": "success", "thread_id": t1, "result": "echo: pong"}
        assert state["async_tasks"][t1]["status"] == "success"

        # 4. Cancel the other one while it sleeps.
        state = _turn(agent, model, "cancel", tool_call("cancel_async_task", task_id=t2), say("Cancelled."))
        assert last_tool_message(state["messages"]).content == f"Cancelled async subagent task: {t2}"
        assert state["async_tasks"][t2]["status"] == "cancelled"

        # 5. List: terminal statuses come from the supervisor's own state, no host is asked.
        state = _turn(agent, model, "list", tool_call("list_async_tasks"), say("Listed."))
        listing = str(last_tool_message(state["messages"]).content)
        assert listing.startswith("2 tracked task(s):")
        assert f"task_id: {t1}  agent: echo  status: success" in listing
        assert f"task_id: {t2}  agent: echo  status: cancelled" in listing

        # 6. A third task loses its host mid-run: error, not a hang.
        state = _turn(
            agent,
            model,
            "one more",
            tool_call("start_async_task", description="sleep 90: gone", subagent_type="echo"),
            say("Launched."),
        )
        t3 = next(tid for tid in state["async_tasks"] if tid not in (t1, t2))
        hosts.delete(host_id=t3)  # "the MicroVM is gone"
        result, state = _check_until(agent, model, t3, lambda r: r["status"] in TERMINAL, timeout=45.0)
        assert result["status"] == "error", result
        assert any(word in result["error"] for word in ("unreachable", "gone", "failed", "destroyed")), result
        assert state["async_tasks"][t3]["status"] == "error"
    finally:
        lazy.close()

    # Teardown killed every remaining local server and said so in the store.
    assert hosts.list_hosts() == []
    store = GatewayStore(tmp_path / "gateway.sqlite")
    try:
        assert {row.status for row in store.list_hosts()} <= {"destroyed", "failed"}
    finally:
        store.close()


def test_supervisor_fetches_files_a_real_host_saved(tmp_path) -> None:
    """Artifacts: two tasks on ONE pooled host save the same file name; each fetch gets its own task's file."""
    hosts = LocalDevServerHost(root_dir=tmp_path / "hosts", ready_timeout=120.0)
    lazy = LazyGateway(
        {"echo": HostSpec(graph_id="echo", reuse="pooled", n_jobs_per_worker=2)},
        settings=GatewaySettings(host_mode="local", token=TOKEN, db_path=tmp_path / "gateway.sqlite"),
        hosts_factory=lambda mode: hosts,
        register_atexit=False,
        export_sdk_api_key=False,
        poll_interval=3600.0,  # ticks are driven below
        pause_grace=0.0,
    )
    model = ScriptedChatModel()
    agent = create_deep_agent(
        model=model,
        subagents=[lazy.subagent(name="echo", description="Replies 'echo: <message>'; 'save NAME:' writes a file")],
        middleware=[lazy.middleware()],
        tools=[lazy.artifacts_tool()],
        checkpointer=InMemorySaver(),
    )
    try:
        state = _turn(
            agent,
            model,
            "launch two",
            tool_calls(
                ("start_async_task", {"description": "save report.md: from task one", "subagent_type": "echo"}),
                ("start_async_task", {"description": "save report.md: from task two", "subagent_type": "echo"}),
            ),
            say("Launched both."),
        )
        t1, t2 = state["async_tasks"]
        assert [h.host_id for h in hosts.list_hosts()] == ["pool-echo"]  # one shared host
        for tid, text in ((t1, "from task one"), (t2, "from task two")):
            result, _ = _check_until(agent, model, tid, lambda r: r["status"] in TERMINAL)
            assert result["result"] == f"echo: {text} (saved /outputs/report.md)"

        # Both idle: the next ticks pause the shared host (SIGSTOP); fetching wakes it.
        lazy.gateway.tick_sync()
        lazy.gateway.tick_sync()
        assert lazy.gateway.store.get_host("pool-echo").status == "paused"

        for tid, text in ((t1, "from task one"), (t2, "from task two")):
            state = _turn(agent, model, "fetch", tool_call("fetch_subagent_artifacts", task_id=tid), say("Fetched."))
            assert f"/subagents/{tid}/report.md" in last_tool_message(state["messages"]).content
            assert state["files"][f"/subagents/{tid}/report.md"]["content"] == text

        hosts.delete(host_id="pool-echo")  # "the MicroVM is gone"
        state = _turn(agent, model, "again", tool_call("fetch_subagent_artifacts", task_id=t1), say("Gone."))
        assert "are gone" in last_tool_message(state["messages"]).content
    finally:
        lazy.close()
