"""Tests for agent_host.supervisor: GatewaySettings, LazyGateway and the async-tool flow.

Fast: hosts are FakeAgentServers behind an httpx.MockTransport
(fake_agent_server.py) and the supervisor's "LLM" is a ScriptedChatModel
(scripted_model.py). What is real: deepagents' AsyncSubAgentMiddleware and its
five tools, the LangGraph SDK client they use, and the embedded gateway. The
same script runs against real `langgraph dev` hosts in test_supervisor_e2e.py
(opt-in).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from deepagents import create_deep_agent
from fake_agent_server import FakeHosts, make_transport
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk import get_sync_client
from scripted_model import (
    ScriptedChatModel,
    check_result,
    last_tool_message,
    say,
    task_ids,
    tool_call,
    tool_calls,
)

from langchain_digitalocean_microvm.agent_host import (
    GatewaySettings,
    HostSpec,
    LazyGateway,
    export_sdk_api_key,
)
from langchain_digitalocean_microvm.agent_host.gateway import SubAgentGateway
from langchain_digitalocean_microvm.agent_host.store import GatewayStore
from langchain_digitalocean_microvm.agent_host.supervisor import (
    ARTIFACTS_TOOL_NAME,
    GATEWAY_START_MIDDLEWARE_NAME,
    GATEWAY_TOKEN_ENV,
    HOST_MODE_ENV,
    SDK_API_KEY_ENV,
)

TOKEN = "gw-lab-token"
CONFIG = {"configurable": {"thread_id": "supervisor"}}


class ClosableFakeHosts(FakeHosts):
    """FakeHosts plus the `close()` LocalDevServerHost has (destroy everything)."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        for host_id in list(self.servers):
            self.delete(host_id=host_id)


@pytest.fixture
def hosts() -> ClosableFakeHosts:
    return ClosableFakeHosts()


def make_lazy(tmp_path: Path, hosts: FakeHosts, **overrides: Any) -> LazyGateway:
    kwargs: dict[str, Any] = {
        "settings": GatewaySettings(host_mode="local", token=TOKEN, db_path=tmp_path / "gateway.sqlite"),
        "hosts_factory": lambda mode: hosts,
        "register_atexit": False,
        "export_sdk_api_key": False,
        "poll_interval": 3600.0,  # ticks are not needed: `check` proxies live runs and caches terminal ones
        "upstream_transport": make_transport(hosts),
    }
    kwargs.update(overrides)
    return LazyGateway({"echo": HostSpec(graph_id="echo-graph")}, **kwargs)


# ---------------------------------------------------------------------------
# GatewaySettings
# ---------------------------------------------------------------------------


def test_settings_default_to_fake_mode_with_a_generated_token(monkeypatch) -> None:
    monkeypatch.delenv(HOST_MODE_ENV, raising=False)
    monkeypatch.delenv(GATEWAY_TOKEN_ENV, raising=False)

    settings = GatewaySettings.from_env(token_factory=lambda: "generated-token")

    assert settings.host_mode == "local"
    assert settings.token == "generated-token"
    assert settings.resolved_db_path.name == "gateway-local.sqlite"


def test_settings_read_mode_and_token_from_env(monkeypatch) -> None:
    monkeypatch.setenv(HOST_MODE_ENV, "microvm")
    monkeypatch.setenv(GATEWAY_TOKEN_ENV, "s3cret")

    settings = GatewaySettings.from_env()

    assert (settings.host_mode, settings.token) == ("microvm", "s3cret")
    assert settings.resolved_db_path.name == "gateway.sqlite"  # real MicroVMs never share the local mode file


def test_settings_generate_a_token_for_real_microvms_too(monkeypatch) -> None:
    monkeypatch.setenv(HOST_MODE_ENV, "microvm")
    monkeypatch.delenv(GATEWAY_TOKEN_ENV, raising=False)
    settings = GatewaySettings.from_env(token_factory=lambda: "per-process")
    assert (settings.host_mode, settings.token) == ("microvm", "per-process")


def test_settings_reject_unknown_modes(monkeypatch) -> None:
    monkeypatch.setenv(HOST_MODE_ENV, "cloud")
    with pytest.raises(ValueError, match=HOST_MODE_ENV):
        GatewaySettings.from_env()
    with pytest.raises(ValueError, match=HOST_MODE_ENV):
        GatewaySettings(host_mode="cloud", token="x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LazyGateway lifecycle
# ---------------------------------------------------------------------------


def test_specs_exist_before_anything_starts(tmp_path, hosts) -> None:
    lazy = make_lazy(tmp_path, hosts)

    spec = lazy.subagent(name="echo", description="echoes")

    assert spec["url"] == lazy.url
    assert spec["graph_id"] == "echo"
    assert spec["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert not lazy.started
    assert hosts.spawned() == []
    with pytest.raises(ValueError, match="Unknown graph"):
        lazy.subagent(name="analyst", description="not served here")
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{lazy.url}/ok", timeout=2.0)  # the port is reserved, nobody listens yet


def test_ensure_started_is_idempotent_and_exports_the_sdk_api_key(tmp_path, hosts, monkeypatch) -> None:
    monkeypatch.delenv(SDK_API_KEY_ENV, raising=False)
    factory_calls: list[str] = []

    def factory(mode: str) -> FakeHosts:
        factory_calls.append(mode)
        return hosts

    lazy = make_lazy(tmp_path, hosts, hosts_factory=factory, export_sdk_api_key=True)
    try:
        assert lazy.ensure_started() == lazy.url
        assert httpx.get(f"{lazy.url}/ok", timeout=5.0).json() == {"ok": True}
        assert lazy.ensure_started() == lazy.url
        assert factory_calls == ["local"]
        assert os.environ[SDK_API_KEY_ENV] == TOKEN
        # the SDK now authenticates with x-api-key alone, i.e. without the Bearer header the spec adds
        client = get_sync_client(url=lazy.url)
        assert "thread_id" in client.threads.create()
    finally:
        lazy.close()


def test_export_sdk_api_key_keeps_a_different_value_quietly(monkeypatch, caplog) -> None:
    """Another gateway's token (or a LangSmith key) stays; the Bearer header authenticates regardless."""
    monkeypatch.setenv(SDK_API_KEY_ENV, "another-gateways-token")
    with caplog.at_level(logging.WARNING, logger="langchain_digitalocean_microvm.agent_host.supervisor"):
        export_sdk_api_key("gw-token")
    assert os.environ[SDK_API_KEY_ENV] == "another-gateways-token"
    assert not caplog.records


def test_middleware_starts_the_gateway_before_the_first_step(tmp_path, hosts) -> None:
    lazy = make_lazy(tmp_path, hosts)
    middleware = lazy.middleware()
    try:
        assert middleware.before_agent({}, None) is None
        assert lazy.started
        assert asyncio.run(middleware.abefore_agent({}, None)) is None  # async hook, as under `langgraph dev`
        assert middleware.name == GATEWAY_START_MIDDLEWARE_NAME
    finally:
        lazy.close()


def test_close_destroys_fake_hosts_and_records_it(tmp_path, hosts) -> None:
    lazy = make_lazy(tmp_path, hosts)
    lazy.ensure_started()
    client = get_sync_client(url=lazy.url, api_key=TOKEN)
    tid = client.threads.create()["thread_id"]
    client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "ping"}]})
    assert hosts.spawned() == [tid]

    lazy.close()

    assert hosts.closed and hosts.servers == {}
    assert not lazy.started
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{lazy.url}/ok", timeout=2.0)
    store = GatewayStore(tmp_path / "gateway.sqlite")
    try:
        assert store.get_host(tid).status == "destroyed"  # a later process reports the task as an error, not a hang
    finally:
        store.close()
    lazy.close()  # idempotent


def test_one_agent_takes_two_gateways_once_each_has_its_own_names(tmp_path, hosts) -> None:
    """Middleware names must be unique per agent; tools with one name would shadow each other silently."""
    first = make_lazy(tmp_path / "a", hosts)
    second = make_lazy(tmp_path / "b", hosts)
    with pytest.raises(AssertionError, match="duplicate middleware"):
        create_deep_agent(model=ScriptedChatModel(), middleware=[first.middleware(), second.middleware()])

    tools = [first.artifacts_tool(), second.artifacts_tool(name="fetch_other_artifacts")]
    create_deep_agent(
        model=ScriptedChatModel(),
        tools=tools,
        middleware=[first.middleware(), second.middleware(name="other-gateway-start")],
    )
    assert [t.name for t in tools] == [ARTIFACTS_TOOL_NAME, "fetch_other_artifacts"]
    assert not first.started and not second.started  # building the agent starts nothing


def test_artifacts_of_a_task_another_gateway_ran_say_why(tmp_path, hosts, caplog) -> None:
    """With two same-named tools the agent keeps one: asked for the other gateway's task, it says so."""
    lazy = make_lazy(tmp_path, hosts)
    try:
        lazy.ensure_started()
        with caplog.at_level(logging.WARNING, logger="langchain_digitalocean_microvm.agent_host.supervisor"):
            answer = lazy.artifacts_tool().invoke({"task_id": "a-task-of-another-gateway"})
        assert "this gateway has no such task" in answer and "artifacts_tool(name=...)" in answer
        assert "a-task-of-another-gateway" in caplog.text
    finally:
        lazy.close()


def test_embedded_gateway_serves_requests_under_the_dev_servers_blocking_detector(tmp_path, hosts) -> None:
    """`langgraph dev` activates blockbuster; the gateway's synchronous SQLite must still work on its own thread."""
    blockbuster = pytest.importorskip("blockbuster")
    detector = blockbuster.BlockBuster()
    detector.activate()
    lazy = make_lazy(tmp_path, hosts)
    try:
        lazy.ensure_started()
        client = get_sync_client(url=lazy.url, api_key=TOKEN)
        tid = client.threads.create()["thread_id"]  # sqlite INSERT inside the gateway's event loop
        run = client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "ping"}]})
        assert client.runs.get(tid, run["run_id"])["status"] == "pending"
    finally:
        detector.deactivate()
        lazy.close()


@pytest.mark.parametrize("flag", ["context", "thread"])
def test_embedded_gateway_thread_sets_either_kind_of_blockbuster_skip_flag(monkeypatch, flag: str) -> None:
    """blockbuster's skip flag is a context variable up to 1.5.28 and a thread-local flag from 1.5.29."""
    module = types.ModuleType("blockbuster.blockbuster")
    if flag == "context":
        skip = contextvars.ContextVar("blockbuster_skip", default=False)
        module.blockbuster_skip = skip  # type: ignore[attr-defined]

        def skipped() -> bool:
            return skip.get()
    else:
        local = threading.local()
        module._skip = local  # type: ignore[attr-defined]

        def skipped() -> bool:
            return getattr(local, "value", False)
    package = types.ModuleType("blockbuster")
    package.blockbuster = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "blockbuster", package)
    monkeypatch.setitem(sys.modules, "blockbuster.blockbuster", module)
    seen: list[bool] = []
    server = SimpleNamespace(run=lambda: seen.append(skipped()))  # uvicorn's run(), on the gateway's thread
    thread = threading.Thread(target=SubAgentGateway._serve_embedded, args=(SimpleNamespace(_server=server),))
    thread.start()
    thread.join()
    assert seen == [True]


# ---------------------------------------------------------------------------
# The five async-subagent tools, through the real middleware
# ---------------------------------------------------------------------------


def _turn(agent, model: ScriptedChatModel, prompt: str, *steps) -> dict[str, Any]:
    model.script(*steps)
    return agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=CONFIG)


def _check(agent, model: ScriptedChatModel, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _turn(agent, model, f"check {task_id}", tool_call("check_async_task", task_id=task_id), say("checked"))
    return check_result(last_tool_message(state["messages"])), state


def test_async_task_tools_flow_through_the_gateway(tmp_path, hosts) -> None:
    lazy = make_lazy(tmp_path, hosts)
    model = ScriptedChatModel()
    agent = create_deep_agent(
        model=model,
        subagents=[lazy.subagent(name="echo", description="Replies 'echo: <message>'")],
        middleware=[lazy.middleware()],
        checkpointer=InMemorySaver(),
    )
    assert not lazy.started  # building the graph (import time under `langgraph dev`) starts nothing
    try:
        # 1. Two tasks launched in one model turn.
        state = _turn(
            agent,
            model,
            "launch two echo tasks",
            tool_calls(
                ("start_async_task", {"description": "ping one", "subagent_type": "echo"}),
                ("start_async_task", {"description": "ping two", "subagent_type": "echo"}),
            ),
            say("Launched both."),
        )
        assert lazy.started
        tasks = state["async_tasks"]
        assert len(tasks) == 2 and all(t["status"] == "running" for t in tasks.values())
        assert sorted(task_ids(state["messages"])) == sorted(tasks)  # ids the model can read back
        assert sorted(hosts.spawned()) == sorted(tasks)  # per-task hosts, host id == task id
        by_text = {
            hosts.servers[tid].threads[tid]["values"]["messages"][0]["content"]: tid for tid in tasks
        }
        t1, t2 = by_text["ping one"], by_text["ping two"]
        run1 = tasks[t1]["run_id"]
        assert hosts.servers[t1].runs[run1]["assistant_id"] == "echo-graph"  # the gateway mapped the name

        # 2. Check while running: proxied to the host.
        result, _ = _check(agent, model, t1)
        assert {k: result[k] for k in ("status", "thread_id")} == {"status": "pending", "thread_id": t1}  # newer: a `note`

        # 3. The host finishes; check returns the result and the gateway caches it.
        hosts.servers[t1].complete(run1, "echo: ping one")
        result, state = _check(agent, model, t1)
        assert result == {"status": "success", "thread_id": t1, "result": "echo: ping one"}
        assert state["async_tasks"][t1]["status"] == "success"
        assert lazy.gateway is not None and lazy.gateway.store.get_thread(t1).cached_run["status"] == "success"

        # 4. Update: a new run on the same thread with interrupt semantics; same task id.
        state = _turn(
            agent, model, "update", tool_call("update_async_task", task_id=t1, message="pong"), say("Updated.")
        )
        assert last_tool_message(state["messages"]).content == f"Updated async subagent. task_id: {t1}"
        run1b = state["async_tasks"][t1]["run_id"]
        assert run1b != run1 and hosts.servers[t1].runs[run1b]["multitask_strategy"] == "interrupt"
        assert state["async_tasks"][t1]["status"] == "running"
        hosts.servers[t1].complete(run1b, "echo: pong")
        result, _ = _check(agent, model, t1)
        assert result["result"] == "echo: pong"

        # 5. Cancel the second task while it is still pending: proxied to its host ...
        state = _turn(agent, model, "cancel", tool_call("cancel_async_task", task_id=t2), say("Cancelled."))
        assert last_tool_message(state["messages"]).content == f"Cancelled async subagent task: {t2}"
        assert hosts.servers[t2].runs[tasks[t2]["run_id"]]["status"] == "interrupted"
        assert state["async_tasks"][t2]["status"] == "cancelled"
        # ... and cancelling a finished run, where the Agent Server answers 404, still succeeds (gateway answers locally).
        state = _turn(agent, model, "cancel again", tool_call("cancel_async_task", task_id=t2), say("Cancelled."))
        assert last_tool_message(state["messages"]).content.startswith("Cancelled async subagent task")

        # 6. List, on the async code path `langgraph dev` uses.
        model.script(tool_call("list_async_tasks"), say("Listed."))
        state = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "list"}]}, config=CONFIG))
        listing = str(last_tool_message(state["messages"]).content)
        assert listing.startswith("2 tracked task(s):")
        assert f"task_id: {t1}  agent: echo  status: success" in listing
        assert f"task_id: {t2}  agent: echo  status: cancelled" in listing

        # 7. A third task whose host vanishes: the task turns into an error instead of hanging.
        state = _turn(
            agent, model, "one more", tool_call("start_async_task", description="gone", subagent_type="echo"), say("Launched.")
        )
        t3 = next(tid for tid in state["async_tasks"] if tid not in (t1, t2))
        hosts.kill(t3)
        result, state = _check(agent, model, t3)
        assert result["status"] == "error" and "unreachable" in result["error"]
        assert state["async_tasks"][t3]["status"] == "error"
    finally:
        lazy.close()
