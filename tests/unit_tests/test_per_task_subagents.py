"""Per-task subagent sandboxes: one session per `task` call (or per key), local mode, real deepagents supervisor.

A scripted model drives the parent and its subagents. Subagent steps are callables that read their own prompt (the
task description, their tool results so far), so parallel tasks may take steps in any order.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from scripted_model import ScriptedChatModel, say, tool_call, tool_calls

from langchain_digitalocean_microvm.agent_host.store import GatewayStore
from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import (
    PER_TASK_SANDBOX_TAG,
    SUBAGENT_SANDBOX_TAG,
)
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import (
    MicroVMForbiddenError,
    SandboxError,
)
from langchain_digitalocean_microvm.provider import MicroVMLifecycle
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.subagents import (
    PerTask,
    PerTaskSandbox,
    TaskKeyMiddleware,
    asubagent_sandboxes,
    provision_subagent_sandboxes,
    subagent_sandbox_id,
    subagent_sandboxes,
)

THREAD = "pt-thread"


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


class CountingLifecycle(MicroVMLifecycle):
    """A local transport lifecycle that counts get-or-creates, creates and pauses, with optional delay and failures."""

    def __init__(self, *, delay: float = 0.0) -> None:
        super().__init__(MicroVMConfig(transport="local"))
        self.delay = delay
        self.fail_for: dict[str, BaseException] = {}  # sandbox_id -> raised once by the next get-or-create
        self.calls = self.creates = self.pauses = 0
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()

    def get_or_create(self, *, sandbox_id: str, **kwargs: Any) -> MicroVMSandbox:
        with self._lock:
            self.calls += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.creates += sandbox_id not in self._local_states
            failure = self.fail_for.pop(sandbox_id, None)
        try:
            time.sleep(self.delay)
            if failure is not None:
                raise failure
            return super().get_or_create(sandbox_id=sandbox_id, **kwargs)
        finally:
            with self._lock:
                self._active -= 1

    def pause(self, backend: MicroVMSandbox | None = None, **kwargs: Any) -> Any:
        with self._lock:
            self.pauses += 1
        return super().pause(backend, **kwargs)


class Lifecycle(CountingLifecycle):
    """Also counts deletes and remembers which sandbox_ids were created."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.deletes: list[str] = []
        self.created: list[str] = []
        self.tags: dict[str, tuple[str, ...]] = {}

    def get_or_create(self, *, sandbox_id: str, **kwargs: Any) -> Any:
        self.tags[sandbox_id] = tuple(kwargs.get("tags") or ())
        if sandbox_id not in self._local_states:
            self.created.append(sandbox_id)
        return super().get_or_create(sandbox_id=sandbox_id, **kwargs)

    def delete(self, *, sandbox_id: str) -> None:
        self.deletes.append(sandbox_id)
        super().delete(sandbox_id=sandbox_id)


def _specs(*tools: Any) -> list[dict[str, Any]]:
    return [
        {"name": "analyst", "description": "Analyses one repository.", "system_prompt": "You analyse.",
         "tools": list(tools), "permissions": []},
        {"name": "reviewer", "description": "Reviews.", "system_prompt": "You review.", "permissions": []},
    ]


def _description(messages: list[BaseMessage]) -> str:
    return next(str(m.content) for m in messages if m.type == "human")


def _since_task(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if m.type == "tool"]


def _out(message: BaseMessage) -> str:
    """A tool result without deepagents' `[Command ... exit code N]` footer."""
    return str(message.content).split("\n[Command")[0].strip()


def worker(messages: list[BaseMessage]) -> Any:
    """A subagent's step: list work/, write /work/<word>.md, list again, answer with both listings.

    Relative paths: the local transport runs commands in the session's root and maps only existing files' paths.
    """
    done = _since_task(messages)
    word = _description(messages).split()[0]
    if not done:
        return tool_call("execute", command="ls work 2>/dev/null; true")
    if len(done) == 1:
        return tool_call("write_file", file_path=f"/work/{word}.md", content=word)
    if len(done) == 2:
        return tool_call("execute", command="ls work")
    return say(f"before: [{_out(done[0])}] after: [{_out(done[2])}]")


def _task_answers(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in messages if m.type == "tool" and m.name == "task"]


def _agent(model: ScriptedChatModel, sb: Any, **kwargs: Any) -> Any:
    return create_deep_agent(model=model, subagents=list(sb.subagents), middleware=[*sb.middleware],
                             backend=sb.mount(StateBackend()), **kwargs)


def _run(agent: Any, text: str = "go", config: dict[str, Any] | None = None) -> dict[str, Any]:
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, config)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_without_per_task_nothing_changes() -> None:
    lifecycle = Lifecycle()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel()) as sb:
        assert sb.middleware == () and dict(sb.per_task) == {}
        assert dict(sb.sandbox_ids) == {n: subagent_sandbox_id(THREAD, n) for n in ("analyst", "reviewer")}


def test_per_task_spec_gets_a_router_and_the_parent_a_middleware() -> None:
    lifecycle = Lifecycle()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                            per_task={"analyst": PerTask()}) as sb:
        assert lifecycle.calls == 1  # only the fixed reviewer, eagerly; per-task sessions start on use
        assert set(sb.backends) == {"reviewer"} and set(sb.per_task) == {"analyst"}
        assert isinstance(sb.per_task["analyst"], PerTaskSandbox)
        (middleware,) = sb.middleware
        assert isinstance(middleware, TaskKeyMiddleware)
        assert set(sb.routes()) == {"/subagents/reviewer/"}  # no parent mount of per-task sessions


def test_bad_configuration_is_refused() -> None:
    lifecycle = Lifecycle()
    with pytest.raises(ValueError, match="unknown"):
        provision_subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                                     per_task={"nobody": PerTask()})
    with pytest.raises(ValueError, match="when_done"):
        PerTask(when_done="keep")
    with pytest.raises(ValueError, match="max_concurrent"):
        PerTask(max_concurrent=0)


def test_outside_a_task_the_router_raises() -> None:
    router = PerTaskSandbox("analyst")
    with pytest.raises(SandboxError, match="'analyst' was used outside a task"):
        router.execute("true")
    with pytest.raises(SandboxError):
        router.current()
    assert router.id == "per-task:analyst" and router.enable_capture_offload is False


# ---------------------------------------------------------------------------
# One session per task call
# ---------------------------------------------------------------------------


def test_each_task_call_gets_its_own_session_deleted_when_it_returns() -> None:
    lifecycle = Lifecycle()
    model = ScriptedChatModel()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True,
                            per_task={"analyst": PerTask()}) as sb:
        agent = _agent(model, sb)
        model.script(
            tool_call("task", subagent_type="analyst", description="alpha first"),
            *[worker] * 4,
            tool_call("task", subagent_type="analyst", description="beta second"),
            *[worker] * 4,
            say("done"),
        )
        result = _run(agent)
        assert sb.started_per_task() == {"analyst": {}}

    first, second = _task_answers(result["messages"])
    assert "after: [alpha.md]" in first
    assert "before: []" in second and "after: [beta.md]" in second  # the second task never saw alpha's file
    assert len(lifecycle.created) == 2 and lifecycle.deletes == lifecycle.created
    assert all(sid.startswith(f"{THREAD}-analyst-call_") for sid in lifecycle.created)
    assert lifecycle._local_states == {}  # nothing left, and the reviewer was never started
    assert lifecycle.pauses == 0


def test_parallel_tasks_run_in_isolated_sessions_forked_from_the_template() -> None:
    lifecycle = Lifecycle()
    template = lifecycle.build_template("analyst", seed_files=[("/work/README", b"from the template\n")])
    barrier = threading.Barrier(3, timeout=5)

    @lc_tool
    def rendezvous() -> str:
        """Wait for the other analysts."""
        barrier.wait()
        return "ok"

    def step(messages: list[BaseMessage]) -> Any:
        if not _since_task(messages):
            return tool_call("rendezvous")
        return worker([m for m in messages if not (m.type == "tool" and m.name == "rendezvous")])

    model = ScriptedChatModel()

    async def main() -> dict[str, Any]:
        async with asubagent_sandboxes(lifecycle, _specs(rendezvous), thread_id=THREAD, default_model=model,
                                       templates={"analyst": template}, on_demand=True,
                                       per_task={"analyst": PerTask()}) as sb:
            agent = _agent(model, sb)
            model.script(
                tool_calls(*[("task", {"subagent_type": "analyst", "description": f"{w} repo"})
                             for w in ("red", "green", "blue")]),
                *[step] * 15,
                say("done"),
            )
            return await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})

    result = asyncio.run(main())
    answers = _task_answers(result["messages"])
    assert len(answers) == 3
    for word in ("red", "green", "blue"):
        (mine,) = [a for a in answers if f"{word}.md" in a]
        after = set(mine.split("after: [")[1].rstrip("]").split())
        assert "before: [README]" in mine and after == {"README", f"{word}.md"}, mine
    assert len(lifecycle.created) == 3 and sorted(lifecycle.deletes) == sorted(lifecycle.created)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_max_concurrent_bounds_running_tasks(mode: str) -> None:
    lifecycle = Lifecycle(delay=0.15)
    model = ScriptedChatModel()

    def step(messages: list[BaseMessage]) -> Any:
        return say("ok") if _since_task(messages) else tool_call("execute", command="true")

    sb = provision_subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True,
                                      per_task={"analyst": PerTask(max_concurrent=2)})
    model.script(
        tool_calls(*[("task", {"subagent_type": "analyst", "description": f"t{i}"}) for i in range(4)]),
        *[step] * 8,
        say("done"),
    )
    agent, prompt = _agent(model, sb), {"messages": [{"role": "user", "content": "go"}]}
    result = agent.invoke(prompt) if mode == "sync" else asyncio.run(agent.ainvoke(prompt))
    assert _task_answers(result["messages"]) == ["ok"] * 4
    assert lifecycle.peak == 2 and len(lifecycle.created) == 4 and len(lifecycle.deletes) == 4


# ---------------------------------------------------------------------------
# Keys, pause, resume
# ---------------------------------------------------------------------------


def _by_repo(call: Any) -> str:
    return call.description.split()[0]


def test_pause_keeps_a_keyed_session_for_a_later_turn() -> None:
    lifecycle = Lifecycle()
    store = GatewayStore(":memory:")
    model = ScriptedChatModel()
    kw = {"thread_id": THREAD, "default_model": model, "on_demand": True, "store": store,
          "per_task": {"analyst": PerTask(key=_by_repo, when_done="pause")}}
    sid_a = subagent_sandbox_id(THREAD, "analyst-alpha")

    with subagent_sandboxes(lifecycle, _specs(), **kw) as sb:
        model.script(tool_call("task", subagent_type="analyst", description="alpha look"), *[worker] * 4, say("ok"))
        _run(_agent(model, sb))
        assert lifecycle._local_states == {sid_a: MicroVMStatus.PAUSED}  # paused when the task returned
    assert lifecycle.pauses == 1  # teardown had nothing left to pause
    assert store.get_sandbox(sid_a).status == "paused"

    with subagent_sandboxes(lifecycle, _specs(), **kw) as sb:
        model.script(
            tool_calls(("task", {"subagent_type": "analyst", "description": "alpha again"}),
                       ("task", {"subagent_type": "analyst", "description": "beta new"})),
            *[worker] * 8,
            say("ok"),
        )
        result = _run(_agent(model, sb))

    again, new = sorted(_task_answers(result["messages"]), key=lambda a: "beta" in a)
    assert "before: [alpha.md]" in again  # resumed: the first turn's file is there
    assert "before: []" in new
    assert lifecycle.created == [sid_a, subagent_sandbox_id(THREAD, "analyst-beta")]
    assert set(lifecycle._local_states.values()) == {MicroVMStatus.PAUSED} and lifecycle.deletes == []


def test_the_same_key_twice_at_once_waits_for_the_first() -> None:
    lifecycle = Lifecycle()
    model = ScriptedChatModel()
    inside = 0
    peak = 0
    lock = threading.Lock()

    @lc_tool
    async def hold() -> str:
        """Hold the session a moment."""
        nonlocal inside, peak
        with lock:
            inside += 1
            peak = max(peak, inside)
        await asyncio.sleep(0.1)
        with lock:
            inside -= 1
        return "held"

    def step(messages: list[BaseMessage]) -> Any:
        return say("ok") if _since_task(messages) else tool_call("hold")

    async def main() -> None:
        async with asubagent_sandboxes(lifecycle, _specs(hold), thread_id=THREAD, default_model=model,
                                       on_demand=True,
                                       per_task={"analyst": PerTask(key=_by_repo, when_done="pause")}) as sb:
            model.script(
                tool_calls(("task", {"subagent_type": "analyst", "description": "alpha one"}),
                           ("task", {"subagent_type": "analyst", "description": "alpha two"})),
                *[step] * 4,
                say("done"),
            )
            await _agent(model, sb).ainvoke({"messages": [{"role": "user", "content": "go"}]})

    asyncio.run(main())
    assert peak == 1


def test_an_interrupted_task_is_paused_and_its_replay_finds_the_same_session() -> None:
    lifecycle = Lifecycle()
    model = ScriptedChatModel()

    @lc_tool
    def approve() -> str:
        """Ask a human."""
        return str(interrupt("approve?"))

    with subagent_sandboxes(lifecycle, _specs(approve), thread_id=THREAD, default_model=model, on_demand=True,
                            per_task={"analyst": PerTask()}) as sb:
        agent = _agent(model, sb, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "interrupt"}}
        model.script(
            tool_call("task", subagent_type="analyst", description="alpha"),
            tool_call("write_file", file_path="/work/draft.md", content="draft"),
            tool_call("approve"),
        )
        first = _run(agent, config=config)
        assert first.get("__interrupt__")
        (sid,) = lifecycle.created
        assert lifecycle._local_states == {sid: MicroVMStatus.PAUSED} and lifecycle.deletes == []

        model.script(
            tool_call("execute", command="cat /work/draft.md"),
            lambda msgs: say(f"saw {msgs[-1].content}"),
            say("done"),
        )
        result = agent.invoke(Command(resume="yes"), config)

    assert "saw draft" in _task_answers(result["messages"])[0]
    assert lifecycle.created == [sid] and lifecycle.deletes == [sid]  # same session, deleted once it returned


# ---------------------------------------------------------------------------
# Refusals and failures
# ---------------------------------------------------------------------------


def test_max_sessions_refuses_with_a_tool_message_counting_paused_ones() -> None:
    lifecycle = Lifecycle()
    store = GatewayStore(":memory:")
    model = ScriptedChatModel()
    kw = {"thread_id": THREAD, "default_model": model, "on_demand": True, "store": store,
          "per_task": {"analyst": PerTask(key=_by_repo, when_done="pause", max_sessions=1)}}

    with subagent_sandboxes(lifecycle, _specs(), **kw) as sb:
        model.script(tool_call("task", subagent_type="analyst", description="alpha"), *[worker] * 4, say("ok"))
        _run(_agent(model, sb))

    with subagent_sandboxes(lifecycle, _specs(), **kw) as sb:
        model.script(
            tool_call("task", subagent_type="analyst", description="beta"),  # a second session: refused
            tool_call("task", subagent_type="analyst", description="alpha"),  # the paused one: allowed
            *[worker] * 4,
            say("ok"),
        )
        result = _run(_agent(model, sb))

    refused, allowed = [m for m in result["messages"] if m.type == "tool" and m.name == "task"]
    assert refused.status == "error" and "holds 1 sandboxes" in str(refused.content) and "limit is 1" in str(
        refused.content)
    assert "before: [alpha.md]" in str(allowed.content)
    assert lifecycle.created == [subagent_sandbox_id(THREAD, "analyst-alpha")]


def test_a_bad_key_refuses_the_call() -> None:
    lifecycle = Lifecycle()
    model = ScriptedChatModel()

    def key(call: Any) -> str:
        if call.description == "boom":
            raise ValueError("no repository named")
        return ""

    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True,
                            per_task={"analyst": PerTask(key=key)}) as sb:
        model.script(
            tool_calls(("task", {"subagent_type": "analyst", "description": "boom"}),
                       ("task", {"subagent_type": "analyst", "description": "empty"})),
            say("ok"),
        )
        result = _run(_agent(model, sb))
    answers = sorted(_task_answers(result["messages"]))
    assert "could not be computed: no repository named" in answers[0] and "key is empty" in answers[1]
    assert lifecycle.calls == 0


def test_a_session_that_cannot_start_is_reported_to_the_subagent() -> None:
    lifecycle = Lifecycle()
    model = ScriptedChatModel()

    class Failing(Lifecycle):
        def get_or_create(self, *, sandbox_id: str, **kwargs: Any) -> Any:
            if "-analyst-" in sandbox_id:
                raise MicroVMForbiddenError(403, "quota reached")
            return super().get_or_create(sandbox_id=sandbox_id, **kwargs)

    lifecycle = Failing()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True,
                            per_task={"analyst": PerTask()}) as sb:
        model.script(
            tool_call("task", subagent_type="analyst", description="alpha"),
            tool_call("execute", command="true"),
            lambda msgs: say(f"failed: {msgs[-1].content}"),
            say("ok"),
        )
        result = _run(_agent(model, sb))
    assert "could not be started" in _task_answers(result["messages"])[0]
    assert "quota reached" in _task_answers(result["messages"])[0]


# ---------------------------------------------------------------------------
# Custom tools, teardown, store
# ---------------------------------------------------------------------------


def test_custom_tools_reach_the_running_tasks_session_through_the_router() -> None:
    lifecycle = Lifecycle()
    router = PerTaskSandbox("analyst")  # built before the specs, like any backend a tool closes over

    @lc_tool
    def load(file: str) -> str:
        """Load a file into this task's session."""
        router.write(f"/work/{file}.txt", router.key)
        return f"loaded into {router.current().id}"

    model = ScriptedChatModel()
    with subagent_sandboxes(lifecycle, _specs(load), thread_id=THREAD, default_model=model, on_demand=True,
                            per_task={"analyst": PerTask(key=_by_repo, when_done="pause")}) as sb:
        model.script(
            tool_call("task", subagent_type="analyst", description="alpha"),
            tool_call("load", file="x"),
            tool_call("execute", command="cat /work/x.txt"),
            lambda msgs: say(_out(msgs[-1])),
            say("ok"),
        )
        result = _run(_agent(model, sb))
    assert _task_answers(result["messages"])[0].startswith("alpha")
    assert lifecycle.created == [subagent_sandbox_id(THREAD, "analyst-alpha")]


def test_teardown_pauses_a_per_task_session_still_live() -> None:
    lifecycle = Lifecycle()
    sb = provision_subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                                      on_demand=True, per_task={"analyst": PerTask()})
    pool = sb._pools["analyst"]
    sandbox = pool.admit("k1")  # as the middleware does when a task starts
    sandbox.execute("true")
    pool.admit("k2")  # admitted but never used: nothing to pause
    assert set(sb.started_per_task()["analyst"]) == {subagent_sandbox_id(THREAD, "analyst-k1")}
    from langchain_digitalocean_microvm.subagents import (
        pause_subagent_sandboxes,
    )

    assert pause_subagent_sandboxes(lifecycle, sb) == []
    assert lifecycle.pauses == 1 and lifecycle._local_states == {
        subagent_sandbox_id(THREAD, "analyst-k1"): MicroVMStatus.PAUSED}


def test_store_rows_and_tags() -> None:
    lifecycle = Lifecycle()
    store = GatewayStore(":memory:")
    model = ScriptedChatModel()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True,
                            store=store, per_task={"analyst": PerTask(key=_by_repo, when_done="pause")}) as sb:
        model.script(tool_call("task", subagent_type="analyst", description="alpha"), *[worker] * 4,
                     tool_call("task", subagent_type="analyst", description="beta"), *[worker] * 4, say("ok"))
        _run(_agent(model, sb))
    rows = {r.sandbox_id: r for r in store.list_sandboxes()}
    for repo in ("alpha", "beta"):
        row = rows[subagent_sandbox_id(THREAD, f"analyst-{repo}")]
        assert (row.thread_id, row.subagent, row.status) == (THREAD, "analyst", "paused")
        assert lifecycle.tags[row.sandbox_id] == (SUBAGENT_SANDBOX_TAG, PER_TASK_SANDBOX_TAG)


def test_deleted_sessions_are_retired_in_a_separate_store() -> None:
    lifecycle = Lifecycle()
    store = GatewayStore(":memory:")
    model = ScriptedChatModel()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True,
                            store=store, per_task={"analyst": PerTask()}) as sb:
        model.script(tool_call("task", subagent_type="analyst", description="alpha"), *[worker] * 4, say("ok"))
        _run(_agent(model, sb))
    (row,) = [r for r in store.list_sandboxes() if r.subagent == "analyst"]
    assert (row.status, row.retired_reason) == ("destroyed", "deleted")


def test_contract_the_task_tool_call_shape_the_middleware_reads() -> None:
    """TaskKeyMiddleware routes on the tool name `task` and its `subagent_type` / `description` arguments."""
    from deepagents.middleware.subagents import SubAgentMiddleware

    spec = {"name": "a", "description": "d", "system_prompt": "s", "model": ScriptedChatModel(), "tools": []}
    (task,) = SubAgentMiddleware(backend=StateBackend(), subagents=[spec]).tools
    assert task.name == "task" and {"subagent_type", "description"} <= set(task.args)
