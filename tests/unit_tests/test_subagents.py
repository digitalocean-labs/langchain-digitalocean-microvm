"""Subagent sandboxes: one MicroVMSandbox per sync subagent.

- Helper: `per_subagent_specs` gives each subagent's FilesystemMiddleware,
  summarization and skills middleware its own backend and permissions, checked
  on the processed specs deepagents hands to SubAgentMiddleware, plus a
  contract test on the middleware names the replacement relies on.
- Provisioning: provisioning per turn on `FakeMicroVMClient` (create/pause/resume
  counters), in parallel on a bounded pool, without blocking the event loop,
  with cleanup on partial failure, activity rows in the store, and the
  sweeper expiring sandboxes.
- Supervisor: a real deepagents supervisor driven by a scripted model: a subagent
  writes in its sandbox, the parent reads it at /subagents/<name>/..., and
  `execute` stays on the parent's backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
import time
from datetime import UTC, datetime
from typing import Any

import deepagents
import pytest
from deepagents import FilesystemPermission, create_deep_agent
from deepagents import graph as deepagents_graph
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from fake_client import FakeMicroVMClient
from scripted_model import ScriptedChatModel, last_tool_message, say, tool_call

from langchain_digitalocean_microvm.agent_host.store import GatewayStore
from langchain_digitalocean_microvm.agent_host.sweeper import apply_sweep, plan_sweep
from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import (
    MANAGED_TAG,
    SUBAGENT_SANDBOX_TAG,
    id_tag,
)
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.provider import MicroVMLifecycle
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.subagents import (
    OnDemandSandbox,
    SubagentSandboxes,
    asubagent_sandboxes,
    pause_subagent_sandboxes,
    per_subagent_specs,
    provision_subagent_sandboxes,
    subagent_sandbox_id,
    subagent_sandboxes,
)
from langchain_digitalocean_microvm.transport.local import LocalTransport

IMAGE = "registry.digitalocean.com/team/sandbox:1"
THREAD = "thread-1"


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    """LocalTransport keys its directories by MicroVM id ("01fake0001"...), which repeat across tests."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _lifecycle(client: FakeMicroVMClient) -> MicroVMLifecycle:
    config = MicroVMConfig(transport="exec", api_token="test-token", image=IMAGE, ready_timeout=5.0)
    return MicroVMLifecycle(config, client=client, transport_factory=LocalTransport)  # type: ignore[arg-type]


def _specs(**extra: Any) -> list[dict[str, Any]]:
    return [
        {"name": "analyst", "description": "Analyses data.", "system_prompt": "You analyse.", **extra},
        {"name": "reviewer", "description": "Reviews work.", "system_prompt": "You review."},
    ]


class _RecordingSubAgentMiddleware(SubAgentMiddleware):
    """Captures the processed specs (with their final middleware stacks) create_deep_agent builds."""

    seen: list[list[Any]] = []

    def __init__(self, *, subagents, **kwargs) -> None:  # type: ignore[no-untyped-def]
        type(self).seen.append(list(subagents))
        super().__init__(subagents=subagents, **kwargs)


@pytest.fixture
def processed_specs(monkeypatch):
    _RecordingSubAgentMiddleware.seen = []
    monkeypatch.setattr(deepagents_graph, "SubAgentMiddleware", _RecordingSubAgentMiddleware)

    def get() -> dict[str, Any]:
        (specs,) = _RecordingSubAgentMiddleware.seen
        return {spec["name"]: spec for spec in specs}

    return get


def _only(middleware: list[Any], cls: type) -> Any:
    found = [m for m in middleware if isinstance(m, cls)]
    assert len(found) == 1, f"expected one {cls.__name__}, got {found}"
    return found[0]


def _summarization(middleware: list[Any]) -> Any:
    found = [m for m in middleware if m.name == "SummarizationMiddleware"]
    assert len(found) == 1
    return found[0]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_contract_deepagents_names_the_replacement_relies_on(processed_specs) -> None:
    """If deepagents renames these, replacement silently stops working: fail here instead."""
    # Supported range: pyproject.toml (deepagents>=0.7.0,<0.8.0). Widen both together.
    assert deepagents.__version__.startswith("0.7."), deepagents.__version__
    model = ScriptedChatModel()
    assert FilesystemMiddleware().name == "FilesystemMiddleware"
    assert create_summarization_middleware(model, StateBackend()).name == "SummarizationMiddleware"
    assert SkillsMiddleware(backend=StateBackend(), sources=["/skills/"]).name == "SkillsMiddleware"

    create_deep_agent(model=model, subagents=[{**_specs()[0], "skills": ["/skills/"]}])
    base = [m.name for m in processed_specs()["analyst"]["middleware"]]
    for name in ("FilesystemMiddleware", "SummarizationMiddleware", "SkillsMiddleware"):
        assert base.count(name) == 1, base

    # The private hooks the helper calls exist with the shapes it expects.
    from deepagents._models import resolve_model
    from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model

    assert resolve_model(model) is model
    assert _harness_profile_for_model(model, None).tool_description_overrides is not None
    from langchain_digitalocean_microvm.subagents import _PERMISSIONS_KWARG

    rules = [FilesystemPermission(["read"], ["/x"])]
    fs = FilesystemMiddleware(backend=StateBackend(), **{_PERMISSIONS_KWARG: rules})
    assert fs.backend is not None and len(fs._permissions) == 1


def test_each_subagent_gets_its_own_backend_and_permissions(processed_specs) -> None:
    model = ScriptedChatModel()
    parent_backend, b1, b2 = StateBackend(), StateBackend(), StateBackend()
    parent_rules = [FilesystemPermission(["write"], ["/**"], mode="deny")]
    analyst_rules = [FilesystemPermission(["read", "write"], ["/secret/**"], mode="deny")]
    specs = _specs(permissions=analyst_rules)
    specs[1]["skills"] = ["/skills/"]

    rewritten = per_subagent_specs(
        specs, backends={"analyst": b1, "reviewer": b2}, default_model=model, parent_permissions=parent_rules,
    )
    create_deep_agent(model=model, subagents=rewritten, backend=parent_backend, permissions=parent_rules)
    seen = processed_specs()

    analyst, reviewer = seen["analyst"]["middleware"], seen["reviewer"]["middleware"]
    assert _only(analyst, FilesystemMiddleware).backend is b1
    assert _only(reviewer, FilesystemMiddleware).backend is b2
    assert _summarization(analyst)._backend is b1
    assert _summarization(reviewer)._backend is b2
    assert _only(reviewer, SkillsMiddleware)._backend is b2
    assert not [m for m in analyst if isinstance(m, SkillsMiddleware)]
    # Own rules win; a spec without rules inherits the parent's (as deepagents does).
    assert _only(analyst, FilesystemMiddleware)._permissions == analyst_rules
    assert _only(reviewer, FilesystemMiddleware)._permissions == parent_rules
    # Replaced in place: same position as deepagents' own FilesystemMiddleware (first).
    assert analyst[0].name == "FilesystemMiddleware"
    # The general-purpose subagent is not ours to move: it keeps the parent's backend.
    assert _only(seen["general-purpose"]["middleware"], FilesystemMiddleware).backend is parent_backend


def test_permissions_on_a_sandbox_are_refused_unless_the_spec_opts_out() -> None:
    """deepagents cannot enforce permission rules on a backend with `execute`; say so per subagent."""
    model = ScriptedChatModel()
    sandbox = MicroVMLifecycle(MicroVMConfig(transport="local")).get_or_create(sandbox_id="perm-check")
    rules = [FilesystemPermission(["write"], ["/etc/**"], mode="deny")]
    with pytest.raises(ValueError, match="Subagent 'analyst'.*permissions: \\[\\]"):
        per_subagent_specs(_specs(), backends={"analyst": sandbox}, default_model=model, parent_permissions=rules)
    (out, _) = per_subagent_specs(
        _specs(permissions=[]), backends={"analyst": sandbox}, default_model=model, parent_permissions=rules,
    )
    assert _only(out["middleware"], FilesystemMiddleware)._permissions == []


def test_specs_without_a_backend_and_other_kinds_pass_through() -> None:
    model = ScriptedChatModel()
    compiled = {"name": "compiled", "description": "d", "runnable": object()}
    remote = {"name": "remote", "description": "d", "graph_id": "g"}
    plain = _specs()[1]
    out = per_subagent_specs(
        [plain, compiled, remote], backends={"reviewer": None, "compiled": StateBackend()}, default_model=model,  # type: ignore[dict-item]
    )
    assert out[0] is plain and out[1] is compiled and out[2] is remote


def test_spec_middleware_comes_after_and_wins_on_a_name_clash() -> None:
    model = ScriptedChatModel()
    mine = FilesystemMiddleware(backend=StateBackend())
    spec = {**_specs()[0], "middleware": [mine]}
    (out,) = per_subagent_specs([spec], backends={"analyst": StateBackend()}, default_model=model)
    assert out["middleware"][-1] is mine
    assert spec["middleware"] == [mine]  # input not mutated


# ---------------------------------------------------------------------------
# Provisioning, teardown, store, sweeper
# ---------------------------------------------------------------------------


def test_per_turn_counters_create_once_then_resume_and_pause_each_turn() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    model = ScriptedChatModel()
    seeds = {"analyst": [("/skills/a/SKILL.md", b"---\nname: a\n---\n")]}

    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, seed_files=seeds) as sb:
        assert set(sb.backends) == {"analyst", "reviewer"}
        assert (client.create_calls, client.resume_calls, client.pause_calls) == (2, 0, 0)
        assert sb.backends["analyst"].execute("cat /skills/a/SKILL.md").output.startswith("---")
        assert sb.backends["reviewer"].execute("ls /skills").exit_code != 0  # seeds are per subagent
        assert all(isinstance(b, MicroVMSandbox) for b in sb.backends.values())
        assert len(sb.subagents) == 2
    assert (client.create_calls, client.resume_calls, client.pause_calls) == (2, 0, 2)
    assert {s.status for s in client.live()} == {MicroVMStatus.PAUSED}

    for turn in (2, 3):
        with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, seed_files=seeds) as sb:
            assert {s.status for s in client.live()} == {MicroVMStatus.RUNNING}
            assert sb.backends["analyst"].execute("cat /skills/a/SKILL.md").exit_code == 0
        assert (client.create_calls, client.resume_calls, client.pause_calls) == (2, 2 * (turn - 1), 2 * turn)

    names = {s.name: s for s in client.live()}
    assert len(names) == 2
    for state in names.values():
        assert MANAGED_TAG in state.tags and SUBAGENT_SANDBOX_TAG in state.tags
    assert {t for s in names.values() for t in s.tags if t.startswith("microvm-sandbox-id:")} == {
        id_tag(subagent_sandbox_id(THREAD, "analyst")), id_tag(subagent_sandbox_id(THREAD, "reviewer")),
    }


def test_threads_get_separate_sandboxes() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    with subagent_sandboxes(lifecycle, _specs()[:1], thread_id="t-a", default_model=ScriptedChatModel()) as a:
        a.backends["analyst"].write("/note.md", "from a")
    with subagent_sandboxes(lifecycle, _specs()[:1], thread_id="t-b", default_model=ScriptedChatModel()) as b:
        assert b.backends["analyst"].execute("cat /note.md").exit_code != 0
    assert client.create_calls == 2


def test_provisioning_runs_in_parallel_on_a_bounded_pool() -> None:
    client = FakeMicroVMClient(create_delay=0.3)
    lifecycle = _lifecycle(client)
    started = time.monotonic()
    sb = provision_subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel())
    assert time.monotonic() - started < 0.55
    assert client.peak_concurrent_creates == 2
    pause_subagent_sandboxes(lifecycle, sb)

    serial = FakeMicroVMClient(create_delay=0.05)
    sb = provision_subagent_sandboxes(
        _lifecycle(serial), _specs(), thread_id=THREAD, default_model=ScriptedChatModel(), max_workers=1,
    )
    assert serial.peak_concurrent_creates == 1 and serial.create_calls == 2


def test_async_provisioning_does_not_block_the_event_loop() -> None:
    client = FakeMicroVMClient(create_delay=0.4)
    lifecycle = _lifecycle(client)

    async def main() -> tuple[int, SubagentSandboxes]:
        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        async with asubagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel()) as sb:
            pass
        stop.set()
        await task
        return ticks, sb

    ticks, sb = asyncio.run(main())
    assert ticks >= 10  # the loop kept running during the ~0.4 s of creates
    assert client.peak_concurrent_creates == 2 and client.pause_calls == 2
    assert threading.current_thread() is threading.main_thread()


def test_a_failed_create_pauses_the_others_and_raises() -> None:
    from langchain_digitalocean_microvm.client import microvm_name
    from langchain_digitalocean_microvm.exceptions import MicroVMForbiddenError

    client = FakeMicroVMClient()
    name = microvm_name("microvm-sandbox", subagent_sandbox_id(THREAD, "reviewer"))
    client.fail_create_for[name] = MicroVMForbiddenError(403, "team MicroVM limit reached")
    lifecycle = _lifecycle(client)
    store = GatewayStore(":memory:")

    with pytest.raises(MicroVMForbiddenError), subagent_sandboxes(
        lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(), store=store,
    ):
        pytest.fail("must not enter the body")
    assert [s.status for s in client.live()] == [MicroVMStatus.PAUSED]
    (row,) = store.list_sandboxes()
    assert (row.subagent, row.status) == ("analyst", "paused")


def test_teardown_errors_are_logged_not_raised(caplog) -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel()) as sb:
        client.delete(sb.backends["reviewer"].id)  # gone behind our back; pause will fail
    assert "Could not pause the sandbox of subagent 'reviewer'" in caplog.text
    assert client.get(sb.backends["analyst"].id).status == MicroVMStatus.PAUSED


def test_store_records_activity_per_turn() -> None:
    client = FakeMicroVMClient()
    store = GatewayStore(":memory:")
    now = [1000.0]
    kw = {"thread_id": THREAD, "default_model": ScriptedChatModel(), "store": store, "clock": lambda: now[0]}
    with subagent_sandboxes(_lifecycle(client), _specs(), **kw) as sb:
        rows = {r.subagent: r for r in store.list_sandboxes()}
        assert {r.status for r in rows.values()} == {"active"}
        assert rows["analyst"].microvm_id == sb.backends["analyst"].id
        assert rows["analyst"].thread_id == THREAD and rows["analyst"].last_active_at == 1000.0
        now[0] = 1060.0
    rows = {r.subagent: r for r in store.list_sandboxes()}
    assert {(r.status, r.last_active_at) for r in rows.values()} == {("paused", 1060.0)}


def test_fake_transport_lifecycle_works_too() -> None:
    lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local"))
    with subagent_sandboxes(lifecycle, _specs(), thread_id="fake-thread", default_model=ScriptedChatModel()) as sb:
        sb.backends["analyst"].write("/outputs/a.md", "hi")
        assert sb.mount(StateBackend()).read("/subagents/analyst/outputs/a.md").file_data["content"] == "hi"
    assert set(lifecycle._local_states.values()) == {MicroVMStatus.PAUSED}


def test_sweeper_expires_subagent_sandboxes_by_store_activity() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    store = GatewayStore(":memory:")
    created = datetime(2026, 10, 1, 22, 46, 36, tzinfo=UTC).timestamp()  # FakeMicroVMClient's created_at
    day = 24 * 3600.0

    stale = provision_subagent_sandboxes(lifecycle, _specs()[:1], thread_id="old", default_model=ScriptedChatModel(),
                                         store=store, clock=lambda: created)
    fresh = provision_subagent_sandboxes(lifecycle, _specs()[:1], thread_id="new", default_model=ScriptedChatModel(),
                                         store=store, clock=lambda: created + 2 * day)
    orphan = provision_subagent_sandboxes(lifecycle, _specs()[:1], thread_id="lost", default_model=ScriptedChatModel())
    plain = lifecycle.get_or_create(sandbox_id="m2-demo")  # not a subagent sandbox: never swept
    gone = provision_subagent_sandboxes(lifecycle, _specs()[:1], thread_id="gone", default_model=ScriptedChatModel(),
                                        store=store, clock=lambda: created)
    client.delete(gone.backends["analyst"].id)

    plan = plan_sweep(client, store=store, now=created + 2 * day + 60, sandbox_ttl=day, orphan_ttl=day)
    deleted = {item.id for item in plan.delete_microvms}
    assert deleted == {stale.backends["analyst"].id, orphan.backends["analyst"].id}
    assert all(item.kind == "sandbox" for item in plan.delete_microvms)
    kept = {item.id: item.reason for item in plan.kept}
    assert "subagent 'analyst' of thread new" in kept[fresh.backends["analyst"].id]
    assert plain.id not in deleted and plain.id not in kept
    assert [(i.kind, i.id) for i in plan.retire_rows] == [("sandbox_row", subagent_sandbox_id("gone", "analyst"))]

    result = apply_sweep(plan, client, store=store)
    assert not result.errors and len(result.deleted_microvms) == 2
    rows = {r.thread_id: r for r in store.list_sandboxes()}
    assert (rows["old"].status, rows["old"].retired_reason) == ("destroyed", "gc")
    assert (rows["gone"].status, rows["gone"].retired_reason) == ("destroyed", "vanished")
    assert rows["new"].status == "active"
    assert plan_sweep(client, store=store, now=created + 2 * day + 60, sandbox_ttl=day, orphan_ttl=day).empty

    # Using the thread again recreates its sandbox and revives the row.
    again = provision_subagent_sandboxes(lifecycle, _specs()[:1], thread_id="old", default_model=ScriptedChatModel(),
                                         store=store)
    assert again.backends["analyst"].id != stale.backends["analyst"].id
    row = store.get_sandbox(subagent_sandbox_id("old", "analyst"))
    assert row is not None and (row.status, row.retired_reason) == ("active", None)


# ---------------------------------------------------------------------------
# CompositeBackend mounts, through a real deepagents supervisor
# ---------------------------------------------------------------------------


def _supervisor(model: ScriptedChatModel, sb: SubagentSandboxes, parent_backend: Any) -> Any:
    return create_deep_agent(model=model, subagents=list(sb.subagents), backend=sb.mount(parent_backend))


def test_parent_reads_a_subagent_file_through_the_mount_and_execute_stays_on_the_parent() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    parent = lifecycle.get_or_create(sandbox_id="parent")  # a sandbox of its own, so the parent can execute
    parent.write("/parent-only.txt", "parent file")
    model = ScriptedChatModel()

    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model) as sb:
        agent = _supervisor(model, sb, parent)
        model.script(
            tool_call("task", subagent_type="analyst", description="Write the analysis to /outputs/a.md"),
            # the analyst's own loop, inside its sandbox
            tool_call("write_file", file_path="/outputs/a.md", content="# Analysis\n42\n"),
            tool_call("execute", command="cat /outputs/a.md && ls /parent-only.txt || echo no-parent-file"),
            say("Wrote /outputs/a.md"),
            # back in the parent
            tool_call("read_file", file_path="/subagents/analyst/outputs/a.md"),
            tool_call("execute", command="cat /parent-only.txt; cat /outputs/a.md || echo not-on-parent"),
            say("done"),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "analyse"}]})

    tool_outputs = [m for m in result["messages"] if m.type == "tool"]
    read, execute = tool_outputs[-2], tool_outputs[-1]
    assert read.name == "read_file" and "42" in str(read.content)
    assert execute.name == "execute"
    assert "parent file" in str(execute.content) and "not-on-parent" in str(execute.content)
    # the subagent's own execute ran in its sandbox, which has no parent file
    sub_prompts = [p for p in model.prompts if any("You analyse." in str(m.content) for m in p)]
    sub_exec = last_tool_message(sub_prompts[-1])
    assert "42" in str(sub_exec.content) and "no-parent-file" in str(sub_exec.content)
    # the file lives only in the analyst's sandbox
    assert sb.backends["analyst"].execute("cat /outputs/a.md").output.startswith("# Analysis")
    assert parent.execute("ls /outputs").exit_code != 0
    assert client.pause_calls == 2


def test_parent_mount_routes_and_writes_go_to_the_subagent() -> None:
    lifecycle = _lifecycle(FakeMicroVMClient())
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel()) as sb:
        assert set(sb.routes()) == {"/subagents/analyst/", "/subagents/reviewer/"}
        assert set(sb.routes("inbox")) == {"/inbox/analyst/", "/inbox/reviewer/"}
        mounted = sb.mount(StateBackend())
        assert isinstance(mounted, CompositeBackend)
        mounted.write("/subagents/reviewer/inputs/brief.md", "review this")
        assert sb.backends["reviewer"].execute("cat /inputs/brief.md").output == "review this"


def test_graph_factory_pattern_provisions_and_pauses_each_run() -> None:
    """The make_graph(config, runtime) shape the lab uses, run twice on one thread."""
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    model = ScriptedChatModel()

    @contextlib.asynccontextmanager
    async def make_graph(config: dict[str, Any]):  # type: ignore[no-untyped-def]
        thread_id = config["configurable"]["thread_id"]
        async with asubagent_sandboxes(lifecycle, _specs(), thread_id=thread_id, default_model=model) as sb:
            yield _supervisor(model, sb, StateBackend())

    async def turn(text: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": "factory-thread"}}
        async with make_graph(config) as graph:
            return await graph.ainvoke({"messages": [{"role": "user", "content": text}]}, config)

    model.script(
        tool_call("task", subagent_type="reviewer", description="save a note"),
        tool_call("write_file", file_path="/notes/n.md", content="kept across turns"),
        say("saved"),
        say("ok"),
    )
    asyncio.run(turn("one"))
    assert (client.create_calls, client.resume_calls, client.pause_calls) == (2, 0, 2)

    model.script(
        tool_call("read_file", file_path="/subagents/reviewer/notes/n.md"),
        say("read it"),
    )
    result = asyncio.run(turn("two"))
    assert "kept across turns" in str(last_tool_message(result["messages"]).content)
    assert (client.create_calls, client.resume_calls, client.pause_calls) == (2, 2, 4)


def test_size_and_tags_on_create() -> None:
    client = FakeMicroVMClient()
    with subagent_sandboxes(_lifecycle(client), _specs()[:1], thread_id=THREAD, default_model=ScriptedChatModel()):
        pass
    (payload,) = client.created
    assert payload["oci_ref"] == IMAGE and payload["size"] == MicroVMSize.parse("1vcpu-2gb")
    assert SUBAGENT_SANDBOX_TAG in payload["tags"]


# ---------------------------------------------------------------------------
# templates and on-demand sandboxes
# ---------------------------------------------------------------------------


def test_templates_clone_each_subagent_from_its_checkpoint_and_seeds_go_on_top() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    template = lifecycle.build_template("analyst", seed_files=[("/data/ref.csv", b"ref\n")])
    seeds = {"analyst": [("/inputs/brief.md", b"brief\n")]}

    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                            templates={"analyst": template}, seed_files=seeds) as sb:
        assert sb.backends["analyst"].execute("cat /inputs/brief.md").output == "brief\n"
    by_name = {p["name"]: p for p in client.created}
    analyst = by_name[client.get(sb.backends["analyst"].id).name]
    reviewer = by_name[client.get(sb.backends["reviewer"].id).name]
    assert analyst["checkpoint_id"] == template.checkpoint_id and analyst["oci_ref"] is None
    assert SUBAGENT_SANDBOX_TAG in analyst["tags"]
    assert reviewer["checkpoint_id"] is None and reviewer["oci_ref"] == IMAGE

    with pytest.raises(ValueError, match="unknown subagents"):
        provision_subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                                     templates={"nobody": template})


def test_on_demand_creates_only_what_is_used_and_pauses_only_that() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    kw = {"thread_id": THREAD, "default_model": ScriptedChatModel(), "on_demand": True}

    with subagent_sandboxes(lifecycle, _specs(), **kw) as sb:
        assert client.create_calls == 0 and sb.started() == {}
        assert all(isinstance(b, OnDemandSandbox) for b in sb.backends.values())
        assert sb.backends["analyst"].id == subagent_sandbox_id(THREAD, "analyst")  # no MicroVM yet
        sb.backends["analyst"].write("/outputs/a.md", "hi")
        assert client.create_calls == 1 and set(sb.started()) == {"analyst"}
        assert sb.backends["analyst"].id == sb.started()["analyst"].id
    assert client.pause_calls == 1

    with subagent_sandboxes(lifecycle, _specs(), **kw) as sb:  # next turn: only a used sandbox resumes
        assert client.resume_calls == 0
        assert sb.mount(StateBackend()).read("/subagents/analyst/outputs/a.md").file_data["content"] == "hi"
        assert sb.backends["reviewer"].started is None
    assert (client.create_calls, client.resume_calls, client.pause_calls) == (1, 1, 2)


def test_on_demand_first_uses_in_parallel_start_in_parallel_once_each() -> None:
    client = FakeMicroVMClient(create_delay=0.3)
    lifecycle = _lifecycle(client)
    sb = provision_subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                                      on_demand=True)

    async def main() -> None:
        await asyncio.gather(*(sb.backends[n].aexecute("true") for n in ("analyst", "reviewer", "analyst")))

    started = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - started < 0.55
    assert client.create_calls == 2 and client.peak_concurrent_creates == 2
    pause_subagent_sandboxes(lifecycle, sb)
    assert client.pause_calls == 2


def test_on_demand_failure_is_reported_to_the_agent_and_retried() -> None:
    from langchain_digitalocean_microvm.client import microvm_name
    from langchain_digitalocean_microvm.exceptions import MicroVMForbiddenError

    client = FakeMicroVMClient()
    name = microvm_name("microvm-sandbox", subagent_sandbox_id(THREAD, "analyst"))
    client.fail_create_for[name] = MicroVMForbiddenError(403, "no capacity")
    lifecycle = _lifecycle(client)
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=ScriptedChatModel(),
                            on_demand=True) as sb:
        failed = sb.backends["analyst"].execute("echo hi")
        assert failed.exit_code == 1 and "could not be started" in failed.output
        client.fail_create_for[name] = MicroVMForbiddenError(403, "no capacity")  # a failure is not cached
        assert "could not be started" in sb.backends["analyst"].upload_files([("/x", b"x")])[0].error
        assert sb.backends["analyst"].execute("echo hi").output.strip() == "hi"  # retried
    assert client.pause_calls == 1


def test_on_demand_supervisor_delegation_starts_the_subagent_sandbox() -> None:
    client = FakeMicroVMClient()
    lifecycle = _lifecycle(client)
    model = ScriptedChatModel()
    with subagent_sandboxes(lifecycle, _specs(), thread_id=THREAD, default_model=model, on_demand=True) as sb:
        agent = _supervisor(model, sb, StateBackend())
        model.script(
            tool_call("task", subagent_type="reviewer", description="note it"),
            tool_call("execute", command="echo reviewed > /review.md && cat /review.md"),
            say("noted"),
            say("done"),
        )
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        assert set(sb.started()) == {"reviewer"}
    assert (client.create_calls, client.pause_calls) == (1, 1)
