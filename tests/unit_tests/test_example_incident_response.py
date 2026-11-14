"""examples/09_incident_response in local mode: the world, the gateway's policy and approvals, the tools, the event log.

No model is called: the graph below is one node that calls the gateway tools the way the remediator does, so the
`allow` / `ask` / `deny` tiers, the approval interrupt, and what each leaves in the event log are checked end to end,
and the investigators' tools run in local MicroVMs cloned from the example's template.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

blockbuster = pytest.importorskip("blockbuster")
pytest.importorskip("rich")

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "09_incident_response"
_COMMON = _EXAMPLE.parent / "incident_common"  # the shared incident scenario
_MODULES = ("agent", "tools", "dashboard", "gateway", "run_script", "prepare", "approvals")


@pytest.fixture
def example(tmp_path, monkeypatch):
    """The example's `agent` module, imported in local mode with its own sandbox root and event log."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("DO_MICROVM_TRANSPORT", "local")
    monkeypatch.setenv("EXAMPLES_MODEL", "anthropic:claude-haiku-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("DO_MODEL_ACCESS_KEY", raising=False)
    monkeypatch.syspath_prepend(str(_EXAMPLE))
    # Other tests import modules of the same names (the agent-host template's `agent`, examples 07-11's `tools`).
    names = lambda: [n for n in sys.modules if n in _MODULES or n.startswith(("tools.", "gateway.", "incident_common"))]  # noqa: E731
    saved = {n: sys.modules.pop(n) for n in names()}
    module = importlib.import_module("agent")
    monkeypatch.setattr(sys.modules["incident_common.events"], "EVENTS", tmp_path / "events.jsonl")
    yield module
    module.gateway.close()
    for name in names():
        del sys.modules[name]
    sys.modules.update(saved)


def _events(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def _script(name: str):
    """A helper from sandbox/ (it runs in a MicroVM), loaded by path."""
    spec = importlib.util.spec_from_file_location(name, _COMMON / "sandbox" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _State(TypedDict, total=False):
    answer: str


def _drive(gateway, tool: str, arguments: dict, decisions: list[str]) -> tuple[str, list[dict]]:
    """Call `tool` through `action_invoke` in a graph; answer each interrupt from `decisions`. Returns the answer and the interrupts."""
    invoke = next(t for t in gateway.tools() if t.name == "action_invoke")
    graph = StateGraph(_State)
    graph.add_node("act", lambda state: {"answer": invoke.invoke({"tool": tool, "arguments": arguments, "rationale": "evidence"})})
    graph.add_edge(START, "act")
    graph.add_edge("act", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"t-{tool}-{json.dumps(arguments, sort_keys=True)}"}}
    out, stops = app.invoke({}, config), []
    while "__interrupt__" in out:
        stops.append(out["__interrupt__"][0].value)
        out = app.invoke(Command(resume=decisions.pop(0)), config)
    return out["answer"], stops


def test_the_world_fails_after_the_deploy_and_recovers_after_a_restore(example) -> None:
    world = sys.modules["incident_common.world"]
    now = [1_000_000.0]
    sim = world.SimulatedWorld(now=lambda: now[0])

    assert sim.health()["status"] == "failing" and sim.health()["version"] == world.BAD_SHA
    found = _script("series").analyse(sim.metrics())
    assert found["previous_version"] == world.GOOD_SHA and found["version_at_onset"] == world.BAD_SHA
    assert 0 <= found["minutes_from_version_change_to_onset"] <= 2 and found["peak_error_rate"] >= 0.4

    restore = sim.actions()["digitalocean_restore-droplet"].run
    assert restore({"ID": 1, "ImageID": 2}) == {"error": "unknown droplet or image"}
    assert "invalid" in restore({"ID": str(world.DROPLET_ID), "ImageID": world.SNAPSHOT_ID})["error"]  # ids are numbers
    result = restore({"ID": world.DROPLET_ID, "ImageID": world.SNAPSHOT_ID})
    assert result["action"]["status"] == "in-progress"
    assert sim.health()["status"] == "failing"  # the restore takes about 20 s
    now[0] += 30
    assert sim.health() == {"status": "healthy", "version": world.GOOD_SHA, "error_rate": 0.01, "droplet": "active"}


def test_the_timeline_names_the_commit_before_the_onset(example) -> None:
    world = sys.modules["incident_common.world"]
    sim = world.SimulatedWorld()
    onset = next(p["t"] for p in sim.metrics() if p["version"] == world.BAD_SHA and p["error_rate"] > 0.05)
    out = _script("timeline").build(sim.actions()["github_list_commits"].run({})["commits"], onset)
    assert out["last_before_onset"]["sha"] == world.BAD_SHA
    assert 0 <= out["last_before_onset"]["minutes_before_onset"] <= 8


def test_commits_made_in_the_same_second_keep_the_api_order(example) -> None:
    now = datetime.now(UTC)
    same, onset = now.strftime("%Y-%m-%dT%H:%M:%SZ"), (now + timedelta(minutes=18)).strftime("%Y-%m-%dT%H:%M:%SZ")
    newest_first = [{"sha": "a9f65fb0", "commit": {"message": "bad", "author": {"date": same}}},
                    {"sha": "8082d270", "commit": {"message": "good", "author": {"date": same}}}]
    out = _script("timeline").build(newest_first, onset)
    assert [c["sha"] for c in out["commits"]] == ["8082d27", "a9f65fb"]  # oldest first
    assert out["last_before_onset"]["sha"] == "a9f65fb"  # the newer commit is the suspect


def test_the_policy_has_one_ask_and_one_deny(example) -> None:
    policy = sys.modules["gateway"].POLICY
    assert list(policy.ask) == ["digitalocean_restore-droplet@3"] and list(policy.deny) == ["digitalocean_droplet-delete@3"]
    bare = {n.split("@")[0] for n in [*policy.allow, *policy.ask, *policy.deny]}
    assert bare <= set(example.service.actions())


def test_reads_run_the_restore_waits_for_a_human_and_delete_is_refused(example, tmp_path) -> None:
    world, gateway = sys.modules["incident_common.world"], example.gateway
    sim = example.service

    answer, stops = _drive(gateway, "github_list_commits", {"owner": world.OWNER, "repo": world.REPO}, [])
    assert stops == [] and "wait for the payment provider" in answer  # allow: ran, no question

    args = {"ID": world.DROPLET_ID, "ImageID": world.SNAPSHOT_ID}
    refused = {**args, "ImageID": world.SNAPSHOT_ID + 1}  # an approval is bound to its arguments
    answer, stops = _drive(gateway, "digitalocean_restore-droplet", refused, ["deny"])
    assert [s["type"] for s in stops] == ["action_approval"] and stops[0]["arguments"] == refused
    assert stops[0]["rationale"] == "evidence" and "denied" in answer.lower()
    assert sim.restored_at is None  # a denied approval ran nothing

    answer, stops = _drive(gateway, "digitalocean_restore-droplet", args, ["approve"])
    assert len(stops) == 1 and "in-progress" in answer and sim.restored_at is not None

    answer, stops = _drive(gateway, "digitalocean_droplet-delete", {"ID": world.DROPLET_ID}, [])
    assert stops == [] and answer.startswith("Error:") and not sim.deleted  # deny: refused, nobody is asked

    seen = [(e["event"], e["tool"], e.get("status") or e["decision"]) for e in _events(tmp_path) if e["event"] in ("gateway", "decision")]
    assert seen == [
        ("gateway", "github_list_commits", "allow"),
        ("gateway", "digitalocean_restore-droplet", "ask"),
        ("decision", "digitalocean_restore-droplet", "deny"),
        ("gateway", "digitalocean_restore-droplet", "ask"),
        ("decision", "digitalocean_restore-droplet", "approve"),
        ("gateway", "digitalocean_restore-droplet", "approved"),
        ("gateway", "digitalocean_droplet-delete", "deny"),
    ]


async def test_investigator_tools_do_not_block_the_event_loop(example) -> None:
    tools, world = sys.modules["tools"], sys.modules["incident_common.world"]
    uploads: dict[str, bytes] = {}
    commands: list[str] = []

    class _Sandbox:
        id = "incident-test-sandbox"

        def upload_files(self, files):
            uploads.update(files)
            return [SimpleNamespace(path=p, error=None) for p, _ in files]

        async def aexecute(self, command: str, timeout: int | None = None):
            commands.append(command)
            return SimpleNamespace(exit_code=0, output='{"onset": null}')

    fetch_metrics, find_onset = tools.metrics_tools(_Sandbox(), example.service, local=True)
    fetch_commits, _ = tools.changes_tools(_Sandbox(), example.gateway, local=True)
    with blockbuster.blockbuster_ctx():
        wrote = await fetch_metrics.ainvoke({"minutes": 30})
        found = await find_onset.ainvoke({})
        listed = await fetch_commits.ainvoke({"owner": world.OWNER, "repo": world.REPO})
    assert wrote.startswith("Wrote 31 points") and len(json.loads(uploads["/data/metrics.json"])) == 31
    assert found == '{"onset": null}' and f"{tools.root_for_host(_Sandbox.id)}/tools/series.py" in commands[0]
    assert "env -u MICROVM_LOCAL_ROOT" in commands[0]
    assert listed == "Wrote 3 commits to /data/commits.json." and len(json.loads(uploads["/data/commits.json"])) == 3

    # a bare list (rather than {"commits": [...]}) is written as it is
    commits = json.loads(uploads["/data/commits.json"])
    stub = SimpleNamespace(call=lambda tool, arguments: json.dumps(commits[:2]))
    uploads.clear()
    bare, _ = tools.changes_tools(_Sandbox(), stub, local=True)
    assert await bare.ainvoke({"owner": "o", "repo": "r"}) == "Wrote 2 commits to /data/commits.json."
    assert json.loads(uploads["/data/commits.json"]) == commits[:2]


def test_the_approval_shows_what_the_arguments_point_at_from_independent_reads(example) -> None:
    world = sys.modules["incident_common.world"]
    approvals = importlib.import_module("approvals")
    facts = lambda stop: approvals.verified_facts(stop, example.gateway)  # noqa: E731
    stop = {"tool": "digitalocean_restore-droplet", "arguments": {"ID": world.DROPLET_ID, "ImageID": world.SNAPSHOT_ID}}
    lines = facts(stop)
    assert lines[0].startswith(f"droplet    {world.DROPLET_NAME} (active)") and world.SNAPSHOT_NAME in lines[1]
    wrong = {"tool": stop["tool"], "arguments": {"ID": world.DROPLET_ID, "ImageID": 1}}
    assert "NOT FOUND" in facts(wrong)[1]
    assert facts({"tool": "github_list_commits", "arguments": {}}) == []
    assert approvals.ask_human({**stop, "type": "action_approval", "rationale": "r"}, "deny", lines) == "deny"


def test_the_postmortem_is_a_file(example, tmp_path, monkeypatch) -> None:
    tools = sys.modules["tools"]
    monkeypatch.setattr(tools, "HERE", tmp_path)
    save = next(t for t in tools.commander_tools() if t.name == "save_postmortem")

    async def run() -> str:
        with blockbuster.blockbuster_ctx():
            return await save.ainvoke({"title": "Checkout errors: sync mode", "body": "## Summary\nbad commit"})

    answer = asyncio.run(run())
    written = next((tmp_path / "outputs" / "postmortems").glob("*checkout-errors-sync-mode.md"))
    assert str(written) in answer and written.read_text().startswith("# Checkout errors: sync mode")


def test_seed_files_hold_the_helpers(example) -> None:
    assert {path for path, _ in example.seed_files()} == {"/tools/series.py", "/tools/timeline.py", "/tools/verdict.py"}
    assert [spec["name"] for spec in example.SUBAGENTS] == ["metrics", "changes", "infra", "verifier"]
    dashboard = importlib.import_module("dashboard")
    assert dashboard.ROLES == [spec["name"] for spec in example.SUBAGENTS] and set(dashboard.ROLE_NAMES) == set(dashboard.ROLES)


def test_a_refusal_is_reported_alone_without_another_way(example) -> None:
    # A refused action ends the answer: the commander and the remediator name it and the policy, and suggest no one else.
    for prompt in (example.SYSTEM_PROMPT, example.REMEDIATOR["system_prompt"]):
        assert "Action Gateway's policy refused it. Add nothing after it" in prompt
    assert "whatever you think of it" in example.SYSTEM_PROMPT  # a human's request goes to the remediator; the policy decides


def test_the_preflight_opens_a_session_so_a_stale_version_fails_before_a_run(example, monkeypatch) -> None:
    prepare = importlib.import_module("prepare")
    from langchain_digitalocean_microvm.integrations.action_gateway import (
        ActionGatewayError,
    )

    opened: list[list[str]] = []

    class Gateway:
        def __init__(self, policy, *, tools):
            self.tools = tools

        def __enter__(self):
            opened.append(self.tools)
            if any(t.endswith("@4") for t in self.tools):
                raise ActionGatewayError('Action Gateway error 400: session selection tool "github_list_commits" version "4" '
                                         'does not resolve to the released version "5"', 400)
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(prepare, "ActionGateway", Gateway)
    ok, detail = prepare.session_opens(["github_list_commits@5", "digitalocean_droplet-get@3"])
    assert ok and "2 tools" in detail
    ok, detail = prepare.session_opens(["github_list_commits@4"])
    assert not ok and 'released version "5"' in detail and "policy.yaml" in detail
    assert opened == [["github_list_commits@5", "digitalocean_droplet-get@3"], ["github_list_commits@4"]]


def test_only_the_remediator_holds_the_action_tools(example, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(example, "create_deep_agent", lambda **kwargs: seen.update(kwargs))

    async def run() -> None:
        async with example.team_sandboxes("t-roles") as sandboxes:
            example.build_commander(sandboxes)

    asyncio.run(run())
    names = lambda tools: {t.name for t in tools}  # noqa: E731
    by_role = {spec["name"]: names(spec["tools"]) for spec in seen["subagents"]}
    assert set(by_role) == {"metrics", "changes", "infra", "verifier", "remediator"}
    assert {"action_invoke", "action_search"} == by_role["remediator"]
    assert all("action_invoke" not in tools for role, tools in by_role.items() if role != "remediator")
    assert names(seen["tools"]) == {"save_postmortem"}  # the commander can write its report and nothing else
    assert {"sample_health", "judge_health", "digitalocean_droplet-get"} == by_role["verifier"]
    assert "remediator" not in {spec["name"] for spec in example.SUBAGENTS}  # no MicroVM of its own: it only calls the gateway


def test_investigators_have_their_microvm_files_and_no_shell(example, monkeypatch) -> None:
    from deepagents.middleware.filesystem import (
        FilesystemMiddleware,
        supports_execution,
    )

    seen: dict = {}
    monkeypatch.setattr(example, "create_deep_agent", lambda **kwargs: seen.update(kwargs))
    tools = sys.modules["tools"]

    async def run() -> str:
        async with example.team_sandboxes("t-shell") as sandboxes:
            example.build_commander(sandboxes)
            filesystems = {spec["name"]: next(m for m in spec.get("middleware", []) if isinstance(m, FilesystemMiddleware))
                           for spec in seen["subagents"] if spec["name"] != "remediator"}
            assert set(filesystems) == {"metrics", "changes", "infra", "verifier"}
            assert not any(supports_execution(fs.backend) for fs in filesystems.values())  # deepagents hides `execute`
            fetch_metrics, _ = tools.metrics_tools(sandboxes.backends["metrics"], example.service, local=True)
            await fetch_metrics.ainvoke({})  # the role's tool writes into the MicroVM...
            return (await filesystems["metrics"].backend.aread("/data/metrics.json")).file_data["content"]  # ...its file tools read it

    assert json.loads(asyncio.run(run()))[0]["version"]


def test_the_verdict_needs_three_healthy_samples_in_a_row(example) -> None:
    verdict = _script("verdict")
    ok = {"status": "healthy", "version": "9f3c2d1", "error_rate": 0.01}
    bad = {"status": "failing", "version": "a1b2c3d", "error_rate": 0.41}
    assert verdict.judge([])["verdict"] == "not_healthy"
    assert verdict.judge([ok, ok])["verdict"] == "not_healthy"  # too few samples
    assert verdict.judge([bad, ok, ok, ok])["verdict"] == "healthy"
    assert verdict.judge([ok, ok, bad, ok])["verdict"] == "not_healthy"  # the latest samples count
    assert verdict.judge([ok, ok, {**ok, "error_rate": 0.2}])["verdict"] == "not_healthy"  # "healthy" with a high error rate


def test_the_verifier_judges_the_service_before_and_after_the_restore(example) -> None:
    tools, world = sys.modules["tools"], sys.modules["incident_common.world"]

    async def check(restored: bool) -> dict:
        example.service.restored_at = time.time() if restored else None
        async with example.team_sandboxes(f"t-verify-{restored}") as sandboxes:
            sample, judge = tools.verifier_tools(sandboxes.backends["verifier"], example.service, local=True, every=0.2)
            await sample.ainvoke({"wait_seconds": 1})
            return json.loads(await judge.ainvoke({}))

    failing, healthy = asyncio.run(check(False)), asyncio.run(check(True))
    assert failing["verdict"] == "not_healthy" and failing["latest"]["version"] == world.BAD_SHA
    assert healthy["verdict"] == "healthy" and healthy["version"] == world.GOOD_SHA and healthy["samples"] >= 3


def test_investigator_microvms_start_on_first_use_and_pause_with_events(example, tmp_path) -> None:
    tools, world = sys.modules["tools"], sys.modules["incident_common.world"]

    async def run() -> set[str]:
        async with example.team_sandboxes("t1") as sandboxes:
            assert sandboxes.started() == {}  # nothing exists before an investigator acts
            fetch_metrics, find_onset = tools.metrics_tools(sandboxes.backends["metrics"], example.service, local=True)
            await fetch_metrics.ainvoke({})
            found = json.loads(await find_onset.ainvoke({}))
            assert found["version_at_onset"] == world.BAD_SHA  # the helper ran in the clone, on what was written there
            return set(sandboxes.started())

    assert asyncio.run(run()) == {"metrics"}
    assert [(e["event"], e["sandbox_id"]) for e in _events(tmp_path) if "sandbox_id" in e] == [
        ("clone-start", "t1-metrics"),
        ("clone", "t1-metrics"),
        ("pause", "t1-metrics"),
    ]


def test_the_dashboard_follows_the_microvms_the_droplet_and_the_gateway(example, tmp_path, monkeypatch) -> None:
    tools, world = sys.modules["tools"], sys.modules["incident_common.world"]
    dashboard = importlib.import_module("dashboard")
    monkeypatch.setattr(dashboard, "EVENTS", tmp_path / "events.jsonl")

    async def investigate() -> None:
        async with example.team_sandboxes("t2") as sandboxes:
            fetch, _ = tools.metrics_tools(sandboxes.backends["metrics"], example.service, local=True)
            await fetch.ainvoke({})

    asyncio.run(investigate())
    _drive(example.gateway, "digitalocean_restore-droplet", {"ID": world.DROPLET_ID, "ImageID": world.SNAPSHOT_ID}, ["approve"])
    _drive(example.gateway, "digitalocean_droplet-delete", {"ID": world.DROPLET_ID}, [])
    state = dashboard.State("t2", from_start=True)
    state.read_events()
    assert {(t.role, t.status) for t in state.tiles.values()} == {("metrics", "paused")}
    assert state.droplet is not None and state.droplet["what"] == "restoring"
    assert [row[1] for row in state.audit] == ["ask", "approve", "approved", "deny"]

    from rich.console import Console

    console = Console(width=200, record=True, file=open(tmp_path / "screen.txt", "w"))  # noqa: SIM115
    console.print(dashboard.render(state, "local"))
    screen = console.export_text()
    assert "Metrics Analyst" in screen and "PAUSED" in screen and f"droplet {world.DROPLET_NAME}" in screen
    assert "DENY" in screen and "digitalocean_droplet-delete" in screen  # the row carries the gateway's own refusal text


def _service_module(mode: str, history: Path, **constants: str):
    """service/app.py with the given mode and constants, imported as a fresh module (its workers start only on request)."""
    source = (_COMMON / "service" / "app.py").read_text().replace('CONFIRMATION_MODE = "async"', f'CONFIRMATION_MODE = "{mode}"', 1)
    source = source.replace('HISTORY = Path("/var/lib/checkout/history.jsonl")', f"HISTORY = Path({str(history)!r})", 1)
    for old, new in constants.items():
        assert old in source, old
        source = source.replace(old, new, 1)
    namespace: dict = {"__name__": "service_app", "__file__": str(_COMMON / "service" / "app.py")}
    exec(compile(source, "app.py", "exec"), namespace)  # noqa: S102 - our own file, to vary its constants
    return namespace


def test_the_service_fails_by_itself_when_it_waits_for_the_provider_inside_the_request(tmp_path) -> None:
    """Measured, not computed: a small pool, a short hold and a fast load keep the test to seconds, with the same shape as the demo."""
    shrunk = {"POOL_SIZE = 8": "POOL_SIZE = 2", '"sync": 3.4': '"sync": 0.4', "QUEUE_LIMIT = 4": "QUEUE_LIMIT = 2",
              "OFFERED_RPS = 4": "OFFERED_RPS = 12"}
    services = {mode: _service_module(mode, tmp_path / f"{mode}.jsonl", **shrunk) for mode in ("async", "sync")}
    try:
        for service in services.values():
            service["start_background"]()
        time.sleep(3)
        results = {mode: service["state"](time.time()) for mode, service in services.items()}
    finally:
        for service in services.values():
            service["STOP"].set()  # the sync workers would otherwise spin for the rest of the session
    # Relative, not absolute: a loaded CI runner slows both services.
    assert results["async"]["error_rate"] < 0.05 and results["async"]["p95_ms"] < results["sync"]["p95_ms"]
    assert results["sync"]["error_rate"] > 0.05 and results["sync"]["p95_ms"] > 350  # the pool saturates, requests are rejected


def test_setup_shows_the_connect_link_instead_of_failing_outside_a_graph(example, monkeypatch) -> None:
    setup = importlib.import_module("incident_common.setup_droplet")
    link = "https://cloud.digitalocean.com/security/connectlinks/confirm?token=t"
    connected = []

    def invoke(session_id, tool, arguments, rationale):
        if not connected:
            return {"error": {"message": f"connect your account: {link}"}}
        return {"output": {"commits": [{"sha": "b" * 40}, {"sha": "a" * 40}]}}

    gateway = setup.ActionGateway(setup.ActionPolicy(allow=["github_list_commits@5"]), tools=["github_list_commits@5"],
                                  client=SimpleNamespace(create_session=lambda *a: "urn:s", delete_session=lambda urn: None,
                                                         invoke=invoke, close=lambda: None))
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="no connection for this action yet. Open https://cloud"):
        setup.versions(gateway, "o/r")
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: connected.append(prompt))
    assert setup.versions(gateway, "o/r") == ("aaaaaaa", "bbbbbbb") and link in connected[0]


def test_the_graph_is_registered_for_langgraph_dev() -> None:
    config = json.loads((_EXAMPLE / "langgraph.json").read_text())
    assert config["graphs"] == {"incident_response": "./agent.py:make_graph"}
