"""examples/10_erasure in local mode: the mock systems, the gateway's policy and approvals, the classifier, the verifier, the report.

No model is called: the graph below is one node that calls `action_invoke` the way the coordinator does, so the
`allow` / `ask` / `deny` tiers and the approval interrupt are checked end to end, and the locators' tools run in
local MicroVMs cloned from the example's template.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

blockbuster = pytest.importorskip("blockbuster")
pytest.importorskip("rich")

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "10_erasure"
_MODULES = ("agent", "models", "tools", "events", "world", "gateway", "run_script", "prepare", "dashboard")
PERSONAL = ("jane.doe@example.com", "jane.doe@example.org", "Jane", "Doe", "555 0134", "Oakland", "Broadway", "94612")


@pytest.fixture
def example(tmp_path, monkeypatch):
    """The example's `agent` module, imported in local mode with its own sandbox root, event log and report path."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("DO_MICROVM_TRANSPORT", "local")
    monkeypatch.setenv("EXAMPLES_MODEL", "anthropic:claude-haiku-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("DO_MODEL_ACCESS_KEY", raising=False)
    monkeypatch.syspath_prepend(str(_EXAMPLE))
    # Other tests import modules of the same names (the agent-host template's `agent`, examples 07-09's `tools`).
    names = lambda: [n for n in sys.modules if n in _MODULES or n.startswith(("tools.", "gateway."))]  # noqa: E731
    saved = {n: sys.modules.pop(n) for n in names()}
    module = importlib.import_module("agent")
    monkeypatch.setattr(sys.modules["events"], "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setattr(sys.modules["tools"], "REPORT", tmp_path / "erasure-report.md")
    yield module
    module.gateway.close()
    for name in names():
        del sys.modules[name]
    sys.modules.update(saved)


def _classifier():
    spec = importlib.util.spec_from_file_location("classify", _EXAMPLE / "sandbox" / "classify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _State(TypedDict, total=False):
    answer: str


def _drive(gateway, tool: str, arguments: dict, decisions: list[str]) -> tuple[str, list[dict]]:
    """Call `tool` through `action_invoke` in a graph; answer each interrupt from `decisions` (none left: leave it open).
    Returns the answer and the interrupts."""
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
        if not decisions:
            return "", stops
        out = app.invoke(Command(resume=decisions.pop(0)), config)
    return out["answer"], stops


def _manifest(example, role: str) -> dict:
    """What `classify.py` makes of one locator's records (run here, on what its fetcher read through the gateway)."""
    tools, world = sys.modules["tools"], sys.modules["world"]
    sources = tools.FETCHERS[role](example.gateway, world.REQUEST_EMAIL, world.REQUEST_NAME)
    request = {"email": world.REQUEST_EMAIL, "name": world.REQUEST_NAME}
    return _classifier().classify(request, {"system": role, "sources": sources})


def test_the_policy_tiers_are_written_out(example) -> None:
    policy = sys.modules["gateway"].POLICY
    bare = lambda names: {n.split("@")[0] for n in names}  # noqa: E731
    assert bare(policy.ask) == {"hubspot_crm_contacts_gdpr_delete_create", "intercom_delete_a_contact",
                                "stripe_post_customers_customer", "mailchimp_forget_contact", "mixpanel_delete_profile"}
    assert bare(policy.deny) == {"stripe_delete_customers_customer", "hubspot_files_gdpr_delete_by_fileid"}
    assert all(any(w in n for w in ("search", "get", "query")) for n in bare(policy.allow))
    assert "intercom_search_contacts" in bare(policy.allow)  # a read, whatever the catalog's hint says
    actions = example.systems.actions()
    assert bare([*policy.allow, *policy.ask, *policy.deny]) <= set(actions)


def test_the_mocks_check_required_arguments_as_the_catalog_has_them(example) -> None:
    actions = example.systems.actions()
    search = actions["hubspot_crm_contacts_search"].run
    assert "after" in search({"filter_groups": [], "properties": [], "sorts": [], "limit": 10})["error"]
    assert "audience_id" in actions["mailchimp_forget_contact"].run({"contact_id": "x"})["error"]
    bad = {"project_id": 31337, "updates": [{"$distinct_id": "u-5c1e9a", "$delete": ""}]}  # no $token
    assert "$token" in actions["mixpanel_delete_profile"].run(bad)["error"]
    assert "u-5c1e9a" in example.systems.profiles  # nothing was deleted


def test_the_classifier_skips_the_decoy_retains_the_invoices_and_names_nobody(example) -> None:
    world = sys.modules["world"]
    crm = _manifest(example, "crm")
    by_id = {r["id"]: r for r in crm["records"]}
    assert by_id[world.CRM_ID]["identity"] == "match" and by_id[world.CRM_ID]["action"] == "erase"
    assert by_id[world.CRM_ID]["call"] == {"tool": "hubspot_crm_contacts_gdpr_delete_create", "arguments": {"object_id": world.CRM_ID}}
    assert by_id[world.DECOY_ID]["identity"] == "different_person" and by_id[world.DECOY_ID]["action"] == "skip"
    assert "call" not in by_id[world.DECOY_ID] and by_id[world.CRM_ID]["related"] == {"notes": 2}

    billing = _manifest(example, "billing")
    actions = {(r["kind"], r["action"]) for r in billing["records"]}
    assert actions == {("customer", "redact"), ("invoice", "retain")} and billing["counts"]["retain"] == 3
    customer = next(r for r in billing["records"] if r["kind"] == "customer")
    assert customer["call"]["tool"] == "stripe_post_customers_customer" and customer["call"]["arguments"]["email"] == ""

    marketing = _manifest(example, "marketing")
    assert [r["call"]["tool"] for r in marketing["records"]] == ["mailchimp_forget_contact", "mixpanel_delete_profile"]
    for manifest in (crm, billing, marketing, _manifest(example, "support")):
        text = json.dumps(manifest)
        assert not any(value.lower() in text.lower() for value in PERSONAL), text  # ids, categories and counts only
        assert manifest["request"].startswith("sha256:")


def test_reads_run_an_erasure_waits_for_a_human_and_the_billing_delete_is_refused(example) -> None:
    world, systems, gateway = sys.modules["world"], example.systems, example.gateway

    answer, stops = _drive(gateway, "intercom_get_a_contact", {"contact_id": world.SUPPORT_ID}, [])
    assert stops == [] and world.SUPPORT_ID in answer  # allow: ran, no question

    answer, stops = _drive(gateway, "intercom_delete_a_contact", {"contact_id": world.SUPPORT_ID}, ["deny"])
    assert [s["type"] for s in stops] == ["action_approval"] and "denied" in answer.lower()
    assert world.SUPPORT_ID in systems.support  # a denied approval ran nothing

    answer, stops = _drive(gateway, "hubspot_crm_contacts_gdpr_delete_create", {"object_id": world.CRM_ID}, ["approve"])
    assert len(stops) == 1 and world.CRM_ID in answer and world.CRM_ID not in systems.crm
    assert world.DECOY_ID in systems.crm  # an approval covers the one record its arguments name

    answer, stops = _drive(gateway, "stripe_delete_customers_customer", {"customer": world.BILLING_ID}, [])
    assert stops == [] and answer.startswith("Error:") and world.BILLING_ID in systems.customers  # deny: nobody is asked

    seen = [(e["event"], e["tool"], e.get("status") or e["decision"]) for e in example.audited.audit]
    assert seen == [
        ("gateway", "intercom_get_a_contact", "allow"),
        ("gateway", "intercom_delete_a_contact", "ask"),
        ("decision", "intercom_delete_a_contact", "deny"),
        ("gateway", "hubspot_crm_contacts_gdpr_delete_create", "ask"),
        ("decision", "hubspot_crm_contacts_gdpr_delete_create", "approve"),
        ("gateway", "hubspot_crm_contacts_gdpr_delete_create", "approved"),
        ("gateway", "stripe_delete_customers_customer", "deny"),
    ]


def test_an_erasure_end_to_end_keeps_the_decoy_and_the_invoices_and_reports_without_personal_data(example, tmp_path) -> None:
    tools, world, systems = sys.modules["tools"], sys.modules["world"], example.systems

    async def locate() -> dict[str, dict]:
        manifests = {}
        async with example.team_sandboxes("t1") as sandboxes:
            for role in example.LOCATED:
                fetch, classify = tools.locator_tools(role, sandboxes.backends[role], example.gateway, local=True)
                assert (await fetch.ainvoke({"email": world.REQUEST_EMAIL, "name": world.REQUEST_NAME})).startswith("Wrote ")
                manifests[role] = json.loads(await classify.ainvoke({}))
                root = Path(tools.root_for_host((await sandboxes.backends[role].aget()).id))
                assert not (root / "data" / "records.json").exists()  # the raw copy is gone from the MicroVM
            assert set(sandboxes.started()) == set(example.LOCATED)
        return manifests

    manifests = asyncio.run(locate())
    plan = [{"system": {"mixpanel": "analytics", "mailchimp": "marketing"}.get(r["source"], role), **r}
            for role, m in manifests.items() for r in m["records"]]
    for record in plan:
        if "call" in record:  # the marketing erasure is denied, the rest approved
            decision = "deny" if record["system"] == "marketing" else "approve"
            _drive(example.gateway, record["call"]["tool"], record["call"]["arguments"], [decision])

    assert world.DECOY_ID in systems.crm and world.CRM_ID not in systems.crm  # the other Jane Doe is untouched
    assert set(systems.invoices) == set(world.INVOICE_IDS)  # retained
    customer = systems.customers[world.BILLING_ID]
    assert not any(customer.get(f) for f in ("name", "email", "phone", "address"))  # redacted, not deleted
    assert world.MAILCHIMP_CONTACT in systems.members and not systems.profiles  # denied stays; approved went

    records = [{k: r[k] for k in ("system", "kind", "id", "action")} for r in plan]
    result = tools.verify(example.gateway, example.ledger, world.REQUEST_EMAIL, world.REQUEST_NAME, records)
    status = {(r["system"], r["id"]): r["status"] for r in result["records"]}
    assert status[("crm", world.CRM_ID)] == "erased" and status[("crm", world.DECOY_ID)] == "present"
    assert status[("billing", world.BILLING_ID)] == "redacted" and status[("billing", world.INVOICE_IDS[0])] == "retained"
    assert status[("marketing", world.MAILCHIMP_CONTACT)] == "still present" and status[("analytics", world.MIXPANEL_ID)] == "erased"
    assert result["missed"] == [] and result["complete"] is False

    report = tools.render_report(example.ledger, f"Erased {world.CRM_ID}; {world.REQUEST_NAME} <{world.REQUEST_EMAIL}> asked.")
    assert not any(value.lower() in report.lower() for value in PERSONAL), report
    assert tools.request_hash(world.REQUEST_EMAIL) in report and "INCOMPLETE (1 open)" in report
    row = next(line for line in report.splitlines() if world.MAILCHIMP_CONTACT in line)
    assert "denied by the approver" in row and "still present ✘" in row
    assert "| erase | approved | erased ✔ |" in next(line for line in report.splitlines() if f"`{world.CRM_ID}`" in line)


def test_the_dashboard_follows_every_record_from_the_event_log(example, tmp_path, monkeypatch) -> None:
    tools, world = sys.modules["tools"], sys.modules["world"]
    dashboard = importlib.import_module("dashboard")
    monkeypatch.setattr(dashboard, "EVENTS", tmp_path / "events.jsonl")

    async def locate() -> None:
        async with example.team_sandboxes("t2") as sandboxes:
            for role in ("crm", "billing", "marketing"):
                fetch, classify = tools.locator_tools(role, sandboxes.backends[role], example.gateway, local=True)
                await fetch.ainvoke({"email": world.REQUEST_EMAIL, "name": world.REQUEST_NAME})
                await classify.ainvoke({})

    asyncio.run(locate())
    gateway = example.gateway
    _drive(gateway, "hubspot_crm_contacts_gdpr_delete_create", {"object_id": world.CRM_ID}, ["approve"])
    _drive(gateway, "mailchimp_forget_contact", {"audience_id": world.AUDIENCE_ID, "contact_id": world.MAILCHIMP_CONTACT}, ["deny"])
    redact = {"customer": world.BILLING_ID, "name": "", "email": "", "phone": "", "address": {"line1": "", "city": "", "postal_code": "", "country": ""}}
    _drive(gateway, "stripe_post_customers_customer", redact, ["approve"])
    _drive(gateway, "stripe_delete_customers_customer", {"customer": world.BILLING_ID}, [])
    _drive(gateway, "mixpanel_engage_query", {"project_id": world.MIXPANEL_PROJECT}, [])  # a read: not listed
    _drive(gateway, "mailchimp_forget_contact", {"audience_id": world.AUDIENCE_ID, "contact_id": "mc_other"}, [])  # left open
    state = dashboard.State("t2", from_start=True)
    state.read_events()

    seen = {(r.system, r.id): (r.action, r.state, r.note) for r in state.records.values()}
    assert seen[("crm", world.CRM_ID)] == ("erase", "erased", "")
    assert seen[("crm", world.DECOY_ID)] == ("skip", "skipped", "")
    assert seen[("billing", world.BILLING_ID)] == ("redact", "redacted", "delete refused by policy")
    assert seen[("billing", world.INVOICE_IDS[0])] == ("retain", "kept", "")
    assert seen[("marketing", world.MAILCHIMP_CONTACT)] == ("erase", "denied", "")
    assert seen[("analytics", world.MIXPANEL_ID)] == ("erase", "found", "")
    assert seen[("marketing", "mc_other")][1] == "awaiting approval"
    assert {t.role for t in state.tiles.values()} == {"crm", "billing", "marketing"}
    assert not any("mixpanel_engage_query" in row[2] for row in state.audit)

    tools.verify(gateway, example.ledger, world.REQUEST_EMAIL, world.REQUEST_NAME,
                 [{"system": "crm", "kind": "contact", "id": world.CRM_ID, "action": "erase"},
                  {"system": "marketing", "kind": "member", "id": world.MAILCHIMP_CONTACT, "action": "erase"}])
    state.read_events()
    assert state.records[("crm", "contact", world.CRM_ID)].verified is True
    assert state.records[("marketing", "member", world.MAILCHIMP_CONTACT)].note == "still present"
    assert state.outcome is not None and state.outcome.startswith("INCOMPLETE")

    from rich.console import Console

    console = Console(width=200, record=True, file=open(tmp_path / "screen.txt", "w"))  # noqa: SIM115
    console.print(dashboard.render(state, "local"))
    screen = console.export_text()
    assert "ERASED ✔" in screen and "DENIED" in screen and "delete refused by policy" in screen
    assert "(another person)" in screen and "verified: INCOMPLETE" in screen
    assert not any(value.lower() in screen.lower() for value in PERSONAL), screen


def test_the_verifier_finds_what_the_plan_missed(example) -> None:
    tools, world = sys.modules["tools"], sys.modules["world"]
    result = tools.verify(example.gateway, example.ledger, world.REQUEST_EMAIL, world.REQUEST_NAME,
                          [{"system": "crm", "kind": "contact", "id": world.CRM_ID, "action": "erase"}])
    assert result["records"][0]["status"] == "still present"
    assert {(m["system"], m["id"]) for m in result["missed"]} == {
        ("support", world.SUPPORT_ID), ("billing", world.BILLING_ID), ("marketing", world.MAILCHIMP_CONTACT),
        ("analytics", world.MIXPANEL_ID)}


def test_the_approval_shows_independent_facts_about_the_record(example) -> None:
    world = sys.modules["world"]
    run_script = importlib.import_module("run_script")
    facts = lambda tool, arguments: "\n".join(run_script.verified_facts({"tool": tool, "arguments": arguments}))  # noqa: E731
    assert "matches the request ✔" in facts("hubspot_crm_contacts_gdpr_delete_create", {"object_id": world.CRM_ID})
    decoy = facts("hubspot_crm_contacts_gdpr_delete_create", {"object_id": world.DECOY_ID})
    assert "DIFFERENT email address (@example.org)" in decoy and world.DECOY_EMAIL not in decoy
    redact = {"customer": world.BILLING_ID, "name": "", "email": "", "address": {"line1": ""}}
    assert "clears name, email, address ✔" in facts("stripe_post_customers_customer", redact)
    assert "3 invoices" in facts("stripe_post_customers_customer", redact)
    assert "SETS email" in facts("stripe_post_customers_customer", {**redact, "email": "someone@example.net"})
    assert "matches the request ✔" in facts("mailchimp_forget_contact", {"audience_id": world.AUDIENCE_ID, "contact_id": world.MAILCHIMP_CONTACT})
    analytics = {"project_id": world.MIXPANEL_PROJECT, "updates": [{"$token": "t", "$distinct_id": "u-nobody", "$delete": ""}]}
    assert "NOT among the profiles" in facts("mixpanel_delete_profile", analytics)
    assert run_script.verified_facts({"tool": "intercom_get_a_contact", "arguments": {}}) == []


async def test_locator_tools_do_not_block_the_event_loop(example) -> None:
    tools, world = sys.modules["tools"], sys.modules["world"]
    uploads: dict[str, bytes] = {}
    commands: list[str] = []

    class _Sandbox:
        id = "erasure-test-sandbox"

        def upload_files(self, files):
            uploads.update(files)
            return [SimpleNamespace(path=p, error=None) for p, _ in files]

        async def aexecute(self, command: str, timeout: int | None = None):
            commands.append(command)
            return SimpleNamespace(exit_code=0, output='{"records": []}')

    fetch, classify = tools.locator_tools("billing", _Sandbox(), example.gateway, local=True)
    with blockbuster.blockbuster_ctx():
        wrote = await fetch.ainvoke({"email": world.REQUEST_EMAIL, "name": world.REQUEST_NAME})
        found = await classify.ainvoke({})
    assert wrote == "Wrote 1 stripe_customers, 3 stripe_invoices to /data/records.json."
    assert json.loads(uploads["/data/request.json"]) == {"email": world.REQUEST_EMAIL, "name": world.REQUEST_NAME}
    assert found == '{"records": []}' and f"{tools.root_for_host(_Sandbox.id)}/tools/classify.py" in commands[0]


def test_the_report_tool_writes_the_file_and_scrubs_the_notes(example, tmp_path) -> None:
    tools, world = sys.modules["tools"], sys.modules["world"]
    example.ledger.name = world.REQUEST_NAME
    save = next(t for t in tools.coordinator_tools(example.gateway, example.ledger) if t.name == "save_report")

    async def run() -> str:
        with blockbuster.blockbuster_ctx():
            return await save.ainvoke({"notes": "jane DOE asked; mail jane.doe+x@example.com"})

    answer = asyncio.run(run())
    text = (tmp_path / "erasure-report.md").read_text()
    assert str(tmp_path / "erasure-report.md") in answer and text.startswith("# Erasure report")
    assert "[name removed] asked; mail [email removed]" in text and "(no verification ran)" in text
    assert "Request: unknown" in text


def test_locators_have_their_microvm_files_and_no_shell(example, monkeypatch) -> None:
    from deepagents.middleware.filesystem import (
        FilesystemMiddleware,
        supports_execution,
    )

    seen: dict = {}
    monkeypatch.setattr(example, "create_deep_agent", lambda **kwargs: seen.update(kwargs))
    tools, world = sys.modules["tools"], sys.modules["world"]

    async def run() -> str:
        async with example.team_sandboxes("t-shell") as sandboxes:
            example.build_coordinator(sandboxes)
            filesystems = {spec["name"]: next(m for m in spec["middleware"] if isinstance(m, FilesystemMiddleware))
                           for spec in seen["subagents"]}
            assert set(filesystems) == {"crm", "support", "billing", "marketing"}
            assert not any(supports_execution(fs.backend) for fs in filesystems.values())  # deepagents hides `execute`
            fetch_records = tools.locator_tools("crm", sandboxes.backends["crm"], example.gateway, local=True)[0]
            await fetch_records.ainvoke({"email": world.REQUEST_EMAIL, "name": world.REQUEST_NAME})  # the tool writes...
            return (await filesystems["crm"].backend.aread("/data/request.json")).file_data["content"]  # ...its file tools read

    assert json.loads(asyncio.run(run()))["email"] == sys.modules["world"].REQUEST_EMAIL


def test_seed_files_hold_the_classifier(example) -> None:
    assert {path for path, _ in example.seed_files()} == {"/tools/classify.py"}
    assert [spec["name"] for spec in example.SUBAGENTS] == ["crm", "support", "billing", "marketing"]
