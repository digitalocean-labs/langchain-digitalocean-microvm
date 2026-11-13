"""ActionGateway against the in-memory gateway: tools, policy, approvals and connect links as interrupts."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langchain_digitalocean_microvm.integrations.action_gateway import (
    SESSION_PREFIX,
    ActionGateway,
    ActionGatewayClient,
    ActionGatewayError,
    ActionPolicy,
    PendingApproval,
)
from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    CONNECT_URL,
    LocalActionGatewayClient,
)

POLICY = ActionPolicy(allow=["exa_web_search"], ask=["digitalocean_droplet-create"], deny=["digitalocean_region-list"])
SLUGS = ["exa_web_search@14", "digitalocean_droplet-create@3", "digitalocean_region-list@3"]
CREATE = {"tool": "digitalocean_droplet-create", "arguments": {"name": "web"}, "rationale": "needed"}


class State(TypedDict, total=False):
    answer: str


def make(gateway: ActionGateway, call: dict[str, Any]):
    """A graph whose one node calls `action_invoke`, as an agent's tool node would."""
    invoke = next(t for t in gateway.tools() if t.name == "action_invoke")

    def node(state: State) -> State:
        return {"answer": invoke.invoke(call)}

    graph = StateGraph(State)
    graph.add_node("act", node)
    graph.add_edge(START, "act")
    graph.add_edge("act", END)
    return graph.compile(checkpointer=InMemorySaver())


def thread(n: str = "t") -> dict[str, Any]:
    return {"configurable": {"thread_id": n}}


@pytest.fixture
def fake() -> LocalActionGatewayClient:
    return LocalActionGatewayClient()


@pytest.fixture
def gateway(fake: LocalActionGatewayClient):
    with ActionGateway(POLICY, tools=SLUGS, client=fake, actor_id="user-1") as gw:
        yield gw


def test_policy_spec_always_allows_the_meta_tools_and_drops_versions():
    spec = ActionPolicy(allow=["a@3"], ask=["b"], deny=["c"]).to_spec()
    assert spec["defaultAction"] == "deny"
    assert spec["rules"] == [
        {"tool": "action_search", "action": "allow"},
        {"tool": "action_invoke", "action": "allow"},
        {"tool": "a", "action": "allow"},
        {"tool": "b", "action": "ask"},
        {"tool": "c", "action": "deny"},
    ]


def test_session_is_named_with_the_prefix_and_deleted_on_close(fake: LocalActionGatewayClient):
    with ActionGateway(POLICY, client=fake, actor_id="user-1") as gw:
        gw.tools()
        (session,) = fake.sessions.values()
        assert session["name"].startswith(SESSION_PREFIX)
        assert session["actor_id"] == "user-1"
        assert session["tools"] == ["exa_web_search", "digitalocean_droplet-create"]  # default: allow + ask
    assert fake.sessions == {}
    assert fake.closed


def test_a_session_prefix_names_the_session_and_lists_it(fake: LocalActionGatewayClient):
    other = ActionGateway(POLICY, client=fake).open()
    with ActionGateway(POLICY, client=fake, session_prefix="microvm-ag-inc1-remediator-") as gw:
        gw.open()
        (mine,) = fake.list_sessions(prefix="microvm-ag-inc1-")
        assert re.fullmatch(r"microvm-ag-inc1-remediator-[0-9a-f]{8}", mine["name"])
        assert mine["sessionUrn"].rsplit(":", 1)[-1] == gw.session_id
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", mine["createdAt"])
        assert len(fake.list_sessions()) == 2
    assert fake.list_sessions(prefix="microvm-ag-inc1-") == []
    other.close()
    with pytest.raises(ValueError, match="session_prefix"):
        ActionGateway(POLICY, client=fake, session_prefix="")


def test_client_lists_every_page_and_filters_by_prefix():
    names = [f"microvm-ag-{i:03d}" for i in range(150)] + ["other-1"]
    pages: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        pages.append(params)
        page, per = int(params["page"]), int(params["per_page"])
        chunk = names[(page - 1) * per: page * per]
        return httpx.Response(200, json={"sessions": [{"sessionUrn": f"urn:{n}", "name": n} for n in chunk],
                                         "pagination": {"page": page, "per_page": per, "total": len(names)}})

    client = ActionGatewayClient("tok")
    client._api = httpx.Client(base_url="https://api.digitalocean.com/v2/action-gateway", transport=httpx.MockTransport(handler))
    assert len(client.list_sessions()) == 151
    assert [p["page"] for p in pages] == ["1", "2"] and pages[0]["per_page"] == "100"
    found = client.list_sessions(prefix="microvm-ag-14")
    assert [s["name"] for s in found] == [f"microvm-ag-14{i}" for i in range(10)]


def test_no_tools_or_unknown_preload_is_refused(fake: LocalActionGatewayClient):
    with pytest.raises(ValueError, match="no tools"):
        ActionGateway(ActionPolicy(), client=fake)
    with pytest.raises(ValueError, match="preload"):
        ActionGateway(POLICY, preload=["other@1"], client=fake)


def test_missing_token_is_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DO_ACTION_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("DO_MICROVM_TRANSPORT", raising=False)
    with pytest.raises(ActionGatewayError, match="DO_ACTION_GATEWAY_TOKEN"):
        ActionGateway(POLICY)


def test_fake_transport_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DO_MICROVM_TRANSPORT", "local")
    with ActionGateway(POLICY) as gw:
        search = next(t for t in gw.tools() if t.name == "action_search")
        assert "exa_web_search" in search.invoke({"query": "web"})


def test_meta_tools_search_and_invoke_an_allowed_action(gateway: ActionGateway, fake: LocalActionGatewayClient):
    search, invoke = gateway.tools()
    assert [t["name"] for t in json.loads(search.invoke({"query": "search web"}))] == ["exa_web_search"]
    out = invoke.invoke({"tool": "exa_web_search@14", "arguments": {"query": "x"}})
    assert json.loads(out) == {"results": ["result for x"]}
    assert fake.calls == [("exa_web_search", {"query": "x"})]


def test_denied_and_unknown_actions_come_back_as_errors(gateway: ActionGateway):
    invoke = gateway.tools()[1]
    assert "denied by the session policy" in invoke.invoke({"tool": "digitalocean_region-list"})
    assert "unknown tool" in invoke.invoke({"tool": "nope"})


def test_preloaded_tools_are_direct_tools_with_their_schema(fake: LocalActionGatewayClient):
    with ActionGateway(POLICY, tools=SLUGS, preload=["exa_web_search@14"], client=fake) as gw:
        assert [t.name for t in gw.tools()] == ["action_search", "action_invoke", "exa_web_search"]
        (direct,) = gw.tools(meta=False)
        assert direct.args["query"]["type"] == "string"
        assert json.loads(direct.invoke({"query": "q"})) == {"results": ["result for q"]}


async def test_async_tools(gateway: ActionGateway):
    invoke = gateway.tools()[1]
    out = await invoke.ainvoke({"tool": "exa_web_search", "arguments": {"query": "q"}})
    assert json.loads(out) == {"results": ["result for q"]}


def test_ask_interrupts_then_approve_runs_the_call_once(gateway: ActionGateway, fake: LocalActionGatewayClient):
    graph = make(gateway, CREATE)
    out = graph.invoke({}, thread())
    (stop,) = out["__interrupt__"]
    assert stop.value["type"] == "action_approval"
    assert stop.value["tool"] == "digitalocean_droplet-create"
    assert stop.value["arguments"] == {"name": "web"}
    approval_id = stop.value["approval_id"]
    assert fake.calls == []

    final = graph.invoke(Command(resume="approve"), thread())
    assert json.loads(final["answer"]) == {"created": {"name": "web"}}
    assert fake.decisions == [(approval_id, "approve")]
    assert len(fake.calls) == 1
    assert len(fake._approvals) == 1  # the resumed run did not ask for a second approval


def test_deny_does_not_run_the_call(gateway: ActionGateway, fake: LocalActionGatewayClient):
    graph = make(gateway, CREATE)
    graph.invoke({}, thread())
    final = graph.invoke(Command(resume={"decision": "deny", "reason": "too big"}), thread())
    assert final["answer"] == "The user denied this action. Reason: too big"
    assert fake.calls == []
    assert [d for _, d in fake.decisions] == ["deny"]


def test_an_unrecognised_resume_value_denies(gateway: ActionGateway, fake: LocalActionGatewayClient):
    graph = make(gateway, CREATE)
    graph.invoke({}, thread())
    graph.invoke(Command(resume="maybe"), thread())
    assert fake.calls == []
    assert [d for _, d in fake.decisions] == ["deny"]


def test_an_expired_approval_is_reported_to_the_agent(gateway: ActionGateway, fake: LocalActionGatewayClient):
    graph = make(gateway, CREATE)
    out = graph.invoke({}, thread())
    fake.expire(out["__interrupt__"][0].value["approval_id"])
    final = graph.invoke(Command(resume="approve"), thread())
    assert "expired" in final["answer"]
    assert fake.calls == []


def test_a_new_call_after_expiry_asks_again(gateway: ActionGateway, fake: LocalActionGatewayClient):
    first = make(gateway, CREATE)
    out = first.invoke({}, thread("a"))
    fake.expire(out["__interrupt__"][0].value["approval_id"])
    second = make(gateway, CREATE).invoke({}, thread("b"))
    assert second["__interrupt__"][0].value["approval_id"] != out["__interrupt__"][0].value["approval_id"]


def test_request_returns_the_pending_approval_and_decide_runs_the_call_once(gateway: ActionGateway, fake: LocalActionGatewayClient):
    pending = gateway.request(**CREATE)  # no graph: the supervisor holds the approval, not an interrupt
    assert isinstance(pending, PendingApproval)
    assert (pending.tool, dict(pending.arguments), pending.rationale) == ("digitalocean_droplet-create", {"name": "web"}, "needed")
    assert fake.calls == []
    assert json.loads(gateway.decide(pending, "approve")) == {"created": {"name": "web"}}
    assert fake.decisions == [(pending.approval_id, "approve")]
    assert len(fake.calls) == 1


def test_decide_deny_or_an_expired_approval_runs_nothing(gateway: ActionGateway, fake: LocalActionGatewayClient):
    denied = gateway.request(**CREATE)
    assert isinstance(denied, PendingApproval)
    assert gateway.decide(denied, "deny", "too big") == "The user denied this action. Reason: too big"
    expired = gateway.request(**{**CREATE, "arguments": {"name": "db"}})  # a denied call stays denied: another one
    assert isinstance(expired, PendingApproval)
    fake.expire(expired.approval_id)
    assert "expired" in gateway.decide(expired, "approve")
    assert fake.calls == []


def test_request_answers_allowed_denied_and_unconnected_actions_as_text(gateway: ActionGateway, fake: LocalActionGatewayClient):
    assert json.loads(gateway.request("exa_web_search", {"query": "q"})) == {"results": ["result for q"]}
    assert gateway.request("digitalocean_region-list").startswith("Error:")  # denied by the policy
    fake.sessions[next(iter(fake.sessions))]["policy"] = ActionPolicy(allow=["digitalocean_region-list"]).to_spec()
    assert CONNECT_URL in gateway.request("digitalocean_region-list")


def test_connect_link_interrupts_and_resume_retries(gateway: ActionGateway, fake: LocalActionGatewayClient):
    gateway.policy = ActionPolicy(allow=["digitalocean_region-list"])  # only for the fake's rules below
    fake.sessions[next(iter(fake.sessions))]["policy"] = gateway.policy.to_spec()
    graph = make(gateway, {"tool": "digitalocean_region-list"})
    out = graph.invoke({}, thread())
    (stop,) = out["__interrupt__"]
    assert stop.value == {"type": "connect_required", "url": CONNECT_URL, "tool": "digitalocean_region-list"}
    fake.connect("digitalocean_region-list")
    final = graph.invoke(Command(resume="connected"), thread())
    assert json.loads(final["answer"]) == {"regions": ["nyc3", "sfo3", "ams3"]}


def test_connect_cancel_and_a_link_still_unopened(gateway: ActionGateway, fake: LocalActionGatewayClient):
    fake.sessions[next(iter(fake.sessions))]["policy"] = ActionPolicy(allow=["digitalocean_region-list"]).to_spec()
    call = {"tool": "digitalocean_region-list"}
    cancelled = make(gateway, call)
    cancelled.invoke({}, thread("c"))
    assert "did not open" in cancelled.invoke(Command(resume="cancel"), thread("c"))["answer"]
    unopened = make(gateway, call)
    unopened.invoke({}, thread("d"))
    assert "Still not connected" in unopened.invoke(Command(resume="done"), thread("d"))["answer"]


def test_client_speaks_the_data_plane_protocol():
    seen: list[httpx.Request] = []
    hit = {"name": "t", "description": "d", "inputSchema": {"type": "object"}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v2/action-gateway/sessions":
            return httpx.Response(200, json={"session": {"sessionUrn": "do:ag:session:abc"}})
        if request.url.path == "/tools/invoke":
            return httpx.Response(200, json={"results": [{"result": {"status": "success", "output": {"ok": 1}}}]})
        if request.url.path == "/tools/search":
            return httpx.Response(200, json={"status": "succeeded", "output": {"results": [{"index": 1, "results": [hit]}]}})
        if request.url.path.startswith("/approvals/"):
            return httpx.Response(410, text="gone")
        return httpx.Response(200, json={})

    client = ActionGatewayClient("tok")
    client._api = httpx.Client(base_url="https://api.digitalocean.com/v2/action-gateway", transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer tok"})
    client._data = httpx.Client(base_url="https://actions.do-ai.run", transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer tok"})
    urn = client.create_session("microvm-ag-1", ["t@1"], ActionPolicy(allow=["t"]).to_spec(), "u")
    assert urn == "do:ag:session:abc"
    assert json.loads(seen[0].content)["actorId"] == "u"
    assert client.invoke("abc", "t", {"a": 1}, "why") == {"status": "success", "output": {"ok": 1}}
    assert seen[1].headers["X-Session-Id"] == "abc" and seen[1].headers["Authorization"] == "Bearer tok"
    assert json.loads(seen[1].content) == {"rationale": "why", "tools": [{"tool": "t", "arguments": {"a": 1}}]}
    assert client.search("abc", "do t", 3) == [hit]
    assert json.loads(seen[2].content) == {"queries": [{"use_case": "do t"}], "limit": 3}
    with pytest.raises(ActionGatewayError) as err:
        client.decide("abc", None, "id1", "approve")
    assert err.value.status_code == 410
    assert seen[3].url.path == "/approvals/session/abc/id1"


def test_ask_on_an_unconnected_action_approves_then_connects_then_runs(fake: LocalActionGatewayClient):
    policy = ActionPolicy(ask=["digitalocean_region-list"])
    with ActionGateway(policy, client=fake) as gw:
        graph = make(gw, {"tool": "digitalocean_region-list"})
        out = graph.invoke({}, thread())
        assert out["__interrupt__"][0].value["type"] == "action_approval"
        out = graph.invoke(Command(resume="approve"), thread())
        (stop,) = out["__interrupt__"]  # the approval went to a call that needed a connection
        assert stop.value["type"] == "connect_required" and stop.value["url"] == CONNECT_URL
        assert fake.calls == []
        fake.connect("digitalocean_region-list")
        final = graph.invoke(Command(resume="connected"), thread())
        assert json.loads(final["answer"]) == {"regions": ["nyc3", "sfo3", "ams3"]}
        assert [d for _, d in fake.decisions] == ["approve", "approve"]
