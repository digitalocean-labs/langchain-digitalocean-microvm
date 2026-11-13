"""The gateway retries a host's 502/503/504 before counting it unreachable.

A MicroVM that paused itself at its idle timeout can answer the first request with a plain-text
503 from DigitalOcean's edge while it resumes; a second request about a second later gets
through. GETs (and the idempotent thread create, cancel and artifact reads) are retried on any
such status; a run creation only when the answer did not come from the Agent Server (which
answers JSON), because the server may already have started the run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fake_agent_server import FakeHosts, make_transport
from test_gateway import FakeClock
from test_gateway_artifacts import FilesystemFakeHosts

from langchain_digitalocean_microvm.agent_host import HostSpec, SubAgentGateway

TOKEN = "gw-retry-token"


class FlakyEdge:
    """Answers chosen requests with injected errors before the local hosts see them; records what got through."""

    def __init__(self, hosts: FakeHosts) -> None:
        self.inner = make_transport(hosts)
        self.injected: list[tuple[str, str, httpx.Response]] = []  # (method, path suffix, response)
        self.seen: list[tuple[str, str]] = []  # requests that reached a host

    def fail(self, method: str, suffix: str, *responses: httpx.Response) -> None:
        self.injected.extend((method, suffix, r) for r in responses)

    def transport(self) -> httpx.AsyncBaseTransport:
        async def handler(request: httpx.Request) -> httpx.Response:
            for i, (method, suffix, response) in enumerate(self.injected):
                if request.method == method and request.url.path.endswith(suffix):
                    del self.injected[i]
                    return response
            self.seen.append((request.method, request.url.path))
            return await self.inner.handle_async_request(request)

        return httpx.MockTransport(handler)


def edge_503() -> httpx.Response:
    return httpx.Response(503, text="Service Unavailable", headers={"content-type": "text/plain"})


def server_error(status: int = 503) -> httpx.Response:
    return httpx.Response(status, json={"detail": "the Agent Server failed"})


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[SubAgentGateway, FilesystemFakeHosts, FlakyEdge, httpx.Client]]:
    hosts = FilesystemFakeHosts(tmp_path / "hosts")
    edge = FlakyEdge(hosts)
    gateway = SubAgentGateway(
        hosts, {"echo": HostSpec(graph_id="echo")}, db_path=tmp_path / "gateway.sqlite", token=TOKEN,
        poll_interval=3600.0, unreachable_threshold=1, upstream_retry_delay=0.01,
        upstream_transport=edge.transport(), clock=FakeClock().time,
    )
    http = httpx.Client(base_url=gateway.start_embedded(), headers={"Authorization": f"Bearer {TOKEN}"}, timeout=20.0)
    try:
        yield gateway, hosts, edge, http
    finally:
        http.close()
        gateway.stop()
        hosts.close()


def launch(http: httpx.Client) -> tuple[str, httpx.Response]:
    tid = http.post("/threads", json={}).json()["thread_id"]
    return tid, http.post(f"/threads/{tid}/runs", json={"assistant_id": "echo", "input": {"messages": []}})


def runs_posted(edge: FlakyEdge) -> int:
    return sum(1 for method, path in edge.seen if method == "POST" and path.endswith("/runs"))


def test_an_edge_503_on_a_run_creation_is_retried(env) -> None:
    gateway, hosts, edge, http = env
    edge.fail("POST", "/runs", edge_503())
    tid, response = launch(http)
    assert response.status_code == 200, response.text
    assert edge.injected == [] and runs_posted(edge) == 1 and len(hosts.servers[tid].runs) == 1
    assert gateway.store.get_host(tid).status == "busy"


def test_a_5xx_from_the_agent_server_on_a_run_creation_is_not_retried(env) -> None:
    gateway, hosts, edge, http = env
    edge.fail("POST", "/runs", server_error(503), server_error(503))
    tid, response = launch(http)
    assert response.status_code == 502
    assert "HTTP 503" in response.json()["detail"]
    assert len(edge.injected) == 1  # one attempt only: the server may have started the run


def test_retries_are_bounded(env) -> None:
    gateway, hosts, edge, http = env
    edge.fail("POST", "/runs", edge_503(), edge_503(), edge_503(), edge_503())
    tid, response = launch(http)
    assert response.status_code == 502
    assert len(edge.injected) == 1 and runs_posted(edge) == 0  # 1 + upstream_retries attempts


def test_polls_are_retried_even_on_a_json_5xx(env) -> None:
    gateway, hosts, edge, http = env
    tid, response = launch(http)
    rid = response.json()["run_id"]
    edge.fail("GET", f"/runs/{rid}", server_error(502), edge_503())
    run = http.get(f"/threads/{tid}/runs/{rid}").json()
    assert run["status"] == "pending" and edge.injected == []
    assert gateway.store.get_host(tid).failures == 0 and gateway.store.get_host(tid).status == "busy"


def test_an_artifact_read_after_an_edge_503_is_retried(env) -> None:
    gateway, hosts, edge, http = env
    tid, response = launch(http)
    hosts.servers[tid].complete(response.json()["run_id"], "done")
    gateway.tick_sync()
    hosts.save(tid, tid, "report.md", "# findings")
    edge.fail("GET", "/report.md", edge_503())
    fetched = http.get(f"/artifacts/{tid}/files/report.md")
    assert fetched.status_code == 200 and fetched.content == b"# findings"
    assert edge.injected == []
