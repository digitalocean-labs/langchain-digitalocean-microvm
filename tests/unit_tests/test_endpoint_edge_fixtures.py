"""EndpointTransport's reading of real responses recorded through DigitalOcean's edge.

`fixtures/microvm_endpoint/` was written by `scripts/probe_endpoint.py --fixtures`: status, headers
(values kept only for generic ones) and the start of the body of responses seen through the edge.
`edge_*` are answers of the edge itself (no access token, a deleted MicroVM, a request over the
time budget); `server_*` came from the workload, which was the probe's own minimal server (its
error bodies say "sandbox-server", not this package's "microvm-sandbox"), so only their headers
are used here: they show which header marks a response as the workload's. If the edge changes
how it answers, re-record with the probe and these tests say what the transport would now do.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from endpoint_harness import ACCESS_TOKEN, API_TOKEN
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm import sandbox_server
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.transport.base import TransportCredentials
from langchain_digitalocean_microvm.transport.endpoint import (
    PROTOCOL,
    EndpointTransport,
    _from_workload,
    _ours,
)
from langchain_digitalocean_microvm.transport.errors import TransportError

FIXTURES = Path(__file__).parent / "fixtures" / "microvm_endpoint"
_DROP = {"content-length", "content-encoding", "transfer-encoding"}


def recorded(name: str, *, body: object | None = None) -> httpx.Response:
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    headers = {k: v for k, v in data["headers"].items() if k not in _DROP}
    content = body if body is not None else data["body"]
    raw = json.dumps(content).encode() if isinstance(content, dict) else str(content).encode()
    return httpx.Response(data["status"], headers=headers, content=raw)


def ours(name: str, body: dict) -> httpx.Response:
    """A recorded workload response's headers with this package's server body."""
    return recorded(name, body={**body, "source": sandbox_server.SERVER_NAME})


@pytest.mark.parametrize("name", ["edge_401_no_token", "edge_deleted", "edge_timeout"])
def test_the_edges_own_answers_are_not_from_the_workload(name: str) -> None:
    resp = recorded(name)
    assert not _from_workload(resp) and not _ours(resp)


@pytest.mark.parametrize("name", ["server_401_no_key", "server_exec_200", "server_get_404", "server_range_206",
                                  "server_healthz_200", "server_put_200"])
def test_workload_responses_keep_the_upstream_header_through_the_edge(name: str) -> None:
    assert _from_workload(recorded(name))


def _connected(answer: httpx.Response, calls: list[httpx.Request], root: Path) -> EndpointTransport:
    """A transport whose /v1/info finds a current server, and every other request gets `answer`."""
    info = ours("server_healthz_200", {"server": sandbox_server.SERVER_NAME, "protocol": PROTOCOL, "origin": "inline"})
    info.headers["content-type"] = "application/json"

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return info if request.url.path == "/v1/info" else answer

    client = FakeMicroVMClient()
    state = client.create(name="fx", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048), http_port=8080)
    client.get(state.id)
    t = EndpointTransport(client, http=httpx.Client(transport=httpx.MockTransport(handler)),  # type: ignore[arg-type]
                          url_for=lambda _s: "http://vm.test", sleep=lambda _s: None, settle_wait=0.0,
                          token_file=str(root / "token"), server_dir=str(root / "srv"))  # exec runs locally
    t.connect(host=state.id, credentials=TransportCredentials(api_token=API_TOKEN, access_token=ACCESS_TOKEN))
    return t


def test_a_deleted_microvm_is_gone_not_retried(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    t = _connected(recorded("edge_deleted"), calls, tmp_path)
    with pytest.raises(TransportError, match="is gone"):
        t.run("true")
    assert [r.url.path for r in calls].count("/v1/exec") == 1


def test_the_edges_401_is_a_rejected_access_token(tmp_path: Path) -> None:
    t = _connected(recorded("edge_401_no_token"), [], tmp_path)
    with pytest.raises(TransportError, match="rejected the access token"):
        t.get_file("/etc/hosts")


def test_the_edges_504_is_retried_then_reported(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    t = _connected(recorded("edge_timeout"), calls, tmp_path)
    with pytest.raises(TransportError, match="failed after 4 attempts: HTTP 504 from the edge"):
        t.run("true")
    exec_calls = [r for r in calls if r.url.path == "/v1/exec"]
    assert len(exec_calls) == 4
    assert len({r.headers["x-idempotency-key"] for r in exec_calls}) == 1  # one logical command


def test_a_401_from_the_server_rekeys_once(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    t = _connected(ours("server_401_wrong_key", {"error": "unauthorized"}), calls, tmp_path)
    exec_before = len(t._client.exec_calls)  # type: ignore[union-attr]
    with pytest.raises(TransportError, match="unauthorized"):
        t.run("true")
    assert len(t._client.exec_calls) == exec_before + 1  # type: ignore[union-attr]  # one re-key, then give up
    assert [r.url.path for r in calls].count("/v1/exec") == 2
