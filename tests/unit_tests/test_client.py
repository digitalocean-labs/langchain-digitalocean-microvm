"""Contract tests for langchain_digitalocean_microvm.client.MicroVMClient.

Responses come from tests/fixtures/microvm_api/, recorded from the real
MicroVMs API by scripts/probe_api.py and re-recorded with
`scripts/probe_api.py --fixtures` (identifying values are never in these
payloads; the token never is). Requests are intercepted with httpx.MockTransport, so
these tests pin both what we send (paths, verbs, JSON) and how we parse what
DigitalOcean actually returns.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from langchain_digitalocean_microvm.client import (
    MANAGED_TAG,
    MicroVMClient,
    _operation,
    api_calls,
    microvm_name,
)
from langchain_digitalocean_microvm.exceptions import (
    MicroVMAlreadyExistsError,
    MicroVMAPIError,
    MicroVMForbiddenError,
    MicroVMNotFoundError,
    MicroVMRateLimitError,
)
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.owner import owner_tag
from langchain_digitalocean_microvm.status import MicroVMStatus

FIXTURES = Path(__file__).parent / "fixtures" / "microvm_api"


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# Run-specific values (ids, names, timestamps) come from the fixtures, so
# re-recording them (scripts/probe_api.py --fixtures) only fails a test when
# the API's behaviour changed.
_CREATED = fixture("create_201")["microvm"]
VM_ID = _CREATED["id"]
VM_NAME = _CREATED["name"]
CLONE_NAME = fixture("create_from_checkpoint_201")["microvm"]["name"]
CHECKPOINT_NAME = fixture("checkpoint_create_201")["checkpoint"]["name"]
NODE_VERSION = fixture("exec_200")["stdout"]


class Recorder:
    """MockTransport handler: answers from a route table and records every request."""

    def __init__(self, routes: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response] | httpx.Response | list]):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get((request.method, request.url.path))
        if answer is None:
            return httpx.Response(404, json={"id": "not_found", "message": f"no route {request.method} {request.url.path}"})
        if isinstance(answer, list):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer(request) if callable(answer) else answer

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) if r.content else None for r in self.requests]


def make_client(recorder: Recorder, **kwargs: Any) -> MicroVMClient:
    return MicroVMClient("test-token", transport=httpx.MockTransport(recorder), sleep=lambda s: None, **kwargs)


# ---------------------------------------------------------------- create


def test_create_from_image_sends_the_documented_payload_and_parses_creating() -> None:
    rec = Recorder({("POST", "/v2/microvms"): httpx.Response(201, json=fixture("create_201"))})
    state = make_client(rec).create(
        name=VM_NAME,
        oci_ref="docker.io/mendhak/http-https-echo:latest",
        region="mkc1",
        size=MicroVMSize(cpu=1, memory=2048),
        http_port=8080,
        http_protocol="http",
        environment={"PROBE_ENV": "hello"},
        idle_timeout="30s",
        tags=["microvm-probe"],
    )
    body = rec.bodies()[0]
    assert body == {
        "name": VM_NAME,
        "tags": [MANAGED_TAG, owner_tag(), "microvm-probe"],
        "source": {"oci_ref": "docker.io/mendhak/http-https-echo:latest"},
        "region": "mkc1",
        "size": {"cpu": 1, "memory": 2048},
        "environment": {"PROBE_ENV": "hello"},
        "http_port": 8080,
        "http_protocol": "http",
        "auto_pause": {"idle_timeout": "30s"},
    }
    assert rec.requests[0].headers["authorization"] == "Bearer test-token"
    assert state.id == VM_ID
    assert state.status is MicroVMStatus.CREATING
    assert state.endpoint is None and state.url is None  # urls[0].status == PENDING, hostname ""
    assert state.http_port == 8080
    assert state.size == MicroVMSize(cpu=1, memory=2048, disk=50)
    # The API sends nanoseconds ("...35.720913606Z"); parsed trimmed to microseconds, in UTC.
    assert state.created_at == datetime.fromisoformat(_CREATED["created_at"][:26]).replace(tzinfo=UTC)


def test_create_from_checkpoint_sends_no_region_size_or_environment() -> None:
    rec = Recorder({("POST", "/v2/microvms"): httpx.Response(201, json=fixture("create_from_checkpoint_201"))})
    state = make_client(rec).create(name=CLONE_NAME, checkpoint_id="cp-1", tags=["x"], idle_timeout="30s")
    assert rec.bodies()[0] == {
        "name": CLONE_NAME,
        "tags": [MANAGED_TAG, owner_tag(), "x"],
        "source": {"checkpoint_id": "cp-1"},
        "auto_pause": {"idle_timeout": "30s"},
    }
    assert state.status is MicroVMStatus.RUNNING  # clones come back running


def test_create_validates_its_arguments_before_calling_the_api() -> None:
    client = make_client(Recorder({}))
    with pytest.raises(ValueError, match="exactly one"):
        client.create(name="x")
    with pytest.raises(ValueError, match="region and size"):
        client.create(name="x", oci_ref="img")
    with pytest.raises(ValueError, match="inherits its environment"):
        client.create(name="x", checkpoint_id="cp", environment={"A": "1"})


# ---------------------------------------------------------------- read


def test_get_running_exposes_the_https_endpoint() -> None:
    rec = Recorder({("GET", f"/v2/microvms/{VM_ID}"): httpx.Response(200, json=fixture("get_running_200"))})
    state = make_client(rec).get(VM_ID)
    assert state.status is MicroVMStatus.RUNNING
    assert state.endpoint == f"{VM_ID}.mkc1.microvm.ondigitalocean.com"
    assert state.url == f"https://{VM_ID}.mkc1.microvm.ondigitalocean.com"
    assert "microvm-probe" in state.tags
    assert state.raw["auto_pause"]["idle_timeout"] == "30s"


def test_list_follows_pagination_and_passes_filters() -> None:
    page1 = fixture("list_200")
    page1["links"] = {"pages": {"next": "https://api.digitalocean.com/v2/microvms?page=2"}}
    page2 = {"links": {}, "meta": {"total": 3}, "microvms": [fixture("get_running_200")["microvm"]]}
    rec = Recorder({("GET", "/v2/microvms"): [httpx.Response(200, json=page1), httpx.Response(200, json=page2)]})
    states = make_client(rec).list(name=VM_NAME, tag="microvm-probe")
    assert len(states) == len(page1["microvms"]) + 1
    first, second = (dict(r.url.params) for r in rec.requests)
    assert first == {"name": VM_NAME, "tag_name": "microvm-probe", "page": "1", "per_page": "200"}
    assert second["page"] == "2"


def test_get_by_name_ignores_a_terminating_leftover() -> None:
    gone = fixture("get_terminating_200")["microvm"]
    rec = Recorder({("GET", "/v2/microvms"): httpx.Response(200, json={"links": {}, "microvms": [gone]})})
    assert make_client(rec).get_by_name(gone["name"]) is None
    live = fixture("get_running_200")["microvm"]
    rec = Recorder({("GET", "/v2/microvms"): httpx.Response(200, json={"links": {}, "microvms": [gone, live]})})
    assert make_client(rec).get_by_name(live["name"]).id == live["id"]


def test_options_returns_limits_and_sizes() -> None:
    rec = Recorder({("GET", "/v2/microvms/options"): httpx.Response(200, json=fixture("options_200"))})
    options = make_client(rec).options()
    limit = options["account_limits"]["max_total_count"]
    assert isinstance(limit, int) and limit > 0  # the team's own value; it changes with the team
    assert options["default_region"] == "mkc1"


# ---------------------------------------------------------------- lifecycle


def test_pause_and_resume_refetch_because_their_response_drops_tags() -> None:
    paused = fixture("pause_200")
    assert paused["microvm"].get("tags", []) == []  # the real API quirk this test guards against (no tags)
    rec = Recorder({
        ("POST", f"/v2/microvms/{VM_ID}/pause"): httpx.Response(200, json=paused),
        ("POST", f"/v2/microvms/{VM_ID}/resume"): httpx.Response(200, json=fixture("resume_200")),
        ("GET", f"/v2/microvms/{VM_ID}"): [httpx.Response(200, json=fixture("get_paused_200")),
                                         httpx.Response(200, json=fixture("get_running_200"))],
    })
    client = make_client(rec)
    state = client.pause(VM_ID)
    assert state.status is MicroVMStatus.PAUSED and "microvm-probe" in state.tags
    state = client.resume(VM_ID)
    assert state.status is MicroVMStatus.RUNNING and "microvm-probe" in state.tags
    assert [(r.method, r.url.path.rsplit("/", 1)[-1]) for r in rec.requests] == [
        ("POST", "pause"), ("GET", VM_ID), ("POST", "resume"), ("GET", VM_ID)]


def test_pause_waits_out_a_stale_running_read() -> None:
    """A GET right after POST .../pause can still read running (e.g. with two pauses in parallel), so pause polls."""
    rec = Recorder({
        ("POST", f"/v2/microvms/{VM_ID}/pause"): httpx.Response(200, json=fixture("pause_200")),
        ("GET", f"/v2/microvms/{VM_ID}"): [httpx.Response(200, json=fixture("get_running_200")),
                                         httpx.Response(200, json=fixture("get_running_200")),
                                         httpx.Response(200, json=fixture("get_paused_200"))],
    })
    state = make_client(rec).pause(VM_ID)
    assert state.status is MicroVMStatus.PAUSED
    assert [r.method for r in rec.requests] == ["POST", "GET", "GET", "GET"]


def test_pause_is_sent_again_when_the_first_one_had_no_effect(monkeypatch) -> None:
    """A pause right after an auto-resume can be accepted (200) and ignored: the MicroVM keeps running."""
    import langchain_digitalocean_microvm.client as client_module

    monkeypatch.setattr(client_module, "_REPAUSE_AFTER", 0.01)
    rec = Recorder({
        ("POST", f"/v2/microvms/{VM_ID}/pause"): httpx.Response(200, json=fixture("pause_200")),
        ("GET", f"/v2/microvms/{VM_ID}"): lambda request: httpx.Response(
            200, json=fixture("get_running_200" if pause_count(rec) < 2 else "get_paused_200")),
    })
    state = make_client(rec).pause(VM_ID)
    assert state.status is MicroVMStatus.PAUSED
    assert pause_count(rec) == 2


def pause_count(rec: Recorder) -> int:
    return sum(r.method == "POST" and r.url.path.endswith("/pause") for r in rec.requests)


def test_wait_for_endpoint_polls_until_running_and_active() -> None:
    rec = Recorder({("GET", f"/v2/microvms/{VM_ID}"): [httpx.Response(200, json=fixture("get_creating_200")),
                                                     httpx.Response(200, json=fixture("get_running_200"))]})
    state = make_client(rec).wait_for_endpoint(VM_ID, timeout=10)
    assert state.endpoint and len(rec.requests) == 2


def test_delete_accepts_204() -> None:
    rec = Recorder({("DELETE", f"/v2/microvms/{VM_ID}"): httpx.Response(204)})
    make_client(rec).delete(VM_ID)
    assert rec.requests[0].method == "DELETE"


def test_exec_sends_argv_and_reports_a_failing_command_as_a_result() -> None:
    rec = Recorder({("POST", f"/v2/microvms/{VM_ID}/exec"): httpx.Response(200, json=fixture("exec_exit_3_200"))})
    result = make_client(rec).exec(VM_ID, ["sh", "-c", "exit 3"], cwd="/tmp")
    assert rec.bodies()[0] == {"argv": ["sh", "-c", "exit 3"], "cwd": "/tmp"}
    assert (result.exit_code, result.stdout, result.stderr, result.truncated) == (3, "out\nhello\n", "err\n", False)
    with pytest.raises(ValueError):
        make_client(rec).exec(VM_ID, [])


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_create_then_wait_until_available() -> None:
    cp = fixture("checkpoint_create_201")["checkpoint"]
    creating = {"checkpoint": {**cp, "status": "CHECKPOINT_CREATING"}}
    rec = Recorder({
        ("POST", f"/v2/microvms/{VM_ID}/checkpoints"): httpx.Response(201, json={"checkpoint": cp}),
        ("GET", f"/v2/microvms/checkpoints/{cp['id']}"): [httpx.Response(200, json=creating),
                                                        httpx.Response(200, json=fixture("checkpoint_available_200"))],
    })
    client = make_client(rec)
    created = client.create_checkpoint(VM_ID, name=CHECKPOINT_NAME)
    assert rec.bodies()[0] == {"name": CHECKPOINT_NAME}
    assert created.status == "CHECKPOINT_CREATING" and created.size == MicroVMSize(cpu=1, memory=2048, disk=50)
    ready = client.wait_for_checkpoint(created.id, timeout=10)
    assert ready.available and ready.microvm_id == VM_ID


# ---------------------------------------------------------------- errors and retries


@pytest.mark.parametrize(
    ("status", "error"),
    [(404, MicroVMNotFoundError), (409, MicroVMAlreadyExistsError), (403, MicroVMForbiddenError), (400, MicroVMAPIError)],
)
def test_error_statuses_map_to_exceptions(status: int, error: type[Exception]) -> None:
    rec = Recorder({("GET", f"/v2/microvms/{VM_ID}"): httpx.Response(status, json={"id": "x", "message": "nope"})})
    with pytest.raises(error, match="nope"):
        make_client(rec).get(VM_ID)


def test_forbidden_explains_scope_or_team_limit() -> None:
    rec = Recorder({("POST", "/v2/microvms"): httpx.Response(403, json={"id": "forbidden", "message": "limit"})})
    with pytest.raises(MicroVMForbiddenError, match="max_total_count|options"):
        make_client(rec).create(name="x", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048))


def test_rate_limits_are_retried_with_backoff_then_surface() -> None:
    sleeps: list[float] = []
    limited = httpx.Response(429, json={"id": "too_many_requests", "message": "60 commands per minute"})
    ok = httpx.Response(200, json=fixture("exec_200"))
    rec = Recorder({("POST", f"/v2/microvms/{VM_ID}/exec"): [limited, limited, ok]})
    client = MicroVMClient("t", transport=httpx.MockTransport(rec), sleep=sleeps.append)
    assert client.exec(VM_ID, ["node", "-v"]).stdout == NODE_VERSION
    assert sleeps == [1.0, 2.0]

    rec = Recorder({("POST", f"/v2/microvms/{VM_ID}/exec"): [limited]})
    client = MicroVMClient("t", transport=httpx.MockTransport(rec), sleep=lambda s: None, max_retries=2)
    with pytest.raises(MicroVMRateLimitError):
        client.exec(VM_ID, ["true"])
    assert len(rec.requests) == 3


def test_every_request_is_counted_by_operation_including_retries() -> None:
    limited = httpx.Response(429, json={"id": "too_many_requests", "message": "60 commands per minute"})
    rec = Recorder({
        ("POST", f"/v2/microvms/{VM_ID}/exec"): [limited, httpx.Response(200, json=fixture("exec_200"))],
        ("GET", f"/v2/microvms/{VM_ID}"): httpx.Response(200, json=fixture("get_running_200")),
    })
    before = api_calls()
    client = make_client(rec)
    client.exec(VM_ID, ["true"])
    client.get(VM_ID)
    after = api_calls()
    delta = {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}
    assert delta == {"exec": 2, "retry:429": 1, "get": 1}  # the rate limit counts every attempt


@pytest.mark.parametrize(("method", "path", "operation"), [
    ("POST", "/microvms", "create"),
    ("GET", "/microvms", "list"),
    ("GET", f"/microvms/{VM_ID}", "get"),
    ("DELETE", f"/microvms/{VM_ID}", "delete"),
    ("POST", f"/microvms/{VM_ID}/exec", "exec"),
    ("POST", f"/microvms/{VM_ID}/pause", "pause"),
    ("POST", f"/microvms/{VM_ID}/resume", "resume"),
    ("GET", "/microvms/options", "options"),
    ("POST", f"/microvms/{VM_ID}/checkpoints", "create checkpoint"),
    ("GET", "/microvms/checkpoints", "list checkpoints"),
    ("GET", "/microvms/checkpoints/cp-1", "get checkpoint"),
    ("DELETE", "/microvms/checkpoints/cp-1", "delete checkpoint"),
])
def test_operation_names(method: str, path: str, operation: str) -> None:
    assert _operation(method, path) == operation


# ---------------------------------------------------------------- names


def test_microvm_name_is_deterministic_valid_and_collision_free() -> None:
    assert microvm_name("microvm-agenthost", "6f1c2a1e-0d7b-4c39-9a3e-1b2c3d4e5f60") == (
        "microvm-agenthost-6f1c2a1e-0d7b-4c39-9a3e-1b2c3d4e5f60"
    )
    weird = microvm_name("microvm-sandbox", "My Sandbox/Notes")
    assert weird == microvm_name("microvm-sandbox", "My Sandbox/Notes")
    assert weird != microvm_name("microvm-sandbox", "my-sandbox-notes")  # sanitized twin gets a hash suffix
    long_name = microvm_name("microvm-sandbox", "x" * 200)
    for name in (weird, long_name):
        assert 1 <= len(name) <= 63 and all(c.islower() or c.isdigit() or c == "-" for c in name)


def test_size_parsing_and_validation() -> None:
    assert MicroVMSize.parse("2vcpu-4gb") == MicroVMSize(cpu=2, memory=4096)
    assert MicroVMSize.parse("1vcpu-2gb").slug == "1vcpu-2gb"
    with pytest.raises(ValueError, match="Unsupported"):
        MicroVMSize.parse("4vcpu-8gb")
    with pytest.raises(ValueError, match="look like"):
        MicroVMSize.parse("mv-1vcpu-2gb")
