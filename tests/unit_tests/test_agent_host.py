"""Tests for agent_host.MicroVMAgentHost on the (faked) MicroVMs API.

FakeMicroVMClient stands in for the API (creating -> running, endpoint
PENDING -> ACTIVE, unique names, `terminating` after delete) and a
scripted probe for GET /ok. No sleeping:
the host manager's `sleep`/`monotonic` hooks are driven by a fake clock.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm.agent_host import (
    HOST_MANAGED_TAG,
    HostHandle,
    HostSpec,
    HostStatus,
    MicroVMAgentHost,
    host_graph_tag,
    host_id_tag,
)
from langchain_digitalocean_microvm.client import MANAGED_TAG, microvm_name
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import (
    HostNotReadyError,
    SandboxError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.status import MicroVMStatus

IMAGE = "registry.digitalocean.com/team/microvm-subagent:1"


class ScriptedProbe:
    """Answers GET /ok from a script, then a default; records (url, token) per call."""

    def __init__(self, results: list[bool] | None = None, *, default: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self._results = list(results or [])
        self._default = default

    def __call__(self, url: str, token: str, timeout: float) -> bool:
        self.calls.append((url, token))
        if self._results:
            return self._results.pop(0)
        return self._default


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _tokens() -> Iterator[str]:
    n = 0
    while True:
        n += 1
        yield f"tok-{n}"


def _config(**overrides) -> MicroVMConfig:
    values = {
        "transport": "exec",
        "api_token": "control-token",
        "access_token": "access-token",
        "agent_image": IMAGE,
        "idle_timeout": "10m",
        "ready_timeout": 10.0,
    }
    values.update(overrides)
    return MicroVMConfig(**values)


def _make_host(client: FakeMicroVMClient, *, probe: ScriptedProbe | None = None, metrics=None,
               config: MicroVMConfig | None = None, tokens: Iterator[str] | None = None):
    probe = probe or ScriptedProbe()
    clock = FakeClock()
    host = MicroVMAgentHost(
        config or _config(),
        client=client,  # type: ignore[arg-type]  # duck-typed double
        probe=probe,
        metrics_fetcher=metrics,
        token_factory=(tokens or _tokens()).__next__,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        probe_interval=1.0,
    )
    return host, probe, clock


SPEC = HostSpec(graph_id="researcher", env={"DO_MODEL_ACCESS_KEY": "model-key"}, n_jobs_per_worker=2)


def test_fresh_host_is_created_with_its_whole_environment_then_probed_over_https() -> None:
    client = FakeMicroVMClient(stuck_transitional_polls=2)
    host, probe, _ = _make_host(client)

    handle = host.get_or_create(host_id="h1", spec=SPEC)

    [payload] = client.created
    assert payload["name"] == microvm_name("microvm-agenthost", "h1")
    assert payload["oci_ref"] == IMAGE
    assert payload["size"] == MicroVMSize(cpu=1, memory=2048)
    assert payload["http_port"] == 2024 and payload["http_protocol"] == "http"
    assert payload["idle_timeout"] == "10m"
    assert payload["environment"] == {
        "DO_MODEL_ACCESS_KEY": "model-key",
        "SUBAGENT_HOST_TOKEN": "tok-1",
        "N_JOBS_PER_WORKER": "2",
    }
    assert payload["tags"] == [HOST_MANAGED_TAG, host_id_tag("h1"), host_graph_tag("researcher")]
    assert isinstance(handle, HostHandle)
    assert handle.status is HostStatus.READY
    assert handle.url == f"https://{handle.microvm_id}.mkc1.microvm.ondigitalocean.com"
    assert handle.token == "tok-1"
    assert handle.booted_at  # MicroVM created_at
    assert probe.calls == [(handle.url, "tok-1")]
    assert MANAGED_TAG in client.get(handle.microvm_id).tags


def test_existing_host_with_known_token_is_reused() -> None:
    client = FakeMicroVMClient()
    host, probe, _ = _make_host(client)
    first = host.get_or_create(host_id="h2", spec=SPEC)

    second = host.get_or_create(host_id="h2", spec=SPEC)

    assert client.create_calls == 1
    assert second.microvm_id == first.microvm_id and second.token == first.token
    assert len(probe.calls) == 2


def test_a_host_whose_token_is_unknown_is_replaced_because_env_is_immutable() -> None:
    client = FakeMicroVMClient()
    old_host, _, _ = _make_host(client)
    original = old_host.get_or_create(host_id="h3", spec=SPEC)

    new_host, _, _ = _make_host(client, tokens=iter(["fresh-token"]))  # a new process: no tokens known
    replacement = new_host.get_or_create(host_id="h3", spec=SPEC)

    assert client.get(original.microvm_id).status is MicroVMStatus.TERMINATING
    assert replacement.microvm_id != original.microvm_id
    assert replacement.token == "fresh-token"
    assert client.created[-1]["environment"]["SUBAGENT_HOST_TOKEN"] == "fresh-token"


def test_remember_token_lets_a_new_process_reuse_and_resume_a_host() -> None:
    client = FakeMicroVMClient()
    old_host, _, _ = _make_host(client)
    original = old_host.get_or_create(host_id="h4", spec=SPEC)
    old_host.pause(host_id="h4")

    new_host, probe, _ = _make_host(client)
    with pytest.raises(SandboxError, match="remember_token"):
        new_host.resume(host_id="h4")
    new_host.remember_token("h4", original.token)
    resumed = new_host.resume(host_id="h4")

    assert resumed.microvm_id == original.microvm_id and resumed.token == original.token
    assert resumed.graph_id == "researcher"  # recovered from the graph tag
    assert client.create_calls == 1 and client.resume_calls == 1
    assert probe.calls[-1] == (resumed.url, original.token)


def test_paused_host_is_resumed_by_get_or_create() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client)
    handle = host.get_or_create(host_id="h5", spec=SPEC)
    host.pause(host_id="h5")
    assert client.get(handle.microvm_id).status is MicroVMStatus.PAUSED

    again = host.get_or_create(host_id="h5", spec=SPEC)

    assert again.microvm_id == handle.microvm_id
    assert client.resume_calls == 1 and client.create_calls == 1


def test_failed_host_is_deleted_and_recreated_with_a_new_token_by_get_or_create() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client)
    original = host.get_or_create(host_id="h6", spec=SPEC)
    client.force_status[original.microvm_id] = MicroVMStatus.FAILED

    recreated = host.get_or_create(host_id="h6", spec=SPEC)

    assert client.delete_calls == 1 and client.create_calls == 2
    assert recreated.microvm_id != original.microvm_id
    assert recreated.token == "tok-2"


def test_resume_refuses_to_recreate_a_failed_host() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client)
    handle = host.get_or_create(host_id="h7", spec=SPEC)
    client.force_status[handle.microvm_id] = MicroVMStatus.FAILED

    with pytest.raises(SandboxError, match="refusing to recreate"):
        host.resume(host_id="h7")
    assert client.create_calls == 1


def test_lost_create_race_replaces_the_winner_whose_token_we_cannot_know() -> None:
    client = FakeMicroVMClient(fail_next_create=True)
    host, _, _ = _make_host(client)

    handle = host.get_or_create(host_id="h8", spec=SPEC)

    assert client.create_calls == 2 and client.delete_calls == 1
    assert handle.token == "tok-1"
    assert [s.id for s in client.live()] == [handle.microvm_id]


def test_probe_timeout_raises_host_not_ready() -> None:
    client = FakeMicroVMClient()
    host, probe, clock = _make_host(client, probe=ScriptedProbe(default=False))

    with pytest.raises(HostNotReadyError, match="did not answer"):
        host.get_or_create(host_id="h9", spec=SPEC)
    assert clock.t >= 10.0 and len(probe.calls) > 5


def test_probe_eventually_succeeds_after_retries() -> None:
    client = FakeMicroVMClient()
    host, probe, _ = _make_host(client, probe=ScriptedProbe([False, False, True]))
    assert host.get_or_create(host_id="h10", spec=SPEC).status is HostStatus.READY
    assert len(probe.calls) == 3


def test_a_host_that_fails_to_boot_is_reported_with_the_reason() -> None:
    client = FakeMicroVMClient(stuck_transitional_polls=1)
    host, _, _ = _make_host(client)
    original_create = client.create

    def create_then_fail(**kwargs):
        state = original_create(**kwargs)
        client.force_status[state.id] = MicroVMStatus.FAILED
        return state

    client.create = create_then_fail  # type: ignore[method-assign]
    with pytest.raises(SandboxError):
        host.get_or_create(host_id="h11", spec=SPEC)


def test_pause_and_delete_by_host_id_and_not_found() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client)
    handle = host.get_or_create(host_id="h12", spec=SPEC)

    host.pause(host_id="h12")
    host.pause(host_id="h12")  # idempotent
    assert client.pause_calls == 1
    host.delete(host_id="h12")
    assert client.get(handle.microvm_id).status is MicroVMStatus.TERMINATING
    with pytest.raises(SandboxNotFoundError):
        host.pause(host_id="h12")
    with pytest.raises(SandboxNotFoundError):
        host.delete(host_id="never")


def test_list_hosts_filters_by_graph_and_reports_status_and_known_tokens() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client)
    a = host.get_or_create(host_id="a", spec=HostSpec(graph_id="echo"))
    host.get_or_create(host_id="b", spec=HostSpec(graph_id="researcher"))
    host.pause(host_id="b")

    stranger, _, _ = _make_host(client)
    everything = {h.host_id: h for h in stranger.list_hosts()}
    assert set(everything) == {"a", "b"}
    assert everything["a"].status is HostStatus.READY and everything["a"].url == a.url
    assert everything["b"].status is HostStatus.PAUSED
    assert everything["a"].token == ""  # not known to that process
    assert [h.host_id for h in host.list_hosts(graph_id="echo")] == ["a"]
    assert host.list_hosts(graph_id="echo")[0].token == a.token


def test_passthrough_env_is_copied_and_spec_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "from-supervisor")
    monkeypatch.setenv("LANGSMITH_PROJECT", "subagents")
    client = FakeMicroVMClient()
    config = _config(subagent_env_passthrough=("DO_MODEL_ACCESS_KEY", "LANGSMITH_PROJECT", "UNSET_VAR"))
    host, _, _ = _make_host(client, config=config)

    host.get_or_create(host_id="h13", spec=HostSpec(graph_id="g", env={"DO_MODEL_ACCESS_KEY": "scoped"}))

    env = client.created[0]["environment"]
    assert env["DO_MODEL_ACCESS_KEY"] == "scoped"
    assert env["LANGSMITH_PROJECT"] == "subagents"
    assert "UNSET_VAR" not in env
    assert env["SUBAGENT_HOST_TOKEN"] == "tok-1"


def test_spec_size_overrides_the_configured_size() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client)
    host.get_or_create(host_id="h14", spec=HostSpec(graph_id="g", size="2vcpu-4gb"))
    assert client.created[0]["size"] == MicroVMSize(cpu=2, memory=4096)


def test_is_idle_reads_queue_metrics_and_fails_closed() -> None:
    client = FakeMicroVMClient()
    responses: list[object] = [
        {"queue": {"n_pending": 0, "n_running": 0}},
        {"queue": {"n_pending": 0, "n_running": 1}},
        {"queue": {}},
        RuntimeError("connection refused"),
    ]

    def metrics(url: str, token: str, timeout: float) -> dict:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    host, _, _ = _make_host(client, metrics=metrics)
    handle = host.get_or_create(host_id="h15", spec=SPEC)

    assert host.is_idle(handle) is True
    assert host.is_idle(handle) is False  # a run is executing
    assert host.is_idle(handle) is True  # empty queue dict => zero pending/running
    assert host.is_idle(handle) is False  # unreachable => never "idle"


def test_edge_headers_carry_the_access_only_token_which_is_required() -> None:
    host, _, _ = _make_host(FakeMicroVMClient())
    assert host.edge_headers() == {"Authorization": "Bearer access-token"}
    with pytest.raises(SandboxError, match="DO_MICROVM_ACCESS_TOKEN"):  # never the control token instead
        _make_host(FakeMicroVMClient(), config=_config(access_token=None))


def test_default_probe_sends_both_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Response:
        status_code = 200

    def fake_get(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return Response()

    monkeypatch.setattr("langchain_digitalocean_microvm.agent_host.host.httpx.get", fake_get)
    client = FakeMicroVMClient()
    host = MicroVMAgentHost(_config(), client=client, token_factory=lambda: "host-token")  # type: ignore[arg-type]
    handle = host.get_or_create(host_id="h17", spec=SPEC)
    assert seen["url"] == f"{handle.url}/ok"
    assert seen["headers"] == {"Authorization": "Bearer access-token", "x-api-key": "host-token"}


def test_missing_agent_image_is_an_explicit_error() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client, config=_config(agent_image=None))
    with pytest.raises(SandboxError, match="image"):
        host.get_or_create(host_id="h18", spec=HostSpec(graph_id="g"))
    assert client.create_calls == 0


def test_configuration_errors_and_independence_from_the_sandbox_transport() -> None:
    with pytest.raises(SandboxError, match="DO_MICROVM_API_TOKEN"):
        MicroVMAgentHost(MicroVMConfig(transport="exec"))
    # DO_MICROVM_TRANSPORT=local keeps the sandbox demo offline; it must not stop real agent hosts.
    host, _, _ = _make_host(FakeMicroVMClient(), config=_config(transport="local"))
    assert host.get_or_create(host_id="h19", spec=SPEC).status is HostStatus.READY


def test_a_host_without_an_endpoint_fails_after_the_endpoint_wait_not_the_ready_timeout() -> None:
    """Booting gets `ready_timeout`; once running, a missing URL gets 60 s and a clear error."""
    class NoUrl(FakeMicroVMClient):
        def __init__(self) -> None:
            super().__init__()
            self.waits: list[tuple[str, float]] = []

        def get(self, microvm_id):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            return replace(super().get(microvm_id), endpoint=None)  # running, but no URL listed

        def wait_for_status(self, microvm_id, *, timeout, poll_interval=2.0):  # type: ignore[no-untyped-def]
            self.waits.append(("status", timeout))
            return super().wait_for_status(microvm_id, timeout=timeout, poll_interval=poll_interval)

        def wait_for_endpoint(self, microvm_id, *, timeout, poll_interval=2.0):  # type: ignore[no-untyped-def]
            self.waits.append(("endpoint", timeout))
            raise TimeoutError(f"MicroVM {microvm_id} endpoint not ACTIVE after {timeout}s (state running)")

    client = NoUrl()
    host, _, _ = _make_host(client)  # type: ignore[arg-type]
    with pytest.raises(HostNotReadyError, match="lists no HTTPS endpoint for it, which an agent host needs"):
        host.get_or_create(host_id="nourl", spec=SPEC)
    assert ("status", host.config.ready_timeout) in client.waits and ("endpoint", 60.0) in client.waits
