"""Gateway behaviour that only matters on real MicroVMs.

- every upstream request carries the host manager's edge headers (the
  DigitalOcean access token) next to the host token;
- tokens persisted in the store are handed back to the host manager when a
  gateway starts (MicroVM environments are immutable, so they cannot be
  re-keyed);
- a poll interval that would let busy hosts auto-pause is refused;
- at the team's MicroVM limit, an idle host whose results are cached is
  evicted to make room; if every host is busy the launch fails clearly.

Hosts are FakeAgentServers behind httpx.MockTransport, as in test_gateway.py.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fake_agent_server import FakeHosts, make_transport
from fake_client import FakeMicroVMClient
from langgraph_sdk import get_sync_client
from test_gateway import FakeClock

from langchain_digitalocean_microvm.agent_host import (
    HostSpec,
    MicroVMAgentHost,
    SubAgentGateway,
)
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import (
    HostCapacityError,
    MicroVMForbiddenError,
    SandboxError,
)
from langchain_digitalocean_microvm.models import MicroVMSize, parse_duration

TOKEN = "gw-token"
EDGE = {"Authorization": "Bearer do-access-token"}


class MicroVMLikeHosts(FakeHosts):
    """FakeHosts plus the MicroVMAgentHost extras: edge headers, remember_token, a capacity limit."""

    def __init__(self, *, limit: int | None = None, idle: float = 600.0) -> None:
        super().__init__()
        self.limit = limit
        self.idle_timeout_seconds = idle
        self.remembered: dict[str, str] = {}

    def edge_headers(self) -> dict[str, str]:
        return dict(EDGE)

    def remember_token(self, host_id: str, token: str) -> None:
        self.remembered[host_id] = token

    def get_or_create(self, *, host_id: str, spec: HostSpec):
        live = [h for h in self.servers if not self.servers[h].dead]
        if self.limit is not None and host_id not in self.servers and len(live) >= self.limit:
            raise HostCapacityError(f"The team is at its MicroVM limit ({len(live)} of {self.limit} MicroVMs).")
        return super().get_or_create(host_id=host_id, spec=spec)


def recording_transport(hosts: FakeHosts, seen: list[httpx.Request]) -> httpx.MockTransport:
    inner = make_transport(hosts)

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return await inner.handle_async_request(request)

    return httpx.MockTransport(handler)


def make_gateway(tmp_path: Path, hosts: FakeHosts, **kwargs) -> SubAgentGateway:
    seen = kwargs.pop("seen", [])
    return SubAgentGateway(
        hosts,
        {"echo": HostSpec(graph_id="echo")},
        db_path=tmp_path / "gateway.sqlite",
        token=TOKEN,
        poll_interval=kwargs.pop("poll_interval", 5.0),
        upstream_transport=recording_transport(hosts, seen),
        **kwargs,
    )


def launch(url: str, content: str = "ping") -> tuple[str, str]:
    client = get_sync_client(url=url, api_key=TOKEN, timeout=httpx.Timeout(20.0))
    tid = client.threads.create()["thread_id"]
    run = client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": content}]})
    return tid, run["run_id"]


def test_upstream_requests_carry_the_edge_token_and_the_host_token(tmp_path) -> None:
    hosts = MicroVMLikeHosts()
    seen: list[httpx.Request] = []
    gateway = make_gateway(tmp_path, hosts, seen=seen)
    url = gateway.start_embedded()
    try:
        tid, rid = launch(url)
        get_sync_client(url=url, api_key=TOKEN).runs.get(tid, rid)
    finally:
        gateway.stop()
    assert seen, "no upstream traffic recorded"
    for request in seen:
        assert request.headers["authorization"] == EDGE["Authorization"]
        assert request.headers["x-api-key"] == f"tok-{tid}"


def test_stored_tokens_are_handed_back_to_the_host_manager_on_start(tmp_path) -> None:
    first = MicroVMLikeHosts()
    gateway = make_gateway(tmp_path, first)
    url = gateway.start_embedded()
    try:
        tid, _ = launch(url)
    finally:
        gateway.stop()

    restarted = MicroVMLikeHosts()
    gateway = make_gateway(tmp_path, restarted)  # same SQLite file, new process' host manager
    gateway.store.close()
    assert restarted.remembered == {tid: f"tok-{tid}"}


def test_a_poll_interval_that_lets_busy_hosts_pause_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="too long"):
        make_gateway(tmp_path, MicroVMLikeHosts(idle=30.0), poll_interval=15.0)
    make_gateway(tmp_path, MicroVMLikeHosts(idle=30.0), poll_interval=10.0).store.close()


def test_at_the_limit_an_idle_cached_host_is_evicted_to_make_room(tmp_path) -> None:
    hosts = MicroVMLikeHosts(limit=2)
    clock = FakeClock()
    gateway = make_gateway(tmp_path, hosts, clock=clock.time)
    url = gateway.start_embedded()
    try:
        client = get_sync_client(url=url, api_key=TOKEN)
        t1, r1 = launch(url, "one")
        t2, r2 = launch(url, "two")
        hosts.servers[t1].complete(r1, "echo: one")
        hosts.servers[t2].complete(r2, "echo: two")
        gateway.tick_sync()  # cache both results; both hosts idle
        clock.advance(1.0)
        gateway.store.update_host(t2, last_active_at=clock.t)  # t2 was used more recently

        # Inside eviction_grace (60 s) both hosts keep their files: the launch is refused.
        tid = client.threads.create()["thread_id"]
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "three"}]})
        assert excinfo.value.response.status_code == 503
        assert "try again in about 59s" in excinfo.value.response.text
        assert set(hosts.servers) == {t1, t2}

        clock.advance(60.0)
        t3, _ = launch(url, "three")  # a third host: one of the idle ones must go

        assert set(hosts.servers) == {t2, t3}  # t1 was the least recently active
        row = gateway.store.get_host(t1)
        assert (row.status, row.retired_reason) == ("destroyed", "evicted")
        # The evicted task's result is still served from the cache.
        assert client.runs.get(t1, r1)["status"] == "success"
        assert client.threads.get(t1)["values"]["messages"][-1]["content"] == "echo: one"
    finally:
        gateway.stop()


def test_at_the_limit_with_every_host_busy_the_launch_fails_clearly(tmp_path) -> None:
    hosts = MicroVMLikeHosts(limit=2)
    gateway = make_gateway(tmp_path, hosts)
    url = gateway.start_embedded()
    try:
        launch(url, "one")
        launch(url, "two")
        client = get_sync_client(url=url, api_key=TOKEN)
        tid = client.threads.create()["thread_id"]
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": "three"}]})
        assert excinfo.value.response.status_code == 503
        assert "MicroVM limit" in excinfo.value.response.text and "in progress" in excinfo.value.response.text
        assert len(hosts.servers) == 2
    finally:
        gateway.stop()


# ---------------------------------------------------------------- host manager side


def _host(client: FakeMicroVMClient) -> MicroVMAgentHost:
    config = MicroVMConfig(transport="exec", api_token="t", access_token="a", agent_image="img", idle_timeout="10m")
    return MicroVMAgentHost(config, client=client, probe=lambda u, t, s: True)  # type: ignore[arg-type]


def test_a_403_at_the_limit_becomes_host_capacity_error() -> None:
    client = FakeMicroVMClient()
    for name in ("a", "b"):
        client.create(name=name, oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048))
    client.fail_create_with = MicroVMForbiddenError(403, "forbidden")
    with pytest.raises(HostCapacityError, match="2 of 2"):
        _host(client).get_or_create(host_id="h", spec=HostSpec(graph_id="echo"))


def test_a_403_below_the_limit_points_at_the_token_scope() -> None:
    client = FakeMicroVMClient()
    client.fail_create_with = MicroVMForbiddenError(403, "forbidden")
    with pytest.raises(SandboxError, match="microvm:create") as excinfo:
        _host(client).get_or_create(host_id="h", spec=HostSpec(graph_id="echo"))
    assert not isinstance(excinfo.value, HostCapacityError)


def test_idle_timeout_is_exposed_in_seconds_for_the_gateway_check() -> None:
    assert _host(FakeMicroVMClient()).idle_timeout_seconds == 600.0
    assert parse_duration("1h30m") == 5400.0
    assert parse_duration("5m0s") == 300.0  # the API's response form
    with pytest.raises(ValueError):
        parse_duration("ten minutes")


def test_a_deleted_microvm_answering_403_fails_its_task_instead_of_hanging(tmp_path) -> None:
    """Real MicroVM endpoints answer 403 'RBAC: access denied' once the VM is deleted, not a connection error."""
    hosts = MicroVMLikeHosts()
    gateway = make_gateway(tmp_path, hosts)
    url = gateway.start_embedded()
    try:
        tid, rid = launch(url)
        server = hosts.servers[tid]

        async def rbac_denied(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="RBAC: access denied")

        server.handle = rbac_denied  # type: ignore[method-assign]  # the edge still answers ...
        hosts.status.pop(tid)  # ... but the VM is gone from the inventory
        hosts.servers.pop(tid)
        hosts.servers[tid] = server  # keep routing to the 403 answer
        hosts.list_hosts = lambda graph_id=None: []  # type: ignore[method-assign]

        run = get_sync_client(url=url, api_key=TOKEN).runs.get(tid, rid)
        assert run["status"] == "error"
        assert "403" in run["error"]
    finally:
        gateway.stop()


def test_on_start_hosts_that_vanished_since_the_last_session_are_marked_destroyed(tmp_path) -> None:
    first = MicroVMLikeHosts()
    gateway = make_gateway(tmp_path, first)
    url = gateway.start_embedded()
    try:
        done_tid, done_rid = launch(url, "finished")
        busy_tid, _ = launch(url, "still running")
        first.servers[done_tid].complete(done_rid, "echo: finished")
        gateway.tick_sync()
    finally:
        gateway.stop()

    later = MicroVMLikeHosts()  # a new session: both VMs were deleted meanwhile
    gateway = make_gateway(tmp_path, later)
    try:
        assert gateway.store.get_host(done_tid).status == "destroyed"
        assert gateway.store.get_host(busy_tid).status == "destroyed"
        assert gateway.store.get_thread(busy_tid).run_status == "error"  # not "running" forever
        assert gateway.store.get_thread(done_tid).run_status == "success"  # cached result kept
    finally:
        gateway.store.close()


def test_concurrent_updates_on_one_thread_track_the_run_created_last(tmp_path) -> None:
    """Parallel update_async_task calls: the host keeps the last-created run (interrupt), so must the gateway."""
    import asyncio
    import threading

    hosts = MicroVMLikeHosts()
    gateway = make_gateway(tmp_path, hosts)
    url = gateway.start_embedded()
    try:
        tid, _ = launch(url)
        server = hosts.servers[tid]
        original = server.handle
        first_create = threading.Event()

        async def slow_first_response(request: httpx.Request) -> httpx.Response:
            response = await original(request)
            if request.method == "POST" and request.url.path.endswith("/runs") and not first_create.is_set():
                first_create.set()
                await asyncio.sleep(1.0)  # the first run's response arrives after the second's
            return response

        server.handle = slow_first_response  # type: ignore[method-assign]
        client = get_sync_client(url=url, api_key=TOKEN, timeout=httpx.Timeout(20.0))
        created: list[dict] = []

        def update(text: str) -> None:
            created.append(client.runs.create(tid, assistant_id="echo", input={"messages": [{"role": "user", "content": text}]},
                                              multitask_strategy="interrupt"))

        a = threading.Thread(target=update, args=("first",))
        a.start()
        first_create.wait(5)
        b = threading.Thread(target=update, args=("second",))
        b.start()
        a.join()
        b.join()

        latest_on_host = max(server.runs.values(), key=lambda r: list(server.runs).index(r["run_id"]))
        assert latest_on_host["status"] == "pending"  # the last-created run is the live one
        assert gateway.store.get_thread(tid).run_id == latest_on_host["run_id"]
    finally:
        gateway.stop()


def test_status_never_exposes_host_tokens_and_the_store_is_owner_only(tmp_path) -> None:
    hosts = MicroVMLikeHosts()
    gateway = make_gateway(tmp_path, hosts)
    url = gateway.start_embedded()
    try:
        tid, _ = launch(url)
        body = httpx.get(f"{url}/gateway/status", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10).json()
        assert [h["host_id"] for h in body["hosts"]] == [tid]
        assert all("token" not in h for h in body["hosts"])
        assert f"tok-{tid}" not in str(body)
        assert (tmp_path / "gateway.sqlite").stat().st_mode & 0o777 == 0o600
    finally:
        gateway.stop()
