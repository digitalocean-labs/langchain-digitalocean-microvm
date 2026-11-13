"""EndpointTransport against the real sandbox server, run locally behind a fake edge.

The server is sandbox_server.py as a local process; FakeMicroVMClient runs the
transport's exec calls (re-key, bootstrap) as local subprocesses, with every
path under a temporary directory. FakeEdge (endpoint_harness.py) re-creates
the edge's timeout-and-retry behaviour at a smaller time scale, so the
"a command never runs twice" property is tested without a MicroVM.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from endpoint_harness import (
    ACCESS_TOKEN,
    API_TOKEN,
    FakeEdge,
    free_port,
    start_server,
    stop,
)
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.exceptions import SandboxError
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.transport.base import TransportCredentials
from langchain_digitalocean_microvm.transport.endpoint import (
    MAX_OUTPUT,
    PROTOCOL,
    TIMEOUT_EXIT_CODE,
    EndpointTransport,
    sandbox_token,
    server_source,
)
from langchain_digitalocean_microvm.transport.errors import (
    TransportError,
    TransportFileNotFoundError,
    TransportIsADirectoryError,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")

CREDS = TransportCredentials(api_token=API_TOKEN, access_token=ACCESS_TOKEN)


@pytest.fixture
def vm() -> tuple[FakeMicroVMClient, str]:
    client = FakeMicroVMClient()
    state = client.create(name="ep", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048), http_port=8080)
    client.get(state.id)  # settle to running
    return client, state.id


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    """A server with NO token yet: the transport has to re-key it over exec, as for a fresh MicroVM."""
    port = free_port()
    proc = start_server(tmp_path, port)
    try:
        yield tmp_path, port
    finally:
        stop(proc)


@pytest.fixture
def edge(server) -> Iterator[FakeEdge]:
    _root, port = server
    fake = FakeEdge(f"http://127.0.0.1:{port}", budget=1.0)
    try:
        yield fake
    finally:
        fake.close()


def _transport(client: FakeMicroVMClient, url: str, root: Path, **kwargs) -> EndpointTransport:
    defaults = dict(url_for=lambda _state: url, token_file=str(root / "token"), server_dir=str(root / "srv"),
                    sync_budget=0.4, ready_timeout=5.0, settle_wait=0.5, sleep=lambda s: time.sleep(min(s, 0.05)))
    return EndpointTransport(client, **{**defaults, **kwargs})  # type: ignore[arg-type]  # duck-typed double


@pytest.fixture
def transport(vm, server, edge) -> Iterator[EndpointTransport]:
    client, microvm_id = vm
    root, _port = server
    t = _transport(client, edge.url, root)
    t.connect(host=microvm_id, credentials=CREDS)
    try:
        yield t
    finally:
        t.close()


# ---------------------------------------------------------------- connect


def test_connect_rekeys_a_server_without_the_token(vm, server, edge) -> None:
    client, microvm_id = vm
    root, _ = server
    t = _transport(client, edge.url, root)
    t.connect(host=microvm_id, credentials=CREDS)
    assert not t.uses_fallback
    assert (root / "token").read_text() == sandbox_token(API_TOKEN, microvm_id)
    assert len(client.exec_calls) == 1  # one exec: the re-key; everything else goes through the endpoint
    t.run("true")
    assert len(client.exec_calls) == 1


def test_sandbox_token_is_per_microvm_and_hides_the_control_token() -> None:
    a, b = sandbox_token(API_TOKEN, "vm-a"), sandbox_token(API_TOKEN, "vm-b")
    assert a != b and len(a) == 64 and API_TOKEN not in a
    assert sandbox_token("other-token", "vm-a") != a


def test_connect_requires_an_access_token(vm, server, edge) -> None:
    client, microvm_id = vm
    t = _transport(client, edge.url, server[0])
    with pytest.raises(TransportError, match="DO_MICROVM_ACCESS_TOKEN"):
        t.connect(host=microvm_id, credentials=TransportCredentials(api_token=API_TOKEN))


def test_a_rejected_access_token_is_an_error_not_a_fallback(vm, server, edge) -> None:
    client, microvm_id = vm
    t = _transport(client, edge.url, server[0])
    with pytest.raises(TransportError, match="rejected the access token"):
        t.connect(host=microvm_id, credentials=TransportCredentials(api_token=API_TOKEN, access_token="wrong"))


def test_a_clone_is_rekeyed_even_if_its_endpoint_is_not_ready_at_first(vm, server, edge) -> None:
    """A checkpoint clone runs the template's server with the template's token, and its endpoint
    can answer edge errors for a moment: wait those out, re-key, and do not fall back to exec."""
    client, microvm_id = vm
    root, _ = server
    (root / "token").write_text(sandbox_token(API_TOKEN, "the-template-build-vm"))
    edge.unavailable_next = 3
    t = _transport(client, edge.url, root)
    t.connect(host=microvm_id, credentials=CREDS)
    assert not t.uses_fallback
    assert (root / "token").read_text() == sandbox_token(API_TOKEN, microvm_id)
    assert [argv[2][:5] for _id, argv in client.exec_calls] == ["umask"]  # the re-key; no bootstrap attempt


class _SlowEndpoint(FakeMicroVMClient):
    """A clone whose URL is listed only after `endpoint_delay`; records the order of events."""

    def __init__(self, *, endpoint_delay: float = 0.3) -> None:
        super().__init__()
        self.endpoint_delay = endpoint_delay
        self.listed = False
        self.events: list[str] = []

    def get(self, microvm_id):  # type: ignore[no-untyped-def]
        state = super().get(microvm_id)
        return state if self.listed else replace(state, endpoint=None)

    def wait_for_endpoint(self, microvm_id, *, timeout, poll_interval=2.0):  # type: ignore[no-untyped-def]
        time.sleep(self.endpoint_delay)
        self.listed = True
        self.events.append("endpoint listed")
        return self.get(microvm_id)

    def exec(self, microvm_id, argv, *, cwd=None):  # type: ignore[no-untyped-def]
        self.events.append("exec")
        return super().exec(microvm_id, argv, cwd=cwd)


def _clone_of_template(client: FakeMicroVMClient, root: Path) -> str:
    """A running MicroVM whose server holds the token of the VM its template was built from."""
    state = client.create(name="clone", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048), http_port=8080)
    client.get(state.id)
    (root / "token").write_text(sandbox_token(API_TOKEN, "the-template-build-vm"))
    return state.id


def test_a_clone_is_rekeyed_only_after_its_server_has_answered(server, edge) -> None:
    """An exec sent to a fresh clone before its workload answers slows the clone's start, so the
    re-key waits for the server's 401 (the one exec a clone needs)."""
    root, _ = server
    client = _SlowEndpoint(endpoint_delay=0.1)
    microvm_id = _clone_of_template(client, root)
    t = _transport(client, edge.url, root)
    t.connect(host=microvm_id, credentials=CREDS)
    assert client.events == ["endpoint listed", "exec"]
    assert (root / "token").read_text() == sandbox_token(API_TOKEN, microvm_id) and not t.uses_fallback
    t.close()


def test_a_server_that_already_has_the_token_needs_no_exec(vm, server, edge) -> None:
    """Resume and reconnect: the token file already holds this MicroVM's token."""
    client, microvm_id = vm
    root, _ = server
    (root / "token").write_text(sandbox_token(API_TOKEN, microvm_id))
    t = _transport(client, edge.url, root)
    t.connect(host=microvm_id, credentials=CREDS)
    assert client.exec_calls == [] and not t.uses_fallback
    t.close()


def test_logs_name_neither_the_microvm_nor_a_token(vm, server, edge, caplog) -> None:
    """Re-key, a page shrink and a fallback all log; none of them names the MicroVM or a token."""
    client, microvm_id = vm
    root, _ = server
    (root / "token").write_text(sandbox_token(API_TOKEN, "the-template-build-vm"))
    caplog.set_level(logging.DEBUG, logger="langchain_digitalocean_microvm")
    t = _transport(client, edge.url, root, download_chunk=4096, min_download_chunk=1024)
    t.connect(host=microvm_id, credentials=CREDS)
    blob = root / "blob.bin"
    blob.write_bytes(b"x" * 8192)
    edge.cut_responses_over = 3000
    assert t.get_file(str(blob)) == b"x" * 8192
    edge.cut_responses_over = None
    t.close()
    strict = _transport(client, edge.url, root, bootstrap=False)
    (root / "token").write_text("someone-else's")
    strict._rekey = lambda: None  # type: ignore[method-assign]  # keep the wrong token: forces the fallback
    strict.connect(host=microvm_id, credentials=CREDS)
    assert strict.uses_fallback
    assert "reading in 2048-byte pages" in caplog.text and "using the exec API instead" in caplog.text
    for secret in (microvm_id, sandbox_token(API_TOKEN, microvm_id), API_TOKEN, ACCESS_TOKEN):
        assert secret not in caplog.text


# ---------------------------------------------------------------- commands


def test_run_returns_output_and_exit_code(transport) -> None:
    result = transport.run("echo out; echo err >&2; exit 3")
    assert (result.stdout, result.stderr, result.exit_code) == ("out\n", "err\n", 3)


def test_a_command_longer_than_the_budget_becomes_a_polled_job(transport) -> None:
    started = time.monotonic()
    result = transport.run("sleep 1.5; echo late")
    assert result.stdout == "late\n" and result.exit_code == 0
    assert time.monotonic() - started >= 1.5


def test_timeout_kills_the_command_and_its_children(transport, tmp_path: Path) -> None:
    marker = tmp_path / "child-survived"
    result = transport.run(f"(sleep 2; touch {marker}) & sleep 30", timeout=1)
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert "[timed out after 1s]" in result.stderr
    time.sleep(2.5)
    assert not marker.exists()  # the whole process group was killed


def test_an_edge_retry_does_not_run_the_command_twice(vm, server, edge, tmp_path: Path) -> None:
    """The edge re-sends a request whose response has not completed within its budget; the
    server attaches the re-sent request to the same job (same idempotency key)."""
    client, microvm_id = vm
    root, _ = server
    counter = tmp_path / "runs"
    t = _transport(client, edge.url, root, sync_budget=2.0)  # longer than the edge's 1 s: forces re-sends
    t.connect(host=microvm_id, credentials=CREDS)
    before = edge.forwarded
    result = t.run(f"echo run >> {counter}; sleep 1.5; echo done")
    assert result.stdout == "done\n"
    assert edge.forwarded - before >= 2  # the edge did re-send
    assert counter.read_text() == "run\n"  # ... and the command still ran once


def test_a_lost_response_is_retried_with_the_same_key(transport, edge, tmp_path: Path) -> None:
    counter = tmp_path / "runs"
    edge.fail_next = 1  # the server ran it, the client gets a 502
    result = transport.run(f"echo run >> {counter}; echo ok")
    assert result.stdout == "ok\n"
    assert counter.read_text() == "run\n"


def test_large_output_is_read_in_pages(transport) -> None:
    result = transport.run("head -c 300000 /dev/zero | tr '\\0' a; echo; echo err >&2")  # over the inline limit
    assert result.stdout == "a" * 300000 + "\n" and result.stderr == "err\n" and result.exit_code == 0


def test_output_beyond_4_mib_is_truncated(transport) -> None:
    result = transport.run("head -c 5000000 /dev/zero | tr '\\0' b; echo tail >&2")
    assert len(result.stdout) == MAX_OUTPUT and set(result.stdout) == {"b"}
    assert result.stderr.endswith("[output truncated at 4 MiB]")


def test_backend_execute_through_the_endpoint(transport) -> None:
    backend = MicroVMSandbox(microvm_id="vm", transport=transport)
    response = backend.execute("printf 'a\\nb\\n' | wc -l")
    assert response.output.strip() == "2" and response.exit_code == 0


# ---------------------------------------------------------------- files


def test_files_round_trip_in_chunks(vm, server, edge, tmp_path: Path) -> None:
    client, microvm_id = vm
    t = _transport(client, edge.url, server[0], upload_chunk=1000, download_chunk=700)
    t.connect(host=microvm_id, credentials=CREDS)
    data = bytes(range(256)) * 20  # 5120 bytes: 6 upload chunks, 8 download ranges
    path = str(tmp_path / "deep" / "dir" / "blob.bin")
    t.put_file(path, data)
    assert Path(path).read_bytes() == data
    assert t.get_file(path) == data
    t.put_file(path, b"short")  # overwrite truncates
    assert t.get_file(path) == b"short"
    t.put_file(path, b"")
    assert t.get_file(path) == b""


def test_downloads_shrink_their_pages_when_a_response_is_cut(vm, server, edge, tmp_path: Path) -> None:
    """On a slow link a large page does not arrive within the edge's budget: halve the page and go on."""
    client, microvm_id = vm
    t = _transport(client, edge.url, server[0], download_chunk=8192, min_download_chunk=1024)
    t.connect(host=microvm_id, credentials=CREDS)
    data = bytes(range(256)) * 80  # 20480 bytes
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    edge.cut_responses_over = 3000
    assert t.get_file(str(path)) == data
    assert t._download_chunk == 2048  # remembered for later reads
    edge.cut_responses_over = 500  # even the smallest page does not get through
    with pytest.raises(TransportError, match="failed after"):
        t.get_file(str(path))


def test_file_errors_map_to_transport_errors(transport, tmp_path: Path) -> None:
    with pytest.raises(TransportFileNotFoundError):
        transport.get_file(str(tmp_path / "missing.txt"))
    with pytest.raises(TransportIsADirectoryError):
        transport.get_file(str(tmp_path))
    with pytest.raises(TransportIsADirectoryError):
        transport.put_file(str(tmp_path), b"x")
    (tmp_path / "plain").write_text("x")
    with pytest.raises(TransportFileNotFoundError):  # parent is a file
        transport.put_file(str(tmp_path / "plain" / "child.txt"), b"x")


def test_a_deleted_microvm_is_reported_as_gone(transport, edge) -> None:
    edge.deleted = True
    with pytest.raises(TransportError, match="is gone"):
        transport.run("true")


# ---------------------------------------------------------------- bootstrap and fallback


@pytest.fixture
def bare_edge(tmp_path: Path) -> Iterator[tuple[FakeEdge, int, Path]]:
    """An edge in front of a port nothing listens on yet (an image without the server)."""
    port = free_port()
    fake = FakeEdge(f"http://127.0.0.1:{port}", budget=2.0)
    try:
        yield fake, port, tmp_path
    finally:
        fake.close()
        subprocess.run(["pkill", "-f", str(tmp_path / "srv" / "sandbox_server.py")], check=False)


def test_bootstrap_installs_and_starts_the_server(vm, bare_edge) -> None:
    client, microvm_id = vm
    edge, port, root = bare_edge
    t = _transport(client, edge.url, root)
    t.connect(host=microvm_id, port=port, credentials=CREDS)
    try:
        assert not t.uses_fallback
        assert (root / "srv" / "sandbox_server.py").read_text() == server_source()
        assert t.run("echo hi").stdout == "hi\n"
    finally:
        t.close()


def test_without_bootstrap_the_transport_falls_back_to_exec(vm, bare_edge) -> None:
    client, microvm_id = vm
    edge, port, root = bare_edge
    t = _transport(client, edge.url, root, bootstrap=False, ready_timeout=0.5)
    t.connect(host=microvm_id, port=port, credentials=CREDS)
    assert t.uses_fallback
    assert t.run("echo via-exec").stdout == "via-exec\n"


def test_a_port_taken_by_the_image_falls_back_quickly(vm, tmp_path: Path) -> None:
    """The old sandbox image's health server answers 200 'ok' to everything on the HTTP port."""
    class Ok(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "3")
            self.end_headers()
            self.wfile.write(b"ok\n")

        def log_message(self, *args) -> None:  # noqa: ANN002
            pass

    health = ThreadingHTTPServer(("0.0.0.0", 0), Ok)  # noqa: S104 - bound like the image's server
    port = health.server_address[1]
    threading.Thread(target=health.serve_forever, daemon=True).start()
    edge = FakeEdge(f"http://127.0.0.1:{port}")
    client, microvm_id = vm
    try:
        t = _transport(client, edge.url, tmp_path, ready_timeout=20.0)
        started = time.monotonic()
        t.connect(host=microvm_id, port=port, credentials=CREDS)
        assert t.uses_fallback
        assert time.monotonic() - started < 5  # did not wait out ready_timeout
    finally:
        edge.close()
        health.shutdown()


def _old_server_source() -> str:
    source = server_source()
    assert f"PROTOCOL = {PROTOCOL}" in source
    return source.replace(f"PROTOCOL = {PROTOCOL}", f"PROTOCOL = {PROTOCOL - 1}", 1)


def test_an_outdated_installed_server_is_replaced(vm, bare_edge) -> None:
    """A server this transport installed over exec (an older package version) is installed again
    (two exec calls: re-key, replace)."""
    client, microvm_id = vm
    edge, port, root = bare_edge
    (root / "srv").mkdir()
    old_file = root / "srv" / "sandbox_server.py"
    old_file.write_text(_old_server_source())
    old = subprocess.Popen([sys.executable, str(old_file), "--port", str(port), "--token-file", str(root / "token"),
                            "--job-root", str(root / "jobs"), "--host", "127.0.0.1"])
    try:
        t = _transport(client, edge.url, root, ready_timeout=10.0)
        t.connect(host=microvm_id, port=port, credentials=CREDS)
        assert not t.uses_fallback
        assert t._server_info["protocol"] == PROTOCOL and t._server_info["origin"] == "file"
        assert len(client.exec_calls) == 2
        assert old.wait(timeout=5) is not None  # the old process was stopped
        assert t.run("echo new").stdout == "new\n"
        t.close()
    finally:
        old.kill()


def test_an_outdated_image_server_is_reported_not_replaced(vm, tmp_path: Path) -> None:
    """The image's own server (run as `python3 -c`) cannot be replaced from here: rebuild the image."""
    client, microvm_id = vm
    port = free_port()
    proc = start_server(tmp_path, port, inline_source=_old_server_source())
    edge = FakeEdge(f"http://127.0.0.1:{port}")
    try:
        strict = _transport(client, edge.url, tmp_path, fallback=False)
        with pytest.raises(TransportError, match="rebuild the image"):
            strict.connect(host=microvm_id, port=port, credentials=CREDS)
        lenient = _transport(client, edge.url, tmp_path)
        lenient.connect(host=microvm_id, port=port, credentials=CREDS)
        assert lenient.uses_fallback
    finally:
        edge.close()
        stop(proc)


def test_the_endpoint_transport_requires_an_access_token_up_front() -> None:
    from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle

    with pytest.raises(SandboxError, match="DO_MICROVM_ACCESS_TOKEN"):  # before any MicroVM is created
        MicroVMLifecycle(MicroVMConfig(api_token=API_TOKEN, transport="endpoint", image="img"),
                         client=FakeMicroVMClient())  # type: ignore[arg-type]


def test_server_routes_need_the_token_except_health(server) -> None:
    root, port = server
    (root / "token").write_text("k")
    base = f"http://127.0.0.1:{port}"
    assert httpx.get(f"{base}/healthz").text == "ok\n"
    for method, path in (("GET", "/v1/info"), ("POST", "/v1/exec"), ("GET", "/v1/files?path=/etc/hosts")):
        resp = httpx.request(method, base + path, headers={"x-api-key": "nope"})
        assert resp.status_code == 401 and resp.json()["source"] == "microvm-sandbox"
    assert httpx.get(f"{base}/v1/info", headers={"x-api-key": "k"}).json()["server"] == "microvm-sandbox"


def test_lifecycle_get_or_create_seeds_through_the_endpoint(server, edge, tmp_path: Path) -> None:
    from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle

    root, _ = server
    client = FakeMicroVMClient()
    config = MicroVMConfig(api_token=API_TOKEN, access_token=ACCESS_TOKEN, transport="endpoint", image="img")
    lifecycle = MicroVMLifecycle(config, client=client,  # type: ignore[arg-type]  # duck-typed double
                                 transport_factory=lambda: _transport(client, edge.url, root))
    seed = str(tmp_path / "seeded" / "hello.txt")
    backend = lifecycle.get_or_create(sandbox_id="ep-life", seed_files=[(seed, b"hi")])
    try:
        assert not backend.transport.uses_fallback  # type: ignore[attr-defined]
        assert Path(seed).read_bytes() == b"hi"
        assert backend.execute(f"cat {seed}").output == "hi"
        assert [argv[:2] for _id, argv in client.exec_calls] == [["sh", "-c"]]  # only the re-key used exec
    finally:
        backend.close()


def test_a_microvm_without_an_endpoint_fails_fast_with_a_clear_message() -> None:
    """A running MicroVM whose API object lists no URL: wait `endpoint_wait`, not the ready timeout."""
    class NoUrl(FakeMicroVMClient):
        waited: list[float] = []

        def get(self, microvm_id):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            return replace(super().get(microvm_id), endpoint=None)

        def wait_for_endpoint(self, microvm_id, *, timeout, poll_interval=2.0):  # type: ignore[no-untyped-def]
            self.waited.append(timeout)
            raise TimeoutError(f"MicroVM {microvm_id} endpoint not ACTIVE after {timeout}s (state running)")

    client = NoUrl()
    state = client.create(name="nourl", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048), http_port=8080)
    t = EndpointTransport(client)  # type: ignore[arg-type]
    with pytest.raises(TransportError, match=r"endpoint not ACTIVE after 60.0s \(state running\): the MicroVMs API lists no HTTPS endpoint.*DO_MICROVM_TRANSPORT=exec"):
        t.connect(host=state.id, credentials=CREDS, timeout=300.0)  # the lifecycle passes its ready timeout
    assert client.waited == [60.0]

