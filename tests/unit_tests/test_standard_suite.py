"""LangChain's standard sandbox suite, offline: the real ExecTransport and EndpointTransport, locally.

FakeMicroVMClient runs each exec argv as a local subprocess and enforces the
MicroVMs exec limits, so this exercises everything the package sends to a
MicroVM (the shell wrapper, chunked file transfer, BaseSandbox's generated
scripts) without credentials. Paths are real local paths under a temporary
directory, as they are real paths inside a MicroVM.

`TestEndpointTransportStandard` runs the same suite through EndpointTransport
against the real sandbox server as a local process behind a fake edge
(tests/unit_tests/endpoint_harness.py).

The same suite against a real MicroVM: tests/integration_tests/test_sandbox_standard.py.
Known upstream and host gaps are xfailed in tests/standard_suite_gaps.py.
LocalTransport is not run through it: its absolute-path rewriting is
best-effort (see transport/local.py) and fails the glob/grep cases by design.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from endpoint_harness import (
    ACCESS_TOKEN,
    API_TOKEN,
    FakeEdge,
    free_port,
    start_server,
    stop,
)
from fake_client import FakeMicroVMClient
from langchain_tests.integration_tests import SandboxIntegrationTests
from standard_suite_gaps import local_host_gaps, upstream_gaps

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.transport.base import TransportCredentials
from langchain_digitalocean_microvm.transport.endpoint import EndpointTransport
from langchain_digitalocean_microvm.transport.exec import ExecTransport

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required"),
    pytest.mark.slow,  # ~5 min on macOS (10 MiB payloads through the exec wrapper)
]

_ROOT = tempfile.mkdtemp(prefix="ldm-standard-")


class TestExecTransportStandard(SandboxIntegrationTests):
    locals().update(upstream_gaps() | local_host_gaps())

    @property
    def sandbox_root_dir(self) -> str:
        return f"{_ROOT}/ops/"

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        client = FakeMicroVMClient()
        state = client.create(name="std", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048))
        client.get(state.id)  # settle to running
        transport = ExecTransport(
            client,  # type: ignore[arg-type]  # duck-typed double
            job_root=f"{_ROOT}/jobs",
            sleep=lambda s: time.sleep(min(s, 0.05)),
        )
        transport.connect(host=state.id, credentials=TransportCredentials(api_token="t"))
        try:
            yield MicroVMSandbox(microvm_id=state.id, transport=transport)
        finally:
            shutil.rmtree(_ROOT, ignore_errors=True)


_EP_ROOT = tempfile.mkdtemp(prefix="ldm-standard-ep-")


class TestEndpointTransportStandard(SandboxIntegrationTests):
    locals().update(upstream_gaps() | local_host_gaps())

    @property
    def sandbox_root_dir(self) -> str:
        return f"{_EP_ROOT}/ops/"

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        from pathlib import Path

        root = Path(_EP_ROOT)
        port = free_port()
        server = start_server(root, port)
        edge = FakeEdge(f"http://127.0.0.1:{port}", budget=15.0)
        client = FakeMicroVMClient()
        state = client.create(name="std-ep", oci_ref="img", region="mkc1", size=MicroVMSize(1, 2048))
        client.get(state.id)  # settle to running
        transport = EndpointTransport(
            client,  # type: ignore[arg-type]  # duck-typed double
            url_for=lambda _state: edge.url,
            token_file=str(root / "token"),
            server_dir=str(root / "srv"),
            fallback=False,
        )
        transport.connect(host=state.id, credentials=TransportCredentials(api_token=API_TOKEN, access_token=ACCESS_TOKEN))
        try:
            yield MicroVMSandbox(microvm_id=state.id, transport=transport)
        finally:
            transport.close()
            edge.close()
            stop(server)
            shutil.rmtree(_EP_ROOT, ignore_errors=True)
