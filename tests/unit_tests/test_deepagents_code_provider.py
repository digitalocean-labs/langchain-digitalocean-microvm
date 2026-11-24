"""MicroVMSandboxProvider, the deepagents_code (`dcode`) adapter (see docs/design.md).

Skipped unless deepagents-code is importable (not a runtime dependency of
this package; the `test` dependency group installs it).
"""

from __future__ import annotations

import importlib.metadata
import tempfile

import pytest

pytest.importorskip("deepagents_code")

from deepagents_code.integrations.sandbox_provider import (  # noqa: E402
    SandboxError,
    SandboxNotFoundError,
    SandboxProvider,
)
from fake_client import FakeMicroVMClient  # noqa: E402

from langchain_digitalocean_microvm.backend import MicroVMSandbox  # noqa: E402
from langchain_digitalocean_microvm.client import MANAGED_TAG  # noqa: E402
from langchain_digitalocean_microvm.config import MicroVMConfig  # noqa: E402
from langchain_digitalocean_microvm.integrations.deepagents_code import (  # noqa: E402
    PROVIDER_NAME,
    MicroVMSandboxProvider,
)
from langchain_digitalocean_microvm.provider import MicroVMLifecycle  # noqa: E402
from langchain_digitalocean_microvm.status import MicroVMStatus  # noqa: E402
from langchain_digitalocean_microvm.transport.local import LocalTransport  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _provider(client: FakeMicroVMClient) -> MicroVMSandboxProvider:
    config = MicroVMConfig(transport="exec", api_token="test-token", image="registry/x:1", ready_timeout=5.0)
    lifecycle = MicroVMLifecycle(config, client=client, transport_factory=LocalTransport)  # type: ignore[arg-type]
    return MicroVMSandboxProvider(lifecycle)


def test_is_a_sandbox_provider_with_metadata_and_no_credentials_needed(monkeypatch) -> None:
    for var in ("DO_MICROVM_API_TOKEN", "DO_MICROVM_TRANSPORT"):
        monkeypatch.delenv(var, raising=False)
    provider = MicroVMSandboxProvider()  # how the dcode registry builds it: no arguments
    assert isinstance(provider, SandboxProvider)
    meta = provider.metadata
    assert (meta.name, meta.working_dir, meta.supports_sandbox_id, meta.supports_snapshot_name) == (
        PROVIDER_NAME, "/", True, False,
    )
    with pytest.raises(SandboxError, match="api_token"):
        provider.get_or_create()


def test_entry_point_is_registered() -> None:
    eps = importlib.metadata.entry_points(group="deepagents_code.sandbox_providers")
    (ep,) = [ep for ep in eps if ep.name == PROVIDER_NAME]
    assert ep.load() is MicroVMSandboxProvider


def test_create_reuse_by_either_id_and_delete() -> None:
    client = FakeMicroVMClient()
    provider = _provider(client)

    backend = provider.get_or_create(seed_files=[("/hello.txt", b"hi")])
    assert isinstance(backend, MicroVMSandbox)
    assert client.create_calls == 1
    (state,) = client.live()
    assert backend.id == state.id and MANAGED_TAG in state.tags
    assert backend.execute("cat /hello.txt").output == "hi"

    # dcode prints backend.id and hands it back later, possibly from a new process.
    again = _provider(client).get_or_create(sandbox_id=backend.id)
    assert again.id == backend.id and client.create_calls == 1

    named = provider.get_or_create(sandbox_id="my-box")
    assert provider.get_or_create(sandbox_id="my-box").id == named.id
    assert client.create_calls == 2

    _provider(client).delete(sandbox_id=backend.id)  # what dcode's cleanup calls
    provider.delete(sandbox_id="my-box")
    assert client.live() == []
    with pytest.raises(SandboxNotFoundError):
        provider.delete(sandbox_id="my-box")


def test_egress_is_a_policy_or_a_destination_list() -> None:
    from langchain_digitalocean_microvm import EgressPolicy

    client = FakeMicroVMClient()
    provider = _provider(client)
    provider.get_or_create(sandbox_id="a", egress="pypi.org,files.pythonhosted.org")
    provider.get_or_create(sandbox_id="b", egress=EgressPolicy.deny_all())
    assert [c["egress_policy"] for c in client.created] == [
        EgressPolicy.allow("pypi.org", "files.pythonhosted.org"), EgressPolicy.deny_all()
    ]
    with pytest.raises(ValueError, match="EgressPolicy or a string"):
        provider.get_or_create(sandbox_id="c", egress=["pypi.org"])


def test_unknown_options_are_refused() -> None:
    provider = _provider(FakeMicroVMClient())
    with pytest.raises(ValueError, match="snapshot"):
        provider.get_or_create(snapshot="golden")
    with pytest.raises(ValueError):
        provider.delete(sandbox_id="x", force=True)


def test_async_twins() -> None:
    import asyncio

    client = FakeMicroVMClient()
    provider = _provider(client)

    async def main() -> None:
        backend = await provider.aget_or_create(sandbox_id="async-box")
        assert client.get(backend.id).status == MicroVMStatus.RUNNING
        await provider.adelete(sandbox_id="async-box")

    asyncio.run(main())
    assert client.live() == []


def test_dcode_create_sandbox_end_to_end_in_fake_mode(monkeypatch) -> None:
    """dcode's own create/cleanup path: registry -> entry point -> provider, delete by backend.id."""
    from deepagents_code.integrations import sandbox_factory

    monkeypatch.setenv("DO_MICROVM_TRANSPORT", "local")
    with sandbox_factory.create_sandbox(PROVIDER_NAME) as backend:
        assert isinstance(backend, MicroVMSandbox)
        backend.write("/x.txt", "local")
        assert backend.execute("cat /x.txt").output == "local"
