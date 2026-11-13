"""Tests for MicroVMLifecycle's real-MicroVM reconciliation logic.

These exercise provider.py's real branch (config.transport == "exec") with
two test seams from MicroVMLifecycle.__init__:

- `client=`: FakeMicroVMClient (tests/fake_client.py), an in-memory double of
  the MicroVMs API that behaves like the recorded real responses (creating ->
  running, unique names, `terminating` after delete).
- `transport_factory=`: LocalTransport, a purely local subprocess transport,
  so MicroVMSandbox.execute() genuinely works end to end without the exec API
  (ExecTransport itself is covered by test_exec_transport.py).
"""

from __future__ import annotations

import pytest
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import MANAGED_TAG, id_tag, microvm_name
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import SandboxError, SandboxNotFoundError
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.provider import MicroVMLifecycle
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.transport.local import LocalTransport

IMAGE = "registry.digitalocean.com/team/agent:1"


def _make_lifecycle(client: FakeMicroVMClient) -> MicroVMLifecycle:
    config = MicroVMConfig(
        transport="exec",
        api_token="test-token",
        image=IMAGE,
        idle_timeout="15m",
        ready_timeout=5.0,
    )
    return MicroVMLifecycle(
        config,
        client=client,  # type: ignore[arg-type]  # duck-typed double, not a real MicroVMClient
        transport_factory=LocalTransport,
    )


def test_fresh_sandbox_creates_and_waits_for_running() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    backend = lifecycle.get_or_create(sandbox_id="sbx-fresh")

    assert isinstance(backend, MicroVMSandbox)
    assert backend.id
    assert client.create_calls == 1
    assert client.get(backend.id).status == MicroVMStatus.RUNNING

    # The returned sandbox is genuinely connected (LocalTransport runs a real
    # local subprocess), not just a state-machine artifact.
    result = backend.execute("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.output
    backend.close()


def test_existing_running_microvm_is_reused_without_recreating() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    first = lifecycle.get_or_create(sandbox_id="sbx-reuse")
    assert client.create_calls == 1
    first.close()

    second = lifecycle.get_or_create(sandbox_id="sbx-reuse")

    assert client.create_calls == 1  # no second create() call happened
    assert second.id == first.id
    second.close()


def test_paused_microvm_is_resumed_on_get_or_create() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    backend = lifecycle.get_or_create(sandbox_id="sbx-pause")
    lifecycle.pause(backend)
    assert client.pause_calls == 1
    assert client.get(backend.id).status == MicroVMStatus.PAUSED

    resumed = lifecycle.get_or_create(sandbox_id="sbx-pause")

    assert client.resume_calls == 1
    assert client.create_calls == 1  # never recreated
    assert resumed.id == backend.id
    assert client.get(backend.id).status == MicroVMStatus.RUNNING
    resumed.close()


def test_transitional_microvm_settles_after_n_polls() -> None:
    n = 3
    client = FakeMicroVMClient(stuck_transitional_polls=n)
    lifecycle = _make_lifecycle(client)

    backend = lifecycle.get_or_create(sandbox_id="sbx-transitional")

    # The freshly created MicroVM starts CREATING and must be polled via
    # wait_for_status() -- proving polling actually looped rather than
    # instantly succeeding (n "still transitional" reads, then settled).
    assert client.wait_for_status_calls == 1
    assert client.get_calls >= n + 1
    assert client.get(backend.id).status == MicroVMStatus.RUNNING
    backend.close()


def test_failed_microvm_is_deleted_then_recreated() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    original = lifecycle.get_or_create(sandbox_id="sbx-failed")
    original_id = original.id
    original.close()
    assert client.create_calls == 1

    client.force_status[original_id] = MicroVMStatus.FAILED

    recreated = lifecycle.get_or_create(sandbox_id="sbx-failed")

    assert client.delete_calls == 1
    assert client.create_calls == 2
    assert recreated.id != original_id
    assert client.get(recreated.id).status == MicroVMStatus.RUNNING
    recreated.close()


def test_create_race_recovers_by_refetching_instead_of_raising() -> None:
    client = FakeMicroVMClient(fail_next_create=True)
    lifecycle = _make_lifecycle(client)

    backend = lifecycle.get_or_create(sandbox_id="sbx-race")

    assert backend.id
    assert client.create_calls == 1
    assert client.fail_next_create is False
    assert client.get(backend.id).status == MicroVMStatus.RUNNING
    backend.close()


def test_delete_with_no_matching_microvm_raises_sandbox_not_found() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    with pytest.raises(SandboxNotFoundError):
        lifecycle.delete(sandbox_id="sbx-never-existed")


def test_pause_then_get_or_create_calls_resume_not_create() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    backend = lifecycle.get_or_create(sandbox_id="sbx-pause-resume")
    assert client.get_by_name(microvm_name("microvm-sandbox", "sbx-pause-resume")) is not None
    lifecycle.pause(backend)

    lifecycle.get_or_create(sandbox_id="sbx-pause-resume")

    assert client.resume_calls == 1
    assert client.create_calls == 1


def test_pause_by_sandbox_id_without_a_live_backend() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)

    backend = lifecycle.get_or_create(sandbox_id="sbx-pause-by-id")
    backend.close()

    lifecycle.pause(sandbox_id="sbx-pause-by-id")

    assert client.pause_calls == 1
    assert client.get(backend.id).status == MicroVMStatus.PAUSED

    # Idempotent: an already-PAUSED MicroVM is left alone.
    lifecycle.pause(sandbox_id="sbx-pause-by-id")
    assert client.pause_calls == 1


def test_pause_by_unknown_sandbox_id_raises_sandbox_not_found() -> None:
    lifecycle = _make_lifecycle(FakeMicroVMClient())

    with pytest.raises(SandboxNotFoundError):
        lifecycle.pause(sandbox_id="sbx-never-existed")


def test_pause_requires_exactly_one_of_backend_or_sandbox_id() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)
    backend = lifecycle.get_or_create(sandbox_id="sbx-pause-args")

    with pytest.raises(ValueError):
        lifecycle.pause()
    with pytest.raises(ValueError):
        lifecycle.pause(backend, sandbox_id="sbx-pause-args")
    backend.close()


def test_create_sends_image_size_region_idle_timeout_and_tags() -> None:
    client = FakeMicroVMClient()
    backend = _make_lifecycle(client).get_or_create(sandbox_id="sbx-payload")
    [payload] = client.created
    assert payload["name"] == "microvm-sandbox-sbx-payload"
    assert payload["oci_ref"] == IMAGE
    assert payload["region"] == "mkc1"
    assert payload["size"] == MicroVMSize(cpu=1, memory=2048)
    assert payload["idle_timeout"] == "15m"
    assert payload["tags"] == [MANAGED_TAG, id_tag("sbx-payload")]
    assert payload["environment"] is None  # sandboxes get no secrets
    backend.close()


def test_missing_image_is_an_explicit_error() -> None:
    lifecycle = MicroVMLifecycle(
        MicroVMConfig(transport="exec", api_token="t"),
        client=FakeMicroVMClient(),  # type: ignore[arg-type]
        transport_factory=LocalTransport,
    )
    with pytest.raises(SandboxError, match="DO_MICROVM_IMAGE"):
        lifecycle.get_or_create(sandbox_id="sbx-no-image")


def test_a_terminating_leftover_does_not_block_a_new_sandbox() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)
    first = lifecycle.get_or_create(sandbox_id="sbx-again")
    first.close()
    lifecycle.delete(sandbox_id="sbx-again")
    assert client.get(first.id).status == MicroVMStatus.TERMINATING  # still listed, as on the real API

    second = lifecycle.get_or_create(sandbox_id="sbx-again")
    assert second.id != first.id and client.create_calls == 2
    second.close()


def test_sandbox_ids_that_are_not_valid_names_are_mapped_deterministically() -> None:
    client = FakeMicroVMClient()
    lifecycle = _make_lifecycle(client)
    backend = lifecycle.get_or_create(sandbox_id="Thread 42/Notes")
    name = client.created[0]["name"]
    assert name == microvm_name("microvm-sandbox", "Thread 42/Notes") and len(name) <= 63
    backend.close()
    assert lifecycle.get_or_create(sandbox_id="Thread 42/Notes").id == backend.id  # found again by name
