"""Sandbox templates: MicroVMLifecycle.build_template / get_or_create(template=...) / delete_template."""

from __future__ import annotations

import tempfile

import pytest
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm import (
    MicroVMConfig,
    MicroVMLifecycle,
    SandboxTemplate,
)
from langchain_digitalocean_microvm.client import MANAGED_TAG
from langchain_digitalocean_microvm.exceptions import SandboxError
from langchain_digitalocean_microvm.provider import TEMPLATE_PREFIX
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.transport.local import LocalTransport

IMAGE = "registry.example/sandbox:1"


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _real(client: FakeMicroVMClient, **overrides) -> MicroVMLifecycle:
    config = MicroVMConfig(**{"transport": "exec", "api_token": "t", "image": IMAGE, "ready_timeout": 5.0, **overrides})
    return MicroVMLifecycle(config, client=client, transport_factory=LocalTransport)  # type: ignore[arg-type]


def test_build_runs_setup_checkpoints_and_deletes_the_build_microvm() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)

    template = lifecycle.build_template("analyst", setup=["touch /setup-ran"], seed_files=[("/data/x.csv", b"a,b\n")])

    assert template.built and template.checkpoint_name.startswith(TEMPLATE_PREFIX)
    assert [c.name for c in client.list_checkpoints()] == [template.checkpoint_name]
    build = client.created[0]
    assert build["oci_ref"] == IMAGE and TEMPLATE_PREFIX in build["tags"]
    assert all(s.status in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED) for s in client.list())  # build VM gone


def test_same_inputs_reuse_the_checkpoint_and_changed_inputs_build_a_new_one() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)
    first = lifecycle.build_template("analyst", setup=["true"])

    again = lifecycle.build_template("analyst", setup=["true"])
    changed = lifecycle.build_template("analyst", setup=["true", "echo more"])
    rebuilt = lifecycle.build_template("analyst", setup=["true"], rebuild=True)

    assert (again.built, again.checkpoint_id) == (False, first.checkpoint_id)
    assert changed.built and changed.checkpoint_name != first.checkpoint_name
    assert rebuilt.built and rebuilt.checkpoint_id != first.checkpoint_id
    assert first.checkpoint_id not in {c.id for c in client.list_checkpoints()}  # replaced, not kept


def test_a_failing_setup_command_raises_and_still_deletes_the_build_microvm() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)

    with pytest.raises(SandboxError, match="setup command failed"):
        lifecycle.build_template("broken", setup=["exit 7"])

    assert client.list_checkpoints() == []
    assert client.delete_calls == 1


def test_get_or_create_clones_new_sandboxes_from_the_template_and_reuses_existing_ones() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)
    template = lifecycle.build_template("analyst")

    backend = lifecycle.get_or_create(sandbox_id="warm-1", template=template, seed_files=[("/x", b"x")])
    again = lifecycle.get_or_create(sandbox_id="warm-1", template=template)

    clone = client.created[-1]
    assert clone["checkpoint_id"] == template.checkpoint_id and clone["oci_ref"] is None
    assert MANAGED_TAG in clone["tags"]
    assert again.id == backend.id and client.create_calls == 2  # build + one clone


def test_clone_does_not_need_an_image() -> None:
    client = FakeMicroVMClient()
    template = _real(client).build_template("analyst")

    backend = _real(client, image=None).get_or_create(sandbox_id="warm-2", template=template)

    assert client.get(backend.id).status == MicroVMStatus.RUNNING


def test_fake_mode_copies_seed_files_skips_setup_and_deletes() -> None:
    lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local"))
    template = lifecycle.build_template("analyst", setup=["touch /should-not-exist"], seed_files=[("/data/x.csv", b"a,b\n")])
    assert isinstance(template, SandboxTemplate) and template.built
    assert not lifecycle.build_template("analyst", setup=["touch /should-not-exist"], seed_files=[("/data/x.csv", b"a,b\n")]).built

    backend = lifecycle.get_or_create(sandbox_id="fake-warm", template=template)

    assert backend.download_files(["/data/x.csv"])[0].content == b"a,b\n"
    assert backend.download_files(["/should-not-exist"])[0].error == "file_not_found"
    lifecycle.delete_template(template)
    assert lifecycle.build_template("analyst", setup=["touch /should-not-exist"], seed_files=[("/data/x.csv", b"a,b\n")]).built


def test_find_template_returns_the_reusable_checkpoint_and_never_builds() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)

    assert lifecycle.find_template("analyst", setup=["true"]) is None
    assert client.created == [] and client.list_checkpoints() == []  # nothing booted, nothing checkpointed

    built = lifecycle.build_template("analyst", setup=["true"])
    found = lifecycle.find_template("analyst", setup=["true"])
    assert found is not None and (found.built, found.checkpoint_id) == (False, built.checkpoint_id)
    assert lifecycle.find_template("analyst", setup=["true", "echo more"]) is None  # other inputs, other template

    lifecycle.delete_template(found)
    assert lifecycle.find_template("analyst", setup=["true"]) is None


def test_find_template_in_local_mode() -> None:
    lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local"))

    assert lifecycle.find_template("analyst", seed_files=[("/x", b"x")]) is None
    built = lifecycle.build_template("analyst", seed_files=[("/x", b"x")])
    found = lifecycle.find_template("analyst", seed_files=[("/x", b"x")])
    assert found is not None and found.checkpoint_id == built.checkpoint_id and not found.built

    lifecycle.delete_template(found)
    assert lifecycle.find_template("analyst", seed_files=[("/x", b"x")]) is None
