"""Tests for MicroVMLifecycle's local transport branch (no client, no network).

test_provider.py drives the REAL branch through FakeMicroVMClient; this
file pins the separate in-memory simulation used when
config.transport == "local", including the id-based pause.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError
from langchain_digitalocean_microvm.provider import MicroVMLifecycle


def _lifecycle() -> MicroVMLifecycle:
    return MicroVMLifecycle(MicroVMConfig(transport="local"))


def test_fake_branch_seeds_pauses_by_id_resumes_and_deletes() -> None:
    lifecycle = _lifecycle()

    backend = lifecycle.get_or_create(
        sandbox_id="fake-sbx", seed_files=[("/notes.md", b"seeded\n")]
    )
    assert backend.read("/notes.md").file_data["content"].strip() == "seeded"

    lifecycle.pause(backend)
    lifecycle.pause(sandbox_id="fake-sbx")  # id-based path, idempotent

    resumed = lifecycle.resume(sandbox_id="fake-sbx")
    result = resumed.execute("cat /notes.md")
    assert result.exit_code == 0
    assert "seeded" in result.output
    resumed.close()

    # A reused sandbox is never re-seeded.
    again = lifecycle.get_or_create(
        sandbox_id="fake-sbx", seed_files=[("/notes.md", b"overwritten\n")]
    )
    assert again.read("/notes.md").file_data["content"].strip() == "seeded"
    again.close()

    lifecycle.delete(sandbox_id="fake-sbx")
    with pytest.raises(SandboxNotFoundError):
        lifecycle.resume(sandbox_id="fake-sbx")


def test_a_deleted_fake_sandbox_leaves_no_files_and_its_id_starts_empty() -> None:
    lifecycle = _lifecycle()
    sandbox_id = f"fake-del-{uuid.uuid4().hex[:8]}"
    root = Path(tempfile.gettempdir()) / "microvm_sandbox_local" / sandbox_id  # where local mode keeps its files
    backend = lifecycle.get_or_create(sandbox_id=sandbox_id, seed_files=[("/notes.md", b"old\n")])
    assert not backend.write("/work.txt", "work\n").error
    backend.close()
    assert (root / "work.txt").exists()

    lifecycle.delete(sandbox_id=sandbox_id)
    assert not root.exists()

    again = lifecycle.get_or_create(sandbox_id=sandbox_id)
    assert again.read("/notes.md").error and again.read("/work.txt").error
    again.close()
    lifecycle.delete(sandbox_id=sandbox_id)


def test_fake_branch_pause_by_unknown_id_raises() -> None:
    with pytest.raises(SandboxNotFoundError):
        _lifecycle().pause(sandbox_id="never-created")
