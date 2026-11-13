"""Clean-up touches what this user created, not what other users of the team did.

On the in-memory `FakeMicroVMClient`; no network, no real state directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient

from langchain_digitalocean_microvm import owner
from langchain_digitalocean_microvm.agent_host import store
from langchain_digitalocean_microvm.agent_host.models import (
    HOST_MANAGED_TAG,
    host_id_tag,
)
from langchain_digitalocean_microvm.agent_host.sweeper import plan_sweep
from langchain_digitalocean_microvm.client import MANAGED_TAG

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import delete_all  # noqa: E402

OTHER = "microvm-owner:ffffffffff"


def test_the_owner_id_is_created_once_and_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(store, "state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(owner, "_cached", None)
    first = owner.owner_id()
    assert len(first) == 10 and (tmp_path / "state" / "owner-id").read_text().strip() == first
    monkeypatch.setattr(owner, "_cached", None)  # a new process reads the file
    assert owner.owner_id() == first
    assert owner.owner_tag() == f"microvm-owner:{first}"


def test_an_unwritable_state_directory_still_gives_an_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(store, "state_dir", lambda: blocker / "state")
    monkeypatch.setattr(owner, "_cached", None)
    assert len(owner.owner_id()) == 10


def test_owner_tags_are_read_back() -> None:
    assert owner.owner_from_tags([MANAGED_TAG, OTHER]) == "ffffffffff"
    assert owner.owner_from_tags([MANAGED_TAG]) is None
    assert owner.is_mine([owner.owner_tag()]) and not owner.is_mine([OTHER]) and not owner.is_mine([MANAGED_TAG])
    assert owner.in_owner_scope([OTHER], all_owners=True) and not owner.in_owner_scope([OTHER])


def _shared_team() -> tuple[FakeMicroVMClient, dict[str, str]]:
    """Mine, someone else's, and one created before owner tags existed."""
    client = FakeMicroVMClient()
    ids = {"mine": client.create(name="microvm-sandbox-mine", oci_ref="img").id,
           "other": client.create(name="microvm-sandbox-other", oci_ref="img", tags=[OTHER]).id}
    old = client._register(name="microvm-sandbox-old", size=None, region=None, tags=[MANAGED_TAG], http_port=None)
    ids["old"] = old.id
    return client, ids


def test_delete_all_selects_only_my_microvms_by_default() -> None:
    client, ids = _shared_team()
    microvms, _, _ = delete_all.find(client, include_templates=False)
    assert {s.id for s in microvms} == {ids["mine"]}


def test_all_owners_adds_other_users_and_untagged_microvms() -> None:
    client, ids = _shared_team()
    microvms, _, _ = delete_all.find(client, include_templates=False, all_owners=True)
    assert {s.id for s in microvms} == set(ids.values())
    microvms, _, _ = delete_all.find(client, include_templates=False, all_namespaces=True)
    assert {s.id for s in microvms} == set(ids.values())  # every namespace implies every owner


def test_a_checkpoint_follows_its_microvm_and_a_bare_name_needs_every_owner() -> None:
    client, ids = _shared_team()
    mine = client.create_checkpoint(ids["mine"], name="microvm-golden-a")
    theirs = client.create_checkpoint(ids["other"], name="microvm-golden-b")
    _, checkpoints, _ = delete_all.find(client, include_templates=False)
    assert [cp.id for cp in checkpoints] == [mine.id]
    _, checkpoints, _ = delete_all.find(client, include_templates=False, all_owners=True)
    assert {cp.id for cp in checkpoints} == {mine.id, theirs.id}


def test_the_warning_says_whose_microvms_go() -> None:
    client, _ = _shared_team()
    mine = delete_all.warning(*delete_all.find(client, include_templates=False))
    assert "you created" in mine and "--all-owners" in mine
    everyone = delete_all.warning(*delete_all.find(client, include_templates=False, all_owners=True), all_owners=True)
    assert "of every owner" in everyone


def _host(client: FakeMicroVMClient, name: str, tags: list[str]) -> str:
    return client.create(name=name, oci_ref="img", tags=[HOST_MANAGED_TAG, host_id_tag(name), *tags]).id


def test_the_sweeper_keeps_an_unknown_host_that_is_not_mine_unless_all_owners() -> None:
    client = FakeMicroVMClient()
    mine = _host(client, "h-mine", [])
    other = _host(client, "h-other", [OTHER])
    old = client._register(name="h-old", size=None, region=None, tags=[HOST_MANAGED_TAG, host_id_tag("h-old")],
                           http_port=None).id
    kwargs = dict(store=None, now=1e12, orphan_ttl=0.0)
    plan = plan_sweep(client, all_owners=False, **kwargs)  # type: ignore[arg-type]
    assert {i.id for i in plan.delete_microvms} == {mine}
    assert {other, old} <= {i.id for i in plan.kept}
    assert "not created by this user" in next(i.reason for i in plan.kept if i.id == other)
    plan = plan_sweep(client, **kwargs)  # type: ignore[arg-type]  # the library default is unchanged
    assert {i.id for i in plan.delete_microvms} == {mine, other, old}
