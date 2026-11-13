"""scripts/delete_all.py keeps templates by default, warns about the scope and asks before deleting.

On `FakeMicroVMClient`: which MicroVMs and checkpoints are selected (template
checkpoints only with --include-templates), the warning's counts, and the
confirmation (the typed word at a terminal, --force otherwise). No network here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fake_client import FakeMicroVMClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import delete_all  # noqa: E402

from langchain_digitalocean_microvm.client import MANAGED_TAG  # noqa: E402


def _team() -> FakeMicroVMClient:
    client = FakeMicroVMClient()
    sandbox = client.create(name="microvm-sandbox-sales-t1", oci_ref="img", tags=[MANAGED_TAG])
    builder = client.create(name="microvm-sandbox-template-build", oci_ref="img", tags=[MANAGED_TAG])
    for state in (sandbox, builder):
        client.wait_for_status(state.id, timeout=5, poll_interval=0)
    client.pause(sandbox.id)
    client.create_checkpoint(builder.id, name="microvm-template-sales-assistant-431ebf1527")
    client.create_checkpoint(builder.id, name="microvm-golden-abc123")
    return client


def test_templates_are_kept_unless_included() -> None:
    client = _team()
    microvms, checkpoints, kept = delete_all.find(client, include_templates=False)
    assert sorted(s.name for s in microvms) == ["microvm-sandbox-sales-t1", "microvm-sandbox-template-build"]
    assert [cp.name for cp in checkpoints] == ["microvm-golden-abc123"]
    assert [cp.name for cp in kept] == ["microvm-template-sales-assistant-431ebf1527"]

    _, checkpoints, kept = delete_all.find(client, include_templates=True)
    assert len(checkpoints) == 2 and kept == []


def test_the_warning_names_the_scope_and_the_counts() -> None:
    text = delete_all.warning(*delete_all.find(_team(), include_templates=False))
    assert "the default namespace (no namespace tag)" in text and "Other namespaces are left alone" in text
    assert "2 MicroVM(s) (1 running), 1 checkpoint(s)" in text
    assert "1 template checkpoint(s) kept" in text

    everything = delete_all.warning(*delete_all.find(_team(), include_templates=False, all_namespaces=True), all_namespaces=True)
    assert "in every namespace, including other people's" in everything


def test_deleting_needs_the_typed_word_or_force(capsys) -> None:
    assert delete_all.confirmed(force=False, interactive=True, ask=lambda _: "delete")
    assert not delete_all.confirmed(force=False, interactive=True, ask=lambda _: "y")
    assert "nothing deleted" in capsys.readouterr().out

    def eof(_: str) -> str:
        raise EOFError

    assert not delete_all.confirmed(force=False, interactive=True, ask=eof)
    assert not delete_all.confirmed(force=False, interactive=False, ask=lambda _: "delete")  # never asked
    assert "--yes --force" in capsys.readouterr().out
    assert delete_all.confirmed(force=True, interactive=False)
