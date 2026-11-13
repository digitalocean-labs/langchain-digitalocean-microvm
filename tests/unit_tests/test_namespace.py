"""Namespaces separate the users of one team: tags, names, store file, clean-up scope.

On the in-memory `FakeMicroVMClient`; no network.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient
from test_agent_host import SPEC, _config, _make_host
from test_provider import IMAGE

from langchain_digitalocean_microvm.agent_host.host import (
    GOLDEN_PREFIX,
    GOLDEN_TAG,
    MicroVMAgentHost,
)
from langchain_digitalocean_microvm.agent_host.models import (
    HOST_MANAGED_TAG,
    host_id_tag,
)
from langchain_digitalocean_microvm.agent_host.store import default_db_path
from langchain_digitalocean_microvm.agent_host.supervisor import GatewaySettings
from langchain_digitalocean_microvm.agent_host.sweeper import apply_sweep, plan_sweep
from langchain_digitalocean_microvm.client import MANAGED_TAG, microvm_name
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.namespace import (
    list_owned,
    namespace_from_name,
    namespace_from_tags,
    namespace_tags,
    namespaced_prefix,
    validate_namespace,
)
from langchain_digitalocean_microvm.provider import MicroVMLifecycle
from langchain_digitalocean_microvm.transport.local import LocalTransport

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _env  # noqa: E402
import delete_all  # noqa: E402


def _lifecycle(client: FakeMicroVMClient, namespace: str | None) -> MicroVMLifecycle:
    config = MicroVMConfig(transport="exec", api_token="t", image=IMAGE, ready_timeout=5.0, namespace=namespace)
    return MicroVMLifecycle(config, client=client, transport_factory=LocalTransport)  # type: ignore[arg-type]


def test_a_namespace_is_lowercase_letters_and_digits() -> None:
    assert validate_namespace(None) is None and validate_namespace("  ") is None
    assert validate_namespace(" dev1 ") == "dev1"
    for bad in ("Dev", "a-b", "a_b", "x" * 17):
        with pytest.raises(ValueError):
            validate_namespace(bad)
    with pytest.raises(ValueError):
        MicroVMConfig(namespace="no way")


def test_config_reads_the_namespace_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_MICROVM_NAMESPACE", "alice")
    assert MicroVMConfig.from_env().namespace == "alice"
    monkeypatch.setenv("DO_MICROVM_NAMESPACE", "")
    assert MicroVMConfig.from_env().namespace is None


def test_names_and_tags_round_trip() -> None:
    assert namespaced_prefix("microvm-sandbox", None) == "microvm-sandbox"
    assert namespaced_prefix("microvm-sandbox", "ab") == "microvm-sandbox-ns-ab"
    assert namespace_from_tags(namespace_tags("ab")) == "ab" and namespace_from_tags([MANAGED_TAG]) is None
    name = microvm_name(namespaced_prefix("microvm-agenthost", "ab"), "pool-x-y")
    assert namespace_from_name(name, ("microvm-agenthost",)) == "ab"
    assert namespace_from_name("microvm-agenthost-pool-x", ("microvm-agenthost",)) is None


def test_the_same_sandbox_id_is_one_sandbox_per_namespace() -> None:
    client = FakeMicroVMClient()
    alice, bob, default = (_lifecycle(client, ns) for ns in ("alice", "bob", None))
    a = alice.get_or_create(sandbox_id="shared")
    b = bob.get_or_create(sandbox_id="shared")
    d = default.get_or_create(sandbox_id="shared")
    assert len({a.id, b.id, d.id}) == 3 and client.create_calls == 3
    assert namespace_from_tags(client.get(a.id).tags) == "alice"
    assert namespace_from_tags(client.get(d.id).tags) is None
    assert alice.sandbox_id_for(b.id) is None and alice.sandbox_id_for(a.id) == "shared"  # not ours to delete
    alice.delete(sandbox_id="shared")
    assert {s.id for s in client.live()} == {b.id, d.id}


def test_the_same_pooled_host_id_is_one_host_per_namespace() -> None:
    client = FakeMicroVMClient()
    alice, _, _ = _make_host(client, config=_config(namespace="alice"))
    bob, _, _ = _make_host(client, config=_config(namespace="bob"))
    first = alice.get_or_create(host_id="pool-researcher", spec=SPEC)
    second = bob.get_or_create(host_id="pool-researcher", spec=SPEC)
    assert first.microvm_id != second.microvm_id
    assert [h.microvm_id for h in alice.list_hosts()] == [first.microvm_id]
    assert [h.microvm_id for h in bob.list_hosts()] == [second.microvm_id]
    tags = client.get(first.microvm_id).tags
    assert HOST_MANAGED_TAG in tags and host_id_tag("pool-researcher") in tags and "microvm-namespace:alice" in tags
    assert client.get(first.microvm_id).name == "microvm-agenthost-ns-alice-pool-researcher"


def test_golden_checkpoints_belong_to_a_namespace() -> None:
    client = FakeMicroVMClient()
    names = []
    for ns in ("alice", None):
        host = MicroVMAgentHost(_config(namespace=ns), client=client)  # type: ignore[arg-type]
        names.append(host._golden_name(SPEC, IMAGE, MicroVMSize(1, 2048)))
    assert names[0].startswith(f"{GOLDEN_PREFIX}-ns-alice-") and names[1].startswith(f"{GOLDEN_PREFIX}-") and "-ns-" not in names[1]


def test_the_store_file_is_per_namespace() -> None:
    assert default_db_path("microvm").name == "gateway.sqlite"
    assert default_db_path("microvm", "alice").name == "gateway-alice.sqlite"
    assert default_db_path("local", "alice").name == "gateway-local-alice.sqlite"
    settings = GatewaySettings(host_mode="microvm", token="t", namespace="alice")
    assert settings.resolved_db_path == default_db_path("microvm", "alice")


def _team() -> tuple[FakeMicroVMClient, dict[str, str]]:
    client = FakeMicroVMClient()
    ids: dict[str, str] = {}
    for ns in ("alice", "bob", None):
        host = client.create(name=microvm_name(namespaced_prefix("microvm-agenthost", ns), "h"), oci_ref="img",
                             region="mkc1", size=MicroVMSize(1, 2048),
                             tags=[HOST_MANAGED_TAG, host_id_tag("h"), *namespace_tags(ns)])
        sandbox = client.create(name=microvm_name(namespaced_prefix("microvm-sandbox", ns), "s"), oci_ref="img",
                                tags=[*namespace_tags(ns)])
        golden = client.create(name=microvm_name(namespaced_prefix(GOLDEN_PREFIX, ns), "g"), oci_ref="img",
                               tags=[GOLDEN_TAG, *namespace_tags(ns)])
        cp = client.create_checkpoint(golden.id, name=microvm_name(namespaced_prefix(GOLDEN_PREFIX, ns), "cp"))
        client._checkpoints[cp.id] = replace(cp, status="CHECKPOINT_FAILED")  # the sweeper collects failed captures
        ids[f"host-{ns}"], ids[f"sandbox-{ns}"], ids[f"golden-{ns}"], ids[f"cp-{ns}"] = host.id, sandbox.id, golden.id, cp.id
    return client, ids


def test_list_owned_is_one_namespace_unless_all() -> None:
    client, ids = _team()
    assert {s.id for s in list_owned(client, namespace="alice")} == {ids["host-alice"], ids["sandbox-alice"], ids["golden-alice"]}
    assert {s.id for s in list_owned(client, namespace=None)} == {ids["host-None"], ids["sandbox-None"], ids["golden-None"]}
    assert len(list_owned(client, namespace="alice", all_namespaces=True)) == 9


def test_the_sweeper_only_plans_its_own_namespace() -> None:
    client, ids = _team()
    for ns, expected in (("alice", "alice"), (None, "None")):
        plan = plan_sweep(client, store=None, now=1e12, orphan_ttl=0.0, checkpoint_ttl=0.0, golden_build_timeout=0.0, namespace=ns)
        assert {i.id for i in plan.delete_microvms} == {ids[f"host-{expected}"], ids[f"golden-{expected}"]}
        assert {i.id for i in plan.delete_checkpoints} == {ids[f"cp-{expected}"]}
    plan = plan_sweep(client, store=None, now=1e12, orphan_ttl=0.0, checkpoint_ttl=0.0, golden_build_timeout=0.0,
                      namespace="alice", all_namespaces=True)
    assert len(plan.delete_microvms) == 6 and len(plan.delete_checkpoints) == 3
    apply_sweep(plan_sweep(client, store=None, now=1e12, orphan_ttl=0.0, namespace="bob"), client, store=None)
    assert ids["host-bob"] not in {s.id for s in client.live()} and ids["host-alice"] in {s.id for s in client.live()}


def test_delete_all_is_one_namespace_unless_all() -> None:
    client, ids = _team()
    microvms, checkpoints, _ = delete_all.find(client, include_templates=False, namespace="alice")
    assert {s.id for s in microvms} == {ids["host-alice"], ids["sandbox-alice"], ids["golden-alice"]}
    assert [c.id for c in checkpoints] == [ids["cp-alice"]]
    microvms, checkpoints, _ = delete_all.find(client, include_templates=False)  # the default namespace
    assert {s.id for s in microvms} == {ids["host-None"], ids["sandbox-None"], ids["golden-None"]}
    microvms, checkpoints, _ = delete_all.find(client, include_templates=False, all_namespaces=True)
    assert len(microvms) == 9 and len(checkpoints) == 3
    text = delete_all.warning(*delete_all.find(client, include_templates=False, namespace="alice"), namespace="alice")
    assert "namespace 'alice'" in text and "Other namespaces are left alone" in text


def test_templates_are_shared_so_deleting_them_needs_all_namespaces() -> None:
    client, _ = _team()
    assert delete_all.templates_need_all_namespaces(client, "alice")  # a namespace is set
    assert delete_all.templates_need_all_namespaces(client, None)  # others are on the team
    solo = FakeMicroVMClient()
    solo.create(name="microvm-sandbox-x", oci_ref="img", tags=[MANAGED_TAG])
    assert not delete_all.templates_need_all_namespaces(solo, None)


def test_probes_refuse_only_while_their_own_namespace_has_microvms(capsys: pytest.CaptureFixture[str]) -> None:
    client, _ = _team()
    assert _env.refuse_if_owned(client, "alice")
    assert "namespace 'alice'" in capsys.readouterr().err
    empty_ns = FakeMicroVMClient()
    empty_ns.create(name="microvm-sandbox-ns-bob-x", oci_ref="img", tags=[MANAGED_TAG, *namespace_tags("bob")])
    assert not _env.refuse_if_owned(empty_ns, "alice")
    assert "outside this namespace" in capsys.readouterr().err
