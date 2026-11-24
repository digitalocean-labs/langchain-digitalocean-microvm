"""Agent hosts with an outbound policy: HostSpec.egress, config.agent_egress, warm start and the sweeper."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient
from test_agent_host import IMAGE, _config, _make_host
from test_warm_start import make as make_warm

from langchain_digitalocean_microvm import EgressPolicy, MicroVMConfig
from langchain_digitalocean_microvm.agent_host import HostSpec
from langchain_digitalocean_microvm.agent_host.host import GOLDEN_PREFIX
from langchain_digitalocean_microvm.agent_host.local import LocalDevServerHost
from langchain_digitalocean_microvm.agent_host.sweeper import plan_sweep
from langchain_digitalocean_microvm.status import MicroVMStatus

MODEL = EgressPolicy.allow_all().with_credential("https://inference.do-ai.run/v1", secret="model-key")
SPEC = HostSpec(graph_id="researcher", env={"DO_MODEL_ACCESS_KEY": "set-by-the-platform"}, egress=MODEL)


def test_a_host_is_created_with_its_spec_policy_else_the_config_one() -> None:
    client = FakeMicroVMClient()
    host, _, _ = _make_host(client, config=_config(agent_egress=EgressPolicy.allow("api.smith.langchain.com")))
    host.get_or_create(host_id="a", spec=SPEC)
    host.get_or_create(host_id="b", spec=HostSpec(graph_id="echo"))
    assert [c["egress_policy"] for c in client.created] == [MODEL, EgressPolicy.allow("api.smith.langchain.com").for_image(IMAGE)]
    assert client.created[0]["environment"]["DO_MODEL_ACCESS_KEY"] == "set-by-the-platform"

    plain, _, _ = _make_host(client)
    plain.get_or_create(host_id="c", spec=HostSpec(graph_id="echo"))
    assert client.created[-1]["egress_policy"] is None


def test_from_env_reads_the_agent_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DO_MICROVM_EGRESS", raising=False)
    monkeypatch.delenv("DO_MICROVM_EGRESS_CREDENTIALS", raising=False)
    monkeypatch.setenv("DO_MICROVM_AGENT_EGRESS", "*")
    monkeypatch.setenv("DO_MICROVM_AGENT_EGRESS_CREDENTIALS", "https://inference.do-ai.run/v1=MODEL_KEY")
    monkeypatch.setenv("MODEL_KEY", "model-key")
    config = MicroVMConfig.from_env()
    assert config.agent_egress == MODEL and config.egress is None


def test_warm_start_with_credentials_keeps_the_golden_vm_paused(tmp_path: Path) -> None:
    client = FakeMicroVMClient()
    host, _ = make_warm(client, tmp_path)

    host.get_or_create(host_id="h1", spec=SPEC)
    host.get_or_create(host_id="h2", spec=SPEC)

    golden, clone1, clone2 = client.created
    assert golden["egress_policy"] == MODEL and clone1["egress_policy"] is None and clone2["checkpoint_id"]
    golden_vm = next(s for s in client.live() if s.name.startswith(GOLDEN_PREFIX))
    assert golden_vm.status is MicroVMStatus.PAUSED and len(client._checkpoints) == 1


def test_warm_start_without_credentials_deletes_the_golden_vm(tmp_path: Path) -> None:
    client = FakeMicroVMClient()
    host, _ = make_warm(client, tmp_path)
    host.get_or_create(host_id="h1", spec=HostSpec(graph_id="echo", egress=EgressPolicy.allow("pypi.org")))
    assert not any(s.name.startswith(GOLDEN_PREFIX) for s in client.live())


def test_the_policy_is_part_of_the_golden_name(tmp_path: Path) -> None:
    host, _ = make_warm(FakeMicroVMClient(), tmp_path)
    image, size = "img", host.config.agent_size
    names = {
        host._golden_name(HostSpec(graph_id="r"), image, size),
        host._golden_name(HostSpec(graph_id="r", egress=MODEL), image, size),
        host._golden_name(HostSpec(graph_id="r", egress=EgressPolicy.allow_all().with_credential(
            "https://inference.do-ai.run/v1", secret="rotated")), image, size),
    }
    assert len(names) == 3


def test_a_golden_checkpoint_whose_vm_is_gone_is_rebuilt(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    client = FakeMicroVMClient()
    host, _ = make_warm(client, tmp_path)
    host.get_or_create(host_id="h1", spec=SPEC)
    golden_vm = next(s for s in client.live() if s.name.startswith(GOLDEN_PREFIX))
    old = next(iter(client._checkpoints))
    client.delete(golden_vm.id)

    with caplog.at_level(logging.WARNING):
        host.get_or_create(host_id="h2", spec=SPEC)

    assert "rebuilding" in caplog.text
    assert old not in {c.id for c in client.list_checkpoints() if c.microvm_id == golden_vm.id}
    assert client.created[-1]["checkpoint_id"]  # h2 is a clone of the new checkpoint, not a cold host


def test_the_sweeper_keeps_a_golden_vm_while_its_checkpoint_is_kept(tmp_path: Path) -> None:
    client = FakeMicroVMClient()
    host, _ = make_warm(client, tmp_path)
    host.get_or_create(host_id="h1", spec=SPEC)
    golden_vm = next(s for s in client.live() if s.name.startswith(GOLDEN_PREFIX))
    created = datetime(2026, 1, 1, tzinfo=UTC)
    now = created.timestamp() + 3600

    from dataclasses import replace

    for cid, cp in list(client._checkpoints.items()):
        client._checkpoints[cid] = replace(cp, created_at=created)
    client._microvms[golden_vm.id] = replace(client._microvms[golden_vm.id], created_at=created)

    plan = plan_sweep(client, store=None, now=now, checkpoint_ttl=24 * 3600, golden_build_timeout=60)
    assert golden_vm.id in {i.id for i in plan.kept} and not plan.delete_checkpoints

    plan = plan_sweep(client, store=None, now=created.timestamp() + 48 * 3600, checkpoint_ttl=24 * 3600)
    assert golden_vm.id in {i.id for i in plan.delete_microvms} and len(plan.delete_checkpoints) == 1


def test_a_local_host_warns_that_the_policy_is_not_applied(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    host = LocalDevServerHost.__new__(LocalDevServerHost)  # only _provision's first lines run
    with caplog.at_level(logging.WARNING), pytest.raises(AttributeError):
        host._provision("h1", SPEC)
    assert "egress policy is not applied" in caplog.text
