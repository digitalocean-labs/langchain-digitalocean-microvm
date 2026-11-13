"""scripts/check_setup.py: what is ready, what is missing, and that nothing real is needed to judge it.

The probes are injected, so no network is used. `--deep` runs against the local transport.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fake_client import FakeMicroVMClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_setup  # noqa: E402

from langchain_digitalocean_microvm.exceptions import MicroVMAPIError  # noqa: E402
from langchain_digitalocean_microvm.models import MicroVMSize  # noqa: E402

REAL = {
    "DO_MICROVM_API_TOKEN": "t", "DO_MICROVM_ACCESS_TOKEN": "a", "DO_MICROVM_IMAGE": "img", "DO_MODEL_ACCESS_KEY": "m",
}


def _probes(team: check_setup.Team | Exception | None = None, model: str | None = None, gateway: str | None = None) -> check_setup.Probes:
    def get_team() -> check_setup.Team:
        if isinstance(team, Exception):
            raise team
        return team or check_setup.Team(limit=20, live=3, mine=1, hourly=0.039)

    return check_setup.Probes(team=get_team, model_key=lambda: model, gateway=lambda: gateway)


def _status(report: check_setup.Report) -> dict[str, tuple[str, list[str]]]:
    return {e.number: (status, missing) for e, status, missing in report.examples}


def test_offline_reports_missing_variables_without_touching_the_network() -> None:
    report = check_setup.evaluate({}, None, local=False)
    status = _status(report)
    assert status["01"][0] == "missing"
    assert {"DO_MICROVM_API_TOKEN", "DO_MICROVM_ACCESS_TOKEN", "DO_MICROVM_IMAGE"} <= set(status["01"][1])
    assert any(line.what == "Control token" and line.ok is False for line in report.checks)


def test_a_complete_real_setup_is_ready_and_own_setup_examples_say_what_else() -> None:
    status = _status(check_setup.evaluate(REAL, _probes(), local=False))
    assert status["01"][0] == "ready" and status["03"][0] == "ready"
    assert status["09"][0] == "missing" and "DO_ACTION_GATEWAY_TOKEN" in status["09"][1]
    assert status["02"][0] == "ready*"  # needs the dcode CLI
    assert status["04"][0] == "missing" and "DO_MICROVM_HOST_MODE=microvm" in status["04"][1]


def test_agent_host_examples_need_their_images_and_host_mode() -> None:
    env = {**REAL, "DO_MICROVM_HOST_MODE": "microvm", "DO_MICROVM_AGENT_IMAGE": "reg/host:1",
           "DO_ACTION_GATEWAY_TOKEN": "g", "DO_ACTION_GATEWAY_ACTOR_ID": "me", "DO_INCIDENT_REPO": "o/r",
           "DO_INCIDENT_HOST_IMAGE": "h", "DO_INCIDENT_SPECIALISTS_IMAGE": "s", "DO_INCIDENT_BROKER_URL": "https://x"}
    status = _status(check_setup.evaluate(env, _probes(), local=False))
    assert status["04"][0] == "ready*" and status["09"][0] == "ready*" and status["12"][0] == "ready*"


def test_fake_mode_needs_no_digitalocean_token_but_still_a_model() -> None:
    report = check_setup.evaluate({"DO_MODEL_ACCESS_KEY": "m"}, None, local=True)
    assert _status(report)["03"][0] == "ready" and _status(report)["04"][0] == "ready"
    assert _status(check_setup.evaluate({}, None, local=True))["01"][0] == "missing"
    assert _status(check_setup.evaluate({"DO_MICROVM_TRANSPORT": "local", "DO_MODEL_ACCESS_KEY": "m"}, None, local=False))["01"][0] == "ready"


def test_the_exec_transport_does_not_need_the_access_token() -> None:
    env = {k: v for k, v in REAL.items() if k != "DO_MICROVM_ACCESS_TOKEN"}
    assert _status(check_setup.evaluate({**env, "DO_MICROVM_TRANSPORT": "exec"}, _probes(), local=False))["01"][0] == "ready"
    assert _status(check_setup.evaluate(env, _probes(), local=False))["01"][0] == "missing"


def test_a_team_without_room_is_reported_per_example() -> None:
    full = check_setup.Team(limit=20, live=19, mine=0, hourly=None)
    status = _status(check_setup.evaluate(REAL, _probes(team=full), local=False))
    assert status["01"][0] == "ready" and status["03"][0] == "no room" and "has 1 free" in status["03"][1][0]


@pytest.mark.parametrize(("code", "word"), [(401, "wrong or expired"), (403, "microvm:read")])
def test_a_refused_token_says_why(code: int, word: str) -> None:
    report = check_setup.evaluate(REAL, _probes(team=MicroVMAPIError(code, "no")), local=False)
    line = next(line for line in report.checks if line.what == "Control token")
    assert line.ok is False and word in line.detail
    assert _status(report)["01"][0] == "ready"  # the variables are there; the failed check is what fails the run


def test_a_refused_model_key_and_a_bad_namespace_fail() -> None:
    report = check_setup.evaluate({**REAL, "DO_MICROVM_NAMESPACE": "Bad-Name"}, _probes(model="401: refused"), local=False)
    assert next(line for line in report.checks if line.what == "Namespace").ok is False
    assert next(line for line in report.checks if line.what.startswith("Model")).ok is False


def test_the_model_can_come_from_another_provider() -> None:
    assert check_setup.model_variables({"EXAMPLES_MODEL": "openai:gpt-5", "OPENAI_API_KEY": "k"}) == (True, "")
    ok, why = check_setup.model_variables({"EXAMPLES_MODEL": "openai:gpt-5"})
    assert not ok and "OPENAI_API_KEY" in why


def test_the_hourly_price_comes_from_the_options() -> None:
    options = {"resource_pricing": {"price_per_vcpu_hour": 0.022, "price_per_gib_memory_hour": 0.0085}}
    assert check_setup.hourly_price(options, MicroVMSize(1, 2048)) == pytest.approx(0.039)
    assert check_setup.hourly_price({}, MicroVMSize(1, 2048)) is None


def test_the_report_never_contains_a_secret() -> None:
    env = {**REAL, "DO_MICROVM_API_TOKEN": "dop_v1_secret", "DO_MODEL_ACCESS_KEY": "sk-secret"}
    text = check_setup.render(check_setup.evaluate(env, _probes(), local=False), local=False)
    assert "secret" not in text and "ready" in text


def test_deep_uses_a_throwaway_sandbox_and_always_deletes_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from langchain_digitalocean_microvm import MicroVMLifecycle
    from langchain_digitalocean_microvm.agent_host import store

    monkeypatch.setattr(store, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(check_setup, "MicroVMClient", lambda token: FakeMicroVMClient())
    deleted: list[str] = []
    real_delete = MicroVMLifecycle.delete

    def delete(self: MicroVMLifecycle, *, sandbox_id: str) -> None:
        deleted.append(sandbox_id)
        real_delete(self, sandbox_id=sandbox_id)

    monkeypatch.setattr(MicroVMLifecycle, "delete", delete)
    probes, _ = check_setup.live_probes({"DO_MICROVM_API_TOKEN": "t", "DO_MICROVM_TRANSPORT": "local"}, deep=True)
    assert probes.deep is not None and probes.deep() is None
    assert len(deleted) == 1 and deleted[0].startswith("check-")

    def boom(self: MicroVMLifecycle, **kwargs: object) -> None:
        raise RuntimeError("no room")

    monkeypatch.setattr(MicroVMLifecycle, "get_or_create", boom)
    assert "no room" in (probes.deep() or "")
    assert len(deleted) == 2  # deleted although the check failed
