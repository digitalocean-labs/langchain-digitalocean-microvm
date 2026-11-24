"""Outbound policy (egress.py): validation, the API body, the config, and how sandboxes and templates use it."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import httpx
import pytest
from fake_client import FakeMicroVMClient
from test_client import Recorder, fixture, make_client

from langchain_digitalocean_microvm import (
    EgressPolicy,
    MicroVMConfig,
    MicroVMLifecycle,
    OutboundCredential,
)
from langchain_digitalocean_microvm.config import egress_from_env
from langchain_digitalocean_microvm.exceptions import CheckpointUnavailableError
from langchain_digitalocean_microvm.provider import TEMPLATE_BUILD_PREFIX
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.template_cache import TemplateCache
from langchain_digitalocean_microvm.transport.local import LocalTransport

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import delete_all  # noqa: E402

IMAGE = "registry.example/sandbox:1"
GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


@pytest.fixture(autouse=True)
def _isolated_fake_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _real(client: FakeMicroVMClient, **overrides) -> MicroVMLifecycle:
    config = MicroVMConfig(**{"transport": "exec", "api_token": "t", "image": IMAGE, "ready_timeout": 5.0, **overrides})
    return MicroVMLifecycle(config, client=client, transport_factory=LocalTransport)  # type: ignore[arg-type]


# ---------------------------------------------------------------- the policy


def test_allow_builds_the_documented_api_body() -> None:
    policy = EgressPolicy.allow("PyPI.org", "*.example.org", "203.0.113.5", "pypi.org")
    assert policy.to_api() == {
        "default_action": "deny",
        "rules": [{"domain": "pypi.org"}, {"domain": "*.example.org"}, {"ip": "203.0.113.5"}],
    }
    assert EgressPolicy.deny_all().to_api() == {"default_action": "deny", "rules": []}
    assert EgressPolicy.allow_all().to_api() == {"default_action": "allow", "rules": []}


def test_with_credential_allows_its_host_and_never_shows_the_secret() -> None:
    policy = EgressPolicy.allow("pypi.org").with_credential("https://api.example.com/v1/", secret="s3cret")
    assert policy.rules == ("pypi.org", "api.example.com")
    assert policy.to_api()["credentials"] == [
        {"host": "api.example.com", "scheme": "bearer", "secret": "s3cret", "path_prefix": "/v1"}
    ]
    assert "s3cret" not in repr(policy)
    raw = EgressPolicy.allow_all().with_credential("https://api.example.com", secret="k", scheme="raw")
    assert raw.rules == () and raw.to_api()["credentials"] == [{"host": "api.example.com", "scheme": "raw", "secret": "k"}]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: EgressPolicy.allow("https://pypi.org"), "not an exact hostname"),
        (lambda: EgressPolicy.allow("pypi.org:443"), "not an exact hostname"),
        (lambda: EgressPolicy.allow("10.0.0.0/8"), "not an exact hostname"),
        (lambda: EgressPolicy.allow(*(f"h{i}.example.com" for i in range(51))), "at most 50 rules"),
        (lambda: EgressPolicy.allow().with_credential("http://api.example.com", secret="k"), "https://"),
        (lambda: EgressPolicy.allow().with_credential("https://api.example.com:8443", secret="k"), "host and a path"),
        (lambda: EgressPolicy.allow().with_credential("https://api.example.com", secret=""), "empty secret"),
        (lambda: EgressPolicy.allow().with_credential("https://api.example.com", secret="k" * 8193), "8192"),
        (
            lambda: EgressPolicy.allow().with_credential("https://a.example.com/x", secret="1")
            .with_credential("https://a.example.com/y", secret="2"),
            "one credential per host",
        ),
        (
            lambda: EgressPolicy(rules=("*.example.com",), credentials=(OutboundCredential("a.example.com", "k"),)),
            "allowed by an exact rule",
        ),
        (lambda: OutboundCredential("*.example.com", "k"), "exact hostname"),
    ],
)
def test_invalid_policies_are_refused_before_any_api_call(build, message) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_parse_reads_the_variable_forms() -> None:
    assert EgressPolicy.parse("") is None
    assert EgressPolicy.parse(" none ") == EgressPolicy.deny_all()
    assert EgressPolicy.parse("*") == EgressPolicy.allow_all()
    assert EgressPolicy.parse("pypi.org, files.pythonhosted.org,") == EgressPolicy.allow("pypi.org", "files.pythonhosted.org")


def test_allows_matches_wildcards_below_their_domain_only() -> None:
    policy = EgressPolicy.allow("*.example.com")
    assert policy.allows("api.example.com") and not policy.allows("example.com")
    assert EgressPolicy.allow_all().allows("anything.test")


def test_digest_changes_with_the_secret_but_does_not_contain_it() -> None:
    base = EgressPolicy.allow("pypi.org")
    one = base.with_credential("https://api.example.com", secret="one")
    two = base.with_credential("https://api.example.com", secret="two")
    assert len({base.digest(b"k"), one.digest(b"k"), two.digest(b"k"), one.digest(b"other")}) == 4
    assert one.digest(b"k") == base.with_credential("https://api.example.com", secret="one").digest(b"k")


# ---------------------------------------------------------------- config


def test_from_env_reads_the_policy_and_the_named_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DO_MICROVM_EGRESS", raising=False)
    monkeypatch.delenv("DO_MICROVM_EGRESS_CREDENTIALS", raising=False)
    assert egress_from_env() is None and MicroVMConfig.from_env().egress is None

    monkeypatch.setenv("DO_MICROVM_EGRESS", "pypi.org")
    monkeypatch.setenv("DO_MICROVM_EGRESS_CREDENTIALS", "https://api.example.com/v1=EXAMPLE_KEY")
    monkeypatch.setenv("EXAMPLE_KEY", "s3cret")
    policy = MicroVMConfig.from_env().egress
    assert policy == EgressPolicy.allow("pypi.org").with_credential("https://api.example.com/v1", secret="s3cret")

    monkeypatch.delenv("DO_MICROVM_EGRESS")
    policy = egress_from_env()
    assert policy is not None and policy.default_action == "allow" and len(policy.credentials) == 1

    monkeypatch.delenv("EXAMPLE_KEY")
    with pytest.raises(ValueError, match="'EXAMPLE_KEY', which is unset"):
        egress_from_env()
    monkeypatch.setenv("DO_MICROVM_EGRESS_CREDENTIALS", "https://api.example.com")
    with pytest.raises(ValueError, match="https-url=VARIABLE"):
        egress_from_env()


# ---------------------------------------------------------------- client


def test_create_from_image_sends_the_policy() -> None:
    from langchain_digitalocean_microvm.models import MicroVMSize

    rec = Recorder({("POST", "/v2/microvms"): httpx.Response(201, json=fixture("create_201"))})
    policy = EgressPolicy.allow("pypi.org").with_credential("https://api.example.com", secret="k")
    make_client(rec).create(name="x", oci_ref="img", region="mkc1", size=MicroVMSize.parse("1vcpu-2gb"), egress_policy=policy)
    assert rec.bodies()[0]["egress_policy"] == policy.to_api()


def test_create_from_checkpoint_refuses_a_policy() -> None:
    with pytest.raises(ValueError, match="inherits its egress policy"):
        make_client(Recorder({})).create(name="x", checkpoint_id="cp", egress_policy=EgressPolicy.deny_all())


# ---------------------------------------------------------------- sandboxes and templates


def test_sandboxes_use_the_config_policy_unless_the_call_sets_one() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client, egress=EgressPolicy.allow("pypi.org"))
    lifecycle.get_or_create(sandbox_id="a")
    lifecycle.get_or_create(sandbox_id="b", egress=EgressPolicy.deny_all())
    assert [c["egress_policy"] for c in client.created] == [EgressPolicy.allow("pypi.org"), EgressPolicy.deny_all()]
    _real(client).get_or_create(sandbox_id="c")
    assert client.created[-1]["egress_policy"] is None


def test_a_clone_inherits_the_template_policy() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client, egress=EgressPolicy.allow("pypi.org"))
    template = lifecycle.build_template("analyst")
    lifecycle.get_or_create(sandbox_id="warm", template=template)
    assert client.created[0]["egress_policy"] == EgressPolicy.allow("pypi.org")
    assert client.created[-1]["checkpoint_id"] == template.checkpoint_id and client.created[-1]["egress_policy"] is None
    with pytest.raises(ValueError, match="template's egress policy"):
        lifecycle.get_or_create(sandbox_id="x", template=template, egress=EgressPolicy.deny_all())


def test_the_policy_is_part_of_the_template_key() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)
    plain = lifecycle.build_template("analyst")
    allowed = lifecycle.build_template("analyst", egress=EgressPolicy.allow("pypi.org"))
    one = lifecycle.build_template("analyst", egress=EgressPolicy.allow_all().with_credential("https://a.example.com", secret="1"))
    two = lifecycle.build_template("analyst", egress=EgressPolicy.allow_all().with_credential("https://a.example.com", secret="2"))
    assert len({plain.checkpoint_name, allowed.checkpoint_name, one.checkpoint_name, two.checkpoint_name}) == 4
    assert lifecycle.find_template("analyst", egress=EgressPolicy.allow("pypi.org")) == allowed.__class__(
        allowed.name, allowed.checkpoint_name, allowed.checkpoint_id, built=False
    )


def test_a_template_with_credentials_keeps_its_origin_paused_until_it_is_deleted() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)
    policy = EgressPolicy.allow("pypi.org").with_credential("https://api.example.com", secret="k")

    template = lifecycle.build_template("analyst", egress=policy)

    (origin,) = client.live()
    assert origin.name.startswith(f"{TEMPLATE_BUILD_PREFIX}-") and origin.status is MicroVMStatus.PAUSED
    lifecycle.delete_template(template)
    assert client.live() == [] and client.list_checkpoints() == []


def test_a_template_without_credentials_deletes_its_origin() -> None:
    client = FakeMicroVMClient()
    template = _real(client).build_template("analyst", egress=EgressPolicy.allow("pypi.org"))
    assert client.live() == []
    _real(client).delete_template(template)  # nothing to delete but the checkpoint
    assert client.list_checkpoints() == []


def test_rebuild_replaces_the_kept_origin() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)
    policy = EgressPolicy.allow_all().with_credential("https://api.example.com", secret="k")
    lifecycle.build_template("analyst", egress=policy)
    first_origin = client.live()[0].id

    rebuilt = lifecycle.build_template("analyst", egress=policy, rebuild=True)

    assert rebuilt.built and client.get(first_origin).status in GONE
    (origin,) = client.live()
    assert [c.microvm_id for c in client.list_checkpoints()] == [origin.id]


def test_delete_all_keeps_the_origin_of_a_kept_template() -> None:
    client = FakeMicroVMClient()
    lifecycle = _real(client)
    lifecycle.build_template("analyst", egress=EgressPolicy.allow_all().with_credential("https://a.example.com", secret="k"))
    lifecycle.get_or_create(sandbox_id="t1")

    microvms, _, kept = delete_all.find(client, include_templates=False)
    assert [s.name for s in microvms] == [lifecycle._name("t1")] and len(kept) == 1
    microvms, checkpoints, _ = delete_all.find(client, include_templates=True)
    assert len(microvms) == 2 and len(checkpoints) == 1


def test_local_mode_warns_that_the_policy_is_not_applied(caplog: pytest.LogCaptureFixture) -> None:
    lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local", egress=EgressPolicy.deny_all()))
    with caplog.at_level(logging.WARNING):
        lifecycle.get_or_create(sandbox_id="a")
        lifecycle.get_or_create(sandbox_id="b")
    assert sum("does not apply an egress policy" in r.message for r in caplog.records) == 1


# ---------------------------------------------------------------- a template whose origin was deleted


def _credentials_template(client: FakeMicroVMClient):
    lifecycle = _real(client)
    policy = EgressPolicy.allow_all().with_credential("https://api.example.com", secret="k")
    template = lifecycle.build_template("analyst", egress=policy)
    (origin,) = client.live()
    return lifecycle, policy, template, origin


def test_a_template_whose_origin_was_deleted_is_not_reused() -> None:
    client = FakeMicroVMClient()
    lifecycle, policy, template, origin = _credentials_template(client)
    client.delete(origin.id)  # by someone else: the service still lists the checkpoint as available

    assert lifecycle.find_template("analyst", egress=policy) is None
    with pytest.raises(CheckpointUnavailableError):
        lifecycle.get_or_create(sandbox_id="t1", template=template)
    rebuilt = lifecycle.build_template("analyst", egress=policy)

    assert rebuilt.built and template.checkpoint_id not in {c.id for c in client.list_checkpoints()} - {rebuilt.checkpoint_id}
    lifecycle.get_or_create(sandbox_id="t1", template=rebuilt)


def test_template_cache_rebuilds_when_the_origin_is_gone() -> None:
    client = FakeMicroVMClient()
    lifecycle, policy, _, origin = _credentials_template(client)
    cache = TemplateCache(lambda: lifecycle.build_template("analyst", egress=policy))
    cache.get()
    client.delete(origin.id)

    backend = cache.create(lambda t: lifecycle.get_or_create(sandbox_id="t1", template=t))

    assert client.get(backend.id).status is MicroVMStatus.RUNNING


def test_the_client_names_an_unrestorable_checkpoint() -> None:
    message = "the origin MicroVM's egress credentials are incomplete; this checkpoint cannot be restored"
    rec = Recorder({("POST", "/v2/microvms"): httpx.Response(400, json={"id": "failed_precondition", "message": message})})
    with pytest.raises(CheckpointUnavailableError, match="cannot be restored"):
        make_client(rec).create(name="x", checkpoint_id="cp")


def test_a_docr_image_gets_the_registry_token_host_under_deny() -> None:
    docr = "registry.digitalocean.com/team/sandbox:1"
    assert EgressPolicy.allow("pypi.org").for_image(docr).rules == ("pypi.org", "api.digitalocean.com")
    assert EgressPolicy.deny_all().for_image(docr).rules == ("api.digitalocean.com",)
    assert EgressPolicy.allow_all().for_image(docr) == EgressPolicy.allow_all()
    assert EgressPolicy.allow("pypi.org").for_image("docker.io/library/python:3.12") == EgressPolicy.allow("pypi.org")

    client = FakeMicroVMClient()
    lifecycle = _real(client, image=docr)
    lifecycle.get_or_create(sandbox_id="a", egress=EgressPolicy.allow("pypi.org"))
    lifecycle.build_template("analyst", egress=EgressPolicy.deny_all())
    assert [c["egress_policy"].rules for c in client.created] == [("pypi.org", "api.digitalocean.com"), ("api.digitalocean.com",)]
