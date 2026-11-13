"""ActionPolicy from a mapping or a YAML file: the same policy as the Python form, and the mistakes it refuses."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGateway,
    ActionPolicy,
)
from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    LocalActionGatewayClient,
)

YAML = """\
allow:
  - github_list_commits@5
  - digitalocean_droplet-get@3
ask:
  - digitalocean_restore-droplet@3
deny:
  - digitalocean_droplet-delete@3
"""
PYTHON = ActionPolicy(
    allow=["github_list_commits@5", "digitalocean_droplet-get@3"],
    ask=["digitalocean_restore-droplet@3"],
    deny=["digitalocean_droplet-delete@3"],
)


def test_yaml_file_gives_the_python_policy(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(YAML)
    for source in (path, str(path)):
        policy = ActionPolicy.from_yaml(source)
        assert policy == PYTHON
        assert policy.to_spec() == PYTHON.to_spec()  # versions dropped, meta tools allowed, default deny


def test_yaml_stream() -> None:
    assert ActionPolicy.from_yaml(io.StringIO(YAML)) == PYTHON


def test_missing_tiers_are_empty() -> None:
    assert ActionPolicy.from_mapping({"deny": ["x"]}) == ActionPolicy(deny=["x"])
    assert ActionPolicy.from_mapping({}) == ActionPolicy()
    assert ActionPolicy.from_mapping({"allow": None, "ask": []}) == ActionPolicy()


def test_a_mistyped_path_is_not_parsed_as_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ActionPolicy.from_yaml(str(tmp_path / "polcy.yaml"))
    with pytest.raises(FileNotFoundError):
        ActionPolicy.from_yaml("allow: [x]")  # YAML text is not accepted


def test_the_file_name_is_in_the_error(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("allow: x\n")
    with pytest.raises(ValueError, match=r"policy\.yaml: 'allow' must be a list"):
        ActionPolicy.from_yaml(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("- a\n- b\n", "expected a mapping"),
        ("allow: [a\n", "not valid YAML"),
        ("!!python/object/apply:os.getcwd []\n", "not valid YAML"),
        ("default: allow\n", "the default is always deny"),
        ("permissions: {default: deny}\n", "not of an ActionPolicy"),
        ("rules: []\n", "not of an ActionPolicy"),
        ("allwo: [a]\n", "unknown key 'allwo'"),
        ("allow: tool_a\n", "must be a list"),
        ("allow: {a: 1}\n", "must be a list"),
        ("allow: [1]\n", "tool names without spaces"),
        ("allow: ['']\n", "tool names without spaces"),
        ("allow: ['two words']\n", "tool names without spaces"),
        ("allow: ['@3']\n", "no tool name before the version"),
        ("allow: [do.actions/digitalocean_droplet-get]\n", "do.actions/"),
        ("deny: [action_invoke]\n", "always allowed"),
        ("allow: [action_search@1]\n", "always allowed"),
        ("allow: [a]\ndeny: [a]\n", "'a' is listed in 'allow' and 'deny'"),
        ("allow: [a@1]\nask: [a@2]\n", "'a' is listed in 'allow' and 'ask'"),
        ("ask: [a, a@1]\n", "twice in 'ask'"),
    ],
)
def test_mistakes_are_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ActionPolicy.from_yaml(io.StringIO(text))


def test_a_loaded_policy_drives_a_gateway() -> None:
    policy = ActionPolicy.from_yaml(io.StringIO(YAML))
    slugs = [*policy.allow, *policy.ask, *policy.deny]
    gateway = ActionGateway(policy, tools=slugs, client=LocalActionGatewayClient())
    assert gateway.tool_slugs == tuple(slugs)
    assert {t.name for t in gateway.tools()} == {"action_search", "action_invoke"}
    gateway.close()
