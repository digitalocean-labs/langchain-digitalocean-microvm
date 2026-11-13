"""Tests for agent_host.env: the environment an agent host runs with."""

from __future__ import annotations

import pytest

from langchain_digitalocean_microvm.agent_host import HostSpec
from langchain_digitalocean_microvm.agent_host.env import (
    TOKEN_ENV,
    build_host_env,
    render_env_file,
)


def test_render_env_file_sorts_quotes_and_escapes() -> None:
    rendered = render_env_file(
        {
            "SUBAGENT_HOST_TOKEN": 'tok"with"quotes',
            "DO_MODEL_ACCESS_KEY": "back\\slash",
            "N_JOBS_PER_WORKER": "2",
        }
    )

    assert rendered == (
        'DO_MODEL_ACCESS_KEY="back\\\\slash"\n'
        'N_JOBS_PER_WORKER="2"\n'
        'SUBAGENT_HOST_TOKEN="tok\\"with\\"quotes"\n'
    )


def test_render_env_file_rejects_bad_keys_and_newlines() -> None:
    with pytest.raises(ValueError):
        render_env_file({"BAD-KEY": "x"})
    with pytest.raises(ValueError):
        render_env_file({"OK": "line1\nline2"})
    assert render_env_file({}) == ""


def test_build_host_env_layers_passthrough_then_spec_then_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "supervisor-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "subagents")
    spec = HostSpec(graph_id="g", env={"DO_MODEL_ACCESS_KEY": "scoped", TOKEN_ENV: "spoofed"}, n_jobs_per_worker=3)

    env = build_host_env(spec, "real-token", passthrough=("DO_MODEL_ACCESS_KEY", "LANGSMITH_PROJECT", "UNSET"))

    assert env == {
        "DO_MODEL_ACCESS_KEY": "scoped",
        "LANGSMITH_PROJECT": "subagents",
        TOKEN_ENV: "real-token",  # the spec cannot override the host token
        "N_JOBS_PER_WORKER": "3",
    }
