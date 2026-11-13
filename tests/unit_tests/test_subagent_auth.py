"""Tests for the in-VM auth handler (images/agent-host/auth.py): env token vs token file."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

AUTH_PATH = Path(__file__).resolve().parents[2] / "images" / "agent-host" / "auth.py"


@pytest.fixture
def auth(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBAGENT_HOST_TOKEN", "env-token")
    monkeypatch.setenv("SUBAGENT_HOST_TOKEN_FILE", str(tmp_path / "token"))
    spec = importlib.util.spec_from_file_location("subagent_auth_under_test", AUTH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _ok(auth, token: str) -> bool:
    try:
        asyncio.run(auth.authenticate({b"x-api-key": token.encode()}))
    except auth.Auth.exceptions.HTTPException:
        return False
    return True


def test_env_token_until_a_token_file_appears_then_only_the_file(auth, tmp_path) -> None:
    assert _ok(auth, "env-token") and not _ok(auth, "other")
    (tmp_path / "token").write_text("clone-token\n")
    assert _ok(auth, "clone-token")
    assert not _ok(auth, "env-token")  # the inherited token stops working


def test_a_rewritten_token_file_is_picked_up(auth, tmp_path) -> None:
    path = tmp_path / "token"
    path.write_text("first")
    assert _ok(auth, "first")
    path.write_text("second")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert _ok(auth, "second") and not _ok(auth, "first")


def test_no_token_at_all_rejects_everything(auth, monkeypatch) -> None:
    monkeypatch.delenv("SUBAGENT_HOST_TOKEN")
    assert not _ok(auth, "")
