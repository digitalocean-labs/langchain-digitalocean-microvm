"""The gateway store lives in the platform's per-user state directory; an older store is still used."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from langchain_digitalocean_microvm.agent_host import store as store_module
from langchain_digitalocean_microvm.agent_host.store import (
    default_db_path,
    legacy_state_dir,
    state_dir,
    store_dirs,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for name in ("XDG_STATE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", "Library/Application Support/langchain_digitalocean_microvm"),
        ("linux", ".local/state/langchain_digitalocean_microvm"),
        ("win32", "AppData/Local/langchain_digitalocean_microvm"),
    ],
)
def test_state_dir_follows_the_platform(home: Path, monkeypatch: pytest.MonkeyPatch, platform: str, expected: str) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert state_dir() == home / expected
    assert store_dirs() == [home / expected, home / ".langchain_digitalocean_microvm"]


def test_xdg_state_home_and_localappdata_are_honoured(home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert state_dir() == tmp_path / "xdg" / "langchain_digitalocean_microvm"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert state_dir() == tmp_path / "local" / "langchain_digitalocean_microvm"


def test_a_fresh_install_uses_the_state_dir(home: Path) -> None:
    assert default_db_path("microvm") == state_dir() / "gateway.sqlite"
    assert default_db_path("local") == state_dir() / "gateway-local.sqlite"
    assert default_db_path("microvm", "alice") == state_dir() / "gateway-alice.sqlite"
    assert default_db_path("local", "alice") == state_dir() / "gateway-local-alice.sqlite"


def test_a_store_in_the_old_folder_keeps_being_used_until_one_exists_in_the_new(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    legacy = legacy_state_dir() / "gateway.sqlite"
    legacy.parent.mkdir()
    legacy.touch()
    with caplog.at_level(logging.INFO, logger=store_module.__name__):
        assert default_db_path("microvm") == legacy
    assert "move it to" in caplog.text
    assert default_db_path("microvm", "alice") == state_dir() / "gateway-alice.sqlite"  # no old file of that name

    (state_dir()).mkdir(parents=True)
    (state_dir() / "gateway.sqlite").touch()
    assert default_db_path("microvm") == state_dir() / "gateway.sqlite"  # the moved file wins; nothing was moved for us
    assert legacy.exists()
