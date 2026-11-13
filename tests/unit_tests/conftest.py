"""Every test runs as the same owner and never touches the real state directory."""

from __future__ import annotations

import pytest

from langchain_digitalocean_microvm import owner

TEST_OWNER = "0123456789"


@pytest.fixture(autouse=True)
def _fixed_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "_cached", TEST_OWNER)
