import pytest

import langchain_digitalocean_microvm.transport  # noqa: F401  (registers the exec, endpoint and local builtins)
from langchain_digitalocean_microvm.transport.registry import (
    available_transports,
    create_transport,
    register_transport,
)


class _FakeInstance:
    pass


def test_register_and_create_transport() -> None:
    register_transport("fresh-fake", _FakeInstance)
    transport = create_transport("fresh-fake")
    assert isinstance(transport, _FakeInstance)


def test_register_duplicate_without_overwrite_raises() -> None:
    register_transport("dup-fake", _FakeInstance)
    with pytest.raises(ValueError):
        register_transport("dup-fake", _FakeInstance)


def test_register_duplicate_with_overwrite_succeeds() -> None:
    register_transport("overwrite-fake", _FakeInstance)
    register_transport("overwrite-fake", _FakeInstance, overwrite=True)


def test_create_unknown_transport_raises() -> None:
    with pytest.raises(ValueError):
        create_transport("no-such-transport")


def test_available_transports_includes_builtins() -> None:
    names = available_transports()
    assert {"exec", "local"} <= set(names)
