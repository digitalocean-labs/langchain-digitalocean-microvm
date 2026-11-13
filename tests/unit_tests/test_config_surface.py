"""`config.py` is the one place that reads `DO_MICROVM_*` environment variables."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from langchain_digitalocean_microvm import config
from langchain_digitalocean_microvm.agent_host import local, supervisor

SRC = Path(config.__file__).parent
ENV_NAME_CONSTANTS = {name for name in dir(config) if name.endswith("_ENV")}


def _environment_reads(path: Path) -> list[tuple[int, str]]:
    """(line, what) of every os.environ.get / os.getenv / os.environ[...] naming a DO_MICROVM_* variable."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        key: ast.expr | None = None
        if isinstance(node, ast.Call) and node.args:
            func = ast.unparse(node.func)
            if func in ("os.environ.get", "os.getenv", "os.environ.pop", "os.environ.setdefault"):
                key = node.args[0]
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            key = node.slice
        if key is None:
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.startswith("DO_MICROVM_"):
            found.append((node.lineno, ast.unparse(node)))
        elif isinstance(key, ast.Name) and key.id in ENV_NAME_CONSTANTS:
            found.append((node.lineno, ast.unparse(node)))
    return found


def test_only_config_reads_do_microvm_variables() -> None:
    offenders = {
        str(path.relative_to(SRC)): reads
        for path in SRC.rglob("*.py")
        if path.name != "config.py" and (reads := _environment_reads(path))
    }
    assert offenders == {}, "read DO_MICROVM_* through the readers in config.py (env_value, env_flag, env_int, env_csv)"


def test_the_settings_modules_share_config_s_variable_names() -> None:
    assert supervisor.HOST_MODE_ENV is config.HOST_MODE_ENV == "DO_MICROVM_HOST_MODE"
    assert supervisor.GATEWAY_TOKEN_ENV is config.GATEWAY_TOKEN_ENV == "DO_MICROVM_GATEWAY_TOKEN"
    assert local.TEMPLATE_DIR_ENV is config.TEMPLATE_DIR_ENV == "DO_MICROVM_SUBAGENT_TEMPLATE"
    assert local.PASSTHROUGH_ENV is config.PASSTHROUGH_ENV == "DO_MICROVM_SUBAGENT_ENV_PASSTHROUGH"


def test_the_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_TEST", raising=False)
    assert config.env_value("X_TEST") is None and config.env_int("X_TEST", 7) == 7
    assert config.env_csv("X_TEST") == () and config.env_flag("X_TEST") is False
    monkeypatch.setenv("X_TEST", "")
    assert config.env_value("X_TEST") is None and config.env_int("X_TEST", 7) == 7
    monkeypatch.setenv("X_TEST", " a, b ,,c ")
    assert config.env_csv("X_TEST") == ("a", "b", "c") and config.env_value("X_TEST") == " a, b ,,c "
    for on in ("1", "TRUE", " yes ", "On"):
        monkeypatch.setenv("X_TEST", on)
        assert config.env_flag("X_TEST") is True
    monkeypatch.setenv("X_TEST", "0")
    assert config.env_flag("X_TEST") is False and config.env_int("X_TEST", 7) == 0


def test_the_module_docstring_names_every_variable() -> None:
    doc = config.__doc__ or ""
    names = {
        node.value
        for node in ast.walk(ast.parse(Path(config.__file__).read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"DO_MICROVM_[A-Z_]+", node.value)
    }
    assert len(names) >= 15
    # the table abbreviates families (`DO_MICROVM_IMAGE`, `_HTTP_PORT`, ...): the part after the prefix must appear
    missing = sorted(n for n in names if n not in doc and n.removeprefix("DO_MICROVM") not in doc)
    assert missing == []
