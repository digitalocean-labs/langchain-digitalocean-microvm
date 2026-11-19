"""scripts/setup_example_09.py: the token checks, the .env writes and the dry run, without network."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import setup_example_09 as setup  # noqa: E402


class _Response:
    def __init__(self, status_code: int, body: object = None) -> None:
        self.status_code = status_code
        self.body = body

    def json(self) -> object:
        return self.body


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], int]:
    """(url suffix, token) -> status; anything else answers 401."""
    answers: dict[tuple[str, str], int] = {}

    def get(url: str, *, headers: dict[str, str], **_: object) -> _Response:
        token = headers["Authorization"].removeprefix("Bearer ")
        return _Response(next((s for (suffix, t), s in answers.items() if url.endswith(suffix) and t == token), 401))

    def post(url: str, *, headers: dict[str, str], **_: object) -> _Response:
        token = headers["Authorization"].removeprefix("Bearer ")
        return _Response(400 if token == "control" else 403 if token == "access" else 401)

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(setup.Setup, "team_room", lambda self: None)  # MicroVMClient, not httpx.get
    monkeypatch.setattr(setup, "_model_probe", lambda key, model: None if key == "model" else "401: refused")
    return answers


def test_access_token_must_not_manage_microvms(api: dict[tuple[str, str], int]) -> None:
    api[("/microvms", "control")] = 200
    assert setup.check_access("control", {"DO_MICROVM_API_TOKEN": "control"}) is not None  # the same token
    assert "microvm:access only" in (setup.check_access("control", {}) or "")  # can create MicroVMs
    assert setup.check_access("access", {}) is None  # 403 on create
    assert setup.check_access("wrong", {}) is not None  # 401


def test_control_and_gateway_checks(api: dict[tuple[str, str], int]) -> None:
    api[("/microvms", "control")] = 200
    api[("/action-gateway/tools", "gw")] = 200
    assert setup.check_control("control", {}) is None
    assert setup.check_control("other", {}) is not None
    assert setup.check_gateway("gw", {}) is None
    assert setup.check_gateway("control", {}) is not None


def test_model_key_checks_both_models(api: dict[tuple[str, str], int], monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(setup, "_model_probe", lambda key, model: asked.append(model))
    assert setup.check_model_key("model", {}) is None
    assert asked == list(setup.MODELS)


def test_save_writes_env_private(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# DO_MICROVM_IMAGE=\nOTHER=1\n")
    env_file.chmod(0o644)
    s = setup.Setup(env_file, dry_run=False)
    s.save("DO_MICROVM_IMAGE", "registry.example/img:1")
    s.save("OTHER", "2")
    text = env_file.read_text()
    assert "DO_MICROVM_IMAGE=registry.example/img:1" in text and "OTHER=2" in text and "OTHER=1" not in text
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_dry_run_changes_nothing(api: dict[tuple[str, str], int], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api[("/microvms", "control")] = 200
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_API_TOKEN=control\nDO_MODEL_ACCESS_KEY=model\n")
    before = env_file.read_text()
    assert setup.main(["--dry-run", "--env-file", str(env_file)]) == 1
    out = capsys.readouterr().out
    assert env_file.read_text() == before
    assert "DO_MICROVM_ACCESS_TOKEN" in out and "microvm:access only" in out
    assert "DO_ACTION_GATEWAY_TOKEN" in out and "To do:" in out


@pytest.mark.parametrize("extra", ["/droplets", "/account"])
def test_broader_tokens_refused_and_never_printed(api: dict[tuple[str, str], int], tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str], extra: str) -> None:
    api[("/microvms", "sekrit-control")] = 200
    api[(extra, "sekrit-control")] = 200  # Full Access, Read Only or an extra scope
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_API_TOKEN=sekrit-control\nDO_MODEL_ACCESS_KEY=model\n")
    assert setup.main(["--dry-run", "--env-file", str(env_file)]) == 1
    out = capsys.readouterr().out
    assert "sekrit" not in out
    assert "[need] DO_MICROVM_API_TOKEN: this token can also read droplets or the account" in out


def test_missing_tool_stops_before_any_prompt(api: dict[tuple[str, str], int], tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda command: None if command == "gh" else f"/bin/{command}")
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(setup.Setup, "tokens", lambda self: pytest.fail("asked for tokens"))
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_IMAGE=img\n")
    assert setup.main(["--env-file", str(env_file)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] gh: not on PATH" in out and "https://cli.github.com" in out
    assert "crane" not in out.split("0. Prerequisites")[1]  # the image is set: crane is not needed


def _gh(answers: dict[str, str | None]):  # noqa: ANN202 - a fake _run
    def run(command: list[str]) -> str | None:
        return answers.get(command[2]) if command[:2] == ["gh", "api"] else None
    return run


def test_verify_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda command: f"/bin/{command}")
    good = {"repos/o/r": "public", "repos/o/r/commits?per_page=2": "2", "repos/o/r/contents/service/app.py": "app.py"}
    monkeypatch.setattr(setup, "_run", _gh(good))
    assert setup.verify_repo("o/r")[0] is True
    monkeypatch.setattr(setup, "_run", _gh({**good, "repos/o/r": "private"}))
    assert "must be public" in setup.verify_repo("o/r")[1]
    monkeypatch.setattr(setup, "_run", _gh({**good, "repos/o/r/commits?per_page=2": "1"}))
    assert setup.verify_repo("o/r")[0] is False
    monkeypatch.setattr(setup, "_run", _gh({**good, "repos/o/r/contents/service/app.py": None}))
    assert "not the demo" in setup.verify_repo("o/r")[1]
    assert setup.verify_repo("-o/r")[0] is False
    monkeypatch.setattr(setup.shutil, "which", lambda command: None)
    assert setup.verify_repo("o/r")[0] is None


def test_verify_image_private_needs_registry_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda command: f"/bin/{command}")
    assert setup.verify_image("registry.digitalocean.com/reg/img:1", None)[0] is None
    monkeypatch.setattr(setup.shutil, "which", lambda command: None)
    assert setup.verify_image("docker.io/library/python:3.12-slim", None)[0] is None


@pytest.mark.parametrize(("body", "ok"), [({"status": "healthy"}, True), ({"status": "failing"}, True),
                                          ({"status": "starting"}, False), (None, False)])
def test_verify_droplet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: object, ok: bool) -> None:
    env = tmp_path / "incident.env"
    env.write_text("DO_INCIDENT_URL=http://203.0.113.10:8080\n")
    monkeypatch.setattr(setup, "DROPLET_ENV", env)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(httpx, "get", lambda url, **_: _Response(200, body))
    assert setup.verify_droplet()[0] is ok


def test_dry_run_ready_when_everything_checks(api: dict[tuple[str, str], int], tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    api[("/microvms", "control")] = 200
    api[("/action-gateway/tools", "gw")] = 200
    state = tmp_path / "incident.json"
    state.write_text("{}")
    monkeypatch.setattr(setup, "DROPLET_STATE", state)
    monkeypatch.setattr(setup, "DROPLET_ENV", tmp_path / "incident.env")
    monkeypatch.setattr(setup, "verify_repo", lambda repo: (True, "demo"))
    monkeypatch.setattr(setup, "verify_image", lambda image, token: (True, "found"))
    monkeypatch.setattr(setup, "verify_droplet", lambda: (True, "healthy"))
    monkeypatch.setattr(setup.Setup, "prerequisites", lambda self, **_: True)
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_API_TOKEN=control\nDO_MICROVM_ACCESS_TOKEN=access\nDO_MODEL_ACCESS_KEY=model\n"
                        "DO_ACTION_GATEWAY_TOKEN=gw\nDO_ACTION_GATEWAY_ACTOR_ID=me\nDO_MICROVM_IMAGE=img\n"
                        "DO_INCIDENT_REPO=o/r\n")
    assert setup.main(["--dry-run", "--env-file", str(env_file)]) == 0
    out = capsys.readouterr().out
    assert "Ready: every read-only check passed" in out
    assert "[--  ] DO_ACTION_GATEWAY_ACTOR_ID" in out and "prepare.py, full run" in out
    monkeypatch.setattr(setup, "verify_droplet", lambda: (False, "the service does not answer"))
    assert setup.main(["--dry-run", "--env-file", str(env_file)]) == 1
    assert "recreate the droplet" in capsys.readouterr().out
