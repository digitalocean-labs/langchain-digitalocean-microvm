"""scripts/setup_examples.py: the plan per example, the token checks, the .env writes and the dry run, without network."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import setup_examples as setup  # noqa: E402


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


@pytest.mark.parametrize(("examples", "models"), [
    (("01",), ["anthropic-claude-haiku-4.5"]),
    (("01", "09"), ["anthropic-claude-haiku-4.5", "anthropic-claude-4.6-sonnet"]),
    (setup.BASE, ["anthropic-claude-haiku-4.5", "anthropic-claude-4.6-sonnet"]),  # 08, 10, 11 use the strong model
])
def test_model_key_checks_each_model_the_examples_use(api: dict[tuple[str, str], int], monkeypatch: pytest.MonkeyPatch,
                                                      examples: tuple[str, ...], models: list[str]) -> None:
    asked: list[str] = []
    monkeypatch.setattr(setup, "_model_probe", lambda key, model: asked.append(model))
    secret = next(s for s in setup.secrets_for(setup.Plan.of(examples), {}) if s.key == "DO_MODEL_ACCESS_KEY")
    assert secret.check("model", {}) is None
    assert asked == models


def test_plan_per_example() -> None:
    base = setup.Plan.of(setup.BASE)
    assert (base.model, base.image, base.gateway, base.incident) == (True, True, False, False)
    assert base.microvms == 7  # 08: six specialists and the template's builder
    keys = lambda plan: [s.key for s in setup.secrets_for(plan, {})]  # noqa: E731
    assert keys(base) == ["DO_MICROVM_API_TOKEN", "DO_MICROVM_ACCESS_TOKEN", "DO_MODEL_ACCESS_KEY"]
    assert "DO_ACTION_GATEWAY_TOKEN" in keys(setup.Plan.of(("09",)))
    assert "DO_MODEL_ACCESS_KEY" not in keys(setup.Plan.of(("02",)))  # dcode brings its own model
    assert keys(setup.Plan.of(("07",)))[-1] == "TAVILY_API_KEY"
    assert setup.Plan.of(("02",)).dcode and setup.Plan.of(("09", "07")).incident


@pytest.mark.parametrize("value", ["13", "4,13"])
def test_unknown_examples_are_refused(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        setup.main(["--dry-run", "--examples", value])
    assert "no example 13" in capsys.readouterr().err


def test_examples_parsed_once_each() -> None:
    import argparse
    assert setup.parse_examples("9, 7,09", argparse.ArgumentParser()) == ("09", "07")
    assert setup.parse_examples(None, argparse.ArgumentParser()) == setup.BASE


def test_save_writes_env_private(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# DO_MICROVM_IMAGE=\nOTHER=1\n")
    env_file.chmod(0o644)
    s = setup.Setup(env_file, setup.Plan.of(setup.BASE), dry_run=False)
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
    assert setup.main(["--dry-run", "--examples", "09", "--env-file", str(env_file)]) == 1
    out = capsys.readouterr().out
    assert env_file.read_text() == before
    assert "DO_MICROVM_ACCESS_TOKEN" in out and "microvm:access only" in out
    assert "DO_ACTION_GATEWAY_TOKEN" in out and "To do:" in out
    setup.main(["--dry-run", "--env-file", str(env_file)])  # the base: no gateway token, no 09 steps
    out = capsys.readouterr().out
    steps = [line for line in out.splitlines() if line.startswith("[")]  # not the shell note, which names variables
    assert not any("DO_ACTION_GATEWAY_TOKEN" in line for line in steps) and "Demo repository" not in out


@pytest.mark.parametrize("extra", ["/droplets", "/account"])
def test_broader_token_in_env_warned_not_failed(api: dict[tuple[str, str], int], tmp_path: Path,
                                               capsys: pytest.CaptureFixture[str], extra: str) -> None:
    api[("/microvms", "sekrit-control")] = 200
    api[(extra, "sekrit-control")] = 200  # Full Access, Read Only or an extra scope
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_API_TOKEN=sekrit-control\nDO_MODEL_ACCESS_KEY=model\n")
    s = setup.Setup(env_file, setup.Plan.of(setup.BASE), dry_run=True)
    s.tokens()
    out = capsys.readouterr().out
    assert "sekrit" not in out
    assert "[warn] DO_MICROVM_API_TOKEN: accepted, and it can also read droplets or the account" in out
    assert not any("control token" in item for item in s.todo)  # kept: a warning, not a missing token


@pytest.mark.parametrize(("keep", "saved"), [("y", True), ("", False)])
def test_pasted_broader_token_saved_only_when_confirmed(api: dict[tuple[str, str], int], tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch, keep: str, saved: bool) -> None:
    api[("/microvms", "full")] = 200
    api[("/droplets", "full")] = 200
    pasted = iter(["full", ""])  # the broader token, then an empty answer (skip)
    monkeypatch.setattr(setup.getpass, "getpass", lambda prompt: next(pasted))
    monkeypatch.setattr("builtins.input", lambda prompt: keep)
    monkeypatch.setattr(setup, "secrets_for", lambda plan, env: [setup.Secret(
        "DO_MICROVM_API_TOKEN", "control", "", "", setup.check_control)])  # the control token only
    env_file = tmp_path / ".env"
    s = setup.Setup(env_file, setup.Plan.of(setup.BASE), dry_run=False)
    assert s.tokens() is saved
    assert (s.env.get("DO_MICROVM_API_TOKEN") == "full") is saved


def test_missing_tool_stops_before_any_prompt(api: dict[tuple[str, str], int], tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda command: None if command == "gh" else f"/bin/{command}")
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(setup.Setup, "tokens", lambda self: pytest.fail("asked for tokens"))
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_IMAGE=img\n")
    assert setup.main(["--examples", "09", "--env-file", str(env_file)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] gh: not found" in out and "https://cli.github.com" in out
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
    monkeypatch.setattr(setup, "verify_connections", lambda token, actor: (True, "active DigitalOcean and GitHub"))
    monkeypatch.setattr(setup.Setup, "prerequisites", lambda self, **_: True)
    env_file = tmp_path / ".env"
    env_file.write_text("DO_MICROVM_API_TOKEN=control\nDO_MICROVM_ACCESS_TOKEN=access\nDO_MODEL_ACCESS_KEY=model\n"
                        "DO_ACTION_GATEWAY_TOKEN=gw\nDO_ACTION_GATEWAY_ACTOR_ID=me\nDO_MICROVM_IMAGE=img\n"
                        "DO_INCIDENT_REPO=o/r\n")
    assert setup.main(["--dry-run", "--examples", "09", "--env-file", str(env_file)]) == 0
    out = capsys.readouterr().out
    assert "Ready: every read-only check passed" in out
    assert "[ok  ] DO_ACTION_GATEWAY_ACTOR_ID me: active" in out and "prepare.py, full run" in out
    assert setup.main(["--dry-run", "--env-file", str(env_file)]) == 0  # the base is ready too
    capsys.readouterr()
    monkeypatch.setattr(setup, "verify_droplet", lambda: (False, "the service does not answer"))
    assert setup.main(["--dry-run", "--examples", "09", "--env-file", str(env_file)]) == 1
    assert "recreate the droplet" in capsys.readouterr().out


def test_plan_for_04() -> None:
    plan = setup.Plan.of(("04",))
    assert (plan.hosts, plan.agents, plan.incident, plan.gateway, plan.image) == (True, False, False, False, False)
    assert plan.hosts_label == "04" and setup.Plan.of(("04", "12")).hosts_label == "04, 12"


@pytest.mark.parametrize(("examples", "keys"), [
    (("04",), ["DO_MICROVM_AGENT_IMAGE"]),
    (("12",), ["DO_INCIDENT_HOST_IMAGE", "DO_INCIDENT_SPECIALISTS_IMAGE"]),
    (("04", "12"), ["DO_MICROVM_AGENT_IMAGE", "DO_INCIDENT_HOST_IMAGE", "DO_INCIDENT_SPECIALISTS_IMAGE"]),
])
def test_host_images_per_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, examples: tuple[str, ...],
                                 keys: list[str]) -> None:
    checked: list[str] = []
    monkeypatch.setattr(setup.Setup, "host_image", lambda self, key, what, spec, name: checked.append(key) or True)
    env_file = tmp_path / ".env"
    env_file.write_text("")
    assert setup.Setup(env_file, setup.Plan.of(examples), dry_run=True).host_images() is True
    assert checked == keys


def test_default_agent_host_is_built_without_a_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class Done:
        returncode, stdout, stderr = 0, "layer: 1 MiB, content hash 5ba6f91221c5\n", ""

    monkeypatch.setattr(setup.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **_: seen.append(command) or Done())
    assert setup.content_hash(None) == "5ba6f91221c5"
    assert "--spec" not in seen[0]
    assert setup.content_hash(setup.HOST_IMAGES[0][2]) == "5ba6f91221c5"
    assert "--spec" in seen[1]


def test_plan_for_12() -> None:
    plan = setup.Plan.of(("12",))
    assert (plan.incident, plan.hosts, plan.agents, plan.gateway, plan.image) == (True, True, True, True, False)
    assert plan.incident_label == "12" and setup.Plan.of(("09", "12")).incident_label == "09, 12"
    assert not setup.Plan.of(("09",)).agents


def _setup_12(tmp_path: Path, env: str, *, dry_run: bool = True) -> setup.Setup:
    env_file = tmp_path / ".env"
    env_file.write_text(env)
    return setup.Setup(env_file, setup.Plan.of(("12",)), dry_run=dry_run)


@pytest.mark.parametrize(("current", "manifest", "ok", "detail"), [
    ("abc123", (True, "found"), True, "built from these sources"),
    ("fff999", (True, "found"), False, "tag abc123, the sources hash to fff999"),
    (None, (True, "found"), True, "was not checked"),  # uv or crane missing: the manifest only
    ("abc123", (False, "cannot be read"), False, "cannot be read"),
])
def test_host_image_checked_against_its_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str], current: str | None,
                                                manifest: tuple[bool, str], ok: bool, detail: str) -> None:
    monkeypatch.setattr(setup, "content_hash", lambda spec: current)
    monkeypatch.setattr(setup, "verify_image", lambda image, token: manifest)
    s = _setup_12(tmp_path, "DO_INCIDENT_HOST_IMAGE=registry.digitalocean.com/reg/commander:abc123\n")
    key, what, spec, name = setup.HOST_IMAGES[0]
    assert s.host_image(key, what, spec, name) is ok
    assert detail in capsys.readouterr().out
    assert bool(s.todo) is not ok


def test_host_image_build_keeps_its_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = "layer: 1 MiB\n\npushed registry.digitalocean.com/reg/commander:fff999\n  digest ref: x\n"
        stderr = ""

    def run(command: list[str], **kwargs: object) -> Done:
        calls.append(command)
        assert "sekrit" not in " ".join(command)  # the registry token goes in the environment, not argv
        assert kwargs["env"]["DO_REGISTRY_TOKEN"] == "sekrit"  # type: ignore[index]
        return Done()

    monkeypatch.setattr(setup, "content_hash", lambda spec: "fff999")
    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    s = _setup_12(tmp_path, "DO_INCIDENT_HOST_IMAGE=registry.digitalocean.com/reg/commander:abc123\n"
                            "DO_REGISTRY_TOKEN=sekrit\n", dry_run=False)
    key, what, spec, name = setup.HOST_IMAGES[0]
    assert s.host_image(key, what, spec, name) is True
    assert calls[0][calls[0].index("--repository") + 1] == "registry.digitalocean.com/reg/commander"
    assert s.env[key] == "registry.digitalocean.com/reg/commander:fff999"


def test_host_mode_and_broker_in_a_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(setup, "verify_broker", lambda url: (True, "the tunnel answers (502)"))
    s = _setup_12(tmp_path, "DO_INCIDENT_BROKER_URL=http://127.0.0.1:8788\n")
    assert s.host_mode() is False and s.broker() is False
    assert any("DO_MICROVM_HOST_MODE=microvm" in item for item in s.todo)
    assert any("cloudflared tunnel --url http://127.0.0.1:8788" in item for item in s.todo)
    s = _setup_12(tmp_path, "DO_MICROVM_HOST_MODE=microvm\nDO_INCIDENT_BROKER_URL=https://a-b.trycloudflare.com \n")
    assert s.host_mode() is True and s.broker() is True
    out = capsys.readouterr().out
    assert "the tunnel answers (502)" in out and not s.unchecked


def _connections(pages: list[list[dict[str, str]]], monkeypatch: pytest.MonkeyPatch, status: int = 200) -> None:
    total = sum(len(p) for p in pages)

    def get(url: str, *, params: dict[str, int], **_: object) -> _Response:
        assert url.endswith("/action-gateway/connections")
        return _Response(status, {"connections": pages[params["page"] - 1], "pagination": {"total": total}})

    monkeypatch.setattr(httpx, "get", get)


def test_verify_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    do = {"user_id": "me", "provider": "digitalocean", "status": "active"}
    gh = {"user_id": "me", "provider": "github", "status": "active"}
    _connections([[do], [gh]], monkeypatch)  # two pages
    assert setup.verify_connections("gw", "me") == (True, "active DigitalOcean and GitHub connections")
    _connections([[do, {**gh, "status": "pending"}]], monkeypatch)
    ok, detail = setup.verify_connections("gw", "me")
    assert ok is False and "no active GitHub connection for me" in detail and "GitHub is pending" in detail
    _connections([[do, gh]], monkeypatch)
    assert setup.verify_connections("gw", "someone-else")[0] is False  # another actor's connections do not count
    _connections([[]], monkeypatch, status=403)
    assert setup.verify_connections("gw", "me")[0] is None


def test_missing_connection_fails_the_dry_run_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(setup, "check_gateway", lambda token, env: None)
    monkeypatch.setattr(setup, "verify_connections", lambda token, actor: (False, "no active GitHub connection for me"))
    env = "DO_ACTION_GATEWAY_TOKEN=gw\nDO_ACTION_GATEWAY_ACTOR_ID=me\n"
    dry = _setup_12(tmp_path, env)
    assert dry.actor() is False and any("connect link" in item for item in dry.todo)
    full = _setup_12(tmp_path, env, dry_run=False)
    assert full.actor() is True  # the next gateway call shows the connect link
    assert "shows the connect link" in capsys.readouterr().out


@pytest.mark.parametrize(("resolves", "answer", "ok", "detail"), [
    (False, 200, False, "does not resolve"),
    (True, 530, False, "not connected"),
    (True, 502, True, "the relay is not running"),
    (True, 404, True, "the tunnel and the relay answer"),
    (True, None, False, "no answer"),
])
def test_verify_broker(monkeypatch: pytest.MonkeyPatch, resolves: bool, answer: int | None, ok: bool, detail: str) -> None:
    def getaddrinfo(host: str, port: int) -> list[object]:
        if not resolves:
            raise setup.socket.gaierror("no such host")
        return [object()]

    def get(url: str, **_: object) -> _Response:
        if answer is None:
            raise httpx.ConnectTimeout("timed out")
        return _Response(answer)

    monkeypatch.setattr(setup.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(httpx, "get", get)
    got_ok, got_detail = setup.verify_broker("https://a-b.trycloudflare.com")
    assert got_ok is ok and detail in got_detail
