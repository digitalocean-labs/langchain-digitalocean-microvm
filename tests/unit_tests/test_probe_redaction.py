"""scripts/probe_api.py writes no tokens or identities to disk, and picks the right fixture responses.

The probe's echo image sends back the request headers it received: the
probe must keep only header names (values for its own headers), and redact
the token, JWTs and emails from everything it writes. No network here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import probe_api  # noqa: E402

TOKEN = "dop_v1_" + "ab12" * 16
JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl"

ECHO = {
    "path": "/hdr", "query": {"x": "1"}, "method": "GET", "ip": "203.0.113.7", "ips": ["203.0.113.7"],
    "os": {"hostname": "vm"}, "connection": {}, "cookies": {},
    "headers": {
        "authorization": f"Bearer {TOKEN}",
        "x-user-email": "someone@example.org",
        "x-org-uuid": "6f1c2a1e-0d7b-4c39-9a3e-1b2c3d4e5f60",
        "do-iam-authorization-data": JWT,
        "cf-connecting-ip": "203.0.113.7",
        "x-api-key": "probe-host-token",
        "x-auth-scheme": "langsmith",
    },
}


def test_echo_responses_keep_header_names_and_only_safe_values() -> None:
    kept = probe_api.scrub_echo(ECHO)
    assert set(kept) == {"method", "path", "query", "headers"}  # no ip / os / connection / cookies
    assert sorted(kept["headers"]) == sorted(ECHO["headers"])
    assert kept["headers"]["x-api-key"] == "probe-host-token" and kept["headers"]["x-auth-scheme"] == "langsmith"
    for name in ("authorization", "x-user-email", "x-org-uuid", "do-iam-authorization-data", "cf-connecting-ip"):
        assert kept["headers"][name] == "<redacted>"
    assert probe_api.scrub_echo("plain text") == "plain text"


def test_nothing_written_carries_the_token_a_jwt_or_an_email(tmp_path: Path) -> None:
    probe = probe_api.Probe(TOKEN, tmp_path)
    probe.event("endpoint", response=probe_api.scrub_echo(ECHO))
    probe.event("api", body={"leak": f"{TOKEN} {JWT} someone@example.org", "other": "dop_v1_" + "cd" * 32})
    probe.note("finding", {"nested": [JWT]})
    probe.log.close()
    written = (tmp_path / "events.jsonl").read_text()
    for secret in (TOKEN, JWT, "someone@example.org", "203.0.113.7", "6f1c2a1e", "dop_v1_cd"):
        assert secret not in written
    assert "<REDACTED_TOKEN>" in written and "<REDACTED_JWT>" in written and "<REDACTED_EMAIL>" in written


def test_the_access_token_is_what_the_endpoint_gets_and_is_redacted_too(tmp_path: Path) -> None:
    access = "dop_v1_" + "ef" * 32
    probe = probe_api.Probe(TOKEN, tmp_path, access_token=access)
    assert probe.access_token == access
    assert probe.redact({"a": f"Bearer {access}", "b": TOKEN}) == {"a": "Bearer <REDACTED_TOKEN>", "b": "<REDACTED_TOKEN>"}
    assert probe_api.Probe(TOKEN, tmp_path).access_token == TOKEN  # no access token: the API token, as before


def test_the_project_id_is_redacted_everywhere(tmp_path: Path) -> None:
    probe = probe_api.Probe(TOKEN, tmp_path)
    project = "6f1c2a1e-0d7b-4c39-9a3e-1b2c3d4e5f60"
    body = {"microvms": [{"id": "vm-1", "project_id": project}], "microvm": {"project_id": project}}
    assert project not in json.dumps(probe.redact(body))
    assert probe.redact(body)["microvms"][0] == {"id": "vm-1", "project_id": "<REDACTED_PROJECT_ID>"}


def test_recorded_fixtures_carry_no_project_id() -> None:
    fixtures = Path(__file__).parent / "fixtures" / "microvm_api"
    for path in fixtures.glob("*.json"):
        assert probe_api._redact_keys(json.loads(path.read_text())) == json.loads(path.read_text()), path.name


def test_fixtures_are_the_first_matching_response_of_the_run(tmp_path: Path) -> None:
    probe = probe_api.Probe(TOKEN, tmp_path)

    def vm(state: str, tags: list[str]) -> dict:
        return {"microvm": {"id": "vm-1", "name": "probe-x-a", "state": state, "tags": tags}}

    probe.responses += [
        ("GET", "options", 200, None, {"account_limits": {"max_total_count": 2}}),
        ("POST", "microvms", 201, {"source": {"oci_ref": "img"}}, vm("creating", ["microvm-probe"])),
        ("GET", "microvm", 200, None, vm("creating", ["microvm-probe"])),
        ("GET", "microvm", 200, None, vm("running", ["microvm-probe"])),
        ("GET", "microvm", 200, None, vm("running", ["later"])),
        ("POST", "pause", 200, None, vm("paused", [])),
        ("POST", "exec", 200, {"argv": ["node", "-v"]}, {"exit_code": 0, "stdout": "v22\n", "stderr": ""}),
        ("POST", "exec", 200, {"argv": ["sh", "-c", "echo $X; exit 3"]}, {"exit_code": 3, "stdout": f"{TOKEN}\n"}),
    ]
    report = probe.write_fixtures(tmp_path / "fx")
    assert report["get_running_200"] == "written" and report["checkpoint_create_201"] == "missing"
    assert json.loads((tmp_path / "fx" / "get_running_200.json").read_text())["microvm"]["tags"] == ["microvm-probe"]
    assert json.loads((tmp_path / "fx" / "exec_exit_3_200.json").read_text())["exit_code"] == 3
    assert TOKEN not in (tmp_path / "fx" / "exec_exit_3_200.json").read_text()
    assert set(report) == set(probe_api.FIXTURES)
    # every fixture the client tests replay is one the probe knows how to record
    assert {p.stem for p in (Path(__file__).parent / "fixtures" / "microvm_api").glob("*.json")} == set(probe_api.FIXTURES)


def test_path_kinds() -> None:
    kinds = {
        "/microvms": "microvms", "/microvms/options": "options", "/microvms/vm-1": "microvm",
        "/microvms/vm-1/exec": "exec", "/microvms/vm-1/pause": "pause", "/microvms/vm-1/checkpoints": "checkpoints",
        "/microvms/checkpoints/cp-1": "checkpoint", "/microvms/checkpoints": "checkpoints-list",
    }
    assert {path: probe_api._path_kind(path) for path in kinds} == kinds
