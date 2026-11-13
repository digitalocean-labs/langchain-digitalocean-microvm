"""Probe the DigitalOcean MicroVMs API and record what it really does.

Answers the questions the design depends on that the documentation leaves open
(see docs/design.md):

  1. account options (sizes, limits) and whether tags may contain ':'
  2. create latency and the raw MicroVM object (fixtures for client.py)
  3. which request headers reach the container through the HTTPS endpoint
     (Authorization? x-api-key? x-auth-scheme?) and what it does without a token
  4. exec: stdout/stderr/exit code shape
  5. what "idle" means: does a busy container (CPU + outbound HTTPS, no inbound
     traffic) still get auto-paused? Are its processes frozen, then resumed?
  6. does inbound polling keep a VM awake; auto-resume latency
  7. explicit pause/resume latency and idempotency
  8. a slow request through the endpoint (edge timeout)
  9. checkpoint capture time, create-from-checkpoint, and whether a process
     running at capture time is still running in the clone
 10. list filters (name, tag_name) and delete semantics

Every MicroVM and checkpoint it creates is deleted in a `finally` block. The
API token is read from .env (DO_MICROVM_API_TOKEN); requests to the
endpoint carry DO_MICROVM_ACCESS_TOKEN when set (else the API token). Nothing written to
disk carries a token or anyone's identity: the echo image sends back the
request headers it received, so endpoint responses are recorded as path, query, method and header
NAMES, with values only for the headers this probe sets itself
(`SAFE_HEADER_VALUES`); the token, JWT-shaped strings and email addresses are
then redacted from every record as a backstop. Output: one JSON-lines event
log + summary.json; with --fixtures, the API responses tests/test_client.py
replays (`FIXTURES`), written from this run.

    uv run python \
        scripts/probe_api.py --out <dir> \
        [--fixtures tests/unit_tests/fixtures/microvm_api]
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from _env import add_env_file_argument, load_env  # scripts/_env.py

API = "https://api.digitalocean.com/v2"
IMAGE = "docker.io/mendhak/http-https-echo:latest"  # echoes request headers as JSON; node:alpine
IMAGE_PORT = 8080
REGION = "mkc1"
SIZE = {"cpu": 1, "memory": 2048}
IDLE = "30s"
PROBE_TAG = "microvm-probe"

# Header values worth recording: the ones this probe sends itself (pass-through
# checks) and generic HTTP ones. Every other forwarded header keeps its name only.
SAFE_HEADER_VALUES = frozenset({
    "x-api-key", "x-auth-scheme", "x-probe-custom", "x-set-response-delay-ms",
    "accept", "accept-encoding", "content-length", "user-agent", "x-forwarded-proto",
})
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]+)?")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_DO_TOKEN = re.compile(r"do[por]_v1_[0-9a-f]{16,}")
_REDACTED_KEYS = {"project_id": "<REDACTED_PROJECT_ID>"}  # identifies the team's project, not the probe's VMs


def _redact_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _REDACTED_KEYS[k] if k in _REDACTED_KEYS and obj[k] else _redact_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_keys(v) for v in obj]
    return obj


def scrub_echo(body: Any) -> Any:
    """What is kept of an echo response: request line and header names (values only if safe)."""
    if not isinstance(body, dict):
        return body
    headers = body.get("headers") if isinstance(body.get("headers"), dict) else {}
    return {
        "method": body.get("method"),
        "path": body.get("path"),
        "query": body.get("query"),
        "headers": {k: (v if k.lower() in SAFE_HEADER_VALUES else "<redacted>") for k, v in sorted(headers.items())},
    }


# fixture name -> how to pick its response out of the run (method, path kind, status, predicate on request/body)
def _vm_state(state: str):
    return lambda req, body: (body.get("microvm") or {}).get("state") == state


FIXTURES: dict[str, tuple[str, str, int, Any]] = {
    "options_200": ("GET", "options", 200, None),
    "create_201": ("POST", "microvms", 201, lambda req, body: "oci_ref" in ((req or {}).get("source") or {})),
    "create_from_checkpoint_201": ("POST", "microvms", 201,
                                   lambda req, body: "checkpoint_id" in ((req or {}).get("source") or {})),
    "get_creating_200": ("GET", "microvm", 200, _vm_state("creating")),
    "get_running_200": ("GET", "microvm", 200, _vm_state("running")),
    "get_paused_200": ("GET", "microvm", 200, _vm_state("paused")),
    "get_terminating_200": ("GET", "microvm", 200, _vm_state("terminating")),
    "pause_200": ("POST", "pause", 200, None),
    "resume_200": ("POST", "resume", 200, None),
    "list_200": ("GET", "microvms", 200, lambda req, body: len(body.get("microvms") or []) == 2),
    "exec_200": ("POST", "exec", 200, lambda req, body: (req or {}).get("argv") == ["node", "-v"]),
    "exec_exit_3_200": ("POST", "exec", 200, lambda req, body: "exit 3" in " ".join((req or {}).get("argv") or [])),
    "checkpoint_create_201": ("POST", "checkpoints", 201, None),
    "checkpoint_available_200": ("GET", "checkpoint", 200,
                                 lambda req, body: (body.get("checkpoint") or {}).get("status") == "CHECKPOINT_AVAILABLE"),
}


def _path_kind(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    if parts == ["microvms"]:
        return "microvms"
    if parts[1:2] == ["options"]:
        return "options"
    if parts[1:2] == ["checkpoints"]:
        return "checkpoint" if len(parts) > 2 else "checkpoints-list"
    return parts[2] if len(parts) > 2 else "microvm"

# Background workload started via exec: every 2 s, append a timestamp and the
# status of an outbound HTTPS GET to /tmp/probe.log. CPU + egress, no ingress.
WORKLOAD = (
    "const https=require('https'),fs=require('fs');"
    "setInterval(()=>{const t=Date.now();"
    "https.get('https://api.digitalocean.com/v2/',r=>{fs.appendFileSync('/tmp/probe.log',t+' '+r.statusCode+'\\n');r.resume();})"
    ".on('error',e=>fs.appendFileSync('/tmp/probe.log',t+' ERR '+e.code+'\\n'));},2000);"
)


class Probe:
    def __init__(self, token: str, out: Path, *, access_token: str | None = None) -> None:
        self.token = token
        # Endpoint requests carry the access-only token when there is one: the
        # (third-party) echo container sees the request headers.
        self.access_token = access_token or token
        self.out = out
        self.log = (out / "events.jsonl").open("a")
        self.summary: dict[str, Any] = {}
        self.vms: list[str] = []
        self.responses: list[tuple[str, str, int, Any, Any]] = []  # method, path kind, status, request, body
        self.checkpoints: list[str] = []
        self.http = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=httpx.Timeout(120.0))
        self.t0 = time.monotonic()

    # ------------------------------------------------------------ plumbing

    def redact(self, obj: Any) -> Any:
        text = json.dumps(obj, default=str).replace(self.token, "<REDACTED_TOKEN>")
        text = text.replace(self.access_token, "<REDACTED_TOKEN>")
        text = _DO_TOKEN.sub("<REDACTED_TOKEN>", _JWT.sub("<REDACTED_JWT>", text))
        return _redact_keys(json.loads(_EMAIL.sub("<REDACTED_EMAIL>", text)))

    def event(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.monotonic() - self.t0, 2), "kind": kind, **data}
        self.log.write(json.dumps(self.redact(record)) + "\n")
        self.log.flush()
        brief = {k: v for k, v in record.items() if k not in ("body", "response")}
        print(json.dumps(self.redact(brief))[:300], flush=True)

    def note(self, key: str, value: Any) -> None:
        self.summary[key] = value
        self.event("finding", key=key, value=value)

    def api(self, method: str, path: str, *, json_body: Any = None, params: Any = None, expect_ok: bool = True) -> httpx.Response:
        for attempt in range(6):
            started = time.monotonic()
            resp = self.http.request(method, f"{API}{path}", json=json_body, params=params)
            elapsed = round(time.monotonic() - started, 3)
            try:
                body = resp.json()
            except ValueError:
                body = resp.text[:2000]
            self.event("api", method=method, path=path, status=resp.status_code, elapsed=elapsed,
                       request=json_body, body=body)
            if isinstance(body, dict):
                self.responses.append((method, _path_kind(path), resp.status_code, json_body, body))
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if expect_ok and resp.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> {resp.status_code}: {str(body)[:500]}")
            return resp
        raise RuntimeError(f"{method} {path}: still rate limited")

    def endpoint(self, host: str, path: str = "/", *, headers: dict[str, str] | None = None, auth: bool = True,
                 timeout: float = 120.0) -> tuple[int | None, Any, float]:
        hdrs = dict(headers or {})
        if auth:
            hdrs["Authorization"] = f"Bearer {self.access_token}"
        started = time.monotonic()
        try:
            resp = httpx.get(f"https://{host}{path}", headers=hdrs, timeout=timeout)
            status, body = resp.status_code, (resp.json() if "json" in resp.headers.get("content-type", "") else resp.text[:1000])
        except httpx.HTTPError as exc:
            status, body = None, f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - started, 3)
        self.event("endpoint", host=host, path=path, auth=auth, sent_headers=sorted(hdrs), status=status,
                   elapsed=elapsed, response=scrub_echo(body))
        return status, body, elapsed  # unscrubbed for the checks in run(); only findings are recorded

    def write_fixtures(self, directory: Path) -> dict[str, str]:
        """Write each FIXTURES response this run saw (first match) to `directory`; report what was missing."""
        directory.mkdir(parents=True, exist_ok=True)
        report: dict[str, str] = {}
        for name, (method, kind, status, matches) in FIXTURES.items():
            found = next((body for m, k, st, req, body in self.responses
                          if (m, k, st) == (method, kind, status) and (matches is None or matches(req, body))), None)
            if found is None:
                report[name] = "missing"
                continue
            (directory / f"{name}.json").write_text(json.dumps(self.redact(found), indent=2, sort_keys=True) + "\n")
            report[name] = "written"
        return report

    def exec(self, vm: str, argv: list[str], cwd: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"argv": argv}
        if cwd:
            body["cwd"] = cwd
        resp = self.api("POST", f"/microvms/{vm}/exec", json_body=body, expect_ok=False)
        return {"status": resp.status_code, **(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text})}

    def get(self, vm: str) -> dict[str, Any]:
        return self.api("GET", f"/microvms/{vm}").json()["microvm"]

    def wait_state(self, vm: str, want: set[str], *, timeout: float, interval: float = 3.0) -> tuple[str, float, list]:
        started = time.monotonic()
        seen: list[tuple[float, str]] = []
        while True:
            obj = self.get(vm)
            state = obj.get("state", "?")
            if not seen or seen[-1][1] != state:
                seen.append((round(time.monotonic() - started, 1), state))
            if state in want:
                return state, round(time.monotonic() - started, 1), seen
            if state == "failed":
                raise RuntimeError(f"MicroVM {vm} failed: {obj.get('failure_reason')}")
            if time.monotonic() - started > timeout:
                return state, round(time.monotonic() - started, 1), seen
            time.sleep(interval)

    def wait_endpoint(self, vm: str, *, timeout: float = 300.0) -> tuple[str, float]:
        started = time.monotonic()
        while True:
            obj = self.get(vm)
            urls = obj.get("urls") or []
            default = next((u for u in urls if u.get("default")), urls[0] if urls else None)
            if default and default.get("status") == "ACTIVE" and default.get("hostname"):
                return default["hostname"], round(time.monotonic() - started, 1)
            if obj.get("state") == "failed":
                raise RuntimeError(f"MicroVM {vm} failed: {obj.get('failure_reason')}")
            if time.monotonic() - started > timeout:
                raise RuntimeError(f"endpoint of {vm} not ACTIVE after {timeout}s: {urls}")
            time.sleep(3)

    def create(self, name: str, *, tags: list[str], extra: dict[str, Any] | None = None, source: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "tags": tags, **(extra or {})}
        if source is None:
            body.update({"region": REGION, "size": SIZE, "source": {"oci_ref": IMAGE}, "http_port": IMAGE_PORT,
                         "http_protocol": "http", "auto_pause": {"idle_timeout": IDLE}})
        else:
            body["source"] = source
            body.setdefault("auto_pause", {"idle_timeout": IDLE})
        resp = self.api("POST", "/microvms", json_body=body, expect_ok=False)
        if resp.status_code >= 400:
            return {"error": resp.status_code, "body": resp.json() if resp.content else None}
        vm = resp.json()["microvm"]
        self.vms.append(vm["id"])
        return vm

    # ------------------------------------------------------------ steps

    def run(self) -> None:
        run_id = secrets.token_hex(3)

        # 1. options + tag charset ------------------------------------------
        options = self.api("GET", "/microvms/options").json()
        self.note("options", options)

        name = f"probe-{run_id}-a"
        colon_tags = [PROBE_TAG, f"{PROBE_TAG}-run:{run_id}", "microvm-agenthost-graph:echo"]
        started = time.monotonic()
        vm = self.create(name, tags=colon_tags, extra={"environment": {"PROBE_ENV": "hello", "HTTP_PORT": str(IMAGE_PORT)}})
        if "error" in vm:
            self.note("tags_with_colon_accepted", False)
            self.note("create_error_with_colon_tags", vm)
            vm = self.create(name, tags=[PROBE_TAG, f"{PROBE_TAG}-run-{run_id}"],
                             extra={"environment": {"PROBE_ENV": "hello", "HTTP_PORT": str(IMAGE_PORT)}})
            if "error" in vm:
                raise RuntimeError(f"create failed even without colon tags: {vm}")
        else:
            self.note("tags_with_colon_accepted", True)
        a = vm["id"]
        self.note("create_response_state", vm.get("state"))
        self.note("create_response_keys", sorted(vm))

        # 2. readiness ---------------------------------------------------------
        state, secs, seen = self.wait_state(a, {"running"}, timeout=300)
        self.note("create_to_running_s", {"seconds": round(time.monotonic() - started, 1), "states": seen})
        host, secs = self.wait_endpoint(a)
        self.note("running_to_endpoint_active_s", secs)
        self.note("endpoint_hostname", host)
        self.note("microvm_object_running", self.get(a))

        # 3. headers through the endpoint -------------------------------------
        status, body, el = self.endpoint(host, "/hdr?x=1", headers={
            "x-api-key": "probe-host-token", "x-auth-scheme": "langsmith", "x-probe-custom": "yes"})
        forwarded = sorted((body.get("headers") or {}).keys()) if isinstance(body, dict) else body
        self.note("endpoint_first_request", {"status": status, "seconds": el})
        self.note("headers_seen_by_container", forwarded)
        if isinstance(body, dict):
            hdrs = body.get("headers") or {}
            self.note("authorization_forwarded_to_container", "authorization" in hdrs)
            self.note("token_value_visible_in_container", self.access_token in json.dumps(hdrs))
            self.note("path_and_query_seen", {"path": body.get("path"), "query": body.get("query")})
        status, body, el = self.endpoint(host, "/", auth=False)
        self.note("endpoint_without_token", {"status": status, "body": body})
        status, body, el = self.endpoint(host, "/", auth=False, headers={"Authorization": "Bearer not-a-real-token"})
        self.note("endpoint_with_bad_token", {"status": status, "body": body})

        # 4. exec ---------------------------------------------------------------
        self.note("exec_simple", self.exec(a, ["node", "-v"]))
        self.note("exec_shell_exit_code", self.exec(a, ["sh", "-c", "echo out; echo err >&2; echo $PROBE_ENV; exit 3"]))
        self.note("exec_cwd", self.exec(a, ["pwd"], cwd="/tmp"))
        self.note("exec_whoami_and_os", self.exec(a, ["sh", "-c", "id; uname -a; cat /etc/os-release | head -2; nproc; free -m 2>/dev/null | head -2"]))
        self.note("exec_env_has_token", self.exec(a, ["sh", "-c", "env | grep -c dop_v1_ || true"]))
        started_bg = self.exec(a, ["sh", "-c", f"nohup node -e \"{WORKLOAD}\" > /tmp/workload.out 2>&1 & echo started"])
        self.note("exec_background_start", started_bg)

        # 5. idle definition: workload busy, NO inbound traffic -----------------
        time.sleep(8)
        before = self.exec(a, ["sh", "-c", "wc -l < /tmp/probe.log; tail -2 /tmp/probe.log"])
        self.note("workload_log_before_idle", before)
        idle_started = time.monotonic()
        state, secs, seen = self.wait_state(a, {"paused"}, timeout=150, interval=5)
        self.note("busy_but_no_ingress_auto_paused", {"final_state": state, "seconds": secs, "states": seen})

        # 6. auto-resume on request, then frozen-process evidence --------------
        if state == "paused":
            status, body, el = self.endpoint(host, "/after-pause")
            self.note("auto_resume_first_request", {"status": status, "seconds": el})
            log = self.exec(a, ["sh", "-c", "cat /tmp/probe.log"])
            stamps = [int(line.split()[0]) for line in (log.get("stdout") or "").splitlines() if line.split()]
            gaps = [round((b - a2) / 1000, 1) for a2, b in zip(stamps, stamps[1:]) if b - a2 > 5000]
            self.note("workload_gaps_over_5s_after_resume", gaps)
            self.note("workload_still_running_after_resume", self.exec(a, ["sh", "-c", "pgrep -f 'node -e' >/dev/null && echo yes || echo no"]))
        self.summary["idle_wait_total_s"] = round(time.monotonic() - idle_started, 1)

        # 6b. does inbound polling every 10 s keep it awake for 75 s? ----------
        awake: list[tuple[float, str]] = []
        poll_started = time.monotonic()
        while time.monotonic() - poll_started < 75:
            self.endpoint(host, "/keepalive")
            awake.append((round(time.monotonic() - poll_started, 1), self.get(a).get("state")))
            time.sleep(10)
        self.note("polling_every_10s_states", awake)

        # 7. explicit pause/resume ----------------------------------------------
        for op in ("pause", "pause", "resume", "resume"):
            started = time.monotonic()
            resp = self.api("POST", f"/microvms/{a}/{op}", expect_ok=False)
            self.note(f"explicit_{op}", {"status": resp.status_code, "seconds": round(time.monotonic() - started, 2),
                                         "state_after": self.get(a).get("state")})

        # 8. slow request through the endpoint ----------------------------------
        status, body, el = self.endpoint(host, "/slow", headers={"x-set-response-delay-ms": "35000"}, timeout=90)
        self.note("slow_request_35s", {"status": status, "seconds": el})

        # 9. checkpoint + clone ---------------------------------------------------
        limit_ok = True
        try:
            limit = options.get("limits") or options.get("options", {}).get("limits")
            self.note("limits_raw", limit)
        except AttributeError:
            pass
        resp = self.api("POST", f"/microvms/{a}/checkpoints", json_body={"name": f"probe-{run_id}-cp"}, expect_ok=False)
        if resp.status_code < 400:
            cp = resp.json().get("checkpoint", {})
            cp_id = cp.get("id")
            if cp_id:
                self.checkpoints.append(cp_id)
            started = time.monotonic()
            status = cp.get("status")
            while status not in ("CHECKPOINT_AVAILABLE", "CHECKPOINT_FAILED") and time.monotonic() - started < 600:
                time.sleep(5)
                status = self.api("GET", f"/microvms/checkpoints/{cp_id}").json()["checkpoint"]["status"]
            self.note("checkpoint_capture", {"status": status, "seconds": round(time.monotonic() - started, 1)})
            if status == "CHECKPOINT_AVAILABLE" and limit_ok:
                started = time.monotonic()
                clone = self.create(f"probe-{run_id}-b", tags=[PROBE_TAG], source={"checkpoint_id": cp_id})
                if "error" in clone:
                    self.note("create_from_checkpoint_error", clone)
                else:
                    b = clone["id"]
                    self.note("create_from_checkpoint_response_state", clone.get("state"))
                    state, secs, seen = self.wait_state(b, {"running"}, timeout=300)
                    self.note("checkpoint_clone_to_running_s", {"seconds": round(time.monotonic() - started, 1), "states": seen})
                    bhost, secs = self.wait_endpoint(b)
                    self.note("checkpoint_clone_endpoint_active_s", secs)
                    self.note("checkpoint_clone_same_hostname", bhost == host)
                    self.note("checkpoint_clone_process_running",
                              self.exec(b, ["sh", "-c", "pgrep -f 'node -e' >/dev/null && echo yes || echo no; tail -1 /tmp/probe.log; echo $PROBE_ENV"]))
                    status_b, _, el_b = self.endpoint(bhost, "/clone")
                    self.note("checkpoint_clone_first_request", {"status": status_b, "seconds": el_b})
                    self.note("checkpoint_object", self.api("GET", f"/microvms/checkpoints/{cp_id}").json())
            # deleting a checkpoint while capturing is a 400; by now it is final
        else:
            self.note("checkpoint_create_error", {"status": resp.status_code, "body": resp.text[:500]})

        # 10. list filters --------------------------------------------------------
        self.note("list_by_name", [m["id"] for m in self.api("GET", "/microvms", params={"name": name}).json().get("microvms", [])])
        self.note("list_by_tag", [m["id"] for m in self.api("GET", "/microvms", params={"tag_name": PROBE_TAG}).json().get("microvms", [])])
        self.note("list_response_keys", sorted(self.api("GET", "/microvms", params={"per_page": 1}).json()))

    def cleanup(self) -> None:
        for vm in self.vms:
            resp = self.api("DELETE", f"/microvms/{vm}", expect_ok=False)
            self.summary.setdefault("delete_status", {})[vm] = resp.status_code
            after = self.api("GET", f"/microvms/{vm}", expect_ok=False)
            self.summary.setdefault("get_after_delete", {})[vm] = {
                "status": after.status_code, "state": (after.json().get("microvm") or {}).get("state") if after.status_code == 200 else None}
        for cp in self.checkpoints:
            for _ in range(24):
                resp = self.api("DELETE", f"/microvms/checkpoints/{cp}", expect_ok=False)
                if resp.status_code != 400:
                    break
                time.sleep(5)
            self.summary.setdefault("checkpoint_delete_status", {})[cp] = resp.status_code
        # Belt and braces: anything still tagged as a probe (e.g. from a crashed earlier run).
        leftovers = self.api("GET", "/microvms", params={"tag_name": PROBE_TAG}, expect_ok=False)
        if leftovers.status_code == 200:
            for m in leftovers.json().get("microvms", []):
                if m.get("state") not in ("terminating", "terminated"):
                    self.api("DELETE", f"/microvms/{m['id']}", expect_ok=False)
                    self.summary.setdefault("leftover_deleted", []).append(m["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, help="also write the recorded API responses tests replay here")
    add_env_file_argument(parser)
    args = parser.parse_args()
    env = load_env(args.env_file)
    token = env.get("DO_MICROVM_API_TOKEN")
    if not token:
        print("DO_MICROVM_API_TOKEN not set", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    probe = Probe(token, args.out, access_token=env.get("DO_MICROVM_ACCESS_TOKEN"))
    code = 0
    try:
        probe.run()
    except Exception as exc:  # noqa: BLE001 - record and still clean up
        probe.event("error", error=f"{type(exc).__name__}: {exc}")
        probe.summary["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        try:
            probe.cleanup()
        except Exception as exc:  # noqa: BLE001
            probe.summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            code = 1
        if args.fixtures is not None and code == 0:
            probe.summary["fixtures"] = probe.write_fixtures(args.fixtures)
        (args.out / "summary.json").write_text(json.dumps(probe.redact(probe.summary), indent=2, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
