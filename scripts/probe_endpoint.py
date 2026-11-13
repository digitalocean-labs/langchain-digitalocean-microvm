"""Probe a MicroVM's HTTPS endpoint as the sandbox data plane and record what it does.

Answers ten questions about the endpoint (E1-E10)
with its own minimal in-VM server (`SERVER` below: healthz, exec with a real
timeout, raw file PUT/GET with offset and Range, plus probe-only /echo, /slow
and /stream routes). That server is the probe's, written before the package's
sandbox_server.py and deliberately kept separate: the probe measures the edge,
not the package (its error bodies say "sandbox-server", which is why
test_endpoint_edge_fixtures.py uses only the headers of its recorded answers). The image is
python:3.12-slim with only its configuration changed (entrypoint = SERVER),
built with crane like scripts/build_sandbox_image.py; E9 uses a second image
from the same base whose entrypoint is `sleep infinity`.

  E1  hostname format, create -> running -> URL ACTIVE -> first 200; server sanity
  E2  which request headers reach the workload (names only)
  E3  edge request timeout (/slow?s=30,55,65,90,130 in parallel), handler kept running?
  E3b (--edge-timeouts: E1, then only this) sequential GET/POST of 5-30 s, headers first,
      heartbeat, stream, slow upload; every handler call logged with the edge's request-id
  E4  upload/download limits and throughput (1-256 MiB, chunked encoding, offsets, Range)
  E5  streaming: does a chunked response arrive incrementally? (n=60, n=90)
  E6  latency and rate: sequential, then concurrent 10/25/50, vs 20 /exec calls
  E7  HTTP/2 (needs h2: `uv run --with h2 ...`)
  E8  explicit pause -> request, checkpoint clone + re-key, idle kept off by traffic,
      auto-pause during an in-flight request, auto-resume latency
  E9  bootstrap the server over /exec on an image without it
  E10 auth failure modes (no edge token, no/wrong x-api-key, deleted MicroVM)

At most two MicroVMs exist at a time (A, its clone B; then C), mkc1, 1vcpu-2gb;
one checkpoint. The run refuses to start unless the team has no MicroVMs or
checkpoints of ours (tag `microvm-probe`, names `probe-`), checks again at the
end, and deletes everything it created in `finally`.

Tokens: DO_MICROVM_API_TOKEN (control: create, exec, pause, delete) and
DO_MICROVM_ACCESS_TOKEN (required; the only token sent to the endpoint). Each
MicroVM gets its own random sandbox token. Nothing written to disk carries a
token, a header value other than the probe's own, or anyone's identity: /echo
returns header names and value hashes, compared in memory only; every record
is redacted for the tokens, JWT-shaped strings and email addresses as a
backstop. Output: events.jsonl + summary.json in --out; with --fixtures,
scrubbed endpoint responses for the transport tests.

    uv run --with h2 python scripts/probe_endpoint.py --out <dir> \\
        [--fixtures tests/unit_tests/fixtures/microvm_endpoint] [--skip E4,E9]
    uv run python scripts/probe_endpoint.py --edge-timeouts --out <dir>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from _env import add_env_file_argument, load_env  # scripts/_env.py
from build_sandbox_image import (  # scripts/build_sandbox_image.py
    BASE_IMAGE,
    PLATFORM,
    _csv_field,
    run,
)

API = "https://api.digitalocean.com/v2"
REGION = "mkc1"
SIZE = {"cpu": 1, "memory": 2048}
PORT = 8080
IDLE_A = "10m"  # long: E3-E7 must not be confounded by auto-pause
IDLE_SHORT = "30s"  # B (clone) and C: E8 idle checks
PROBE_TAG = "microvm-probe"
TOKEN_FILE = "/var/lib/microvm-sandbox/token"
MIB = 1 << 20

# The minimal sandbox server (§2.1 of the plan), stdlib only. The token file
# wins over $SANDBOX_TOKEN and is read on every request, so a clone is re-keyed
# by writing the file. Never logs requests.
SERVER = r'''
import hashlib, hmac, json, os, signal, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PORT = int(os.environ.get("PORT", "8080"))
TOKEN_FILE = os.environ.get("SANDBOX_TOKEN_FILE", "/var/lib/microvm-sandbox/token")
MAX_OUTPUT = int(os.environ.get("SANDBOX_MAX_OUTPUT", str(1 << 20)))
CHUNK = 1 << 20
ECHO_VALUES = ("x-probe-custom",)


def expected_token():
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return os.environ.get("SANDBOX_TOKEN", "")


def mark(name, text=""):
    with open("/tmp/" + name, "w") as f:
        f.write(text)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_json(self, status, body, close=False):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def body_chunks(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return
                left = size
                while left:
                    buf = self.rfile.read(min(left, CHUNK))
                    if not buf:
                        raise ConnectionError("short body")
                    left -= len(buf)
                    yield buf
                self.rfile.readline()
        left = int(self.headers.get("Content-Length") or 0)
        while left:
            buf = self.rfile.read(min(left, CHUNK))
            if not buf:
                raise ConnectionError("short body")
            left -= len(buf)
            yield buf

    def handle_any(self, method):
        url = urlsplit(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if method == "GET" and url.path == "/healthz":
            data = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        want = expected_token()
        got = self.headers.get("x-api-key", "")
        if not want or not hmac.compare_digest(got.encode(), want.encode()):
            return self.send_json(401, {"error": "unauthorized", "source": "sandbox-server"}, close=True)
        route = ROUTES.get((method, url.path))
        if url.path in ("/slow", "/stream", "/v1/exec"):  # one line per handler call: retries show up twice
            with open("/tmp/calls.log", "a") as f:
                f.write("%s %s %s %s %.3f %.3f\n" % (url.path, url.query or "-", self.headers.get("request-id", "-"),
                                                     self.headers.get("x-request-id", "-"), time.monotonic(), time.time()))
        if route is None:
            return self.send_json(404, {"error": "no such route", "source": "sandbox-server"}, close=True)
        try:
            route(self, q)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            self.send_json(500, {"error": type(exc).__name__, "source": "sandbox-server"}, close=True)

    def do_GET(self):
        self.handle_any("GET")

    def do_POST(self):
        self.handle_any("POST")

    def do_PUT(self):
        self.handle_any("PUT")


def r_exec(h, q):
    req = json.loads(b"".join(h.body_chunks()) or b"{}")
    timeout = float(req.get("timeout") or 30)
    started = time.monotonic()
    proc = subprocess.Popen(["sh", "-c", req["command"]], cwd=req.get("cwd") or "/", stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
    h.send_json(200, {"exit_code": proc.returncode, "timed_out": timed_out,
                      "stdout": out[:MAX_OUTPUT].decode("utf-8", "replace"),
                      "stderr": err[:MAX_OUTPUT].decode("utf-8", "replace"),
                      "truncated": len(out) > MAX_OUTPUT or len(err) > MAX_OUTPUT,
                      "seconds": round(time.monotonic() - started, 3)})


def r_put_file(h, q):
    path = q.get("path", "")
    if not path.startswith("/"):
        return h.send_json(400, {"error": "path must be absolute", "source": "sandbox-server"}, close=True)
    os.makedirs(os.path.dirname(path) or "/", exist_ok=True)
    offset = int(q.get("offset") or 0)
    digest, size, started = hashlib.sha256(), 0, time.monotonic()
    mode = "r+b" if offset and os.path.exists(path) else "wb"
    with open(path, mode) as f:
        f.seek(offset)
        for buf in h.body_chunks():
            f.write(buf)
            digest.update(buf)
            size += len(buf)
    want = h.headers.get("X-Content-Sha256")
    if want and want != digest.hexdigest():
        return h.send_json(422, {"error": "sha256 mismatch", "source": "sandbox-server"})
    h.send_json(200, {"bytes": size, "sha256": digest.hexdigest(), "seconds": round(time.monotonic() - started, 3)})


def r_get_file(h, q):
    path = q.get("path", "")
    try:
        total = os.stat(path).st_size
    except OSError:
        return h.send_json(404, {"error": "not found", "source": "sandbox-server"})
    start, end, status = 0, total - 1, 200
    rng = h.headers.get("Range", "")
    if rng.startswith("bytes=") and "," not in rng:
        a, _, b = rng[6:].partition("-")
        start = int(a) if a else max(total - int(b), 0)
        end = min(int(b), total - 1) if a and b else total - 1
        status = 206
    h.send_response(status)
    h.send_header("Content-Type", "application/octet-stream")
    h.send_header("Content-Length", str(max(end - start + 1, 0)))
    h.send_header("Accept-Ranges", "bytes")
    if status == 206:
        h.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, total))
    h.end_headers()
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            buf = f.read(min(left, CHUNK))
            if not buf:
                break
            h.wfile.write(buf)
            left -= len(buf)


def r_echo(h, q):
    names = [k.lower() for k in h.headers.keys()]
    h.send_json(200, {
        "header_names": sorted(names),
        "values": {k: h.headers.get(k) for k in ECHO_VALUES if h.headers.get(k) is not None},
        "value_sha256": sorted(hashlib.sha256(v.encode()).hexdigest() for v in h.headers.values()),
    })


def r_slow(h, q):
    secs, beat = int(q.get("s", "1")), int(q.get("beat", "0"))
    mark("slow-%d.started" % secs, str(time.time()))
    if not (beat or q.get("headers_first")):
        time.sleep(secs)
        mark("slow-%d.done" % secs, str(time.time()))
        return h.send_json(200, {"slept": secs})
    # headers now, body later (headers_first=1), or one chunk every `beat` seconds
    h.send_response(200)
    h.send_header("Content-Type", "text/plain")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()
    h.wfile.flush()
    end = time.monotonic() + secs
    while time.monotonic() < end:
        time.sleep(min(beat or secs, max(end - time.monotonic(), 0)))
        h.wfile.write(b"2\r\n.\n\r\n")
        h.wfile.flush()
    h.wfile.write(b"0\r\n\r\n")
    mark("slow-%d.done" % secs, str(time.time()))


def r_stream(h, q):
    n, sent = int(q.get("n", "10")), 0
    h.send_response(200)
    h.send_header("Content-Type", "text/plain")
    h.send_header("Transfer-Encoding", "chunked")
    h.send_header("Cache-Control", "no-cache")
    h.end_headers()
    try:
        for i in range(n):
            line = ("%d %.3f\n" % (i, time.time())).encode()
            h.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
            h.wfile.flush()
            sent += 1
            time.sleep(1)
        h.wfile.write(b"0\r\n\r\n")
    finally:
        mark("stream-%d.done" % n, str(sent))


ROUTES = {("POST", "/v1/exec"): r_exec, ("PUT", "/v1/files"): r_put_file, ("GET", "/v1/files"): r_get_file,
          ("GET", "/echo"): r_echo, ("GET", "/slow"): r_slow, ("GET", "/stream"): r_stream}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


Server(("0.0.0.0", PORT), Handler).serve_forever()
'''
SERVER_ENTRYPOINT = ["python3", "-c", SERVER]
PLAIN_ENTRYPOINT = ["sleep", "infinity"]
IMAGE_ENV = {"PORT": str(PORT), "HOME": "/root", "PYTHONUNBUFFERED": "1"}

# Response headers whose values are recorded; any other keeps its name only.
SAFE_RESPONSE_HEADERS = frozenset({
    "content-type", "content-length", "content-range", "accept-ranges", "transfer-encoding", "connection",
    "server", "via", "retry-after", "alt-svc", "cache-control",
})
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]+)?")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_DO_TOKEN = re.compile(r"do[por]_v1_[0-9a-f]{16,}")


def scrub_response(resp: httpx.Response, *, body_chars: int = 400) -> dict[str, Any]:
    """Status, response headers (values only if safe) and the start of the body."""
    try:
        body: Any = resp.json()
    except ValueError:
        body = resp.text[:body_chars]
    return {
        "status": resp.status_code,
        "http_version": resp.http_version,
        "headers": {k.lower(): (v if k.lower() in SAFE_RESPONSE_HEADERS else "<redacted>")
                    for k, v in sorted(resp.headers.items())},
        "body": body,
    }


def payload(size: int, seed: int) -> Iterator[bytes]:
    """Incompressible, reproducible bytes in 1 MiB blocks."""
    rnd = random.Random(seed)
    left = size
    while left > 0:
        n = min(left, MIB)
        yield rnd.randbytes(n)
        left -= n


def payload_sha(size: int, seed: int) -> str:
    digest = hashlib.sha256()
    for block in payload(size, seed):
        digest.update(block)
    return digest.hexdigest()


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 4)


def latency_stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95), "max": pct(values, 1.0),
            "mean": round(statistics.fmean(values), 4) if values else None}


def clip(obj: Any, limit: int = 4000) -> Any:
    """Long strings shortened before the redaction regexes run (they backtrack badly on MiB-long words)."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else f"{obj[:limit]}...<{len(obj)} chars>"
    if isinstance(obj, dict):
        return {k: clip(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clip(v, limit) for v in obj]
    return obj


class Probe:
    def __init__(self, token: str, access_token: str, out: Path) -> None:
        self.token = token
        self.access_token = access_token
        self.secrets: list[str] = [token, access_token]
        self.out = out
        self.log = (out / "events.jsonl").open("a")
        self.summary: dict[str, Any] = {}
        self.fixtures: dict[str, Any] = {}
        self.vms: list[str] = []
        self.checkpoints: list[str] = []
        self.deleted: set[str] = set()
        self.run_tag = ""
        self.edge_only = False  # --edge-timeouts: E1, then E3b only
        self.http = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=httpx.Timeout(120.0))
        # The endpoint only ever sees the access-only token.
        self.edge = httpx.Client(headers={"Authorization": f"Bearer {access_token}"},
                                 timeout=httpx.Timeout(200.0, connect=30.0),
                                 limits=httpx.Limits(max_connections=80, max_keepalive_connections=80))
        self.t0 = time.monotonic()
        self.lock = threading.Lock()

    # ------------------------------------------------------------ plumbing

    def new_secret(self) -> str:
        value = secrets.token_urlsafe(32)
        self.secrets.append(value)
        return value

    def redact(self, obj: Any) -> Any:
        text = json.dumps(obj, default=str)
        for value in self.secrets:  # before clipping, so that no token is cut in half
            text = text.replace(value, "<REDACTED_TOKEN>")
        text = json.dumps(clip(json.loads(text)))
        text = _DO_TOKEN.sub("<REDACTED_TOKEN>", _JWT.sub("<REDACTED_JWT>", text))
        return json.loads(_EMAIL.sub("<REDACTED_EMAIL>", text))

    def event(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.monotonic() - self.t0, 2), "kind": kind, **data}
        with self.lock:
            self.log.write(json.dumps(self.redact(record)) + "\n")
            self.log.flush()
        brief = {k: v for k, v in record.items() if k not in ("body", "response", "request")}
        print(json.dumps(self.redact(brief))[:300], flush=True)

    def note(self, item: str, key: str, value: Any) -> None:
        self.summary.setdefault(item, {})[key] = value
        self.event("finding", item=item, key=key, value=value)

    def fixture(self, name: str, resp: httpx.Response) -> None:
        self.fixtures.setdefault(name, scrub_response(resp))

    def api(self, method: str, path: str, *, json_body: Any = None, params: Any = None,
            expect_ok: bool = True) -> httpx.Response:
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
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if expect_ok and resp.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> {resp.status_code}: {str(body)[:500]}")
            return resp
        raise RuntimeError(f"{method} {path}: still rate limited")

    def paginate(self, path: str, key: str) -> list[dict[str, Any]]:
        items, page = [], 1
        while True:
            body = self.api("GET", path, params={"page": page, "per_page": 200}).json()
            items += body.get(key) or []
            if not ((body.get("links") or {}).get("pages") or {}).get("next"):
                return items
            page += 1

    def ours(self) -> dict[str, Any]:
        """Counts of the team's resources, and the ids of ours (probe tag or name prefix)."""
        vms = [m for m in self.paginate("/microvms", "microvms") if m.get("state") not in ("terminating", "terminated")]
        cps = self.paginate("/microvms/checkpoints", "checkpoints")
        our_vms = [m["id"] for m in vms if PROBE_TAG in (m.get("tags") or []) or str(m.get("name", "")).startswith("probe-")]
        our_cps = [c["id"] for c in cps if str(c.get("name", "")).startswith("probe-")]
        return {"team_microvms": len(vms), "team_checkpoints": len(cps), "our_microvms": our_vms, "our_checkpoints": our_cps}

    def control_exec(self, vm: str, argv: list[str]) -> dict[str, Any]:
        resp = self.api("POST", f"/microvms/{vm}/exec", json_body={"argv": argv}, expect_ok=False)
        try:
            return {"status": resp.status_code, **resp.json()}
        except ValueError:
            return {"status": resp.status_code, "text": resp.text[:500]}

    def get(self, vm: str) -> dict[str, Any]:
        return self.api("GET", f"/microvms/{vm}").json()["microvm"]

    def wait_state(self, vm: str, want: set[str], *, timeout: float, interval: float = 2.0) -> tuple[str, float, list]:
        started = time.monotonic()
        seen: list[tuple[float, str]] = []
        while True:
            obj = self.get(vm)
            state = obj.get("state", "?")
            if not seen or seen[-1][1] != state:
                seen.append((round(time.monotonic() - started, 1), state))
            if state in want or time.monotonic() - started > timeout:
                return state, round(time.monotonic() - started, 1), seen
            if state == "failed":
                raise RuntimeError(f"MicroVM {vm} failed: {obj.get('failure_reason')}")
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
            time.sleep(2)

    def create(self, name: str, *, image: str | None = None, checkpoint: str | None = None, idle: str,
               environment: dict[str, str] | None = None, tags: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "tags": tags, "auto_pause": {"idle_timeout": idle}}
        if image:
            body.update({"region": REGION, "size": SIZE, "source": {"oci_ref": image}, "http_port": PORT,
                         "http_protocol": "http"})
            if environment:
                body["environment"] = environment
        else:
            body["source"] = {"checkpoint_id": checkpoint}
        resp = self.api("POST", "/microvms", json_body=body, expect_ok=False)
        if resp.status_code >= 400:
            raise RuntimeError(f"create {name} -> {resp.status_code}: {resp.text[:500]}")
        vm = resp.json()["microvm"]
        self.vms.append(vm["id"])
        return vm

    def delete_vm(self, vm: str) -> int:
        resp = self.api("DELETE", f"/microvms/{vm}", expect_ok=False)
        if resp.status_code < 400 or resp.status_code == 404:
            self.deleted.add(vm)
        return resp.status_code

    def delete_checkpoint(self, cp: str) -> int:
        status = 0
        for _ in range(24):  # 400 while still capturing
            status = self.api("DELETE", f"/microvms/checkpoints/{cp}", expect_ok=False).status_code
            if status != 400:
                break
            time.sleep(5)
        if status < 400 or status == 404:
            self.checkpoints.remove(cp)
        return status

    # Endpoint calls -------------------------------------------------------

    def call(self, host: str, method: str, path: str, *, key: str | None, edge_auth: bool = True,
             client: httpx.Client | None = None, log: bool = True, **kwargs: Any) -> tuple[httpx.Response | None, float, str | None]:
        headers = dict(kwargs.pop("headers", None) or {})
        if key is not None:
            headers["x-api-key"] = key
        started = time.monotonic()
        try:
            if edge_auth:
                resp = (client or self.edge).request(method, f"https://{host}{path}", headers=headers, **kwargs)
            else:  # a bare request: no Authorization at all
                resp = httpx.request(method, f"https://{host}{path}", headers=headers, timeout=60.0, **kwargs)
            error = None
        except httpx.HTTPError as exc:
            resp, error = None, f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - started, 4)
        if log:
            self.event("endpoint", method=method, path=path.split("?")[0], status=resp.status_code if resp else None,
                       elapsed=elapsed, error=error,
                       response=scrub_response(resp) if resp is not None and path.split("?")[0] != "/echo" else None)
        return resp, elapsed, error

    def sx(self, host: str, key: str, command: str, timeout: float = 30) -> dict[str, Any]:
        """Run a command through the sandbox server."""
        resp, elapsed, error = self.call(host, "POST", "/v1/exec", key=key, json={"command": command, "timeout": timeout})
        if resp is None:
            return {"error": error, "elapsed": elapsed}
        try:
            return {"status": resp.status_code, "elapsed": elapsed, **resp.json()}
        except ValueError:
            return {"status": resp.status_code, "elapsed": elapsed, "text": resp.text[:300]}

    def bench(self, label: str, fn: Callable[[], tuple[httpx.Response | None, float, str | None]], *,
              n: int, concurrency: int) -> dict[str, Any]:
        latencies: list[float] = []
        statuses: Counter[str] = Counter()
        versions: Counter[str] = Counter()

        def one(_: int) -> None:
            resp, elapsed, error = fn()
            with self.lock:
                statuses[str(resp.status_code) if resp is not None else (error or "error").split(":")[0]] += 1
                if resp is not None:
                    versions[resp.http_version] += 1
                    if resp.status_code == 200:
                        latencies.append(elapsed)

        started = time.monotonic()
        if concurrency == 1:
            for i in range(n):
                one(i)
        else:
            with ThreadPoolExecutor(concurrency) as pool:
                list(pool.map(one, range(n)))
        wall = time.monotonic() - started
        result = {"n": n, "concurrency": concurrency, "wall_s": round(wall, 2), "rps": round(n / wall, 1),
                  "statuses": dict(statuses), "http_versions": dict(versions), "latency_s": latency_stats(latencies)}
        self.event("bench", label=label, **result)
        return result

    # ------------------------------------------------------------ steps

    def run(self, image: str, plain_image: str, skip: set[str]) -> None:
        run_id = secrets.token_hex(3)
        self.run_tag = f"{PROBE_TAG}-run-{run_id}"
        tags = [PROBE_TAG, self.run_tag]
        key_a = self.new_secret()

        # E1: create, hostname, readiness, server sanity --------------------------
        started = time.monotonic()
        vm = self.create(f"probe-ep-{run_id}-a", image=image, idle=IDLE_A, tags=tags,
                         environment={"SANDBOX_TOKEN": key_a})
        a = vm["id"]
        self.note("E1", "create_response_state", vm.get("state"))
        self.note("E1", "create_response_hostname", self._hostname_pattern(vm, a))
        state, _, seen = self.wait_state(a, {"running"}, timeout=300)
        self.note("E1", "create_to_running_s", {"seconds": round(time.monotonic() - started, 1), "states": seen})
        host, secs = self.wait_endpoint(a)
        self.note("E1", "url_active_after_running_s", secs)
        obj = self.get(a)
        self.note("E1", "hostname_pattern", host.replace(a, "<id>"))
        self.note("E1", "hostname_field_vs_urls", {"hostname_field": self._hostname_pattern(obj, a),
                                                   "urls": [{**u, "hostname": str(u.get("hostname", "")).replace(a, "<id>")}
                                                            for u in obj.get("urls") or []]})
        statuses: list[tuple[float, int | None]] = []
        while time.monotonic() - started < 400:
            resp, _, _ = self.call(host, "GET", "/healthz", key=None, log=False)
            code = resp.status_code if resp is not None else None
            if not statuses or statuses[-1][1] != code:
                statuses.append((round(time.monotonic() - started, 1), code))
            if code == 200:
                break
            time.sleep(0.5)
        self.note("E1", "create_to_first_healthz_200_s", {"seconds": round(time.monotonic() - started, 1),
                                                          "statuses": statuses})
        self.note("E1", "server_exit_code", self.sx(host, key_a, "echo out; echo err >&2; exit 3"))
        r =self.sx(host, key_a, "sleep 30 & sleep 30", timeout=2)
        self.note("E1", "server_timeout_kills_group", {
            "result": {k: r.get(k) for k in ("status", "exit_code", "timed_out", "seconds")},
            "sleeps_left": self.sx(host, key_a, "pgrep -c -x sleep || true").get("stdout")})
        r = self.sx(host, key_a, "head -c 3000000 /dev/zero | tr '\\0' a")
        self.note("E1", "server_output_cap", {"truncated": r.get("truncated"), "stdout_len": len(r.get("stdout") or "")})
        self.note("E1", "server_cwd", self.sx(host, key_a, "pwd").get("stdout"))
        self.note("E1", "vm_resources", self.sx(host, key_a, "id -u; nproc; free -m | head -2; df -m / /tmp | tail -n +2").get("stdout"))
        resp, _, _ = self.call(host, "POST", "/v1/exec", key=key_a, json={"command": "echo hi", "timeout": 5})
        if resp is not None:
            self.fixture("server_exec_200", resp)
        if self.edge_only:
            self.edge_timeouts(host, key_a)
            return

        # E2: header forwarding --------------------------------------------------
        if "E2" not in skip:
            resp, _, _ = self.call(host, "GET", "/echo?x=1", key=key_a, headers={"x-probe-custom": "yes",
                                                                                   "x-auth-scheme": "langsmith"})
            if resp is not None and resp.status_code == 200:
                body = resp.json()
                hashes = set(body.get("value_sha256") or [])
                sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
                self.note("E2", "header_names_seen_by_workload", body.get("header_names"))
                self.note("E2", "authorization_forwarded", "authorization" in (body.get("header_names") or []))
                self.note("E2", "access_token_value_visible", bool(hashes & {sha(self.access_token),
                                                                             sha(f"Bearer {self.access_token}")}))
                self.note("E2", "x_api_key_passes", sha(key_a) in hashes)
                self.note("E2", "probe_header_values", body.get("values"))
            else:
                self.note("E2", "echo_failed", scrub_response(resp) if resp is not None else None)

        # E10 (part 1): auth failure modes on a live MicroVM -----------------------
        if "E10" not in skip:
            for label, kwargs in (("no_edge_token", {"key": key_a, "edge_auth": False}),
                                  ("edge_token_no_x_api_key", {"key": None}),
                                  ("wrong_x_api_key", {"key": "wrong-key"}),
                                  ("healthz_no_x_api_key", {"key": None})):
                path = "/healthz" if label.startswith("healthz") else "/v1/files?path=/etc/hostname"
                resp, el, err = self.call(host, "GET", path, **kwargs)
                self.note("E10", label, scrub_response(resp) if resp is not None else err)
                if resp is not None:
                    self.fixture({"no_edge_token": "edge_401_no_token", "edge_token_no_x_api_key": "server_401_no_key",
                                  "wrong_x_api_key": "server_401_wrong_key"}.get(label, "server_healthz_200"), resp)

        # E3: edge request timeout -------------------------------------------------
        if "E3" not in skip:
            durations = (30, 55, 65, 90, 130)
            results: dict[int, Any] = {}

            def slow(s: int) -> None:
                resp, el, err = self.call(host, "GET", f"/slow?s={s}", key=key_a, timeout=s + 60.0)
                results[s] = {"status": resp.status_code if resp is not None else None, "seconds": el, "error": err,
                              "response": scrub_response(resp) if resp is not None and resp.status_code != 200 else None}
                if resp is not None and resp.status_code >= 500:
                    self.fixture("edge_timeout", resp)

            threads = [threading.Thread(target=slow, args=(s,)) for s in durations]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            time.sleep(5)
            markers = self.sx(host, key_a, "cd /tmp && for f in slow-*; do echo $f; done")
            self.note("E3", "slow_requests", {str(s): results.get(s) for s in durations})
            self.note("E3", "handler_markers", (markers.get("stdout") or "").split())

        # E5: streaming --------------------------------------------------------------
        if "E5" not in skip:
            streams: dict[int, Any] = {}

            def stream(n: int) -> None:
                started_s = time.monotonic()
                arrivals: list[float] = []
                status, error = None, None
                try:
                    with self.edge.stream("GET", f"https://{host}/stream?n={n}", headers={"x-api-key": key_a},
                                          timeout=n + 60.0) as resp:
                        status = resp.status_code
                        for line in resp.iter_lines():
                            if line.strip():
                                arrivals.append(round(time.monotonic() - started_s, 2))
                except httpx.HTTPError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                gaps = [round(b - a2, 2) for a2, b in zip(arrivals, arrivals[1:])]
                streams[n] = {"status": status, "error": error, "lines": len(arrivals),
                              "first_line_s": arrivals[0] if arrivals else None,
                              "last_line_s": arrivals[-1] if arrivals else None,
                              "max_gap_s": max(gaps) if gaps else None,
                              "incremental": bool(arrivals) and (arrivals[-1] - arrivals[0]) > 0.5 * max(len(arrivals) - 1, 1),
                              "total_s": round(time.monotonic() - started_s, 2)}

            threads = [threading.Thread(target=stream, args=(n,)) for n in (60, 90)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.note("E5", "streams", {str(n): streams.get(n) for n in (60, 90)})
            self.note("E5", "server_side_lines_sent", self.sx(host, key_a, "cd /tmp && for f in stream-*; do echo $f $(cat $f); done").get("stdout"))

        # E4: body and response limits ------------------------------------------------
        if "E4" not in skip:
            sizes = (1, 8, 32, 100, 256)
            results4: dict[str, Any] = {}
            for i, mib in enumerate(sizes):
                results4[f"{mib}MiB"] = self._transfer(host, key_a, mib * MIB, seed=i, chunked=False)
            results4["8MiB_chunked_encoding"] = self._transfer(host, key_a, 8 * MIB, seed=100, chunked=True)
            self.note("E4", "transfers", results4)
            # 32 MiB as 4 x 8 MiB with offsets, then two Range reads
            path = "/root/probe-data/offset.bin"
            whole = b"".join(payload(32 * MIB, 200))
            parts = []
            for k in range(4):
                part = whole[k * 8 * MIB:(k + 1) * 8 * MIB]
                resp, el, err = self.call(host, "PUT", f"/v1/files?path={path}&offset={k * 8 * MIB}", key=key_a,
                                          content=part, headers={"X-Content-Sha256": hashlib.sha256(part).hexdigest()})
                parts.append({"status": resp.status_code if resp is not None else err, "seconds": el})
            r = self.sx(host, key_a, f"sha256sum {path} | cut -c1-64")
            ranges = {}
            for label, rng, want in (("first_1KiB", "bytes=0-1023", whole[:1024]),
                                     ("middle_1MiB", f"bytes={10 * MIB}-{11 * MIB - 1}", whole[10 * MIB:11 * MIB]),
                                     ("suffix_100B", "bytes=-100", whole[-100:])):
                resp, el, err = self.call(host, "GET", f"/v1/files?path={path}", key=key_a, headers={"Range": rng})
                ranges[label] = {"status": resp.status_code if resp is not None else err,
                                 "content_range": resp.headers.get("content-range") if resp is not None else None,
                                 "matches": resp is not None and resp.content == want}
                if resp is not None and resp.status_code == 206:
                    self.fixture("server_range_206", resp)
            self.note("E4", "offset_upload", {"parts": parts, "sha_matches": (r.get("stdout") or "").strip() == hashlib.sha256(whole).hexdigest()})
            self.note("E4", "range_reads", ranges)
            resp, _, _ = self.call(host, "GET", "/v1/files?path=/root/no-such-file", key=key_a)
            if resp is not None:
                self.fixture("server_get_404", resp)
            self.sx(host, key_a, "rm -rf /root/probe-data")
            del whole

        # E6: latency and rate ------------------------------------------------------------
        if "E6" not in skip or "E7" not in skip:
            self.sx(host, key_a, "head -c 1024 /dev/urandom > /root/small.bin")
        if "E6" not in skip:
            read = lambda: self.call(host, "GET", "/v1/files?path=/root/small.bin", key=key_a, log=False)  # noqa: E731
            ex = lambda: self.call(host, "POST", "/v1/exec", key=key_a, json={"command": "true"}, log=False)  # noqa: E731
            self.note("E6", "exec_sequential_100", self.bench("exec seq", ex, n=100, concurrency=1))
            self.note("E6", "read_sequential_100", self.bench("read seq", read, n=100, concurrency=1))
            for c in (10, 25, 50):
                self.note("E6", f"read_concurrent_{c}_x500", self.bench(f"read c{c}", read, n=500, concurrency=c))
            self.note("E6", "exec_concurrent_10_x200", self.bench("exec c10", ex, n=200, concurrency=10))
            lat: list[float] = []
            codes: Counter[int] = Counter()
            for _ in range(20):
                t = time.monotonic()
                r = self.control_exec(a, ["true"])
                codes[r["status"]] += 1
                if r["status"] == 200:
                    lat.append(time.monotonic() - t)
            self.note("E6", "control_exec_sequential_20", {"statuses": dict(codes), "latency_s": latency_stats(lat)})

        # E7: HTTP/2 -----------------------------------------------------------------------
        if "E7" not in skip:
            try:
                import h2  # noqa: F401
            except ImportError:
                self.note("E7", "skipped", "h2 not installed (run with `uv run --with h2`)")
            else:
                with httpx.Client(http2=True, headers={"Authorization": f"Bearer {self.access_token}"},
                                  timeout=httpx.Timeout(60.0, connect=30.0),
                                  limits=httpx.Limits(max_connections=30, max_keepalive_connections=30)) as h2c:
                    read2 = lambda: self.call(host, "GET", "/v1/files?path=/root/small.bin", key=key_a,  # noqa: E731
                                              client=h2c, log=False)
                    self.note("E7", "read_sequential_100_h2", self.bench("h2 read seq", read2, n=100, concurrency=1))
                    self.note("E7", "read_concurrent_25_x500_h2", self.bench("h2 read c25", read2, n=500, concurrency=25))

        # E8: pause, clones, idle, auto-resume ------------------------------------------------
        if "E8" not in skip:
            for sample in (1, 2):
                t = time.monotonic()
                self.api("POST", f"/microvms/{a}/pause")
                st, secs, seen = self.wait_state(a, {"paused"}, timeout=60, interval=1)
                pause_s = round(time.monotonic() - t, 2)
                resp, el, err = self.call(host, "GET", "/v1/files?path=/root/small.bin", key=key_a)
                self.note("E8", f"explicit_pause_then_request_{sample}", {
                    "pause_to_paused_s": pause_s, "state": st, "request_status": resp.status_code if resp is not None else err,
                    "request_s": el, "state_after": self.get(a).get("state")})
            self.sx(host, key_a, "nohup sleep 3600 >/dev/null 2>&1 & echo started")
            resp = self.api("POST", f"/microvms/{a}/checkpoints", json_body={"name": f"probe-ep-{run_id}-cp"})
            cp = resp.json()["checkpoint"]
            self.checkpoints.append(cp["id"])
            t = time.monotonic()
            status = cp.get("status")
            while status not in ("CHECKPOINT_AVAILABLE", "CHECKPOINT_FAILED") and time.monotonic() - t < 600:
                time.sleep(3)
                status = self.api("GET", f"/microvms/checkpoints/{cp['id']}").json()["checkpoint"]["status"]
            self.note("E8", "checkpoint_capture", {"status": status, "seconds": round(time.monotonic() - t, 1)})
            if status != "CHECKPOINT_AVAILABLE":
                raise RuntimeError(f"checkpoint {status}")
            t = time.monotonic()
            clone = self.create(f"probe-ep-{run_id}-b", checkpoint=cp["id"], idle=IDLE_SHORT, tags=tags)
            b = clone["id"]
            st, _, seen = self.wait_state(b, {"running"}, timeout=300)
            bhost, _ = self.wait_endpoint(b)
            self.note("E8", "clone_to_url_active_s", {"seconds": round(time.monotonic() - t, 1), "states": seen,
                                                      "auto_pause_in_response": clone.get("auto_pause")})
            resp, el, _ = self.call(bhost, "GET", "/v1/files?path=/etc/hostname", key=key_a)
            self.note("E8", "clone_first_request_inherited_token", {"status": resp.status_code if resp is not None else None,
                                                                    "seconds": el, "since_create_s": round(time.monotonic() - t, 1)})
            self.note("E8", "clone_process_survived", self.sx(bhost, key_a, "pgrep -x sleep >/dev/null && echo yes || echo no").get("stdout"))
            key_b = self.new_secret()
            t = time.monotonic()
            r = self.control_exec(b, ["sh", "-c", f"mkdir -p {os.path.dirname(TOKEN_FILE)} && umask 077 && "
                                                  f"printf %s \"$1\" > {TOKEN_FILE}", "sh", key_b])
            rekey = {"exec_status": r.get("status"), "exec_s": round(time.monotonic() - t, 2)}
            for label, k in (("old_token", key_a), ("new_token", key_b)):
                resp, el, _ = self.call(bhost, "GET", "/v1/files?path=/etc/hostname", key=k)
                rekey[label] = resp.status_code if resp is not None else None
            resp, _, _ = self.call(host, "GET", "/v1/files?path=/etc/hostname", key=key_a)
            rekey["source_still_accepts_its_token"] = resp.status_code if resp is not None else None
            self.note("E8", "clone_rekey", rekey)
            self.note("E8", "clone_hostname_differs", bhost != host)

            # B has a 30 s idle timeout: traffic every 10 s for 75 s keeps it running?
            states = []
            t = time.monotonic()
            while time.monotonic() - t < 75:
                resp, el, _ = self.call(bhost, "GET", "/v1/files?path=/etc/hostname", key=key_b, log=False)
                states.append((round(time.monotonic() - t), resp.status_code if resp is not None else None,
                               round(el, 2), self.get(b).get("state")))
                time.sleep(10)
            self.note("E8", "traffic_every_10s_idle_30s", states)

            # one in-flight 45 s request and nothing else: does B pause under it?
            seen_states: list[tuple[float, str]] = []
            done = threading.Event()

            def watch() -> None:
                t0 = time.monotonic()
                while not done.is_set():
                    s = self.get(b).get("state")
                    if not seen_states or seen_states[-1][1] != s:
                        seen_states.append((round(time.monotonic() - t0, 1), s))
                    done.wait(4)

            watcher = threading.Thread(target=watch)
            watcher.start()
            resp, el, err = self.call(bhost, "GET", "/slow?s=45", key=key_b, timeout=120.0)
            done.set()
            watcher.join()
            self.note("E8", "inflight_45s_request_idle_30s", {"status": resp.status_code if resp is not None else err,
                                                               "seconds": el, "states": seen_states})

            # then silence: auto-pause, and the auto-resume latency of the next request
            st, secs, seen = self.wait_state(b, {"paused"}, timeout=150, interval=3)
            self.note("E8", "auto_pause_after_silence", {"state": st, "seconds": secs, "states": seen})
            if st == "paused":
                time.sleep(5)
                for sample in (1, 2):
                    resp, el, _ = self.call(bhost, "GET", "/v1/files?path=/etc/hostname", key=key_b)
                    self.note("E8", f"auto_resume_request_{sample}", {"status": resp.status_code if resp is not None else None,
                                                                      "seconds": el, "state_after": self.get(b).get("state")})
                    if sample == 1:
                        st, _, _ = self.wait_state(b, {"paused"}, timeout=150, interval=3)
                        time.sleep(5)

            # E10 (part 2): a deleted MicroVM's endpoint
            self.note("E8", "delete_clone", self.delete_vm(b))
            if "E10" not in skip:
                after = []
                t = time.monotonic()
                for wait in (0, 15, 45):
                    time.sleep(max(0.0, wait - (time.monotonic() - t)))
                    resp, el, err = self.call(bhost, "GET", "/v1/files?path=/etc/hostname", key=key_b)
                    after.append({"after_s": wait, "status": resp.status_code if resp is not None else err})
                    if resp is not None and resp.status_code >= 400:
                        self.fixture("edge_deleted", resp)
                self.note("E10", "deleted_microvm", after)
            self.note("E8", "delete_checkpoint", self.delete_checkpoint(cp["id"]))

        self.note("E1", "delete_a", self.delete_vm(a))

        # E9: bootstrap on an image without the server ------------------------------------------
        if "E9" not in skip:
            key_c = self.new_secret()
            t = time.monotonic()
            vm = self.create(f"probe-ep-{run_id}-c", image=plain_image, idle=IDLE_SHORT, tags=tags)
            c = vm["id"]
            st, _, seen = self.wait_state(c, {"running"}, timeout=300)
            self.note("E9", "create_to_running_s", {"seconds": round(time.monotonic() - t, 1), "states": seen})
            urls = self.get(c).get("urls") or []
            self.note("E9", "url_status_before_server", [u.get("status") for u in urls])
            script = base64.b64encode(SERVER.encode()).decode()
            tb = time.monotonic()
            r = self.control_exec(c, ["sh", "-c",
                                      "mkdir -p /opt/microvm-sandbox " + os.path.dirname(TOKEN_FILE) + " && "
                                      "printf %s \"$1\" | base64 -d > /opt/microvm-sandbox/server.py && "
                                      f"(umask 077 && printf %s \"$2\" > {TOKEN_FILE}) && "
                                      "setsid nohup python3 /opt/microvm-sandbox/server.py >/dev/null 2>&1 </dev/null & "
                                      "echo started", "sh", script, key_c])
            self.note("E9", "bootstrap_exec", {k: r.get(k) for k in ("status", "exit_code", "stdout", "stderr")}
                      | {"seconds": round(time.monotonic() - tb, 2), "script_bytes": len(SERVER)})
            chost, secs = self.wait_endpoint(c, timeout=120)
            codes: list[tuple[float, int | None]] = []
            while time.monotonic() - tb < 120:
                resp, _, _ = self.call(chost, "GET", "/healthz", key=None, log=False)
                code = resp.status_code if resp is not None else None
                if not codes or codes[-1][1] != code:
                    codes.append((round(time.monotonic() - tb, 1), code))
                if code == 200:
                    break
                time.sleep(0.5)
            resp, _, _ = self.call(chost, "GET", "/v1/files?path=/etc/hostname", key=key_c)
            self.note("E9", "bootstrap_to_ready", {"seconds": round(time.monotonic() - tb, 1), "healthz": codes,
                                                   "authenticated_status": resp.status_code if resp is not None else None})
            read = lambda: self.call(chost, "GET", "/v1/files?path=/etc/hostname", key=key_c, log=False)  # noqa: E731
            ex = lambda: self.call(chost, "POST", "/v1/exec", key=key_c, json={"command": "true"}, log=False)  # noqa: E731
            self.note("E9", "read_sequential_50", self.bench("c read seq", read, n=50, concurrency=1))
            self.note("E9", "exec_sequential_20", self.bench("c exec seq", ex, n=20, concurrency=1))
            self.note("E9", "delete_c", self.delete_vm(c))

    def edge_timeouts(self, host: str, key: str) -> None:
        """E3b: which response timeout the edge applies, to what, and whether it retries (sequential calls)."""
        def calls() -> list[list[str]]:
            out = self.sx(host, key, "cat /tmp/calls.log; : > /tmp/calls.log").get("stdout") or ""
            return [line.split() for line in out.splitlines() if line.strip()]

        def one(label: str, method: str, path: str, headers: dict[str, str] | None = None, **kwargs: Any) -> None:
            calls()  # reset the log
            first: list[float] = []
            t = time.monotonic()
            status, error, body_bytes = None, None, 0
            try:
                with self.edge.stream(method, f"https://{host}{path}", headers={"x-api-key": key, **(headers or {})},
                                      timeout=httpx.Timeout(150.0, connect=30.0), **kwargs) as resp:
                    status = resp.status_code
                    first.append(round(time.monotonic() - t, 2))  # response headers
                    for block in resp.iter_raw():
                        body_bytes += len(block)
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = round(time.monotonic() - t, 2)
            # the reset call's own line was truncated with the log; the read call's own line is the last one
            handler = calls()[:-1]
            starts = [round(float(c[4]) - float(handler[0][4]), 1) for c in handler] if handler else []
            self.note("E3b", label, {"status": status, "error": error, "seconds": elapsed,
                                     "headers_at_s": first[0] if first else None, "body_bytes": body_bytes,
                                     "handler_calls": len(handler), "handler_starts_s": starts,
                                     "distinct_request_ids": len({c[2] for c in handler})})

        for s in (5, 10, 14, 16, 20, 30):
            one(f"get_slow_{s}s", "GET", f"/slow?s={s}")
        for s in (10, 14, 16, 20, 30):
            one(f"post_exec_sleep_{s}s", "POST", "/v1/exec", json={"command": f"sleep {s}", "timeout": s + 10})
        one("get_headers_first_30s", "GET", "/slow?s=30&headers_first=1")
        one("get_heartbeat_5s_for_40s", "GET", "/slow?s=40&beat=5")
        one("get_stream_1s_for_30s", "GET", "/stream?n=30")
        # a slow upload (~20 s at 1 MiB/s), then an immediate response: does the timer start after the body?
        size = 24 * MIB
        one("put_24MiB_then_respond", "PUT", "/v1/files?path=/root/up.bin", content=payload(size, 7),
            headers={"Content-Length": str(size)})

    @staticmethod
    def _hostname_pattern(obj: dict[str, Any], vm_id: str) -> Any:
        value = obj.get("hostname")
        return value.replace(vm_id, "<id>") if isinstance(value, str) else value

    def _transfer(self, host: str, key: str, size: int, *, seed: int, chunked: bool) -> dict[str, Any]:
        path = f"/root/probe-data/{size // MIB}{'c' if chunked else ''}.bin"
        want = payload_sha(size, seed)
        headers = {"X-Content-Sha256": want, "Content-Type": "application/octet-stream"}
        if not chunked:
            headers["Content-Length"] = str(size)
        resp, el, err = self.call(host, "PUT", f"/v1/files?path={path}", key=key, content=payload(size, seed),
                                  headers=headers, timeout=httpx.Timeout(900.0, connect=30.0))
        up: dict[str, Any] = {"status": resp.status_code if resp is not None else None, "error": err, "seconds": el,
                              "mib_per_s": round(size / MIB / el, 2) if resp is not None and resp.status_code == 200 else None}
        if resp is not None and resp.status_code != 200:
            up["response"] = scrub_response(resp)
        elif resp is not None:
            self.fixture("server_put_200", resp)
            up["server_seconds"] = resp.json().get("seconds")
        down: dict[str, Any] = {}
        if up["status"] == 200:
            digest, got, t = hashlib.sha256(), 0, time.monotonic()
            try:
                with self.edge.stream("GET", f"https://{host}/v1/files?path={path}", headers={"x-api-key": key},
                                      timeout=httpx.Timeout(900.0, connect=30.0)) as r:
                    down["status"] = r.status_code
                    for block in r.iter_bytes(MIB):
                        digest.update(block)
                        got += len(block)
            except httpx.HTTPError as exc:
                down["error"] = f"{type(exc).__name__}: {exc}"
            secs = time.monotonic() - t
            down.update({"bytes": got, "seconds": round(secs, 2), "sha_matches": digest.hexdigest() == want,
                         "mib_per_s": round(got / MIB / secs, 2) if got else None})
            self.sx(host, key, f"rm -f {path}")  # keep the disk small
        return {"upload": up, "download": down}

    def cleanup(self) -> None:
        for vm in self.vms:
            if vm not in self.deleted:
                self.summary.setdefault("cleanup", {}).setdefault("deleted_microvms", {})[vm] = self.delete_vm(vm)
        for cp in list(self.checkpoints):
            self.summary.setdefault("cleanup", {}).setdefault("deleted_checkpoints", {})[cp] = self.delete_checkpoint(cp)
        # Belt and braces: anything of this run that `self.vms` missed (e.g. a create that timed out client-side).
        if self.run_tag:
            for m in self.api("GET", "/microvms", params={"tag_name": self.run_tag}, expect_ok=False).json().get("microvms") or []:
                if m.get("state") not in ("terminating", "terminated") and m["id"] not in self.deleted:
                    self.summary.setdefault("cleanup", {}).setdefault("leftover_deleted", []).append(m["id"])
                    self.delete_vm(m["id"])


def build_images(repository: str, registry_token: str) -> tuple[str, str]:
    """Push the server image and the plain (`sleep infinity`) image; return both references."""
    base_digest = run(["crane", "digest", "--platform", PLATFORM, BASE_IMAGE]).strip()
    base_ref = f"{BASE_IMAGE.rsplit(':', 1)[0]}@{base_digest}"
    refs = []
    with tempfile.TemporaryDirectory(prefix="microvm-probe-build-") as tmp:
        config_dir = Path(tmp) / "docker-config"
        config_dir.mkdir(mode=0o700)
        auth = base64.b64encode(f"{registry_token}:{registry_token}".encode()).decode()
        (config_dir / "config.json").write_text(json.dumps({"auths": {repository.split("/", 1)[0]: {"auth": auth}}}))
        (config_dir / "config.json").chmod(0o600)
        env = {**os.environ, "DOCKER_CONFIG": str(config_dir)}
        for label, entrypoint in (("server", SERVER_ENTRYPOINT), ("plain", PLAIN_ENTRYPOINT)):
            tag = hashlib.sha256(json.dumps([base_digest, entrypoint, IMAGE_ENV], sort_keys=True).encode()).hexdigest()[:12]
            reference = f"{repository}:{label}-{tag}"
            mutate = ["crane", "mutate", base_ref, "-t", reference, "--user", "0:0", "--workdir", "/",
                      "--entrypoint", _csv_field(entrypoint), "--cmd", "", "--exposed-ports", str(PORT)]
            for key, value in IMAGE_ENV.items():
                mutate += ["--env", f"{key}={value}"]
            run(mutate, env=env)
            config = json.loads(run(["crane", "config", reference], env=env))["config"]
            if config.get("Entrypoint") != entrypoint:
                raise SystemExit(f"pushed image {reference} has an unexpected entrypoint")
            refs.append(reference)
    return refs[0], refs[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, help="also write scrubbed endpoint responses for the transport tests here")
    parser.add_argument("--repository", help="default: DO_MICROVM_IMAGE's registry + /microvm-endpoint-probe")
    parser.add_argument("--image", help="skip the build: server image reference")
    parser.add_argument("--plain-image", help="skip the build: `sleep infinity` image reference (E9)")
    parser.add_argument("--skip", default="", help="comma-separated E-items to skip (E2-E10)")
    parser.add_argument("--edge-timeouts", action="store_true",
                        help="E1, then only E3b: sequential timeout and retry checks (one MicroVM, ~10 min)")
    add_env_file_argument(parser)
    args = parser.parse_args()
    env = load_env(args.env_file)
    token, access = env.get("DO_MICROVM_API_TOKEN"), env.get("DO_MICROVM_ACCESS_TOKEN")
    if not token or not access:
        print("DO_MICROVM_API_TOKEN and DO_MICROVM_ACCESS_TOKEN are required", file=sys.stderr)
        return 2
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    image, plain = args.image, args.plain_image
    if not (image and plain):
        repository = args.repository or (env.get("DO_MICROVM_IMAGE", "").rsplit("/", 1)[0] + "/microvm-endpoint-probe")
        registry_token = os.environ.get("DO_REGISTRY_TOKEN") or env.get("DO_REGISTRY_TOKEN")
        if not repository.startswith("registry.") or not registry_token or shutil.which("crane") is None:
            print("building needs --repository (or DO_MICROVM_IMAGE), DO_REGISTRY_TOKEN and crane", file=sys.stderr)
            return 2
        built = build_images(repository, registry_token)
        image, plain = image or built[0], plain or built[1]
    args.out.mkdir(parents=True, exist_ok=True)
    probe = Probe(token, access, args.out)
    probe.edge_only = args.edge_timeouts
    probe.summary["images"] = {"server": image, "plain": plain}
    before = probe.ours()
    probe.summary["team_before"] = before
    if before["our_microvms"] or before["our_checkpoints"]:
        print(f"refusing to start: the team has probe resources: {before}", file=sys.stderr)
        (args.out / "summary.json").write_text(json.dumps(probe.redact(probe.summary), indent=2) + "\n")
        return 3
    code = 0
    try:
        probe.run(image, plain, skip)
    except Exception as exc:  # noqa: BLE001 - record and still clean up
        probe.event("error", error=f"{type(exc).__name__}: {exc}")
        probe.summary["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        try:
            probe.cleanup()
            time.sleep(10)
            probe.summary["team_after"] = probe.ours()
        except Exception as exc:  # noqa: BLE001
            probe.summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            code = 1
        if args.fixtures is not None and code == 0:
            args.fixtures.mkdir(parents=True, exist_ok=True)
            for name, data in probe.fixtures.items():
                (args.fixtures / f"{name}.json").write_text(json.dumps(probe.redact(data), indent=2, sort_keys=True) + "\n")
            probe.summary["fixtures"] = sorted(probe.fixtures)
        (args.out / "summary.json").write_text(json.dumps(probe.redact(probe.summary), indent=2, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
