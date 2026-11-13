"""The sandbox server: runs INSIDE a MicroVM and is the data plane of EndpointTransport.

One file, Python standard library only, so that the same source is both the
sandbox image's entrypoint (scripts/build_sandbox_image.py) and what
EndpointTransport installs over exec into any image that has python3
(transport/endpoint.py). Importing this module starts nothing.

    python3 sandbox_server.py [--port 8080] [--token-file PATH]

Routes (everything except the first two needs `x-api-key: <sandbox token>`):

    GET  /, /healthz           liveness, constant body
    GET  /v1/info              {"server": "microvm-sandbox", "protocol": 2, "origin", ...}
    POST /v1/exec              {"command", "timeout", "budget"?, "cwd"?}: starts the
                               command (bash, else sh) and waits up to `budget`
                               seconds; answers its result, or {"status": "running",
                               "job_id"} when it is still running
    GET  /v1/jobs/<id>?wait=S  status (and result once done) of such a job, waiting
                               up to S seconds for it to finish
    GET  /v1/jobs/<id>/stdout  a finished job's raw output (also /stderr), `Range` supported
    POST /v1/jobs/<id>/cancel  kills the job's process group
    PUT  /v1/files?path=&offset=  raw body written at `offset` (0 truncates);
                               parents created; optional X-Content-Sha256
    GET  /v1/files?path=       raw bytes, single `Range` supported, X-File-Size

Every answer is complete within `MAX_SYNC_BUDGET` seconds plus a bounded
transfer: a request through the MicroVM endpoint has to be answered within
about 15 s, so longer commands are jobs that the client polls, and a result
carries its output inline only up to `INLINE_OUTPUT` bytes (larger output is
read from the job's output routes in `Range` pages, like files). A command is identified by
the client's `x-idempotency-key`: a repeated request with the same key (a
retry) attaches to the same job instead of running the command again.

The sandbox token is read from the token file on every request (cached by
modification time), falling back to $SANDBOX_TOKEN; with neither, every
authenticated route answers 401. The server logs nothing: request headers
carry credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

SERVER_NAME = "microvm-sandbox"
PROTOCOL = 2  # bump on any change a client depends on; EndpointTransport checks it
# "file" when run from a file (installed over exec), "inline" when run as `python3 -c` (the image's entrypoint)
ORIGIN = "file" if "__file__" in globals() else "inline"
DEFAULT_PORT = 8080
DEFAULT_TOKEN_FILE = "/var/lib/microvm-sandbox/token"
DEFAULT_JOB_ROOT = "/tmp/.microvm-sandbox/jobs"
MAX_SYNC_BUDGET = 10.0  # seconds a request may wait for a command
MAX_OUTPUT = 4 * 1024 * 1024  # bytes of stdout + stderr a client reads per command (it truncates beyond)
INLINE_OUTPUT = 256 * 1024  # stdout + stderr up to this size come back in the result itself
JOB_TTL = 600.0  # seconds a finished job (and its idempotency key) is kept
TIMEOUT_EXIT_CODE = 124
_CHUNK = 1024 * 1024


class _Token:
    """The expected sandbox token: the token file if it exists, else $SANDBOX_TOKEN."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._mtime: float | None = None
        self._value = ""
        self._lock = threading.Lock()

    def get(self) -> str:
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            return os.environ.get("SANDBOX_TOKEN", "")
        with self._lock:
            if mtime != self._mtime:
                try:
                    with open(self.path) as f:
                        self._value = f.read().strip()
                except OSError:
                    self._value = ""
                self._mtime = mtime
            return self._value


class _Job:
    """One command, its output captured to files in its own directory."""

    def __init__(self, root: str, command: str, timeout: float, cwd: str | None) -> None:
        self.id = uuid.uuid4().hex
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, mode=0o700)
        self.timeout = timeout
        shell = "bash" if shutil.which("bash") else "sh"
        with open(os.path.join(self.dir, "out"), "wb") as out, open(os.path.join(self.dir, "err"), "wb") as err:
            # The command travels as an argument, never written to disk (a recursive
            # grep inside the VM must not find its own pattern), in its own session so
            # that a timeout or cancel kills everything it started.
            self.proc = subprocess.Popen([shell, "-c", command], cwd=cwd or "/", stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=err, start_new_session=True)
        self.started = time.monotonic()
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.timed_out = False
        self.cancelled = False
        self.done = threading.Event()
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self) -> None:
        try:
            code = self.proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.timed_out = True
            self.kill()
            code = self.proc.wait()
        self.exit_code = TIMEOUT_EXIT_CODE if self.timed_out else (128 - code if code < 0 else code)
        self.finished_at = time.monotonic()
        self.done.set()

    def kill(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except OSError:
                return
            if sig == signal.SIGTERM:
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    continue
                # the shell is gone; still kill whatever it left in its group
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                return

    def status(self) -> dict:
        if not self.done.is_set():
            return {"job_id": self.id, "status": "running", "seconds": round(time.monotonic() - self.started, 3)}
        out_size = os.path.getsize(self.output_path("stdout"))
        err_size = os.path.getsize(self.output_path("stderr"))
        result = {"job_id": self.id, "status": "done", "exit_code": self.exit_code, "timed_out": self.timed_out,
                  "cancelled": self.cancelled, "stdout_bytes": out_size, "stderr_bytes": err_size,
                  "seconds": round((self.finished_at or time.monotonic()) - self.started, 3)}
        if out_size + err_size <= INLINE_OUTPUT:
            for stream in ("stdout", "stderr"):
                with open(self.output_path(stream), "rb") as f:
                    result[stream] = f.read().decode("utf-8", "replace")
        return result

    def output_path(self, stream: str) -> str:
        return os.path.join(self.dir, {"stdout": "out", "stderr": "err"}[stream])


class _Jobs:
    def __init__(self, root: str) -> None:
        self.root = root
        self._jobs: dict[str, _Job] = {}
        self._keys: dict[str, str] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._collect_garbage, daemon=True).start()

    def start(self, key: str | None, command: str, timeout: float, cwd: str | None) -> _Job:
        with self._lock:
            if key and key in self._keys and self._keys[key] in self._jobs:
                return self._jobs[self._keys[key]]  # a retry of a request we already ran
            os.makedirs(self.root, mode=0o700, exist_ok=True)
            job = _Job(self.root, command, timeout, cwd)
            self._jobs[job.id] = job
            if key:
                self._keys[key] = job.id
            return job

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _collect_garbage(self) -> None:
        while True:
            time.sleep(30)
            now = time.monotonic()
            with self._lock:
                old = [j for j in self._jobs.values() if j.finished_at is not None and now - j.finished_at > JOB_TTL]
                for job in old:
                    del self._jobs[job.id]
                    shutil.rmtree(job.dir, ignore_errors=True)
                gone = {job.id for job in old}
                for key in [k for k, v in self._keys.items() if v in gone]:
                    del self._keys[key]


def make_handler(token: _Token, jobs: _Jobs) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = SERVER_NAME

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - never log: headers carry credentials
            pass

        # ------------------------------------------------------------ helpers

        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, body: dict) -> None:
            self._send(status, json.dumps(body).encode(), "application/json")

        def _error(self, status: int, kind: str, detail: str = "") -> None:
            if self.command in ("PUT", "POST") and not self._body_read:
                self.close_connection = True  # unread body: do not reuse the connection
            body = {"error": kind, "detail": detail, "source": SERVER_NAME}
            self._send(status, json.dumps(body).encode(), "application/json",
                       {"Connection": "close"} if self.close_connection else None)

        def _body(self):  # noqa: ANN202 - generator of bytes
            self._body_read = True
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return
                    left = size
                    while left:
                        buf = self.rfile.read(min(left, _CHUNK))
                        if not buf:
                            raise ConnectionError("request body ended early")
                        left -= len(buf)
                        yield buf
                    self.rfile.readline()
            left = int(self.headers.get("Content-Length") or 0)
            while left:
                buf = self.rfile.read(min(left, _CHUNK))
                if not buf:
                    raise ConnectionError("request body ended early")
                left -= len(buf)
                yield buf

        def _json_body(self) -> dict:
            raw = b"".join(self._body())
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        # ------------------------------------------------------------ dispatch

        def _dispatch(self) -> None:
            self._body_read = False
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if self.command == "GET" and url.path in ("/", "/healthz"):
                return self._send(200, b"ok\n", "text/plain")
            want = token.get()
            got = self.headers.get("x-api-key", "")
            if not want or not hmac.compare_digest(got.encode(), want.encode()):
                return self._error(401, "unauthorized")
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts == ["v1", "info"] and self.command == "GET":
                    return self._json(200, {"server": SERVER_NAME, "protocol": PROTOCOL, "origin": ORIGIN,
                                            "max_sync_budget": MAX_SYNC_BUDGET, "max_output": MAX_OUTPUT,
                                            "inline_output": INLINE_OUTPUT})
                if parts == ["v1", "exec"] and self.command == "POST":
                    return self._exec()
                if parts[:2] == ["v1", "jobs"] and len(parts) == 3 and self.command == "GET":
                    return self._job(parts[2], query)
                if parts[:2] == ["v1", "jobs"] and len(parts) == 4 and parts[3] == "cancel" and self.command == "POST":
                    return self._cancel(parts[2])
                if (parts[:2] == ["v1", "jobs"] and len(parts) == 4 and parts[3] in ("stdout", "stderr")
                        and self.command == "GET"):
                    return self._job_output(parts[2], parts[3])
                if parts == ["v1", "files"] and self.command == "PUT":
                    return self._put_file(query)
                if parts == ["v1", "files"] and self.command == "GET":
                    return self._get_file(query)
                return self._error(404, "no_such_route")
            except (BrokenPipeError, ConnectionResetError, ConnectionError):
                self.close_connection = True
            except (ValueError, KeyError, TypeError) as exc:
                self._error(400, "bad_request", f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - answer instead of dropping the connection
                self._error(500, "internal", type(exc).__name__)

        do_GET = do_POST = do_PUT = _dispatch

        # ------------------------------------------------------------ commands

        def _budget(self, raw: object) -> float:
            try:
                return max(0.0, min(float(raw), MAX_SYNC_BUDGET))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return MAX_SYNC_BUDGET

        def _exec(self) -> None:
            req = self._json_body()
            command = req["command"]
            if not isinstance(command, str):
                raise TypeError("command must be a string")
            timeout = float(req.get("timeout") or 86400)
            cwd = req.get("cwd")
            if cwd is not None and not os.path.isdir(cwd):
                return self._error(400, "bad_cwd", str(cwd))
            job = jobs.start(self.headers.get("x-idempotency-key") or None, command, timeout, cwd)
            job.done.wait(self._budget(req.get("budget", MAX_SYNC_BUDGET)))
            self._json(200, job.status())

        def _job(self, job_id: str, query: dict) -> None:
            job = jobs.get(job_id)
            if job is None:
                return self._error(404, "no_such_job", job_id)
            job.done.wait(self._budget(query.get("wait", 0)))
            self._json(200, job.status())

        def _cancel(self, job_id: str) -> None:
            job = jobs.get(job_id)
            if job is None:
                return self._error(404, "no_such_job", job_id)
            if not job.done.is_set():
                job.cancelled = True
                job.kill()
                job.done.wait(5.0)
            self._json(200, job.status())

        def _job_output(self, job_id: str, stream: str) -> None:
            job = jobs.get(job_id)
            if job is None:
                return self._error(404, "no_such_job", job_id)
            if not job.done.is_set():
                return self._error(409, "job_running", job_id)
            self._send_file(job.output_path(stream))

        # ------------------------------------------------------------ files

        def _file_error(self, exc: OSError, path: str) -> None:
            if isinstance(exc, (IsADirectoryError,)):
                return self._error(409, "is_directory", path)
            if isinstance(exc, PermissionError):
                return self._error(403, "permission_denied", path)
            if isinstance(exc, (FileNotFoundError, NotADirectoryError, FileExistsError)):
                return self._error(404, "not_found", path)
            return self._error(500, "io_error", f"{type(exc).__name__}: {path}")

        def _put_file(self, query: dict) -> None:
            path = query.get("path", "")
            if not path.startswith("/"):
                return self._error(400, "invalid_path", path)
            offset = int(query.get("offset") or 0)
            if os.path.isdir(path):
                return self._error(409, "is_directory", path)
            try:
                os.makedirs(os.path.dirname(path) or "/", exist_ok=True)
                mode = "r+b" if offset and os.path.exists(path) else "wb"
                digest, size = hashlib.sha256(), 0
                with open(path, mode) as f:
                    if offset:
                        f.seek(offset)
                    for buf in self._body():
                        f.write(buf)
                        digest.update(buf)
                        size += len(buf)
            except OSError as exc:
                return self._file_error(exc, path)
            want = self.headers.get("X-Content-Sha256")
            if want and want != digest.hexdigest():
                return self._error(422, "checksum_mismatch", path)
            self._json(200, {"bytes": size, "offset": offset, "sha256": digest.hexdigest()})

        def _get_file(self, query: dict) -> None:
            path = query.get("path", "")
            if not path.startswith("/"):
                return self._error(400, "invalid_path", path)
            self._send_file(path)

        def _send_file(self, path: str) -> None:
            if os.path.isdir(path):
                return self._error(409, "is_directory", path)
            try:
                f = open(path, "rb")  # noqa: SIM115 - closed below
            except OSError as exc:
                return self._file_error(exc, path)
            with f:
                total = os.fstat(f.fileno()).st_size
                start, end, status = 0, total - 1, 200
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes=") and "," not in rng and total > 0:
                    first, _, last = rng[6:].partition("-")
                    if first:
                        start = int(first)
                        end = min(int(last), total - 1) if last else total - 1
                    else:
                        start = max(total - int(last), 0)
                    if start >= total:
                        return self._send(416, b"", "application/octet-stream",
                                          {"Content-Range": f"bytes */{total}", "X-File-Size": str(total)})
                    status = 206
                length = max(end - start + 1, 0)
                self.send_response(status)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(length))
                self.send_header("X-File-Size", str(total))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.end_headers()
                f.seek(start)
                while length > 0:
                    buf = f.read(min(length, _CHUNK))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    length -= len(buf)

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def serve(port: int = DEFAULT_PORT, *, token_file: str = DEFAULT_TOKEN_FILE, job_root: str = DEFAULT_JOB_ROOT,
          host: str = "0.0.0.0") -> None:  # noqa: S104 - the MicroVM's port is only reachable through its endpoint
    server = _Server((host, port), make_handler(_Token(token_file), _Jobs(job_root)))
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MicroVM sandbox server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or DEFAULT_PORT))
    parser.add_argument("--token-file", default=os.environ.get("SANDBOX_TOKEN_FILE") or DEFAULT_TOKEN_FILE)
    parser.add_argument("--job-root", default=os.environ.get("SANDBOX_JOB_ROOT") or DEFAULT_JOB_ROOT)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    args = parser.parse_args(argv)
    serve(args.port, token_file=args.token_file, job_root=args.job_root, host=args.host)


if __name__ == "__main__":
    main()
