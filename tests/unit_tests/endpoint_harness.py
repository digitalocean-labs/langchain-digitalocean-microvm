"""Test harness for EndpointTransport: the real sandbox server as a local process, and a fake edge.

`FakeEdge` stands in for DigitalOcean's edge in front of a MicroVM endpoint,
with the behaviour measured through it (scaled down in time): an attempt
whose upstream response has not completed within `budget` seconds is
re-sent upstream, up to `attempts` times, after which the client gets a 504
HTML page; workload responses gain `x-envoy-upstream-service-time`, the
edge's own answers do not; a deleted MicroVM answers 404 HTML; a request
without the access token gets the edge's JSON 401.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from langchain_digitalocean_microvm import sandbox_server

ACCESS_TOKEN = "access-token"
API_TOKEN = "control-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def start_server(root: Path, port: int, *, token: str | None = None, inline_source: str | None = None) -> subprocess.Popen:
    """The package's sandbox_server.py as a local process, token file and jobs under `root`.

    With `inline_source`, that source runs as `python -c` instead, as the sandbox image runs its server.
    """
    token_file = root / "token"
    if token is not None:
        token_file.write_text(token)
    # the file itself, not `-m langchain_digitalocean_microvm.sandbox_server`: that would import the whole
    # package (deepagents included) for nothing, and the server never imports it in a MicroVM either
    program = ["-c", inline_source] if inline_source is not None else [sandbox_server.__file__]
    proc = subprocess.Popen(
        [sys.executable, *program, "--port", str(port),
         "--token-file", str(token_file), "--job-root", str(root / "jobs"), "--host", "127.0.0.1"],
        env={k: v for k, v in os.environ.items() if k != "SANDBOX_TOKEN"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    wait_for_port(port)
    return proc


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class FakeEdge:
    """A threaded HTTP proxy to `upstream` with the edge's timeout, retry and error behaviour."""

    def __init__(self, upstream: str, *, budget: float = 1.0, attempts: int = 4) -> None:
        self.upstream = upstream
        self.budget = budget
        self.attempts = attempts
        self.deleted = False
        self.fail_next: int = 0  # answer the next N (completed) requests with a 502, response discarded
        self.unavailable_next: int = 0  # answer the next N requests with a 502 without forwarding (VM not ready yet)
        # Cut a workload response after this many body bytes (a response that did not finish within the
        # edge's budget on a slow link): the client sees an incomplete body.
        self.cut_responses_over: int | None = None
        self.forwarded = 0  # upstream attempts, including re-sends
        self._pool = ThreadPoolExecutor(32)
        self._lock = threading.Lock()
        edge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # noqa: ANN002
                pass

            def _answer(self, status: int, body: bytes, headers: dict[str, str], *, cut: int | None = None) -> None:
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if cut is not None and len(body) > cut:
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                if self.headers.get("Authorization") != f"Bearer {ACCESS_TOKEN}":
                    return self._answer(401, b'{"id":"Unauthorized","message":"Unable to authenticate you"}',
                                        {"Content-Type": "application/json"})
                if edge.deleted:
                    return self._answer(404, b"<!DOCTYPE html><title>DigitalOcean - Not found</title>",
                                        {"Content-Type": "text/html"})
                with edge._lock:
                    if edge.unavailable_next:
                        edge.unavailable_next -= 1
                        return self._answer(502, b"<html>502 Bad Gateway</html>", {"Content-Type": "text/html"})
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in ("host", "authorization", "content-length", "connection")}
                for _ in range(edge.attempts):
                    with edge._lock:
                        edge.forwarded += 1
                    future = edge._pool.submit(httpx.request, self.command, edge.upstream + self.path,
                                               headers=headers, content=body, timeout=60.0)
                    try:
                        resp = future.result(timeout=edge.budget)
                    except FutureTimeout:
                        continue  # the upstream attempt keeps running; its answer is dropped
                    except httpx.HTTPError:
                        return self._answer(502, b"<html>502 Bad Gateway</html>", {"Content-Type": "text/html"})
                    with edge._lock:
                        if edge.fail_next:
                            edge.fail_next -= 1
                            return self._answer(502, b"<html>502 Bad Gateway</html>", {"Content-Type": "text/html"})
                    out = {k: v for k, v in resp.headers.items()
                           if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding")}
                    out["x-envoy-upstream-service-time"] = "1"
                    return self._answer(resp.status_code, resp.content, out, cut=edge.cut_responses_over)
                return self._answer(504, b"<!DOCTYPE html><title>504: Gateway time-out</title>",
                                    {"Content-Type": "text/html"})

            do_GET = do_POST = do_PUT = _handle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._pool.shutdown(wait=False, cancel_futures=True)
