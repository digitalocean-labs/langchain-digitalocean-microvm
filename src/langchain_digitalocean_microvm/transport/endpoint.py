"""Endpoint transport: commands and files over the MicroVM's HTTPS endpoint.

The MicroVMs exec API runs single commands and is not built for high volume
(60 commands/min and 32 concurrent per team, one process per call, base64
file payloads), so this transport
talks to a small server inside the MicroVM instead (sandbox_server.py),
through the MicroVM's own HTTPS endpoint, and uses exec only to install,
upgrade or re-key that server. `connect(host=<microvm id>)`.

Requests carry two credentials: `Authorization: Bearer <access token>` for
DigitalOcean's edge (a token scoped to `microvm:access` only; never the
control token) and `x-api-key: <sandbox token>` for the server. The sandbox
token is derived from the control token and the MicroVM id
(`sandbox_token`), so any process holding the control token can reach a
sandbox without stored state, and a checkpoint clone (new id) gets its own
token. The server reads its token from a file; when it answers 401 the
transport writes the right token there over exec ("re-key") and retries.

Limits this transport is built around (measured through the endpoint):

- a response has to be complete within about 15 s of the request body
  arriving, and a request whose response has not started by then is re-sent
  by the edge. So `run()` asks the server to wait at most `sync_budget`
  seconds; a longer command continues as a job that is polled
  (`GET /v1/jobs/<id>?wait=`). Every command carries an idempotency key: a
  re-sent request attaches to the same job instead of running it again, and
  this transport retries failed requests with the same key. A result carries
  its output inline only when it is small; larger output, like files, is read
  in `Range` pages.
- uploads do not count against that budget but very large bodies are refused:
  files go up in `upload_chunk` pieces (`PUT ?offset=`, SHA-256 checked).
  Downloads do count against it: pages start at `download_chunk` bytes and
  are halved (down to `min_download_chunk`) whenever one does not arrive
  complete, so a slow link still gets every page through in time.

Connecting checks the server (`GET /v1/info`), including its protocol
version. A server this transport installed itself (over exec, into any image
with python3 whose own command does not use the HTTP port) is installed
again when its version differs; a server that is the image's own entrypoint
cannot be replaced from here (rebuild the image). When no compatible server
can run and `fallback` is on, the transport delegates to ExecTransport and
logs a warning saying why.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time
from collections.abc import Callable
from importlib import resources
from typing import Any

import httpx

from langchain_digitalocean_microvm import sandbox_server
from langchain_digitalocean_microvm.client import DEFAULT_BASE_URL, MicroVMClient
from langchain_digitalocean_microvm.config import DEFAULT_ENDPOINT_WAIT
from langchain_digitalocean_microvm.exceptions import (
    MicroVMClientError,
    MicroVMNotFoundError,
)
from langchain_digitalocean_microvm.models import ExecResult, MicroVMState
from langchain_digitalocean_microvm.transport.base import (
    SandboxTransport,
    TransportCredentials,
    TransportExecResult,
)
from langchain_digitalocean_microvm.transport.errors import (
    TransportError,
    TransportFileNotFoundError,
    TransportIsADirectoryError,
    TransportPermissionError,
)
from langchain_digitalocean_microvm.transport.exec import ExecTransport

logger = logging.getLogger(__name__)

DEFAULT_SERVER_DIR = "/opt/microvm-sandbox"
PROTOCOL = sandbox_server.PROTOCOL
TIMEOUT_EXIT_CODE = sandbox_server.TIMEOUT_EXIT_CODE
MAX_OUTPUT = sandbox_server.MAX_OUTPUT
_TOKEN_CONTEXT = b"langchain-digitalocean-microvm/sandbox-token/v1\0"
# Responses that came from the workload carry this header; the edge's own
# 401/404/502/504 do not (see `_from_workload`).
_UPSTREAM_HEADER = "x-envoy-upstream-service-time"
_RETRY_STATUSES = frozenset({502, 503, 504})
_TRUNCATED_NOTE = "[output truncated at 4 MiB]"

_REKEY_SCRIPT = (
    'umask 077 && mkdir -p "$(dirname "$1")" && printf %s "$2" > "$1.tmp" && mv "$1.tmp" "$1"'
)
# $1 server dir, $2 token file, $3 base64 server source, $4 token, $5 port, $6 "replace" to stop a
# server installed earlier from $1 first (never the image's own: that one runs as `python3 -c`).
_INSTALL_SCRIPT = """\
command -v python3 >/dev/null 2>&1 || exit 127
mkdir -p "$1" "$(dirname "$2")" || exit 125
if [ "$6" = replace ]; then
  if command -v pkill >/dev/null 2>&1; then pkill -f "$1/sandbox_server.py"
  else for p in /proc/[0-9]*; do
    case "$(tr '\\000' ' ' < "$p/cmdline" 2>/dev/null)" in *"$1/sandbox_server.py"*) kill "${p#/proc/}" 2>/dev/null;; esac
  done; fi
fi
printf '%s' "$3" | base64 -d > "$1/sandbox_server.py.tmp" && mv "$1/sandbox_server.py.tmp" "$1/sandbox_server.py" || exit 125
(umask 077 && printf %s "$4" > "$2.tmp" && mv "$2.tmp" "$2") || exit 125
__i=0
until python3 -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", int(sys.argv[1])))' "$5" 2>/dev/null; do
  __i=$((__i + 1)); [ "$__i" -ge 50 ] && exit 98; sleep 0.1
done
if command -v setsid >/dev/null 2>&1; then __detach=setsid; else __detach=; fi
SANDBOX_TOKEN_FILE="$2" PORT="$5" $__detach nohup python3 "$1/sandbox_server.py" >/dev/null 2>&1 </dev/null &
echo started
"""


def sandbox_token(api_token: str, microvm_id: str) -> str:
    """The sandbox server token for one MicroVM: HMAC-SHA256 of its id, keyed with the control token.

    One-way: the token inside the VM reveals nothing about the control token.
    """
    return hmac.new(api_token.encode(), _TOKEN_CONTEXT + microvm_id.encode(), hashlib.sha256).hexdigest()


def server_source() -> str:
    """Source of sandbox_server.py, as installed into a MicroVM over exec and by the image build."""
    return resources.files("langchain_digitalocean_microvm").joinpath("sandbox_server.py").read_text()


class _AttemptsExhausted(TransportError):
    """Every attempt of one request failed in a way that may succeed with a smaller request."""


class EndpointTransport(SandboxTransport):
    """SandboxTransport over the MicroVM's HTTPS endpoint and the in-VM sandbox server."""

    def __init__(
        self,
        client: MicroVMClient | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        http: httpx.Client | None = None,
        url_for: Callable[[MicroVMState], str] | None = None,
        sync_budget: float = 8.0,
        upload_chunk: int = 32 * 1024 * 1024,
        download_chunk: int = 16 * 1024 * 1024,
        min_download_chunk: int = 1024 * 1024,
        request_timeout: float = 30.0,
        probe_timeout: float = 5.0,
        max_attempts: int = 4,
        bootstrap: bool = True,
        fallback: bool = True,
        server_dir: str = DEFAULT_SERVER_DIR,
        token_file: str = sandbox_server.DEFAULT_TOKEN_FILE,
        ready_timeout: float = 30.0,
        endpoint_wait: float = DEFAULT_ENDPOINT_WAIT,
        settle_wait: float = 10.0,
        cancel_grace: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        new_key: Callable[[], str] = lambda: secrets.token_hex(16),
        source: Callable[[], str] = server_source,
    ) -> None:
        """Production code passes nothing. `client`, `http`, `url_for`, `source` and the timing/key
        hooks are test seams; `endpoint_wait` is how long a running MicroVM may take to report its
        HTTPS endpoint (normally well under a second); `settle_wait` is how long edge errors right
        after a create, clone or resume (or a server still starting) are waited out before deciding
        there is no server."""
        self._client = client
        self._owns_client = client is None
        self._base_url = base_url
        self._http = http
        self._owns_http = http is None
        self._url_for = url_for
        self._sync_budget = min(sync_budget, sandbox_server.MAX_SYNC_BUDGET)
        self._upload_chunk = upload_chunk
        self._download_chunk = download_chunk
        self._min_download_chunk = min(min_download_chunk, download_chunk)
        self._request_timeout = request_timeout
        self._probe_timeout = probe_timeout
        self._max_attempts = max(1, max_attempts)
        self._bootstrap = bootstrap
        self._fallback_allowed = fallback
        self._server_dir = server_dir
        self._token_file = token_file
        self._ready_timeout = ready_timeout
        self._endpoint_wait = endpoint_wait
        self._settle_wait = settle_wait
        self._cancel_grace = cancel_grace
        self._sleep = sleep
        self._monotonic = monotonic
        self._new_key = new_key
        self._source = source
        self._microvm_id: str | None = None
        self._url = ""
        self._port = 0
        self._token = ""
        self._fallback: ExecTransport | None = None
        self._last_probe = ""
        self._server_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ connection

    @property
    def uses_fallback(self) -> bool:
        """True when no compatible server can run and commands go through ExecTransport."""
        return self._fallback is not None

    def connect(
        self,
        *,
        host: str,
        port: int = 0,
        credentials: TransportCredentials,
        timeout: float | None = None,
    ) -> None:
        if not credentials.api_token:
            raise TransportError("EndpointTransport needs credentials.api_token (the control token, for exec)")
        if not credentials.access_token:
            raise TransportError(
                "EndpointTransport needs credentials.access_token: set DO_MICROVM_ACCESS_TOKEN to a token "
                "scoped to microvm:access only (endpoint requests never carry the control token)"
            )
        if self._client is None:
            self._client = MicroVMClient(credentials.api_token, base_url=self._base_url, max_retries=6)
        wait = min(timeout, self._endpoint_wait) if timeout else self._endpoint_wait
        try:
            state = self._client.get(host)
            if state.endpoint is None:
                state = self._client.wait_for_endpoint(host, timeout=wait)
        except MicroVMNotFoundError as exc:
            raise TransportError(f"MicroVM {host} does not exist") from exc
        except TimeoutError as exc:
            raise TransportError(
                f"{exc}: the MicroVMs API lists no HTTPS endpoint for it, which the endpoint transport needs "
                "(DO_MICROVM_TRANSPORT=exec uses the exec API instead)"
            ) from exc
        except MicroVMClientError as exc:
            raise TransportError(f"MicroVM {host} has no usable endpoint: {exc}") from exc
        self._microvm_id = host
        self._url = self._url_for(state) if self._url_for else f"https://{state.endpoint}"
        self._port = port or state.http_port or sandbox_server.DEFAULT_PORT
        self._token = sandbox_token(credentials.api_token, host)
        if self._http is None:
            self._http = httpx.Client(
                headers={"Authorization": f"Bearer {credentials.access_token}"},
                timeout=httpx.Timeout(self._request_timeout, connect=10.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
            )
        self._fallback = None
        self._ensure_server()

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
        self._microvm_id = None
        if self._owns_http and self._http is not None:
            self._http.close()
            self._http = None
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._microvm_id is not None

    def _ensure_server(self) -> None:
        # A checkpoint clone always answers 401 at first (its server holds the template's token).
        # Re-key only then: an exec sent to a fresh clone before its workload answers slows the
        # clone's start by more than the exec itself takes.
        state = self._wait_ready(self._settle_wait)
        if state == "unauthorized":
            self._rekey()
            state = self._wait_ready(self._settle_wait)
        if state == "outdated" and self._server_info.get("origin") == "file" and self._bootstrap:
            state = self._install_and_wait(replace=True)
        elif state == "unavailable" and self._bootstrap:
            state = self._install_and_wait(replace=False)
        if state == "ok":
            return
        reason = {
            "unauthorized": "the sandbox server rejects its token",
            "foreign": "another program answers on the HTTP port",
            "outdated": (f"the image's sandbox server speaks protocol {self._server_info.get('protocol')}, "
                         f"this package needs {PROTOCOL} (rebuild the image with scripts/build_sandbox_image.py)"),
        }.get(state, "no sandbox server answers")
        if not self._fallback_allowed:
            raise TransportError(f"MicroVM {self._microvm_id}: {reason} on {self._url} ({self._last_probe})")
        logger.warning("Sandbox: %s on its endpoint (%s); using the exec API instead (slower, rate limited)",
                       reason, self._last_probe)
        fallback = ExecTransport(self._client)
        fallback.connect(host=self._microvm_id or "", credentials=TransportCredentials())
        self._fallback = fallback

    def _install_and_wait(self, *, replace: bool) -> str:
        installed = self._install(replace=replace)
        if installed == "started":
            state = self._wait_ready(self._ready_timeout)
        elif installed == "port_taken":  # by something that may still be starting: maybe our own server
            state = self._wait_ready(self._settle_wait)
        else:
            return "unavailable"
        if state == "unauthorized":
            self._rekey()
            state = self._wait_ready(self._settle_wait)
        return state

    def _probe(self) -> str:
        """What answers on the endpoint: 'ok' (our server), 'outdated' (our server, other protocol),
        'unauthorized' (our server, wrong token), 'foreign' (another program on the HTTP port) or
        'unavailable' (nothing yet: edge errors, connection failures)."""
        try:
            resp = self._send("GET", "/v1/info", timeout=self._probe_timeout)
        except httpx.HTTPError as exc:
            self._last_probe = type(exc).__name__
            return "unavailable"
        self._last_probe = f"HTTP {resp.status_code}" + ("" if _from_workload(resp) else " from the edge")
        info = _json(resp)
        if resp.status_code == 200 and info.get("server") == sandbox_server.SERVER_NAME:
            self._server_info = info
            return "ok" if info.get("protocol") == PROTOCOL else "outdated"
        if resp.status_code == 401 and info.get("source") == sandbox_server.SERVER_NAME:
            return "unauthorized"
        if not _from_workload(resp):
            if resp.status_code == 401:
                raise TransportError(f"the MicroVM endpoint rejected the access token (HTTP 401 from {self._url})")
            return "unavailable"
        return "foreign"

    def _wait_ready(self, timeout: float) -> str:
        deadline = self._monotonic() + timeout
        while True:
            state = self._probe()
            if state != "unavailable" or self._monotonic() >= deadline:
                return state
            self._sleep(0.5)

    def _rekey(self) -> None:
        result = self._control_exec(["sh", "-c", _REKEY_SCRIPT, "_", self._token_file, self._token])
        if result.exit_code != 0:
            raise TransportError(f"could not write the sandbox token in MicroVM {self._microvm_id}: "
                                 f"exit {result.exit_code}: {result.stderr.strip()[:200]}")

    def _install(self, *, replace: bool) -> str:
        """Install (or replace) and start the server over exec: 'started', 'port_taken' or 'failed'."""
        source = base64.b64encode(self._source().encode()).decode()
        result = self._control_exec(["sh", "-c", _INSTALL_SCRIPT, "_", self._server_dir, self._token_file,
                                     source, self._token, str(self._port), "replace" if replace else ""])
        if result.exit_code == 98:
            logger.info("Sandbox: port %s is taken, not starting the sandbox server", self._port)
            return "port_taken"
        if result.exit_code == 127:
            logger.warning("Sandbox: the image has no python3, so the sandbox server cannot be installed")
            return "failed"
        if result.exit_code != 0:
            logger.warning("Sandbox: installing the sandbox server failed (exit %s): %s",
                           result.exit_code, result.stderr.strip()[:200])
            return "failed"
        return "started"

    def _control_exec(self, argv: list[str]) -> ExecResult:
        assert self._client is not None and self._microvm_id is not None
        try:
            return self._client.exec(self._microvm_id, argv)
        except MicroVMClientError as exc:
            raise TransportError(f"exec in MicroVM {self._microvm_id} failed: {exc}") from exc

    # ------------------------------------------------------------------ HTTP

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if self._http is None or self._microvm_id is None:
            raise TransportError("EndpointTransport is not connected")
        headers = {"x-api-key": self._token, **kwargs.pop("headers", {})}
        return self._http.request(method, f"{self._url}{path}", headers=headers, **kwargs)

    def _request(self, method: str, path: str, *, key: str | None = None, attempts: int | None = None,
                 **kwargs: Any) -> httpx.Response:
        """One logical request: retried (same idempotency key) on connection errors, cut bodies and edge 5xx.

        Raises _AttemptsExhausted when every attempt failed that way, TransportError for an answer
        that no retry can fix (the MicroVM is gone, the access token is rejected).
        """
        headers = dict(kwargs.pop("headers", {}))
        if key:
            headers["x-idempotency-key"] = key
        rekeyed = False
        last = ""
        tries = attempts or self._max_attempts
        for attempt in range(tries):
            if attempt:
                self._sleep(min(2.0 ** (attempt - 1), 8.0))
            try:
                resp = self._send(method, path, headers=headers, **kwargs)
            except httpx.TransportError as exc:  # connection refused/reset, timeout, body cut short
                last = f"{type(exc).__name__}: {exc}"
                continue
            if resp.status_code in _RETRY_STATUSES and not _ours(resp):
                last = f"HTTP {resp.status_code} from the edge"
                continue
            if resp.status_code == 401 and _ours(resp) and not rekeyed:
                rekeyed = True
                self._rekey()
                continue
            if not _from_workload(resp) and not _ours(resp):
                if resp.status_code == 404:
                    raise TransportError(f"MicroVM {self._microvm_id} is gone (its endpoint answers 404)")
                if resp.status_code in (401, 403):
                    raise TransportError(f"the MicroVM endpoint rejected the access token (HTTP {resp.status_code})")
            return resp
        raise _AttemptsExhausted(f"{method} {path} on MicroVM {self._microvm_id} failed after {tries} attempts: {last}")

    def _read(self, path: str, params: dict[str, Any] | None, *, limit: int | None, what: str) -> tuple[bytes, int]:
        """All of a file-like resource in Range pages (or its first `limit` bytes), and its total size.

        A page that does not arrive complete is asked for again at half the size: the page size is
        kept for later reads, since a slow link stays slow.
        """
        data = bytearray()
        total: int | None = None
        while total is None or len(data) < (total if limit is None else min(total, limit)):
            end = len(data) + self._download_chunk - 1
            if limit is not None:
                end = min(end, limit - 1)
            try:
                resp = self._request("GET", path, params=params, headers={"Range": f"bytes={len(data)}-{end}"},
                                     attempts=2)
            except _AttemptsExhausted:
                if self._download_chunk <= self._min_download_chunk:
                    raise
                self._download_chunk = max(self._min_download_chunk, self._download_chunk // 2)
                logger.info("Sandbox: reading in %d-byte pages (a page did not arrive in time)",
                            self._download_chunk)
                continue
            if resp.status_code == 200:  # small or empty: all of it
                content = resp.content
                return (content if limit is None else content[:limit]), len(content)
            if resp.status_code != 206:
                _raise_file_error(resp, what, "read")
            total = int(resp.headers.get("x-file-size") or resp.headers["content-range"].rsplit("/", 1)[1])
            if not resp.content:
                raise TransportError(f"cannot read {what}: empty range response at offset {len(data)}")
            data += resp.content
        return bytes(data), total

    # ------------------------------------------------------------------ commands

    def run(self, command: str, *, timeout: float | None = None) -> TransportExecResult:
        if self._fallback is not None:
            return self._fallback.run(command, timeout=timeout)
        started = self._monotonic()
        body: dict[str, Any] = {"command": command, "budget": self._sync_budget}
        if timeout is not None:
            body["timeout"] = timeout
        data = self._ok_json(self._request("POST", "/v1/exec", key=self._new_key(), json=body), "exec")
        while data.get("status") == "running":
            job = str(data["job_id"])
            if timeout is not None and self._monotonic() - started > timeout + self._cancel_grace:
                # the server enforces `timeout` itself; this only covers a server that stopped doing so
                data = self._ok_json(self._request("POST", f"/v1/jobs/{job}/cancel"), "cancel")
                data["timed_out"] = True
                break
            data = self._ok_json(self._request("GET", f"/v1/jobs/{job}", params={"wait": self._sync_budget}), "job")
        return self._result(data, timeout)

    def _result(self, data: dict[str, Any], timeout: float | None) -> TransportExecResult:
        sizes = int(data.get("stdout_bytes") or 0), int(data.get("stderr_bytes") or 0)
        truncated = sum(sizes) > MAX_OUTPUT
        if "stdout" in data:  # small enough to come back inline
            stdout, stderr = str(data.get("stdout") or ""), str(data.get("stderr") or "")
        else:
            job = data["job_id"]
            out, _ = self._read(f"/v1/jobs/{job}/stdout", None, limit=MAX_OUTPUT, what="the command's stdout")
            err = b""
            if MAX_OUTPUT - len(out) > 0 and sizes[1]:
                err, _ = self._read(f"/v1/jobs/{job}/stderr", None, limit=MAX_OUTPUT - len(out),
                                    what="the command's stderr")
            stdout, stderr = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        notes = []
        if truncated:
            notes.append(_TRUNCATED_NOTE)
        if data.get("timed_out"):
            notes.append(f"[timed out after {timeout:g}s]" if timeout is not None else "[timed out]")
        if notes:
            stderr = "\n".join([stderr, *notes]) if stderr else "\n".join(notes)
        code = TIMEOUT_EXIT_CODE if data.get("timed_out") else data.get("exit_code")
        return TransportExecResult(stdout=stdout, stderr=stderr, exit_code=int(code) if code is not None else None)

    # ------------------------------------------------------------------ files

    def put_file(self, path: str, content: bytes) -> None:
        if self._fallback is not None:
            return self._fallback.put_file(path, content)
        size = self._upload_chunk
        chunks = [content[i : i + size] for i in range(0, len(content), size)] or [b""]
        for index, chunk in enumerate(chunks):
            resp = self._request(
                "PUT", "/v1/files", params={"path": path, "offset": index * size}, content=chunk,
                headers={"X-Content-Sha256": hashlib.sha256(chunk).hexdigest(),
                         "Content-Type": "application/octet-stream"},
            )
            if resp.status_code != 200:
                _raise_file_error(resp, path, "write")
        return None

    def get_file(self, path: str) -> bytes:
        if self._fallback is not None:
            return self._fallback.get_file(path)
        data, _total = self._read("/v1/files", {"path": path}, limit=None, what=path)
        return data

    # ------------------------------------------------------------------ internals

    def _ok_json(self, resp: httpx.Response, what: str) -> dict[str, Any]:
        data = _json(resp)
        if resp.status_code != 200 or not data:
            raise TransportError(f"{what} in MicroVM {self._microvm_id} failed: HTTP {resp.status_code} "
                                 f"{data.get('error') or resp.text[:200]!r}")
        return data


def _json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _ours(resp: httpx.Response) -> bool:
    """An error answer of the sandbox server itself."""
    return resp.headers.get("content-type", "").startswith("application/json") and \
        _json(resp).get("source") == sandbox_server.SERVER_NAME


def _from_workload(resp: httpx.Response) -> bool:
    """Did the MicroVM's workload produce this response (as opposed to DigitalOcean's edge)?"""
    return _UPSTREAM_HEADER in resp.headers or resp.headers.get("server", "").startswith(sandbox_server.SERVER_NAME)


def _raise_file_error(resp: httpx.Response, path: str, verb: str) -> None:
    kind = _json(resp).get("error")
    if kind == "not_found" or resp.status_code == 404:
        raise TransportFileNotFoundError(f"cannot {verb} {path}: no such file or directory")
    if kind == "permission_denied" or resp.status_code == 403:
        raise TransportPermissionError(f"cannot {verb} {path}: permission denied")
    if kind == "is_directory" or resp.status_code == 409:
        raise TransportIsADirectoryError(f"cannot {verb} {path}: is a directory")
    raise TransportError(f"cannot {verb} {path}: HTTP {resp.status_code} {kind or resp.text[:200]!r}")
