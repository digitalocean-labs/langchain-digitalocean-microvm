# images/agent-host/artifacts_app.py
"""Artifact routes, served by the Agent Server next to the Agent Protocol.

Registered in langgraph.json as the server's custom app:

    "http": {"app": "./artifacts_app.py:app", "enable_custom_route_auth": true}

A host's endpoint has one HTTPS port and the Agent Server holds it, so the
gateway reads a task's files through these two routes on that same server
(langchain_digitalocean_microvm.agent_host.artifacts is the client):

    GET /microvm/artifacts/{tid}?limit=N          regular files under the task's
                                                  outputs/ (relative path, size),
                                                  sorted, at most N
    GET /microvm/artifacts/{tid}/files/{path}     one file; `Range: bytes=a-b`
                                                  answers 206 with that slice

A response that does not arrive within the endpoint's time limit is asked for
again by the client in a smaller range, so a file of any allowed size gets
through a slow link.

Every request must carry the host token in `x-api-key`. auth.py checks it
(enable_custom_route_auth) and this app checks it again, so the routes stay
closed even if the Agent Server stops applying auth to custom routes.
Every answer of this app carries `x-microvm-artifacts`, so the client can tell
it from an Agent Server without these routes (an older image) or the edge.

The subagent controls everything under its task directory and can read the
host token and its model key, so nothing here follows a symlink: every path
component below SUBAGENT_WORKDIR is opened with O_NOFOLLOW relative to its
parent's descriptor (a component swapped for a symlink fails the open instead
of being followed), a listing skips everything that is not a regular file and
does not descend into symlinked directories, and a read opens the file with
O_NOFOLLOW | O_NONBLOCK (a FIFO cannot block it) and checks fstat.
"""

from __future__ import annotations

import errno
import hmac
import importlib.util
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

ROUTE_PREFIX = "/microvm/artifacts"
MARKER_HEADER = "x-microvm-artifacts"
TASKS_DIR = "tasks"  # must match agent.py
OUTPUTS_DIR = "outputs"
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000
DEFAULT_WORKDIR = "/work"
_SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RANGE = re.compile(r"^bytes=(\d+)-(\d*)$")
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
# A symlink met O_NOFOLLOW: ELOOP on Linux and macOS, EMLINK on FreeBSD. ENOTDIR (a file where a
# directory belongs, or a symlink opened with O_DIRECTORY on some systems) is refused the same way.
_SYMLINK_ERRNOS = {errno.ELOOP, getattr(errno, "EMLINK", -1)}


class _Refused(Exception):
    def __init__(self, status: int, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.reason, self.detail = status, reason, detail


def _load_auth_token() -> Callable[[], Awaitable[str]]:
    """auth.py's expected_token, loaded from next to this file (the server may not put it on sys.path)."""
    spec = importlib.util.spec_from_file_location("microvm_agent_host_auth", Path(__file__).with_name("auth.py"))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.expected_token


def _respond(status: int, body: bytes | dict[str, Any], headers: dict[str, str] | None = None) -> Response:
    if isinstance(body, dict):
        return Response(json.dumps(body), status_code=status, media_type="application/json",
                        headers={MARKER_HEADER: "1", **(headers or {})})
    return Response(body, status_code=status, media_type="application/octet-stream",
                    headers={MARKER_HEADER: "1", **(headers or {})})


def _error(exc: _Refused) -> Response:
    return _respond(exc.status, {"error": exc.reason, "detail": exc.detail})


@contextmanager
def _open_dir(parent: int | None, name: str, *, what: str):  # type: ignore[no-untyped-def]
    """Open directory `name` under the descriptor `parent` without following a symlink."""
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=parent) if parent is not None else os.open(name, _DIR_FLAGS)
    except FileNotFoundError:
        raise _Refused(404, "not_found", f"{what}: no such directory") from None
    except OSError as exc:
        if exc.errno in _SYMLINK_ERRNOS or exc.errno == errno.ENOTDIR:
            raise _Refused(403, "symlink", f"{what} is a symlink or not a directory") from None
        raise _Refused(403, "unreadable", f"{what}: {exc.strerror}") from None
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _outputs_fd(workdir: str, thread_id: str):  # type: ignore[no-untyped-def]
    """A descriptor of <workdir>/tasks/<thread_id>/outputs, each component opened without following symlinks."""
    if not _SAFE_THREAD_ID.match(thread_id):
        raise _Refused(400, "bad_path", f"not a valid task id: {thread_id!r}")
    with _open_dir(None, workdir, what="the work directory") as work, \
            _open_dir(work, TASKS_DIR, what="the tasks directory") as tasks, \
            _open_dir(tasks, thread_id, what="the task directory") as task, \
            _open_dir(task, OUTPUTS_DIR, what="the outputs directory") as outputs:
        yield outputs


def _relative_parts(path: str) -> list[str]:
    parts = [p for p in path.split("/") if p]
    if not parts or any(p in (".", "..") or "\x00" in p for p in parts):
        raise _Refused(400, "bad_path", f"not a path inside the outputs directory: {path!r}")
    return parts


def list_outputs(workdir: str, thread_id: str, limit: int) -> dict[str, Any]:
    """{"exists", "files": [{"path", "size"}], "truncated"}: regular files only, sorted by path."""
    files: list[dict[str, Any]] = []
    truncated = False
    try:
        with _outputs_fd(workdir, thread_id) as outputs:
            # fwalk(follow_symlinks=False) lists a symlinked directory but never descends into it
            for dirpath, dirnames, filenames, dirfd in os.fwalk(".", dir_fd=outputs, follow_symlinks=False):
                dirnames.sort()
                for name in sorted(filenames):
                    try:
                        info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                    except OSError:
                        continue  # removed while listing
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    if len(files) >= limit:
                        truncated = True
                        break
                    rel = os.path.normpath(os.path.join(dirpath, name))
                    files.append({"path": rel, "size": info.st_size})
                if truncated:
                    break
    except _Refused as exc:
        if exc.reason == "not_found":
            return {"exists": False, "files": [], "truncated": False}
        raise
    files.sort(key=lambda f: f["path"])
    return {"exists": True, "files": files, "truncated": truncated}


def read_output(workdir: str, thread_id: str, path: str, byte_range: tuple[int, int | None] | None
                ) -> tuple[bytes, int, int]:
    """(data, start, file size) of one regular file under the task's outputs/, or of a slice of it."""
    parts = _relative_parts(path)
    with _outputs_fd(workdir, thread_id) as outputs:
        fds: list[int] = []
        try:
            parent = outputs
            for part in parts[:-1]:
                try:
                    parent = os.open(part, _DIR_FLAGS, dir_fd=parent)
                except FileNotFoundError:
                    raise _Refused(404, "not_found", f"{path}: no such file") from None
                except OSError as exc:
                    if exc.errno in _SYMLINK_ERRNOS or exc.errno == errno.ENOTDIR:
                        raise _Refused(403, "symlink", f"{path}: symlink or non-directory in the path") from None
                    raise _Refused(403, "unreadable", f"{path}: {exc.strerror}") from None
                fds.append(parent)
            try:
                fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                raise _Refused(404, "not_found", f"{path}: no such file") from None
            except OSError as exc:
                if exc.errno in _SYMLINK_ERRNOS:
                    raise _Refused(403, "symlink", f"{path} is a symlink") from None
                raise _Refused(403, "unreadable", f"{path}: {exc.strerror}") from None
        finally:
            for opened in fds:
                os.close(opened)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _Refused(403, "not_a_file", f"{path} is not a regular file")
        size = info.st_size
        if byte_range is None:
            start, end = 0, size - 1
        else:
            start, last = byte_range
            if size and start >= size:
                raise _Refused(416, "bad_range", f"range starts at {start}, the file has {size} bytes")
            end = size - 1 if last is None else min(last, size - 1)
        length = max(0, end - start + 1)
        chunks = bytearray()
        while len(chunks) < length:
            piece = os.pread(fd, length - len(chunks), start + len(chunks))
            if not piece:
                break  # the file shrank while it was read
            chunks += piece
        return bytes(chunks), start, size
    finally:
        os.close(fd)


def create_app(*, workdir: str | None = None,
               expected_token: Callable[[], Awaitable[str]] | None = None) -> Starlette:
    """The routes as a Starlette app. `workdir` defaults to $SUBAGENT_WORKDIR (read per request);
    `expected_token` to auth.py's (token file, then $SUBAGENT_HOST_TOKEN). Tests pass both."""
    token_source = expected_token or _load_auth_token()

    def root() -> str:
        return workdir or os.environ.get("SUBAGENT_WORKDIR") or DEFAULT_WORKDIR

    async def authorized(request: Request) -> bool:
        expected = await token_source()
        presented = request.headers.get("x-api-key", "")
        return bool(expected) and hmac.compare_digest(presented.encode(), expected.encode())

    async def list_route(request: Request) -> Response:
        if not await authorized(request):
            return _respond(401, {"error": "unauthorized", "detail": "invalid or missing host token"})
        try:
            limit = min(MAX_LIMIT, max(1, int(request.query_params.get("limit", DEFAULT_LIMIT))))
        except ValueError:
            return _respond(400, {"error": "bad_request", "detail": "limit must be an integer"})
        try:
            listing = await run_in_threadpool(list_outputs, root(), request.path_params["tid"], limit)
        except _Refused as exc:
            return _error(exc)
        return _respond(200, listing)

    async def file_route(request: Request) -> Response:
        if not await authorized(request):
            return _respond(401, {"error": "unauthorized", "detail": "invalid or missing host token"})
        header = request.headers.get("range")
        byte_range: tuple[int, int | None] | None = None
        if header:
            match = _RANGE.match(header.strip())
            if not match or (match.group(2) and int(match.group(2)) < int(match.group(1))):
                return _respond(416, {"error": "bad_range", "detail": f"unsupported Range {header!r}"})
            byte_range = (int(match.group(1)), int(match.group(2)) if match.group(2) else None)
        try:
            data, start, size = await run_in_threadpool(
                read_output, root(), request.path_params["tid"], request.path_params["path"], byte_range)
        except _Refused as exc:
            return _error(exc)
        headers = {"x-file-size": str(size), "accept-ranges": "bytes"}
        if byte_range is None or size == 0:
            return _respond(200, data, headers)
        headers["content-range"] = f"bytes {start}-{start + len(data) - 1}/{size}"
        return _respond(206, data, headers)

    return Starlette(routes=[
        Route(ROUTE_PREFIX + "/{tid}", list_route, methods=["GET"]),
        Route(ROUTE_PREFIX + "/{tid}/files/{path:path}", file_route, methods=["GET"]),
    ])


app = create_app()
