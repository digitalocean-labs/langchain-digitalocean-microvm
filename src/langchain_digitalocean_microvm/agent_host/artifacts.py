"""Artifacts: files a subagent left in its host's per-task outputs directory.

The subagent template (images/agent-host/agent.py) gives every
task its own directory, keyed by the run's thread id, which is also the
supervisor's task id:

    <workdir>/tasks/<thread_id>/outputs/

`<workdir>` is SUBAGENT_WORKDIR inside the host (/work in the image).

The host's Agent Server serves that directory next to the Agent Protocol
(images/agent-host/artifacts_app.py), on the host's own endpoint and with the
same credentials. This module is the client the gateway uses:

- `list_artifacts()`: regular files under the directory, relative paths and
  sizes, capped at `max_files` (one request).
- `read_artifact()`: one file in `Range` pages. A page that does not arrive
  (the endpoint gives a response about 15 s) is asked for again at half the
  size, down to `min_page_bytes`.

The server never follows symlinks (the subagent controls the directory, and
its environment holds the host token and a model key): see artifacts_app.py.
Every answer of the server carries `MARKER_HEADER`; a response without it came
from something else (DigitalOcean's edge for a deleted host, or an Agent
Server built from an image without these routes) and is reported as
`ArtifactError(reason="not_served")` with its status, for the caller to tell
those apart.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote

import httpx

from langchain_digitalocean_microvm.exceptions import SandboxError

TASKS_DIR = "tasks"  # must match images/agent-host/agent.py
OUTPUTS_DIR = "outputs"
ROUTE_PREFIX = "/microvm/artifacts"  # must match images/agent-host/artifacts_app.py
MARKER_HEADER = "x-microvm-artifacts"
DEFAULT_MAX_FILES = 200
DEFAULT_PAGE_BYTES = 4 * 1024 * 1024
MIN_PAGE_BYTES = 512 * 1024
_SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

# GET `path` (with these extra headers) on the host; raises ArtifactUpstreamError when the host
# does not answer (connection error, timeout, 5xx).
HostGet = Callable[[str, Mapping[str, str]], Awaitable[httpx.Response]]


class ArtifactError(SandboxError):
    """Reading an artifact failed; `reason` is machine-readable, `status` the host's HTTP status."""

    def __init__(self, reason: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


class ArtifactUpstreamError(SandboxError):
    """The host did not answer an artifact request (connection error, timeout, 5xx)."""


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    path: str  # relative to the task's outputs directory, "/"-separated
    size: int


@dataclass(frozen=True, slots=True)
class ArtifactListing:
    exists: bool  # False: the task never created its outputs directory (or it is gone)
    files: tuple[ArtifactInfo, ...]
    truncated: bool  # more than max_files regular files


def outputs_dir(workdir: str, thread_id: str) -> str:
    """`<workdir>/tasks/<thread_id>/outputs`."""
    _check_thread_id(thread_id)
    return str(PurePosixPath(workdir) / TASKS_DIR / thread_id / OUTPUTS_DIR)


def safe_relative_path(path: str) -> str:
    """Normalize a caller-supplied artifact path; reject anything that could leave the directory."""
    cleaned = path.strip().removeprefix(f"/{OUTPUTS_DIR}/").lstrip("/")
    parts = PurePosixPath(cleaned).parts
    if not parts or any(part in ("..", ".") for part in parts) or "\x00" in cleaned:
        raise ArtifactError("bad_path", f"not a path inside the outputs directory: {path!r}")
    return str(PurePosixPath(*parts))


async def list_artifacts(get: HostGet, thread_id: str, *, max_files: int = DEFAULT_MAX_FILES) -> ArtifactListing:
    """List the regular files a task left under its outputs/ (one request)."""
    _check_thread_id(thread_id)
    response = await get(f"{ROUTE_PREFIX}/{thread_id}?limit={max_files}", {})
    _raise_for_error(response, thread_id)
    body = response.json()
    files = tuple(sorted((ArtifactInfo(path=str(f["path"]), size=int(f["size"])) for f in body.get("files", [])),
                         key=lambda f: f.path))
    return ArtifactListing(exists=bool(body.get("exists")), files=files, truncated=bool(body.get("truncated")))


async def read_artifact(
    get: HostGet,
    thread_id: str,
    path: str,
    *,
    max_bytes: int,
    page_bytes: int = DEFAULT_PAGE_BYTES,
    min_page_bytes: int = MIN_PAGE_BYTES,
) -> bytes:
    """Read one file in Range pages. ArtifactError(reason="too_large") past `max_bytes`."""
    _check_thread_id(thread_id)
    rel = safe_relative_path(path)
    url = f"{ROUTE_PREFIX}/{thread_id}/files/{quote(rel, safe='/')}"
    page = max(1, page_bytes)
    data = bytearray()
    size: int | None = None
    while size is None or len(data) < size:
        try:
            response = await get(url, {"Range": f"bytes={len(data)}-{len(data) + page - 1}"})
        except ArtifactUpstreamError:
            if page <= min_page_bytes:
                raise
            page = max(min_page_bytes, page // 2)
            continue
        if response.status_code == 416 and size is not None:
            break  # the file shrank while it was read
        _raise_for_error(response, rel)
        total = int(response.headers.get("x-file-size", len(response.content)))
        if size is None:
            size = total
            if size > max_bytes:
                raise ArtifactError("too_large", f"{rel} is {size} bytes; the limit is {max_bytes}", status=413)
        if response.status_code == 200:  # empty file (or a server that ignores Range): all of it
            return response.content[:size]
        if not response.content:
            break  # the file shrank while it was read
        data += response.content
    return bytes(data[: size or 0])


def _check_thread_id(thread_id: str) -> None:
    if not _SAFE_THREAD_ID.match(thread_id):
        raise ValueError(f"not a valid task id: {thread_id!r}")


def _raise_for_error(response: httpx.Response, what: str) -> None:
    if MARKER_HEADER not in response.headers:
        raise ArtifactError("not_served", f"{what}: HTTP {response.status_code} without the artifact routes' marker",
                            status=response.status_code)
    if response.status_code < 400:
        return
    try:
        body = response.json()
    except ValueError:
        body = {}
    reason = str(body.get("error") or "unreadable")
    raise ArtifactError(reason, f"{what}: {body.get('detail') or reason}", status=response.status_code)
