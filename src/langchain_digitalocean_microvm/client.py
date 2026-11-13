"""REST client for the DigitalOcean MicroVMs API.

Source of truth: the MicroVMs documentation (API Reference,
"Endpoints"), corrected by what the API actually returns when probed
(scripts/probe_api.py; recorded responses live in
tests/fixtures/microvm_api/). Every DigitalOcean-specific detail -- paths,
field names, response envelopes, status codes -- stays in this file.

Facts the rest of the package relies on:

- Base path /v2/microvms; envelopes `microvm`, `microvms`, `checkpoint`,
  `checkpoints`; list endpoints paginate (`page`, `per_page` <= 200,
  `links.pages.next`).
- Everything is set at create time (image, size, env, tags, ports, idle
  timeout, auto-resume); there is no update endpoint.
- POST .../pause and .../resume return when the change completes and are
  idempotent, but their response body carries `"tags": []` even for a
  tagged MicroVM, so callers get a fresh GET instead. That GET can still
  report the old state (`running`, not `pausing`) for a few seconds (for
  example with two pauses in parallel), so pause/resume poll briefly for
  the state they asked for.
- DELETE returns 204, after which GET still answers with `state:
  terminating` for a while.
- POST .../exec runs argv directly (no shell) inside the workload container:
  60 s limit, 4 MiB output, 1 MiB request body, 60 commands/min and 32
  concurrent per team (HTTP 429). It auto-resumes a paused MicroVM.
- Names are 1-63 chars of [a-z0-9-], unique per team and region; `?name=` is an exact
  match, which is how this package looks MicroVMs up (see microvm_name()).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Any

import httpx

from langchain_digitalocean_microvm.exceptions import (
    MicroVMAlreadyExistsError,
    MicroVMAPIError,
    MicroVMClientError,
    MicroVMForbiddenError,
    MicroVMNotFoundError,
    MicroVMRateLimitError,
)
from langchain_digitalocean_microvm.models import (
    Checkpoint,
    ExecResult,
    MicroVMSize,
    MicroVMState,
)
from langchain_digitalocean_microvm.owner import OWNER_TAG_PREFIX, owner_tag
from langchain_digitalocean_microvm.status import (
    TRANSITIONAL_STATUSES,
    MicroVMStatus,
    parse_status,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.digitalocean.com/v2"

# Every MicroVM this client creates carries MANAGED_TAG (bulk scans, cleanup
# scripts). Exported so provider.py and cleanup tooling use the same value.
MANAGED_TAG = "microvm-sandbox"
# Subagent sandboxes: a sandbox owned by one subagent of one thread.
# Lets gc_microvms.py expire those without touching other sandboxes.
SUBAGENT_SANDBOX_TAG = "microvm-subagent-sandbox"
# Thread sandboxes (threads.py): one sandbox per conversation thread, expired the same way.
THREAD_SANDBOX_TAG = "microvm-thread-sandbox"
# A per-task subagent sandbox (subagents.PerTask), recorded next to SUBAGENT_SANDBOX_TAG.
PER_TASK_SANDBOX_TAG = "microvm-per-task-sandbox"
_MAX_NAME = 63
_NAME_UNSAFE = re.compile(r"[^a-z0-9-]+")
_MAX_PER_PAGE = 200
_RETRYABLE = {429, 502, 503, 504}
# After pause/resume: how long a stale `running`/`paused` read is waited out, and how often re-read.
_SETTLE_SECONDS = 15.0
_SETTLE_POLL = 1.0
_REPAUSE_AFTER = 3.0  # pause: seconds of still reading `running` before the pause is sent again


_ID_TAG_PREFIX = "microvm-sandbox-id:"

# Every HTTP request any MicroVMClient in this process sends, by operation
# ("exec", "create", "get", "pause", ...; retries also count as
# "retry:<status>"). Process-wide because each ExecTransport owns its own
# client; exec requests are what the team's 60/min budget counts.
_CALLS: Counter[str] = Counter()
_CALLS_LOCK = threading.Lock()


def _operation(method: str, path: str) -> str:
    """The counter key for a request to `path` (relative to /v2, e.g. /microvms/<id>/exec)."""
    parts = [p for p in path.split("/") if p][1:]  # drop "microvms"
    if not parts:
        return "list" if method == "GET" else "create"
    if parts[0] == "options":
        return "options"
    if parts[0] == "checkpoints":
        if method == "GET":
            return "get checkpoint" if len(parts) > 1 else "list checkpoints"
        return f"{method.lower()} checkpoint"
    if len(parts) > 1:  # exec, pause, resume, checkpoints (create)
        return "create checkpoint" if parts[1] == "checkpoints" else parts[1]
    return {"GET": "get", "DELETE": "delete"}.get(method, method.lower())


def api_calls() -> dict[str, int]:
    """Snapshot of the requests sent to the MicroVMs API by this process, by operation."""
    with _CALLS_LOCK:
        return dict(_CALLS)


def _count(key: str) -> None:
    with _CALLS_LOCK:
        _CALLS[key] += 1


def id_tag(sandbox_id: str) -> str:
    """Informational tag recording which sandbox_id a MicroVM backs (lookups go by name)."""
    return f"{_ID_TAG_PREFIX}{sandbox_id}"


def sandbox_id_from_tags(tags: tuple[str, ...]) -> str | None:
    """The sandbox_id recorded by `id_tag`, or None (agent hosts and foreign MicroVMs have none)."""
    for tag in tags:
        if tag.startswith(_ID_TAG_PREFIX):
            return tag[len(_ID_TAG_PREFIX):]
    return None


def microvm_name(prefix: str, key: str) -> str:
    """Deterministic, API-valid MicroVM name for `key` (a sandbox_id or host_id).

    Names must be 1-63 lowercase letters, digits and hyphens. A key that is
    already safe and short maps to `<prefix>-<key>`; anything else is
    sanitized and suffixed with a short hash of the original so two
    different keys can never collide on the same name.
    """
    base = f"{prefix}-{key}"
    safe = _NAME_UNSAFE.sub("-", base.lower()).strip("-")
    safe = re.sub(r"-{2,}", "-", safe)
    if safe == base and len(safe) <= _MAX_NAME:
        return safe
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    head = safe[: _MAX_NAME - len(digest) - 1].rstrip("-") or "mv"
    return f"{head}-{digest}"


def _parse_time(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).replace("Z", "+00:00")
    # The API returns up to 9 fractional digits (nanoseconds); Python accepts 6.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Unparseable timestamp %r from the MicroVMs API", raw)
        return None


def _parse_size(raw: Any) -> MicroVMSize | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return MicroVMSize(cpu=int(raw["cpu"]), memory=int(raw["memory"]), disk=raw.get("disk"))
    except (KeyError, TypeError, ValueError):
        return None


def _parse_microvm(payload: Mapping[str, Any]) -> MicroVMState:
    urls = payload.get("urls") or []
    default = next((u for u in urls if isinstance(u, Mapping) and u.get("default")), urls[0] if urls else None)
    endpoint = None
    http_port = None
    if isinstance(default, Mapping):
        http_port = default.get("port")
        if default.get("status") == "ACTIVE" and default.get("hostname"):
            endpoint = str(default["hostname"])
    return MicroVMState(
        id=str(payload["id"]),
        name=str(payload.get("name", "")),
        status=parse_status(str(payload.get("state", ""))),
        size=_parse_size(payload.get("size")),
        region=payload.get("region"),
        endpoint=endpoint,
        http_port=int(http_port) if http_port is not None else None,
        created_at=_parse_time(payload.get("created_at")),
        tags=tuple(payload.get("tags") or ()),
        failure_reason=payload.get("failure_reason"),
        raw=dict(payload),
    )


def _parse_checkpoint(payload: Mapping[str, Any]) -> Checkpoint:
    return Checkpoint(
        id=str(payload["id"]),
        name=str(payload.get("name", "")),
        microvm_id=payload.get("microvm_id"),
        region=payload.get("region"),
        size=_parse_size(payload.get("size")),
        status=str(payload.get("status", "")),
        created_at=_parse_time(payload.get("created_at")),
        raw=dict(payload),
    )


def _error_message(response: httpx.Response) -> str:
    """Prefer the API's own {"id": ..., "message": ...} error body over the raw text."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict) and "message" in body:
        return str(body["message"])
    return response.text[:500]


class MicroVMClient:
    """Synchronous client for /v2/microvms.

    Sync-only on purpose: async callers in this package wrap it with
    `asyncio.to_thread`. Retries 429 and 502/503/504 with exponential
    backoff (honouring Retry-After), `max_retries` times.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 90.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
        backoff: float = 1.0,
        sleep: Any = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=transport,
        )
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ plumbing

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        operation = _operation(method, path)
        while True:
            _count(operation)
            response = self._client.request(method, path, **kwargs)
            if response.status_code not in _RETRYABLE or attempt >= self._max_retries:
                break
            _count(f"retry:{response.status_code}")
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else self._backoff * 2**attempt
            logger.info("MicroVMs API %s -> %s; retrying in %.1fs", operation, response.status_code, delay)
            self._sleep(delay)
            attempt += 1
        self._raise_for_status(response)
        return response

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        message = _error_message(response)
        status = response.status_code
        if status == 404:
            raise MicroVMNotFoundError(message)
        if status == 409:
            raise MicroVMAlreadyExistsError(message)
        if status == 403:
            raise MicroVMForbiddenError(
                status,
                f"{message} (missing microvm:* scope, or the team's MicroVM limit is reached; "
                "see GET /v2/microvms/options)",
            )
        if status == 429:
            raise MicroVMRateLimitError(status, message)
        raise MicroVMAPIError(status, message)

    @staticmethod
    def _unwrap(response: httpx.Response, key: str) -> Any:
        body = response.json()
        try:
            return body[key]
        except (KeyError, TypeError) as exc:
            raise MicroVMClientError(f"Unexpected MicroVMs API response: missing {key!r} in {body!r}") from exc

    def _paginate(self, path: str, key: str, params: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        page = 1
        while True:
            response = self._request("GET", path, params={**params, "page": page, "per_page": _MAX_PER_PAGE})
            body = response.json()
            yield from body.get(key) or []
            if not ((body.get("links") or {}).get("pages") or {}).get("next"):
                return
            page += 1

    # ------------------------------------------------------------------ MicroVMs

    def create(
        self,
        *,
        name: str,
        oci_ref: str | None = None,
        checkpoint_id: str | None = None,
        region: str | None = None,
        size: MicroVMSize | None = None,
        http_port: int | None = None,
        http_protocol: str | None = None,
        environment: Mapping[str, str] | None = None,
        tags: list[str] | None = None,
        idle_timeout: str | None = None,
        auto_resume: bool | None = None,
        networking: str | None = None,
        vpc_uuid: str | None = None,
    ) -> MicroVMState:
        """POST /microvms from an image (`oci_ref`, needs region+size) or a `checkpoint_id`.

        Always adds MANAGED_TAG and the owner tag (owner.py). From a checkpoint, region, size and
        environment are inherited and must not be sent (the API rejects
        `environment`).
        """
        if (oci_ref is None) == (checkpoint_id is None):
            raise ValueError("pass exactly one of oci_ref or checkpoint_id")
        body_tags = [MANAGED_TAG]
        for tag in tags or []:
            if tag not in body_tags:
                body_tags.append(tag)
        if not any(tag.startswith(OWNER_TAG_PREFIX) for tag in body_tags):
            body_tags.insert(1, owner_tag())  # an explicit owner tag (a MicroVM made for someone else) is kept
        payload: dict[str, Any] = {"name": name, "tags": body_tags}
        if oci_ref is not None:
            if region is None or size is None:
                raise ValueError("creating from oci_ref requires region and size")
            payload["source"] = {"oci_ref": oci_ref}
            payload["region"] = region
            payload["size"] = size.to_api()
            if environment:
                payload["environment"] = dict(environment)
        else:
            if environment:
                raise ValueError("a MicroVM created from a checkpoint inherits its environment; do not pass one")
            payload["source"] = {"checkpoint_id": checkpoint_id}
        if http_port is not None:
            payload["http_port"] = http_port
        if http_protocol is not None:
            payload["http_protocol"] = http_protocol
        if idle_timeout is not None:
            payload["auto_pause"] = {"idle_timeout": idle_timeout}
        if auto_resume is not None:
            payload["auto_resume"] = auto_resume
        if networking is not None:
            payload["networking"] = networking
        if vpc_uuid is not None:
            payload["vpc_uuid"] = vpc_uuid
        response = self._request("POST", "/microvms", json=payload)
        return _parse_microvm(self._unwrap(response, "microvm"))

    def get(self, microvm_id: str) -> MicroVMState:
        response = self._request("GET", f"/microvms/{microvm_id}")
        return _parse_microvm(self._unwrap(response, "microvm"))

    def list(self, *, name: str | None = None, tag: str | None = None, region: str | None = None) -> list[MicroVMState]:
        """GET /microvms, all pages. Filters combine with AND; `name` is an exact match."""
        params = {k: v for k, v in (("name", name), ("tag_name", tag), ("region", region)) if v}
        return [_parse_microvm(item) for item in self._paginate("/microvms", "microvms", params)]

    def get_by_name(self, name: str) -> MicroVMState | None:
        """The live MicroVM called `name`, or None. A `terminating` leftover with that name is ignored."""
        matches = [s for s in self.list(name=name) if s.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)]
        return matches[0] if matches else None

    def delete(self, microvm_id: str) -> None:
        self._request("DELETE", f"/microvms/{microvm_id}")

    def pause(self, microvm_id: str, *, timeout: float = 60.0) -> MicroVMState:
        """POST .../pause (idempotent), then GETs until it reads `paused` (see module docstring).

        A pause sent right after inbound requests to the MicroVM's endpoint can
        be accepted and have no effect, so while it still reads `running` the
        pause is sent again every `_REPAUSE_AFTER` seconds of the settle window.

        Returns the last state read: still `running` only if it did not read
        `paused` within the settle window (e.g. inbound requests keep resuming it).
        """
        self._request("POST", f"/microvms/{microvm_id}/pause")
        state = self.wait_for_status(microvm_id, timeout=timeout)
        now = time.monotonic()
        deadline, repause_at = now + min(timeout, _SETTLE_SECONDS), now + _REPAUSE_AFTER
        while state.status in (MicroVMStatus.RUNNING,) and time.monotonic() < deadline:
            if time.monotonic() >= repause_at:
                self._request("POST", f"/microvms/{microvm_id}/pause")
                repause_at = time.monotonic() + _REPAUSE_AFTER
            self._sleep(_SETTLE_POLL)
            state = self.wait_for_status(microvm_id, timeout=timeout)
        return state

    def resume(self, microvm_id: str, *, timeout: float = 60.0) -> MicroVMState:
        """POST .../resume (idempotent), then GETs until it reads `running` (see `pause`)."""
        self._request("POST", f"/microvms/{microvm_id}/resume")
        return self._settle(microvm_id, MicroVMStatus.RUNNING, timeout=timeout)

    def _settle(self, microvm_id: str, target: MicroVMStatus, *, timeout: float) -> MicroVMState:
        state = self.wait_for_status(microvm_id, timeout=timeout)
        deadline = time.monotonic() + min(timeout, _SETTLE_SECONDS)
        while (
            state.status != target
            and state.status in (MicroVMStatus.RUNNING, MicroVMStatus.PAUSED)
            and time.monotonic() < deadline
        ):
            self._sleep(_SETTLE_POLL)
            state = self.wait_for_status(microvm_id, timeout=timeout)
        return state

    def wait_for_status(self, microvm_id: str, *, timeout: float, poll_interval: float = 2.0) -> MicroVMState:
        """Poll until the state is no longer creating/pausing/resuming."""
        deadline = time.monotonic() + timeout
        state = self.get(microvm_id)
        while state.status in TRANSITIONAL_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MicroVM {microvm_id} still {state.status} after {timeout}s")
            self._sleep(poll_interval)
            state = self.get(microvm_id)
        return state

    def wait_for_endpoint(self, microvm_id: str, *, timeout: float, poll_interval: float = 2.0) -> MicroVMState:
        """Poll until the MicroVM is running AND its default endpoint is ACTIVE."""
        deadline = time.monotonic() + timeout
        state = self.get(microvm_id)
        while state.status != MicroVMStatus.RUNNING or state.endpoint is None:
            if state.status in (MicroVMStatus.FAILED, MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED):
                raise MicroVMClientError(
                    f"MicroVM {microvm_id} is {state.status}" + (f": {state.failure_reason}" if state.failure_reason else "")
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MicroVM {microvm_id} endpoint not ACTIVE after {timeout}s (state {state.status})")
            self._sleep(poll_interval)
            state = self.get(microvm_id)
        return state

    def exec(self, microvm_id: str, argv: list[str], *, cwd: str | None = None) -> ExecResult:
        """POST .../exec. HTTP 200 means the command ran; check `exit_code`.

        Runs `argv` directly, not through a shell. 60 s limit server-side;
        callers needing longer work detach it (see transport/exec.py).
        """
        if not argv:
            raise ValueError("argv must not be empty")
        body: dict[str, Any] = {"argv": list(argv)}
        if cwd is not None:
            body["cwd"] = cwd
        data = self._request("POST", f"/microvms/{microvm_id}/exec", json=body).json()
        return ExecResult(
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            exit_code=int(data.get("exit_code", -1)),
            truncated=bool(data.get("truncated", False)),
        )

    def options(self) -> dict[str, Any]:
        """GET /microvms/options: default region, sizes (with availability), account limits."""
        return dict(self._request("GET", "/microvms/options").json())

    # ------------------------------------------------------------------ checkpoints

    def create_checkpoint(self, microvm_id: str, *, name: str | None = None) -> Checkpoint:
        body = {"name": name} if name else {}
        response = self._request("POST", f"/microvms/{microvm_id}/checkpoints", json=body)
        return _parse_checkpoint(self._unwrap(response, "checkpoint"))

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        response = self._request("GET", f"/microvms/checkpoints/{checkpoint_id}")
        return _parse_checkpoint(self._unwrap(response, "checkpoint"))

    def list_checkpoints(self, *, microvm_id: str | None = None) -> list[Checkpoint]:
        params = {"microvm_id": microvm_id} if microvm_id else {}
        return [_parse_checkpoint(item) for item in self._paginate("/microvms/checkpoints", "checkpoints", params)]

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """DELETE; the API answers 400 while the checkpoint is still being captured."""
        self._request("DELETE", f"/microvms/checkpoints/{checkpoint_id}")

    def wait_for_checkpoint(self, checkpoint_id: str, *, timeout: float, poll_interval: float = 3.0) -> Checkpoint:
        deadline = time.monotonic() + timeout
        checkpoint = self.get_checkpoint(checkpoint_id)
        while checkpoint.status == "CHECKPOINT_CREATING":
            if time.monotonic() >= deadline:
                raise TimeoutError(f"checkpoint {checkpoint_id} still capturing after {timeout}s")
            self._sleep(poll_interval)
            checkpoint = self.get_checkpoint(checkpoint_id)
        return checkpoint
