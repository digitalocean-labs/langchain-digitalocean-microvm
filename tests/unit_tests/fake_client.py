"""In-memory double for langchain_digitalocean_microvm.client.MicroVMClient.

Duck-typed against the real client's public surface (create/get/list/
get_by_name/delete/pause/resume/wait_for_status/wait_for_endpoint/exec/
options/checkpoints) so provider.py, agent_host/host.py and
transport/exec.py can be tested without the network. It mirrors the
behaviours the real API showed when probed (tests/fixtures/microvm_api/):

- create returns `creating` with a PENDING endpoint; the MicroVM reaches
  `running` with an ACTIVE endpoint after `stuck_transitional_polls` GETs;
- names are unique: a second create with a live name is rejected (409);
- DELETE leaves the MicroVM visible as `terminating` (until `purge()`);
- exec really runs argv as a local subprocess (no shell added), so the
  shell wrapper in transport/exec.py is exercised for real, and it enforces
  the API's limits: each argument <= 128 KiB (MAX_ARG_STRLEN on Linux), the
  JSON body <= 1 MiB, output truncated at 4 MiB, `exec_time_limit` seconds.

Failure injection: `fail_next_create` (lose a create race), `force_status`
(id -> status, applied by get), `fail_create_with` (an exception to raise),
`fail_create_for` (name -> exception, for one MicroVM among several).

Thread-safe (one lock around state changes) so parallel provisioning can be
tested; `create_delay` makes each create take that long outside the lock, and
`peak_concurrent_creates` records how many overlapped.
"""

from __future__ import annotations

import functools
import json
import subprocess
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from langchain_digitalocean_microvm.client import MANAGED_TAG
from langchain_digitalocean_microvm.egress import EgressPolicy
from langchain_digitalocean_microvm.exceptions import (
    CheckpointUnavailableError,
    MicroVMAlreadyExistsError,
    MicroVMAPIError,
    MicroVMClientError,
    MicroVMNotFoundError,
)
from langchain_digitalocean_microvm.models import (
    Checkpoint,
    ExecResult,
    MicroVMSize,
    MicroVMState,
)
from langchain_digitalocean_microvm.owner import OWNER_TAG_PREFIX, owner_tag
from langchain_digitalocean_microvm.status import TRANSITIONAL_STATUSES, MicroVMStatus

_MAX_SIMULATED_POLLS = 10_000
MAX_ARG = 131_072
MAX_BODY = 1024 * 1024
MAX_OUTPUT = 4 * 1024 * 1024
_GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


def _locked(method):  # type: ignore[no-untyped-def]
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class FakeMicroVMClient:
    def __init__(
        self,
        *,
        fail_next_create: bool = False,
        stuck_transitional_polls: int = 0,
        force_status: dict[str, MicroVMStatus] | None = None,
        exec_time_limit: float = 60.0,
        create_delay: float = 0.0,
    ) -> None:
        self._lock = threading.RLock()
        self.create_delay = create_delay
        self._creates_in_flight = 0
        self.peak_concurrent_creates = 0
        self.fail_create_for: dict[str, Exception] = {}
        self.fail_next_create = fail_next_create
        self.fail_create_with: Exception | None = None
        self.stuck_transitional_polls = stuck_transitional_polls
        self.force_status: dict[str, MicroVMStatus] = force_status if force_status is not None else {}
        self.exec_time_limit = exec_time_limit

        self._microvms: dict[str, MicroVMState] = {}
        self._countdown: dict[str, int] = {}
        self._target: dict[str, MicroVMStatus] = {}
        self._checkpoints: dict[str, Checkpoint] = {}
        self._egress: dict[str, dict[str, Any]] = {}  # microvm id -> its egress_policy as the API returns it
        self._next_id = 1

        self.created: list[dict[str, Any]] = []  # every create() payload, in order
        self.exec_calls: list[tuple[str, list[str]]] = []
        self.create_calls = 0
        self.get_calls = 0
        self.delete_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.wait_for_status_calls = 0

    # ------------------------------------------------------------------ helpers

    def _new_id(self) -> str:
        microvm_id = f"01fake{self._next_id:04d}"
        self._next_id += 1
        return microvm_id

    def _register(self, *, name: str, size: MicroVMSize | None, region: str | None, tags: list[str],
                  http_port: int | None) -> MicroVMState:
        microvm_id = self._new_id()
        state = MicroVMState(
            id=microvm_id,
            name=name,
            status=MicroVMStatus.CREATING,
            size=size,
            region=region,
            endpoint=None,
            http_port=http_port,
            created_at=datetime(2026, 10, 1, 22, 46, 36, tzinfo=UTC),
            tags=tuple(tags),
        )
        self._microvms[microvm_id] = state
        self._countdown[microvm_id] = self.stuck_transitional_polls
        self._target[microvm_id] = MicroVMStatus.RUNNING
        return state

    def _begin_transition(self, microvm_id: str, *, transitional: MicroVMStatus, target: MicroVMStatus) -> None:
        state = self._require(microvm_id)
        if state.status in (MicroVMStatus.CREATING, MicroVMStatus.FAILED, *_GONE):
            raise MicroVMAPIError(400, f"cannot change a MicroVM that is {state.status}")
        if state.status == target:
            return  # idempotent, like the real API
        self._microvms[microvm_id] = replace(state, status=transitional)
        self._countdown[microvm_id] = self.stuck_transitional_polls
        self._target[microvm_id] = target

    def _require(self, microvm_id: str) -> MicroVMState:
        if microvm_id not in self._microvms:
            raise MicroVMNotFoundError(microvm_id)
        return self._microvms[microvm_id]

    def purge(self, microvm_id: str) -> None:
        """Test hook: a `terminating` MicroVM finally disappears."""
        self._microvms.pop(microvm_id, None)

    def live(self) -> list[MicroVMState]:
        return [s for s in self._microvms.values() if s.status not in _GONE]

    # ------------------------------------------------------------------ MicroVMs

    def create(self, *, name: str, oci_ref: str | None = None, checkpoint_id: str | None = None,
               region: str | None = None, size: MicroVMSize | None = None, http_port: int | None = None,
               http_protocol: str | None = None, environment: dict[str, str] | None = None,
               tags: list[str] | None = None, idle_timeout: str | None = None, auto_resume: bool | None = None,
               networking: str | None = None, vpc_uuid: str | None = None,
               egress_policy: EgressPolicy | None = None) -> MicroVMState:
        if checkpoint_id is not None and egress_policy is not None:
            raise ValueError("a MicroVM created from a checkpoint inherits its egress policy; do not pass one")
        payload = {k: v for k, v in locals().items() if k != "self"}
        with self._lock:
            self._creates_in_flight += 1
            self.peak_concurrent_creates = max(self.peak_concurrent_creates, self._creates_in_flight)
        try:
            if self.create_delay:
                time.sleep(self.create_delay)
            with self._lock:
                return self._create_locked(payload)
        finally:
            with self._lock:
                self._creates_in_flight -= 1

    def _create_locked(self, payload: dict[str, Any]) -> MicroVMState:
        name: str = payload["name"]
        checkpoint_id, region, size = payload["checkpoint_id"], payload["region"], payload["size"]
        tags, http_port = payload["tags"], payload["http_port"]
        self.create_calls += 1
        self.created.append(payload)
        if name in self.fail_create_for:
            raise self.fail_create_for.pop(name)
        if self.fail_create_with is not None:
            exc, self.fail_create_with = self.fail_create_with, None
            raise exc
        body_tags = [MANAGED_TAG] + [t for t in tags or [] if t != MANAGED_TAG]
        if not any(t.startswith(OWNER_TAG_PREFIX) for t in body_tags):
            body_tags.insert(1, owner_tag())  # the real client adds this user's owner tag
        if checkpoint_id is not None:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint is None or not checkpoint.available:
                raise MicroVMAPIError(400, "checkpoint not available")
            origin = self._microvms.get(checkpoint.microvm_id or "")
            if checkpoint.has_credentials and (origin is None or origin.status in _GONE):
                raise CheckpointUnavailableError(
                    "the origin MicroVM's egress credentials are incomplete; this checkpoint cannot be restored"
                )
            region, size = checkpoint.region, checkpoint.size
        if self.fail_next_create:
            self.fail_next_create = False
            self._register(name=name, size=size, region=region, tags=body_tags, http_port=http_port)
            raise MicroVMAlreadyExistsError(f"simulated create race for {name!r}")
        if any(s.name == name for s in self.live()):
            raise MicroVMAlreadyExistsError(f"a MicroVM named {name!r} already exists in this region")
        state = self._register(name=name, size=size, region=region, tags=body_tags, http_port=http_port)
        policy: EgressPolicy | None = payload["egress_policy"]
        if policy is not None:
            api = policy.to_api()
            for credential in api.get("credentials", []):
                del credential["secret"]  # the API never returns a secret
            self._egress[state.id] = api
        elif checkpoint_id is not None and checkpoint_id in self._checkpoints:
            self._egress[state.id] = dict(self._checkpoints[checkpoint_id].raw.get("egress_policy") or {})
        if checkpoint_id is not None:  # clones come back already running
            state = self._settle(state.id, MicroVMStatus.RUNNING)
        return state

    def _settle(self, microvm_id: str, target: MicroVMStatus) -> MicroVMState:
        state = self._microvms[microvm_id]
        endpoint = state.endpoint
        if target == MicroVMStatus.RUNNING and endpoint is None:
            endpoint = f"{microvm_id}.mkc1.microvm.ondigitalocean.com"
        state = replace(state, status=target, endpoint=endpoint)
        self._microvms[microvm_id] = state
        self._countdown.pop(microvm_id, None)
        self._target.pop(microvm_id, None)
        return state

    @_locked
    def get(self, microvm_id: str) -> MicroVMState:
        self.get_calls += 1
        state = self._require(microvm_id)
        if microvm_id in self.force_status:
            return replace(state, status=self.force_status[microvm_id])
        if microvm_id in self._countdown:
            if self._countdown[microvm_id] > 0:
                self._countdown[microvm_id] -= 1
                return state
            state = self._settle(microvm_id, self._target[microvm_id])
        return state

    @_locked
    def list(self, *, name: str | None = None, tag: str | None = None, region: str | None = None) -> list[MicroVMState]:
        results = []
        for microvm_id in list(self._microvms):
            state = self.get(microvm_id)
            if name is not None and state.name != name:
                continue
            if tag is not None and tag not in state.tags:
                continue
            if region is not None and state.region != region:
                continue
            results.append(state)
        return results

    def get_by_name(self, name: str) -> MicroVMState | None:
        matches = [s for s in self.list(name=name) if s.status not in _GONE]
        return matches[0] if matches else None

    @_locked
    def delete(self, microvm_id: str) -> None:
        self.delete_calls += 1
        self._microvms[microvm_id] = replace(self._require(microvm_id), status=MicroVMStatus.TERMINATING, endpoint=None)
        self._countdown.pop(microvm_id, None)
        self._target.pop(microvm_id, None)
        self.force_status.pop(microvm_id, None)

    @_locked
    def pause(self, microvm_id: str, *, timeout: float = 60.0) -> MicroVMState:
        self.pause_calls += 1
        self._begin_transition(microvm_id, transitional=MicroVMStatus.PAUSING, target=MicroVMStatus.PAUSED)
        return self.wait_for_status(microvm_id, timeout=timeout)

    @_locked
    def resume(self, microvm_id: str, *, timeout: float = 60.0) -> MicroVMState:
        self.resume_calls += 1
        self._begin_transition(microvm_id, transitional=MicroVMStatus.RESUMING, target=MicroVMStatus.RUNNING)
        return self.wait_for_status(microvm_id, timeout=timeout)

    @_locked
    def wait_for_status(self, microvm_id: str, *, timeout: float, poll_interval: float = 2.0) -> MicroVMState:
        self.wait_for_status_calls += 1
        state = self.get(microvm_id)
        polls = 0
        while state.status in TRANSITIONAL_STATUSES:
            polls += 1
            if polls > _MAX_SIMULATED_POLLS:
                raise TimeoutError(f"{microvm_id} stuck in {state.status}")
            state = self.get(microvm_id)
        return state

    def wait_for_endpoint(self, microvm_id: str, *, timeout: float, poll_interval: float = 2.0) -> MicroVMState:
        state = self.wait_for_status(microvm_id, timeout=timeout)
        if state.status != MicroVMStatus.RUNNING or state.endpoint is None:
            raise MicroVMClientError(f"MicroVM {microvm_id} is {state.status}")
        return state

    def exec(self, microvm_id: str, argv: list[str], *, cwd: str | None = None) -> ExecResult:
        self.exec_calls.append((microvm_id, list(argv)))
        state = self.get(microvm_id)
        if state.status in _GONE or state.status == MicroVMStatus.FAILED:
            raise MicroVMAPIError(502, "the command did not reach the workload")
        if state.status == MicroVMStatus.PAUSED:  # exec auto-resumes
            self._settle(microvm_id, MicroVMStatus.RUNNING)
        if not argv:
            raise MicroVMAPIError(400, "argv is missing or empty")
        if any(len(a.encode()) > MAX_ARG for a in argv):
            raise MicroVMAPIError(400, "argument longer than MAX_ARG_STRLEN (simulated E2BIG)")
        if len(json.dumps({"argv": argv, "cwd": cwd})) > MAX_BODY:
            raise MicroVMAPIError(400, "request body larger than 1 MiB")
        try:
            done = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=self.exec_time_limit, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise MicroVMAPIError(504, f"command exceeded {self.exec_time_limit}s") from exc
        out, err = done.stdout, done.stderr
        truncated = len(out) + len(err) > MAX_OUTPUT
        if truncated:
            out = out[:MAX_OUTPUT]
            err = err[: max(0, MAX_OUTPUT - len(out))]
        return ExecResult(stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"),
                          exit_code=done.returncode, truncated=truncated)

    def options(self) -> dict[str, Any]:
        return {"account_limits": {"max_total_count": 2, "max_memory_bytes": 4 * 1024**3, "max_idle_timeout_seconds": 3600},
                "default_region": "mkc1"}

    # ------------------------------------------------------------------ checkpoints

    def create_checkpoint(self, microvm_id: str, *, name: str | None = None) -> Checkpoint:
        state = self._require(microvm_id)
        raw = {"egress_policy": self._egress[microvm_id]} if microvm_id in self._egress else {}
        checkpoint = Checkpoint(id=f"cp-{len(self._checkpoints) + 1}", name=name or "checkpoint", microvm_id=microvm_id,
                                region=state.region, size=state.size, status="CHECKPOINT_AVAILABLE", created_at=None,
                                raw=raw)
        self._checkpoints[checkpoint.id] = checkpoint
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        if checkpoint_id not in self._checkpoints:
            raise MicroVMNotFoundError(checkpoint_id)
        return self._checkpoints[checkpoint_id]

    def list_checkpoints(self, *, microvm_id: str | None = None) -> list[Checkpoint]:
        return [c for c in self._checkpoints.values() if microvm_id is None or c.microvm_id == microvm_id]

    def wait_for_checkpoint(self, checkpoint_id: str, *, timeout: float, poll_interval: float = 3.0) -> Checkpoint:
        return self.get_checkpoint(checkpoint_id)

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        self._checkpoints.pop(checkpoint_id, None)
