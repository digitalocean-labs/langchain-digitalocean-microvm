"""Create-or-reuse lifecycle management for MicroVMSandbox instances.

MicroVMLifecycle is the class in this package that turns the full
MicroVM FAQ lifecycle (Launch -> Run -> Pause/Resume -> Destroy) into a
single `get_or_create(sandbox_id)` call that always hands back a
ready-to-use MicroVMSandbox, regardless of whether that sandbox_id's
MicroVM already existed, was paused, was mid-transition, or had previously
failed. This mirrors the create-or-reuse-with-seeding pattern of the Deep Agents
course's LangSmith sandbox assistant (m5/sales_assistant_sandbox), with a
typed status enum driving the branching instead of string comparisons.

The state machine itself (settle to RUNNING, transport connect) lives in
reconcile.py so that agent_host.MicroVMAgentHost --
the "agent in sandbox" sibling of this class -- runs the identical logic.

Deliberately NOT a subclass of deepagents_code's SandboxProvider, and this
module does not import that (optional, companion) package at all --
langchain_digitalocean_microvm must work standalone with zero dependency on it. The
method names/shapes below (get_or_create/pause/resume/delete plus async
twins via asyncio.to_thread) are chosen to structurally match
SandboxProvider so that wrapping this class as a real SandboxProvider later
is a thin adapter, not a rewrite -- see exceptions.py's docstring for the
same reasoning applied to SandboxError/SandboxNotFoundError.

Two independent branches live side by side in every method:

- real (config.transport == "exec" or "endpoint"): talks to the MicroVMs API
  through MicroVMClient and runs commands through transport.exec.ExecTransport
  or, with "endpoint", through the in-VM sandbox server over the MicroVM's
  HTTPS endpoint (transport/endpoint.py). The
  MicroVM is found by its deterministic name (client.microvm_name). Note
  that MicroVMs auto-pause after `config.idle_timeout` without inbound
  requests; exec calls auto-resume them, so a sandbox paused by the platform
  between agent turns simply wakes up on the next command.
- local (config.transport == "local"): no MicroVMClient, no network at all -- just an in-memory dict simulating RUNNING/PAUSED, paired
  with transport.local.LocalTransport's real-subprocess-on-local-disk
  semantics. This is what lets the whole package (and anything built on top
  of it) be demoed and tested with zero DigitalOcean credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import shutil
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.client import (
    MANAGED_TAG,
    MicroVMClient,
    id_tag,
    microvm_name,
    sandbox_id_from_tags,
)
from langchain_digitalocean_microvm.config import MicroVMConfig
from langchain_digitalocean_microvm.egress import EgressPolicy
from langchain_digitalocean_microvm.exceptions import (
    MicroVMAlreadyExistsError,
    MicroVMAPIError,
    MicroVMNotFoundError,
    SandboxError,
    SandboxNotFoundError,
)
from langchain_digitalocean_microvm.models import Checkpoint, MicroVMState
from langchain_digitalocean_microvm.namespace import (
    namespace_from_tags,
    namespace_tags,
    namespaced_prefix,
)
from langchain_digitalocean_microvm.reconcile import (
    connect_transport,
    settle_to_running,
)
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.transport.base import (
    SandboxTransport,
    TransportCredentials,
)
from langchain_digitalocean_microvm.transport.local import root_for_host
from langchain_digitalocean_microvm.transport.registry import create_transport

logger = logging.getLogger(__name__)

SANDBOX_NAME_PREFIX = "microvm-sandbox"
TEMPLATE_PREFIX = "microvm-template"  # checkpoint names, and the temporary build MicroVM's tag
TEMPLATE_BUILD_PREFIX = "microvm-template-build"
DEFAULT_SETUP_TIMEOUT = 900  # seconds per setup command (pip installs); ExecTransport detaches and polls


@dataclass(frozen=True, slots=True)
class SandboxTemplate:
    """A checkpoint of a prepared sandbox (`MicroVMLifecycle.build_template`).

    A sandbox created from it (`get_or_create(..., template=...)`) starts with
    the template's files, installed packages and running processes, in a few
    seconds instead of a cold boot plus setup.
    """

    name: str
    checkpoint_name: str  # includes a hash of image, size, setup, seed files and outbound policy
    checkpoint_id: str  # local mode: the template's directory key
    built: bool  # True if this call built it, False if an existing one was reused


def _template_digest(
    name: str,
    config: MicroVMConfig,
    setup: Sequence[str],
    seed_files: Sequence[tuple[str, bytes]],
    egress: EgressPolicy | None,
) -> str:
    """Changes whenever anything that shapes the template does, so a stale checkpoint is never reused."""
    digest = hashlib.sha256()
    for part in (name, config.image or "", repr(config.size), str(config.http_port), *setup):
        digest.update(part.encode() + b"\0")
    for path, content in seed_files:
        digest.update(path.encode() + b"\0" + hashlib.sha256(content).digest())
    if egress is not None:  # the secrets enter keyed with the API token, so the name reveals nothing about them
        digest.update(b"egress\0" + egress.digest((config.api_token or "").encode()).encode())
    return digest.hexdigest()[:10]


class MicroVMLifecycle:
    """Create-or-reuse manager for MicroVMSandbox instances, one per sandbox_id."""

    def __init__(
        self,
        config: MicroVMConfig | None = None,
        *,
        client: MicroVMClient | None = None,
        transport_factory: Callable[[], SandboxTransport] | None = None,
    ) -> None:
        """Build a lifecycle manager.

        `client` and `transport_factory` are optional, additive hooks that
        exist purely for tests (see tests/test_provider.py and
        tests/fake_client.py): pass a pre-built MicroVMClient (or any
        duck-typed stand-in exposing the same methods) to skip building one
        from `config`, and/or a zero-arg factory controlling exactly what
        SandboxTransport _connect_backend() connects, instead of always
        deferring to transport.registry.create_transport(config.transport).
        Production callers never need either and can keep just passing
        `config`.
        """
        self.config = config or MicroVMConfig.from_env()

        if client is None and self.config.transport != "local" and not self.config.api_token:
            raise SandboxError("MicroVMConfig.api_token is required unless transport='local'")
        if self.config.transport == "endpoint" and not self.config.access_token:
            raise SandboxError(
                "DO_MICROVM_TRANSPORT=endpoint needs DO_MICROVM_ACCESS_TOKEN: a token scoped to microvm:access "
                "only, for requests to the sandboxes' endpoints (or set DO_MICROVM_TRANSPORT=exec)"
            )
        self._client: MicroVMClient | None = client
        if self._client is None and self.config.transport != "local":
            assert self.config.api_token is not None
            self._client = MicroVMClient(self.config.api_token, base_url=self.config.base_url)

        self._transport_factory: Callable[[], SandboxTransport] = (
            transport_factory
            if transport_factory is not None
            else lambda: create_transport(self.config.transport)
        )

        # Local transport state only: sandbox_id -> status, cycling between
        # RUNNING and PAUSED. Never populated or consulted when a real
        # MicroVMClient is in use.
        self._local_states: dict[str, MicroVMStatus] = {}

        # Bounds the get-or-create calls (creates and resumes) in flight, whoever calls them.
        limit = self.config.max_concurrent_creates
        self._creates = threading.BoundedSemaphore(limit) if limit else None
        self._warned_local_egress = False

    # ------------------------------------------------------------------
    # public sync API
    # ------------------------------------------------------------------

    def get_or_create(
        self,
        *,
        sandbox_id: str,
        seed_files: list[tuple[str, bytes]] | None = None,
        tags: Sequence[str] = (),
        template: SandboxTemplate | None = None,
        egress: EgressPolicy | None = None,
    ) -> MicroVMSandbox:
        """Return a connected MicroVMSandbox for `sandbox_id`, creating one if needed.

        With `template` (`build_template`), a sandbox this call creates is
        cloned from the template's checkpoint instead of booted from the image;
        an existing sandbox is reused as it is, like `seed_files`.

        `egress` is the outbound policy of a sandbox this call creates from the
        image (default: `config.egress`). A sandbox created from a template has
        the template's policy, so `egress` and `template` cannot be combined. An
        existing sandbox keeps the policy it was created with.

        `seed_files` is uploaded (via MicroVMSandbox.upload_files) only
        when this call is the one that actually created the underlying
        MicroVM -- a reused, resumed, or raced-into-existence MicroVM is
        never re-seeded. `tags` are added to MANAGED_TAG and the id tag on
        create (tags cannot change afterwards); cleanup tooling finds a class
        of sandboxes by them (e.g. subagents.SUBAGENT_SANDBOX_TAG).
        """
        if egress is not None and template is not None:
            raise ValueError("a sandbox created from a template has the template's egress policy; pass one or the other")
        policy = None if template is not None else (egress or self.config.egress)
        with self._create_slot():
            if self.config.transport == "local":
                self._warn_local_egress(policy)
                return self._local_get_or_create(sandbox_id=sandbox_id, seed_files=seed_files, template=template)
            return self._real_get_or_create(
                sandbox_id=sandbox_id, seed_files=seed_files, tags=tuple(tags), template=template, egress=policy
            )

    def _warn_local_egress(self, policy: EgressPolicy | None) -> None:
        if policy is not None and not self._warned_local_egress:
            self._warned_local_egress = True
            logger.warning("transport='local' does not apply an egress policy: commands use this machine's network")

    def _create_slot(self) -> AbstractContextManager[Any]:
        """A slot of `config.max_concurrent_creates` (a no-op without a limit), held while provisioning."""
        return self._creates if self._creates is not None else contextlib.nullcontext()

    def pause(
        self,
        backend: MicroVMSandbox | None = None,
        *,
        sandbox_id: str | None = None,
    ) -> MicroVMState | None:
        """Pause a MicroVM, given either its live `backend` or its `sandbox_id`.

        With `backend`, its transport is closed first. With `sandbox_id`
        alone -- a caller that never connected to the MicroVM in this
        process, e.g. a supervisor checkpointing a child by id -- the MicroVM
        is looked up by name and paused. Already-PAUSED MicroVMs are left
        alone. (MicroVMs also pause by themselves after `idle_timeout`.)
        Returns the MicroVM's state afterwards (None in local mode); it is not
        `paused` only if something resumed it at once.
        """
        if (backend is None) == (sandbox_id is None):
            raise ValueError("pass exactly one of `backend` or `sandbox_id`")

        if self.config.transport == "local":
            if backend is not None:
                backend.close()
                self._local_states[backend.id] = MicroVMStatus.PAUSED
                return None
            assert sandbox_id is not None
            if sandbox_id not in self._local_states:
                raise SandboxNotFoundError(
                    f"No local MicroVM registered for sandbox_id={sandbox_id!r}"
                )
            self._local_states[sandbox_id] = MicroVMStatus.PAUSED
            return None

        client = self._client_or_raise()
        if backend is not None:
            backend.close()
            return client.pause(backend.id, timeout=self.config.ready_timeout)

        assert sandbox_id is not None
        state = self._lookup(sandbox_id)
        if state.status == MicroVMStatus.PAUSED:
            return state
        return client.pause(state.id, timeout=self.config.ready_timeout)

    def resume(self, *, sandbox_id: str) -> MicroVMSandbox:
        """Resume the MicroVM backing `sandbox_id` and return a freshly connected sandbox."""
        if self.config.transport == "local":
            if sandbox_id not in self._local_states:
                raise SandboxNotFoundError(
                    f"No local MicroVM registered for sandbox_id={sandbox_id!r}"
                )
            self._local_states[sandbox_id] = MicroVMStatus.RUNNING
            return self._local_connect(sandbox_id)

        client = self._client_or_raise()
        state = self._lookup(sandbox_id)
        resumed = client.resume(state.id, timeout=self.config.ready_timeout)
        return self._connect_backend(resumed)

    def delete(self, *, sandbox_id: str) -> None:
        """Permanently destroy the MicroVM backing `sandbox_id`."""
        if self.config.transport == "local":
            if sandbox_id not in self._local_states:
                raise SandboxNotFoundError(
                    f"No local MicroVM registered for sandbox_id={sandbox_id!r}"
                )
            del self._local_states[sandbox_id]
            shutil.rmtree(root_for_host(sandbox_id), ignore_errors=True)  # a later create with this id starts empty
            return

        client = self._client_or_raise()
        state = self._lookup(sandbox_id)
        client.delete(state.id)

    def sandbox_id_for(self, microvm_id: str) -> str | None:
        """The sandbox_id behind a MicroVM id (`MicroVMSandbox.id`), or None if it is not one of ours.

        Callers that only kept `backend.id` (deepagents_code prints and later
        deletes sandboxes by it) map it back with this. In local mode a
        backend's id IS its sandbox_id.
        """
        if self.config.transport == "local":
            return microvm_id if microvm_id in self._local_states else None
        try:
            state = self._client_or_raise().get(microvm_id)
        except MicroVMNotFoundError:
            return None
        except MicroVMAPIError as exc:
            if exc.status_code in (400, 422):  # not a MicroVM id at all
                return None
            raise
        if MANAGED_TAG not in state.tags or state.status in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED):
            return None
        if namespace_from_tags(state.tags) != self.config.namespace:  # another namespace's sandbox
            return None
        return sandbox_id_from_tags(state.tags)

    # ------------------------------------------------------------------
    # public async API -- thin asyncio.to_thread wrappers, matching
    # SandboxProvider's async-twin shape without importing that package.
    # ------------------------------------------------------------------

    def build_template(
        self,
        name: str,
        *,
        setup: Sequence[str] = (),
        seed_files: Sequence[tuple[str, bytes]] = (),
        rebuild: bool = False,
        setup_timeout: int = DEFAULT_SETUP_TIMEOUT,
        egress: EgressPolicy | None = None,
    ) -> SandboxTemplate:
        """Prepare a sandbox once and checkpoint it, so new sandboxes start from it warm.

        Boots a temporary MicroVM from the configured image, uploads
        `seed_files`, runs each `setup` command (e.g. `pip install ...`;
        a non-zero exit raises), checkpoints it and deletes the temporary
        MicroVM. Reuses an existing checkpoint built from the same inputs
        unless `rebuild=True`. Checkpoints outlive their MicroVM and do not
        count as MicroVMs; `delete_template` removes one.

        `egress` (default: `config.egress`) is the outbound policy of the
        temporary MicroVM, so `setup` runs under it, and every sandbox created
        from the template inherits it. A policy with credentials keeps the
        temporary MicroVM, paused: a checkpoint cannot be restored once the
        MicroVM it came from is deleted with its credentials. `delete_template`
        deletes it with the checkpoint.

        Local mode copies a directory instead and does NOT run `setup` (the
        commands would run on this machine); only `seed_files` are applied.
        """
        policy = egress or self.config.egress
        digest = _template_digest(name, self.config, setup, seed_files, policy)
        if self.config.transport == "local":
            self._warn_local_egress(policy)
            return self._local_build_template(name, digest, setup=setup, seed_files=seed_files, rebuild=rebuild)
        return self._real_build_template(
            name, digest, setup=setup, seed_files=seed_files, rebuild=rebuild, setup_timeout=setup_timeout,
            egress=policy,
        )

    def find_template(
        self,
        name: str,
        *,
        setup: Sequence[str] = (),
        seed_files: Sequence[tuple[str, bytes]] = (),
        egress: EgressPolicy | None = None,
    ) -> SandboxTemplate | None:
        """The template `build_template` would reuse for these inputs, or None; never builds one."""
        digest = _template_digest(name, self.config, setup, seed_files, egress or self.config.egress)
        if self.config.transport == "local":
            key = f"{TEMPLATE_PREFIX}-{name}-{digest}"
            ready = (root_for_host(key).parent / f"{key}.ready").exists()
            return SandboxTemplate(name, key, key, built=False) if ready else None
        checkpoint_name = microvm_name(TEMPLATE_PREFIX, f"{name}-{digest}")
        for checkpoint in self._client_or_raise().list_checkpoints():
            if checkpoint.name == checkpoint_name and checkpoint.available and self._restorable(checkpoint):
                return SandboxTemplate(name, checkpoint_name, checkpoint.id, built=False)
        return None

    def _restorable(self, checkpoint: Checkpoint) -> bool:
        """False for a checkpoint with credentials whose origin MicroVM is gone (its status does not say so)."""
        if not checkpoint.has_credentials:
            return True
        if checkpoint.microvm_id is None:
            return False
        try:
            origin = self._client_or_raise().get(checkpoint.microvm_id)
        except MicroVMNotFoundError:
            return False
        return origin.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)

    def delete_template(self, template: SandboxTemplate) -> None:
        """Delete a template's checkpoint, and the MicroVM it was taken from if that was kept.

        Sandboxes created from it are not affected.
        """
        if self.config.transport == "local":
            shutil.rmtree(root_for_host(template.checkpoint_id), ignore_errors=True)
            (root_for_host(template.checkpoint_id).parent / f"{template.checkpoint_id}.ready").unlink(missing_ok=True)
            return
        client = self._client_or_raise()
        try:
            checkpoint = client.get_checkpoint(template.checkpoint_id)
        except MicroVMNotFoundError:
            return
        self._delete_template_checkpoint(checkpoint)

    def _delete_template_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Delete a template checkpoint, then its origin MicroVM if one was kept (a policy with credentials)."""
        client = self._client_or_raise()
        try:
            client.delete_checkpoint(checkpoint.id)
        except MicroVMNotFoundError:
            pass
        if checkpoint.microvm_id is None:
            return
        try:
            origin = client.get(checkpoint.microvm_id)
        except MicroVMNotFoundError:
            return
        if origin.name.startswith(f"{TEMPLATE_BUILD_PREFIX}-") and origin.status not in (
            MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED
        ):
            try:
                client.delete(origin.id)
            except MicroVMNotFoundError:
                pass

    async def abuild_template(self, name: str, **kwargs: Any) -> SandboxTemplate:
        return await asyncio.to_thread(lambda: self.build_template(name, **kwargs))

    async def afind_template(self, name: str, **kwargs: Any) -> SandboxTemplate | None:
        return await asyncio.to_thread(lambda: self.find_template(name, **kwargs))

    async def adelete_template(self, template: SandboxTemplate) -> None:
        await asyncio.to_thread(self.delete_template, template)

    async def aget_or_create(self, **kwargs: Any) -> MicroVMSandbox:
        return await asyncio.to_thread(self.get_or_create, **kwargs)

    async def apause(self, backend: MicroVMSandbox | None = None, **kwargs: Any) -> MicroVMState | None:
        return await asyncio.to_thread(self.pause, backend, **kwargs)

    async def aresume(self, **kwargs: Any) -> MicroVMSandbox:
        return await asyncio.to_thread(self.resume, **kwargs)

    async def adelete(self, **kwargs: Any) -> None:
        return await asyncio.to_thread(self.delete, **kwargs)

    # ------------------------------------------------------------------
    # real-transport internals
    # ------------------------------------------------------------------

    def _client_or_raise(self) -> MicroVMClient:
        assert self._client is not None
        return self._client

    def _name(self, sandbox_id: str) -> str:
        return microvm_name(namespaced_prefix(SANDBOX_NAME_PREFIX, self.config.namespace), sandbox_id)

    def _lookup(self, sandbox_id: str) -> MicroVMState:
        name = self._name(sandbox_id)
        state = self._client_or_raise().get_by_name(name)
        if state is None:
            raise SandboxNotFoundError(
                f"No MicroVM registered for sandbox_id={sandbox_id!r}"
            ) from MicroVMNotFoundError(f"No MicroVM named {name!r}")
        return state

    def _real_get_or_create(
        self,
        *,
        sandbox_id: str,
        seed_files: list[tuple[str, bytes]] | None,
        tags: tuple[str, ...] = (),
        template: SandboxTemplate | None = None,
        egress: EgressPolicy | None = None,
    ) -> MicroVMSandbox:
        client = self._client_or_raise()

        state = client.get_by_name(self._name(sandbox_id))
        freshly_created = False
        if state is None:
            state, freshly_created = self._create_real(sandbox_id=sandbox_id, tags=tags, template=template, egress=egress)

        state, freshly_created = settle_to_running(
            client,
            state,
            label=f"sandbox_id={sandbox_id!r}",
            ready_timeout=self.config.ready_timeout,
            recreate=lambda: self._create_real(sandbox_id=sandbox_id, tags=tags, template=template, egress=egress),
            freshly_created=freshly_created,
        )

        backend = self._connect_backend(state)
        if freshly_created and seed_files:
            self._seed(backend, seed_files)
        return backend

    def _create_real(
        self,
        *,
        sandbox_id: str,
        tags: tuple[str, ...] = (),
        template: SandboxTemplate | None = None,
        egress: EgressPolicy | None = None,
    ) -> tuple[MicroVMState, bool]:
        """Create a fresh, tagged MicroVM (possibly still transitional).

        The returned state is handed to reconcile.settle_to_running exactly
        like any other newly-found state -- this function does not itself
        wait for RUNNING, so that logic lives in exactly one place.

        Returns (state, True) on a normal create. Returns (state, False) if
        another caller's create won the race between our lookup miss and this
        call (names are unique per team and region, so the API rejects ours): that
        MicroVM was not created by US, so it is not eligible for seeding.
        """
        client = self._client_or_raise()
        if template is None and not self.config.image:
            raise SandboxError(
                "A sandbox image is required: set DO_MICROVM_IMAGE to a linux/amd64 container image "
                "that keeps running and ships sh, coreutils and python3; for deepagents its user must be able "
                "to write to / (scripts/build_sandbox_image.py builds one). Not DO_MICROVM_AGENT_IMAGE: that "
                "image runs an Agent Server as uid 1000"
            )
        name = self._name(sandbox_id)
        try:
            if template is not None:  # a clone inherits image, size, region, port and environment
                return (
                    client.create(
                        name=name,
                        checkpoint_id=template.checkpoint_id,
                        idle_timeout=self.config.idle_timeout,
                        tags=[MANAGED_TAG, id_tag(sandbox_id), *namespace_tags(self.config.namespace), *tags],
                    ),
                    True,
                )
            return (
                client.create(
                    name=name,
                    oci_ref=self.config.image,
                    region=self.config.region,
                    size=self.config.size,
                    http_port=self.config.http_port,
                    idle_timeout=self.config.idle_timeout,
                    tags=[MANAGED_TAG, id_tag(sandbox_id), *namespace_tags(self.config.namespace), *tags],
                    egress_policy=egress.for_image(self.config.image or "") if egress is not None else None,
                ),
                True,
            )
        except (MicroVMAlreadyExistsError, MicroVMAPIError) as exc:
            if isinstance(exc, MicroVMAPIError) and exc.status_code not in (400, 409, 422):
                raise
            existing = client.get_by_name(name)
            if existing is None:
                raise  # rejected for some other reason; don't paper over it
            return existing, False

    def _connect_backend(self, state: MicroVMState) -> MicroVMSandbox:
        """Build+connect the configured transport for `state` and wrap it as a sandbox."""
        transport = connect_transport(self.config, state, self._transport_factory())
        return MicroVMSandbox(microvm_id=state.id, transport=transport)

    def _real_build_template(
        self,
        name: str,
        digest: str,
        *,
        setup: Sequence[str],
        seed_files: Sequence[tuple[str, bytes]],
        rebuild: bool,
        setup_timeout: int,
        egress: EgressPolicy | None = None,
    ) -> SandboxTemplate:
        client = self._client_or_raise()
        if not self.config.image:
            raise SandboxError("build_template needs DO_MICROVM_IMAGE (the image the template starts from)")
        checkpoint_name = microvm_name(TEMPLATE_PREFIX, f"{name}-{digest}")
        existing = [c for c in client.list_checkpoints() if c.name == checkpoint_name]
        if rebuild:
            for checkpoint in existing:
                self._delete_template_checkpoint(checkpoint)
        else:
            for checkpoint in existing:
                if not checkpoint.available:
                    continue
                if self._restorable(checkpoint):
                    return SandboxTemplate(name, checkpoint_name, checkpoint.id, built=False)
                logger.warning("Template %r: the MicroVM it was taken from is gone with its credentials; rebuilding", name)
                self._delete_template_checkpoint(checkpoint)

        build_name = microvm_name(TEMPLATE_BUILD_PREFIX, f"{name}-{digest}")
        state = client.get_by_name(build_name)  # left over from an interrupted build
        if state is None:
            with self._create_slot():
                state = client.create(
                    name=build_name,
                    oci_ref=self.config.image,
                    region=self.config.region,
                    size=self.config.size,
                    http_port=self.config.http_port,
                    idle_timeout=self.config.idle_timeout,
                    tags=[MANAGED_TAG, TEMPLATE_PREFIX, *namespace_tags(self.config.namespace)],
                    egress_policy=egress.for_image(self.config.image or "") if egress is not None else None,
                )
        keep_origin = False
        try:
            with self._create_slot():
                state, _ = settle_to_running(
                    client, state, label=f"template {name!r}", ready_timeout=self.config.ready_timeout, recreate=None
                )
            backend = self._connect_backend(state)
            for result in backend.upload_files(list(seed_files)):
                if result.error:
                    raise SandboxError(f"template {name!r}: could not upload {result.path}: {result.error}")
            for command in setup:
                result = backend.execute(command, timeout=setup_timeout)
                if result.exit_code != 0:
                    raise SandboxError(
                        f"template {name!r}: setup command failed (exit {result.exit_code}): {command}\n"
                        f"{result.output[-2000:]}"
                    )
            for attempt in (1, 2):  # a failed capture is tried once more
                checkpoint = client.create_checkpoint(state.id, name=checkpoint_name)
                checkpoint = client.wait_for_checkpoint(checkpoint.id, timeout=max(self.config.ready_timeout, 300.0))
                if checkpoint.available:
                    break
                reason = checkpoint.raw.get("failure_reason") or "no reason given"
                if checkpoint.status != "CHECKPOINT_FAILED":
                    raise SandboxError(f"template {name!r}: checkpoint {checkpoint.id} ended {checkpoint.status} ({reason})")
                with contextlib.suppress(MicroVMNotFoundError):
                    client.delete_checkpoint(checkpoint.id)
                if attempt == 2:
                    raise SandboxError(f"template {name!r}: checkpoint {checkpoint.id} ended {checkpoint.status} ({reason})")
                logger.warning("Template %r: checkpoint %s failed (%s); trying again", name, checkpoint.id, reason)
            # Deleting the origin would delete its stored credentials, and its checkpoints could no longer be restored.
            keep_origin = bool(egress is not None and egress.credentials)
            if keep_origin:
                try:
                    client.pause(state.id, timeout=self.config.ready_timeout)
                except Exception:  # noqa: BLE001 - the template is ready; the MicroVM also pauses when idle
                    logger.warning("Could not pause the origin MicroVM of template %r", name, exc_info=True)
            return SandboxTemplate(name, checkpoint_name, checkpoint.id, built=True)
        finally:
            try:
                if not keep_origin:
                    client.delete(state.id)  # the checkpoint outlives it
            except MicroVMNotFoundError:
                pass
            except Exception:  # noqa: BLE001 - must not mask the build's own outcome
                logger.warning("Could not delete the build MicroVM of template %r", name, exc_info=True)

    # ------------------------------------------------------------------
    # local transport internals -- no MicroVMClient and no network calls
    # anywhere below this line.
    # ------------------------------------------------------------------

    def _local_get_or_create(
        self,
        *,
        sandbox_id: str,
        seed_files: list[tuple[str, bytes]] | None,
        template: SandboxTemplate | None = None,
    ) -> MicroVMSandbox:
        freshly_created = sandbox_id not in self._local_states
        if freshly_created:
            if template is not None:
                source = root_for_host(template.checkpoint_id)
                if not (source.parent / f"{template.checkpoint_id}.ready").exists():
                    raise SandboxNotFoundError(f"Local template {template.checkpoint_id!r} no longer exists")
                shutil.copytree(source, root_for_host(sandbox_id), dirs_exist_ok=True)
            self._local_states[sandbox_id] = MicroVMStatus.RUNNING
        elif self._local_states[sandbox_id] == MicroVMStatus.PAUSED:
            self._local_states[sandbox_id] = MicroVMStatus.RUNNING  # simulate resume
        # else: already RUNNING -- reuse as-is.

        backend = self._local_connect(sandbox_id)
        if freshly_created and seed_files:
            self._seed(backend, seed_files)
        return backend

    def _local_build_template(
        self,
        name: str,
        digest: str,
        *,
        setup: Sequence[str],
        seed_files: Sequence[tuple[str, bytes]],
        rebuild: bool,
    ) -> SandboxTemplate:
        key = f"{TEMPLATE_PREFIX}-{name}-{digest}"
        root = root_for_host(key)
        marker = root.parent / f"{key}.ready"
        if marker.exists() and not rebuild:
            return SandboxTemplate(name, key, key, built=False)
        shutil.rmtree(root, ignore_errors=True)
        if setup:
            logger.warning("Local mode: template %r setup commands are not run (they would run on this machine)", name)
        backend = self._local_connect(key)
        for result in backend.upload_files(list(seed_files)):
            if result.error:
                raise SandboxError(f"template {name!r}: could not upload {result.path}: {result.error}")
        marker.touch()
        return SandboxTemplate(name, key, key, built=True)

    def _local_connect(self, sandbox_id: str) -> MicroVMSandbox:
        """Build a LocalTransport keyed off sandbox_id itself.

        Passing sandbox_id as `host` is what makes LocalTransport.connect()
        recreate/reattach to the SAME on-disk directory every time (see
        transport/local.py's _root_for_host) -- this is the whole trick that
        lets the local branch simulate pause/resume without a real backend.
        """
        transport = create_transport("local")
        transport.connect(host=sandbox_id, port=0, credentials=TransportCredentials(), timeout=None)
        return MicroVMSandbox(microvm_id=sandbox_id, transport=transport)

    # ------------------------------------------------------------------
    # shared internals
    # ------------------------------------------------------------------

    def _seed(self, backend: MicroVMSandbox, seed_files: list[tuple[str, bytes]]) -> None:
        """Upload `seed_files` into a freshly created sandbox.

        Per-file failures are logged, not raised -- a partially-seeded sandbox is still a
        usable sandbox.
        """
        results = backend.upload_files(seed_files)
        for (path, _content), result in zip(seed_files, results):
            if result.error:
                logger.warning(
                    "Failed to seed %s into the sandbox: %s", path, result.error
                )
