"""Warm start for agent hosts: golden checkpoints and the clone's token install.

Warm start (config.warm_start, DO_MICROVM_WARM_START=1): new hosts are
created from a checkpoint of an already-booted host instead of from the image
(about 6 s instead of ~14 s). One "golden" checkpoint per distinct host
configuration (image, graph, environment, size, outbound policy; named by a
hash of them) is built on first use -- boot a MicroVM, wait for GET /ok,
checkpoint it, delete the VM (checkpoints do not count against the team's
MicroVM limit). When the host's outbound policy has credentials the VM is
kept, paused: the service deletes stored credentials with the MicroVM, and
its checkpoints cannot be restored afterwards. A golden checkpoint whose VM
is gone is deleted and built again. A clone
inherits the golden VM's environment, token included, so right after
creating one the manager writes the clone's own token to the token file
(`token_file`, via exec, once the clone's Agent Server answers GET /ok); the
template's auth.py prefers that file. Any failure along the warm path falls
back to a cold create.

WarmStart is a helper owned by one MicroVMAgentHost; it uses that host's
client, config, token factory and readiness probe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import TYPE_CHECKING

from langchain_digitalocean_microvm.agent_host.env import build_host_env
from langchain_digitalocean_microvm.agent_host.models import (
    HOST_MANAGED_TAG,
    HostHandle,
    HostSpec,
    HostStatus,
    host_graph_tag,
    host_id_tag,
)
from langchain_digitalocean_microvm.exceptions import (
    HostNotReadyError,
    MicroVMClientError,
    MicroVMForbiddenError,
    MicroVMNotFoundError,
    SandboxError,
)
from langchain_digitalocean_microvm.models import Checkpoint, MicroVMSize, MicroVMState
from langchain_digitalocean_microvm.namespace import namespace_tags, namespaced_prefix
from langchain_digitalocean_microvm.reconcile import checkpoint_restorable

if TYPE_CHECKING:
    from langchain_digitalocean_microvm.agent_host.host import MicroVMAgentHost

logger = logging.getLogger(__name__)

GOLDEN_PREFIX = "microvm-golden"
GOLDEN_TAG = "microvm-golden"


class WarmStart:
    """Golden checkpoints, clones and pending token installs for one MicroVMAgentHost."""

    def __init__(self, host: MicroVMAgentHost) -> None:
        self._host = host
        # Hosts created from a checkpoint whose token is not yet in the token file.
        self._token_pending: set[str] = set()
        self._golden_locks: dict[str, threading.Lock] = {}
        self._golden_locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # golden checkpoints and clones
    # ------------------------------------------------------------------

    def golden_name(self, spec: HostSpec, image: str, size: MicroVMSize) -> str:
        """Checkpoint name for this host configuration; changes whenever the env would."""
        config = self._host.config
        env = build_host_env(spec, "", passthrough=config.subagent_env_passthrough)
        env_digest = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(env.items())}  # no secrets in the hash input log
        policy = self._host.egress_for(spec)
        material = json.dumps(
            {"image": image, "graph": spec.graph_id, "env": env_digest, "size": size.slug, "port": config.agent_port,
             "egress": policy.digest((config.api_token or "").encode()) if policy is not None else None},
            sort_keys=True,
        )
        return f"{namespaced_prefix(GOLDEN_PREFIX, config.namespace)}-{hashlib.sha256(material.encode()).hexdigest()[:12]}"

    def golden_checkpoint(self, spec: HostSpec, image: str, size: MicroVMSize) -> Checkpoint | None:
        """The AVAILABLE golden checkpoint for this configuration, building it if needed; None to go cold."""
        name = self.golden_name(spec, image, size)
        with self._golden_locks_guard:
            lock = self._golden_locks.setdefault(name, threading.Lock())
        with lock:
            client = self._host._client_or_raise()
            try:
                existing = [c for c in client.list_checkpoints() if c.name == name and c.available]
            except MicroVMClientError:
                logger.warning("Could not list checkpoints; starting %s cold", name, exc_info=True)
                return None
            for checkpoint in existing:
                if checkpoint_restorable(client, checkpoint):
                    return checkpoint
                logger.warning("Warm-start checkpoint %s: its MicroVM is gone with its credentials; rebuilding", name)
                self._delete_golden(checkpoint)
            try:
                return self._build_golden(name, spec, image, size)
            except (SandboxError, MicroVMClientError, TimeoutError) as exc:
                logger.warning("Could not build warm-start checkpoint %s (%s); starting cold", name, exc)
                return None

    def _build_golden(self, name: str, spec: HostSpec, image: str, size: MicroVMSize) -> Checkpoint | None:
        host = self._host
        config = host.config
        client = host._client_or_raise()
        token = host._token_factory()
        env = build_host_env(spec, token, passthrough=config.subagent_env_passthrough)
        policy = host.egress_for(spec)
        keep_origin = False
        started = host._monotonic()
        try:
            state = client.create(
                name=name, oci_ref=image, region=config.region, size=size, http_port=config.agent_port,
                http_protocol="http", environment=env, idle_timeout=config.idle_timeout,
                tags=[GOLDEN_TAG, *namespace_tags(config.namespace)],
                egress_policy=policy.for_image(image) if policy is not None else None,
            )
        except MicroVMForbiddenError as exc:
            raise host._explain_forbidden(exc) from exc
        try:
            state = host._wait_for_url(state.id)
            probe_handle = HostHandle(
                host_id=name, microvm_id=state.id, graph_id=spec.graph_id, url=state.url or "", token=token,
                status=HostStatus.BOOTING, booted_at="",
            )
            host.wait_ready(probe_handle)  # the server must be up before the snapshot
            checkpoint = client.create_checkpoint(state.id, name=name)
            checkpoint = client.wait_for_checkpoint(checkpoint.id, timeout=config.ready_timeout)
            logger.info("Built warm-start checkpoint %s in %.0fs", name, host._monotonic() - started)
            if not checkpoint.available:
                return None
            # Deleting the VM would delete its stored credentials, and the checkpoint could no longer be restored.
            keep_origin = bool(policy is not None and policy.credentials)
            if keep_origin:
                try:
                    client.pause(state.id, timeout=config.ready_timeout)
                except MicroVMClientError:
                    logger.warning("Could not pause the MicroVM of warm-start checkpoint %s", name, exc_info=True)
            return checkpoint
        finally:
            if not keep_origin:
                host._delete_quietly(state.id)  # frees the slot; the checkpoint outlives it

    def _delete_golden(self, checkpoint: Checkpoint) -> None:
        """Delete a golden checkpoint and the MicroVM it was taken from, if that still exists."""
        client = self._host._client_or_raise()
        try:
            client.delete_checkpoint(checkpoint.id)
        except MicroVMClientError:
            logger.warning("Could not delete warm-start checkpoint %s", checkpoint.name, exc_info=True)
        if checkpoint.microvm_id is None:
            return
        try:
            origin = client.get(checkpoint.microvm_id)
            if origin.name.startswith(GOLDEN_PREFIX):
                client.delete(origin.id)
        except MicroVMNotFoundError:
            pass
        except MicroVMClientError:
            logger.warning("Could not delete the MicroVM of warm-start checkpoint %s", checkpoint.name, exc_info=True)

    def create_from_checkpoint(self, name: str, checkpoint: Checkpoint, host_id: str, spec: HostSpec) -> MicroVMState | None:
        """Create host `host_id` as a clone of `checkpoint`; None (after logging) to go cold."""
        host = self._host
        config = host.config
        client = host._client_or_raise()
        try:
            return client.create(
                name=name, checkpoint_id=checkpoint.id, idle_timeout=config.idle_timeout,
                tags=[HOST_MANAGED_TAG, host_id_tag(host_id), host_graph_tag(spec.graph_id),
                      *namespace_tags(config.namespace)],
            )
        except MicroVMForbiddenError as exc:
            raise host._explain_forbidden(exc) from exc
        except MicroVMClientError:
            logger.warning("Creating host %s from checkpoint %s failed; starting cold", host_id, checkpoint.name, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # the clone's own token
    # ------------------------------------------------------------------

    def mark_token_pending(self, host_id: str) -> None:
        """Record that `host_id` is a clone whose own token is not yet in the token file."""
        self._token_pending.add(host_id)

    def is_token_pending(self, host_id: str) -> bool:
        return host_id in self._token_pending

    def install_token(self, state: MicroVMState, host_id: str) -> None:
        """Write the clone's own token to the token file (the inherited env token then stops working)."""
        host = self._host
        script = 'umask 077; mkdir -p "$(dirname "$1")" && printf %s "$2" > "$1.tmp" && mv "$1.tmp" "$1"'
        try:
            result = host._client_or_raise().exec(state.id, ["sh", "-c", script, "_", host.token_file, host._tokens[host_id]])
        except MicroVMClientError as exc:
            raise HostNotReadyError(f"Could not install the token on host {host_id}: {exc}") from exc
        if result.exit_code != 0:
            raise HostNotReadyError(f"Could not install the token on host {host_id}: {result.stderr.strip()}")
        self._token_pending.discard(host_id)
