"""Namespaces: which MicroVMs, checkpoints and store files belong to which owner or deployment.

A team can be shared by several users or deployments of this package. Setting
`DO_MICROVM_NAMESPACE` (`MicroVMConfig.namespace`) gives each its own namespace:

- every MicroVM it creates carries the tag `microvm-namespace:<ns>`;
- the namespace is part of the names of its sandboxes, agent hosts and warm-start
  checkpoints (`microvm-sandbox-ns-<ns>-<id>`), so equal ids never meet;
- its gateway store is a separate file (`gateway-<ns>.sqlite`).

Without a namespace nothing changes: names and the store file are as before, and a
MicroVM with no namespace tag belongs to the default namespace. Clean-up (`list_owned`,
scripts/delete_all.py, scripts/gc_microvms.py) touches one namespace unless told otherwise.
Sandbox templates are content-addressed and shared by every namespace.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from langchain_digitalocean_microvm.client import MANAGED_TAG, MicroVMClient
from langchain_digitalocean_microvm.models import MicroVMState
from langchain_digitalocean_microvm.status import MicroVMStatus

NAMESPACE_TAG_PREFIX = "microvm-namespace:"
_NAME_MARK = "ns"
# Lowercase letters and digits only: no hyphen, so a namespace can be read back out of a name.
_VALID = re.compile(r"[a-z0-9]{1,16}")
_GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


def validate_namespace(namespace: str | None) -> str | None:
    """`namespace` stripped, or None for the default namespace; ValueError unless 1-16 lowercase letters/digits."""
    if namespace is None or not namespace.strip():
        return None
    value = namespace.strip()
    if not _VALID.fullmatch(value):
        raise ValueError(f"namespace must be 1-16 lowercase letters or digits, got {namespace!r}")
    return value


def namespace_tags(namespace: str | None) -> list[str]:
    """The tag to add to a MicroVM of `namespace` (none for the default namespace)."""
    return [f"{NAMESPACE_TAG_PREFIX}{namespace}"] if namespace else []


def namespace_from_tags(tags: Iterable[str]) -> str | None:
    """The namespace a MicroVM's tags record; None: the default namespace."""
    for tag in tags:
        if tag.startswith(NAMESPACE_TAG_PREFIX):
            return tag[len(NAMESPACE_TAG_PREFIX):] or None
    return None


def namespaced_prefix(prefix: str, namespace: str | None) -> str:
    """`prefix`, or `<prefix>-ns-<namespace>`: the name prefix of one namespace's MicroVMs and checkpoints."""
    return f"{prefix}-{_NAME_MARK}-{namespace}" if namespace else prefix


def namespace_from_name(name: str, prefixes: Iterable[str]) -> str | None:
    """The namespace in a MicroVM or checkpoint name starting with one of `prefixes`; None: default or unknown."""
    for prefix in prefixes:
        match = re.fullmatch(rf"{re.escape(prefix)}-{_NAME_MARK}-([a-z0-9]{{1,16}})(?:-.*)?", name)
        if match:
            return match.group(1)
    return None


def in_scope(found: str | None, namespace: str | None, *, all_namespaces: bool = False) -> bool:
    """Whether a resource of namespace `found` is in the scope of `namespace` (or of every namespace)."""
    return all_namespaces or found == namespace


def list_owned(
    client: MicroVMClient,
    *,
    namespace: str | None,
    all_namespaces: bool = False,
    tag: str = MANAGED_TAG,
) -> list[MicroVMState]:
    """Live MicroVMs carrying `tag` that belong to `namespace` (None: the default one), or to any with `all_namespaces`."""
    return [
        state for state in client.list(tag=tag)
        if state.status not in _GONE and in_scope(namespace_from_tags(state.tags), namespace, all_namespaces=all_namespaces)
    ]


def namespaces_present(states: Iterable[MicroVMState]) -> set[str | None]:
    """The namespaces (None: default) the MicroVMs belong to."""
    return {namespace_from_tags(state.tags) for state in states}
