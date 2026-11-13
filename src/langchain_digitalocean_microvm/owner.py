"""Owners: which MicroVMs were created by this user, as opposed to someone else on the same team.

Every MicroVM `MicroVMClient.create` makes carries the tag `microvm-owner:<id>`. The id is a random
string kept in a file in the package's state directory, so it is stable for one user's account on one
machine and differs between users. Clean-up (scripts/delete_all.py, scripts/gc_microvms.py) and the
examples' `prepare.py --clean` touch only the MicroVMs that carry this user's id unless told otherwise.

MicroVMs created before the tag existed carry no owner tag; they count as nobody's and are selected
only with `--all-owners` (or `--all-namespaces`).
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterable

logger = logging.getLogger(__name__)

OWNER_TAG_PREFIX = "microvm-owner:"
_FILE = "owner-id"
_cached: str | None = None


def owner_id() -> str:
    """This user's owner id: read from the state directory, created on first use (10 hex characters)."""
    global _cached
    if _cached is not None:
        return _cached
    from langchain_digitalocean_microvm.agent_host.store import (
        state_dir,  # late: that package imports the client
    )

    path = state_dir() / _FILE
    try:
        value = path.read_text().strip()
    except OSError:
        value = ""
    if len(value) != 10 or any(c not in "0123456789abcdef" for c in value):
        value = secrets.token_hex(5)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value + "\n")
        except OSError:
            logger.warning("could not write %s; this process uses a temporary owner id", path)
    _cached = value
    return value


def owner_tag() -> str:
    """The tag that marks a MicroVM as created by this user."""
    return f"{OWNER_TAG_PREFIX}{owner_id()}"


def owner_from_tags(tags: Iterable[str]) -> str | None:
    """The owner id a MicroVM's tags record; None: created before owner tags existed."""
    for tag in tags:
        if tag.startswith(OWNER_TAG_PREFIX):
            return tag[len(OWNER_TAG_PREFIX):] or None
    return None


def is_mine(tags: Iterable[str]) -> bool:
    """Whether a MicroVM with these tags was created by this user."""
    return owner_from_tags(tags) == owner_id()


def in_owner_scope(tags: Iterable[str], *, all_owners: bool = False) -> bool:
    """Whether a MicroVM is in the scope of this user's clean-up (or of every owner's with `all_owners`)."""
    return all_owners or is_mine(tags)
