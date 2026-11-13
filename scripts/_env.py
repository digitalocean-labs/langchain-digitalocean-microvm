"""Shared settings loading for the scripts in this directory.

Every script reads the same things the same way: the repository's .env, with any DO_MICROVM_* variable already set in the shell
taking precedence. Values are returned, never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def add_env_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help=f"default: {DEFAULT_ENV_FILE}")


def load_env(env_file: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """.env values, overridden by DO_MICROVM_* variables from the environment."""
    values = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    values.update({k: v for k, v in os.environ.items() if k.startswith("DO_MICROVM_")})
    return values


def env_namespace(env: dict[str, str]) -> str | None:
    """DO_MICROVM_NAMESPACE from `load_env`'s values, validated; None is the default namespace."""
    from langchain_digitalocean_microvm.namespace import validate_namespace

    return validate_namespace(env.get("DO_MICROVM_NAMESPACE"))


def refuse_if_owned(client, namespace: str | None) -> bool:  # noqa: ANN001 - a MicroVMClient (or local)
    """True (after saying why on stderr) if live MicroVMs of `namespace` exist: live smoke tests start on none of their own.

    MicroVMs of other namespaces, or created otherwise, do not count; they are listed so the team limit can be judged.
    """
    from langchain_digitalocean_microvm.namespace import list_owned, namespace_from_tags
    from langchain_digitalocean_microvm.status import MicroVMStatus

    gone = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)
    owned = list_owned(client, namespace=namespace)
    if owned:
        which = f"namespace {namespace!r}" if namespace else "the default namespace"
        print(f"refusing to start: {len(owned)} MicroVM(s) of {which} exist:", file=sys.stderr)
        for state in owned:
            print(f"  {state.id} {state.name} {state.status.value}", file=sys.stderr)
        return True
    others = [s for s in client.list() if s.status not in gone]
    if others:
        print(f"note: the team also has {len(others)} MicroVM(s) outside this namespace; "
              f"they count against the team limit: {sorted({namespace_from_tags(s.tags) or '(default)' for s in others})}",
              file=sys.stderr)
    return False
