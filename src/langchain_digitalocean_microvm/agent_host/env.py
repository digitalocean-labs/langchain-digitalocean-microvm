"""The environment an agent host's Agent Server runs with.

On a real MicroVM the environment is passed once, at create time
(`environment` in POST /v2/microvms; immutable afterwards, never returned by
the API). In local mode LocalDevServerHost writes the same variables to the
host's ./.env with render_env_file().
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

from langchain_digitalocean_microvm.agent_host.models import HostSpec

TOKEN_ENV = "SUBAGENT_HOST_TOKEN"
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_host_env(spec: HostSpec, token: str, *, passthrough: Iterable[str] = ()) -> dict[str, str]:
    """Passthrough variables, then `spec.env`, then the host token and worker count (always last)."""
    env: dict[str, str] = {}
    for name in passthrough:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env.update(spec.env)
    env[TOKEN_ENV] = token
    env["N_JOBS_PER_WORKER"] = str(spec.n_jobs_per_worker)
    return env


def render_env_file(env: Mapping[str, str]) -> str:
    """Render `env` as a dotenv file (KEY="value", one per line).

    Keys are sorted for a stable file; values are double-quoted with `\\`
    and `"` escaped. Newlines in values and non-identifier keys are rejected
    rather than silently producing a file that would be misparsed.
    """
    lines: list[str] = []
    for key in sorted(env):
        if not _ENV_KEY.match(key):
            raise ValueError(f"Invalid environment variable name {key!r}")
        value = str(env[key])
        if "\n" in value or "\r" in value:
            raise ValueError(f"Environment variable {key!r} must not contain newlines")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{escaped}"')
    return "\n".join(lines) + ("\n" if lines else "")
