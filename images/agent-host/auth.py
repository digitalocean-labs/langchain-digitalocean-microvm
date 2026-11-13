# images/agent-host/auth.py
"""Per-host token check for the in-MicroVM Agent Server.

Registered in langgraph.json as:

    "auth": {"path": "./auth.py:auth", "disable_studio_auth": true}

Every Agent Protocol request (threads, runs, assistants, store, ...) must carry
the host's token in `x-api-key`. The token is SUBAGENT_HOST_TOKEN, unless the
token file ($SUBAGENT_HOST_TOKEN_FILE, default /work/.microvm-agent/token)
exists, which then wins (hosts cloned from a checkpoint, see expected_token). The token is written to the env file by the
MicroVM's environment (set at create time by MicroVMAgentHost) or by
LocalDevServerHost's ./.env; the gateway is the only
other party that knows it. See docs/design.md.

Two facts about langgraph-api that shaped this file (verified against 0.11.x):

1. `disable_studio_auth: true` is NOT optional. Under `langgraph dev` the
   server sets LANGSMITH_LANGGRAPH_API_VARIANT=local_dev, and with the default
   `disable_studio_auth: false` ANY request carrying `x-auth-scheme: langsmith`
   skips the custom handler and is accepted as a Studio user. The deepagents
   async-subagent middleware sends exactly that header by default, so without
   this flag the token check would never run for the supervisor's own traffic
   (or anyone else who adds the header).
2. The meta routes `/ok`, `/`, `/info`, `/metrics`, `/docs`, `/openapi.json`
   are mounted OUTSIDE the auth middleware and answer without a token. That is
   fine for readiness probes (GET /ok) but means queue statistics on /metrics
   are readable by anyone who can reach the endpoint (on a MicroVM: any token
   of the team with microvm:access).

`x-api-key` rather than `Authorization: Bearer` because that is the header the
LangGraph SDK sends when a client is built with `api_key=` (and the one it
reserves, so nothing else can set it by accident).
"""

from __future__ import annotations

import asyncio
import hmac
import os
from pathlib import Path

from langgraph_sdk import Auth

TOKEN_ENV = "SUBAGENT_HOST_TOKEN"
TOKEN_FILE_ENV = "SUBAGENT_HOST_TOKEN_FILE"
DEFAULT_TOKEN_FILE = "/work/.microvm-agent/token"

auth = Auth()

_cache: dict[str, object] = {"mtime": None, "token": ""}


def _token_file() -> Path:
    return Path(os.environ.get(TOKEN_FILE_ENV, DEFAULT_TOKEN_FILE))


def _file_token() -> str:
    """The token in the token file, re-read only when the file changes; "" if absent."""
    path = _token_file()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return ""
    if mtime != _cache["mtime"]:
        _cache["token"] = path.read_text().strip()
        _cache["mtime"] = mtime
    return str(_cache["token"])


async def expected_token() -> str:
    """The token file wins over the environment.

    A MicroVM created from a checkpoint inherits the checkpointed VM's
    environment, token included, and environments cannot be changed. So the
    host manager writes each clone's own token to the token file (via the
    exec API) right after creating it; from then on the inherited token is
    no longer accepted.
    """
    return await asyncio.to_thread(_file_token) or os.environ.get(TOKEN_ENV, "")


@auth.authenticate
async def authenticate(headers: dict[bytes, bytes] | None) -> str:
    expected = await expected_token()
    presented = (headers or {}).get(b"x-api-key", b"").decode("utf-8", "replace")
    if not expected or not hmac.compare_digest(presented, expected):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="invalid or missing host token"
        )
    return "gateway"
