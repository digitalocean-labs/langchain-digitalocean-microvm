"""The incident team's tools: the investigators' and the verifier's run in their own MicroVMs, the commander's saves the report.

Investigator tools are bound to one subagent's sandbox. They fetch evidence on this side (the service's metrics over
HTTP, the commit list through the Action Gateway: credentials never enter a MicroVM), write it into the MicroVM's
/data/, and run a helper script there (sandbox/, seeded into the template under /tools/). The gateway's other read
tools are handed to the investigators as they are.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain.tools import tool

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGateway,
)
from langchain_digitalocean_microvm.transport.local import root_for_host

HERE = Path(__file__).resolve().parent


class FilesOnly(BackendProtocol):
    """A sandbox's files without its shell: what an investigator's file tools get.

    deepagents offers the model `execute` only on a backend that can run commands; this one cannot, so the agent has
    read, write, ls, grep and glob in its MicroVM and no shell. Its tools hold the sandbox itself and run their scripts.
    """

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def ls(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.ls(*args, **kwargs)

    async def als(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.als(*args, **kwargs)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.read(*args, **kwargs)

    async def aread(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aread(*args, **kwargs)

    def grep(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.grep(*args, **kwargs)

    async def agrep(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.agrep(*args, **kwargs)

    def glob(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.glob(*args, **kwargs)

    async def aglob(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aglob(*args, **kwargs)

    def write(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.write(*args, **kwargs)

    async def awrite(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.awrite(*args, **kwargs)

    def edit(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.edit(*args, **kwargs)

    async def aedit(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aedit(*args, **kwargs)

    def upload_files(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.upload_files(*args, **kwargs)

    async def aupload_files(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aupload_files(*args, **kwargs)

    def download_files(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.download_files(*args, **kwargs)

    async def adownload_files(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.adownload_files(*args, **kwargs)


async def _started(backend: Any) -> Any:
    """The MicroVM behind an on-demand sandbox, started if it is not yet (a local root must exist before use)."""
    return await backend.aget() if hasattr(backend, "aget") else backend


async def _put(backend: Any, path: str, content: str) -> None:
    real = await _started(backend)
    uploads = await asyncio.to_thread(real.upload_files, [(path, content.encode())])
    failed = [u for u in uploads if u.error]
    if failed:
        raise RuntimeError(f"could not write {path}: {failed[0].error}")


async def _script(backend: Any, name: str, args: list[str], *, local: bool) -> str:
    """Run /tools/<name> in the MicroVM; its JSON output, or a JSON error."""
    try:
        real = await _started(backend)
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        return json.dumps({"error": f"MicroVM could not be started: {exc}"})
    # local mode: this process's interpreter, without the local transport's path shim; the scripts resolve their paths
    python = f"env -u MICROVM_LOCAL_ROOT {shlex.quote(sys.executable)}" if local else "python3"
    path = await asyncio.to_thread(lambda: f"{root_for_host(real.id)}/tools/{name}") if local else f"/tools/{name}"
    result = await real.aexecute(" ".join([python, shlex.quote(path), *map(shlex.quote, args)]))
    if result.exit_code != 0:
        return json.dumps({"error": f"command failed ({result.exit_code}): {result.output.strip()[-600:]}"})
    return result.output.strip()


def metrics_tools(backend: Any, service: Any, *, local: bool) -> list:
    """`fetch_metrics` and `find_onset`, bound to the metrics investigator's MicroVM."""

    @tool
    async def fetch_metrics(minutes: int = 45) -> str:
        """Fetch the service's per-minute metrics (version, error rate, p95 latency) into your MicroVM at /data/metrics.json."""
        points = await asyncio.to_thread(service.metrics, min(max(minutes, 10), 120))
        await _put(backend, "/data/metrics.json", json.dumps(points))
        return f"Wrote {len(points)} points to /data/metrics.json ({points[0]['t']} to {points[-1]['t']})."

    @tool
    async def find_onset() -> str:
        """Analyse /data/metrics.json in your MicroVM: baseline, when the errors started, the peak, and the versions before and at the onset."""
        return await _script(backend, "series.py", [], local=local)

    return [fetch_metrics, find_onset]


def changes_tools(backend: Any, gateway: ActionGateway, *, local: bool) -> list:
    """`fetch_commits` and `build_timeline`, bound to the change investigator's MicroVM."""

    @tool
    async def fetch_commits(owner: str, repo: str) -> str:
        """List the repository's recent commits (through the Action Gateway) into your MicroVM at /data/commits.json."""
        answer = await asyncio.to_thread(gateway.call, "github_list_commits", {"owner": owner, "repo": repo})
        try:
            commits = json.loads(answer)
        except ValueError:
            return answer  # an error message from the gateway
        if isinstance(commits, dict):  # the catalog's tool wraps the list: {"commits": [...]}
            commits = commits.get("commits", [])
        await _put(backend, "/data/commits.json", json.dumps(commits))
        return f"Wrote {len(commits)} commits to /data/commits.json."

    @tool
    async def build_timeline(onset: str = "") -> str:
        """Order the commits in /data/commits.json by time; given the error onset (ISO time), name the last commit before it and the minutes between."""
        return await _script(backend, "timeline.py", [onset] if onset else [], local=local)

    return [fetch_commits, build_timeline]


def verifier_tools(backend: Any, service: Any, *, local: bool, every: float = 5.0) -> list:
    """`sample_health` and `judge_health`, bound to the verifier's MicroVM. It reads; it cannot change anything."""
    samples: list[dict[str, Any]] = []

    @tool
    async def sample_health(wait_seconds: int = 20) -> str:
        """Probe the service's /health every 5 seconds for up to 30 seconds and add the samples to /data/health.json in your MicroVM."""
        deadline = time.monotonic() + min(max(wait_seconds, 0), 30)
        while True:
            health = await asyncio.to_thread(service.health)
            await asyncio.to_thread(service.announce, "check")
            samples.append({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": health.get("status"),
                            "version": health.get("version"), "error_rate": health.get("error_rate")})
            if time.monotonic() + every > deadline:
                break
            await asyncio.sleep(every)
        await _put(backend, "/data/health.json", json.dumps(samples))
        return f"{len(samples)} samples in /data/health.json; the latest is {samples[-1]['status']} on {samples[-1]['version']}."

    @tool
    async def judge_health() -> str:
        """Analyse /data/health.json in your MicroVM: the verdict ("healthy" or "not_healthy"), the reason and the latest sample."""
        return await _script(backend, "verdict.py", [], local=local)

    return [sample_health, judge_health]


def commander_tools() -> list:
    """`save_postmortem`: the commander reads through the investigators and the verifier, and acts through the remediator."""

    @tool
    async def save_postmortem(title: str, body: str) -> str:
        """Save the postmortem (a title and a Markdown body: summary, timeline, root cause, what was done, follow-ups) as a file on this machine."""
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "postmortem"
        path = HERE / "outputs" / "postmortems" / f"{time.strftime('%Y%m%dT%H%M%S')}-{slug}.md"

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {title}\n\n{body}\n")

        await asyncio.to_thread(write)
        return f"Saved the postmortem to {path}."

    return [save_postmortem]
