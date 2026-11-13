"""Live benchmark of the MicroVM primitives this package builds on (one MicroVM at a time).

Measures, `--runs` times each (default 5), through the same code paths the backend uses:

  - api_get         one control-plane round trip (`GET /microvms/{id}`): the client-to-API latency
  - create          `get_or_create` from the image until the sandbox answers (create, running, endpoint,
                    the server's token set over exec)
  - endpoint_call   `backend.execute("true")` over the MicroVM's HTTPS endpoint (`--calls` per run)
  - exec_call       one command through the exec API (`--calls` per run)
  - pause           `lifecycle.pause(backend)` until the MicroVM reads `paused`
  - resume          `lifecycle.resume(...)` until the sandbox answers again
  - delete          `lifecycle.delete`
  - template_build  once: temporary MicroVM, checkpoint, delete the MicroVM
  - clone           `get_or_create(template=...)` until the sandbox answers

Numbers depend on where the client runs (each request pays the client-to-region round trip), so run it
from a machine in the MicroVMs region for the service's own figures and from your laptop for the
client's view, and name the machine with `--client-label`. Model time is not measured.

Refuses to start while MicroVMs of its namespace (DO_MICROVM_NAMESPACE) exist. Other MicroVMs and
checkpoints of the team are only counted, never touched: it deletes only what it created (named
`bench-<run id>-...`), in a `finally` block. Tokens come from .env (DO_MICROVM_API_TOKEN,
DO_MICROVM_ACCESS_TOKEN); results go to ~/.langchain_digitalocean_microvm/probe/<date>/bench-<label>.json.

    uv run python scripts/bench.py --client-label laptop [--runs 5] [--calls 20] [--image IMAGE]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import secrets
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from _env import add_env_file_argument, env_namespace, load_env, refuse_if_owned

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle
from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.status import MicroVMStatus

GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


def summarize(samples: list[float]) -> dict[str, float | int]:
    """n, min, p50, p95, max in seconds (nearest rank: with few samples p95 is the maximum)."""
    ordered = sorted(samples)

    def rank(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {"n": len(ordered), "min": round(ordered[0], 3), "p50": round(rank(0.50), 3),
            "p95": round(rank(0.95), 3), "max": round(ordered[-1], 3)}


def timed(samples: list[float], fn: Any, *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        samples.append(time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="default: DO_MICROVM_IMAGE")
    parser.add_argument("--runs", type=int, default=5, help="MicroVM cycles and clones (default 5)")
    parser.add_argument("--calls", type=int, default=20, help="endpoint and exec calls per cycle (default 20)")
    parser.add_argument("--client-label", default="unlabelled",
                        help="where this runs, e.g. laptop or droplet-mkc1 (goes into the result and its file name)")
    add_env_file_argument(parser)
    args = parser.parse_args()
    if args.runs < 1 or args.calls < 1:
        parser.error("--runs and --calls must be at least 1")

    env = load_env(args.env_file)
    token, access = env.get("DO_MICROVM_API_TOKEN"), env.get("DO_MICROVM_ACCESS_TOKEN")
    image = args.image or env.get("DO_MICROVM_IMAGE")
    if not token or not access or not image:
        print("needs DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN and an image", file=sys.stderr)
        return 2
    client = MicroVMClient(token)
    if refuse_if_owned(client, env_namespace(env)):
        return 3

    run_id = secrets.token_hex(3)
    config = MicroVMConfig(transport="endpoint", api_token=token, access_token=access, image=image,
                           idle_timeout="10m", ready_timeout=300.0)
    lifecycle = MicroVMLifecycle(config)
    team_before = {"microvms": len([s for s in client.list() if s.status not in GONE]),
                   "checkpoints": len(client.list_checkpoints())}
    metrics: dict[str, list[float]] = {name: [] for name in (
        "api_get", "create", "endpoint_call", "exec_call", "pause", "resume", "first_call_after_resume",
        "delete", "template_build", "clone", "delete_clone")}
    results: dict[str, Any] = {
        "client_label": args.client_label, "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": metadata.version("langchain-digitalocean-microvm"), "python": platform.python_version(),
        "platform": platform.platform(), "region": config.region, "size": str(config.size), "image": image,
        "run_id": run_id, "runs": args.runs, "calls": args.calls, "team_before": team_before}
    sandboxes: list[str] = []
    templates: list[Any] = []
    t0 = time.monotonic()

    def say(message: str) -> None:
        print(f"[{time.monotonic() - t0:6.1f}s] {message}", flush=True)

    say(f"client {args.client_label!r}, region {config.region}, size {config.size}, team before {team_before}")
    code = 0
    try:
        for i in range(args.runs):
            sandbox_id = f"bench-{run_id}-{i}"
            sandboxes.append(sandbox_id)
            backend = timed(metrics["create"], lifecycle.get_or_create, sandbox_id=sandbox_id)
            microvm_id = backend.id
            backend.execute("true")  # not timed: the first call also pays one-off connection setup
            for _ in range(args.calls):
                timed(metrics["api_get"], client.get, microvm_id)
            for _ in range(args.calls):
                timed(metrics["endpoint_call"], backend.execute, "true")
            for _ in range(args.calls):
                timed(metrics["exec_call"], client.exec, microvm_id, ["true"])
            state = timed(metrics["pause"], lifecycle.pause, backend)
            if state is None or state.status != MicroVMStatus.PAUSED:
                raise RuntimeError(f"cycle {i}: the MicroVM did not read paused after the pause")
            backend = timed(metrics["resume"], lifecycle.resume, sandbox_id=sandbox_id)
            timed(metrics["first_call_after_resume"], backend.execute, "true")
            backend.close()
            timed(metrics["delete"], lifecycle.delete, sandbox_id=sandbox_id)
            sandboxes.remove(sandbox_id)
            say(f"cycle {i + 1}/{args.runs}: create {metrics['create'][-1]:.1f}s, pause {metrics['pause'][-1]:.1f}s, "
                f"resume {metrics['resume'][-1]:.1f}s, endpoint call p50 "
                f"{sorted(metrics['endpoint_call'][-args.calls:])[args.calls // 2]:.2f}s")

        template = timed(metrics["template_build"], lifecycle.build_template, f"bench-{run_id}",
                         seed_files=[("/opt/bench/hello.txt", b"hello\n")], rebuild=True)
        templates.append(template)
        say(f"template built in {metrics['template_build'][-1]:.1f}s")
        for i in range(args.runs):
            clone_id = f"bench-{run_id}-clone-{i}"
            sandboxes.append(clone_id)
            clone = timed(metrics["clone"], lifecycle.get_or_create, sandbox_id=clone_id, template=template)
            if clone.download_files(["/opt/bench/hello.txt"])[0].content != b"hello\n":
                raise RuntimeError(f"clone {i}: the template's file is missing")
            clone.close()
            timed(metrics["delete_clone"], lifecycle.delete, sandbox_id=clone_id)
            sandboxes.remove(clone_id)
            say(f"clone {i + 1}/{args.runs}: {metrics['clone'][-1]:.1f}s")
    except Exception as exc:  # noqa: BLE001
        code = 1
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"failed: {results['error']}", file=sys.stderr)
    finally:
        for sandbox_id in sandboxes:
            try:
                lifecycle.delete(sandbox_id=sandbox_id)
            except Exception as exc:  # noqa: BLE001
                print(f"delete {sandbox_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        for template in templates:
            lifecycle.delete_template(template)
        leftovers = [s for s in client.list() if f"bench-{run_id}" in s.name and s.status not in GONE]
        for state in leftovers:  # only this run's own names
            client.delete(state.id)
        results["leftovers_deleted"] = [s.id for s in leftovers]
        results["team_after"] = {"microvms": len([s for s in client.list() if s.status not in GONE]),
                                 "checkpoints": len(client.list_checkpoints())}
        if results["team_after"] != team_before:
            print(f"note: the team differs from before ({team_before} -> {results['team_after']}); "
                  "other users' changes, or a leftover of this run to check", file=sys.stderr)
        results["seconds"] = {name: summarize(values) for name, values in metrics.items() if values}
        results["samples"] = {name: [round(v, 3) for v in values] for name, values in metrics.items() if values}
        print(f"\n{'primitive':<26}{'n':>4}{'min':>9}{'p50':>9}{'p95':>9}{'max':>9}   (seconds)")
        for name, row in results["seconds"].items():
            print(f"{name:<26}{row['n']:>4}{row['min']:>9.2f}{row['p50']:>9.2f}{row['p95']:>9.2f}{row['max']:>9.2f}")
        out = Path.home() / ".langchain_digitalocean_microvm" / "probe" / datetime.now(UTC).date().isoformat()
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"bench-{args.client_label}.json"
        path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"\nresults: {path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
