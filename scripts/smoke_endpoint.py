"""Live smoke test: MicroVMLifecycle + EndpointTransport against real MicroVMs.

Refuses to start while MicroVMs of its namespace (DO_MICROVM_NAMESPACE) exist. Then, at most two MicroVMs at
a time, through the same code paths the deepagents backend uses:

  1. a sandbox from the sandbox image (scripts/build_sandbox_image.py; the
     server is its entrypoint): get_or_create with a seed file (the transport
     re-keys the server over exec) -> execute -> deepagents file ops ->
     1.5 MB and 40 MiB round trips (chunked) -> 3 MB of command output (read
     in pages) -> a 25 s command (longer than
     the endpoint's request budget: becomes a job; must run exactly once) ->
     a timed-out command is really killed -> 100 small commands, and how many
     control-plane API calls all of that cost -> pause -> get_or_create
     resumes with the filesystem intact -> delete;
  2. a template (build_template with a seed file) and a sandbox cloned from
     it: the clone inherits the template's token file, so the transport
     re-keys it -> seeded file present -> delete both;
  3. optionally (--fallback-image, e.g. an older sandbox image whose own
     process holds the HTTP port): the transport falls back to exec quickly.

Everything it creates is deleted in a `finally` block, and the team is
listed again at the end. Tokens come from .env (DO_MICROVM_API_TOKEN,
DO_MICROVM_ACCESS_TOKEN); results go to
~/.langchain_digitalocean_microvm/probe/<date>/smoke_endpoint.json.

    uv run python scripts/smoke_endpoint.py [--image IMAGE] [--fallback-image IMAGE]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from _env import (  # scripts/_env.py
    add_env_file_argument,
    env_namespace,
    load_env,
    refuse_if_owned,
)

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle
from langchain_digitalocean_microvm.client import MicroVMClient, api_calls
from langchain_digitalocean_microvm.status import MicroVMStatus

GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="default: DO_MICROVM_IMAGE")
    parser.add_argument("--fallback-image", help="an image whose own process holds port 8080 (step 3)")
    add_env_file_argument(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("langchain_digitalocean_microvm.transport").setLevel(logging.INFO)
    env = load_env(args.env_file)
    token, access = env.get("DO_MICROVM_API_TOKEN"), env.get("DO_MICROVM_ACCESS_TOKEN")
    image = args.image or env.get("DO_MICROVM_IMAGE")
    if not token or not access or not image:
        print("needs DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN and an image", file=sys.stderr)
        return 2
    client = MicroVMClient(token)
    namespace = env_namespace(env)
    if refuse_if_owned(client, namespace):
        return 3

    run_id = secrets.token_hex(3)
    config = MicroVMConfig(transport="endpoint", api_token=token, access_token=access, image=image,
                           idle_timeout="5m", ready_timeout=300.0)
    lifecycle = MicroVMLifecycle(config)
    results: dict[str, Any] = {"image": image, "run_id": run_id}
    sandboxes: list[str] = []
    templates: list[Any] = []
    t0 = time.monotonic()

    def step(name: str, value: Any) -> None:
        results[name] = value
        print(f"[{time.monotonic() - t0:6.1f}s] {name}: {json.dumps(value, default=str)[:300]}", flush=True)

    def control_calls() -> int:
        return sum(api_calls().values())

    code = 0
    try:
        # 1. sandbox from the image ------------------------------------------------
        sandbox_id = f"smoke-ep-{run_id}"
        sandboxes.append(sandbox_id)
        started, calls = time.monotonic(), control_calls()
        backend = lifecycle.get_or_create(sandbox_id=sandbox_id, seed_files=[("/tmp/smoke/seed.txt", b"seeded\n")])
        step("get_or_create", {"seconds": round(time.monotonic() - started, 1), "control_calls": control_calls() - calls,
                               "fallback": backend.transport.uses_fallback})  # type: ignore[attr-defined]
        microvm_id = backend.id
        calls = control_calls()

        r = backend.execute("uname -sm; id -un; python3 --version")
        step("environment", {"exit": r.exit_code, "output": r.output})
        step("seed_file", backend.download_files(["/tmp/smoke/seed.txt"])[0].content)
        path = "/tmp/smoke/notes.md"
        step("write", backend.write(path, "alpha\nbeta\n").error)
        step("read_has_beta", "beta" in (backend.read(path).file_data or {}).get("content", ""))
        step("edit_occurrences", backend.edit(path, "beta", "gamma").occurrences)
        step("ls_sees_file", any(e["path"] == path for e in (backend.ls("/tmp/smoke").entries or [])))
        grep = backend.grep("gamma", path="/tmp/smoke")
        step("grep", {"error": grep.error, "paths": [m["path"] for m in grep.matches or []]})

        for label, size in (("blob_1_5MB", 1_500_000), ("blob_40MiB", 40 * 1024 * 1024)):
            blob = os.urandom(size)
            started = time.monotonic()
            up = backend.upload_files([(f"/tmp/smoke/{label}.bin", blob)])[0]
            up_s = time.monotonic() - started
            started = time.monotonic()
            down = backend.download_files([f"/tmp/smoke/{label}.bin"])[0]
            down_s = time.monotonic() - started
            step(label, {"ok": up.error is None and down.content == blob, "upload_s": round(up_s, 1),
                         "download_s": round(down_s, 1), "upload_mib_s": round(size / 2**20 / up_s, 2),
                         "download_mib_s": round(size / 2**20 / down_s, 2)})
            backend.execute(f"rm -f /tmp/smoke/{label}.bin")

        started = time.monotonic()
        r = backend.execute("head -c 3000000 /dev/zero | tr '\\0' x")  # above the inline limit: read in pages
        step("large_output_3MB", {"ok": r.output == "x" * 3_000_000, "seconds": round(time.monotonic() - started, 1)})

        started = time.monotonic()
        r = backend.execute("echo run >> /tmp/smoke/long-runs; sleep 25; echo long-done", timeout=120)
        runs = backend.execute("wc -l < /tmp/smoke/long-runs").output.strip()
        step("long_command", {"stdout": r.output.strip(), "exit": r.exit_code, "runs": runs,
                              "seconds": round(time.monotonic() - started, 1)})

        r = backend.execute("echo begun; sleep 20; touch /tmp/smoke/not-killed", timeout=6)
        step("timeout_kill", {"exit": r.exit_code, "output_tail": r.output[-60:]})
        time.sleep(18)
        step("timeout_kill_verified", backend.execute("test -e /tmp/smoke/not-killed && echo SURVIVED || echo killed").output)

        latencies = []
        for _ in range(100):
            started = time.monotonic()
            backend.execute("true")
            latencies.append(time.monotonic() - started)
        latencies.sort()
        step("small_commands_100", {"p50_s": round(statistics.median(latencies), 3),
                                    "p95_s": round(latencies[94], 3), "max_s": round(latencies[-1], 3)})
        step("control_calls_for_all_of_the_above", control_calls() - calls)

        backend.close()
        started = time.monotonic()
        state = lifecycle.pause(sandbox_id=sandbox_id)
        step("paused", {"state": state.status if state else None, "seconds": round(time.monotonic() - started, 1)})
        started = time.monotonic()
        backend = lifecycle.get_or_create(sandbox_id=sandbox_id)
        step("resumed", {"same_vm": backend.id == microvm_id, "seconds": round(time.monotonic() - started, 1),
                         "fallback": backend.transport.uses_fallback})  # type: ignore[attr-defined]
        step("filesystem_after_resume", (backend.read(path).file_data or {}).get("content"))
        backend.close()
        lifecycle.delete(sandbox_id=sandbox_id)
        sandboxes.remove(sandbox_id)

        # 2. template + clone --------------------------------------------------------
        started = time.monotonic()
        template = lifecycle.build_template(f"smoke-ep-{run_id}", seed_files=[("/opt/tpl/hello.txt", b"from template\n")])
        templates.append(template)
        step("template_built", {"seconds": round(time.monotonic() - started, 1), "built": template.built})
        clone_id = f"smoke-ep-{run_id}-clone"
        sandboxes.append(clone_id)
        started = time.monotonic()
        clone = lifecycle.get_or_create(sandbox_id=clone_id, template=template)
        step("clone", {"seconds": round(time.monotonic() - started, 1),
                       "fallback": clone.transport.uses_fallback,  # type: ignore[attr-defined]
                       "template_file": clone.download_files(["/opt/tpl/hello.txt"])[0].content})
        clone.close()
        lifecycle.delete(sandbox_id=clone_id)
        sandboxes.remove(clone_id)
        lifecycle.delete_template(template)
        templates.remove(template)

        # 3. fallback ------------------------------------------------------------------
        if args.fallback_image:
            old = MicroVMLifecycle(MicroVMConfig(transport="endpoint", api_token=token, access_token=access,
                                                 image=args.fallback_image, idle_timeout="5m", ready_timeout=300.0))
            old_id = f"smoke-ep-{run_id}-old"
            sandboxes.append(old_id)
            started = time.monotonic()
            legacy = old.get_or_create(sandbox_id=old_id)
            step("fallback_image", {"seconds": round(time.monotonic() - started, 1),
                                    "fallback": legacy.transport.uses_fallback,  # type: ignore[attr-defined]
                                    "works": legacy.execute("echo via-exec").output.strip()})
            legacy.close()
            old.delete(sandbox_id=old_id)
            sandboxes.remove(old_id)
    except Exception as exc:  # noqa: BLE001 - report and still clean up
        step("error", f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        for sandbox_id in sandboxes:
            try:
                lifecycle.delete(sandbox_id=sandbox_id)
            except Exception as exc:  # noqa: BLE001
                step(f"delete_error_{sandbox_id}", f"{type(exc).__name__}: {exc}")
        for template in templates:
            lifecycle.delete_template(template)
        leftovers = [s for s in client.list() if run_id in s.name and s.status not in GONE]
        for state in leftovers:
            client.delete(state.id)
        step("leftovers_deleted", [s.id for s in leftovers])
        cps = [c.name for c in client.list_checkpoints() if run_id in c.name]
        step("team_after", {"microvms": len([s for s in client.list() if s.status not in GONE]),
                            "checkpoints_of_this_run": cps})
        out = Path.home() / ".langchain_digitalocean_microvm" / "probe" / date.today().isoformat()
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_endpoint.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
