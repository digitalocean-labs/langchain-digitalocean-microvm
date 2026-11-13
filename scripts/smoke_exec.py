"""Live smoke test: MicroVMLifecycle + ExecTransport against the real MicroVMs API.

Creates ONE MicroVM (sandbox_id `smoke-<random>`) from a public image that
keeps running and ships sh, coreutils and python3, then checks, through the
same code paths the deepagents backend uses:

  get_or_create with seed files -> execute -> deepagents file ops (write/read/
  edit/ls/grep) -> 1.5 MB upload/download -> a command that outlives the
  inline wait (detach + poll) -> a timed-out command is really killed ->
  pause -> get_or_create resumes with the filesystem intact -> delete.

Everything it creates is deleted in a `finally` block. The control-plane
token comes from .env (DO_MICROVM_API_TOKEN); results go to
~/.langchain_digitalocean_microvm/probe/<date>/smoke_exec.json.

    uv run python \\
        scripts/smoke_exec.py [--image IMAGE --http-port PORT]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from _env import add_env_file_argument, load_env  # scripts/_env.py

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle
from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.status import MicroVMStatus
from langchain_digitalocean_microvm.transport.base import TransportCredentials
from langchain_digitalocean_microvm.transport.exec import (
    TIMEOUT_EXIT_CODE,
    ExecTransport,
)

DEFAULT_IMAGE = "docker.io/kennethreitz/httpbin:latest"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--http-port", type=int, default=80)
    add_env_file_argument(parser)
    args = parser.parse_args()
    token = load_env(args.env_file).get("DO_MICROVM_API_TOKEN")
    if not token:
        print("DO_MICROVM_API_TOKEN not set", file=sys.stderr)
        return 2

    sandbox_id = f"smoke-{secrets.token_hex(3)}"
    config = MicroVMConfig(transport="exec", api_token=token, image=args.image, http_port=args.http_port,
                           idle_timeout="5m", ready_timeout=300.0)
    lifecycle = MicroVMLifecycle(config)
    client = MicroVMClient(token)
    results: dict[str, Any] = {"sandbox_id": sandbox_id, "image": args.image}
    t0 = time.monotonic()

    def step(name: str, value: Any) -> None:
        results[name] = value
        print(f"[{time.monotonic() - t0:6.1f}s] {name}: {json.dumps(value, default=str)[:300]}", flush=True)

    code = 0
    try:
        started = time.monotonic()
        backend = lifecycle.get_or_create(sandbox_id=sandbox_id, seed_files=[("/tmp/smoke/seed.txt", b"seeded\n")])
        step("get_or_create_seconds", round(time.monotonic() - started, 1))
        step("microvm_id", backend.id)

        r = backend.execute("uname -sm; id -un; for c in bash python3 base64 dd pkill grep; do command -v $c >/dev/null && echo has:$c || echo missing:$c; done; grep --version | head -1")
        step("environment", {"exit": r.exit_code, "output": r.output})
        step("seed_file", backend.download_files(["/tmp/smoke/seed.txt"])[0].content)

        path = "/tmp/smoke/notes.md"
        step("write", backend.write(path, "alpha\nbeta\n").error)
        step("read_has_beta", "beta" in (backend.read(path).file_data or {}).get("content", ""))
        step("edit_occurrences", backend.edit(path, "beta", "gamma").occurrences)
        step("ls_sees_file", any(e["path"] == path for e in (backend.ls("/tmp/smoke").entries or [])))
        grep = backend.grep("gamma", path="/tmp/smoke")
        step("grep", {"error": grep.error, "paths": [m["path"] for m in grep.matches or []]})

        blob = os.urandom(1_500_000)
        started = time.monotonic()
        upload = backend.upload_files([("/tmp/smoke/blob.bin", blob)])[0]
        download = backend.download_files(["/tmp/smoke/blob.bin"])[0]
        step("blob_roundtrip", {"ok": upload.error is None and download.content == blob,
                                "seconds": round(time.monotonic() - started, 1)})

        detached = ExecTransport(client, inline_wait=3, poll_interval=2)
        detached.connect(host=backend.id, credentials=TransportCredentials(api_token=token))
        started = time.monotonic()
        r = detached.run("sleep 8; echo slow-done", timeout=60)
        step("detach_and_poll", {"stdout": r.stdout, "exit": r.exit_code, "seconds": round(time.monotonic() - started, 1)})
        r = detached.run("echo begun; sleep 20; touch /tmp/smoke/not-killed", timeout=6)
        step("timeout_kill", {"exit": r.exit_code, "expected": TIMEOUT_EXIT_CODE, "stderr": r.stderr[-80:]})
        time.sleep(18)
        r = backend.execute("test -e /tmp/smoke/not-killed && echo SURVIVED || echo killed; ls /tmp/.microvm-exec 2>/dev/null | wc -l")
        step("timeout_kill_verified", r.output)

        backend.close()
        started = time.monotonic()
        lifecycle.pause(sandbox_id=sandbox_id)
        step("paused", {"state": client.get(results["microvm_id"]).status, "seconds": round(time.monotonic() - started, 1)})
        started = time.monotonic()
        backend = lifecycle.get_or_create(sandbox_id=sandbox_id)
        step("resumed", {"same_vm": backend.id == results["microvm_id"], "seconds": round(time.monotonic() - started, 1)})
        step("filesystem_after_resume", backend.read(path).file_data["content"] if backend.read(path).file_data else None)
        backend.close()
    except Exception as exc:  # noqa: BLE001 - report and still clean up
        step("error", f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        try:
            lifecycle.delete(sandbox_id=sandbox_id)
            step("deleted", True)
        except Exception as exc:  # noqa: BLE001
            step("delete_error", f"{type(exc).__name__}: {exc}")
        leftovers = [s for s in client.list(tag="microvm-sandbox")
                     if s.name.startswith("microvm-sandbox-smoke-") and s.status not in (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)]
        for state in leftovers:
            client.delete(state.id)
        step("leftovers_deleted", [s.id for s in leftovers])
        out = Path.home() / ".langchain_digitalocean_microvm" / "probe" / date.today().isoformat()
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_exec.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
