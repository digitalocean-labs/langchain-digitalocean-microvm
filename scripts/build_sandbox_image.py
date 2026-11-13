"""Build the sandbox image for MicroVMSandbox and push it to a registry, without Docker.

A sandbox MicroVM needs an image that keeps running, serves HTTP on one port
(MicroVMs require one), ships sh + coreutils + python3 (BaseSandbox's file
operations are python3 scripts) and, for deepagents subagents, runs as a user
that can write anywhere: the model writes absolute paths it picks itself, and
deepagents offloads to /large_tool_results/ and /conversation_history/
(see docs/design.md). python:3.12-slim already has all of that and runs as
root, so the image is that base's linux/amd64 manifest, pinned by digest,
with only its configuration changed (no extra layer):

- entrypoint: the package's sandbox server (src/langchain_digitalocean_microvm/
  sandbox_server.py, standard library only) on $PORT (default 8080), the data
  plane of DO_MICROVM_TRANSPORT=endpoint. It answers `/` and `/healthz`
  without a token and everything else only with the sandbox's own token,
  which EndpointTransport writes to /var/lib/microvm-sandbox/token over exec
  on first connect; it logs nothing. The exec transport works with this image
  too (it never talks to the server);
- working directory /, HOME=/root, user root.

The tag is a hash of the base digest and that configuration, so rebuilding
reproduces the same tag. Credentials: DO_REGISTRY_TOKEN (.env or the
environment; registry:read + registry:update), written to a throwaway
DOCKER_CONFIG directory and deleted afterwards, never to ~/.docker or argv.

    uv run python \\
        scripts/build_sandbox_image.py \\
        --repository registry.digitalocean.com/<registry>/microvm-sandbox

Prints the pushed reference; set it as DO_MICROVM_IMAGE (DO_MICROVM_HTTP_PORT
stays at its default, 8080). Needs `crane` (github.com/google/go-containerregistry).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _env import add_env_file_argument, load_env  # scripts/_env.py

from langchain_digitalocean_microvm.transport.endpoint import server_source

BASE_IMAGE = "docker.io/library/python:3.12-slim"
PLATFORM = "linux/amd64"
PORT = 8080
ENTRYPOINT = ["python3", "-c", server_source()]
ENV = {"PORT": str(PORT), "HOME": "/root", "PYTHONUNBUFFERED": "1"}


def run(args: list[str], *, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(a if len(a) < 100 and "\n" not in a else a[:60].split("\n")[0] + "..." for a in args), flush=True)
    done = subprocess.run(args, env=env, text=True, capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"command failed ({done.returncode}): {args[0]} {(done.stderr or done.stdout).strip()[-2000:]}")
    return done.stdout


def _csv_field(items: list[str]) -> str:
    """crane's list flags are parsed as one CSV record: quote fields that contain commas, quotes or newlines."""
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow(items)
    return out.getvalue()


def content_tag(base_digest: str) -> str:
    digest = hashlib.sha256(json.dumps([base_digest, ENTRYPOINT, ENV, PORT], sort_keys=True).encode())
    return digest.hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", default=os.environ.get("DO_MICROVM_SANDBOX_REPOSITORY"),
                        help="e.g. registry.digitalocean.com/<registry>/microvm-sandbox")
    parser.add_argument("--tag", help="default: content hash of the base digest and configuration")
    parser.add_argument("--no-push", action="store_true", help="resolve the base and print the reference only")
    add_env_file_argument(parser)
    args = parser.parse_args()
    if not args.repository:
        parser.error("--repository (or DO_MICROVM_SANDBOX_REPOSITORY) is required")
    if shutil.which("crane") is None:
        raise SystemExit("crane is not on PATH")

    base_digest = run(["crane", "digest", "--platform", PLATFORM, BASE_IMAGE]).strip()
    base_ref = f"{BASE_IMAGE.rsplit(':', 1)[0]}@{base_digest}"  # the amd64 image itself, not the index
    reference = f"{args.repository}:{args.tag or content_tag(base_digest)}"
    print(f"base {base_ref}\ntarget {reference}", flush=True)
    if args.no_push:
        return 0

    token = os.environ.get("DO_REGISTRY_TOKEN") or load_env(args.env_file).get("DO_REGISTRY_TOKEN")
    if not token:
        raise SystemExit("DO_REGISTRY_TOKEN is not set (.env or the environment)")
    registry = args.repository.split("/", 1)[0]
    with tempfile.TemporaryDirectory(prefix="microvm-sandbox-build-") as tmp:
        config_dir = Path(tmp) / "docker-config"
        config_dir.mkdir(mode=0o700)
        # DOCR takes the API token as both username and password.
        auth = base64.b64encode(f"{token}:{token}".encode()).decode()
        (config_dir / "config.json").write_text(json.dumps({"auths": {registry: {"auth": auth}}}))
        (config_dir / "config.json").chmod(0o600)
        env = {**os.environ, "DOCKER_CONFIG": str(config_dir)}
        mutate = ["crane", "mutate", base_ref, "-t", reference, "--user", "0:0", "--workdir", "/",
                  "--entrypoint", _csv_field(ENTRYPOINT), "--cmd", "", "--exposed-ports", str(PORT)]
        for key, value in ENV.items():
            mutate += ["--env", f"{key}={value}"]
        pushed = run(mutate, env=env).strip().splitlines()[-1]
        config = json.loads(run(["crane", "config", reference], env=env))["config"]
    if config.get("Entrypoint") != ENTRYPOINT or config.get("User") != "0:0":
        raise SystemExit(f"pushed image has an unexpected configuration: {config}")
    print(f"\npushed {reference}\n  digest ref: {pushed}\n\nSet DO_MICROVM_IMAGE={reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
