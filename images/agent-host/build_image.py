# images/agent-host/build_image.py
"""Build the agent-host image and push it to a registry, without Docker.

MicroVMs run one linux/amd64 container image (see Dockerfile for the same
image built the conventional way). This script needs only `uv` and `crane`
(github.com/google/go-containerregistry), so it works on a Mac with no
Docker daemon and no emulation:

1. `uv export` the locked dependencies of this package;
2. `uv pip install --target` them for linux/amd64 + CPython 3.12
   (manylinux wheels; source-only pure-Python packages are built locally);
3. pack them, agent.py, artifacts_app.py, auth.py and langgraph.json into one layer
   (owned by uid 1000, the user the server runs as);
4. `crane append` that layer onto python:3.12-slim and `crane mutate` the
   entrypoint, env, user and working directory.

The tag is a hash of everything that goes into the layer, so rebuilding
unchanged sources reproduces the same tag. Registry credentials come from
DO_REGISTRY_TOKEN (.env), a DigitalOcean token with registry:read +
registry:update; they are written to a throwaway DOCKER_CONFIG directory and
deleted afterwards, never to ~/.docker.

    uv run python images/agent-host/build_image.py \\
        --repository registry.digitalocean.com/<registry>/microvm-subagent

Prints the pushed reference; put it in .env as DO_MICROVM_AGENT_IMAGE.

Another agent host's image (an example's graphs on the same base and entrypoint) is described by a JSON
spec, `--spec <file>`; paths in it are relative to the spec's directory:

    {"project": ".",                  the uv project whose locked dependencies go in the layer
     "root": "../../..",              the base of `files`: each file lands at /opt/agent/<its path from root>
     "files": ["images/agent-host/auth.py", "examples/x/agent.py", "examples/x/sandbox/*.py", ...],
     "writable": ["examples/x/outputs"],   directories the server's user writes to
     "config": "examples/x/host/langgraph.json"}   the langgraph.json to serve, from root

A path dependency of the project (this package, for example) is built into a wheel and installed like the
others, so its sources are part of the tag.
Stdlib only on purpose: it must run before any project dependency exists.
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
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_FILES = ("agent.py", "artifacts_app.py", "auth.py", "langgraph.json")
BASE_IMAGE = "docker.io/library/python:3.12-slim"
PYTHON_VERSION = "3.12"
WHEEL_PLATFORM = "x86_64-manylinux_2_28"
UID = GID = 1000
APP_DIR = "opt/agent"
SITE_DIR = f"{APP_DIR}/site"
BASE_PATH = "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
# One string for `sh -c` (passed to crane as a CSV record, see _csv_field).
ENTRYPOINT = (
    'exec python -m langgraph_cli dev --config /opt/agent/langgraph.json --host 0.0.0.0 --port "${PORT:-2024}" '
    '--no-browser --no-reload --allow-blocking --n-jobs-per-worker "${N_JOBS_PER_WORKER:-2}"'
)
ENV = {
    "PYTHONPATH": f"/{SITE_DIR}:/{APP_DIR}",
    "PATH": f"/{SITE_DIR}/bin:{BASE_PATH}",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": "/work",
    "SUBAGENT_WORKDIR": "/work",
    "PORT": "2024",
}


def run(args: list[str], *, env: dict[str, str] | None = None, capture: bool = False, cwd: Path | None = None) -> str:
    print("+", " ".join(a if len(a) < 120 else a[:117] + "..." for a in args), flush=True)
    done = subprocess.run(args, env=env, text=True, capture_output=capture, check=False, cwd=cwd)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip() if capture else ""
        raise SystemExit(f"command failed ({done.returncode}): {args[0]} {detail[-2000:]}")
    return done.stdout if capture else ""


def _csv_field(items: list[str]) -> str:
    """crane's list flags are parsed as one CSV record: quote fields that contain quotes."""
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow(items)
    return out.getvalue()


def read_token(env_file: Path) -> str:
    token = os.environ.get("DO_REGISTRY_TOKEN")
    if not token and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "DO_REGISTRY_TOKEN":
                token = value.split(" #", 1)[0].strip().strip('"').strip("'")
    if not token:
        raise SystemExit("DO_REGISTRY_TOKEN is not set (.env or the environment)")
    return token


class Spec:
    """What goes into the layer: the default is this directory's project and APP_FILES (see the module docstring)."""

    def __init__(self, project: Path, root: Path, files: list[Path], writable: list[str], config: str) -> None:
        self.project, self.root, self.files, self.writable, self.config = project, root, files, writable, config

    @classmethod
    def default(cls) -> Spec:
        return cls(HERE, HERE, [HERE / name for name in APP_FILES], [], "langgraph.json")

    @classmethod
    def load(cls, path: Path) -> Spec:
        base = path.resolve().parent
        data = json.loads(path.read_text())
        root = (base / data.get("root", ".")).resolve()
        files: list[Path] = []
        for pattern in data["files"]:
            matches = sorted(root.glob(pattern))
            if not matches:
                raise SystemExit(f"{path}: {pattern!r} matches no file under {root}")
            files += [m for m in matches if m.is_file() and "__pycache__" not in m.parts]
        if not (root / data["config"]).is_file():
            raise SystemExit(f"{path}: config {data['config']!r} is not a file under {root}")
        return cls((base / data.get("project", ".")).resolve(), root, files, list(data.get("writable", [])), data["config"])

    def entrypoint(self) -> str:
        return ENTRYPOINT.replace("/opt/agent/langgraph.json", f"/{APP_DIR}/{self.config}")

    def workdir(self) -> str:
        """The server runs from its config's directory: the Agent Server reads langgraph.json's relative paths from
        its working directory, not from the file's (it also keeps its saved runs there, so the directory is owned by
        the server's user)."""
        parent = Path(self.config).parent.as_posix()
        return f"/{APP_DIR}" if parent == "." else f"/{APP_DIR}/{parent}"


def install_site(work: Path, spec: Spec) -> Path:
    requirements = work / "requirements.txt"
    # --no-editable: a path dependency is installed as a wheel built from its sources, not linked to this machine.
    run(["uv", "export", "--color", "never", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
         "--no-editable", "--project", str(spec.project), "-o", str(requirements)])
    site = work / "site"
    run([
        "uv", "pip", "install", "--color", "never", "--quiet", "--target", str(site),
        "--python-platform", WHEEL_PLATFORM, "--python-version", PYTHON_VERSION,
        "--no-compile", "-r", str(requirements),
    ], cwd=None if spec.project == HERE else spec.project)  # a spec's relative paths are relative to its project
    # Source-only dependencies (e.g. forbiddenfruit, pure Python) get built here; uv refuses any
    # resulting wheel whose tags do not match the linux/amd64 target. Belt and braces:
    native = [p for p in site.rglob("*.so") if b"\x7fELF" != p.read_bytes()[:4]]
    if native:
        raise SystemExit(f"non-Linux native extensions in the layer: {native[:5]}")
    for cache in site.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    return site


def _owned(info: tarfile.TarInfo, uid: int = UID, gid: int = GID) -> tarfile.TarInfo:
    info.uid, info.gid, info.uname, info.gname = uid, gid, "", ""
    info.mtime = 0  # reproducible layer
    return info


def _parents(rel: str) -> list[str]:
    parts = rel.split("/")[:-1]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def build_layer(site: Path, out: Path, spec: Spec | None = None) -> str:
    """Write the layer tarball; return a content hash of everything in it."""
    spec = spec or Spec.default()
    digest = hashlib.sha256()
    names = [f.relative_to(spec.root).as_posix() for f in spec.files]
    directories = [APP_DIR, SITE_DIR, "work"]
    for rel in [*names, *(f"{w}/x" for w in spec.writable)]:  # the app's own directories, before their files
        directories += [f"{APP_DIR}/{d}" for d in _parents(rel) if f"{APP_DIR}/{d}" not in directories]
    with tarfile.open(out, "w") as tar:
        for directory in directories:
            info = _owned(tarfile.TarInfo(directory))
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        for name, path in zip(names, spec.files):
            data = path.read_bytes()
            digest.update(name.encode() + data)
            info = _owned(tarfile.TarInfo(f"{APP_DIR}/{name}"))
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
        for writable in spec.writable:
            digest.update(f"writable:{writable}".encode())
        for path in sorted(site.rglob("*")):
            rel = path.relative_to(site).as_posix()
            arcname = f"{SITE_DIR}/{rel}"
            if path.is_symlink():
                info = _owned(tarfile.TarInfo(arcname))
                info.type, info.linkname = tarfile.SYMTYPE, os.readlink(path)
                tar.addfile(info)
            elif path.is_dir():
                info = _owned(tarfile.TarInfo(arcname))
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            else:
                data = path.read_bytes()
                digest.update(rel.encode() + hashlib.sha256(data).digest())
                info = _owned(tarfile.TarInfo(arcname))
                info.size, info.mode = len(data), 0o755 if os.access(path, os.X_OK) else 0o644
                tar.addfile(info, io.BytesIO(data))
    digest.update(BASE_IMAGE.encode() + spec.entrypoint().encode())
    if spec.workdir() != f"/{APP_DIR}":
        digest.update(f"workdir:{spec.workdir()}".encode())
    return digest.hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", default=os.environ.get("DO_MICROVM_AGENT_REPOSITORY"),
                        help="e.g. registry.digitalocean.com/<registry>/microvm-subagent")
    parser.add_argument("--tag", help="default: content hash of the layer")
    parser.add_argument("--env-file", type=Path, default=HERE.parents[1] / ".env")
    parser.add_argument("--no-push", action="store_true", help="build the layer only")
    parser.add_argument("--spec", type=Path, help="JSON spec of another agent host's image (see the module docstring)")
    args = parser.parse_args()
    if not args.repository:
        parser.error("--repository (or DO_MICROVM_AGENT_REPOSITORY) is required")
    for tool in ("uv", "crane"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is not on PATH")

    spec = Spec.load(args.spec) if args.spec else Spec.default()
    with tempfile.TemporaryDirectory(prefix="microvm-subagent-build-") as tmp:
        work = Path(tmp)
        site = install_site(work, spec)
        layer = work / "layer.tar"
        content_hash = build_layer(site, layer, spec)
        print(f"layer: {layer.stat().st_size / 2**20:.1f} MiB, content hash {content_hash}", flush=True)
        if args.no_push:
            return 0

        reference = f"{args.repository}:{args.tag or content_hash}"
        registry = args.repository.split("/", 1)[0]
        token = read_token(args.env_file)
        config_dir = work / "docker-config"
        config_dir.mkdir(mode=0o700)
        # DOCR takes the API token as both username and password. Written to a
        # throwaway config (not argv, where `ps` would show it; not ~/.docker).
        auth = base64.b64encode(f"{token}:{token}".encode()).decode()
        (config_dir / "config.json").write_text(json.dumps({"auths": {registry: {"auth": auth}}}))
        (config_dir / "config.json").chmod(0o600)
        env = {**os.environ, "DOCKER_CONFIG": str(config_dir)}
        run(["crane", "append", "--platform", "linux/amd64", "-b", BASE_IMAGE, "-f", str(layer), "-t", reference], env=env)
        mutate = ["crane", "mutate", reference, "--user", f"{UID}:{GID}", "--workdir", spec.workdir(),
                  "--entrypoint", _csv_field(["sh", "-c", spec.entrypoint()]), "--exposed-ports", "2024"]
        for key, value in ENV.items():
            mutate += ["--env", f"{key}={value}"]
        pushed = run(mutate, env=env, capture=True).strip().splitlines()[-1]
        variable = "DO_MICROVM_AGENT_IMAGE" if args.spec is None else "the image variable your spec's README names"
        print(f"\npushed {reference}\n  digest ref: {pushed}\n\nSet in .env:\n  {variable}={reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
