"""End-to-end: local agent hosts work from an installed wheel, not only from a source checkout.

Marked `e2e` (deselected by default; `uv run pytest -m e2e`): it runs `uv build` (sdist, then the wheel from the
sdist, as a release does), installs the wheel with the `agent-host` extra into a fresh virtualenv outside the
repository, and from there starts a `LocalDevServerHost` on the `echo` graph until `GET /ok` answers. It needs
`uv` on PATH, and network access unless uv's cache already holds the extra's dependencies.

The install is the only package install here: the host runs the model-free `echo` graph, no agent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_FILES = ("agent.py", "auth.py", "artifacts_app.py", "langgraph.json")
LEFT_OUT = ("pyproject.toml", "uv.lock", "Dockerfile", "build_image.py", "README.md")

# Runs on the fresh venv's interpreter, with a working directory outside the repository.
HOST_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import httpx

    import langchain_digitalocean_microvm
    from langchain_digitalocean_microvm.agent_host import HostSpec, HostStatus, LocalDevServerHost
    from langchain_digitalocean_microvm.agent_host.local import PACKAGED_TEMPLATE_DIR, default_template_dir

    package = Path(langchain_digitalocean_microvm.__file__).resolve().parent
    assert "site-packages" in package.parts, package
    assert default_template_dir() == PACKAGED_TEMPLATE_DIR, default_template_dir()
    assert PACKAGED_TEMPLATE_DIR.is_relative_to(package), PACKAGED_TEMPLATE_DIR

    with LocalDevServerHost(root_dir=Path(sys.argv[1]), ready_timeout=120.0) as hosts:
        assert hosts.template_dir == PACKAGED_TEMPLATE_DIR.resolve(), hosts.template_dir
        handle = hosts.get_or_create(host_id="wheel", spec=HostSpec(graph_id="echo"))
        assert handle.status is HostStatus.READY, handle.status
        assert httpx.get(f"{handle.url}/ok", timeout=5).status_code == 200
    print("WHEEL-HOST-OK")
    """
)


def _clean_env() -> dict[str, str]:
    """This process's environment minus anything that would point the venv back at the repository."""
    drop = {"DO_MICROVM_SUBAGENT_TEMPLATE", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["DO_MICROVM_TRANSPORT"] = "local"
    env["DO_MICROVM_HOST_MODE"] = "local"
    return env


def test_installed_wheel_serves_the_packaged_template(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv on PATH")
    env = _clean_env()

    dist = tmp_path / "dist"
    subprocess.run([uv, "build", "--out-dir", str(dist)], cwd=REPO, env=env, check=True, capture_output=True)
    (wheel,) = dist.glob("*.whl")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    packaged = {
        name.rsplit("/", 1)[-1]
        for name in names
        if name.startswith("langchain_digitalocean_microvm/agent_host/_template/")
    }
    assert packaged == set(TEMPLATE_FILES)
    assert not any(name.startswith("images/") for name in names)
    assert not set(LEFT_OUT) & packaged

    venv = tmp_path / "venv"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run(
        [uv, "venv", "--python", python_version, str(venv)], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    python = venv / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), f"{wheel}[agent-host]"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )

    script = tmp_path / "scripts" / "run_host.py"  # a directory of its own: nothing importable next to it
    script.parent.mkdir()
    script.write_text(HOST_SCRIPT)
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        [str(python), "-I", str(script), str(tmp_path / "hosts")],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "WHEEL-HOST-OK" in result.stdout
