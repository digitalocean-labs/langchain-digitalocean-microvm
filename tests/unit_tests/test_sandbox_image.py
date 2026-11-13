"""scripts/build_sandbox_image.py: the sandbox image's entrypoint and tag (no registry, no crane).

The entrypoint (the package's sandbox server) is run for real, as a local
python3 process: it must keep running, answer the health routes without a
token, and serve nothing else without the sandbox's own token (the MicroVM
endpoint is reachable by any `microvm:access` token of the team).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_sandbox_image as image  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path: Path):
    port = _free_port()
    env = {"PORT": str(port), "SANDBOX_TOKEN_FILE": str(tmp_path / "token"), "SANDBOX_JOB_ROOT": str(tmp_path / "jobs")}
    proc = subprocess.Popen([sys.executable, *image.ENTRYPOINT[1:]], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 10
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                raise AssertionError(f"sandbox server did not start: {proc.communicate()}") from None
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", proc
    proc.terminate()
    proc.wait(timeout=5)


def test_entrypoint_answers_health_and_serves_nothing_without_the_token(server) -> None:
    url, proc = server
    for path in ("/", "/healthz"):
        with urllib.request.urlopen(url + path, timeout=5) as response:
            assert (response.status, response.read()) == (200, b"ok\n")
    for path in ("/etc/passwd", "/../../etc/passwd", "/v1/files?path=/etc/passwd", "/v1/info"):
        request = urllib.request.Request(url + path, headers={"Authorization": "Bearer secret-token"})
        with pytest.raises(urllib.error.HTTPError) as err:  # no token file yet: everything is refused
            urllib.request.urlopen(request, timeout=5)
        assert err.value.code == 401 and b"secret-token" not in err.value.read()
    proc.terminate()
    out, err = proc.communicate(timeout=5)
    assert b"secret-token" not in out + err and b"GET" not in out + err  # logs nothing


def test_entrypoint_is_the_packaged_sandbox_server() -> None:
    from langchain_digitalocean_microvm.transport.endpoint import server_source

    assert image.ENTRYPOINT == ["python3", "-c", server_source()]


def test_entrypoint_is_python3_dash_c_and_the_image_runs_as_root() -> None:
    assert image.ENTRYPOINT[:2] == ["python3", "-c"]
    assert image.ENV["PORT"] == str(image.PORT) == "8080"  # MicroVMConfig.http_port's default
    assert image.ENV["HOME"] == "/root"


def test_tag_is_a_content_hash() -> None:
    a = image.content_tag("sha256:" + "a" * 64)
    assert a == image.content_tag("sha256:" + "a" * 64)
    assert a != image.content_tag("sha256:" + "b" * 64)
    assert len(a) == 12 and int(a, 16) >= 0


def test_list_flags_survive_crane_csv_parsing() -> None:
    import csv

    field = image._csv_field(image.ENTRYPOINT)
    assert next(csv.reader([field])) == image.ENTRYPOINT  # commas, quotes and newlines in the script
