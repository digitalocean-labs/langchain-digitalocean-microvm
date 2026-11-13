"""examples/tour.py: which steps run, the local mode environment, and a step's output tail."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import tour  # noqa: E402


def test_the_default_tour_is_the_quick_steps_and_full_has_all_four() -> None:
    assert [s.number for s in tour.select(None)] == [1, 4]
    assert [s.number for s in tour.select(None, full=True)] == [1, 2, 3, 4]
    assert [s.number for s in tour.select("3,2")] == [2, 3]
    with pytest.raises(SystemExit):
        tour.select("9")


def test_every_step_script_exists() -> None:
    for step in tour.STEPS:
        assert (tour.HERE / step.script).is_file(), step.script


def test_the_environment_is_fake_keeps_credentials_out_and_puts_files_under_the_temporary_directory(tmp_path: Path) -> None:
    env = tour.tour_env(tmp_path, {"DO_MICROVM_API_TOKEN": "real", "HOME": "/h", "PATH": "/bin"})
    assert env["DO_MICROVM_TRANSPORT"] == "local" and env["DO_MICROVM_HOST_MODE"] == "local"
    assert env["DO_MICROVM_API_TOKEN"] == "" and env["DO_MICROVM_ACCESS_TOKEN"] == "" and env["DO_ACTION_GATEWAY_TOKEN"] == ""
    assert env["TMPDIR"] == str(tmp_path) and env["HOME"].startswith(str(tmp_path))
    assert env["XDG_STATE_HOME"].startswith(str(tmp_path)) and env["PATH"] == "/bin"


def test_a_step_shows_the_tail_of_its_output_and_reports_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = tmp_path / "noisy.py"
    script.write_text("import sys\nfor i in range(30): print('line', i)\nsys.exit(0 if len(sys.argv) < 2 else 3)\n")
    ok_step = tour.Step(9, "Noisy", "prints", "noisy.py")
    ok, _ = tour.run_step(ok_step, {}, lines=3, verbose=False, base=tmp_path)
    shown = capsys.readouterr().out
    assert ok and "line 29" in shown and "line 26" not in shown and "27 earlier lines" in shown
    bad = tour.Step(9, "Noisy", "prints", "noisy.py", ("x",))
    assert tour.run_step(bad, {}, lines=1, verbose=False, base=tmp_path)[0] is False
