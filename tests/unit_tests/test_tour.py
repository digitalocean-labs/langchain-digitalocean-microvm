"""examples/tour.py: which steps run, the local mode environment, a step's output, and the dashboard window."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import tour  # noqa: E402


def test_the_default_tour_is_example_11_with_its_dashboard_and_full_has_all_five() -> None:
    default = tour.select(None)
    assert [s.number for s in default] == [5] and default[0].dashboard
    assert [s.number for s in tour.select(None, full=True)] == [1, 2, 3, 4, 5]
    assert [s.number for s in tour.select("3,2")] == [2, 3]
    with pytest.raises(SystemExit):
        tour.select("9")


def test_every_step_script_and_dashboard_exists() -> None:
    for step in tour.STEPS:
        assert (tour.HERE / step.script).is_file(), step.script
        if step.dashboard:
            assert (tour.HERE / step.dashboard).is_file(), step.dashboard


def test_the_environment_is_fake_keeps_credentials_out_and_puts_files_under_the_temporary_directory(tmp_path: Path) -> None:
    env = tour.tour_env(tmp_path, {"DO_MICROVM_API_TOKEN": "real", "HOME": "/h", "PATH": "/bin"})
    assert env["DO_MICROVM_TRANSPORT"] == "local" and env["DO_MICROVM_HOST_MODE"] == "local"
    assert env["DO_MICROVM_API_TOKEN"] == "" and env["DO_MICROVM_ACCESS_TOKEN"] == "" and env["DO_ACTION_GATEWAY_TOKEN"] == ""
    assert env["TMPDIR"] == str(tmp_path) and env["HOME"].startswith(str(tmp_path))
    assert env["XDG_STATE_HOME"].startswith(str(tmp_path)) and env["PATH"] == "/bin"
    assert set(tour.TOUR_VARIABLES) == set(env) - {"PATH"}  # what a dashboard window gets


def test_a_step_shows_the_tail_of_its_output_and_reports_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = tmp_path / "noisy.py"
    script.write_text("import sys\nfor i in range(30): print('line', i)\nsys.exit(0 if len(sys.argv) < 2 else 3)\n")
    ok_step = tour.Step(9, "Noisy", "prints", "noisy.py")
    ok, _ = tour.run_step(ok_step, {}, lines=3, verbose=False, base=tmp_path)
    shown = capsys.readouterr().out
    assert ok and "line 29" in shown and "line 26" not in shown and "27 earlier lines" in shown
    bad = tour.Step(9, "Noisy", "prints", "noisy.py", ("x",))
    assert tour.run_step(bad, {}, lines=1, verbose=False, base=tmp_path)[0] is False


def test_the_window_script_escapes_quotes_and_backslashes_and_picks_the_app() -> None:
    terminal = tour.window_script('echo "a\\b"', "Terminal")
    assert 'tell application "Terminal"' in terminal and 'do script "echo \\"a\\\\b\\""' in terminal
    iterm = tour.window_script('echo "x"', "iTerm")
    assert 'tell application "iTerm"' in iterm and 'write text "echo \\"x\\""' in iterm


def test_no_window_off_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tour.sys, "platform", "linux")
    monkeypatch.setattr(tour.subprocess, "run", lambda *a, **k: pytest.fail("osascript must not run"))
    assert tour.open_window(["true"], {}, tmp_path / "pid") is False


def test_the_closing_lines_start_at_the_gateway_audit_without_the_deletions() -> None:
    output = ["thread tour", "lead -> task", "\x1b[2mgateway audit:\x1b[0m", "  ask  github_create_issue",
              "report: outputs/vulnerability-report.md", "deleted tour-analyst-docs-site"]
    assert tour.closing_lines(output, 2) == output[2:5]
    assert tour.closing_lines(["a", "b", "c"], 2) == ["b", "c"]


def test_a_watched_step_runs_the_dashboard_in_this_terminal_and_stops_it(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "view.py").write_text(
        "import pathlib, signal, sys, time\n"
        "(pathlib.Path(__file__).parent / 'view.args').write_text(' '.join(sys.argv[1:]))\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\ntime.sleep(30)\n")
    (tmp_path / "agent.py").write_text(
        "import sys\nprint(' '.join(sys.argv[1:]))\nprint('gateway audit:')\nprint('deleted x')\nprint('report: r.md')\n")
    step = tour.Step(9, "Watched", "shows", "agent.py", ("--approve",), dashboard="view.py")
    ok, _ = tour.run_watched(step, {}, tmp=tmp_path, same_terminal=True, base=tmp_path, settle=0.5, linger=0.1)
    shown = capsys.readouterr().out
    assert ok and "gateway audit:\nreport: r.md" in shown and "deleted x" not in shown
    assert (tmp_path / "view.args").read_text() == "--thread tour"
    assert "--thread tour --approve" in (tmp_path / "step-9.log").read_text()
