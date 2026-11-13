import logging

import pytest

from langchain_digitalocean_microvm.status import (
    RESUMABLE_STATUSES,
    REUSABLE_STATUSES,
    TERMINAL_FAILURE_STATUSES,
    TRANSITIONAL_STATUSES,
    MicroVMStatus,
    parse_status,
)


@pytest.mark.parametrize("status", list(MicroVMStatus))
def test_parse_status_round_trips(status: MicroVMStatus) -> None:
    assert parse_status(status.value) is status


def test_parse_status_unknown_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result = parse_status("totally-made-up-status")
    assert result is MicroVMStatus.UNKNOWN
    assert "totally-made-up-status" in caplog.text


def test_reusable_statuses() -> None:
    assert REUSABLE_STATUSES == {MicroVMStatus.RUNNING}


def test_resumable_statuses() -> None:
    assert RESUMABLE_STATUSES == {MicroVMStatus.PAUSED}


def test_transitional_statuses() -> None:
    assert TRANSITIONAL_STATUSES == {
        MicroVMStatus.CREATING,
        MicroVMStatus.PAUSING,
        MicroVMStatus.RESUMING,
    }


def test_terminal_failure_statuses() -> None:
    assert TERMINAL_FAILURE_STATUSES == {
        MicroVMStatus.FAILED,
        MicroVMStatus.TERMINATING,
        MicroVMStatus.TERMINATED,
    }


def test_states_match_the_documented_api_vocabulary() -> None:
    documented = {"creating", "running", "pausing", "paused", "resuming", "terminating", "terminated", "failed", "unknown"}
    assert {s.value for s in MicroVMStatus} == documented
