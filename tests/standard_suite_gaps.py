"""Known gaps between langchain_tests' SandboxIntegrationTests and deepagents.

Shared by tests/unit_tests/test_standard_suite.py (offline) and
tests/integration_tests/test_sandbox_standard.py (real MicroVM). Every entry
is a non-strict xfail with a reason (the only override the suite allows), so
it starts passing, unnoticed, once upstream agrees again: remove it then.

Upstream (deepagents 0.7.21 vs langchain-tests 1.1.9; a bare BaseSandbox
running local subprocesses fails the same cases):
- glob: BaseSandbox returns absolute paths on purpose (`_absolutize_glob_path`:
  permission rules only match absolute paths); the suite expects paths
  relative to the search root.
- write: BaseSandbox overwrites an existing file; the suite expects
  "already exists".

Host: BaseSandbox's grep needs GNU grep. A MicroVM image has it; macOS ships
BSD grep or ugrep, so the offline run on a Mac cannot pass the grep cases.
"""

from __future__ import annotations

import functools
import subprocess

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

UPSTREAM_GLOB = "deepagents >= 0.7.21 BaseSandbox.glob returns absolute paths; langchain-tests 1.1.9 expects relative"
UPSTREAM_WRITE = "deepagents >= 0.7.21 BaseSandbox.write overwrites; langchain-tests 1.1.9 expects 'already exists'"
NO_GNU_GREP = "BaseSandbox.grep needs GNU grep; this host's grep is not GNU"

GLOB_TESTS = (
    "test_glob",
    "test_glob_basic_pattern",
    "test_glob_with_directories",
    "test_glob_hidden_files_explicitly",
    "test_glob_with_character_class",
    "test_glob_with_question_mark",
)
GREP_TESTS = tuple(
    name
    for name in dir(SandboxIntegrationTests)
    if name.startswith("test_grep") and name not in {"test_grep_no_matches", "test_grep_empty_directory"}
)


def gnu_grep() -> bool:
    out = subprocess.run(["grep", "--version"], capture_output=True, text=True, check=False).stdout
    return out.startswith("grep (GNU grep)")


def _xfail(name: str, reason: str, condition: bool = True):
    base = getattr(SandboxIntegrationTests, name)

    @pytest.mark.xfail(condition, reason=reason, strict=False)
    @functools.wraps(base)
    def test(self, *args, **kwargs):
        return base(self, *args, **kwargs)

    return test


def upstream_gaps() -> dict[str, object]:
    """Overrides for the upstream disagreements (any sandbox, any host)."""
    overrides: dict[str, object] = {name: _xfail(name, UPSTREAM_GLOB) for name in GLOB_TESTS}
    overrides["test_write_existing_file_fails"] = _xfail("test_write_existing_file_fails", UPSTREAM_WRITE)
    return overrides


def local_host_gaps() -> dict[str, object]:
    """Overrides for a run whose commands execute on this machine."""
    return {name: _xfail(name, NO_GNU_GREP, condition=not gnu_grep()) for name in GREP_TESTS}
