"""Tests for langchain_digitalocean_microvm.backend.MicroVMSandbox wired to a real
langchain_digitalocean_microvm.transport.local.LocalTransport.

LocalTransport is not a mock: `LocalTransport.run()` shells out to a genuine
local `bash -lc <command>` subprocess (see transport/local.py), so every test
below actually exercises the shell/python3 scripts that
deepagents.backends.sandbox.BaseSandbox generates for ls/read/write/edit/
grep/glob -- not a stand-in for them.

Path convention: file paths passed to write/read/edit/delete/ls/glob/grep
below are given *without* a leading '/' (e.g. "notes.txt", not "/notes.txt"),
purely so each assertion reads the same regardless of which convention is
under test. Both conventions actually work: LocalTransport.run() isolates
absolute paths from the real host filesystem too (see
transport/local.py's module docstring -- "Absolute-path isolation" -- and
test_absolute_paths_are_isolated_from_the_real_host_filesystem below), the
same as put_file/get_file (upload_files/download_files) already did.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_digitalocean_microvm.backend import MicroVMSandbox
from langchain_digitalocean_microvm.transport.base import TransportCredentials
from langchain_digitalocean_microvm.transport.local import LocalTransport


@pytest.fixture
def sandbox() -> Iterator[MicroVMSandbox]:
    """A MicroVMSandbox wired to a fresh LocalTransport.

    Each test gets its own uniquely-hosted LocalTransport, so each test's
    LocalTransport sandbox root (see `_root_for_host`) is a distinct real
    directory on local disk -- no state or files leak between tests.
    """
    transport = LocalTransport()
    transport.connect(
        host="test-backend-" + uuid.uuid4().hex,
        port=0,
        credentials=TransportCredentials(),
    )
    sb = MicroVMSandbox(microvm_id="test-microvm", transport=transport)
    yield sb
    sb.close()


def _grep_null_separator_fix(tmp_path: Path) -> Path:
    """A BASH_ENV script that fixes up `grep`'s null-separator flag.

    `deepagents.backends.sandbox._build_grep_cmd` always runs
    `grep -rHnFZ ...`, assuming GNU grep semantics where the short `-Z`
    means `--null` (NUL-separate the filename from the match, exactly what
    `_parse_grep_output` expects). A real MicroVM's Linux image ships GNU
    coreutils and would never hit this, but LocalTransport genuinely shells
    out to *this* machine's `grep` -- and on macOS, the system `/usr/bin/
    grep` is BSD grep, where the short `-Z` instead means `--decompress`
    (see `man grep`), silently producing no NUL byte at all and breaking
    `_parse_grep_output`.

    Sourced via `BASH_ENV` before the generated grep command runs, this
    shell function rewrites any clustered short-option argument containing
    "Z" (e.g. "-rHnFZ") to drop the "Z" and append the long option
    "--null" instead: a no-op rewrite on GNU grep (where `-Z` and `--null`
    already mean the same thing) and a correctness fix on BSD grep.
    """
    script = tmp_path / "grep_null_separator_fix.sh"
    script.write_text(
        "grep() {\n"
        '    local args=()\n'
        '    for a in "$@"; do\n'
        '        case "$a" in\n'
        '            -*Z*) args+=("${a//Z/}" "--null") ;;\n'
        '            *) args+=("$a") ;;\n'
        "        esac\n"
        "    done\n"
        '    command grep "${args[@]}"\n'
        "}\n"
    )
    return script


def test_write_then_read_round_trips_content(sandbox: MicroVMSandbox) -> None:
    write_result = sandbox.write("/notes.txt", "hello from microvm sandbox\n")

    assert write_result.error is None
    assert write_result.path == "/notes.txt"

    read_result = sandbox.read("/notes.txt")

    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "hello from microvm sandbox"


def test_edit_replaces_substring_and_reports_occurrence_count(sandbox: MicroVMSandbox) -> None:
    sandbox.write("/counts.txt", "cat cat cat\n")

    edit_result = sandbox.edit("/counts.txt", "cat", "dog", replace_all=True)

    assert edit_result.error is None
    assert edit_result.path == "/counts.txt"
    assert edit_result.occurrences == 3

    read_result = sandbox.read("/counts.txt")
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "dog dog dog"


def test_ls_lists_a_written_file(sandbox: MicroVMSandbox) -> None:
    sandbox.write("/listed.txt", "content\n")

    ls_result = sandbox.ls(".")

    assert ls_result.error is None
    assert ls_result.entries is not None
    assert any(entry["path"].endswith("/listed.txt") for entry in ls_result.entries)


def test_glob_finds_file_by_pattern(sandbox: MicroVMSandbox) -> None:
    sandbox.write("/data.json", "{}\n")
    sandbox.write("/notes.md", "# notes\n")

    glob_result = sandbox.glob("*.json", ".")

    assert glob_result.error is None
    assert glob_result.matches is not None
    assert any(entry["path"].endswith("/data.json") for entry in glob_result.matches)
    assert not any(entry["path"].endswith("/notes.md") for entry in glob_result.matches)


def test_grep_finds_line_by_pattern(
    sandbox: MicroVMSandbox,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BASH_ENV", str(_grep_null_separator_fix(tmp_path)))
    sandbox.write("/haystack.txt", "first line\nsecond line has needle\nthird line\n")

    grep_result = sandbox.grep("needle", ".")

    assert grep_result.error is None
    assert grep_result.matches is not None
    assert any(
        match["path"].endswith("/haystack.txt") and match["line"] == 2 and "needle" in match["text"]
        for match in grep_result.matches
    )


def test_delete_removes_file_and_subsequent_read_reports_error(sandbox: MicroVMSandbox) -> None:
    sandbox.write("/ephemeral.txt", "temporary\n")

    delete_result = sandbox.delete("/ephemeral.txt")

    assert delete_result.error is None
    assert delete_result.path == "/ephemeral.txt"

    read_result = sandbox.read("/ephemeral.txt")
    assert read_result.error is not None
    assert "file_not_found" in read_result.error
    assert read_result.file_data is None


def test_execute_returns_exit_code_and_output(sandbox: MicroVMSandbox) -> None:
    success = sandbox.execute("echo hello")

    assert success.exit_code == 0
    assert "hello" in success.output

    failure = sandbox.execute("exit 3")

    assert failure.exit_code == 3


def test_upload_and_download_files_round_trip_and_missing_file_reports_error(
    sandbox: MicroVMSandbox,
) -> None:
    upload_results = sandbox.upload_files(
        [
            ("/binary_a.bin", b"\x00\x01\x02\xfe\xff"),
            ("/binary_b.bin", b"some more \x00 bytes here"),
        ]
    )

    assert [r.error for r in upload_results] == [None, None]

    download_results = sandbox.download_files(["/binary_a.bin", "/binary_b.bin", "/does_not_exist.bin"])

    assert download_results[0].content == b"\x00\x01\x02\xfe\xff"
    assert download_results[0].error is None
    assert download_results[1].content == b"some more \x00 bytes here"
    assert download_results[1].error is None
    assert download_results[2].content is None
    assert download_results[2].error == "file_not_found"


def test_absolute_paths_round_trip_through_write_read_edit_delete(
    sandbox: MicroVMSandbox,
) -> None:
    """The same read/write/edit/delete flow as the tests above, with a
    directory component -- the convention every real transport requires, since there an absolute path IS naturally scoped to that one
    MicroVM. LocalTransport must honor it too (see transport/local.py's
    "Absolute-path isolation").
    """
    write_result = sandbox.write("/abs-notes.txt", "hello\n")
    assert write_result.error is None

    read_result = sandbox.read("/abs-notes.txt")
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "hello"

    edit_result = sandbox.edit("/abs-notes.txt", "hello", "goodbye")
    assert edit_result.error is None
    assert sandbox.read("/abs-notes.txt").file_data["content"] == "goodbye"

    delete_result = sandbox.delete("/abs-notes.txt")
    assert delete_result.error is None
    assert sandbox.read("/abs-notes.txt").error is not None


def test_absolute_paths_are_isolated_from_the_real_host_filesystem(
    sandbox: MicroVMSandbox, tmp_path: Path
) -> None:
    """A regression test for a real bug: before transport/local.py's
    absolute-path isolation, `sandbox.read("/etc/hosts")` (or any other
    absolute path that happens to exist on the machine running the tests)
    would read the REAL file, and `sandbox.delete(...)` would really delete
    it -- silently defeating the entire point of a "local", offline,
    zero-risk transport. A file outside the sandbox root must be invisible.
    """
    real_file = tmp_path / "host-only.txt"
    real_file.write_text("real host content, must not leak\n")

    read_result = sandbox.read(str(real_file))
    assert read_result.error is not None
    assert "file_not_found" in read_result.error

    delete_result = sandbox.delete(str(real_file))
    assert delete_result.error is not None
    assert real_file.read_text() == "real host content, must not leak\n"


class _ToolCapableFakeModel(FakeListChatModel):
    """`FakeListChatModel` doesn't implement `bind_tools` -- `BaseChatModel`
    raises `NotImplementedError` by default -- but `create_deep_agent`'s
    model node always calls `bind_tools` once `FilesystemMiddleware`
    registers its filesystem tools. This thin override makes the fake model
    accept (and ignore) tool binding while still returning its scripted
    plain-text response with no tool calls: exactly enough to prove the
    backend wiring works end-to-end without needing the model to actually
    use a tool.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def test_create_deep_agent_smoke_test(sandbox: MicroVMSandbox) -> None:
    """A real deepagents agent, built against this backend, completes one
    invocation without raising -- proving MicroVMSandbox implements
    SandboxBackendProtocol well enough for FilesystemMiddleware to build all
    of its tools at construction time, with no real LLM API key involved.
    """
    model = _ToolCapableFakeModel(responses=["Hello from the fake model."])

    agent = create_deep_agent(model=model, backend=sandbox)
    result = agent.invoke({"messages": [{"role": "user", "content": "Hi there"}]})

    assert result["messages"][-1].content == "Hello from the fake model."


def test_capture_offload_only_where_absolute_redirects_stay_in_the_sandbox() -> None:
    """deepagents' execute wrapper writes to /large_tool_results/ with a shell redirect (see LocalTransport)."""
    from langchain_digitalocean_microvm.transport.exec import ExecTransport
    from langchain_digitalocean_microvm.transport.local import LocalTransport

    assert MicroVMSandbox(microvm_id="a", transport=ExecTransport()).enable_capture_offload is True
    assert MicroVMSandbox(microvm_id="b", transport=LocalTransport()).enable_capture_offload is False


def test_relative_paths_are_invalid(sandbox: MicroVMSandbox) -> None:
    """deepagents' protocol (and langchain_tests' standard suite): relative paths -> invalid_path."""
    assert [r.error for r in sandbox.upload_files([("rel.txt", b"x")])] == ["invalid_path"]
    download = sandbox.download_files(["rel/path.txt"])[0]
    assert (download.content, download.error) == (None, "invalid_path")


def test_directory_paths_report_is_directory(sandbox: MicroVMSandbox) -> None:
    sandbox.execute("mkdir -p /somedir")
    download = sandbox.download_files(["/somedir"])[0]
    assert (download.content, download.error) == (None, "is_directory")
    assert [r.error for r in sandbox.upload_files([("/somedir", b"x")])] == ["is_directory"]
