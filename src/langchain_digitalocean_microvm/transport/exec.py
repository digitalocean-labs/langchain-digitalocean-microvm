"""Exec transport: run commands and move files through the MicroVMs exec API.

MicroVMs have no SSH and no inbound ports besides one HTTPS endpoint; the
only way to run something inside one is POST /v2/microvms/{id}/exec, which
runs argv (no shell) in the workload container with hard limits: 60 s per
command, 4 MiB of output, 1 MiB of request body, and 60 commands/min / 32
concurrent per team. This transport hides those limits behind
SandboxTransport:

- run(): every command goes through a small POSIX-sh wrapper that starts it
  in the background with stdout/stderr/exit code captured to files under
  /tmp/.microvm-exec/<job>/ (`job_root`), and waits up to `inline_wait`
  seconds (default 50, safely under the 60 s limit). A command that finishes
  in time costs exactly ONE exec call and returns normally. One that does
  not keeps running detached; the transport then polls the job directory
  every `poll_interval` seconds until it finishes or `timeout` elapses, at
  which point the job is killed and exit code 124 is returned.
- put_file()/get_file(): base64 through exec, in chunks sized to stay under
  the request and output limits. A file up to `read_chunk` bytes is read in
  one call.

Payloads (the command text, file chunks) travel base64-encoded as separate
positional arguments after the script (`sh -c <script> _ <part> <part> ...`),
never inside the script string: exec runs argv directly, and Linux rejects
any single argument over 128 KiB (MAX_ARG_STRLEN), while the API caps the
whole request at 1 MiB. Commands run with bash when the image has it, else sh.

The command text is handed to that shell as an argument (`bash -c "$cmd"`),
not stored in the job directory, so a recursive grep inside the VM never
finds the command's own pattern on disk. Only a command too long for one
argument (> `_INLINE_COMMAND_BYTES`) is written to `<job>/cmd` first.
"""

from __future__ import annotations

import base64
import logging
import secrets
import shlex
import time
from collections.abc import Callable

from langchain_digitalocean_microvm.client import DEFAULT_BASE_URL, MicroVMClient
from langchain_digitalocean_microvm.exceptions import (
    MicroVMClientError,
    MicroVMNotFoundError,
)
from langchain_digitalocean_microvm.models import ExecResult
from langchain_digitalocean_microvm.transport.base import (
    SandboxTransport,
    TransportCredentials,
    TransportExecResult,
)
from langchain_digitalocean_microvm.transport.errors import (
    TransportError,
    TransportFileNotFoundError,
    TransportIsADirectoryError,
    TransportPermissionError,
)

logger = logging.getLogger(__name__)

JOB_ROOT = "/tmp/.microvm-exec"
DETACHED = "__MICROVM_EXEC_DETACHED__"
PENDING = "__MICROVM_EXEC_PENDING__"
TIMEOUT_EXIT_CODE = 124
_EXIT_NOT_FOUND = 44
_EXIT_PERMISSION = 45
_EXIT_IS_DIR = 47
_ARG_CHARS = 96_000  # per positional argument: well under MAX_ARG_STRLEN (131072)
_PAYLOAD_CHARS = 880_000  # base64 chars per exec request: under the API's 1 MiB body limit
_INLINE_COMMAND_BYTES = 100_000  # decoded command run as `sh -c <cmd>`: one argv element, so < 128 KiB
_TRUNCATED_NOTE = "[output truncated by the MicroVM exec API at 4 MiB]"

_PICK_SHELL = 'if command -v bash >/dev/null 2>&1; then __sh=bash; else __sh=sh; fi'


def _split_args(encoded: str) -> list[str]:
    return [encoded[i : i + _ARG_CHARS] for i in range(0, len(encoded), _ARG_CHARS)] or [""]


def _wrapper(d: str, job: str, mode: str, wait: int) -> str:
    """Shell script (run via `sh -c`) that starts a job and waits up to `wait` seconds for it.

    `d` is the job directory. `mode`:
      "arg"      -- base64 command in the positional args; run as `sh -c "$cmd"`
      "file"     -- base64 command in the positional args; written to $d/cmd first
      "uploaded" -- $d/cmd already exists (uploaded with put_file)
    """
    if mode == "arg":
        prepare = "__cmd=$(printf '%s' \"$@\" | base64 -d) || exit 125"
        launch = '$__sh -c "$__cmd"'
    elif mode == "file":
        prepare = "printf '%s' \"$@\" | base64 -d > \"$d/cmd\" || exit 125"
        launch = '$__sh "$d/cmd"'
    else:
        prepare = ""
        launch = '$__sh "$d/cmd"'
    return f"""d={d}
mkdir -p "$d" || exit 125
{prepare}
{_PICK_SHELL}
set -m 2>/dev/null  # job control: the job gets its own process group, so a timeout can kill all of it
( {launch} > "$d/out" 2> "$d/err" < /dev/null; echo $? > "$d/rc.tmp"; mv "$d/rc.tmp" "$d/rc" ) > /dev/null 2>&1 < /dev/null &
echo $! > "$d/pid"
set +m 2>/dev/null  # the group exists now; without job control the shell prints no "Done" notices
__start=$(date +%s)
while [ ! -f "$d/rc" ]; do
  if [ $(( $(date +%s) - __start )) -ge {wait} ]; then echo '{DETACHED} {job}'; exit 0; fi
  sleep 0.05 2>/dev/null || sleep 1
done
cat "$d/out"; cat "$d/err" >&2; __rc=$(cat "$d/rc"); rm -rf "$d"; exit "$__rc"
"""


def _collect(d: str) -> str:
    return (
        f'd={d}; if [ -f "$d/rc" ]; then cat "$d/out"; cat "$d/err" >&2; __rc=$(cat "$d/rc"); '
        f'rm -rf "$d"; exit "$__rc"; else echo \'{PENDING}\'; fi'
    )


def _kill(d: str) -> str:
    # Kill the job's process group (it leads one, see `set -m` in _wrapper); fall
    # back to the pid and its direct children when job control was unavailable.
    return (
        f'd={d}; if [ -f "$d/pid" ]; then p=$(cat "$d/pid"); '
        f'kill -TERM -"$p" 2>/dev/null || {{ pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; }}; sleep 1; '
        f'kill -KILL -"$p" 2>/dev/null || {{ pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; }}; fi; '
        f'cat "$d/out" 2>/dev/null; cat "$d/err" >&2 2>/dev/null; rm -rf "$d"; exit 0'
    )


class ExecTransport(SandboxTransport):
    """SandboxTransport over the MicroVMs exec API. `connect(host=<microvm id>)`."""

    def __init__(
        self,
        client: MicroVMClient | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        inline_wait: float = 50.0,
        poll_interval: float = 3.0,
        write_chunk: int = 640 * 1024,
        read_chunk: int = 2 * 1024 * 1024,
        job_root: str = JOB_ROOT,
        job_id: Callable[[], str] = lambda: secrets.token_hex(8),
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """`client` and the timing/job-id hooks are test seams; production code passes nothing."""
        self._client = client
        self._owns_client = client is None
        self._base_url = base_url
        self._inline_wait = inline_wait
        self._poll_interval = poll_interval
        self._write_chunk = write_chunk
        self._read_chunk = read_chunk
        self._job_root = job_root.rstrip("/")
        self._job_id = job_id
        self._sleep = sleep
        self._monotonic = monotonic
        self._microvm_id: str | None = None

    # ------------------------------------------------------------------ connection

    def connect(
        self,
        *,
        host: str,
        port: int = 0,
        credentials: TransportCredentials,
        timeout: float | None = None,
    ) -> None:
        if self._client is None:
            if not credentials.api_token:
                raise TransportError("ExecTransport needs credentials.api_token (a microvm:update-scoped token)")
            self._client = MicroVMClient(credentials.api_token, base_url=self._base_url, max_retries=6)
        try:
            self._client.get(host)
        except MicroVMNotFoundError as exc:
            raise TransportError(f"MicroVM {host} does not exist") from exc
        except MicroVMClientError as exc:
            raise TransportError(f"Could not reach the MicroVMs API for {host}: {exc}") from exc
        self._microvm_id = host

    def close(self) -> None:
        self._microvm_id = None
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._microvm_id is not None

    # ------------------------------------------------------------------ commands

    def run(self, command: str, *, timeout: float | None = None) -> TransportExecResult:
        job = self._job_id()
        d = f"{self._job_root}/{job}"
        started = self._monotonic()
        wait = self._inline_wait if timeout is None else min(self._inline_wait, timeout)
        raw = command.encode()
        encoded = base64.b64encode(raw).decode()
        seconds = max(1, int(wait))
        if len(encoded) > _PAYLOAD_CHARS:
            self.put_file(f"{d}/cmd", raw)
            argv = ["sh", "-c", _wrapper(d, job, "uploaded", seconds)]
        else:
            mode = "arg" if len(raw) <= _INLINE_COMMAND_BYTES else "file"
            argv = ["sh", "-c", _wrapper(d, job, mode, seconds), "_", *_split_args(encoded)]
        result = self._exec(argv)
        if result.stdout.strip() != f"{DETACHED} {job}":
            return self._to_transport(result)

        logger.info("Command still running after %ss; polling it as a job", wait)
        while True:
            if timeout is not None and self._monotonic() - started >= timeout:
                partial = self._exec(["sh", "-c", _kill(d)])
                stderr = (partial.stderr + "\n" if partial.stderr else "") + f"[timed out after {timeout:g}s]"
                return TransportExecResult(stdout=partial.stdout, stderr=stderr, exit_code=TIMEOUT_EXIT_CODE)
            self._sleep(self._poll_interval)
            result = self._exec(["sh", "-c", _collect(d)])
            if result.stdout.strip() != PENDING:
                return self._to_transport(result)

    # ------------------------------------------------------------------ files

    def put_file(self, path: str, content: bytes) -> None:
        quoted = shlex.quote(path)
        chunks = [content[i : i + self._write_chunk] for i in range(0, len(content), self._write_chunk)] or [b""]
        for index, chunk in enumerate(chunks):
            encoded = base64.b64encode(chunk).decode()
            redirect = ">" if index == 0 else ">>"
            prepare = (
                f'mkdir -p "$(dirname {quoted})" 2>/dev/null; [ -d "$(dirname {quoted})" ] || exit {_EXIT_NOT_FOUND}; '
                f'[ ! -e {quoted} ] || [ -w {quoted} ] || exit {_EXIT_PERMISSION}; '
                if index == 0
                else ""
            )
            script = f"{prepare}printf '%s' \"$@\" | base64 -d {redirect} {quoted}"
            self._raise_for_file_error(self._exec(["sh", "-c", script, "_", *_split_args(encoded)]), path, "write")

    def get_file(self, path: str) -> bytes:
        quoted = shlex.quote(path)
        head = (
            f"[ -e {quoted} ] || exit {_EXIT_NOT_FOUND}; [ -d {quoted} ] && exit {_EXIT_IS_DIR}; "
            f"[ -r {quoted} ] || exit {_EXIT_PERMISSION}; "
            f"__n=$(wc -c < {quoted} | tr -d ' '); echo \"$__n\"; "
            f'if [ "$__n" -le {self._read_chunk} ]; then base64 < {quoted}; fi'
        )
        first = self._exec(["sh", "-c", head])
        self._raise_for_file_error(first, path, "read")
        size_line, _, rest = first.stdout.partition("\n")
        try:
            size = int(size_line.strip())
        except ValueError as exc:
            raise TransportError(f"Unexpected size output reading {path}: {size_line!r}") from exc
        if size <= self._read_chunk:
            return base64.b64decode(rest)
        data = bytearray()
        blocks = (size + self._read_chunk - 1) // self._read_chunk
        for block in range(blocks):
            script = f"dd if={quoted} bs={self._read_chunk} skip={block} count=1 2>/dev/null | base64"
            part = self._exec(["sh", "-c", script])
            self._raise_for_file_error(part, path, "read")
            data += base64.b64decode(part.stdout)
        return bytes(data)

    # ------------------------------------------------------------------ internals

    def _exec(self, argv: list[str]) -> ExecResult:
        if self._client is None or self._microvm_id is None:
            raise TransportError("ExecTransport is not connected")
        try:
            return self._client.exec(self._microvm_id, argv)
        except MicroVMClientError as exc:
            raise TransportError(f"exec in MicroVM {self._microvm_id} failed: {exc}") from exc

    @staticmethod
    def _to_transport(result: ExecResult) -> TransportExecResult:
        stderr = result.stderr
        if result.truncated:
            stderr = f"{stderr}\n{_TRUNCATED_NOTE}" if stderr else _TRUNCATED_NOTE
        return TransportExecResult(stdout=result.stdout, stderr=stderr, exit_code=result.exit_code)

    @staticmethod
    def _raise_for_file_error(result: ExecResult, path: str, verb: str) -> None:
        if result.exit_code == 0:
            return
        detail = (result.stderr or result.stdout).strip()
        if result.exit_code == _EXIT_NOT_FOUND:
            raise TransportFileNotFoundError(f"cannot {verb} {path}: no such file or directory")
        if result.exit_code == _EXIT_PERMISSION or "Permission denied" in detail:
            raise TransportPermissionError(f"cannot {verb} {path}: permission denied")
        if result.exit_code == _EXIT_IS_DIR:
            raise TransportIsADirectoryError(f"cannot {verb} {path}: is a directory")
        raise TransportError(f"cannot {verb} {path}: exit code {result.exit_code}: {detail}")
