"""deepagents sandbox backend adapter for one attached MicroVM.

MicroVMSandbox implements deepagents.backends.sandbox.BaseSandbox by
delegating command execution and file transfer to a pluggable
langchain_digitalocean_microvm.transport.base.SandboxTransport (ssh/http/local) -- this
class never talks to the MicroVM API or a concrete transport SDK directly,
so it stays correct regardless of which transport is selected at runtime.
"""

from __future__ import annotations

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from langchain_digitalocean_microvm.transport.base import (
    SandboxTransport,
    TransportExecResult,
)
from langchain_digitalocean_microvm.transport.errors import (
    TransportError,
    TransportFileNotFoundError,
    TransportIsADirectoryError,
    TransportPermissionError,
)


def _execute_response(result: TransportExecResult) -> ExecuteResponse:
    """Combine stdout+stderr into one ExecuteResponse, mirroring LangSmithSandbox._execute_response()."""
    output = result.stdout or ""
    if result.stderr:
        output += "\n" + result.stderr if output else result.stderr
    return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=False)


class MicroVMSandbox(BaseSandbox):
    """deepagents sandbox backend for one running MicroVM, reached via a SandboxTransport."""

    # MicroVM's base OCI image is assumed to ship a POSIX sh plus coreutils
    # (head/tail/wc/mkdir/rm) -- the same assumption BaseSandbox's generated
    # ls/read/grep/glob/edit scripts already require (python3 + coreutils).
    enable_capture_offload = True

    def __init__(
        self,
        *,
        microvm_id: str,
        transport: SandboxTransport,
        default_timeout: int = 1800,
    ) -> None:
        self._id = microvm_id
        self._transport = transport
        self._default_timeout = default_timeout
        # deepagents' execute wrapper redirects output to an absolute path,
        # which only a real MicroVM contains (SandboxTransport.supports_capture_offload).
        self.enable_capture_offload = getattr(transport, "supports_capture_offload", True)

    @property
    def id(self) -> str:
        return self._id

    @property
    def transport(self) -> SandboxTransport:
        """The channel this sandbox runs commands and moves files through (e.g. to check
        `EndpointTransport.uses_fallback`)."""
        return self._transport

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run `command` over the transport and return combined stdout+stderr."""
        result = self._transport.run(command, timeout=timeout or self._default_timeout)
        return _execute_response(result)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        """Async version of `execute`, using the transport's native async path."""
        result = await self._transport.arun(command, timeout=timeout or self._default_timeout)
        return _execute_response(result)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload each file via the transport, catching failures per-file for partial success."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._transport.put_file(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except TransportFileNotFoundError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except TransportPermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except TransportIsADirectoryError:
                responses.append(FileUploadResponse(path=path, error="is_directory"))
            except TransportError as exc:
                responses.append(FileUploadResponse(path=path, error=str(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download each file via the transport, catching failures per-file for partial success."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            try:
                content = self._transport.get_file(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except TransportFileNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except TransportPermissionError:
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
            except TransportIsADirectoryError:
                responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
            except TransportError as exc:
                responses.append(FileDownloadResponse(path=path, content=None, error=str(exc)))
        return responses

    def close(self) -> None:
        self._transport.close()
