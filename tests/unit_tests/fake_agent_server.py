"""Test doubles for the gateway: an in-memory Agent Protocol server and a host manager.

FakeAgentServer answers the subset of the Agent Protocol the gateway forwards
(threads, runs, cancel, /ok, /metrics) exactly as the in-memory Agent Server
does where it matters: 401 without the host token, `thread_id` honoured on
create, 404 when cancelling a terminal run, runs stay `pending` until a test
calls `complete()`/`fail()`. FakeHosts is a HostManager whose hosts are those
servers, reachable through `make_transport()` (an httpx.MockTransport that
routes by hostname), so gateway tests never open a socket to a host.

A server given a `workdir` also serves the agent-host image's artifact routes:
the real images/agent-host/artifacts_app.py, called in-process over ASGI.
Without one it answers those paths like an Agent Server built from an image
that lacks them (a plain 404).
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from functools import cache
from pathlib import Path
from typing import Any

import httpx

from langchain_digitalocean_microvm.agent_host.models import (
    HostHandle,
    HostSpec,
    HostStatus,
)

TERMINAL = {"success", "error", "timeout", "interrupted"}
FAKE_HOST_SUFFIX = ".fakehost"
ARTIFACTS_APP_PATH = Path(__file__).resolve().parents[2] / "images" / "agent-host" / "artifacts_app.py"
_FORWARDED = ("x-api-key", "range")
_DROPPED = ("content-length", "content-encoding", "transfer-encoding")


@cache
def artifacts_app_module() -> Any:
    """images/agent-host/artifacts_app.py, imported by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("agent_host_artifacts_app_under_test", ARTIFACTS_APP_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class FakeAgentServer:
    def __init__(self, token: str, *, workdir: Path | None = None) -> None:
        self.token = token
        self.artifact_requests: list[dict[str, str]] = []
        self._artifacts: httpx.AsyncClient | None = None
        if workdir is not None:
            async def expected_token() -> str:
                return self.token

            app = artifacts_app_module().create_app(workdir=str(workdir), expected_token=expected_token)
            self._artifacts = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent-host")
        self.threads: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.dead = False
        self.deleted = False
        self.paused = False
        self.requests: list[tuple[str, str]] = []

    # -- test controls ---------------------------------------------------

    def complete(self, run_id: str, content: str) -> None:
        run = self.runs[run_id]
        run["status"] = "success"
        thread = self.threads[run["thread_id"]]
        thread["values"]["messages"].append({"type": "ai", "content": content})
        thread.update(status="idle", interrupts={})

    def pause_for_human(self, run_id: str, value: Any, interrupt_id: str = "interrupt-1") -> None:
        """The run hit `interrupt(value)`: it ends `success`, its thread waits (`interrupted`) with the interrupt."""
        run = self.runs[run_id]
        run["status"] = "success"
        self.threads[run["thread_id"]].update(status="interrupted", interrupts={"node": [{"id": interrupt_id, "value": value}]})

    def fail(self, run_id: str, error: str) -> None:
        run = self.runs[run_id]
        run["status"] = "error"
        run["error"] = error

    def active_runs(self) -> list[dict[str, Any]]:
        return [r for r in self.runs.values() if r["status"] not in TERMINAL]

    # -- protocol --------------------------------------------------------

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if self.dead:
            raise httpx.ConnectError("connection refused", request=request)
        if self.deleted:  # what DigitalOcean's edge answers for a deleted MicroVM
            return httpx.Response(404, text="<!DOCTYPE html><title>Not found</title>", headers={"content-type": "text/html"})
        if self.paused:
            raise httpx.ReadTimeout("paused host does not answer", request=request)
        path = request.url.path
        self.requests.append((request.method, path))
        parts = [p for p in path.split("/") if p]
        if path.startswith("/microvm/artifacts/") and self._artifacts is not None:
            headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARDED}
            self.artifact_requests.append(headers)
            answer = await self._artifacts.request(request.method, request.url.raw_path.decode(), headers=headers)
            return httpx.Response(answer.status_code, content=answer.content,
                                  headers={k: v for k, v in answer.headers.items() if k.lower() not in _DROPPED})

        if path == "/ok":
            return httpx.Response(200, json={"ok": True})
        if path == "/metrics":
            active = self.active_runs()
            return httpx.Response(
                200,
                json={
                    "queue": {
                        "n_pending": sum(r["status"] == "pending" for r in active),
                        "n_running": sum(r["status"] == "running" for r in active),
                    },
                    "workers": {},
                },
            )
        if request.headers.get("x-api-key") != self.token:
            return httpx.Response(401, json={"detail": "invalid or missing host token"})

        body = json.loads(request.content) if request.content else {}
        if request.method == "POST" and parts == ["threads"]:
            thread_id = str(body.get("thread_id") or uuid.uuid4())
            if thread_id in self.threads and body.get("if_exists") != "do_nothing":
                return httpx.Response(409, json={"detail": "exists"})
            self.threads.setdefault(thread_id, {"thread_id": thread_id, "values": {"messages": []}, "status": "idle"})
            return httpx.Response(200, json=self.threads[thread_id])

        if len(parts) >= 2 and parts[0] == "threads":
            thread_id = parts[1]
            thread = self.threads.get(thread_id)
            if thread is None:
                return httpx.Response(404, json={"detail": "thread not found"})
            if len(parts) == 2 and request.method == "GET":
                return httpx.Response(200, json=thread)
            if len(parts) == 3 and parts[2] == "runs" and request.method == "POST":
                if body.get("multitask_strategy") == "interrupt":
                    for run in self.active_runs():
                        if run["thread_id"] == thread_id:
                            run["status"] = "interrupted"
                run_id = str(uuid.uuid4())
                run = {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "assistant_id": body.get("assistant_id"),
                    "status": "pending",
                    "created_at": "2026-09-03T00:01:00Z",
                    "updated_at": "2026-09-03T00:01:00Z",
                    "metadata": {},
                    "kwargs": {"webhook": body.get("webhook"), "input": body.get("input"), "command": body.get("command")},
                    "multitask_strategy": body.get("multitask_strategy", "reject"),
                }
                self.runs[run_id] = run
                thread.update(status="busy", interrupts={})
                for message in (body.get("input") or {}).get("messages", []):
                    thread["values"]["messages"].append({"type": "human", "content": message.get("content", "")})
                return httpx.Response(200, json=run)
            if len(parts) == 4 and parts[2] == "runs" and request.method == "GET":
                run = self.runs.get(parts[3])
                if run is None or run["thread_id"] != thread_id:
                    return httpx.Response(404, json={"detail": "run not found"})
                return httpx.Response(200, json=run)
            if len(parts) == 5 and parts[2] == "runs" and parts[4] == "cancel" and request.method == "POST":
                run = self.runs.get(parts[3])
                if run is None or run["status"] in TERMINAL:
                    return httpx.Response(404, json={"detail": "No matching runs to cancel."})
                run["status"] = "interrupted"
                return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": f"unhandled {request.method} {path}"})


class FakeHosts:
    """HostManager double whose hosts are FakeAgentServers."""

    def __init__(self, workdir_root: Path | None = None) -> None:
        """`workdir_root`: give every host a work directory (root/<host id>) and the artifact routes."""
        self.workdir_root = workdir_root
        self.servers: dict[str, FakeAgentServer] = {}
        self.status: dict[str, HostStatus] = {}
        self.graphs: dict[str, str] = {}
        self.booted_at: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self._boots = 0

    @staticmethod
    def url_for(host_id: str) -> str:
        return f"http://{host_id}{FAKE_HOST_SUFFIX}"

    def _handle(self, host_id: str) -> HostHandle:
        return HostHandle(
            host_id=host_id,
            microvm_id=f"fake-{host_id}",
            graph_id=self.graphs.get(host_id, ""),
            url=self.url_for(host_id),
            token=self.servers[host_id].token,
            status=self.status[host_id],
            booted_at=self.booted_at[host_id],
        )

    def _boot(self, host_id: str) -> None:
        self._boots += 1
        self.booted_at[host_id] = f"2026-09-03T00:00:{self._boots:02d}Z"  # always older than run starts at 00:01
        self.status[host_id] = HostStatus.READY

    def workdir(self, host_id: str) -> Path | None:
        if self.workdir_root is None:
            return None
        path = (self.workdir_root / host_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- HostManager surface --------------------------------------------

    def get_or_create(self, *, host_id: str, spec: HostSpec) -> HostHandle:
        self.calls.append(("get_or_create", host_id))
        self.graphs[host_id] = spec.graph_id
        server = self.servers.get(host_id)
        if server is None:
            self.servers[host_id] = FakeAgentServer(token=f"tok-{host_id}", workdir=self.workdir(host_id))
            self._boot(host_id)
        elif server.dead or self.status[host_id] in (HostStatus.FAILED, HostStatus.DESTROYED):
            server.dead = False  # respawn in place: runs stay as they were (orphans)
            self._boot(host_id)
        elif self.status[host_id] == HostStatus.PAUSED:
            server.paused = False
            self.status[host_id] = HostStatus.READY
        return self._handle(host_id)

    def resume(self, *, host_id: str, spec: HostSpec | None = None) -> HostHandle:
        self.calls.append(("resume", host_id))
        self.servers[host_id].paused = False
        self.status[host_id] = HostStatus.READY
        return self._handle(host_id)

    def pause(self, *, host_id: str) -> None:
        self.calls.append(("pause", host_id))
        self.servers[host_id].paused = True
        self.status[host_id] = HostStatus.PAUSED

    def delete(self, *, host_id: str) -> None:
        self.calls.append(("delete", host_id))
        self.servers.pop(host_id, None)
        self.status.pop(host_id, None)

    def list_hosts(self, *, graph_id: str | None = None) -> list[HostHandle]:
        return [self._handle(h) for h in self.servers
                if h in self.status and (graph_id is None or self.graphs.get(h) == graph_id)]

    def is_idle(self, handle: HostHandle) -> bool:
        server = self.servers.get(handle.host_id)
        return server is not None and not server.active_runs()

    # -- test controls ---------------------------------------------------

    def kill(self, host_id: str) -> None:
        """The VM vanished: connections fail and the inventory says FAILED."""
        self.servers[host_id].dead = True
        self.status[host_id] = HostStatus.FAILED

    def delete_outside(self, host_id: str) -> None:
        """The MicroVM was deleted behind the gateway's back: the edge answers 404, the inventory has no entry."""
        self.servers[host_id].deleted = True
        self.status.pop(host_id, None)

    def restart(self, host_id: str) -> None:
        """The server process restarted inside the VM: runs in flight are orphaned."""
        self._boots += 1
        self.booted_at[host_id] = f"2026-09-03T00:02:{self._boots:02d}Z"  # newer than run starts at 00:01

    def spawned(self) -> list[str]:
        return [h for op, h in self.calls if op == "get_or_create"]

    def paused(self) -> list[str]:
        return [h for op, h in self.calls if op == "pause"]

    def resumed(self) -> list[str]:
        return [h for op, h in self.calls if op == "resume"]


def make_transport(hosts: FakeHosts) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host_id = (request.url.host or "").removesuffix(FAKE_HOST_SUFFIX)
        server = hosts.servers.get(host_id)
        if server is None:
            raise httpx.ConnectError(f"no such host {host_id!r}", request=request)
        return await server.handle(request)

    return httpx.MockTransport(handler)
