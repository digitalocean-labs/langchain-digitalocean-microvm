"""The simulated world over HTTP, for examples whose agents run in separate processes (local mode only).

One process holds the `SimulatedWorld` and serves it; every other process reaches the same droplet, the same commits and the same
restore through it:

    GET  /health                 as service/app.py answers it (plus "droplet")
    GET  /metrics                one point a minute for the last 45 minutes, as service/app.py
    POST /actions/{tool}         {"arguments": {...}}: the local Action Gateway's action, run on the world
    POST /announce               {"what"}: the world writes its state to the server's event log
    GET  /sessions               the local gateway's sessions recorded here, as the service lists its sessions
    POST /sessions               {"sessionUrn", "name"}: record one;  DELETE /sessions/{urn}: forget it

`HttpService` (world.py) reads /health and /metrics as it reads the real droplet; `remote_actions(url)` gives a
`LocalActionGatewayClient` actions that run here; `WorldSessionsClient(url)` is a local gateway client whose sessions are
recorded here, so another process lists and deletes them (`list_sessions`, `delete_session`) as on the real gateway.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from incident_common.world import SimulatedWorld
from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    LocalAction,
    LocalActionGatewayClient,
)

LOOPBACK = "127.0.0.1"


def make_app(world: SimulatedWorld) -> Any:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    actions = world.actions()
    sessions: dict[str, dict[str, str]] = {}  # urn -> {"sessionUrn", "name", "createdAt"}

    async def health(request: Request) -> JSONResponse:
        return JSONResponse(world.health())

    async def metrics(request: Request) -> JSONResponse:
        return JSONResponse(world.metrics(45))

    async def action(request: Request) -> JSONResponse:
        name = request.path_params["tool"]
        if name not in actions:
            return JSONResponse({"error": f"no action {name}"}, status_code=404)
        body = await request.json()
        arguments = body.get("arguments") if isinstance(body, dict) else None
        return JSONResponse(actions[name].run(arguments if isinstance(arguments, dict) else {}))

    async def announce(request: Request) -> JSONResponse:
        body = await request.json()
        world.announce(str((body or {}).get("what") or "check"))
        return JSONResponse({"ok": True})

    async def list_sessions(request: Request) -> JSONResponse:
        return JSONResponse({"sessions": list(sessions.values())})

    async def add_session(request: Request) -> JSONResponse:
        body = await request.json()
        urn, name = str(body.get("sessionUrn") or ""), str(body.get("name") or "")
        if not urn or not name:
            return JSONResponse({"error": "sessionUrn and name are required"}, status_code=400)
        sessions[urn] = {"sessionUrn": urn, "name": name, "createdAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        return JSONResponse(sessions[urn])

    async def delete_session(request: Request) -> JSONResponse:
        sessions.pop(request.path_params["urn"], None)
        return JSONResponse({"ok": True})

    return Starlette(routes=[Route("/health", health, methods=["GET"]), Route("/metrics", metrics, methods=["GET"]),
                             Route("/actions/{tool}", action, methods=["POST"]), Route("/announce", announce, methods=["POST"]),
                             Route("/sessions", list_sessions, methods=["GET"]), Route("/sessions", add_session, methods=["POST"]),
                             Route("/sessions/{urn}", delete_session, methods=["DELETE"])])


def serve(world: SimulatedWorld, *, port: int = 0) -> tuple[str, Any]:
    """Serve `world` on a loopback port in a background thread; returns its URL and the server (`server.should_exit = True`)."""
    import socket

    import uvicorn

    if not port:
        with socket.socket() as s:
            s.bind((LOOPBACK, 0))
            port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(make_app(world), host=LOOPBACK, port=port, log_level="warning"))
    threading.Thread(target=server.run, name="simulated-world", daemon=True).start()
    url = f"http://{LOOPBACK}:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return url, server
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise RuntimeError("the simulated world did not start")


def remote_actions(url: str) -> dict[str, LocalAction]:
    """The local gateway's actions, as `SimulatedWorld.actions()` names and describes them, each run by the world server at `url`."""

    def runner(name: str):  # noqa: ANN202
        def run(arguments: Any) -> Any:
            return httpx.post(f"{url.rstrip('/')}/actions/{name}", json={"arguments": dict(arguments)}, timeout=30).json()
        return run

    return {name: LocalAction(a.description, runner(name), a.input_schema) for name, a in SimulatedWorld().actions().items()}


class WorldSessionsClient(LocalActionGatewayClient):
    """The local gateway, its sessions also recorded on the world server at `url`: a process that never created them (the
    relay) lists and deletes them there. Session ids are unique across processes (the local client's own are only per process)."""

    def __init__(self, url: str, actions: Mapping[str, LocalAction] | None = None) -> None:
        super().__init__(actions=actions)
        self.url = url.rstrip("/")

    def create_session(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        import secrets

        urn = super().create_session(name, tools, policy, actor_id)
        session = self.sessions.pop(urn)
        urn = f"{urn}-{secrets.token_hex(4)}"
        self.sessions[urn] = session
        httpx.post(f"{self.url}/sessions", json={"sessionUrn": urn, "name": name}, timeout=10).raise_for_status()
        return urn

    def delete_session(self, urn: str) -> None:
        super().delete_session(urn)
        try:
            httpx.delete(f"{self.url}/sessions/{urn}", timeout=10).raise_for_status()
        except httpx.ConnectError:  # the world server stopped: its record of the session went with it
            pass

    def list_sessions(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.url}/sessions", timeout=10)
        response.raise_for_status()
        found = [s for s in response.json().get("sessions") or [] if isinstance(s, dict)]
        return [s for s in found if prefix is None or str(s.get("name") or "").startswith(prefix)]
