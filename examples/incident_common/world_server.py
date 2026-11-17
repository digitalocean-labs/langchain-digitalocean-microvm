"""The simulated world over HTTP, for examples whose agents run in separate processes (local mode only).

One process holds the `SimulatedWorld` and serves it; every other process reaches the same droplet, the same commits and the same
restore through it:

    GET  /health                 as service/app.py answers it (plus "droplet")
    GET  /metrics                one point a minute for the last 45 minutes, as service/app.py
    POST /actions/{tool}         {"arguments": {...}}: the local Action Gateway's action, run on the world
    POST /announce               {"what"}: the world writes its state to the server's event log
    GET  /endpoints              the local gateway's endpoints recorded here, as the service lists its endpoints
    POST /endpoints              {"id", "name"}: record one;  DELETE /endpoints/{id}: forget it

`HttpService` (world.py) reads /health and /metrics as it reads the real droplet; `remote_actions(url)` gives a
`LocalActionGatewayClient` actions that run here; `WorldEndpointsClient(url)` is a local gateway client whose endpoints are
recorded here, so another process lists and deletes them (`list_endpoints`, `delete_endpoint`) as on the real gateway.
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
    endpoints: dict[str, dict[str, str]] = {}  # id -> {"id", "name", "created_at"}

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

    async def list_endpoints(request: Request) -> JSONResponse:
        return JSONResponse({"endpoints": list(endpoints.values())})

    async def add_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        endpoint_id, name = str(body.get("id") or ""), str(body.get("name") or "")
        if not endpoint_id or not name:
            return JSONResponse({"error": "id and name are required"}, status_code=400)
        endpoints[endpoint_id] = {"id": endpoint_id, "name": name, "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        return JSONResponse(endpoints[endpoint_id])

    async def delete_endpoint(request: Request) -> JSONResponse:
        endpoints.pop(request.path_params["endpoint_id"], None)
        return JSONResponse({"ok": True})

    return Starlette(routes=[Route("/health", health, methods=["GET"]), Route("/metrics", metrics, methods=["GET"]),
                             Route("/actions/{tool}", action, methods=["POST"]), Route("/announce", announce, methods=["POST"]),
                             Route("/endpoints", list_endpoints, methods=["GET"]), Route("/endpoints", add_endpoint, methods=["POST"]),
                             Route("/endpoints/{endpoint_id}", delete_endpoint, methods=["DELETE"])])


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


class WorldEndpointsClient(LocalActionGatewayClient):
    """The local gateway, its endpoints also recorded on the world server at `url`: a process that never created them (the
    relay) lists and deletes them there. Endpoint ids are unique across processes (the local client's own are only per process)."""

    def __init__(self, url: str, actions: Mapping[str, LocalAction] | None = None) -> None:
        super().__init__(actions=actions)
        self.url = url.rstrip("/")

    def create_endpoint(self, name: str, tools: Sequence[str], policy: Mapping[str, Any], actor_id: str | None) -> str:
        import secrets

        endpoint_id = super().create_endpoint(name, tools, policy, actor_id)
        endpoint = self.endpoints.pop(endpoint_id)
        endpoint_id = f"{endpoint_id}-{secrets.token_hex(4)}"
        self.endpoints[endpoint_id] = endpoint
        httpx.post(f"{self.url}/endpoints", json={"id": endpoint_id, "name": name}, timeout=10).raise_for_status()
        return endpoint_id

    def delete_endpoint(self, endpoint_id: str) -> None:
        super().delete_endpoint(endpoint_id)
        try:
            httpx.delete(f"{self.url}/endpoints/{endpoint_id}", timeout=10).raise_for_status()
        except httpx.ConnectError:  # the world server stopped: its record of the endpoint went with it
            pass

    def list_endpoints(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.url}/endpoints", timeout=10)
        response.raise_for_status()
        found = [s for s in response.json().get("endpoints") or [] if isinstance(s, dict)]
        return [s for s in found if prefix is None or str(s.get("name") or "").startswith(prefix)]
