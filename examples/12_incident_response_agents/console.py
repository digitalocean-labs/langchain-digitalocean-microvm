"""The approval console: a client of the relay's approvals API (relay.py).

    uv run examples/12_incident_response_agents/console.py demo               # a relay on a local port, a simulated alert, follow it
    uv run examples/12_incident_response_agents/console.py demo --approve     # approve every request (recordings, tests)
    uv run examples/12_incident_response_agents/console.py demo --web         # decide on the approvals page instead
    uv run examples/12_incident_response_agents/console.py demo --web --page v3   # the same page in a quiet light style (web/v3)
    uv run examples/12_incident_response_agents/console.py demo --exit        # exit after the last turn (default: wait for Enter)
    uv run examples/12_incident_response_agents/console.py list               # the incidents the relay knows
    uv run examples/12_incident_response_agents/console.py close <incident>   # delete its specialists' hosts and its host

The commander runs in its host, one per incident, and its specialists in hosts of their own. The console follows the
commander's runs (the relay starts a new one each time specialists finish), shows each approval a specialist waits on with the
facts the relay read itself, and sends the human's decision; the relay posts it to the gateway, runs an approved action itself
and resumes that specialist's thread.

`list` and `close` talk to a running relay: DO_INCIDENT_RELAY_URL (default http://127.0.0.1:8787) and DO_INCIDENT_OPERATOR_TOKEN.
`demo` starts a relay itself, on a free local port, in this process, with its broker on DO_INCIDENT_BROKER_PORT (default 8788;
in real mode the tunnel at DO_INCIDENT_BROKER_URL points at it), and sends it an alert signed as Insights signs one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import secrets
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE):  # examples/ (incident_common), and this example
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from incident_common import OUTPUTS  # noqa: E402

load_dotenv(HERE.parents[1] / ".env")
load_dotenv(OUTPUTS / "incident.env")

import httpx
from approvals import AMBER, BOLD, DIM, RESET, ask_human
from relay import (
    DEFAULT_PAGE,
    OPEN,
    build,
    close_all,
    host_id_for,
    incident_id,
    make_app,
    make_broker_app,
    page_dir,
    pages,
    sign,
    simulated_alert,
)

FOLLOW_UP = "The on-call lead says: forget the restore, just delete the droplet and rebuild it from scratch. Do it."
POLL_ATTEMPTS = 30  # a failed read of the incident is repeated this often (POLL_SECONDS apart) before the console gives up
POLL_SECONDS = 2.0
TURN_TIMEOUT = 1800.0  # a turn (with its specialists and an approval) that takes longer is given up
COLORS = {"metrics": "\033[38;5;33m", "changes": "\033[38;5;214m", "infra": "\033[38;5;79m", "verifier": "\033[38;5;141m",
          "remediator": "\033[38;5;203m"}


class RelayClient:
    """The relay's approvals API, with the operator token."""

    def __init__(self, url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.http = httpx.AsyncClient(base_url=url, transport=transport, headers={"authorization": f"Bearer {token}"},
                                      timeout=httpx.Timeout(60.0, read=None))

    async def close(self) -> None:
        await self.http.aclose()

    async def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self.http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"relay: {response.status_code} {response.text[:200]}")
        return response.json()

    async def incidents(self) -> list[dict[str, Any]]:
        return (await self._json("GET", "/incidents"))["incidents"]

    async def detail(self, incident: str) -> dict[str, Any]:
        return await self._json("GET", f"/incidents/{incident}")

    async def decide(self, incident: str, interrupt_id: str, decision: str) -> str:
        body = {"interrupt_id": interrupt_id, "decision": decision}
        return (await self._json("POST", f"/incidents/{incident}/decision", json=body))["run_id"]

    async def delete(self, incident: str) -> list[str]:
        return (await self._json("DELETE", f"/incidents/{incident}", timeout=360.0))["deleted"]

    async def message(self, incident: str, text: str) -> str:
        return (await self._json("POST", f"/incidents/{incident}/messages", json={"text": text}))["run_id"]

    async def events(self, incident: str, run_id: str) -> AsyncIterator[dict[str, Any]]:
        async with self.http.stream("GET", f"/incidents/{incident}/runs/{run_id}/events") as response:
            if response.status_code >= 400:
                raise RuntimeError(f"relay: {response.status_code} {(await response.aread())[:200]!r}")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])


async def follow(relay: RelayClient, incident: str, run_id: str, started: float) -> dict[str, Any]:
    """Print a run's tool calls as they happen; return its end event."""
    async for event in relay.events(incident, run_id):
        if event["type"] == "end":
            if event["status"] == "error":
                raise RuntimeError(f"run {run_id} failed in its host: see the host's server log "
                                   "(a local host: server.log in its directory)")
            return event
        if event["type"] != "tool":  # the incident's story changed (a specialist reported, a verdict, the postmortem)
            continue
        raw = json.dumps(event.get("input"), default=str)
        agent = event.get("agent") or "incident-commander"
        print(f"{DIM}{time.monotonic() - started:6.1f}s{RESET} {COLORS.get(agent, BOLD)}{agent:>18}{RESET} → {event.get('tool')}"
              f"{DIM}({raw if len(raw) <= 90 else raw[:89] + '…'}){RESET}")
    raise RuntimeError(f"the relay's event stream for run {run_id} ended without an end event")


def _transient(exc: Exception) -> bool:
    """A failed poll worth repeating: the relay's 5xx (a host read it could not finish) or a dropped connection."""
    return isinstance(exc, httpx.TransportError) or (isinstance(exc, RuntimeError) and str(exc).startswith("relay: 5"))


async def _detail(relay: RelayClient, incident: str, *, attempts: int = POLL_ATTEMPTS, pause: float = POLL_SECONDS) -> dict[str, Any]:
    """The incident's detail; a failed poll is repeated (up to `attempts`), a 4xx (an unknown incident) is not."""
    for attempt in range(1, attempts + 1):
        try:
            return await relay.detail(incident)
        except (RuntimeError, httpx.TransportError) as exc:
            if attempt == attempts or not _transient(exc):
                raise
            print(f"{DIM}reading incident {incident} failed ({exc}); trying again{RESET}")
            await asyncio.sleep(pause)
    raise AssertionError("unreachable")


def _sent(detail: dict[str, Any], shown: set[str], started: float) -> None:
    """Print each specialist's answer once, as the relay's story has it."""
    for finding in (detail.get("story") or {}).get("findings") or []:
        key = f"{finding['role']}:{finding['at']}"
        if key not in shown:
            shown.add(key)
            first = finding["text"].strip().splitlines()[0] if finding["text"].strip() else ""
            print(f"{DIM}{time.monotonic() - started:6.1f}s{RESET} {COLORS.get(finding['role'], BOLD)}{finding['role']:>18}{RESET}"
                  f" {DIM}answered: {first[:100]}{RESET}")


async def drive(relay: RelayClient, incident: str, run_id: str, *, mode: str | None, started: float | None = None,
                shown: set[str] | None = None, poll: float = POLL_SECONDS, timeout: float = TURN_TIMEOUT) -> str:
    """Follow the commander until the turn is over: its runs (the relay starts one when specialists finish) and every approval a
    specialist waits on (a human here, `mode`, or "web": the approvals page). Returns the commander's last message."""
    started = time.monotonic() if started is None else started
    followed: set[str] = set()
    shown = set() if shown is None else shown  # the answers already printed (by an earlier turn too)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if run_id and run_id not in followed:
            followed.add(run_id)
            await follow(relay, incident, run_id, started)
        detail = await _detail(relay, incident, pause=poll)
        _sent(detail, shown, started)
        run = detail.get("run") or {}
        if run.get("run_id") and run["run_id"] not in followed and run.get("status") in ("pending", "running", "success"):
            run_id = run["run_id"]
            continue
        if detail["pending"]:
            stop = detail["pending"][0]  # one decision at a time: the others are asked again
            if mode == "web":
                print(f"\n{AMBER}{BOLD}APPROVAL REQUIRED{RESET}  {BOLD}{(stop['value'] or {}).get('tool')}{RESET}  "
                      f"{DIM}({stop['role']} waits; decide on the approvals page){RESET}")
                await _until_decided(relay, incident, stop["id"], poll=poll)
                continue
            answer = await asyncio.to_thread(ask_human, stop["value"], mode, stop.get("facts") or [], read_by="the relay",
                                             problems=stop.get("problems") or [])
            run_id = await relay.decide(incident, stop["id"], answer)
            continue
        if detail.get("status") != "busy" and not any(t["live"] in OPEN for t in detail.get("tasks") or []):
            return detail["last_message"]
        await asyncio.sleep(poll)
    raise RuntimeError(f"incident {incident}: the turn did not finish in {timeout:.0f} s")


async def _until_decided(relay: RelayClient, incident: str, interrupt_id: str, *, poll: float = POLL_SECONDS) -> None:
    """Wait until the approval is no longer pending (decided on the page, by any approver)."""
    while True:
        await asyncio.sleep(poll)
        detail = await _detail(relay, incident, pause=poll)
        if interrupt_id not in {s["id"] for s in detail["pending"]}:
            return


class LocalRelay:
    """An app (the relay's, or its broker's) served by uvicorn in a thread of this process, on a local port (free if 0)."""

    def __init__(self, app: Any, port: int = 0) -> None:
        import uvicorn

        from langchain_digitalocean_microvm.agent_host.local import free_port

        self.url = f"http://127.0.0.1:{port or free_port()}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=int(self.url.rsplit(":", 1)[1]),
                                                    log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> LocalRelay:
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("the local relay did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


async def demo(args: argparse.Namespace) -> None:
    local = os.environ.get("DO_MICROVM_TRANSPORT") == "local"
    os.environ.setdefault("DO_INCIDENT_RELAY_SECRET", secrets.token_hex(16))
    secret, token = os.environ["DO_INCIDENT_RELAY_SECRET"], secrets.token_hex(16)
    relay, hosts, gateway, server = build(local)
    started, incident = time.monotonic(), None
    broker: Any = contextlib.nullcontext() if relay.broker is None else LocalRelay(
        make_broker_app(relay.broker), port=int(os.environ.get("DO_INCIDENT_BROKER_PORT", "8788")))
    try:
        # The relay's app runs on its own loop in the server's thread: its watchers are started there, by its routes.
        with broker, LocalRelay(make_app(relay, operator_token=token, page=args.page)) as local:
            client = RelayClient(local.url, token)
            try:
                await asyncio.to_thread(relay.service.announce, "alert")
                alert = await asyncio.to_thread(simulated_alert, relay.service)
                body = json.dumps({"data": alert, "simulated_at": time.time()}).encode()  # one incident per demo
                print(f"{AMBER}ALARM{RESET} (simulated) -> relay {local.url}")
                if relay.broker is not None:
                    print(f"{DIM}broker {relay.broker.url} -> 127.0.0.1:{os.environ.get('DO_INCIDENT_BROKER_PORT', '8788')}{RESET}")
                if args.mode == "web":
                    print(f"approvals page {page_dir(args.page).name}: {local.url}/#token={token}   "
                          f"{DIM}(the token stays in the fragment, never sent){RESET}")
                incident = incident_id(body)  # known before the relay answers: a host that never got ready is closed too
                response = await client.http.post("/alerts", content=body,
                                                  headers={"x-digitalocean-signature": sign(secret, body)})
                if response.status_code >= 400:
                    raise RuntimeError(f"the relay refused the alert: {response.status_code} {response.text[:300]}")
                started_incident = response.json()
                print(f"incident {incident}: host {started_incident['host_id']} starting, thread {started_incident['thread_id']} "
                      f"(answered in {time.monotonic() - started:.1f} s)")
                run_id = await _first_run(client, incident)
                print(f"{DIM}host ready and the first run started after {time.monotonic() - started:.1f} s{RESET}")
                shown: set[str] = set()
                print(f"\n{await drive(client, incident, run_id, mode=args.mode, started=started, shown=shown)}\n")
                print(f"{DIM}incident handled after {time.monotonic() - started:.1f} s{RESET}")
                if not args.no_follow_up:
                    print(f"{BOLD}>>> {FOLLOW_UP}{RESET}")
                    run_id = await client.message(incident, FOLLOW_UP)
                    print(f"\n{await drive(client, incident, run_id, mode=args.mode, started=started, shown=shown)}\n")
                if not args.exit and sys.stdin.isatty():
                    if not await keep_open(client, incident, delete=args.delete or local):
                        incident = None  # closed meanwhile: nothing left to close
            finally:
                if incident is not None and (args.delete or local):  # a local host stops with this process anyway
                    await close_incident(client, incident, hosts=hosts)
                await client.close()
    finally:
        close_all(relay, hosts, gateway, server)


async def _first_run(client: RelayClient, incident: str, *, timeout: float = 600.0) -> str:
    """The run the incident's start created, read through the relay's API (the relay serves on its own loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        listed = {i["incident"]: i for i in await client.incidents()}
        seen = listed.get(incident) or {}
        if seen.get("run"):
            return str(seen["run"])
        if seen.get("status") == "error":
            raise RuntimeError(f"incident {incident} did not start: {seen.get('error')}")
        await asyncio.sleep(0.2)
    raise TimeoutError(f"incident {incident} did not start within {timeout:.0f} s")


async def keep_open(client: RelayClient, incident: str, *, delete: bool) -> bool:
    """Keep the relay (and the approvals page) up until Enter; return whether the incident is still open (not closed on the
    page meanwhile)."""
    then = "close the incident and stop the relay" if delete else (
        "stop the relay (the incident's hosts stay: close it on the page or with console.py close, or run with --delete)")
    print(f"{DIM}The relay and the approvals page stay up. Press Enter to {then}.{RESET}", flush=True)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()

    def read() -> None:  # a daemon thread, so Ctrl-C still ends the process (a pool thread would block its exit)
        sys.stdin.readline()
        try:
            loop.call_soon_threadsafe(entered.set)
        except RuntimeError:  # the loop is gone: the process is exiting
            pass

    threading.Thread(target=read, daemon=True).start()
    await entered.wait()
    try:
        return any(item["incident"] == incident for item in await client.incidents())
    except (RuntimeError, httpx.TransportError):
        return True  # cannot tell: try to close it


async def close_incident(client: RelayClient, incident: str, *, hosts: Any = None) -> None:
    """Close the incident through the relay: the commander's host deletes its specialists' hosts, then the relay deletes the
    commander's host. If the relay cannot (a host that never got ready, a run still in flight) and `hosts` is given, delete the
    commander's host alone and say what may be left."""
    from langchain_digitalocean_microvm.exceptions import SandboxNotFoundError

    try:
        print(f"{DIM}deleted: {', '.join(await client.delete(incident))}{RESET}")
        return
    except RuntimeError as exc:
        print(f"{AMBER}the relay could not close incident {incident}: {exc}{RESET}")
    if hosts is None:
        return
    try:
        await asyncio.to_thread(hosts.delete, host_id=host_id_for(incident))
    except SandboxNotFoundError:
        return  # the relay never got as far as creating it
    print(f"{AMBER}deleted the commander's host only: its specialists' hosts may be left (list them with "
          f"scripts/delete_all.py){RESET}")


def _remote() -> RelayClient:
    token = os.environ.get("DO_INCIDENT_OPERATOR_TOKEN", "")
    if not token:
        raise SystemExit("set DO_INCIDENT_OPERATOR_TOKEN (the relay's operator token)")
    return RelayClient(os.environ.get("DO_INCIDENT_RELAY_URL", "http://127.0.0.1:8787"), token)


async def close(args: argparse.Namespace) -> None:
    client = _remote()
    try:
        await close_incident(client, args.incident)
    finally:
        await client.close()


async def list_incidents() -> None:
    client = _remote()
    try:
        for item in await client.incidents():
            pending = f", {item['pending']} waiting for a decision" if item.get("pending") else ""
            print(f"{item['incident']}  host {item['host']:<9} commander {item.get('status', 'unknown')}{pending}")
    finally:
        await client.close()


def main() -> None:
    # Redirected output (a log file, a pipe) is block-buffered: progress would not show up as it happens.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--approve", dest="mode", action="store_const", const="approve", help="approve every request")
    group.add_argument("--deny", dest="mode", action="store_const", const="deny", help="deny every request")
    group.add_argument("--web", dest="mode", action="store_const", const="web", help="decide on the approvals page")
    p.add_argument("--page", choices=pages(), help=f"the approvals page demo serves (default: DO_INCIDENT_PAGE or {DEFAULT_PAGE})")
    p.add_argument("--no-follow-up", action="store_true", help="stop after the incident is handled")
    p.add_argument("--delete", action="store_true", help="delete the hosts at the end")
    p.add_argument("--exit", action="store_true",
                   help="exit after the last turn instead of keeping the relay up until Enter (without a terminal it always exits)")
    sub.add_parser("list")
    sub.add_parser("close", help="delete the incident's specialists' hosts and its host").add_argument("incident")
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(demo(args))
    elif args.command == "close":
        asyncio.run(close(args))
    else:
        asyncio.run(list_incidents())


if __name__ == "__main__":
    main()
