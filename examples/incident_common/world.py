"""The incident's world: a small service on a droplet, its commits, its snapshot and its metrics.

`SimulatedWorld` simulates it in memory, so the example runs with no account (`DO_MICROVM_TRANSPORT=local`): a
checkout service whose latest deploy exhausts its worker pool. It answers as the Action Gateway's code-host and
DigitalOcean tools do (`actions()` gives the local gateway one handler per tool, named as the gateway's catalog names
them) and as the real service's `/metrics` does (`metrics()`), and it writes what changes to the event log, which the
dashboard shows. Its links point at `code-host.mock`, which does not exist. In real mode the same tools run on a real
droplet (service/app.py, README.md).

Fixed timeline, relative to when the world is created: healthy for hours, the bad commit deployed
`DEPLOYED_MIN_AGO` minutes ago, a restore from the snapshot makes it healthy again within a minute.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from incident_common.events import log_event
from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    LocalAction,
)

OWNER, REPO = "acme", "checkout-service"
DROPLET_ID = 424242
SNAPSHOT_ID = 777001
DROPLET_NAME, SNAPSHOT_NAME = "checkout-svc", "checkout-svc-snapshot"  # the names setup_droplet.py gives the real ones
DEPLOYED_MIN_AGO = 24
GOOD_SHA, BAD_SHA = "9f3c2d1", "a1b2c3d"

_BAD_PATCH = (
    "@@ -21,7 +21,7 @@\n"
    '-CONFIRMATION_MODE = "async"  # "async": the provider calls our webhook. "sync": wait for its answer inside the request\n'
    '+CONFIRMATION_MODE = "sync"  # wait for the provider answer inside the request (about 3.4 s)\n'
    " POOL_SIZE = 40  # workers\n"
)
COMMITS = [  # newest first: (minutes before the world was created, sha, message, files, patch of the first file)
    (DEPLOYED_MIN_AGO, BAD_SHA, "checkout: wait for the payment provider's answer inside the request", ["service/app.py"], _BAD_PATCH),
    (190, "5e7a9b0", "docs: update the runbook", ["docs/runbook.md"], "@@ -1 +1,2 @@\n+Escalate to the on-call lead.\n"),
    (310, GOOD_SHA, "checkout: confirm payments through the provider's webhook", ["service/app.py"], "@@ -1 +1,2 @@\n+async confirmation\n"),
]


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class SimulatedWorld:
    """One incident. `now` is injectable so tests can move time."""

    def __init__(self, now: Callable[[], float] = time.time) -> None:
        self._now = now
        self.created = now()
        self.restored_at: float | None = None
        self.deleted = False
        self._lock = threading.Lock()

    # -- the service ---------------------------------------------------------------------------

    def _t(self, minutes_ago: float) -> datetime:
        return datetime.fromtimestamp(self.created - minutes_ago * 60, UTC)

    def version_at(self, moment: float) -> str:
        if self.restored_at is not None and moment >= self.restored_at:
            return GOOD_SHA
        return BAD_SHA if moment >= self.created - DEPLOYED_MIN_AGO * 60 else GOOD_SHA

    def health(self) -> dict[str, Any]:
        now = self._now()
        version = self.version_at(now)
        failing = version == BAD_SHA
        return {"status": "failing" if failing else "healthy", "version": version,
                "error_rate": 0.41 if failing else 0.01, "droplet": "deleted" if self.deleted else "active"}

    def metrics(self, minutes: int = 45) -> list[dict[str, Any]]:
        """One point a minute for the last `minutes`: version, error rate, p95 latency."""
        now = self._now()
        points = []
        for ago in range(minutes, -1, -1):
            moment = now - ago * 60
            version = self.version_at(moment)
            since = (moment - (self.created - DEPLOYED_MIN_AGO * 60)) / 60
            if version == BAD_SHA:  # the pool fills: errors climb for six minutes, then stay high
                error_rate = round(0.01 + min(since, 6) / 6 * 0.40, 3)
                p95 = round(180 + min(since, 6) / 6 * 4200)
            else:
                error_rate, p95 = 0.01 + (ago % 3) * 0.002, 180 + (ago % 4) * 6
            points.append({"t": _iso(datetime.fromtimestamp(moment, UTC)), "version": version,
                           "error_rate": error_rate, "p95_ms": p95})
        return points

    droplet_id, repo = DROPLET_ID, f"{OWNER}/{REPO}"

    def announce(self, what: str) -> None:
        """Write the droplet's current state to the event log (the dashboard's droplet strip)."""
        log_event(event="world", what=what, **self.health())

    # -- the gateway's tools -------------------------------------------------------------------

    def actions(self) -> dict[str, LocalAction]:
        """The local gateway's actions, named and shaped like the catalog's."""
        obj = {"type": "object"}

        def schema(**props: str) -> dict[str, Any]:
            return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": list(props)}

        return {
            "github_list_commits": LocalAction("List commits in a repository.", self._list_commits,
                                              schema(owner="string", repo="string")),
            "github_compare_commits": LocalAction("Compare two commits and see the commits and file changes between them.",
                                                 self._compare, schema(owner="string", repo="string", base="string", head="string")),
            "digitalocean_droplet-get": LocalAction("Get a droplet by its ID.", self._droplet, schema(ID="number")),
            "digitalocean_image-list": LocalAction("List images (snapshots, backups, distributions).", self._images, obj),
            "digitalocean_restore-droplet": LocalAction("Restore a droplet from a backup or snapshot.", self._restore,
                                                       schema(ID="number", ImageID="number")),
            "digitalocean_droplet-delete": LocalAction("Delete a droplet.", self._delete, schema(ID="number")),
        }

    def _list_commits(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"commits": [{"sha": sha, "commit": {"message": message, "author": {"name": "dev", "date": _iso(self._t(ago))}},
                             "html_url": f"https://code-host.mock/{OWNER}/{REPO}/commit/{sha}"}
                            for ago, sha, message, _, _ in COMMITS]}

    def _compare(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        head = str(arguments.get("head", ""))
        sha = next((c for c in COMMITS if head.startswith(c[1])), COMMITS[0])
        return {"status": "ahead", "total_commits": 1,
                "commits": [{"sha": sha[1], "commit": {"message": sha[2]}}],
                "files": [{"filename": f, "additions": 1, "deletions": 1, "patch": sha[4]} for f in sha[3]]}

    def _droplet(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"droplet": {"id": DROPLET_ID, "name": DROPLET_NAME, "status": "off" if self.deleted else "active",
                            "size_slug": "s-1vcpu-1gb", "region": {"slug": "sfo2"},
                            "networks": {"v4": [{"ip_address": "203.0.113.10", "type": "public"}]}, "snapshot_ids": [SNAPSHOT_ID]}}

    def _images(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"images": [{"id": SNAPSHOT_ID, "name": SNAPSHOT_NAME, "type": "snapshot",
                            "created_at": _iso(self._t(DEPLOYED_MIN_AGO + 30)), "min_disk_size": 25}]}

    def _restore(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not all(isinstance(arguments.get(k), int) for k in ("ID", "ImageID")):
            return {"error": "Request is invalid. Fix request arguments and retry."}  # ids are numbers, as in the catalog's schema
        if arguments["ID"] != DROPLET_ID or arguments["ImageID"] != SNAPSHOT_ID:
            return {"error": "unknown droplet or image"}
        with self._lock:
            self.restored_at = self._now() + 20  # the droplet is back after about 20 seconds
        self.announce("restoring")
        return {"action": {"id": 900001, "type": "restore", "status": "in-progress"}}

    def _delete(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.deleted = True  # unreachable: the policy denies it
        return {"deleted": True}


class HttpService:
    """The real service on a real droplet (service/app.py): its `/health` and `/metrics`, over HTTP.

    DO_INCIDENT_URL (for example http://203.0.113.10:8080) and DO_INCIDENT_DROPLET_ID are written by setup_droplet.py
    to outputs/incident.env; DO_INCIDENT_REPO (`owner/name`) is yours. The agents learn about the droplet only from the
    alert text and the gateway's tools, as an on-call engineer would.
    """

    def __init__(self, url: str, droplet_id: int, repo: str) -> None:
        self.url, self.droplet_id, self.repo = url.rstrip("/"), droplet_id, repo

    def health(self) -> dict[str, Any]:
        try:
            body = httpx.get(f"{self.url}/health", timeout=10).json()
            return {**body, "droplet": "active"}
        except (httpx.HTTPError, ValueError):
            return {"status": "unreachable", "version": "?", "error_rate": None, "droplet": "unknown"}

    def metrics(self, minutes: int = 45) -> list[dict[str, Any]]:
        return httpx.get(f"{self.url}/metrics", timeout=20).json()[-(minutes + 1):]

    def announce(self, what: str) -> None:
        log_event(event="world", what=what, **self.health())


def alert_text(service: SimulatedWorld | HttpService) -> str:
    """What the on-call alert says: the symptom as measured now, and where the service lives; nothing about the cause."""
    rate = service.health().get("error_rate")
    symptom = f"error rate {rate:.0%}" if isinstance(rate, float) else "the service is not answering"
    return (f"ALERT checkout-service: {symptom}. A stateless service (orders are in an external database) on droplet "
            f"{service.droplet_id} ({DROPLET_NAME}), code in {service.repo}. Find the cause and restore service.")


def make_service(local: bool) -> SimulatedWorld | HttpService:
    """The simulated world in local mode, else the real service named by DO_INCIDENT_* (setup_droplet.py)."""
    if local:
        return SimulatedWorld()
    try:
        return HttpService(os.environ["DO_INCIDENT_URL"], int(os.environ["DO_INCIDENT_DROPLET_ID"]), os.environ["DO_INCIDENT_REPO"])
    except KeyError as exc:
        raise RuntimeError(f"{exc.args[0]} is not set: run setup_droplet.py, or DO_MICROVM_TRANSPORT=local") from None
