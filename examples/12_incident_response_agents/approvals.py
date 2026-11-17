"""The approval prompt, used by the console (console.py, through relay.py) and the relay.

An `action_approval` interrupt carries the tool, its arguments and the Remediator's evidence. Beside them the console prints
what the arguments point at, read with its own gateway endpoint, so an approval does not rest on the agents' words alone.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

DIM, BOLD, RESET, AMBER = "\033[2m", "\033[1m", "\033[0m", "\033[33m"


RESTORE = "digitalocean_restore-droplet"


def read_target(stop: dict[str, Any], gateway: Any) -> dict[str, Any] | None:
    """What a restore's arguments point at, read through `gateway` (not taken from the commander's evidence): {"droplet", "image"}
    (the image None when the account has no such snapshot), {"error"} when it could not be read, None for any other action."""
    if stop.get("tool") != RESTORE:
        return None
    arguments = stop.get("arguments", {})
    try:
        droplet = json.loads(gateway.call("digitalocean_droplet-get", {"ID": arguments["ID"]}))
        droplet = droplet.get("droplet", droplet)
        images = json.loads(gateway.call("digitalocean_image-list", {"Type": "user"})).get("images", [])
        image = next((i for i in images if i.get("id") == arguments.get("ImageID")), None)
    except (KeyError, ValueError, TypeError, AttributeError):
        return {"error": "could not read the droplet and the snapshot"}
    return {"droplet": droplet, "image": image}


def facts_of(target: dict[str, Any] | None) -> list[str]:
    if target is None:
        return []
    if "error" in target:
        return ["could not read the droplet and the snapshot: check them before you approve"]
    droplet, image = target["droplet"], target["image"]
    return [f"droplet    {droplet.get('name')} ({droplet.get('status')})",
            f"snapshot   {image.get('name')}, created {image.get('created_at')}" if image
            else "snapshot   NOT FOUND among the account's snapshots"]


def verified_facts(stop: dict[str, Any], gateway: Any) -> list[str]:
    """What the arguments point at, read by this console through the gateway, not taken from the commander's evidence."""
    return facts_of(read_target(stop, gateway))


def _epoch(stamp: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def problems_of(stop: dict[str, Any], target: dict[str, Any] | None, *, droplet_id: Any = None, alerted_at: float | None = None) -> list[str]:
    """Why a restore must not be approved, from facts the relay holds: it names another droplet than the incident's, the snapshot
    does not exist, or it was taken after the alert (so it cannot predate the failure). An empty list is no proof the restore is
    right, only that none of these failed; an unreadable target adds nothing here (the card already says to check it)."""
    if stop.get("tool") != RESTORE:
        return []
    problems = []
    asked = (stop.get("arguments") or {}).get("ID")
    if droplet_id and str(asked) != str(droplet_id):
        problems.append(f"it restores droplet {asked}, not this incident's droplet {droplet_id}")
    if target is not None and "error" not in target:
        image = target["image"]
        if image is None:
            problems.append("the snapshot is not among the account's snapshots")
        elif alerted_at is not None and (created := _epoch(image.get("created_at"))) is not None and created >= alerted_at:
            problems.append("the snapshot was created after the alert, so it cannot predate the failure")
    return problems


def ask_human(stop: dict[str, Any], mode: str | None, facts: list[str], *, read_by: str = "this console",
              problems: list[str] | None = None) -> str:
    """Show an `action_approval` (or connect) interrupt with the `facts` read independently (by `read_by`); return the resume value."""
    if stop["type"] == "connect_required":
        print(f"\n{AMBER}CONNECT REQUIRED{RESET} {stop['tool']}: open {stop['url']}")
        if mode is None:
            input("Connect the account, then press Enter: ")
        return "connected"
    print(f"\n{AMBER}{BOLD}APPROVAL REQUIRED{RESET}  {BOLD}{stop['tool']}{RESET}")
    print(f"  arguments  {json.dumps(stop['arguments'])}")
    print(f"  evidence   {stop.get('rationale', '')}   {DIM}(the remediator's rationale){RESET}")
    for line in facts:
        print(f"  {line}   {DIM}(read by {read_by}){RESET}")
    print(f"  expires    {stop.get('expires_at')}")
    if problems:  # the relay refuses an approval with these: do not ask for one
        for problem in problems:
            print(f"  {AMBER}{BOLD}NOT APPROVABLE{RESET}  {problem}")
        print(f"  {DIM}answering deny{RESET}")
        return "deny"
    if mode:
        print(f"  {DIM}--{mode}: answering {mode}{RESET}")
        return mode
    while True:
        answer = input("  [a]pprove / [d]eny: ").strip().lower()
        if answer in ("a", "approve"):
            return "approve"
        if answer in ("d", "deny"):
            return "deny"
