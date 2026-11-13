"""The approval prompt of the console (run_script.py).

An `action_approval` interrupt carries the tool, its arguments and the commander's evidence. Beside them the console prints
what the arguments point at, read with its own gateway session, so an approval does not rest on the commander's words alone.
"""

from __future__ import annotations

import json
from typing import Any

DIM, BOLD, RESET, AMBER = "\033[2m", "\033[1m", "\033[0m", "\033[33m"


def verified_facts(stop: dict[str, Any], gateway: Any) -> list[str]:
    """What the arguments point at, read by this console through the gateway, not taken from the commander's evidence."""
    if stop.get("tool") != "digitalocean_restore-droplet":
        return []
    arguments = stop.get("arguments", {})
    try:
        droplet = json.loads(gateway.call("digitalocean_droplet-get", {"ID": arguments["ID"]}))
        droplet = droplet.get("droplet", droplet)
        images = json.loads(gateway.call("digitalocean_image-list", {"Type": "user"})).get("images", [])
        image = next((i for i in images if i.get("id") == arguments.get("ImageID")), None)
    except (KeyError, ValueError, TypeError, AttributeError):
        return ["could not read the droplet and the snapshot: check them before you approve"]
    return [f"droplet    {droplet.get('name')} ({droplet.get('status')})",
            f"snapshot   {image.get('name')}, created {image.get('created_at')}" if image
            else "snapshot   NOT FOUND among the account's snapshots"]


def ask_human(stop: dict[str, Any], mode: str | None, facts: list[str], *, read_by: str = "this console") -> str:
    """Show an `action_approval` (or connect) interrupt with the `facts` read independently (by `read_by`); return the resume value."""
    if stop["type"] == "connect_required":
        print(f"\n{AMBER}CONNECT REQUIRED{RESET} {stop['tool']}: open {stop['url']}")
        if mode is None:
            input("Connect the account, then press Enter: ")
        return "connected"
    print(f"\n{AMBER}{BOLD}APPROVAL REQUIRED{RESET}  {BOLD}{stop['tool']}{RESET}")
    print(f"  arguments  {json.dumps(stop['arguments'])}")
    print(f"  evidence   {stop.get('rationale', '')}   {DIM}(the commander's words){RESET}")
    for line in facts:
        print(f"  {line}   {DIM}(read by {read_by}){RESET}")
    print(f"  expires    {stop.get('expires_at')}")
    if mode:
        print(f"  {DIM}--{mode}: answering {mode}{RESET}")
        return mode
    while True:
        answer = input("  [a]pprove / [d]eny: ").strip().lower()
        if answer in ("a", "approve"):
            return "approve"
        if answer in ("d", "deny"):
            return "deny"
