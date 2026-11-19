"""Real mode only: create the incident's droplet, snapshot it healthy, start the incident, and clean up.

    uv run examples/incident_common/setup_droplet.py                  # droplet + service + snapshot (healthy)
    uv run examples/incident_common/setup_droplet.py --deploy-bad     # the bad deploy: the service starts failing
    uv run examples/incident_common/setup_droplet.py --verify-restore # deploy-bad, restore from the snapshot, wait for health
    uv run examples/incident_common/setup_droplet.py --cleanup        # delete the droplet, the snapshot and the ssh key

Everything goes through an Action Gateway endpoint of its own (DO_ACTION_GATEWAY_TOKEN, DO_ACTION_GATEWAY_ACTOR_ID, the
actor's DigitalOcean connection), so the droplet is created in the account that connection belongs to: the droplet
`checkout-svc` (s-1vcpu-1gb in sfo2 unless DO_INCIDENT_SIZE, DO_INCIDENT_REGION), the snapshot `checkout-svc-snapshot`, the ssh key `incident-key` (a temporary key pair kept
under outputs/). The gateway's `droplet-create` has no first-boot script, so the service is installed over ssh: the droplet
clones DO_INCIDENT_REPO (`owner/name`, a public repository you own, built with service/make_demo_repo.sh: two commits, the
newest is the bad deploy) and runs service/app.py from it. A deploy is a `git checkout` of a commit and a restart, and the
service reports the commit it runs, so the investigators find the cause in the diff. The snapshot is taken with the
droplet shut down, so the image a restore uses is a consistent disk. What was created is recorded in outputs/incident.json,
and DO_INCIDENT_URL / _DROPLET_ID / _REPO go to outputs/incident.env, which the examples load.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

import httpx
from dotenv import dotenv_values
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STATE, ENV_FILE, KEY = OUT / "incident.json", OUT / "incident.env", OUT / "incident_key"
for key, value in dotenv_values(HERE.parents[1] / ".env").items():
    if value:
        os.environ.setdefault(key, value)

from langchain_digitalocean_microvm.integrations.action_gateway import (
    ActionGateway,
    ActionPolicy,
)

DROPLET, SNAPSHOT, SSH_KEY = "checkout-svc", "checkout-svc-snapshot", "incident-key"
IMAGE = "ubuntu-24-04-x64"
SIZE = os.environ.get("DO_INCIDENT_SIZE", "s-1vcpu-1gb")
REGION = os.environ.get("DO_INCIDENT_REGION", "sfo2")  # another region when the size is not available in this one
TOOLS = [
    "digitalocean_key-create", "digitalocean_key-delete", "digitalocean_droplet-create", "digitalocean_droplet-get",
    "digitalocean_droplet-delete", "digitalocean_snapshot-droplet", "digitalocean_image-list", "digitalocean_image-delete",
    "github_list_commits", "digitalocean_shutdown-droplet", "digitalocean_power-off-droplet", "digitalocean_power-on-droplet",
    "digitalocean_restore-droplet",
]
SSH = ["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


class _Call(TypedDict, total=False):
    tool: str
    arguments: dict[str, Any]
    answer: str


def _ask_until_connected(gateway: ActionGateway, tool: str, arguments: dict[str, Any]) -> str:
    """One gateway call in a one-node graph: an action the actor has not connected yet interrupts it with a connect link,
    which is shown here; once you have connected (press Enter), the call runs again."""
    graph = StateGraph(_Call)
    graph.add_node("call", lambda state: {"answer": gateway.call(state["tool"], state["arguments"], f"setup_droplet.py: {state['tool']}")})
    graph.add_edge(START, "call")
    graph.add_edge("call", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
    out = app.invoke({"tool": tool, "arguments": arguments}, config)
    while "__interrupt__" in out:
        stop = out["__interrupt__"][0].value
        if stop.get("type") != "connect_required" or not sys.stdin.isatty():
            raise SystemExit(f"{tool}: the actor has no connection for this action yet. Open {stop.get('url')} "
                             "(signed in as the actor), connect, and run this again.")
        input(f"{tool} needs a connection: open {stop['url']}, connect, then press Enter: ")
        out = app.invoke(Command(resume="connected"), config)
    return out["answer"]


def call(gateway: ActionGateway, tool: str, **arguments: Any) -> Any:
    """One gateway call; the answer as JSON when it is (the catalog's tools return text or JSON)."""
    answer = _ask_until_connected(gateway, tool, arguments)
    if answer.startswith("Error:"):
        raise SystemExit(f"{tool} failed: {answer}")
    try:
        decoded = json.loads(answer)
    except ValueError:
        return answer
    return json.loads(decoded["text"]) if isinstance(decoded, dict) and set(decoded) == {"text"} and decoded["text"].lstrip()[:1] in "{[" else decoded


def find(value: Any, key: str) -> Any:
    """The first `key` anywhere in a nested answer (the tools' result shapes are the providers' own)."""
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            if (found := find(item, key)) is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            if (found := find(item, key)) is not None:
                return found
    return None


def ssh(ip: str, command: str, *, check: bool = True) -> str:
    done = subprocess.run([*SSH, f"root@{ip}", command], capture_output=True, text=True, timeout=120, check=False)
    if check and done.returncode:
        raise SystemExit(f"ssh {command[:60]!r} failed: {done.stderr.strip()[-300:]}")
    return done.stdout


def wait(label: str, probe, seconds: int = 300) -> Any:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if (result := probe()) is not None:
            return result
        time.sleep(5)
    raise SystemExit(f"timed out waiting for {label}")


def save(state: dict[str, Any]) -> None:
    OUT.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    ENV_FILE.write_text(
        f"DO_INCIDENT_URL=http://{state['ip']}:8080\nDO_INCIDENT_DROPLET_ID={state['droplet_id']}\n"
        f"DO_INCIDENT_REPO={state['repo']}\n"
    )


def versions(gateway: ActionGateway, repo: str) -> tuple[str, str]:
    """(good, bad): the short ids of the repository's second-newest and newest commits."""
    owner, name = repo.split("/", 1)
    commits = call(gateway, "github_list_commits", owner=owner, repo=name)
    shas = [c["sha"][:7] for c in (commits if isinstance(commits, list) else find(commits, "commits") or [])]
    if len(shas) < 2:
        raise SystemExit(f"{repo} needs at least two commits (the newest plays the bad deploy)")
    return shas[1], shas[0]


def create(gateway: ActionGateway) -> None:
    repo = os.environ.get("DO_INCIDENT_REPO")
    if not repo or "/" not in repo:
        raise SystemExit("set DO_INCIDENT_REPO=owner/name (a public repository you own, built with service/make_demo_repo.sh)")
    if STATE.exists():
        raise SystemExit(f"{STATE} exists: run --cleanup first")
    OUT.mkdir(exist_ok=True)
    good, bad = versions(gateway, repo)
    state: dict[str, Any] = {"repo": repo, "good": good, "bad": bad}
    for stale in (KEY, Path(f"{KEY}.pub")):  # left by a run that failed before it recorded anything
        stale.unlink(missing_ok=True)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(KEY)], check=True)
    try:
        key = call(gateway, "digitalocean_key-create", Name=SSH_KEY, PublicKey=Path(f"{KEY}.pub").read_text().strip())
        state["key_id"] = find(key, "id")
        save({**state, "ip": "", "droplet_id": None})  # from here on --cleanup finds the key, whatever fails next
        # SSHKeys items are strings in the tool's schema: the key's fingerprint (or its id as text)
        droplet = call(gateway, "digitalocean_droplet-create", Name=DROPLET, Size=SIZE, Region=REGION, ImageSlug=IMAGE,
                       SSHKeys=[str(find(key, "fingerprint") or state["key_id"])], Tags=["incident-example"])
        state["droplet_id"] = find(droplet, "id")
        save({**state, "ip": ""})

        def address() -> str | None:
            got = call(gateway, "digitalocean_droplet-get", ID=state["droplet_id"])
            if find(got, "status") != "active":
                return None
            return next((n["ip_address"] for n in (find(got, "v4") or []) if n.get("type") == "public"), None)

        state["ip"] = wait("the droplet's address", address)
        save(state)
        print(f"droplet {state['droplet_id']} at {state['ip']}; installing the service over ssh")
        wait("ssh", lambda: True if subprocess.run([*SSH, f"root@{state['ip']}", "true"], capture_output=True, timeout=30, check=False).returncode == 0 else None)
        unit = ("[Unit]\nDescription=checkout service\nAfter=network-online.target\n[Service]\nWorkingDirectory=/opt/checkout\n"
                "ExecStart=/usr/bin/python3 /opt/checkout/service/app.py 8080\nRestart=always\n[Install]\nWantedBy=multi-user.target\n")
        ssh(state["ip"], "command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git); "
                         f"git clone -q https://github.com/{repo}.git /opt/checkout && git -C /opt/checkout checkout -q {good}; "
                         f"printf '%s' '{unit}' > /etc/systemd/system/checkout.service; systemctl daemon-reload; systemctl enable --now checkout")
        wait("the service", lambda: True if _healthy(state["ip"]) else None, 120)
        # Snapshot the droplet shut down: the image a restore uses is then a consistent disk.
        call(gateway, "digitalocean_shutdown-droplet", ID=state["droplet_id"])
        if wait("the droplet to power off", lambda: True if _droplet_status(gateway, state["droplet_id"]) == "off" else None, 120) is None:
            call(gateway, "digitalocean_power-off-droplet", ID=state["droplet_id"])
        snap = call(gateway, "digitalocean_snapshot-droplet", ID=state["droplet_id"], Name=SNAPSHOT)
        print("snapshot requested (droplet off):", json.dumps(snap)[:120])
        state["snapshot_id"] = wait("the snapshot", lambda: _snapshot_id(gateway), 900)
        save(state)
        call(gateway, "digitalocean_power-on-droplet", ID=state["droplet_id"])
        wait("the service after power-on", lambda: True if _healthy(state["ip"]) else None, 300)
        print(f"healthy, snapshot {state['snapshot_id']}. Next: --deploy-bad, then the example (it loads outputs/incident.env).")
    except BaseException:
        print("setup failed; run --cleanup to delete what was created", file=sys.stderr)
        raise


def _status_code(ip: str) -> int | None:
    try:
        return httpx.get(f"http://{ip}:8080/health", timeout=5).status_code
    except httpx.HTTPError:
        return None


def _healthy(ip: str) -> bool:
    return _status_code(ip) == 200


def _droplet_status(gateway: ActionGateway, droplet_id: int) -> str | None:
    return find(call(gateway, "digitalocean_droplet-get", ID=droplet_id), "status")


def _snapshot_id(gateway: ActionGateway) -> int | None:
    images = call(gateway, "digitalocean_image-list", Type="user")
    for image in find(images, "images") or []:
        if image.get("name") == SNAPSHOT:
            return image["id"]
    return None


def _load_state() -> dict:
    if not STATE.exists():
        raise SystemExit(f"{STATE} not found: run setup_droplet.py without a flag first")
    return json.loads(STATE.read_text())


def deploy_bad() -> None:
    state = _load_state()
    ssh(state["ip"], f"git -C /opt/checkout checkout -q {state['bad']} && systemctl restart checkout")
    print(f"deployed {state['bad']}: the service fails from now on; start the example")


def verify_restore(gateway: ActionGateway) -> None:
    """The remedy the example's commander asks for, run by hand: bad deploy, restore, wait until healthy."""
    state = _load_state()
    deploy_bad()
    wait("the failure", lambda: True if _status_code(state["ip"]) == 503 else None, 120)
    print("failing; restoring from the snapshot")
    started = time.monotonic()
    answer = call(gateway, "digitalocean_restore-droplet", ID=state["droplet_id"], ImageID=state["snapshot_id"])
    print("restore:", json.dumps(answer)[:160])
    wait("the service after the restore", lambda: True if _healthy(state["ip"]) else None, 600)
    print(f"healthy {time.monotonic() - started:.0f} s after the restore request")


def cleanup(gateway: ActionGateway) -> None:
    if not STATE.exists():
        raise SystemExit("nothing recorded in outputs/incident.json")
    state = json.loads(STATE.read_text())
    if state.get("droplet_id") is None and state.get("key_id") is None:
        print("nothing to delete beyond the local files")
    for tool, arg, value in (("digitalocean_droplet-delete", "ID", state.get("droplet_id")),
                             ("digitalocean_image-delete", "ID", state.get("snapshot_id")),
                             ("digitalocean_key-delete", "ID", state.get("key_id"))):
        if value is not None:
            print(f"{tool} {value}: {str(call(gateway, tool, **{arg: value}))[:120]}")
    for path in (STATE, ENV_FILE, KEY, Path(f"{KEY}.pub")):
        path.unlink(missing_ok=True)
    print("cleaned up")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--deploy-bad", action="store_true", help="start the incident")
    group.add_argument("--verify-restore", action="store_true", help="deploy the bad commit, restore from the snapshot, wait for health")
    group.add_argument("--cleanup", action="store_true", help="delete the droplet, the snapshot and the ssh key")
    args = parser.parse_args()
    if args.deploy_bad:
        return deploy_bad()
    with ActionGateway(ActionPolicy(allow=TOOLS), tools=TOOLS) as gateway:
        if args.cleanup:
            cleanup(gateway)
        elif args.verify_restore:
            verify_restore(gateway)
        else:
            create(gateway)


if __name__ == "__main__":
    main()
