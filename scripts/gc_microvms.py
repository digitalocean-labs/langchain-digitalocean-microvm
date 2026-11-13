"""Find and delete stale agent hosts and warm-start checkpoints (dry run unless --yes).

The gateway deletes expired hosts itself while a supervisor runs (its
`tick()`: per-task hosts after `release_after`, any host after `paused_ttl`).
This script is the backstop for everything that outlives the supervisor; it
can run from cron. It lists, and with --yes deletes (agent_host/sweeper.py):

- agent hosts the gateway store marks destroyed/failed, or unused for --ttl
  with no task in progress (the store's last activity);
- agent hosts no store row describes, older than --orphan-ttl;
- leftover golden-checkpoint builds, and golden checkpoints that failed or are
  older than --checkpoint-ttl (rebuilt on their next use);
- subagent and thread sandboxes (subagents, threads; tagged
  `microvm-subagent-sandbox` / `microvm-thread-sandbox`) last used more than
  --sandbox-ttl ago per the store, or older than --orphan-ttl when no store
  row describes them;
- store rows whose MicroVM no longer exists (marked destroyed).

Deleted hosts and sandboxes are marked destroyed in the store, so a gateway
sharing it answers `check` from its cache and artifact fetches with a 410.
An agent host or sandbox that no store row describes is collected only if you created it (owner tag, see
owner.py); another user's MicroVMs of the namespace are kept unless --all-owners.
Other sandboxes (plain MicroVMLifecycle users) are not touched;
scripts/delete_all.py removes everything this package created.

Only one namespace is collected: --namespace, else DO_MICROVM_NAMESPACE, else the
default one (MicroVMs without a `microvm-namespace:` tag), with that namespace's store.
Other namespaces are left alone; --all-namespaces collects every one, but with one
store, so the hosts and sandboxes of other namespaces (their rows are in other stores)
count as orphans and go once older than --orphan-ttl.

    uv run python \\
        scripts/gc_microvms.py [--ttl 24h] [--yes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _env import add_env_file_argument, load_env  # scripts/_env.py

from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    default_db_path,
)
from langchain_digitalocean_microvm.agent_host.sweeper import (
    DEFAULT_CHECKPOINT_TTL,
    DEFAULT_GOLDEN_BUILD_TIMEOUT,
    DEFAULT_HOST_TTL,
    SweepItem,
    apply_sweep,
    plan_sweep,
)
from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.models import parse_duration
from langchain_digitalocean_microvm.namespace import validate_namespace


def _duration(text: str) -> float:
    try:
        return parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _show(title: str, items: tuple[SweepItem, ...]) -> None:
    print(f"{title} ({len(items)}):")
    for item in items:
        print(f"  {item.kind:<10} {item.id}  {item.name:<44} {item.reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="actually delete (default: list only)")
    parser.add_argument("--ttl", type=_duration, default=DEFAULT_HOST_TTL, help="unused agent hosts older than this (default 24h)")
    parser.add_argument("--orphan-ttl", type=_duration, default=None, help="hosts no store knows, by age (default: --ttl)")
    parser.add_argument("--checkpoint-ttl", type=_duration, default=DEFAULT_CHECKPOINT_TTL, help="golden checkpoints (default 24h)")
    parser.add_argument("--sandbox-ttl", type=_duration, default=None,
                        help="subagent and thread sandboxes unused this long (default: --ttl)")
    parser.add_argument("--golden-build-timeout", type=_duration, default=DEFAULT_GOLDEN_BUILD_TIMEOUT,
                        help="a golden build VM older than this was interrupted (default 30m)")
    parser.add_argument("--namespace", default=None,
                        help="the namespace to collect (default: DO_MICROVM_NAMESPACE, else the default namespace)")
    parser.add_argument("--all-namespaces", action="store_true",
                        help="collect every namespace's MicroVMs; every host counts as an orphan unless this store describes it")
    parser.add_argument("--all-owners", action="store_true",
                        help="also collect hosts and sandboxes no store describes that other users created (or with no owner tag)")
    parser.add_argument("--db", type=Path, default=None,
                        help="gateway store (default: the namespace's file in the package's state directory)")
    add_env_file_argument(parser)
    args = parser.parse_args()
    env = load_env(args.env_file)
    token = env.get("DO_MICROVM_API_TOKEN")
    if not token:
        print("DO_MICROVM_API_TOKEN is not set (.env); nothing to do.")
        return 0
    if args.all_namespaces and args.namespace:
        parser.error("--namespace and --all-namespaces are exclusive")
    try:
        namespace = validate_namespace(args.namespace if args.namespace is not None else env.get("DO_MICROVM_NAMESPACE"))
    except ValueError as exc:
        parser.error(str(exc))
    if args.db is None:
        args.db = default_db_path("microvm", namespace)
    client = MicroVMClient(token)
    store = GatewayStore(args.db) if args.db.exists() else None
    try:
        print(f"Scope: {'every namespace' if args.all_namespaces else ('namespace ' + repr(namespace) if namespace else 'the default namespace (no namespace tag)')}")
        print(f"Gateway store: {args.db} ({'present' if store else 'absent: every agent host counts as an orphan'})")
        plan = plan_sweep(client, store=store, host_ttl=args.ttl, orphan_ttl=args.orphan_ttl,
                          checkpoint_ttl=args.checkpoint_ttl, golden_build_timeout=args.golden_build_timeout,
                          sandbox_ttl=args.sandbox_ttl, namespace=namespace, all_namespaces=args.all_namespaces,
                          all_owners=args.all_owners or args.all_namespaces)
        _show("Delete MicroVMs", plan.delete_microvms)
        _show("Delete checkpoints", plan.delete_checkpoints)
        _show("Mark store rows destroyed", plan.retire_rows)
        _show("Keep", plan.kept)
        if plan.empty:
            print("\nNothing to collect.")
            return 0
        if not args.yes:
            print("\nDry run. Re-run with --yes to delete the above.")
            return 0
        result = apply_sweep(plan, client, store=store)
    finally:
        if store is not None:
            store.close()
    for item in result.deleted_microvms:
        print(f"deleted MicroVM {item.name} ({item.id})")
    for item in result.skipped:
        print(f"kept MicroVM {item.name} ({item.id}): touched since the plan")
    for item in result.deleted_checkpoints:
        print(f"deleted checkpoint {item.name} ({item.id})")
    if result.retired_rows:
        print(f"marked {len(result.retired_rows)} store row(s) destroyed")
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
