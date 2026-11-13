"""Delete the MicroVMs and checkpoints you created with this package, in one namespace of the team.

Clean-up tool: paused MicroVMs also count against the team limit, so remove
what is no longer needed. This lists, and with --yes deletes:

- MicroVMs created by you (they carry your owner tag, see owner.py) and tagged `microvm-sandbox` (MicroVMClient adds that tag to every
  MicroVM it creates: sandboxes and agent hosts) or `microvm-probe`
  (scripts/probe_api.py), that belong to the namespace. MicroVMs of other users on the team,
  and MicroVMs created before owner tags existed, are left alone unless --all-owners;
- checkpoints taken from those MicroVMs, and with --all-owners also those whose source MicroVM name starts
  with one of this package's prefixes in the namespace (a checkpoint outlives its MicroVM and then
  records no owner), incl. warm-start golden checkpoints. Sandbox templates
  (`microvm-template-*`) are kept unless --include-templates;
- host and subagent/thread sandbox rows in the namespace's gateway store
  (`gateway[-<namespace>].sqlite` in the package's state directory, see docs/reference.md), marked destroyed so the next run
  does not try to reuse a deleted host.

The namespace is --namespace, else DO_MICROVM_NAMESPACE, else the default one:
MicroVMs without a `microvm-namespace:` tag. Other namespaces on the team are
never touched, but --all-namespaces selects every one (and the stores of all of
them), including other users' sandboxes and agent hosts, also mid-run (it implies --all-owners).
Templates are shared by all users and namespaces, so --include-templates needs --all-owners, and
--all-namespaces whenever a namespace is set or other namespaces have MicroVMs
on the team. MicroVMs created otherwise are never touched.

Dry run unless --yes. With --yes the script prints what it will delete and asks
you to type `delete`; without a terminal (scripts, cron) it refuses unless
--force is given as well.

    uv run python \\
        scripts/delete_all.py [--namespace NS [--all-owners] | --all-namespaces] [--yes [--force]] [--include-templates]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from _env import add_env_file_argument, load_env  # scripts/_env.py

from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    default_db_path,
    store_dirs,
)
from langchain_digitalocean_microvm.agent_host.sweeper import (
    delete_checkpoint_patiently,
)
from langchain_digitalocean_microvm.client import MANAGED_TAG, MicroVMClient
from langchain_digitalocean_microvm.exceptions import MicroVMNotFoundError
from langchain_digitalocean_microvm.models import Checkpoint, MicroVMState
from langchain_digitalocean_microvm.namespace import (
    in_scope,
    namespace_from_name,
    namespace_from_tags,
    namespaces_present,
    validate_namespace,
)
from langchain_digitalocean_microvm.owner import in_owner_scope
from langchain_digitalocean_microvm.status import MicroVMStatus

TAGS = (MANAGED_TAG, "microvm-probe")
NAME_PREFIXES_BARE = ("microvm-sandbox", "microvm-agenthost", "microvm-golden", "microvm-template", "probe")
NAME_PREFIXES = tuple(f"{p}-" for p in NAME_PREFIXES_BARE)
TEMPLATE_PREFIX = "microvm-template-"
GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)
CONFIRM_WORD = "delete"


def find(
    client: MicroVMClient, *, include_templates: bool, namespace: str | None = None, all_namespaces: bool = False,
    all_owners: bool = False,
) -> tuple[list[MicroVMState], list[Checkpoint], list[Checkpoint]]:
    """(MicroVMs to delete, checkpoints to delete, template checkpoints kept) in the namespace.

    Only this user's MicroVMs (owner tag) unless `all_owners` or `all_namespaces`; a checkpoint is selected by
    its source MicroVM, or by name only with every owner (its name says nothing about who made it).
    """
    everyone = all_owners or all_namespaces
    microvms: dict[str, MicroVMState] = {}
    for tag in TAGS:
        for state in client.list(tag=tag):
            if state.status not in GONE and in_owner_scope(state.tags, all_owners=everyone) and in_scope(
                namespace_from_tags(state.tags), namespace, all_namespaces=all_namespaces
            ):
                microvms[state.id] = state
    ours = set(microvms)
    checkpoints = [
        cp for cp in client.list_checkpoints()
        if cp.status not in ("CHECKPOINT_DELETING", "CHECKPOINT_DELETED")
        and (cp.microvm_id in ours or (everyone and _name_in_scope(cp, namespace, all_namespaces)))
    ]
    kept = [] if include_templates else [cp for cp in checkpoints if cp.name.startswith(TEMPLATE_PREFIX)]
    return list(microvms.values()), [cp for cp in checkpoints if cp not in kept], kept


def _name_in_scope(cp: Checkpoint, namespace: str | None, all_namespaces: bool) -> bool:
    """A checkpoint whose source MicroVM (or own name) carries one of our prefixes, in the namespace."""
    source = str(cp.raw.get("microvm_name", ""))
    if not source.startswith(NAME_PREFIXES):
        return False
    return in_scope(namespace_from_name(source, NAME_PREFIXES_BARE), namespace, all_namespaces=all_namespaces)


def templates_need_all_namespaces(client: MicroVMClient, namespace: str | None) -> bool:
    """Templates are shared: deleting them is only allowed with --all-namespaces if a namespace is set or others exist."""
    if namespace is not None:
        return True
    others = {
        ns for tag in TAGS for ns in namespaces_present(s for s in client.list(tag=tag) if s.status not in GONE)
    } - {None}
    return bool(others)


def warning(
    microvms: list[MicroVMState], checkpoints: list[Checkpoint], kept: list[Checkpoint],
    *, namespace: str | None = None, all_namespaces: bool = False, all_owners: bool = False,
) -> str:
    running = sum(state.status == MicroVMStatus.RUNNING for state in microvms)
    templates = sum(cp.name.startswith(TEMPLATE_PREFIX) for cp in checkpoints)
    if all_namespaces:
        scope = ["This deletes every MicroVM and checkpoint created by langchain-digitalocean-microvm on this team,",
                 "in every namespace, including other people's: their sandboxes and agent hosts too, also mid-run."]
    else:
        who = "of every owner" if all_owners else "you created"
        scope = [f"This deletes the MicroVMs and checkpoints {who} in {describe_namespace(namespace)} on this team.",
                 "Other namespaces are left alone (--all-namespaces selects every one)."]
        if not all_owners:
            scope.append("Other users' MicroVMs, and ones created before owner tags existed, are left alone (--all-owners).")
    lines = [
        *scope,
        f"  {len(microvms)} MicroVM(s) ({running} running), {len(checkpoints)} checkpoint(s)"
        + (f" ({templates} template(s))" if templates else ""),
    ]
    if kept:
        lines.append(f"  {len(kept)} template checkpoint(s) kept (--include-templates deletes them too)")
    return "\n".join(lines)


def _stores(namespaces: set[str | None]) -> list:
    """The default store file of each namespace (the stores of namespaces with no MicroVM left included if present)."""
    paths = {default_db_path("microvm", ns) for ns in namespaces}
    for directory in store_dirs():
        paths.update(p for p in directory.glob("gateway-*.sqlite") if not p.name.startswith("gateway-local"))
        paths.update(p for p in directory.glob("gateway.sqlite"))
    return sorted(paths)


def describe_namespace(namespace: str | None) -> str:
    return f"namespace {namespace!r}" if namespace else "the default namespace (no namespace tag)"


def confirmed(*, force: bool, interactive: bool, ask: Callable[[str], str] = input) -> bool:
    """--force, or the word typed at a terminal. Without a terminal and --force: refuse."""
    if force:
        return True
    if not interactive:
        print("\nRefusing to delete without a terminal to confirm on; re-run with --yes --force.")
        return False
    try:
        answer = ask(f"\nType '{CONFIRM_WORD}' to delete all of the above: ")
    except EOFError:
        answer = ""
    if answer.strip() != CONFIRM_WORD:
        print("Not confirmed; nothing deleted.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="delete (default: list only); asks for confirmation")
    parser.add_argument("--force", action="store_true", help="with --yes: do not ask (scripts, cron)")
    parser.add_argument("--include-templates", action="store_true",
                        help="also delete sandbox templates (microvm-template-* checkpoints)")
    parser.add_argument("--namespace", default=None,
                        help="the namespace to clean (default: DO_MICROVM_NAMESPACE, else the default namespace)")
    parser.add_argument("--all-owners", action="store_true",
                        help="also MicroVMs other users of the team created, and ones with no owner tag (in the namespace)")
    parser.add_argument("--all-namespaces", action="store_true",
                        help="every namespace on the team, other users' included (and all their stores)")
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
    client = MicroVMClient(token)
    if args.include_templates and not (args.all_owners or args.all_namespaces):
        print("Templates are shared by every user of the team: --include-templates needs --all-owners "
              "(or --all-namespaces).", file=sys.stderr)
        return 2
    if args.include_templates and not args.all_namespaces and templates_need_all_namespaces(client, namespace):
        print("Templates are shared by every namespace: --include-templates needs --all-namespaces here "
              "(a namespace is set, or other namespaces have MicroVMs on the team).", file=sys.stderr)
        return 2

    all_owners = args.all_owners or args.all_namespaces
    microvms, checkpoints, kept = find(
        client, include_templates=args.include_templates, namespace=namespace, all_namespaces=args.all_namespaces,
        all_owners=all_owners,
    )
    print(f"Scope: {'every namespace' if args.all_namespaces else describe_namespace(namespace)}, "
          f"{'every owner' if all_owners else 'created by you'}")
    print(f"MicroVMs ({len(microvms)}):")
    for state in microvms:
        print(f"  {state.id}  {state.name:<60} {state.status.value:<10} tags={list(state.tags)}")
    print(f"Checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        print(f"  {cp.id}  {cp.name:<40} {cp.status:<22} from {cp.raw.get('microvm_name', cp.microvm_id)}")
    if kept:
        print(f"Kept template checkpoints ({len(kept)}):")
        for cp in kept:
            print(f"  {cp.id}  {cp.name}")
    dbs = _stores(namespaces_present(microvms) | {namespace}) if args.all_namespaces else [default_db_path("microvm", namespace)]
    for db in dbs:
        print(f"Gateway store: {db} ({'present' if db.exists() else 'absent'})")
    print(f"\n{warning(microvms, checkpoints, kept, namespace=namespace, all_namespaces=args.all_namespaces, all_owners=all_owners)}")
    if not args.yes:
        print("\nDry run. Re-run with --yes to delete all of the above.")
        return 0
    # Nothing on the team to delete: only local store rows are marked, no need to ask.
    if (microvms or checkpoints) and not confirmed(force=args.force, interactive=sys.stdin.isatty()):
        return 2

    for state in microvms:
        try:
            client.delete(state.id)
            print(f"deleted MicroVM {state.name} ({state.id})")
        except MicroVMNotFoundError:
            pass
    for cp in checkpoints:
        delete_checkpoint_patiently(client, cp.id)  # waits out the 400 while a checkpoint is still being captured
        print(f"deleted checkpoint {cp.name} ({cp.id})")
    for db in dbs:
        if db.exists():
            store = GatewayStore(db)
            try:
                for row in store.list_hosts():
                    if row.status != "destroyed":
                        store.retire_host(row.host_id, "deleted")
                for srow in store.list_sandboxes():
                    if srow.status != "destroyed":
                        store.retire_sandbox(srow.sandbox_id, "deleted")
            finally:
                store.close()
            print(f"marked every host and sandbox in {db} destroyed")
    left = [
        s for tag in TAGS for s in client.list(tag=tag)
        if s.status not in GONE and in_owner_scope(s.tags, all_owners=all_owners)
        and in_scope(namespace_from_tags(s.tags), namespace, all_namespaces=args.all_namespaces)
    ]
    print(f"\nremaining live MicroVMs with our tags in scope: {len(left)}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
