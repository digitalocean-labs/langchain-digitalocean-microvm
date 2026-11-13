"""Check your setup and say, per example, what is ready and what is missing.

Reads `.env` (variables set in the shell win), prints no secret, creates nothing. Checks:

- the control token: a read-only call (the MicroVMs `options` and a list of the team's MicroVMs):
  a missing or wrong token, the team's MicroVM limit, how many MicroVMs are on the team and how many
  of them you created, and the hourly price of a running MicroVM of your size as the API states it;
- the model: `DO_MODEL_ACCESS_KEY` is accepted by Serverless Inference (a model-list read), or
  `EXAMPLES_MODEL` plus the provider key it names;
- the Action Gateway token (a one-item tool-catalog read);
- the variables each example reads, and whether the team has room for it.

The access token, the image and the scopes that create, pause and delete cannot be checked without a
MicroVM: `--deep` creates one throwaway MicroVM (`microvm-sandbox-check-*`), runs `echo`, pauses and
deletes it.

    uv run python scripts/check_setup.py [--offline] [--deep] [--local] [--examples 01,05,10]

Exit status 0 when every example you asked for (default: all) can run in the mode checked, else 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from _env import DEFAULT_ENV_FILE, add_env_file_argument
from dotenv import dotenv_values

from langchain_digitalocean_microvm.client import MicroVMClient
from langchain_digitalocean_microvm.exceptions import (
    MicroVMAPIError,
    MicroVMClientError,
)
from langchain_digitalocean_microvm.models import MicroVMSize
from langchain_digitalocean_microvm.namespace import list_owned, validate_namespace
from langchain_digitalocean_microvm.owner import is_mine
from langchain_digitalocean_microvm.status import MicroVMStatus

INFERENCE_URL = "https://inference.do-ai.run/v1/models"
ACTION_GATEWAY_URL = "https://api.digitalocean.com/v2/action-gateway/tools"

# Where a provider's key lives, for EXAMPLES_MODEL ("anthropic:claude-haiku-4-5" -> ANTHROPIC_API_KEY).
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY"}
GONE = (MicroVMStatus.TERMINATING, MicroVMStatus.TERMINATED)


@dataclass(frozen=True)
class Example:
    """What an example needs. `microvms`: held at once, a template's temporary builder included."""

    number: str
    name: str
    microvms: int
    model: bool = True
    image: bool = True  # DO_MICROVM_IMAGE (a sandbox image)
    host_mode: bool = False  # real agent hosts: DO_MICROVM_HOST_MODE=microvm and DO_MICROVM_AGENT_IMAGE
    gateway: bool = False  # real Action Gateway (DO_ACTION_GATEWAY_*)
    other: tuple[str, ...] = ()  # more variables the real run reads (all required)
    setup: str = ""  # what else the real run needs that no variable shows
    maintainer: bool = False  # setup of your own (see examples/README.md)


DROPLET = "a droplet and a snapshot (examples/incident_common/setup_droplet.py)"

EXAMPLES = (
    Example("01", "quickstart", 1),
    Example("02", "deepagents_code", 1, model=False, setup="the `dcode` CLI (Python 3.12+)"),
    Example("03", "subagent_sandboxes", 2),
    Example("04", "async_subagents", 2, image=False, host_mode=True, maintainer=True,
            setup="an agent-host image built with `uv` and `crane` (images/agent-host)"),
    Example("05", "data_analyst", 1),
    Example("06", "warm_start", 1),
    Example("07", "sales_assistant", 2),
    Example("08", "board_pack", 7),
    Example("09", "incident_response", 5, gateway=True, other=("DO_INCIDENT_REPO",), maintainer=True, setup=DROPLET),
    Example("10", "erasure", 5),
    Example("11", "vulnerability_response", 6),
    Example("12", "incident_response_agents", 6, image=False, host_mode=True, gateway=True,
            other=("DO_INCIDENT_HOST_IMAGE", "DO_INCIDENT_SPECIALISTS_IMAGE", "DO_INCIDENT_BROKER_URL",
                   "DO_INCIDENT_REPO"),
            maintainer=True, setup=DROPLET),
)


@dataclass(frozen=True)
class Team:
    """The control token's view of the team."""

    limit: int | None  # account_limits.max_total_count
    live: int  # MicroVMs on the team that are not terminating or terminated
    mine: int  # of those, the ones you created in your namespace
    hourly: float | None  # price of a running MicroVM of your size, per hour


@dataclass
class Probes:
    """What touches the network, so the evaluation is testable."""

    team: Callable[[], Team]  # raises MicroVMClientError or httpx.HTTPError
    model_key: Callable[[], str | None]  # None when accepted, else why not
    gateway: Callable[[], str | None]
    deep: Callable[[], str | None] | None = None


@dataclass
class Line:
    ok: bool | None  # None: not checked
    what: str
    detail: str = ""


@dataclass
class Report:
    checks: list[Line] = field(default_factory=list)
    examples: list[tuple[Example, str, list[str]]] = field(default_factory=list)  # (example, status, missing)
    team: Team | None = None


def merged_env(env_file: Path) -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(env_file).items() if v}
    values.update({k: v for k, v in os.environ.items()
                   if v and (k.startswith(("DO_", "EXAMPLES_")) or k in PROVIDER_KEYS.values())})
    return values


def model_variables(env: Mapping[str, str]) -> tuple[bool, str]:
    """(set up?, what is missing)"""
    if env.get("DO_MODEL_ACCESS_KEY"):
        return True, ""
    model = env.get("EXAMPLES_MODEL", "anthropic:claude-haiku-4-5")
    provider = model.split(":", 1)[0] if ":" in model else ""
    key = PROVIDER_KEYS.get(provider)
    if key is None:
        return False, "DO_MODEL_ACCESS_KEY (or EXAMPLES_MODEL with its provider's key)"
    return (True, "") if env.get(key) else (False, f"DO_MODEL_ACCESS_KEY, or {key} for EXAMPLES_MODEL={model}")


def hourly_price(options: Mapping[str, object], size: MicroVMSize) -> float | None:
    """The price of one running MicroVM of `size` per hour, from the `options` pricing; None when it is not there."""
    pricing = options.get("resource_pricing")
    if not isinstance(pricing, Mapping):
        return None
    try:
        return float(pricing["price_per_vcpu_hour"]) * size.cpu + float(pricing["price_per_gib_memory_hour"]) * size.memory / 1024
    except (KeyError, TypeError, ValueError):
        return None


def evaluate(env: Mapping[str, str], probes: Probes | None, *, local: bool, only: set[str] | None = None) -> Report:
    """`probes` None: offline (the variables only)."""
    report = Report()
    add = report.checks.append
    transport = env.get("DO_MICROVM_TRANSPORT", "endpoint").lower()
    local = local or transport == "local"
    token = bool(env.get("DO_MICROVM_API_TOKEN"))
    host_local = env.get("DO_MICROVM_HOST_MODE", "local").lower() != "microvm"

    add(Line(True if local else None, "transport",
             "local: sandboxes are directories on this machine" if local else f"{transport}: real MicroVMs"))
    try:
        namespace = validate_namespace(env.get("DO_MICROVM_NAMESPACE"))
        add(Line(True, "Namespace", namespace or "none (the default namespace)"))
    except ValueError as exc:
        add(Line(False, "Namespace", str(exc)))
    if not local:
        if not token:
            add(Line(False, "Control token", "DO_MICROVM_API_TOKEN is not set"))
        elif probes is None:
            add(Line(None, "Control token", "set (not checked: offline)"))
        else:
            try:
                team = report.team = probes.team()
                limit = "no limit reported" if team.limit is None else f"limit {team.limit}, {max(team.limit - team.live, 0)} free"
                price = "" if team.hourly is None else f"; a running {env.get('DO_MICROVM_SIZE', '1vcpu-2gb')} MicroVM is ${team.hourly:.3f}/hour"
                add(Line(True, "Control token", f"accepted; {team.live} MicroVM(s) on the team ({team.mine} yours), {limit}{price}"))
            except MicroVMAPIError as exc:
                hint = {401: "the token is wrong or expired", 403: "the token lacks the microvm:read scope"}.get(
                    exc.status_code, exc.message)
                add(Line(False, "Control token", f"{exc.status_code}: {hint}"))
            except (MicroVMClientError, httpx.HTTPError) as exc:
                add(Line(False, "Control token", f"could not reach the API: {exc}"))
        if transport == "endpoint":
            have = bool(env.get("DO_MICROVM_ACCESS_TOKEN"))
            add(Line(have, "Access token", "set (checked by --deep)" if have else "DO_MICROVM_ACCESS_TOKEN is not set "
                                                                                    "(the endpoint transport needs it)"))
        if token and probes is not None and probes.deep is not None:
            problem = probes.deep()
            add(Line(problem is None, "create, exec, pause, delete (scopes, image, access token)",
                     problem or "a throwaway MicroVM worked"))

    have_model, missing_model = model_variables(env)
    if not have_model:
        add(Line(False, "Model", f"missing {missing_model}"))
    elif env.get("DO_MODEL_ACCESS_KEY") and probes is not None:
        problem = probes.model_key()
        add(Line(problem is None, "Model (DO_MODEL_ACCESS_KEY)", problem or "accepted by Serverless Inference"))
    else:
        add(Line(None if probes is None else True, "Model",
                 "variables set" + (" (not checked: offline)" if probes is None else "")))

    gateway_vars = [k for k in ("DO_ACTION_GATEWAY_TOKEN", "DO_ACTION_GATEWAY_ACTOR_ID") if not env.get(k)]
    if gateway_vars:
        add(Line(None, "Action Gateway (09, 12 real mode)", f"not set up: {', '.join(gateway_vars)}"))
    elif probes is not None:
        problem = probes.gateway()
        add(Line(problem is None, "Action Gateway token", problem or "accepted"))
    else:
        add(Line(None, "Action Gateway token", "set (not checked: offline)"))

    for example in EXAMPLES:
        if only and example.number not in only:
            continue
        missing: list[str] = []
        if example.model and not have_model:
            missing.append(missing_model)
        if not local:
            if not token:
                missing.append("DO_MICROVM_API_TOKEN")
            if transport == "endpoint" and not env.get("DO_MICROVM_ACCESS_TOKEN"):
                missing.append("DO_MICROVM_ACCESS_TOKEN")
            if example.image and not env.get("DO_MICROVM_IMAGE"):
                missing.append("DO_MICROVM_IMAGE")
            if example.host_mode:
                if host_local:
                    missing.append("DO_MICROVM_HOST_MODE=microvm")
                if example.number == "04" and not env.get("DO_MICROVM_AGENT_IMAGE"):
                    missing.append("DO_MICROVM_AGENT_IMAGE")
            if example.gateway:
                missing += gateway_vars
            missing += [k for k in example.other if not env.get(k)]
        status = "ready" if not missing else "missing"
        if not missing and not local and report.team is not None and report.team.limit is not None:
            free = report.team.limit - report.team.live
            if free < example.microvms:
                status = "no room"
                missing.append(f"needs {example.microvms} MicroVM(s), the team has {max(free, 0)} free")
        if status == "ready" and example.setup and not local:
            status = "ready*"
        report.examples.append((example, status, missing))
    return report


def render(report: Report, *, local: bool) -> str:
    out = []
    for line in report.checks:
        mark = {True: "ok  ", False: "FAIL", None: "--  "}[line.ok]
        out.append(f"[{mark}] {line.what}: {line.detail}")
    out.append("")
    out.append(f"{'Example':<34} {'MicroVMs':>8}  Status")
    for example, status, missing in report.examples:
        label = f"{example.number} {example.name}" + (" [own setup]" if example.maintainer else "")
        tail = "" if not missing else ": " + "; ".join(missing)
        extra = f" (also needs {example.setup})" if status == "ready*" else ""
        out.append(f"{label:<34} {example.microvms:>8}  {status}{tail}{extra}")
    out.append("")
    out.append("MicroVMs: what a run holds at once (paused ones count against the team's limit); keep that many free. "
               "`ready*`: the real run also needs what is named.")
    if local:
        out.append("Local mode checked: no DigitalOcean token needed. Re-run without --local for real MicroVMs. "
                   "In local mode an agent's shell commands run on your machine.")
    return "\n".join(out)


def _http_probe(url: str, key: str, *, params: Mapping[str, str] | None = None) -> str | None:
    try:
        response = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, params=params, timeout=15.0)
    except httpx.HTTPError as exc:
        return f"could not reach it: {exc}"
    if response.status_code == 200:
        return None
    if response.status_code in (401, 403):
        return f"{response.status_code}: the key or token was refused"
    return f"unexpected answer {response.status_code}"


def live_probes(env: Mapping[str, str], *, deep: bool) -> tuple[Probes, MicroVMClient | None]:
    token = env.get("DO_MICROVM_API_TOKEN")
    client = MicroVMClient(token) if token else None
    namespace = validate_namespace(env.get("DO_MICROVM_NAMESPACE"))

    def team() -> Team:
        assert client is not None
        options = client.options()
        limits = options.get("account_limits")
        limit = limits.get("max_total_count") if isinstance(limits, Mapping) else None
        live = [s for s in client.list() if s.status not in GONE]
        mine = [s for s in list_owned(client, namespace=namespace) if is_mine(s.tags)]
        size = MicroVMSize.parse(env.get("DO_MICROVM_SIZE", "1vcpu-2gb"))
        return Team(limit=int(limit) if limit is not None else None, live=len(live), mine=len(mine),
                    hourly=hourly_price(options, size))

    def deep_check() -> str | None:
        assert client is not None
        from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle

        sandbox_id = f"check-{os.getpid()}"
        config = MicroVMConfig(api_token=token, access_token=env.get("DO_MICROVM_ACCESS_TOKEN"),
                               image=env.get("DO_MICROVM_IMAGE"), namespace=namespace,
                               transport=env.get("DO_MICROVM_TRANSPORT", "endpoint"))  # type: ignore[arg-type]
        lifecycle = MicroVMLifecycle(config, client=client)
        try:
            backend = lifecycle.get_or_create(sandbox_id=sandbox_id)
            result = backend.execute("echo ok")
            lifecycle.pause(backend)
            return None if "ok" in result.output else f"exec returned {result.output!r}"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            try:
                lifecycle.delete(sandbox_id=sandbox_id)
            except Exception as exc:  # leave it for scripts/delete_all.py
                print(f"could not delete the throwaway MicroVM for sandbox {sandbox_id}: {exc}", file=sys.stderr)

    return Probes(
        team=team,
        model_key=lambda: _http_probe(INFERENCE_URL, env.get("DO_MODEL_ACCESS_KEY", "")),
        gateway=lambda: _http_probe(ACTION_GATEWAY_URL, env.get("DO_ACTION_GATEWAY_TOKEN", ""), params={"per_page": "1"}),
        deep=deep_check if deep and client else None,
    ), client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="check the variables only; no network")
    parser.add_argument("--deep", action="store_true",
                        help="also create, exec in, pause and delete one throwaway MicroVM (checks scopes, image, access token)")
    parser.add_argument("--local", action="store_true", help="check for local mode (no DigitalOcean token needed)")
    parser.add_argument("--examples", help="comma-separated numbers, e.g. 01,05,10 (default: all)")
    add_env_file_argument(parser)
    args = parser.parse_args(argv)
    env = merged_env(args.env_file if args.env_file else DEFAULT_ENV_FILE)
    only = {n.strip().zfill(2) for n in args.examples.split(",")} if args.examples else None

    client = None
    probes = None
    if not args.offline and not args.local:
        probes, client = live_probes(env, deep=args.deep)
    elif not args.offline:  # local mode still checks the model key
        probes, client = live_probes({k: v for k, v in env.items() if k != "DO_MICROVM_API_TOKEN"}, deep=False)
    try:
        report = evaluate(env, probes, local=args.local, only=only)
    finally:
        if client is not None:
            client.close()
    local = args.local or env.get("DO_MICROVM_TRANSPORT", "").lower() == "local"
    print(render(report, local=local))
    failed = any(line.ok is False for line in report.checks) or any(s in ("missing", "no room") for _, s, _ in report.examples)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
