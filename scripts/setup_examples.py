"""Set up what the examples need for a real run, step by step; `--dry-run` checks it without creating anything.

    uv run python scripts/setup_examples.py [--examples 09,07] [--dry-run] [--skip-repo] [--skip-droplet] [--refresh-repo]

Without `--examples`, the common base: the MicroVMs control and access tokens, the model access key, the sandbox image
and room on the team, which is what examples 01, 03, 05, 06, 08, 10 and 11 need. `--examples` adds each example's own
steps (09: the Action Gateway token and actor, the demo repository, the droplet; 12: the same, plus the agent-host
images, the host mode and the broker URL; 04: the default agent-host image and the host mode; 02: the `dcode` CLI;
07: the optional Tavily key); steps shared by several examples run once. The script is the write side of
scripts/check_setup.py and reads the same per-example requirements.

Each step checks first and changes something only when it is missing or refused, so the script can be run again. It
starts with recommendations (a team or namespace of its own, token expiry, a spend threshold, the team limit, clean-up):

- Prerequisites: the commands the selected steps run (for 09 and 12: git, bash, gh and its login, ssh, ssh-keygen; for
  04 and 12: uv and crane; crane when the sandbox image is built; for 02: dcode), each with where to get it. A missing one stops the script before it asks for
  anything.
- Tokens. DigitalOcean personal access tokens and model access keys are created in the Control Panel, so for each one
  that is missing or refused the script names the page and the scopes, reads the value without echoing it, checks it
  and writes it to `.env` (mode 0600). Checked: the control token lists MicroVMs; the access token cannot create
  MicroVMs (checked with an empty create request, which creates nothing); the model key can call each model the
  selected examples use; the Action Gateway token reads the tool catalog. A DigitalOcean token that can also read
  droplets or the account (Full Access, Read Only, or extra custom scopes) is reported as broader than needed; a
  pasted one is saved only if you confirm, one already in `.env` is kept with the warning. The control token then
  reports the team's MicroVM limit against what each selected example holds.
- `DO_MICROVM_IMAGE`: kept when set; otherwise built with scripts/build_sandbox_image.py into your DigitalOcean
  Container Registry. The registry token is passed to the build in its environment and is written to `.env` only if
  you say so.
- 09: `DO_ACTION_GATEWAY_ACTOR_ID`, asked for when unset, and the actor's connections (a read-only list): an active
  DigitalOcean and GitHub connection each; a missing one is shown with where its connect link appears. `DO_INCIDENT_REPO`: a public GitHub repository of your own
  (through the `gh` CLI and its login), filled with examples/incident_common/service/make_demo_repo.sh. An existing
  repository is overwritten only after you type its name; `--refresh-repo` rebuilds it (the change analyst reports the
  commit's age, so rebuild it shortly before a run). The droplet: examples/incident_common/setup_droplet.py, which shows
  the Action Gateway connect links (GitHub and DigitalOcean) for the actor the first time; then
  examples/09_incident_response/prepare.py --clean (09).
- 04 and 12: `DO_MICROVM_HOST_MODE=microvm` (asked before it is written). 04: the default agent-host image
  (`DO_MICROVM_AGENT_IMAGE`, images/agent-host/build_image.py without a spec). 12: the commander's and the
  specialists' images (`DO_INCIDENT_HOST_IMAGE`, `DO_INCIDENT_SPECIALISTS_IMAGE`), built with images/agent-host/build_image.py --spec into
  the same registry; a set image is checked against its sources (its tag is the content hash build_image.py computes
  locally with --no-push) and rebuilt when they differ. `DO_INCIDENT_BROKER_URL`: the script prints the tunnel
  command and asks for the https URL; it does not start the tunnel. A set URL is checked with one GET: a name that
  does not resolve, no answer or a 530 fails; any other answer passes, a 502, 503 or 504 included (the relay does
  not run during setup). Action Gateway endpoints named `microvm-ag-*` (a running relay's, or left by earlier runs)
  are counted (read-only).
- Last: scripts/check_setup.py for the selected examples.

What is already set is checked, not only looked up: the image's manifest (crane; with DO_REGISTRY_TOKEN for a
DigitalOcean registry), the repository (public, the demo's two commits and service/app.py), the droplet's service
(`/health`), the actor's connections, 12's images (their tags against their sources) and its broker URL. A full run treats a failed check as missing and offers to redo that step.

`--dry-run` runs only the read-only checks, creates nothing (no MicroVM, endpoint, droplet or repository) and exits 0
when everything it can check passes; it lists what only the full run checks (for 09: the template and a gateway
endpoint with the example's tools, through prepare.py). Nothing secret is printed or passed on a command line.
The full run requires a terminal (the values are typed in).
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from _env import DEFAULT_ENV_FILE, add_env_file_argument
from check_setup import (
    ACTION_GATEWAY_URL,
    EXAMPLES,
    _http_probe,
    _model_probe,
    _needed_models,
)
from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.digitalocean.com/v2"
TOKENS_PAGE = "https://cloud.digitalocean.com/account/api/tokens"
MODEL_KEYS_PAGE = "https://cloud.digitalocean.com/model-studio/manage-keys"
REGISTRY_PAGE = "https://cloud.digitalocean.com/registry"
TAVILY_PAGE = "https://app.tavily.com"
DEMO_REPO_SCRIPT = ROOT / "examples" / "incident_common" / "service" / "make_demo_repo.sh"
DROPLET_STATE = ROOT / "examples" / "incident_common" / "outputs" / "incident.json"
DROPLET_ENV = DROPLET_STATE.with_name("incident.env")
REPO_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._][A-Za-z0-9._-]*")  # never an option to gh or git
AGENT_HOST_BUILD = ROOT / "images" / "agent-host" / "build_image.py"
EXAMPLE_12 = ROOT / "examples" / "12_incident_response_agents"
# (variable, what runs it, spec, repository name when the variable is unset)
HOST_IMAGES = (
    ("DO_INCIDENT_HOST_IMAGE", "the commander's host", EXAMPLE_12 / "host" / "image.json", "microvm-incident-commander"),
    ("DO_INCIDENT_SPECIALISTS_IMAGE", "the specialists' hosts", EXAMPLE_12 / "specialists" / "image.json",
     "microvm-incident-specialists"),
)
# 04's hosts: the default agent host (images/agent-host, no spec)
AGENT_IMAGE = ("DO_MICROVM_AGENT_IMAGE", "04's agent hosts", None, "microvm-subagent")
CLOUDFLARED_PAGE = "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"

BY_NUMBER = {example.number: example for example in EXAMPLES}
BASE = ("01", "03", "05", "06", "08", "10", "11")  # the examples the common base covers


@dataclass(frozen=True)
class Plan:
    """What the selected examples need, so each step runs once for all of them."""

    examples: tuple[str, ...]
    model: bool
    image: bool
    gateway: bool
    incident: bool  # 09 and 12: the actor, the demo repository, the droplet
    hosts: bool  # 04 and 12: DO_MICROVM_HOST_MODE=microvm and agent-host images
    agents: bool  # 12: its two images and the broker URL
    dcode: bool  # 02
    tavily: bool  # 07, optional

    @classmethod
    def of(cls, numbers: Sequence[str]) -> Plan:
        chosen = [BY_NUMBER[n] for n in numbers]
        return cls(examples=tuple(numbers), model=any(e.model for e in chosen), image=any(e.image for e in chosen),
                   gateway=any(e.gateway for e in chosen), incident=bool({"09", "12"} & set(numbers)),
                   hosts=bool({"04", "12"} & set(numbers)), agents="12" in numbers, dcode="02" in numbers,
                   tavily="07" in numbers)

    @property
    def incident_label(self) -> str:
        """Which selected examples the incident steps serve, for the headings: "09", "12" or "09, 12"."""
        return ", ".join(n for n in ("09", "12") if n in self.examples)

    @property
    def hosts_label(self) -> str:
        return ", ".join(n for n in ("04", "12") if n in self.examples)

    @property
    def microvms(self) -> int:
        return max(BY_NUMBER[n].microvms for n in self.examples)


@dataclass(frozen=True)
class Secret:
    """A value created in a provider's console: where, with which scopes, and how to check it."""

    key: str
    what: str
    page: str
    scopes: str
    check: Callable[[str, Mapping[str, str]], str | None]  # None when the value works, else why not
    optional: bool = False


def _status(url: str, token: str) -> int | None:
    try:
        return httpx.get(url, headers={"Authorization": f"Bearer {token}"}, params={"per_page": "1"}, timeout=15.0).status_code
    except httpx.HTTPError:
        return None


def check_control(token: str, env: Mapping[str, str]) -> str | None:
    return _http_probe(f"{API}/microvms", token, params={"per_page": "1"})


def check_access(token: str, env: Mapping[str, str]) -> str | None:
    if token == env.get("DO_MICROVM_API_TOKEN"):
        return "this is the control token; create a separate token with microvm:access only"
    # An empty create request, which creates nothing; the check reads 400 or 422 (the body) as a token that may create.
    try:
        status = httpx.post(f"{API}/microvms", headers={"Authorization": f"Bearer {token}"}, json={}, timeout=15.0).status_code
    except httpx.HTTPError:
        return "could not reach the API"
    if status in (400, 422):
        return "this token can create MicroVMs; create one with microvm:access only"
    if status == 401:
        return "401: the token was refused"
    return None  # no create scope; its use on an endpoint is checked when a sandbox is reached


def model_checker(models: Sequence[str]) -> Callable[[str, Mapping[str, str]], str | None]:
    def check(key: str, env: Mapping[str, str]) -> str | None:
        for model in models:
            if problem := _model_probe(key, model):
                return f"{model}: {problem}"
        return None
    return check


def check_gateway(token: str, env: Mapping[str, str]) -> str | None:
    return _http_probe(ACTION_GATEWAY_URL, token, params={"per_page": "1"})


def secrets_for(plan: Plan, env: Mapping[str, str]) -> list[Secret]:
    """The tokens the plan needs, in the order they are asked for."""
    secrets = [
        Secret("DO_MICROVM_API_TOKEN", "MicroVMs control token", TOKENS_PAGE,
               "Custom Scopes: microvm:create, microvm:read, microvm:update, microvm:delete", check_control),
        Secret("DO_MICROVM_ACCESS_TOKEN", "MicroVM endpoint access token", TOKENS_PAGE,
               "Custom Scopes: microvm:access only", check_access),
    ]
    if plan.model:
        models = _needed_models(env, set(plan.examples))
        secrets.append(Secret("DO_MODEL_ACCESS_KEY", "Serverless Inference model access key", MODEL_KEYS_PAGE,
                              f"Foundation models: {' and '.join(models)} (or All models)", model_checker(models)))
    if plan.gateway:
        secrets.append(Secret("DO_ACTION_GATEWAY_TOKEN", "Action Gateway token", TOKENS_PAGE,
                              "Custom Scopes: every action_gateway scope", check_gateway))
    if plan.tavily:
        secrets.append(Secret("TAVILY_API_KEY", "Tavily API key (07: web search for the newsletter agent)", TAVILY_PAGE,
                              "optional: without it 07 runs without web search", lambda value, env: None, optional=True))
    return secrets


BROADER_CHECK = ("DO_MICROVM_API_TOKEN", "DO_MICROVM_ACCESS_TOKEN", "DO_ACTION_GATEWAY_TOKEN")
BROADER_WARNING = ("accepted, and it can also read droplets or the account: it has more scopes than this setup needs "
                   "(Full Access, Read Only or extra custom scopes). Create it with only the scopes named to limit what "
                   "it can do")
# Reads none of these tokens needs; a 200 on either means more scopes than the setup asks for (Full Access included).
UNNEEDED_READS = (f"{API}/droplets", f"{API}/account")


def recommendations(plan: Plan) -> str:
    lines = [
        "Before you start",
        "- Use a DigitalOcean team of its own for examples, or set DO_MICROVM_NAMESPACE (1-16 lowercase letters or",
        "  digits), so these MicroVMs and the clean-up scripts stay apart from other work on the team.",
        "- Give each token a short expiration; a run takes minutes. Create each one with Custom Scopes, exactly the",
        "  scopes named; a token with more (Full Access, Read Only or extra scopes) is reported, and you choose whether",
        "  to keep it.",
    ]
    if plan.model:
        lines.append("- Set a spend threshold on the model access key (a spend alert on Managed Access Keys (Serverless "
                     "Inference)).")
    lines += [f"- The team needs room for up to {plan.microvms} MicroVMs at once (paused ones count); the limit is "
              "checked below.",
              "- `uv run scripts/delete_all.py` lists and deletes MicroVMs a stopped run left."]
    if plan.incident:
        label = plan.incident_label
        lines += [f"- {label}: the droplet and its snapshot are billed until "
                  "`uv run examples/incident_common/setup_droplet.py --cleanup`.",
                  f"- {label}: the actor's GitHub and DigitalOcean connections stay after the clean-up; delete them when",
                  "  you no longer need them (Action Gateway documentation: How to Manage Connections)."]
    if plan.hosts:
        lines += [f"- {plan.hosts_label}: the agent-host images contain this package's code: keep them in a private",
                  "  registry. Rebuild them after changing the package or the example; this script reports images built",
                  "  from older sources."]
    if plan.agents:
        lines += ["- 12: with a tunnel, the specialists' action calls and their answers pass through the tunnel's provider;",
                  "  for anything longer-lived, see the example's README (The broker's URL)."]
    return "\n".join(lines)


# (command, why, where to get it, needed when)
TOOLS: tuple[tuple[str, str, str, str], ...] = (
    ("git", "builds and pushes the demo repository", "https://git-scm.com/downloads", "repo"),
    ("bash", "runs make_demo_repo.sh", "preinstalled on macOS and Linux", "repo"),
    ("gh", "creates the public demo repository with your GitHub login", "https://cli.github.com", "repo"),
    ("ssh", "installs the service on the droplet", "OpenSSH, preinstalled on macOS and most Linux", "droplet"),
    ("ssh-keygen", "creates the droplet's temporary key pair", "OpenSSH", "droplet"),
    ("uv", "builds the agent-host images (04, 12)", "https://docs.astral.sh/uv/getting-started/installation/", "agents"),
    ("crane", "pushes the images (the sandbox image when DO_MICROVM_IMAGE is unset; the agent-host images of 04 and 12)",
     "https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md", "image"),
    ("dcode", "Deep Agents Code, example 02's CLI (Python 3.12+)",
     "`uv sync --all-extras` in this checkout, or examples/02_deepagents_code/README.md", "dcode"),
)


def _which(command: str) -> str | None:
    """On PATH, or in this interpreter's environment (where `uv sync` installs dcode)."""
    beside = Path(sys.executable).parent / command
    return shutil.which(command) or (str(beside) if beside.exists() else None)


def broader_than_needed(token: str) -> bool:
    return any(_status(url, token) == 200 for url in UNNEEDED_READS)


# ── read-only checks of what is already set: (True | False | None for not checked, detail) ─────────────────────────
Verdict = tuple[bool | None, str]


def verify_repo(repo: str) -> Verdict:
    """Public, at least two commits, and the demo service's file: what setup_droplet.py clones and deploys."""
    if not REPO_NAME.fullmatch(repo):
        return False, "expected owner/name"
    if shutil.which("gh") is None:
        return None, "not checked: gh is not installed"
    visibility = _run(["gh", "api", f"repos/{repo}", "--jq", ".visibility"])
    if visibility is None:
        return False, "not found (or gh cannot see it)"
    if visibility != "public":
        return False, f"is {visibility}; the droplet clones it without credentials, so it must be public"
    commits = _run(["gh", "api", f"repos/{repo}/commits?per_page=2", "--jq", "length"])
    if commits is None or not commits.isdigit() or int(commits) < 2:
        return False, "needs the demo's two commits (make_demo_repo.sh)"
    if _run(["gh", "api", f"repos/{repo}/contents/service/app.py", "--jq", ".name"]) != "app.py":
        return False, "has no service/app.py: not the demo repository"
    return True, "public, the demo's commits and service/app.py"


def verify_image(image: str, registry_token: str | None) -> Verdict:
    """The image's manifest can be read: anonymously, or with the registry token for a DigitalOcean registry."""
    if shutil.which("crane") is None:
        return None, "not checked: crane is not installed (the first sandbox boots it)"
    registry = image.split("/", 1)[0]
    private = registry == "registry.digitalocean.com"
    if private and not registry_token:
        return None, "not checked: set DO_REGISTRY_TOKEN (registry:read) to check it here (the first sandbox boots it)"
    with tempfile.TemporaryDirectory(prefix="microvm-image-check-") as tmp:
        env = {**os.environ, "DOCKER_CONFIG": tmp}
        if private and registry_token:
            auth = base64.b64encode(f"{registry_token}:{registry_token}".encode()).decode()
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"auths": {registry: {"auth": auth}}}))
            config.chmod(0o600)
        done = subprocess.run(["crane", "manifest", "--platform", "linux/amd64", image], env=env,
                              capture_output=True, text=True, check=False)
    if done.returncode:
        return False, f"cannot be read: {done.stderr.strip().splitlines()[-1][:200] if done.stderr.strip() else 'crane failed'}"
    return True, "found (linux/amd64)"


INCIDENT_PROVIDERS = {"digitalocean": "DigitalOcean", "github": "GitHub"}  # the actor's accounts 09 and 12 use


def verify_connections(token: str, actor: str) -> Verdict:
    """The actor has an active DigitalOcean and GitHub connection (a read of the token user's connections)."""
    connections: list[dict[str, object]] = []
    page = 1
    try:
        while page <= 20:
            response = httpx.get(f"{API}/action-gateway/connections", headers={"Authorization": f"Bearer {token}"},
                                 params={"page": page, "per_page": 100}, timeout=15.0)
            if response.status_code != 200:
                return None, f"{actor}; its connections not listed ({response.status_code})"
            body = response.json()
            connections += body.get("connections") or []
            if len(connections) >= int((body.get("pagination") or {}).get("total") or 0) or not body.get("connections"):
                break
            page += 1
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"{actor}; its connections not listed: {exc}"
    mine = [c for c in connections if c.get("user_id") == actor]
    active = {c.get("provider") for c in mine if c.get("status") == "active"}
    missing = [name for slug, name in INCIDENT_PROVIDERS.items() if slug not in active]
    if not missing:
        return True, f"active {' and '.join(INCIDENT_PROVIDERS.values())} connections"
    other = {str(c.get("provider")): str(c.get("status")) for c in mine if c.get("provider") in INCIDENT_PROVIDERS
             and c.get("status") != "active"}
    states = "".join(f"; {INCIDENT_PROVIDERS[p]} is {s}" for p, s in other.items())
    return False, f"no active {' or '.join(missing)} connection for {actor}{states}"


def verify_broker(url: str) -> Verdict:
    """One GET of the broker URL. Fails when the name does not resolve, nothing answers, or the answer is 530; a 502,
    503 or 504 passes (the relay is not running during setup), and so does any other answer."""
    host = urlparse(url).hostname or ""
    try:
        socket.getaddrinfo(host, 443)
    except (socket.gaierror, UnicodeError):
        return False, "the name does not resolve: start the tunnel and set the URL it prints"
    try:
        status = httpx.get(url, timeout=10.0).status_code
    except httpx.HTTPError as exc:
        return False, f"no answer ({type(exc).__name__}): start the tunnel and set the URL it prints"
    if status == 530:
        return False, "answers 530: start the tunnel and set the URL it prints"
    if status in (502, 503, 504):
        return True, f"answers {status}; the broker answers once the relay runs"
    return True, f"answers {status}"


def verify_droplet() -> Verdict:
    """The droplet's service answers /health (healthy, or failing after --deploy-bad)."""
    url = dotenv_values(DROPLET_ENV).get("DO_INCIDENT_URL") if DROPLET_ENV.exists() else None
    if not url:
        return False, f"{DROPLET_ENV.relative_to(ROOT)} names no DO_INCIDENT_URL: run setup_droplet.py --cleanup, then again"
    try:
        status = httpx.get(f"{url.rstrip('/')}/health", timeout=10.0).json().get("status")
    except (httpx.HTTPError, ValueError, AttributeError):
        status = None
    if status not in ("healthy", "failing"):
        return False, "the service does not answer: run setup_droplet.py --cleanup, then this script again"
    return True, f"the service answers: {status}"


class Setup:
    def __init__(self, env_file: Path, plan: Plan, *, dry_run: bool) -> None:
        self.env_file = env_file
        self.plan = plan
        self.dry_run = dry_run
        self.env = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.exists() else {}
        self.todo: list[str] = []
        self.unchecked: list[str] = []  # set, but only the full run can check it
        self.step = 0
        self._registry_token = ""  # typed in once per run when DO_REGISTRY_TOKEN is unset
        self._registry_token_offered = False

    # ── output and input ──────────────────────────────────────────────────────
    def heading(self, title: str) -> None:
        print(f"\n{self.step}. {title}")
        self.step += 1

    @staticmethod
    def say(mark: str, what: str, detail: str = "") -> None:
        print(f"[{mark:<4}] {what}" + (f": {detail}" if detail else ""), flush=True)

    def verdict(self, what: str, verdict: Verdict, fix: str) -> bool | None:
        """Report a read-only check; a failure in a dry run adds `fix` to the to-do list."""
        ok, detail = verdict
        self.say({True: "ok", False: "FAIL", None: "--"}[ok], what, detail)
        if ok is None:
            self.unchecked.append(what)
        if ok is False and self.dry_run:
            self.todo.append(fix)
        return ok

    def save(self, key: str, value: str) -> None:
        if not self.env_file.exists():
            self.env_file.touch(mode=0o600)
        set_key(str(self.env_file), key, value, quote_mode="never")
        self.env_file.chmod(0o600)
        self.env[key] = value

    @staticmethod
    def ask(prompt: str, default: str = "") -> str:
        answer = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
        return answer or default

    def confirm(self, prompt: str) -> bool:
        return self.ask(f"{prompt} (y/N)").lower() in ("y", "yes")

    def broader(self, secret: Secret, value: str) -> bool:
        """True (after a warning) when a DigitalOcean token has scopes this setup does not need; using it is your choice."""
        if secret.key not in BROADER_CHECK or not broader_than_needed(value):
            return False
        self.say("warn", secret.key, BROADER_WARNING)
        return True

    # ── steps ─────────────────────────────────────────────────────────────────
    def prerequisites(self, *, repo: bool, droplet: bool) -> bool:
        self.heading("Prerequisites")
        needed = {"repo": repo, "droplet": droplet, "agents": self.plan.hosts, "dcode": self.plan.dcode,
                  "image": (self.plan.image and not self.env.get("DO_MICROVM_IMAGE")) or self.plan.hosts}
        ok = True
        for command, why, where, when in TOOLS:
            if not needed[when]:
                continue
            if _which(command):
                self.say("ok", command)
            else:
                self.say("FAIL", command, f"not found; {why}. Install: {where}")
                ok = False
        if self.plan.dcode and sys.version_info < (3, 12):
            self.say("FAIL", "Python", f"{sys.version.split()[0]}; dcode (02) needs Python 3.12+")
            ok = False
        if repo and shutil.which("gh"):
            if _run(["gh", "auth", "status"]) is None:
                self.say("FAIL", "gh login", "run `gh auth login` (scope: repo)")
                ok = False
            else:
                self.say("ok", "gh login")
        if not any(needed.values()):
            self.say("ok", "nothing to install for the selected steps")
        if not ok:
            self.todo.append("install the tools marked FAIL under Prerequisites")
        return ok

    def team_room(self) -> None:
        """The team's MicroVM limit against what each selected example holds, as a recommendation; it changes nothing."""
        from check_setup import live_probes

        probes, client = live_probes(self.env, deep=False)
        try:
            team = probes.team()
        except Exception as exc:  # noqa: BLE001 - only a report
            self.say("warn", "team limit", f"not read: {exc}")
            return
        finally:
            if client is not None:
                client.close()
        if team.limit is None:
            self.say("ok", "team limit", f"none reported; {team.live} MicroVM(s) on the team")
            return
        free = team.limit - team.live
        short = [f"{n} ({BY_NUMBER[n].microvms})" for n in self.plan.examples if BY_NUMBER[n].microvms > free]
        detail = f"{team.limit}, {team.live} MicroVM(s) on the team, {max(free, 0)} free"
        if short:
            self.say("warn", "team limit", f"{detail}; too few for {', '.join(short)}: delete some, or have the team's "
                     "limit raised")
        else:
            self.say("ok", "team limit", f"{detail}; enough for each selected example (up to {self.plan.microvms})")

    def tokens(self) -> bool:
        self.heading("Tokens")
        ok = True
        for secret in secrets_for(self.plan, self.env):
            value = self.env.get(secret.key, "")
            problem = secret.check(value, self.env) if value else "not set"
            if problem is None:
                if not self.broader(secret, value):
                    self.say("ok", secret.key, "set" if secret.optional else "accepted")
                continue
            if secret.optional and self.dry_run:
                self.verdict(secret.key, (None, f"not set ({secret.scopes})"), "")
                continue
            self.say("need", secret.key, problem)
            if self.dry_run:
                self.todo.append(f"create the {secret.what} ({secret.scopes}) at {secret.page}")
                ok = False
                continue
            print(f"       Create the {secret.what} at {secret.page}\n       {secret.scopes}")
            while True:
                value = getpass.getpass(f"       paste {secret.key} (input hidden, empty to skip): ").strip()
                if not value:
                    ok &= secret.optional
                    break
                if (problem := secret.check(value, self.env)) is None:
                    if self.broader(secret, value) and not self.confirm("       keep this token anyway?"):
                        continue
                    self.save(secret.key, value)
                    self.say("ok", secret.key, f"accepted, saved to {self.env_file.name}")
                    break
                self.say("FAIL", secret.key, problem)
        if self.env.get("DO_MICROVM_API_TOKEN") and check_control(self.env["DO_MICROVM_API_TOKEN"], self.env) is None:
            self.team_room()
        return ok

    def actor(self) -> bool:
        self.heading(f"Action Gateway actor ({self.plan.incident_label})")
        actor = self.env.get("DO_ACTION_GATEWAY_ACTOR_ID", "")
        if not actor:
            if self.dry_run:
                self.todo.append("set DO_ACTION_GATEWAY_ACTOR_ID (the part of your DigitalOcean email before the @)")
                return False
            actor = self.ask("       DO_ACTION_GATEWAY_ACTOR_ID (the part of your DigitalOcean email before the @)")
            if not actor:
                return False
            self.save("DO_ACTION_GATEWAY_ACTOR_ID", actor)
        token = self.env.get("DO_ACTION_GATEWAY_TOKEN", "")
        verdict = (verify_connections(token, actor) if token and check_gateway(token, self.env) is None
                   else (None, f"{actor}; its connections not listed: the Action Gateway token is not accepted"))
        ok = self.verdict(f"DO_ACTION_GATEWAY_ACTOR_ID {actor}", verdict,
                          f"connect {actor}'s missing accounts: setup_droplet.py shows the connect link on its first call")
        if ok is False and not self.dry_run:
            # Not a stop: the first gateway call that needs the account shows its connect link (setup_droplet.py, the run).
            self.say("warn", "connections", "the next gateway call that needs the account shows the connect link")
            return True
        return ok is not False

    def image(self) -> bool:
        self.heading("Sandbox image")
        if image := self.env.get("DO_MICROVM_IMAGE"):
            ok = self.verdict(f"DO_MICROVM_IMAGE {image}", verify_image(image, self.env.get("DO_REGISTRY_TOKEN")),
                              "rebuild the sandbox image (unset DO_MICROVM_IMAGE and run this script)")
            if ok is not False:
                return True
            if self.dry_run or not self.confirm("       build a new sandbox image?"):
                return False
        if self.dry_run:
            self.todo.append("build the sandbox image (needs a registry token: registry:read, registry:update)")
            return False
        token = self.registry_token()
        if not token:
            return False
        registry = None if self.env.get("DO_MICROVM_SANDBOX_REPOSITORY") else self._registry(token)
        repository = self.env.get("DO_MICROVM_SANDBOX_REPOSITORY") or (registry and f"{registry}/microvm-sandbox")
        if not repository:
            return False
        build_env = {**os.environ, "DO_REGISTRY_TOKEN": token}
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_sandbox_image.py"), "--repository", repository,
                               "--env-file", os.devnull], env=build_env, capture_output=True, text=True, check=False)
        line = next((ln for ln in done.stdout.splitlines() if ln.startswith("Set DO_MICROVM_IMAGE=")), None)
        if done.returncode or line is None:
            self.say("FAIL", "image build", (done.stderr or done.stdout).strip()[-300:])
            return False
        self.save("DO_MICROVM_IMAGE", line.split("=", 1)[1])
        self.save("DO_MICROVM_SANDBOX_REPOSITORY", repository)
        self.offer_to_keep_registry_token()
        self.say("ok", "DO_MICROVM_IMAGE", self.env["DO_MICROVM_IMAGE"])
        return True

    def registry_token(self) -> str:
        """DO_REGISTRY_TOKEN from .env, or typed in once per run (kept in memory, saved only if you say so)."""
        if not self._registry_token:
            self._registry_token = self.env.get("DO_REGISTRY_TOKEN") or getpass.getpass(
                f"       Create a token at {TOKENS_PAGE} with Custom Scopes registry:read, registry:update\n"
                "       paste it (input hidden): ").strip()
        return self._registry_token

    def offer_to_keep_registry_token(self) -> None:
        if "DO_REGISTRY_TOKEN" not in self.env and not self._registry_token_offered:
            self._registry_token_offered = True
            if self.confirm(f"       save the registry token in {self.env_file.name} for rebuilds?"):
                self.save("DO_REGISTRY_TOKEN", self._registry_token)

    def _registry(self, token: str) -> str | None:
        try:
            response = httpx.get(f"{API}/registry", headers={"Authorization": f"Bearer {token}"}, timeout=15.0)
        except httpx.HTTPError as exc:
            self.say("FAIL", "registry", f"could not reach the API: {exc}")
            return None
        if response.status_code == 404:
            self.say("FAIL", "registry", f"the team has no container registry; create one at {REGISTRY_PAGE}")
            return None
        if response.status_code != 200:
            self.say("FAIL", "registry", f"{response.status_code}: the token was refused (registry:read)")
            return None
        return f"registry.digitalocean.com/{response.json()['registry']['name']}"

    def repo(self, *, refresh: bool) -> bool:
        self.heading(f"Demo repository ({self.plan.incident_label})")
        repo = self.env.get("DO_INCIDENT_REPO")
        if repo and not refresh:
            ok = self.verdict(f"DO_INCIDENT_REPO {repo}", verify_repo(repo), "rebuild the demo repository (--refresh-repo)")
            if ok is not False:
                return True
            if self.dry_run:
                return False
        if self.dry_run:
            self.todo.append("create or rebuild the public demo repository (gh) and set DO_INCIDENT_REPO")
            return False
        login = _run(["gh", "api", "user", "--jq", ".login"])
        if login is None:
            self.say("FAIL", "gh", "not logged in: run `gh auth login`")
            return False
        repo = self.ask("       public repository to use (owner/name)", repo or f"{login}/checkout-demo")
        if not REPO_NAME.fullmatch(repo):
            self.say("FAIL", repo, "expected owner/name (letters, digits, - . _)")
            return False
        visibility = _run(["gh", "repo", "view", repo, "--json", "visibility", "--jq", ".visibility"])
        if visibility is None:
            if not self.confirm(f"       create the public repository {repo}?"):
                return False
            if _run(["gh", "repo", "create", repo, "--public", "--description", "Demo service for an incident-response example"]) is None:
                self.say("FAIL", "gh repo create", repo)
                return False
        elif visibility != "PUBLIC":
            self.say("FAIL", repo, "is not public (the droplet clones it without credentials)")
            return False
        elif self.ask(f"       {repo} exists; its history will be replaced. Type its name to confirm") != repo:
            return False
        with tempfile.TemporaryDirectory(prefix="checkout-demo-") as tmp:
            dest = Path(tmp) / "repo"
            if _run(["bash", str(DEMO_REPO_SCRIPT), str(dest)]) is None:
                self.say("FAIL", "make_demo_repo.sh", "failed")
                return False
            # gh's credential helper, so no token is written to the remote URL or argv
            push = ["git", "-C", str(dest), "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
                    "push", "--force", f"https://github.com/{repo}.git", "HEAD:main"]
            if _run(push) is None:
                self.say("FAIL", "git push", repo)
                return False
        self.save("DO_INCIDENT_REPO", repo)
        self.say("ok", "DO_INCIDENT_REPO", f"{repo}, two commits pushed")
        if DROPLET_STATE.exists():
            self.say("warn", "droplet", "it runs the commits it cloned at creation: run setup_droplet.py --cleanup, "
                     "then this script again, so it clones these")
        return True

    def droplet(self) -> bool:
        self.heading(f"Droplet and connections ({self.plan.incident_label})")
        if DROPLET_STATE.exists():
            if not self.verdict("droplet checkout-svc", verify_droplet(),
                                "recreate the droplet: setup_droplet.py --cleanup, then this script"):
                return False
            repo = dotenv_values(DROPLET_ENV).get("DO_INCIDENT_REPO")
            if repo and repo != self.env.get("DO_INCIDENT_REPO"):
                self.say("warn", "droplet", f"was set up from {repo}, not DO_INCIDENT_REPO; recreate it (--cleanup)")
            if self.dry_run:
                if "09" in self.plan.examples:
                    self.unchecked.append("09: the template and a gateway endpoint with the example's tools (prepare.py, "
                                          "full run)")
                return True
        elif self.dry_run:
            self.say("need", "droplet", "not created")
            self.todo.append("create the droplet (setup_droplet.py; this script runs it, with the connect links)")
            return False
        else:
            print("       creates droplet checkout-svc, ssh key incident-key and snapshot checkout-svc-snapshot (billed until "
                  "setup_droplet.py --cleanup)")
            if not self.confirm("       create them now?") or not _step("examples/incident_common/setup_droplet.py"):
                return False
        return "09" not in self.plan.examples or _step("examples/09_incident_response/prepare.py", "--clean")

    # ── 04 and 12 ─────────────────────────────────────────────────────────────
    def host_mode(self) -> bool:
        self.heading(f"Agent hosts ({self.plan.hosts_label})")
        if self.env.get("DO_MICROVM_HOST_MODE", "").lower() == "microvm":
            self.say("ok", "DO_MICROVM_HOST_MODE", "microvm")
            return True
        self.say("need", "DO_MICROVM_HOST_MODE", f"{self.env.get('DO_MICROVM_HOST_MODE') or 'unset (local)'}; the real "
                 "run needs microvm (examples 04 and 12 then start their agent hosts as MicroVMs)")
        if self.dry_run:
            self.todo.append(f"set DO_MICROVM_HOST_MODE=microvm in {self.env_file.name}")
            return False
        if not self.confirm(f"       set DO_MICROVM_HOST_MODE=microvm in {self.env_file.name}?"):
            return False
        self.save("DO_MICROVM_HOST_MODE", "microvm")
        self.say("ok", "DO_MICROVM_HOST_MODE", "microvm")
        return True

    def host_images(self) -> bool:
        images = ([AGENT_IMAGE] if "04" in self.plan.examples else []) + (list(HOST_IMAGES) if self.plan.agents else [])
        ok = True
        for key, what, spec, name in images:
            ok &= self.host_image(key, what, spec, name)
        return ok

    def host_image(self, key: str, what: str, spec: Path | None, name: str) -> bool:
        """`spec` None: the default agent host (images/agent-host)."""
        image = self.env.get(key, "")
        current = content_hash(spec)
        if image:
            tag = image.rsplit(":", 1)[-1] if ":" in image.rsplit("/", 1)[-1] else ""
            if current and tag and tag != current:
                verdict: Verdict = (False, f"built from other sources (tag {tag}, the sources hash to {current}): rebuild it")
            else:
                ok_, detail = verify_image(image, self.env.get("DO_REGISTRY_TOKEN"))
                fresh = "" if current else "; whether it matches the sources was not checked (uv or crane missing)"
                verdict = (ok_, f"{detail}{'; built from these sources' if current and ok_ else ''}{fresh}")
            if self.verdict(f"{key} ({what})", verdict, f"rebuild {key} (this script, without --dry-run)") is not False:
                return True
        else:
            self.say("need", key, f"not set ({what})")
            if self.dry_run:
                self.todo.append(f"build {key} ({what}; this script, without --dry-run)")
        if self.dry_run or not self.confirm(f"       build and push {key} ({what})?"):
            return False
        token = self.registry_token()
        if not token:
            return False
        repository = (image.rsplit(":", 1)[0] if ":" in image.rsplit("/", 1)[-1] else "") or (
            self.env.get("DO_MICROVM_AGENT_REPOSITORY", "") if spec is None else "")
        if not repository:
            registry = self._registry(token)
            if not registry:
                return False
            repository = f"{registry}/{name}"
        source = spec.relative_to(ROOT) if spec else AGENT_HOST_BUILD.parent.relative_to(ROOT)
        print(f"       building {repository} from {source} (a few minutes the first time)")
        done = subprocess.run([sys.executable, str(AGENT_HOST_BUILD), *_spec_args(spec), "--repository", repository,
                               "--env-file", os.devnull], env={**os.environ, "DO_REGISTRY_TOKEN": token},
                              capture_output=True, text=True, check=False)
        pushed = next((ln.split()[1] for ln in done.stdout.splitlines() if ln.startswith("pushed ")), None)
        if done.returncode or pushed is None:
            self.say("FAIL", f"{key} build", (done.stderr or done.stdout).strip()[-300:])
            return False
        self.save(key, pushed)
        self.offer_to_keep_registry_token()
        self.say("ok", key, pushed)
        return True

    def broker(self) -> bool:
        port = self.env.get("DO_INCIDENT_BROKER_PORT") or "8788"
        tunnel = f"cloudflared tunnel --url http://127.0.0.1:{port}"
        url = self.env.get("DO_INCIDENT_BROKER_URL", "").strip()
        hint = (f"start a tunnel to the broker port, `{tunnel}` (cloudflared: {CLOUDFLARED_PAGE}), and set "
                "DO_INCIDENT_BROKER_URL to the https URL it prints")
        if url and url.startswith("https://"):
            ok = self.verdict(f"DO_INCIDENT_BROKER_URL {url}", verify_broker(url), hint)
            if ok is not False or self.dry_run:
                return ok is not False
        else:
            self.say("need", "DO_INCIDENT_BROKER_URL", "not https" if url else "not set")
        if self.dry_run:
            self.todo.append(hint)
            return False
        print(f"       In another terminal: {tunnel}\n       (cloudflared: {CLOUDFLARED_PAGE})")
        url = self.ask("       paste the https URL it prints (empty to skip)")
        if not url.startswith("https://"):
            return False
        self.save("DO_INCIDENT_BROKER_URL", url)
        return self.verdict(f"DO_INCIDENT_BROKER_URL {url}", verify_broker(url), hint) is not False

    def leftovers(self) -> None:
        """Action Gateway endpoints named microvm-ag-*: a running relay's, or left by earlier runs (read-only)."""
        from langchain_digitalocean_microvm.integrations.action_gateway import (
            ActionGatewayClient,
            ActionGatewayError,
        )

        client = ActionGatewayClient(self.env["DO_ACTION_GATEWAY_TOKEN"])
        try:
            left = client.list_endpoints(prefix="microvm-ag-")
        except (ActionGatewayError, httpx.HTTPError) as exc:
            self.say("warn", "gateway endpoints", f"not listed: {str(exc)[:120]}")
            return
        finally:
            client.close()
        if left:
            self.say("warn", "gateway endpoints", f"{len(left)} named microvm-ag-*: a running relay's, or left by earlier "
                     "runs; a relay deletes those of closed incidents when it starts, the others as the README describes "
                     "(Run (real mode))")
        else:
            self.say("ok", "gateway endpoints", "none named microvm-ag-*")

    def transport(self) -> bool:
        if self.env.get("DO_MICROVM_TRANSPORT", "").lower() != "local":
            return True
        self.say("FAIL", "DO_MICROVM_TRANSPORT", f"is `local` in {self.env_file.name}; remove it for a real run")
        self.todo.append(f"remove DO_MICROVM_TRANSPORT=local from {self.env_file.name}")
        return False


def _spec_args(spec: Path | None) -> list[str]:
    return ["--spec", str(spec)] if spec else []


def content_hash(spec: Path | None) -> str | None:
    """The tag build_image.py gives an image of `spec`'s current sources (it builds the layer locally, pushes nothing);
    `spec` None: the default agent host."""
    if shutil.which("uv") is None or shutil.which("crane") is None:
        return None
    done = subprocess.run([sys.executable, str(AGENT_HOST_BUILD), *_spec_args(spec), "--repository", "local/check",
                           "--no-push"], capture_output=True, text=True, check=False)
    found = re.search(r"content hash ([0-9a-f]+)", done.stdout)
    return found.group(1) if done.returncode == 0 and found else None


def _run(command: list[str]) -> str | None:
    """stdout of a command that prints no secret, or None when it failed."""
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def _step(script: str, *args: str) -> bool:
    print(f"\n$ uv run python {script} {' '.join(args)}".rstrip(), flush=True)
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=False).returncode == 0


def parse_examples(value: str | None, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    if not value:
        return BASE
    numbers = tuple(dict.fromkeys(n.strip().zfill(2) for n in value.split(",") if n.strip()))
    unknown = [n for n in numbers if n not in BY_NUMBER]
    if unknown:
        parser.error(f"no example {', '.join(unknown)}; the examples are {', '.join(BY_NUMBER)}")
    return numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--examples", help=f"comma-separated numbers, e.g. 09,07 (default: the base, {', '.join(BASE)})")
    parser.add_argument("--dry-run", action="store_true", help="check and report each step; change nothing")
    parser.add_argument("--skip-repo", action="store_true", help="09: leave DO_INCIDENT_REPO as it is")
    parser.add_argument("--skip-droplet", action="store_true", help="09: stop before the droplet and the pre-flight")
    parser.add_argument("--refresh-repo", action="store_true", help="09: rebuild the demo repository's two commits")
    add_env_file_argument(parser)
    args = parser.parse_args(argv)
    plan = Plan.of(parse_examples(args.examples, parser))
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("needs a terminal (tokens are typed in); --dry-run works without one")
    shadowed = [k for k in os.environ if k.startswith(("DO_MICROVM_", "DO_ACTION_GATEWAY_", "DO_MODEL_", "DO_INCIDENT_"))]
    if shadowed:
        print(f"note: set in the shell (the examples use these over {args.env_file.name}): {', '.join(sorted(shadowed))}")

    setup = Setup(args.env_file if args.env_file else DEFAULT_ENV_FILE, plan, dry_run=args.dry_run)
    print(f"Examples: {', '.join(plan.examples)}\n\n{recommendations(plan)}")
    repo, droplet = plan.incident and not args.skip_repo, plan.incident and not args.skip_droplet
    if not setup.prerequisites(repo=repo, droplet=droplet) and not args.dry_run:
        print("\nInstall what is marked FAIL, then run this again (nothing was changed).")
        return 1
    ready = setup.transport() & setup.tokens()
    if plan.incident:
        ready &= setup.actor()
    if plan.image:
        ready &= setup.image()
    if repo:
        ready &= setup.repo(refresh=args.refresh_repo)
    if ready and droplet:
        ready &= setup.droplet()
    if plan.hosts:
        ready &= setup.host_mode() & setup.host_images()
    if plan.agents:
        ready &= setup.broker()
        if setup.env.get("DO_ACTION_GATEWAY_TOKEN") and check_gateway(setup.env["DO_ACTION_GATEWAY_TOKEN"], setup.env) is None:
            setup.leftovers()
    if ready and not args.dry_run:
        ready &= _step("scripts/check_setup.py", "--examples", ",".join(plan.examples))
    if setup.todo:
        print("\nTo do:\n" + "\n".join(f"- {item}" for item in setup.todo))
    if setup.unchecked:
        print("\nNot checked by this run (marked --):\n" + "\n".join(f"- {item}" for item in setup.unchecked))
    if args.dry_run:
        print("\n" + ("Ready: every read-only check passed. Run without --dry-run to finish the setup and run "
                      "scripts/check_setup.py." if ready else "Not ready: see To do. Run without --dry-run to set it up."))
    elif ready:
        print("\nReady. Run the examples as their READMEs describe (examples/README.md).")
        if droplet and "09" in plan.examples:
            print("09 (examples/09_incident_response/README.md, Run (real mode)):\n"
                  "  uv run examples/incident_common/setup_droplet.py --deploy-bad\n"
                  "  uv run examples/09_incident_response/dashboard.py          # terminal 2\n"
                  "  uv run examples/09_incident_response/run_script.py --delete\n"
                  "  uv run examples/incident_common/setup_droplet.py --cleanup  # afterwards")
        if "04" in plan.examples:
            print("04 (examples/04_async_subagents/README.md, Real MicroVMs):\n"
                  "  uv run examples/04_async_subagents/run_script.py --with-researcher --delete")
        if plan.agents:
            print("12 (examples/12_incident_response_agents/README.md, Run (real mode)), with the tunnel running:\n"
                  "  uv run examples/incident_common/setup_droplet.py --deploy-bad\n"
                  "  uv run examples/12_incident_response_agents/console.py demo --web --delete\n"
                  "  uv run examples/incident_common/setup_droplet.py --cleanup  # afterwards")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
