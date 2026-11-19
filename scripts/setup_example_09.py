"""Set up everything example 09 (incident response) needs for a real run, step by step.

    uv run python scripts/setup_example_09.py [--dry-run] [--skip-repo] [--skip-droplet] [--refresh-repo]

Each step checks first and changes something only when it is missing or refused, so the script can be run again. It
starts with recommendations (a team or namespace of its own, token expiry, a spend threshold, the team limit, clean-up):

0. Prerequisites: the commands the later steps run (git, bash, gh and its login, ssh, ssh-keygen, crane when the image
   is built), each with where to get it. A missing one stops the script before it asks for anything.
1. Tokens. DigitalOcean personal access tokens and model access keys are created in the Control Panel, so for each one
   that is missing or refused the script names the page and the scopes, reads the value without echoing it, checks it
   and writes it to `.env` (mode 0600). Checked: the control token lists MicroVMs; the access token cannot create
   MicroVMs (an empty create request, which creates nothing, is refused with 403); the model key can call both
   models example 09 uses; the Action Gateway token reads the tool catalog. A DigitalOcean token that can also read
   droplets or the account (Full Access, Read Only, or extra custom scopes) is reported as broader than needed; a
   pasted one is saved only if you confirm, one already in `.env` is kept with the warning. The control token then
   reports the team's MicroVM limit against what a run holds.
2. `DO_ACTION_GATEWAY_ACTOR_ID`, asked for when unset.
3. `DO_MICROVM_IMAGE`: kept when set; otherwise built with scripts/build_sandbox_image.py into your DigitalOcean
   Container Registry. The registry token is passed to the build in its environment and is written to `.env` only if
   you say so.
4. `DO_INCIDENT_REPO`: a public GitHub repository of your own (through the `gh` CLI and its login), filled with
   examples/incident_common/service/make_demo_repo.sh. An existing repository is overwritten only after you type its
   name; `--refresh-repo` rebuilds it (the change analyst reports the commit's age, so rebuild it shortly before a run).
5. The droplet: examples/incident_common/setup_droplet.py, which shows the Action Gateway connect links (GitHub and
   DigitalOcean) for the actor the first time; then examples/09_incident_response/prepare.py --clean and
   scripts/check_setup.py --examples 09.

What is already set is checked, not only looked up: the image's manifest (crane; with DO_REGISTRY_TOKEN for a
DigitalOcean registry), the repository (public, the demo's two commits and service/app.py), the droplet's service
(`/health`). A full run treats a failed check as missing and offers to redo that step.

`--dry-run` runs only the read-only checks, creates nothing (no MicroVM, endpoint, droplet or repository) and exits 0
when everything it can check passes; it lists what only the full run checks (the actor, the template and a gateway
endpoint with the example's tools, through prepare.py). Nothing secret is printed or passed on a command line.
The full run requires a terminal (the values are typed in); step 0 lists the commands the other steps need.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
from _env import DEFAULT_ENV_FILE, add_env_file_argument
from check_setup import ACTION_GATEWAY_URL, _http_probe, _model_probe
from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.digitalocean.com/v2"
TOKENS_PAGE = "https://cloud.digitalocean.com/account/api/tokens"
MODEL_KEYS_PAGE = "https://cloud.digitalocean.com/model-studio/manage-keys"
REGISTRY_PAGE = "https://cloud.digitalocean.com/registry"
MODELS = ("anthropic-claude-haiku-4.5", "anthropic-claude-4.6-sonnet")
DEMO_REPO_SCRIPT = ROOT / "examples" / "incident_common" / "service" / "make_demo_repo.sh"
DROPLET_STATE = ROOT / "examples" / "incident_common" / "outputs" / "incident.json"
DROPLET_ENV = DROPLET_STATE.with_name("incident.env")
REPO_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._][A-Za-z0-9._-]*")  # never an option to gh or git


@dataclass(frozen=True)
class Secret:
    """A value created in the Control Panel: where, with which scopes, and how to check it."""

    key: str
    what: str
    page: str
    scopes: str
    check: Callable[[str, Mapping[str, str]], str | None]  # None when the value works, else why not


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
    # An empty create request creates nothing: a token with microvm:create gets 400 (the body), one without it 403.
    try:
        status = httpx.post(f"{API}/microvms", headers={"Authorization": f"Bearer {token}"}, json={}, timeout=15.0).status_code
    except httpx.HTTPError:
        return "could not reach the API"
    if status in (400, 422):
        return "this token can create MicroVMs; create one with microvm:access only"
    if status == 401:
        return "401: the token was refused"
    return None  # 403: no create scope; its use on an endpoint is checked by prepare.py


def check_model_key(key: str, env: Mapping[str, str]) -> str | None:
    for model in (env.get("DO_INFERENCE_MODEL") or MODELS[0], env.get("DO_INFERENCE_STRONG_MODEL") or MODELS[1]):
        if problem := _model_probe(key, model):
            return f"{model}: {problem}"
    return None


def check_gateway(token: str, env: Mapping[str, str]) -> str | None:
    return _http_probe(ACTION_GATEWAY_URL, token, params={"per_page": "1"})


SECRETS = (
    Secret("DO_MICROVM_API_TOKEN", "MicroVMs control token", TOKENS_PAGE,
           "Custom Scopes: microvm:create, microvm:read, microvm:update, microvm:delete", check_control),
    Secret("DO_MICROVM_ACCESS_TOKEN", "MicroVM endpoint access token", TOKENS_PAGE,
           "Custom Scopes: microvm:access only", check_access),
    Secret("DO_MODEL_ACCESS_KEY", "Serverless Inference model access key", MODEL_KEYS_PAGE,
           f"Foundation models: {' and '.join(MODELS)} (or All models)", check_model_key),
    Secret("DO_ACTION_GATEWAY_TOKEN", "Action Gateway token", TOKENS_PAGE,
           "Custom Scopes: every action_gateway scope", check_gateway),
)
BROADER_CHECK = ("DO_MICROVM_API_TOKEN", "DO_MICROVM_ACCESS_TOKEN", "DO_ACTION_GATEWAY_TOKEN")
BROADER_WARNING = ("accepted, and it can also read droplets or the account: it has more scopes than this setup needs "
                   "(Full Access, Read Only or extra custom scopes). A token with only the scopes named limits what a "
                   "leaked .env can do")
# Reads none of these tokens needs; a 200 on either means more scopes than the setup asks for (Full Access included).
UNNEEDED_READS = (f"{API}/droplets", f"{API}/account")
MICROVMS_NEEDED = 5  # what a run holds at once, the template's builder included (scripts/check_setup.py)

RECOMMENDATIONS = f"""Before you start
- Use a DigitalOcean team of its own for examples, or set DO_MICROVM_NAMESPACE (1-16 lowercase letters or digits),
  so these MicroVMs, the droplet and the clean-up scripts stay apart from other work on the team.
- Give each token a short expiration; a run takes minutes. Create each one with Custom Scopes, exactly the scopes
  named; a token with more (Full Access, Read Only or extra scopes) is reported, and you choose whether to keep it.
- Set a spend threshold on the model access key (a spend alert on Managed Access Keys (Serverless Inference)).
- The team needs room for {MICROVMS_NEEDED} MicroVMs at once (paused ones count); the limit is checked below.
- The droplet and its snapshot are billed until `uv run examples/incident_common/setup_droplet.py --cleanup`;
  `uv run scripts/delete_all.py` lists and deletes MicroVMs a stopped run left.
- The actor's GitHub and DigitalOcean connections stay after the clean-up; delete them when you no longer need
  them (Action Gateway documentation: How to Manage Connections)."""

# (command, why, where to get it, needed when)
TOOLS: tuple[tuple[str, str, str, str], ...] = (
    ("git", "builds and pushes the demo repository", "https://git-scm.com/downloads", "repo"),
    ("bash", "runs make_demo_repo.sh", "preinstalled on macOS and Linux", "repo"),
    ("gh", "creates the public demo repository with your GitHub login", "https://cli.github.com", "repo"),
    ("ssh", "installs the service on the droplet", "OpenSSH, preinstalled on macOS and most Linux", "droplet"),
    ("ssh-keygen", "creates the droplet's temporary key pair", "OpenSSH", "droplet"),
    ("crane", "pushes the sandbox image (only when DO_MICROVM_IMAGE is unset)",
     "https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md", "image"),
)


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
        return None, "not checked: crane is not installed (prepare.py boots it)"
    registry = image.split("/", 1)[0]
    private = registry == "registry.digitalocean.com"
    if private and not registry_token:
        return None, "not checked: set DO_REGISTRY_TOKEN (registry:read) to check it here (prepare.py boots it)"
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
    def __init__(self, env_file: Path, *, dry_run: bool) -> None:
        self.env_file = env_file
        self.dry_run = dry_run
        self.env = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.exists() else {}
        self.todo: list[str] = []
        self.unchecked: list[str] = []  # set, but only the full run can check it

    # ── output and input ──────────────────────────────────────────────────────
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
        print("\n0. Prerequisites")
        needed = {"repo": repo, "droplet": droplet, "image": not self.env.get("DO_MICROVM_IMAGE")}
        ok = True
        for command, why, where, when in TOOLS:
            if not needed[when]:
                continue
            if shutil.which(command):
                self.say("ok", command)
            else:
                self.say("FAIL", command, f"not on PATH; {why}. Install: {where}")
                ok = False
        if repo and shutil.which("gh"):
            if _run(["gh", "auth", "status"]) is None:
                self.say("FAIL", "gh login", "run `gh auth login` (scope: repo)")
                ok = False
            else:
                self.say("ok", "gh login")
        if not ok:
            self.todo.append("install the tools marked FAIL under Prerequisites")
        return ok

    def team_room(self) -> None:
        """The team's MicroVM limit against what a run holds, as a recommendation; it changes nothing."""
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
        self.say("ok" if free >= MICROVMS_NEEDED else "warn", "team limit",
                 f"{team.limit}, {team.live} MicroVM(s) on the team, {max(free, 0)} free; a run needs {MICROVMS_NEEDED}"
                 + ("" if free >= MICROVMS_NEEDED else ": delete some, or have the team's limit raised"))

    def tokens(self) -> bool:
        print("\n1. Tokens")
        ok = True
        for secret in SECRETS:
            value = self.env.get(secret.key, "")
            problem = secret.check(value, self.env) if value else "not set"
            if problem is None:
                if not self.broader(secret, value):
                    self.say("ok", secret.key, "accepted")
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
                    ok = False
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
        print("\n2. Action Gateway actor")
        if actor := self.env.get("DO_ACTION_GATEWAY_ACTOR_ID"):
            self.verdict("DO_ACTION_GATEWAY_ACTOR_ID", (None, f"{actor}; not checked here (setup_droplet.py and prepare.py "
                                                              "call the gateway as this actor)"), "")
            return True
        if self.dry_run:
            self.todo.append("set DO_ACTION_GATEWAY_ACTOR_ID (the part of your DigitalOcean email before the @)")
            return False
        actor = self.ask("       DO_ACTION_GATEWAY_ACTOR_ID (the part of your DigitalOcean email before the @)")
        if not actor:
            return False
        self.save("DO_ACTION_GATEWAY_ACTOR_ID", actor)
        self.say("ok", "DO_ACTION_GATEWAY_ACTOR_ID", actor)
        return True

    def image(self) -> bool:
        print("\n3. Sandbox image")
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
        token = self.env.get("DO_REGISTRY_TOKEN") or getpass.getpass(
            f"       Create a token at {TOKENS_PAGE} with Custom Scopes registry:read, registry:update\n"
            "       paste it (input hidden): ").strip()
        if not token:
            return False
        repository = self.env.get("DO_MICROVM_SANDBOX_REPOSITORY") or self._repository(token)
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
        if "DO_REGISTRY_TOKEN" not in self.env and self.confirm(f"       save the registry token in {self.env_file.name} for rebuilds?"):
            self.save("DO_REGISTRY_TOKEN", token)
        self.say("ok", "DO_MICROVM_IMAGE", self.env["DO_MICROVM_IMAGE"])
        return True

    def _repository(self, token: str) -> str | None:
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
        return f"registry.digitalocean.com/{response.json()['registry']['name']}/microvm-sandbox"

    def repo(self, *, refresh: bool) -> bool:
        print("\n4. Demo repository")
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
        print("\n5. Droplet, connections, pre-flight")
        if self.env.get("DO_MICROVM_TRANSPORT", "").lower() == "local":
            self.say("FAIL", "DO_MICROVM_TRANSPORT", f"is `local` in {self.env_file.name}; remove it for a real run")
            self.todo.append(f"remove DO_MICROVM_TRANSPORT=local from {self.env_file.name}")
            return False
        if DROPLET_STATE.exists():
            if not self.verdict("droplet incident-svc", verify_droplet(),
                                "recreate the droplet: setup_droplet.py --cleanup, then this script"):
                return False
            repo = dotenv_values(DROPLET_ENV).get("DO_INCIDENT_REPO")
            if repo and repo != self.env.get("DO_INCIDENT_REPO"):
                self.say("warn", "droplet", f"was set up from {repo}, not DO_INCIDENT_REPO; recreate it (--cleanup)")
            if self.dry_run:
                self.unchecked.append("the template and a gateway endpoint with the example's tools (prepare.py, full run)")
                return True
        elif self.dry_run:
            self.say("need", "droplet", "not created")
            self.todo.append("create the droplet (setup_droplet.py; this script runs it, with the connect links)")
            return False
        else:
            print("       creates droplet incident-svc, ssh key incident-key and snapshot incident-good (billed until "
                  "setup_droplet.py --cleanup)")
            if not self.confirm("       create them now?") or not _step("examples/incident_common/setup_droplet.py"):
                return False
        return _step("examples/09_incident_response/prepare.py", "--clean") and _step("scripts/check_setup.py", "--examples", "09")


def _run(command: list[str]) -> str | None:
    """stdout of a command that prints no secret, or None when it failed."""
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def _step(script: str, *args: str) -> bool:
    print(f"\n$ uv run python {script} {' '.join(args)}".rstrip(), flush=True)
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=False).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="check and report each step; change nothing")
    parser.add_argument("--skip-repo", action="store_true", help="leave DO_INCIDENT_REPO as it is")
    parser.add_argument("--skip-droplet", action="store_true", help="stop before the droplet and the pre-flight")
    parser.add_argument("--refresh-repo", action="store_true", help="rebuild the demo repository's two commits")
    add_env_file_argument(parser)
    args = parser.parse_args(argv)
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("needs a terminal (tokens are typed in); --dry-run works without one")
    shadowed = [k for k in os.environ if k.startswith(("DO_MICROVM_", "DO_ACTION_GATEWAY_", "DO_MODEL_", "DO_INCIDENT_"))]
    if shadowed:
        print(f"note: set in the shell (the examples use these over {args.env_file.name}):{', '.join(sorted(shadowed))}")

    setup = Setup(args.env_file if args.env_file else DEFAULT_ENV_FILE, dry_run=args.dry_run)
    print(RECOMMENDATIONS)
    if not setup.prerequisites(repo=not args.skip_repo, droplet=not args.skip_droplet) and not args.dry_run:
        print("\nInstall what is marked FAIL, then run this again (nothing was changed).")
        return 1
    ready = setup.tokens() & setup.actor() & setup.image()
    if not args.skip_repo:
        ready &= setup.repo(refresh=args.refresh_repo)
    if ready and not args.skip_droplet:
        ready &= setup.droplet()
    if setup.todo:
        print("\nTo do:\n" + "\n".join(f"- {item}" for item in setup.todo))
    if setup.unchecked:
        print("\nNot checked by this run (marked --):\n" + "\n".join(f"- {item}" for item in setup.unchecked))
    if args.dry_run:
        print("\n" + ("Ready: every read-only check passed. Run without --dry-run for the pre-flight (prepare.py)."
                      if ready else "Not ready: see To do. Run without --dry-run to set it up."))
    elif ready and not args.skip_droplet:
        print("\nReady. Start the incident and run it (examples/09_incident_response/README.md, Run (real mode)):\n"
              "  uv run examples/incident_common/setup_droplet.py --deploy-bad\n"
              "  uv run examples/09_incident_response/dashboard.py          # terminal 2\n"
              "  uv run examples/09_incident_response/run_script.py --delete\n"
              "  uv run examples/incident_common/setup_droplet.py --cleanup  # afterwards")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
