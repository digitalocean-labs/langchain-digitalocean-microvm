# Package reference

Module-by-module reference for `langchain_digitalocean_microvm`. Design and decisions: [`design.md`](design.md).

## Architecture

**`status.py` / `models.py` / `exceptions.py` / `config.py`** -- shared types.
`MicroVMStatus` uses the API's states (`creating`, `running`, `pausing`,
`paused`, `resuming`, `terminating`, `terminated`, `failed`, `unknown`);
`parse_status()` degrades anything else to `UNKNOWN`. `MicroVMState` is a
frozen snapshot (size, endpoint hostname, tags, `raw` payload);
`MicroVMSize` validates the four cpu/memory pairs the API accepts.

**`client.py`** -- `MicroVMClient` for `/v2/microvms`: create (from an image
or a checkpoint, with environment, tags, idle timeout), get/list (paginated,
exact `name` and `tag_name` filters), pause/resume (re-read afterwards: their
response drops `tags`), delete, exec, checkpoints, options. Retries 429 and
502-504 with backoff. MicroVMs are found by a deterministic name
(`microvm_name()`: `<prefix>-<id>`, hashed when the id is not a valid name).

**`backend.py`** -- `MicroVMSandbox(BaseSandbox)`, the deepagents adapter:
`execute`/`upload_files`/`download_files` through a transport; everything
else (`ls`/`read`/`write`/`edit`/`grep`/`glob`) comes from `BaseSandbox`'s
generated shell/python3 scripts.

**`transport/`** -- `exec.py` (`ExecTransport`, `DO_MICROVM_TRANSPORT=exec`) runs commands through the
exec API, within its limits per command and per team
([Limits](https://docs.digitalocean.com/products/microvms/details/limits/#command-and-console-limits)). Each command runs under a small shell wrapper
that waits up to 50 s: short commands cost one API call; longer ones keep
running detached and are polled, and a timed-out command's whole process
group is killed (exit 124). Files move as base64 in chunks, with payloads in
separate argv elements because Linux caps one argument at 128 KiB.
`endpoint.py` (`EndpointTransport`, the default, `DO_MICROVM_TRANSPORT=endpoint`) talks
to a small server inside the MicroVM (`sandbox_server.py`, standard library
only) through the MicroVM's HTTPS endpoint instead, so commands and files do
not go through the rate-limited control plane. Requests carry the access-only
token for the edge and a per-sandbox token in `x-api-key`, derived from the
control token and the MicroVM id (a checkpoint clone gets its own); the
server reads it from a file that the transport writes over exec when the
server rejects it. Every request is kept short: a command waits a few seconds
in the request, then continues as a job that is polled, and output larger
than 256 KiB is read in pages like a file; each command carries an
idempotency key, so retrying a request never starts the command twice. Files
go up in 32 MiB chunks (SHA-256 checked) and come down in pages of up to
16 MiB, made smaller when a slow connection needs it. The sandbox image runs
the server as its entrypoint; on another image with `python3` and a free HTTP
port it is installed over exec on first connect, and installed again when its
protocol version no longer matches the package's (a server that is the
image's own entrypoint is reported instead: rebuild the image). Where no
compatible server can run, the transport falls back to exec with a warning;
a missing or rejected access token is an error.
`local.py` runs commands locally against a per-sandbox directory, rewriting
absolute paths into it (see its docstring for the one known gap). The
directory lasts until `MicroVMLifecycle.delete` removes it, so a process that
reconnects with the same sandbox id finds its files again.

**`subagents.py`** -- one sandbox per subagent (below).
**`threads.py`** -- one sandbox per conversation thread (below).
**`integrations/deepagents_code.py`** -- the dcode sandbox provider (below).

**`provider.py`** -- `MicroVMLifecycle.get_or_create()` reconciles whatever it
finds (`reconcile.settle_to_running`): reuse `running`, resume `paused`, wait
out `creating`/`pausing`/`resuming`, delete and recreate `failed`, retry once
on `unknown`. A `terminating` leftover with the same name is ignored. The
local branch keeps an in-memory status per sandbox instead of calling the API.

## Sandbox templates (warm start)

`MicroVMLifecycle.build_template(name, setup=[...], seed_files=[...], rebuild=False)` boots a
temporary MicroVM from `DO_MICROVM_IMAGE` (tags `microvm-sandbox`, `microvm-template`), uploads
the seed files, runs each setup command through `execute` (detached and polled, so long
`pip install`s are fine; `setup_timeout`, default 900 s, per command; a non-zero exit raises),
checkpoints it as `microvm-template-<name>-<hash>` and deletes the temporary MicroVM. The hash
covers the name, image, size, port, setup commands, seed files (path and content) and outbound policy, so the
same inputs reuse an AVAILABLE checkpoint and any change builds a new one. It returns a
`SandboxTemplate` (`name`, `checkpoint_name`, `checkpoint_id`, `built`).
`get_or_create(sandbox_id, template=t)` creates a missing sandbox from the checkpoint (it
inherits image, size, region, port and environment; `DO_MICROVM_IMAGE` is not needed) and
reuses an existing one unchanged. `find_template(name, setup=..., seed_files=...)` returns the template
those inputs would reuse, or `None`, without building one; `delete_template(t)` deletes the checkpoint
(and a kept temporary MicroVM, see [Outbound policy](#outbound-policy-egress-rules-and-credentials));
`scripts/delete_all.py --yes --include-templates --all-owners` deletes every `microvm-template-*` checkpoint
(without `--include-templates` it keeps them; templates are shared by every user and namespace, so it needs
`--all-owners`, and `--all-namespaces` once a namespace is set or other namespaces have MicroVMs on the team). Local mode stores a
template as a directory (copied into new sandboxes) and skips `setup`. Example:
`examples/06_warm_start/`.

A server that serves many threads builds its template once and keeps it:
`TemplateCache(lambda: lifecycle.build_template(...))` (`template_cache.py`) builds it on first
use (`get()`), and `create(lambda t: lifecycle.get_or_create(sandbox_id=..., template=t))`
retries a clone once if its checkpoint was deleted meanwhile (the API answers a `404`,
`MicroVMNotFoundError`): `refresh(stale)` rebuilds it, once even when several threads fail at the
same moment, and `build_template` reuses the checkpoint if it still exists. Other errors are
raised at once (`retry_on` widens the set). Examples 07 and 08 use it.

## Outbound policy (egress rules and credentials)

A MicroVM's outbound policy limits the destinations it can connect to and can hold credentials the
platform adds to its requests ([How to Restrict a MicroVM's Outbound
Traffic](https://docs.digitalocean.com/products/microvms/how-to/restrict-outbound-traffic/)). The
policy is set at create and cannot change; a MicroVM created from a checkpoint inherits it.
`EgressPolicy` (`egress.py`) describes one:

```python
from langchain_digitalocean_microvm import EgressPolicy

EgressPolicy.allow("pypi.org", "files.pythonhosted.org", "*.example.org", "203.0.113.5")  # everything else denied
EgressPolicy.deny_all()
EgressPolicy.allow_all()  # traffic still goes through the platform proxy, so credentials apply
policy = EgressPolicy.allow("pypi.org").with_credential("https://api.example.com/v1", secret=api_key)
```

- **Rules**: up to 50, each an exact hostname, a `*.` wildcard (which does not match the domain
  itself) or an IPv4 address; no scheme, port, path or CIDR range. `default_action` (`deny` for
  `allow()` and `deny_all()`) applies to every destination no rule matches.
- **Credentials**: up to 16; the package accepts one per host. `with_credential(url, secret=..., scheme="bearer")`
  binds a secret to the URL's exact host and path prefix and adds the host to the rules. The platform
  sets the `Authorization` header of HTTPS requests to that host and prefix to `Bearer <secret>`
  (`scheme="raw"`: the secret itself). Code in the MicroVM should not set that header for the host (MicroVMs
  documentation). The secret is stored outside the
  MicroVM and the API never returns it. A response that echoes request headers contains the secret, so
  bind a credential to the narrowest prefix that works. `repr()` never shows a secret.
- **Where it applies**: `MicroVMConfig.egress` is the default for every sandbox and template created
  from the image; `get_or_create(..., egress=...)` and `build_template(..., egress=...)` override it. A
  sandbox created from a template has the template's policy, so `get_or_create` refuses `egress`
  together with `template`. An existing sandbox keeps the policy it was created with. The
  deepagents-code provider takes `egress` as an `EgressPolicy` or a `DO_MICROVM_EGRESS` string.
- **Environment**: `DO_MICROVM_EGRESS` is `none` (deny every destination), `*` (allow every one) or a
  comma-separated list of destinations. `DO_MICROVM_EGRESS_CREDENTIALS` is a comma-separated list of
  `https-url=VARIABLE` items; each secret is read from the variable named after `=`. Credentials without
  `DO_MICROVM_EGRESS` allow every destination.
- **Templates**: `build_template` creates its temporary MicroVM with the policy, so `setup` commands run
  under it: allow the hosts they download from (pip: `pypi.org` and `files.pythonhosted.org`). The policy
  is part of the template's hash (secrets through an HMAC keyed with the API token), so a changed secret
  builds a new template. The service deletes a MicroVM's stored credentials with the MicroVM, and a
  checkpoint whose credentials are gone cannot be restored; a template whose policy has credentials
  therefore keeps its temporary MicroVM, paused (`microvm-template-build-*`; it counts against the team
  limit). `delete_template` deletes both, and `scripts/delete_all.py` keeps both unless
  `--include-templates`. For a checkpoint with credentials, `find_template` and `build_template` check
  that the MicroVM it came from exists; when it does not, `find_template` returns `None` and
  `build_template` deletes the checkpoint and builds a new one. A clone of it fails
  with `CheckpointUnavailableError` (a `MicroVMNotFoundError`, so `TemplateCache` rebuilds the template).
- **Inside the MicroVM**: outbound traffic goes through a platform proxy set in `HTTP_PROXY` and
  `HTTPS_PROXY`; only HTTP and HTTPS from clients that use those variables reach a destination (no SSH,
  database or other TCP and UDP connections). For a host with a credential the platform presents a
  certificate from its own certificate authority, which `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`,
  `CURL_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS` point to; clients that pin certificates fail for that host.
- **Images in DigitalOcean Container Registry**: for a `registry.digitalocean.com/...` image the package
  adds `api.digitalocean.com` to a policy whose default action is `deny` (`EgressPolicy.for_image`). Code
  in the MicroVM can then reach the DigitalOcean API, with whatever token it has; the package puts no
  DigitalOcean token into a sandbox.
- **Service conditions**: public networking only; credentials only in the regions the MicroVMs
  documentation lists and only on a create from an image.
- **Local mode** applies no policy and logs a warning.

## Thread sandboxes (one MicroVM per conversation thread)

`ThreadSandboxes` (`threads.py`) gives each thread of a server its own sandbox, for a
per-thread graph factory (`examples/07_sales_assistant/agent.py`):

```python
from langchain_digitalocean_microvm import MicroVMLifecycle, TemplateCache, ThreadSandboxes
from langchain_digitalocean_microvm.agent_host.store import GatewayStore

lifecycle = MicroVMLifecycle()
templates = TemplateCache(lambda: lifecycle.build_template("assistant", setup=[...], seed_files=[...]))
threads = ThreadSandboxes(lifecycle, prefix="assistant", template=templates,
                          store=GatewayStore(default_db_path("microvm")))

@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    if not runtime.execution_runtime:            # introspection: no MicroVM
        yield create_deep_agent(..., backend=StateBackend())
        return
    async with threads.asandbox(config["configurable"]["thread_id"]) as backend:
        yield create_deep_agent(..., backend=backend)
```

- `asandbox(thread_id)` / `sandbox(thread_id)` (sync, for scripts) get or create
  `<prefix>-<thread id>` (`sandbox_id(thread_id)`): a new thread's sandbox is cloned from
  `template` (a `SandboxTemplate`, or a `TemplateCache` that is rebuilt once if its checkpoint
  was deleted), a paused one is resumed with its files. On exit the sandbox is paused, also
  after a failed run; a failed pause is logged, not raised (the MicroVM pauses itself when
  idle). The async variant runs the blocking calls off the event loop. An empty thread id
  raises `ValueError`.
- `max_concurrent_creates` (default 4) bounds the get-or-create calls (creates and resumes)
  one instance runs at once; a burst of new threads waits in the process. Create one instance
  per process. `MicroVMConfig.max_concurrent_creates` (below) bounds all callers together.
- Sandboxes carry `microvm-thread-sandbox`. With `store`, each run claims the sandbox's row
  before get-or-create and records `active`, then `paused` (`sandboxes` table, `subagent`
  empty), so `scripts/gc_microvms.py --sandbox-ttl` deletes threads unused that long, and a
  sandbox a sweep is deleting is replaced by a fresh one, not resumed. Without a store the
  sweeper deletes them once older than `--orphan-ttl`.
- Paused MicroVMs keep their files and count against the team's MicroVM limit; delete a
  finished thread's with `lifecycle.delete(sandbox_id=threads.sandbox_id(thread_id))`.

## Agent hosts (subagents as separate processes)

`langchain_digitalocean_microvm.agent_host` is the "agent in sandbox" counterpart of
`MicroVMSandbox`: instead of the local agent shipping shell commands
into a MicroVM, a whole subagent (in-memory LangGraph Agent Server + graph +
filesystem) runs *inside* the MicroVM and the supervisor reaches it over the
Agent Protocol. Design and decisions: [`design.md`](design.md).

The host manager:

```python
from langchain_digitalocean_microvm import EgressPolicy, HostSpec, MicroVMAgentHost

hosts = MicroVMAgentHost()  # reads DO_MICROVM_* incl. DO_MICROVM_AGENT_IMAGE
handle = hosts.get_or_create(
    host_id="task-123",
    spec=HostSpec(
        graph_id="researcher",
        env={"DO_MODEL_ACCESS_KEY": "set-by-the-platform"},  # the model client needs a value
        egress=EgressPolicy.allow_all().with_credential("https://inference.do-ai.run/v1", secret=key),
    ),
)
handle.url, handle.token   # https://<id>.mkc1.microvm.ondigitalocean.com and the host's x-api-key
hosts.edge_headers()       # {"Authorization": "Bearer <DO_MICROVM_ACCESS_TOKEN>"} for DigitalOcean's edge
hosts.pause(host_id="task-123")
handle = hosts.resume(host_id="task-123")   # re-probes GET /ok
hosts.delete(host_id="task-123")
```

- `get_or_create` creates the MicroVM from the agent image with the host's
  whole environment -- including a fresh random `SUBAGENT_HOST_TOKEN` -- in the
  create call, settles it to `running`, waits for the HTTPS endpoint to be
  `ACTIVE`, then probes `GET /ok` with both credentials. MicroVM environments
  are immutable, so a host whose token this process does not know cannot be
  re-keyed: pass a stored token back with `remember_token()`, otherwise
  `get_or_create` replaces the host and `resume` refuses.
- `HostSpec.egress` is the host's outbound policy (default: `MicroVMConfig.agent_egress`, from
  `DO_MICROVM_AGENT_EGRESS` and `DO_MICROVM_AGENT_EGRESS_CREDENTIALS`, same forms as the sandbox pair;
  see [Outbound policy](#outbound-policy-egress-rules-and-credentials)). With a credential for the
  inference endpoint the model key is not in the host: the platform sets the `Authorization` header of
  the host's requests to that URL prefix. Under an
  allowlist, allow every other host the graph calls (for tracing, `api.smith.langchain.com`); the
  gateway's completion webhook is reached only when `webhook_base_url` is set. Warm start keeps the
  golden host of a policy with credentials, paused (see design.md §4.9). A local host applies no
  policy and logs a warning.
- `resume` never recreates a `failed` host: its in-VM state is the point.
- `MicroVMLifecycle.pause(sandbox_id=...)` exists alongside
  `pause(backend)`, so callers can checkpoint a MicroVM they never connected to.
- Local mode: `LocalDevServerHost` has the same surface but spawns one
  local `langgraph dev` per host from `images/agent-host/` (own working
  directory, generated `langgraph.json`, `./.env`; pause = `SIGSTOP`, resume =
  `SIGCONT`). Each server runs under `agent_host/_guard.py`, a stdlib wrapper
  that terminates it when the spawning process disappears, because a
  `langgraph dev` reload worker dies by a re-raised SIGTERM and never runs
  `atexit`. Needs `uv sync --extra agent-host`. `uv run pytest -m e2e
  tests/unit_tests/test_local_host_e2e.py` drives a real server through the Agent Protocol.
- Local mode template: the first match of `DO_MICROVM_SUBAGENT_TEMPLATE` (or
  `template_dir=`), then `images/agent-host/` of a source checkout when it has a
  `langgraph.json`, then the copy the wheel ships in
  `langchain_digitalocean_microvm/agent_host/_template/` (`agent.py`, `auth.py`,
  `artifacts_app.py`, `langgraph.json`), so `pip install
  "langchain-digitalocean-microvm[agent-host]"` is enough to run the `echo` graph.
  The server runs in the caller's environment, so the `researcher` graph also needs
  `langchain-anthropic` there. `uv run pytest -m e2e
  tests/unit_tests/test_wheel_template_e2e.py` builds the wheel, installs it into a
  fresh virtualenv and starts a host from it.
- Gateway: `SubAgentGateway` gives a supervisor ONE stable Agent
  Protocol URL in front of many hosts. It allocates a host per task on the
  first run (reusing the supervisor's thread id), forwards the calls the
  deepagents middleware makes (including listing a task's latest run), caches finished runs so `check` never wakes a
  paused host, pauses idle hosts after a grace period, and turns a dead or
  restarted host into a `status: error` run instead of a hang.

  ```python
  from langchain_digitalocean_microvm.agent_host import (
      HostSpec, LocalDevServerHost, SubAgentGateway, async_subagent_spec,
  )
  from deepagents import create_deep_agent

  hosts = LocalDevServerHost()                       # or MicroVMAgentHost()
  gateway = SubAgentGateway(
      hosts, {"researcher": HostSpec(graph_id="researcher")},
      db_path="<state dir>/gateway.sqlite", token="<gateway token>",
  )
  url = gateway.start_embedded()                     # uvicorn on a loopback port, in-process
  agent = create_deep_agent(
      model=model,
      subagents=[async_subagent_spec(
          name="researcher", description="Deep research in its own VM",
          graph_id="researcher", gateway_url=url, gateway_token="<gateway token>",
      )],
  )
  ```

  Set `LANGGRAPH_API_KEY=<gateway token>` in the supervisor's environment so the
  SDK's mandatory `x-api-key` header carries the gateway token rather than your
  LangSmith key (design doc §4.8). The gateway polls hosts for run status.
  Webhooks are off by default and only useful when the hosts can reach the
  gateway (a standalone https deployment, or local mode with
  `webhook_policy_env()`, since the Agent Server rejects plain-http and loopback
  webhook URLs by default); real MicroVMs cannot reach an embedded gateway.
- Supervisor wiring: `LazyGateway` bundles the host manager and the gateway
  for a supervisor process and starts them on the agent's first turn rather than
  at import (so `langgraph dev` reloads stay cheap), reserving the loopback port
  up front so specs can be built eagerly. `GatewaySettings.from_env()` reads
  `DO_MICROVM_HOST_MODE` (`local` default / `microvm`) and
  `DO_MICROVM_GATEWAY_TOKEN` (random when unset); `close()` (also registered with `atexit`) kills
  local hosts and records them as destroyed, and only stops the gateway in
  microvm mode.

  ```python
  from langchain_digitalocean_microvm.agent_host import HostSpec, LazyGateway

  gateway = LazyGateway({"echo": HostSpec(graph_id="echo")}, poll_interval=2.0)
  agent = create_deep_agent(
      model=model,
      subagents=[gateway.subagent(name="echo", description="Replies 'echo: <message>'")],
      middleware=[gateway.middleware()],   # starts the gateway before the first step
  )
  ```

  The example that uses it is `examples/04_async_subagents/` (see its README for
  the walkthrough); `tests/unit_tests/test_supervisor.py` drives deepagents' real
  `start/check/update/cancel/list_async_task` tools through the gateway with a
  scripted chat model, and `uv run pytest -m e2e tests/unit_tests/test_supervisor_e2e.py`
  does the same against real `langgraph dev` hosts.
- Real MicroVMs: client, exec transport, sandbox lifecycle and host
  manager are on the real API. The gateway sends `hosts.edge_headers()` (the access
  token) with every upstream call, restores stored host tokens on start, refuses a
  `poll_interval` above a third of the hosts' idle timeout, and at the team's MicroVM
  limit evicts an idle host whose results are cached (an evicted task can still be
  checked, not updated). The host image is built by `images/agent-host/build_image.py`.
  A host answering 401/403 counts as unreachable, and so does a 404 for a run once the
  inventory no longer lists the host (a deleted MicroVM's endpoint answers 404 rather than
  refusing connections; a host that is still listed gets a plain "run not found"). On
  start the gateway marks hosts that vanished since the last session as destroyed. The example runs in `microvm` mode with one
  shared host per subagent type; `scripts/delete_all.py` removes everything this
  package created on the team (templates only with `--include-templates`).
- Warm start (`DO_MICROVM_WARM_START=1`): new hosts are clones of a golden checkpoint
  per host configuration, built on first use (a booted, probed host, checkpointed then
  deleted). Each clone's token is written to `/work/.microvm-agent/token` via exec once
  the clone answers `/ok`, and the template's `auth.py` prefers it over the inherited env
  token. A serving host starts in a clone's time instead of a cold boot; the first host pays for building the checkpoint once.
- Artifacts: each task works in its own directory inside its host,
  `<SUBAGENT_WORKDIR>/tasks/<task id>/` (the template keys it by the run's thread id,
  which is the supervisor's task id), so tasks sharing a pooled host never see each
  other's files. The gateway lists and returns a task's `outputs/` with
  `GET /artifacts/<task id>` and `GET /artifacts/<task id>/files/<path>` (max 200 files,
  10 MiB per file by default), reading them from routes the host's own Agent Server
  serves next to the Agent Protocol (`images/agent-host/artifacts_app.py`, same host
  token, never follows symlinks) in `Range` pages of 4 MiB (halved when a page does not
  arrive within the endpoint's time limit). A paused host is resumed for it and paused
  again by `tick()`; a host that was deleted, evicted, or replaced since the task ran
  answers 410 (the files only ever lived there); a host built from an image without the
  routes answers 502 (rebuild the image). The supervisor side is one tool:

  ```python
  agent = create_deep_agent(
      model=model,
      subagents=[gateway.subagent(name="researcher", description="...")],
      middleware=[gateway.middleware()],
      tools=[gateway.artifacts_tool()],    # fetch_subagent_artifacts(task_id, paths=None)
  )
  ```

  It copies the files into the supervisor's backend (default `StateBackend`, i.e. its
  state) at `/subagents/<task id>/<path>`, where `read_file` sees them.
  `fetch_artifacts_tool(gateway_url=..., gateway_token=..., backend=...)` builds the
  same tool for a gateway you run yourself.

  Several gateways in one agent: the start middleware is named `microvm-gateway-start`
  (`gateway.middleware(name=...)` for a second one of this package), and every artifacts tool
  defaults to `fetch_subagent_artifacts`, so give all but one their own name
  (`gateway.artifacts_tool(name=...)`): an agent keeps one tool per name.
- Lifecycle (`agent_host/lifecycle.py`, the gateway's base class `HostLifecycle`): hosts
  do not outlive their usefulness. The gateway's `tick()` (its poll loop) deletes a FAILED host on the next tick, and a READY/PAUSED host with no
  active run once it has been unused for its TTL: `release_after` for per-task hosts
  (default 1 h), `paused_ttl` for every host (default 24 h; `None` disables either). Any
  use restarts the clock (run, update, artifact fetch). Because a task's files live only
  in its host, the TTL is also the artifact retention window; an idle host is also safe
  from eviction at the MicroVM limit for `eviction_grace` (default 60 s), during which a
  launch that needs the slot fails with "try again in about N s". A deleted host stays in
  the store as `destroyed` with a `retired_reason`, so `check` is answered from the cache,
  an update is refused and an artifact fetch answers 410 saying why:

  ```python
  LazyGateway(graphs, release_after=1800.0, paused_ttl=6 * 3600.0, eviction_grace=120.0)
  ```

  For what outlives the supervisor, `scripts/gc_microvms.py` (dry run unless `--yes`;
  `--ttl`, `--orphan-ttl`, `--checkpoint-ttl`; one namespace, see "Namespaces" below) deletes agent hosts the store marks
  destroyed/failed or that are unused past the TTL with no active run, hosts no store row
  describes, interrupted golden builds and failed or old golden checkpoints (rebuilt on
  next use), and marks store rows whose MicroVM is gone. It can run from cron, next to a
  running gateway: each deletion is re-checked in the store and the row marked `retiring` in
  one transaction, while the gateway claims a row before it resumes a host, so a host used
  since the plan is kept ("touched since the plan") and one being deleted is replaced by a
  fresh host instead of resumed.
  `scripts/smoke_lifecycle.py` checks both on one real MicroVM. A warm pool of
  pre-created hosts was left out: a pre-created host takes a slot of the team quota that
  real tasks need, and warm start already shortens a host's start to a clone's (design doc §4.5).

## Subagent sandboxes (one MicroVM per subagent)

`langchain_digitalocean_microvm.subagents` keeps ordinary (synchronous) deepagents
subagents in the supervisor's process but gives each one its own MicroVM for
`execute` and its files (design doc §5). deepagents builds every subagent's
`FilesystemMiddleware`, summarization and skills middleware on the parent's
backend; `per_subagent_specs` rebuilds them on the subagent's sandbox, and
deepagents swaps them in by name.

```python
import contextlib

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_digitalocean_microvm import MicroVMLifecycle
from langchain_digitalocean_microvm.subagents import asubagent_sandboxes

lifecycle = MicroVMLifecycle()
SPECS = [{"name": "analyst", "description": "...", "system_prompt": "..."},
         {"name": "reviewer", "description": "...", "system_prompt": "..."}]

@contextlib.asynccontextmanager
async def make_graph(config, runtime):          # langgraph.json: "./agent.py:make_graph"
    thread_id = config["configurable"]["thread_id"]
    async with asubagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model,
                                   seed_files={"analyst": [("/data/in.csv", b"...")]}) as sb:
        yield create_deep_agent(model=model, subagents=list(sb.subagents),
                                backend=sb.mount(StateBackend()))   # /subagents/<name>/ -> its VM
```

- Per run: the thread's sandboxes (`<thread id>-<subagent>`) are fetched or
  created **in parallel** on a bounded thread pool, off the event loop
  (`max_workers`, default 4); new ones get their `seed_files`; all are paused
  when the run ends (teardown logs failures instead of raising). If one
  cannot be provisioned, the others are paused again and the error raised.
- `sb.mount(parent_backend)` is a `CompositeBackend` routing
  `/subagents/<name>/` to that subagent's sandbox: the parent reads (and
  writes) subagent files there; `execute` only ever runs on the parent's own
  backend.
- Sync twins (`subagent_sandboxes`, `provision_subagent_sandboxes`,
  `pause_subagent_sandboxes`) for scripts; `per_subagent_specs(specs,
  backends=...)` alone if you hold the backends yourself.
- Sandboxes carry the `microvm-subagent-sandbox` tag. Pass `store=GatewayStore(...)`
  (the examples use `default_db_path("microvm")`) and every run records each
  sandbox's last use; `scripts/gc_microvms.py --sandbox-ttl 24h` deletes the
  unused ones (and, by age, ones no store knows). `delete_all.py` removes them
  with everything else.
- Not covered: the default general-purpose subagent keeps the parent's
  backend (pass your own `general-purpose` spec to sandbox it); async and
  compiled subagents pass through unchanged.
- deepagents cannot enforce `permissions` on a backend that runs commands, so
  a sandboxed subagent with (or inheriting) rules is refused with a clear
  error; `permissions: []` on the spec opts out (the VM is the boundary).
- The sandbox image's user must be able to write to `/`: the model picks
  absolute paths itself (`/outputs/...`, `/report.md`) and deepagents offloads
  to `/large_tool_results/` and `/conversation_history/`. The `microvm-subagent`
  image (uid 1000) does not qualify. `scripts/build_sandbox_image.py` pushes
  one that does, with crane alone (no Docker, no extra layer): the
  `python:3.12-slim` linux/amd64 manifest pinned by digest, run as root, with
  the sandbox server (`sandbox_server.py`) on port 8080 as its entrypoint: it
  answers `/healthz` without a token, serves nothing else without the
  sandbox's own token, and logs nothing. Both transports work with it. Set the
  printed reference as `DO_MICROVM_IMAGE`:

  ```bash
  uv run python \
      scripts/build_sandbox_image.py \
      --repository registry.digitalocean.com/<registry>/microvm-sandbox
  ```

  `docker.io/kennethreitz/httpbin:latest` (`DO_MICROVM_HTTP_PORT=80`) also
  works but is larger (creates are slower).
- `langchain_digitalocean_microvm.client.api_calls()` counts every request this process sent
  to the MicroVMs API by operation (`exec`, `create`, `pause`, ...; retries as
  `retry:<status>`), which is how example 03 reports its exec budget.

### Bounding concurrent creates (`max_concurrent_creates`)

`ThreadSandboxes(max_concurrent_creates=)`, `provision_subagent_sandboxes(max_workers=)` and
`PerTask(max_concurrent=)` each bound their own callers, so several threads or subagents together
can still start many MicroVMs at once. `MicroVMConfig.max_concurrent_creates`
(`DO_MICROVM_MAX_CONCURRENT_CREATES`, default none) is one bound below all of them: a
`MicroVMLifecycle` runs at most that many `get_or_create` calls (creates and resumes, until the
sandbox is connected and seeded) and the boot of a template build at once, and the rest wait.
It is not held while a sandbox is in use or while a template's `setup` commands run. The bound is
per lifecycle instance, so per process: separate processes are not coordinated, and the team's
MicroVM limit is the only bound across them. It limits concurrency, not how many sandboxes exist.

### Templates and on-demand sandboxes

Two options of `asubagent_sandboxes` / `provision_subagent_sandboxes` (and their sync twins):

```python
template = lifecycle.build_template("analyst", setup=["pip install pandas matplotlib"],
                                    seed_files=[("/data/sales.db", db_bytes)])
async with asubagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model,
                               templates={"analyst": template, "reviewer": template},
                               on_demand=True) as sb:
    ...
```

- `templates={name: SandboxTemplate}`: a subagent's sandbox that has to be created is cloned
  from that checkpoint instead of booted from the image (`seed_files` still go on top, e.g.
  per-thread inputs). A `TemplateCache` in place of the template rebuilds it if a clone finds
  its checkpoint deleted, then retries the clone. A name that is not a sync subagent in
  `specs` raises `ValueError`.
- `on_demand=True`: nothing is created or resumed when the run starts. Each subagent gets an
  `OnDemandSandbox` (a `BaseSandbox`) that runs the same get-or-create on its first command or
  file operation, once even under concurrent first uses, off the event loop on the async paths.
  If it fails, the agent gets a failed command / per-file error and the next use retries.
  `sb.started()` lists the sandboxes the run actually used; teardown pauses only those. The
  parent's mount is on-demand too: reading `/subagents/<name>/...` starts that sandbox.

Why this matters in production:

- **Pay for and wait on what the request needs.** A supervisor with ten specialists usually
  calls two or three per request. Eager provisioning resumes (or creates) all ten on every
  run; on-demand starts exactly the ones the model delegates to, and subagents it calls in
  parallel start their VMs in parallel. A follow-up that needs one specialist wakes one VM.
- **Cold start leaves the critical path.** A clone is running in seconds with packages,
  tools, reference data and even warmed-up processes already in place (example 06 compares a
  clone with a cold sandbox). Without a template, every new thread pays boot +
  upload + `pip install` per subagent, and does it while the user waits.
- **Reproducible, auditable environments.** Every clone starts from the same checkpoint,
  named by a hash of image, size, setup and seed files. No "works on the first thread"
  drift from installs at run time, no network dependency (PyPI) inside a user's request, and
  a changed setup is a new template rather than a mutated one.
- **Smaller attack and failure surface per request.** Setup that needs the network or
  credentials (private package indexes, data pulls) happens once at build time; the
  subagents' VMs never see those credentials.
- **Cheap recovery.** A sandbox that was wiped, wedged or deleted is replaced by the next
  get-or-create with a fresh clone, so "delete it" is a valid repair.

`examples/08_board_pack/` uses both: six specialists, one template, one VM per specialist per
thread, started when the orchestrator delegates.

The eager, template-less example is `examples/03_subagent_sandboxes/`; `scripts/smoke_subagents.py`
checks the whole flow on two real MicroVMs.

### Per-task sandboxes (one MicroVM per `task` call)

When a subagent's units of work only appear at run time (repositories found by a search, tickets
from a queue) and must not share files, `per_task=` gives that subagent one sandbox per `task`
call, or per key, instead of one per thread:

```python
from langchain_digitalocean_microvm.subagents import PerTask, asubagent_sandboxes

async with asubagent_sandboxes(
    lifecycle, SPECS, thread_id=thread_id, default_model=model,
    templates={"analyst": template},
    per_task={"analyst": PerTask(
        key=lambda call: call.description.split()[0],   # default: the tool call id
        max_sessions=8, max_concurrent=4, when_done="delete",
    )},
) as sb:
    agent = create_deep_agent(model=model, subagents=list(sb.subagents),
                              middleware=[*sb.middleware],    # required with per_task
                              backend=sb.mount(StateBackend()))
```

- The subagent's built-in file tools, `execute`, summarization and skills all run in the sandbox of
  the task that is running: its spec gets a `PerTaskSandbox`, which resolves the sandbox on every
  call from the task the parent's `TaskKeyMiddleware` (`sb.middleware`) is running. Custom tools take
  `sb.per_task[name]` (or their own `PerTaskSandbox(name)`) as their backend, or call
  `current()` / `acurrent()` for the task's connected `MicroVMSandbox`; `.key` is the task's key.
  Used outside a task of its subagent, the router raises `SandboxError`.
- Sandbox ids: `<thread id>-<subagent>-<key>` (the MicroVM name is hashed when the key is long or
  not name-safe). A sandbox starts on the task's first command or file operation, cloned from the
  spec's template if it has one (seeds go on top); parallel tasks start in parallel. A task that
  never touches its sandbox creates none. The key function maps a `TaskCall` (`id`,
  `subagent_type`, `description`, `args`) to a key; one that raises or returns an empty key refuses
  the call with a message the model reads. Calls with the same key in a thread use the same sandbox,
  one at a time.
- `when_done`: `"delete"` (default) deletes the sandbox when its task returns; `"pause"` keeps it for
  the next task with the same key, in this turn or a later one. A task interrupted inside the
  subagent (an approval) is always paused: on resume, langgraph replays the call with the same tool
  call id, so the default key finds the same sandbox. Teardown pauses any per-task sandbox still live.
- `max_sessions` (default 8, `None` for no limit) counts the subagent's sandboxes in the thread,
  running and paused (with a `store`, the records of earlier turns count too); a call beyond it is
  refused with a tool message the model reads. Paused MicroVMs count against the team's limit, so
  lower it on a small team. `max_concurrent` (default 4) bounds tasks running at once; further calls
  wait.
- A sandbox that cannot start (for example at the team's MicroVM limit) reaches the subagent as a
  failed command, as with `on_demand`.
- The MicroVMs carry `microvm-subagent-sandbox` and `microvm-per-task-sandbox`, so the
  `--sandbox-ttl` clean-up covers them; with a `store`, deleting a sandbox marks its row destroyed.
  `sb.started_per_task()` lists the live ones.
- Not supported: the parent mount (`sb.mount()` routes fixed subagents only; per-task results come
  back through the subagent's answer or its tools).

### deepagents-code (`dcode --sandbox microvm`)

`langchain_digitalocean_microvm.integrations.deepagents_code.MicroVMSandboxProvider` is a
deepagents-code `SandboxProvider` over `MicroVMLifecycle`, registered as the
`microvm` provider (entry point `deepagents_code.sandbox_providers`). Only
dcode imports it; this package does not depend on deepagents-code.
`get_or_create(sandbox_id=None)` creates `dcode-<hex>`; dcode reports and later
deletes a sandbox by `backend.id` (the MicroVM id), so both ids work everywhere.
Options: `seed_files`, `tags`, `egress` (an `EgressPolicy` or a `DO_MICROVM_EGRESS` string); anything else is refused. Its tests run where
deepagents-code is installed:
`uv run pytest tests/unit_tests/test_deepagents_code_provider.py` (the `test` dependency group installs deepagents-code).

## Action Gateway tools (a supervisor outside any sandbox)

`integrations/action_gateway.py` gives a deep agent running in your own process DigitalOcean and
third-party actions through an Action Gateway endpoint of its own. It is independent of the sandbox
classes and unrelated to the agent-host gateway; it needs no extra.

```python
from langchain_digitalocean_microvm.integrations.action_gateway import ActionGateway, ActionPolicy

policy = ActionPolicy(allow=["exa_web_search"], ask=["digitalocean_droplet-create"])
with ActionGateway(policy) as gateway:  # the endpoint's tools: the names in allow and ask
    agent = create_deep_agent(tools=gateway.tools(), checkpointer=checkpointer)
```

- **Configuration.** `DO_ACTION_GATEWAY_TOKEN` (or `token=`): a token with `action_gateway:*`
  scopes. `DO_ACTION_GATEWAY_ACTOR_ID` (or `actor_id=`): the actor DigitalOcean actions run as; by
  default the part of your DigitalOcean email before the `@`, up to 64 letters, digits, `.`, `_` and
  `-`, and your user UUID if the email gives nothing usable. `DO_MICROVM_TRANSPORT=local` (or
  `client=LocalActionGatewayClient()`) runs everything in memory.
- **Endpoints.** The gateway is reached through an endpoint: `POST /v2/action-gateway/endpoints` creates one with its
  first version (the tools, the policy and the actor), which is active at once. Calls go to the data plane the endpoint's
  `mcp_url` names (`POST /tools/invoke`, `POST /tools/search`, approvals), with the
  header `X-Endpoint-Id`. The client accepts only an `https` host on `do-ai.run` or `do-ai-test.run`; for anything else it
  deletes the endpoint and raises `ActionGatewayError`. `DO_ACTION_GATEWAY_DATA_URL` (or `data_url=`) sends every call
  to the host it names instead, and the returned `mcp_url` is not used.
- **Policy.** `ActionPolicy(allow=, ask=, deny=)` lists tool names without versions; everything else
  is denied. `action_search` and `action_invoke` are always allowed, as the endpoint's policy needs.
  `tools=` lists the endpoint's slugs, by default the names in `allow` and `ask`. A slug without a version
  (`github_list_commits`) is sent as it is; `name@version` pins one. When the service refuses the endpoint (for example
  over a pinned version), `ActionGateway` raises `ActionGatewayError` with its answer.
- **Policy as a file.** `ActionPolicy.from_yaml(path)` (a path or an open text stream; a `str` is a path, never YAML text)
  and `ActionPolicy.from_mapping(dict)` build the same policy from data:

  ```yaml
  # policy.yaml: three lists of tool names; the default is always deny
  allow: [github_list_commits, digitalocean_droplet-get]
  ask: [digitalocean_restore-droplet]
  deny: [digitalocean_droplet-delete]
  ```

  Entries are kept as written, with a version where one is pinned, so the lists can also feed `tools=`
  (`ActionGateway(policy, tools=[*policy.allow, *policy.ask, *policy.deny])`); the endpoint's policy drops the versions.
  The loader raises `ValueError`, naming the file, for an unknown key (`default`, `rules` and `permissions` get a hint),
  a value that is not a list of names, a `do.actions/` prefix, `action_search` or `action_invoke`, and a tool that is
  listed twice or in two tiers. It reads plain YAML only (`yaml.safe_load`). The Python constructor does not run these checks.
- **Tools.** `gateway.tools()` returns `action_search` (the endpoint's tools that fit a plain-English
  description of the task) and `action_invoke` (`tool`, `arguments`, `rationale`); `preload=[slug, ...]`
  adds one direct tool per slug with its own input schema, and `tools(meta=False)` returns only those.
  Sync and async.
- **Approvals.** An `ask` rule makes the gateway answer `requires_approval`. The tool interrupts
  with `{"type": "action_approval", "tool", "arguments", "rationale", "approval_id", "expires_at"}`.
  Resume the graph with `"approve"`, `"deny"` or `{"decision": ..., "reason": ...}` (anything else
  denies). The decision is posted with the gateway token, so the code that resumes the graph is the
  approver, and the package gives the agent no tool to decide. The same token also invokes actions,
  so an agent that can run code in the process holding it could decide too: keep the token out of
  its reach. After an approval the package repeats the call once. An approval covers one call with
  the same arguments and expires: a decision after that is reported to the agent, which can call
  the tool again. A graph needs a checkpointer, as for any interrupt.
- **Approvals outside the agent's process.** `request(tool, arguments, rationale)` calls an action
  without interrupting: it returns the answer as text, or a `PendingApproval` (`tool`, `arguments`,
  `rationale`, `approval_id`, `expires_at`) for an `ask` rule. `decide(pending, "approve" | "deny",
  reason=None)` posts the decision and, once approved, repeats the call, returning the same texts as
  the tool. Use them in the code that holds the token (a relay or a supervisor) when the agent runs
  elsewhere, for example in a MicroVM: the agent proposes, that code asks the human and acts, and no
  credential that could approve reaches the agent. Example 12's relay works this way.
- **Connect links.** A DigitalOcean action the actor has not connected answers with an OAuth connect
  link. The tool interrupts with `{"type": "connect_required", "url", "tool"}`; show the link, and
  resume with any value but `"cancel"` to retry the call once. For an `ask` action and an actor that is
  not connected yet, an approved call may not run and the connect interrupt follows; after connecting,
  the resumed tool requests a fresh approval, which the
  decision already given answers.
- **Lifecycle.** The endpoint is created on first use and deleted by `close()`, the context manager
  (`with` / `async with`) or the garbage collector. A process killed before that leaves its endpoint
  behind; the package's clean-up scripts do not list these endpoints.
- **Endpoint names.** An endpoint is named `endpoint_prefix` plus 8 random hex digits; the default prefix
  is `microvm-ag-` (`ENDPOINT_PREFIX`). `gateway.endpoint_id` is its id once created. Give each owner a prefix of its
  own (`ActionGateway(policy, endpoint_prefix="microvm-ag-<job>-")`), and another process can find what a killed one
  left: `ActionGatewayClient(token).list_endpoints(prefix=...)` returns every endpoint the token can see whose name
  starts with the prefix (`id`, `name`, `actor_id`, `created_at`, ...; the client reads every page and filters), and
  `delete_endpoint(endpoint_id)` deletes one. `LocalActionGatewayClient` lists its endpoints the same way.
- **Live check.** `scripts/check_setup.py --deep` calls one read-only action through a throwaway
  `microvm-ag-check-*` endpoint and deletes it. `scripts/smoke_action_gateway.py` drives one endpoint through
  an allowed call, a connect link and an approval (needs a terminal).

## Where state lives

The gateway store (hosts, threads, sandbox rows) is a SQLite file in the package's per-user state
directory, `agent_host.store.state_dir()`: `~/Library/Application Support/langchain_digitalocean_microvm`
on macOS, `%LOCALAPPDATA%\langchain_digitalocean_microvm` on Windows, else
`$XDG_STATE_HOME/langchain_digitalocean_microvm` (`~/.local/state/...` when unset). It is
`gateway.sqlite` (`gateway-<ns>.sqlite` with a namespace), and `gateway-local.sqlite` in local mode;
`default_db_path(mode, namespace)` returns the path. Earlier versions kept it in
`~/.langchain_digitalocean_microvm`: a store there is still used until a file of that name exists in
the state directory, so move it to adopt the new location (nothing is moved for you).
`scripts/delete_all.py` marks the rows of every store it finds in either directory.
The smoke tests' result files (`probe/<date>/`) stay in `~/.langchain_digitalocean_microvm`.

## Namespaces (several users or deployments on one team)

Set `DO_MICROVM_NAMESPACE` (`MicroVMConfig(namespace=...)`: 1-16 lowercase letters or digits) so that
users or deployments sharing a team do not touch each other's MicroVMs. With a namespace `<ns>`:

- every MicroVM carries the tag `microvm-namespace:<ns>`;
- sandboxes, agent hosts and golden checkpoints are named `microvm-sandbox-ns-<ns>-<id>`,
  `microvm-agenthost-ns-<ns>-<id>` and `microvm-golden-ns-<ns>-…`, so the same sandbox id, or the same pooled
  host (`pool-<graph>`), is a different MicroVM per namespace;
- the gateway store is `gateway-<ns>.sqlite` in the state directory (`gateway-local-<ns>.sqlite` in
  local mode), also for `GatewaySettings(namespace=...)` and `default_db_path(mode, namespace)`;
- `MicroVMAgentHost.list_hosts()` and `MicroVMLifecycle.sandbox_id_for()` see only that namespace.

Without a namespace nothing changes: the names above carry no `ns` part, the store is `gateway.sqlite`,
and a MicroVM with no namespace tag belongs to the default namespace. Setting a namespace later does not
move existing MicroVMs: tags cannot be changed after creation and the names differ, so they stay in the
default namespace and are no longer found by their sandbox id.
Sandbox templates (`microvm-template-…`) are content-addressed and shared by every namespace
(their build MicroVM carries the namespace tag).

Clean-up touches one namespace: `--namespace NS`, else `DO_MICROVM_NAMESPACE`, else the default namespace.
Within it, only what you created: every MicroVM `MicroVMClient.create` makes carries `microvm-owner:<id>`,
where the id is a random string in the file `owner-id` in the state directory (`owner.py`: `owner_id()`,
`is_mine(tags)`). MicroVMs of other users and ones created before owner tags existed are left alone unless
`--all-owners`; checkpoints follow their source MicroVM (a bare name says nothing about its maker).

```bash
uv run scripts/delete_all.py [--namespace NS]    # dry run; only what you created in that namespace
uv run scripts/delete_all.py --all-owners        # also other users' MicroVMs in the namespace
uv run scripts/delete_all.py --all-namespaces    # every namespace on the team, other users' included
uv run scripts/gc_microvms.py [--namespace NS]   # that namespace's hosts and sandboxes, with its store;
                                                 # --all-owners: also unknown hosts other users created
```

`--all-namespaces` with `gc_microvms.py` reads one store, so other namespaces' hosts and sandboxes count as
orphans and go once older than `--orphan-ttl`; prefer one run per namespace. `list_owned(client, namespace=...)`
(`namespace` module) lists a namespace's live MicroVMs. The live smoke tests refuse to start while MicroVMs
of their own namespace exist, and say how many belong to others.

## Performance

Typical times of the primitives the package is built on, from `scripts/bench.py`: five cycles of create, 20 endpoint
calls, 20 exec calls, pause, resume and delete on one `1vcpu-2gb` MicroVM from the default sandbox image in `mkc1`, then one
template build and five clones. Times are seconds (median, and the 95th percentile of the in-region run).

| Primitive | In region | From a remote client | p95 (in region) |
| --- | --- | --- | --- |
| API round trip (`GET` a MicroVM) | 0.24 | 0.34 | 0.34 |
| Endpoint call (`execute` through the sandbox server) | 0.13 | 0.20 | 0.17 |
| Exec API call (a short command) | 0.32 | 0.41 | 0.45 |
| Create from the image, until the sandbox answers | 8.6 | 9.9 | 9.3 |
| Pause, until it reads `paused` | 0.8 | 2.0 | 1.8 |
| Resume, until the sandbox answers | 1.5 | 2.1 | 2.9 |
| Delete | 0.5 | 0.7 | 0.6 |
| Template build (once: boot, seed, checkpoint, delete) | 19 | 22 | n/a |
| Clone from the template, until the sandbox answers | 6.9 | 5.8 | 8.3 |

- **In region** is a client on a droplet in `mkc1`; **remote** is a laptop on the US west coast. A remote client pays the
  distance on every request, so an operation made of several requests (pause and resume confirm their state with `GET`s)
  gains the most from running next to the MicroVMs. The API itself needs a couple of tenths of a second per request wherever
  it is called from.
- **What the rows include.** Create covers the MicroVM reaching `running`, its endpoint becoming active and the one exec that
  gives the sandbox server its token. Clone covers the same steps for a checkpoint. Pause and resume include the `GET`s that
  confirm the state.
- **A clone is about as quick as booting a small image.** What a template saves is the setup that would otherwise follow
  the boot (packages, data, warmed-up processes), so the gain grows with the setup
  (example [06](../examples/06_warm_start) compares the two).
- **Not included.** Model time, which dominates an agent run; the first request to a freshly started server (not timed);
  and anything that depends on the image or the checkpoint: a larger one takes longer to start.
- **Few samples.** Five runs each, so a difference of a second between two columns, as in the clone row, is noise.

Run `scripts/bench.py --client-label <where it runs>` for your own region, size, image and client; it deletes only what it
created, and writes the same table to its results file.

## Service limits

How the package copes with the service's limits. The limits themselves (sizes, regions, idle
timeout, exec) are in the
[MicroVMs documentation](https://docs.digitalocean.com/products/microvms/details/limits/);
`GET /v2/microvms/options` lists your team's own values.

- Team limits: a maximum number of MicroVMs (paused ones count) and of total memory. Each
  example's README states the minimum it needs.
- Nothing about a MicroVM can change after creation, so the package finds MicroVMs by a
  deterministic name and replaces them rather than updating them.
- "Idle" means no inbound requests: a busy VM with no traffic still gets paused (and frozen).
  Anything long-running must be polled more often than `DO_MICROVM_IDLE_TIMEOUT`.
- The endpoint transport retries `502`/`503`/`504` about a second later; so does the agent-host
  gateway (`upstream_retries`, `upstream_retry_delay`), except a run creation the Agent Server
  itself answered, which is never sent twice.
- Use an access-only token (`microvm:access`) for endpoint traffic, never the control token.
- exec runs single commands and is not built for high volume (a time limit per command and a
  per-team rate limit). With the default `endpoint` transport sandboxes use it only once per
  new sandbox; with `DO_MICROVM_TRANSPORT=exec` every file operation is one or more exec calls,
  so very chatty agents can hit the limit (the client backs off and retries).
- A response through the endpoint has to be complete within about 15 s (otherwise 504); the
  endpoint transport keeps every request shorter and runs longer commands as jobs.
- A deleted MicroVM's endpoint answers 404 instead of refusing connections.
- `MicroVMClient.pause`/`resume` confirm the requested state with a GET, polling up to 15 s.
- `LocalTransport` cannot redirect shell output (`> /abs/path`) into its root, so
  `MicroVMSandbox` turns off deepagents' capture-at-source `execute` offload for it
  (`SandboxTransport.supports_capture_offload`); real MicroVMs keep it.
