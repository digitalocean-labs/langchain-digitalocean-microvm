# langchain-digitalocean-microvm

[DigitalOcean MicroVMs](https://www.digitalocean.com/) as sandboxes for
[LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview).

Everything DigitalOcean-specific lives in `client.py`.

[MicroVMs](https://docs.digitalocean.com/products/microvms/) are lightweight
virtual machines that run one container image in an isolated kernel, **pause**
after an idle timeout with memory, files and processes intact, and **resume** on
the next request. This package gives Deep Agents:

| | What | Module |
| --- | --- | --- |
| **Sandbox backend** | `MicroVMSandbox`, a deepagents `BaseSandbox`: `execute`, file tools and file transfer in a MicroVM | `backend`, `transport` |
| **Lifecycle** | `MicroVMLifecycle`: get-or-create by a stable id, seed files, pause, resume, delete | `provider` |
| **Templates** | `build_template` / `get_or_create(template=...)`: prepare a sandbox once, checkpoint it, clone new ones warm; `TemplateCache` keeps a server's template and rebuilds it if its checkpoint is deleted | `provider`, `template_cache` |
| **Thread sandboxes** | `ThreadSandboxes`: one MicroVM per conversation thread for a per-thread graph factory; get-or-create on enter (optionally from a template), pause on exit, a bound on concurrent creates, expiry by `gc_microvms.py` | `threads` |
| **Deep Agents Code** | `dcode --sandbox microvm` (entry point `deepagents_code.sandbox_providers`) | `integrations.deepagents_code` |
| **Subagent sandboxes** *(experimental)* | One MicroVM per synchronous subagent, provisioned in parallel or on demand, optionally cloned from a template, paused between runs | `subagents` |
| **Agent hosts** *(experimental, `[agent-host]` extra)* | Async subagents that run inside MicroVMs (an Agent Server each) behind an embedded Agent Protocol gateway | `agent_host` |
| **Action Gateway tools** | DigitalOcean and third-party actions for a supervisor agent in your own process, with allow/ask/deny policy and approvals as LangGraph interrupts | `integrations.action_gateway` |

Everything also runs in **local mode**: no DigitalOcean account, no network.

## Try it without a DigitalOcean account

The tests need no keys. The example needs a model: `DO_MODEL_ACCESS_KEY` (DigitalOcean Serverless Inference) or
another provider's key, see [`examples/README.md`](examples/README.md#setup).

```bash
uv sync --all-extras
uv run pytest                                              # unit tests: no keys, no network
export DO_MODEL_ACCESS_KEY=...                             # or EXAMPLES_MODEL plus that provider's key
DO_MICROVM_TRANSPORT=local uv run examples/01_quickstart/quickstart.py   # a deep agent in a local directory
```

The tour runs example 11 with its live dashboard in local mode in a temporary directory (about 2 minutes; on macOS the
dashboard opens in a window of its own):

```bash
uv run examples/tour.py
uv run examples/tour.py --same-terminal   # keep the dashboard in this terminal
uv run examples/tour.py --full            # adds 01, 03, 06 and 10
```

For real MicroVMs, check what your `.env` is missing, per example (read-only), then set up what is missing:

```bash
uv run python scripts/check_setup.py             # --offline checks the variables only
uv run python scripts/check_setup.py --deep      # also creates, uses and deletes one throwaway MicroVM
uv run python scripts/setup_examples.py          # sets up what is missing (--dry-run checks only)
```

In local mode the agent's shell
commands run on your machine, so use a scratch checkout and read what an example does before you run it.

## Installation

Install from GitHub (pin a tag or commit for reproducible installs):

```bash
pip install "langchain-digitalocean-microvm @ git+https://github.com/digitalocean-labs/langchain-digitalocean-microvm"                  # sandbox backend, lifecycle, subagent sandboxes, dcode provider
pip install "langchain-digitalocean-microvm[agent-host] @ git+https://github.com/digitalocean-labs/langchain-digitalocean-microvm"    # + agent hosts and gateway
```

With uv: `uv add git+https://github.com/digitalocean-labs/langchain-digitalocean-microvm` (add `--extra agent-host` for
agent hosts and the gateway).

From a checkout of this repository, for development and the examples:

```bash
uv sync --all-extras
```

Requires Python 3.11+ (3.12+ for `dcode`, example 02) and `deepagents>=0.7,<0.8`.

## Quickstart

```python
from deepagents import create_deep_agent
from langchain_digitalocean_microvm import MicroVMLifecycle

lifecycle = MicroVMLifecycle()  # reads DO_MICROVM_* from the environment
backend = lifecycle.get_or_create(
    sandbox_id="my-sandbox",
    seed_files=[("/notes.md", b"hello\n")],  # uploaded only when this call creates the MicroVM
)

agent = create_deep_agent(model="anthropic:claude-haiku-4-5", backend=backend)
agent.invoke({"messages": [{"role": "user", "content": "Read /notes.md, then write /fact.txt."}]})

lifecycle.pause(backend)  # optional: MicroVMs also pause on their own when idle

# ... later, possibly in another process ...
backend = lifecycle.get_or_create(sandbox_id="my-sandbox")  # resumes it: same files
```

With `DO_MICROVM_TRANSPORT=local` the sandbox is a local directory.

## Configuration

| Variable | Description |
| --- | --- |
| `DO_MICROVM_API_TOKEN` | DigitalOcean API token with `microvm:create/read/update/delete` (update covers pause, resume and exec). |
| `DO_MICROVM_IMAGE` | Sandbox image: `linux/amd64`, keeps running, ships `sh`, coreutils and `python3`. For agents, its user must be able to write to `/`; `scripts/build_sandbox_image.py` builds one (no Docker needed). |
| `DO_MICROVM_TRANSPORT` | `endpoint` (default): commands and files over the MicroVM's HTTPS endpoint to a small server inside it (the sandbox image runs it; any image with `python3` gets it installed on first connect); needs `DO_MICROVM_ACCESS_TOKEN`. `exec`: everything through the MicroVMs exec API (slower, rate limited). `local`: everything local. |
| `DO_MICROVM_HTTP_PORT` | The port the image serves HTTP on (default `8080`). |
| `DO_MICROVM_SIZE` | `1vcpu-2gb` (default), `2vcpu-4gb`, ... (`GET /v2/microvms/options` lists your team's). |
| `DO_MICROVM_REGION` | Default `mkc1`. |
| `DO_MICROVM_MAX_CONCURRENT_CREATES` | At most this many sandboxes are created or resumed at once per process, whoever asks (threads, subagents, per-task sandboxes); default: no limit. Eases bursts against the API's rate limits and the team's MicroVM limit. |
| `DO_MICROVM_NAMESPACE` | Separates your MicroVMs from other users' or deployments' on a shared team: 1-16 lowercase letters or digits (default: none). Names, a tag and the gateway store carry it; clean-up scripts touch only what you created in one namespace unless told otherwise. |
| `DO_MICROVM_IDLE_TIMEOUT` | Pause after this long without an **inbound** request (default `10m`). CPU work and outbound calls do not count. |
| `DO_MICROVM_ACCESS_TOKEN` | Required by agent hosts and the `endpoint` transport: token for requests *to* a MicroVM's endpoint, scoped to `microvm:access` only. The control token `DO_MICROVM_API_TOKEN` is never used for it. |
| `DO_ACTION_GATEWAY_TOKEN`, `DO_ACTION_GATEWAY_ACTOR_ID` | Action Gateway tools (`integrations/action_gateway.py`): a [personal access token](https://docs.digitalocean.com/reference/api/create-personal-access-token/) authorized for Action Gateway ([quickstart](https://docs.digitalocean.com/products/managed-agents/action-gateway/quickstart/)) and the actor the actions run as (default: the part of your DigitalOcean email before the `@`). |

Agent-host settings (`DO_MICROVM_HOST_MODE`, `DO_MICROVM_AGENT_IMAGE`, warm
start, ...) are in [`.env.example`](.env.example) and
[`docs/reference.md`](docs/reference.md#agent-hosts-subagents-as-separate-processes).

## Examples

| | |
| --- | --- |
| [`01_quickstart`](examples/01_quickstart) | A deep agent in a MicroVM across a pause/resume |
| [`02_deepagents_code`](examples/02_deepagents_code) | `dcode --sandbox microvm` |
| [`03_subagent_sandboxes`](examples/03_subagent_sandboxes) | One MicroVM per synchronous subagent |
| [`04_async_subagents`](examples/04_async_subagents) | Async subagents running inside MicroVMs |
| [`05_data_analyst`](examples/05_data_analyst) | SQL and charts in a MicroVM that pauses between runs |
| [`06_warm_start`](examples/06_warm_start) | Sandbox templates from checkpoints: clone a prepared sandbox in seconds |
| [`07_sales_assistant`](examples/07_sales_assistant) | The course capstone: one MicroVM per conversation thread, started warm from a template, paused between turns |
| [`08_board_pack`](examples/08_board_pack) | Demo app: six specialist subagents, one MicroVM each, cloned on demand in parallel and in series |
| [`09_incident_response`](examples/09_incident_response) | An incident team of agents in MicroVMs; a restore waits for a human's approval (Action Gateway) |
| [`10_erasure`](examples/10_erasure) | An erasure request: locator MicroVMs classify records so the models see manifests only |
| [`11_vulnerability_response`](examples/11_vulnerability_response) | One analyst MicroVM per repository; patches re-tested in fresh MicroVMs |
| [`12_incident_response_agents`](examples/12_incident_response_agents) | Example 09's incident handled by whole agents, each in an agent host of its own |

01, 04, 05 and 07 are ports of labs of the
[Deep Agents course](https://github.com/langchain-ai/lca-deepagents) that use sandboxes; the others
were written for this package. See [`examples/README.md`](examples/README.md) for setup and what each
example needs.

## Known limitations

What the service documents or what this package measured; details in
[`docs/reference.md`](docs/reference.md#service-limits).

- The team's MicroVM limit applies to paused MicroVMs too, so examples state the minimum they need.
- A token with `microvm:access` reaches every MicroVM of the team (see the MicroVMs documentation); endpoint traffic uses an access-only token, never the control token.
- The exec API is meant for single commands; the `endpoint` transport carries the data.
- A response through the endpoint has to complete within about 15 s, so longer commands run as jobs the client polls.
- Agent hosts and subagent sandboxes are experimental and may change.

**Tested:** unit tests and LangChain's standard sandbox suite, offline and on a real MicroVM. Examples 10 and 11 use simulated systems. Found a bug? Please let us know.

## Documentation

- [`docs/design.md`](docs/design.md): design, decisions and the MicroVM facts they rely on.
- [`docs/examples-architecture.md`](docs/examples-architecture.md): what runs inside the MicroVM in each example (sandbox as a tool, agent in sandbox).
- [`docs/reference.md`](docs/reference.md): module-by-module reference and service limits.
- [`images/agent-host`](images/agent-host): the agent-host image (graphs served inside a MicroVM).
- [`scripts/`](scripts): API probe, live smoke tests, image builds, clean-up (`gc_microvms.py`, `delete_all.py`).

## Development

```bash
uv sync --all-extras
uv run pytest                                   # unit tests, local mode (about 1-2 minutes)
uv run pytest -m e2e                            # local end-to-end tests (real `langgraph dev`)
uv run pytest -m slow                           # LangChain's standard sandbox suite, offline
uv run pytest -m live                           # the standard suite against a real MicroVM
uv run ruff check .
uv run pyright                                  # type check src/
```

`tests/standard_suite_gaps.py` lists the standard-suite cases expected to fail
and why (two disagreements between deepagents 0.7.21 and langchain-tests 1.1.9,
and BaseSandbox's grep needing GNU grep).

## License

Apache License 2.0. See [LICENSE](LICENSE). Third-party software and attribution to LangChain: [NOTICE](NOTICE).

This project is not affiliated with or endorsed by LangChain, Inc. LangChain, LangGraph, LangSmith and Deep Agents are trademarks of LangChain, Inc.
