# Examples

What DigitalOcean MicroVMs add as sandboxes for Deep Agents, from a single sandbox to whole teams of agents.

01, 04, 05 and 07 are ports of labs of the [Deep Agents course](https://github.com/langchain-ai/lca-deepagents) that use sandboxes; each one's README names the lab. The others were written for this package: 02, 03 and 06 each show one feature (the `dcode` provider, subagent sandboxes, templates), and 08-12 are larger applications built on several of them.

Each example has its own model setup, run commands and README.


| Example | Min. MicroVMs | Also needed for the real run | Shows |
| --- | --- | --- | --- |
| [`01_quickstart`](01_quickstart) | 1 | sandbox image, model key | A deep agent whose filesystem and shell are a MicroVM; pause and resume keep its files |
| [`02_deepagents_code`](02_deepagents_code) | 1 | sandbox image, `dcode` (Python 3.12+) | `dcode --sandbox microvm` (Deep Agents Code) |
| [`03_subagent_sandboxes`](03_subagent_sandboxes) | 2 per thread | sandbox image, model key | Synchronous subagents with one MicroVM each, created in parallel, paused between turns |
| [`04_async_subagents`](04_async_subagents) | 2 | agent-host image built with `uv` and `crane`, pushed to your registry | Async subagents that run *inside* MicroVMs (Agent Server per host) behind an embedded gateway (experimental) |
| [`05_data_analyst`](05_data_analyst) | 1 | sandbox image, model key | SQL and matplotlib charts in a MicroVM; the paused sandbox keeps the database and packages between runs |
| [`06_warm_start`](06_warm_start) | 1 | sandbox image, model key | Sandbox templates: prepare once, checkpoint, clone new sandboxes in seconds |
| [`07_sales_assistant`](07_sales_assistant) | 1 per thread | sandbox image, model key; optional Tavily key | A sales assistant with one MicroVM per conversation thread (template, seeded skills and database, mail MCP server), paused between turns |
| [`08_board_pack`](08_board_pack) | 6 | sandbox image, model key | The Board Pack app: an orchestrator and six specialists, one MicroVM each, cloned from a template on demand (four in parallel, then two in series); live dashboard, LangSmith spans |
| [`09_incident_response`](09_incident_response) | 4 per thread | sandbox image, model key, Action Gateway token with DigitalOcean and code-host connections, a public GitHub repository you own, a small billed droplet | An incident team: three investigator MicroVMs and a verifier MicroVM (cloned from a template, paused while a human decides), a remediator that alone can act; the restore of a real or simulated droplet is an Action Gateway approval, the delete is refused; live dashboard |
| [`10_erasure`](10_erasure) | 4 per thread | sandbox image, model key (simulated systems) | An erasure request: four locator MicroVMs (cloned from a template, paused while a human decides) classify a person's records so the models see manifests only; Action Gateway approvals, evidence report, live dashboard |
| [`11_vulnerability_response`](11_vulnerability_response) | 6 | sandbox image, model key (simulated systems) | A security lead and one analyst MicroVM per repository (cloned from a template, paused when done); patches re-tested in fresh MicroVMs; Action Gateway approvals; live dashboard |
| [`12_incident_response_agents`](12_incident_response_agents) | 6 per incident | two images in your registry, everything example 09 needs, a public HTTPS URL for the relay's broker port (a tunnel is enough) | Example 09's incident handled by specialist agents, each a whole deep agent in a MicroVM agent host of its own with its own model: a commander with no action tools delegates to them as async sub-agents; every specialist reaches its Action Gateway actions through the relay's broker (a token per role and incident; no gateway token in any MicroVM); the Remediator proposes the restore, a human decides in the console or on an approvals page (`--web`), and the relay runs it |

The incident of examples 09 and 12 (the droplet, its service, the simulated world) is in [`incident_common`](incident_common).

The minimum is the team limit needed to run the example on real MicroVMs; paused MicroVMs count against it, so delete finished sandboxes. Local mode needs none.

**Try them in local mode.** `uv run examples/tour.py` runs example 11 with its live dashboard and no DigitalOcean account (about 2 minutes): the security lead, one sandbox per analysed repository, a row per repository and every approval and refusal. On macOS the dashboard opens in a window of its own; `--same-terminal` keeps it in this terminal. `--full` adds 01, 03, 06 and 10, `--only 2,3` picks steps, `--verbose` streams everything without the dashboard. Sandboxes and the store live in a temporary directory that is removed at the end; it needs a model, and the agent's shell commands run on your machine.

**Check your setup.** `uv run python scripts/check_setup.py` reads `.env` and prints, per example, `ready`, `ready*` (also needs the setup named) or what is missing, plus the team's MicroVM limit, how many are free and the hourly price of a running MicroVM of your size as the API states it. It is read-only; `--offline` checks the variables only, `--local` checks for local mode, `--deep` creates, uses and deletes one throwaway MicroVM to verify the scopes, image and access token. `uv run python scripts/setup_examples.py` sets up what is missing (`--examples 09,12` for the examples' own steps; without it, the base most examples need: tokens, model key, image); `--dry-run` only checks.

**Where to start.** 01, 03 and 06 need only a sandbox image and a model key. A real run bills MicroVM time and model tokens; this repository gives no cost estimate, so delete what a run leaves (`uv run scripts/delete_all.py` lists your MicroVMs first and touches only the ones you created). Examples 04, 09 and 12 need setup of your own (the last column); the sandbox image is built once with `scripts/build_sandbox_image.py` and pushed to your own container registry. Agent-host images (04, 12) contain this package's code, so use a private registry. Local mode runs the agent's commands on your machine, so keep it to scratch checkouts.

## Setup

```bash
uv sync --all-extras          # from the repository root; installs this package and the example dependencies
cp .env.example .env          # then fill in what you use
```

**Model.** The examples use DigitalOcean Serverless Inference when
`DO_MODEL_ACCESS_KEY` is set (`anthropic-claude-haiku-4.5`; change with
`DO_INFERENCE_MODEL`; examples 07 to 12 also use `anthropic-claude-4.6-sonnet`
for their orchestrating agents, change with `DO_INFERENCE_STRONG_MODEL` in 07 to 11). A model
access key can be scoped to specific models: give yours access to the models you
run, or the calls fail (there is no fallback to another model). Otherwise `EXAMPLES_MODEL` (default
`anthropic:claude-haiku-4-5`) goes to LangChain's `init_chat_model`, with that
provider's own API key, and `EXAMPLES_STRONG_MODEL` (default: `EXAMPLES_MODEL`) is the
orchestrating agents' model in 07 to 12. The `examples` extra installs only `langchain-anthropic`: for another
provider, install its LangChain integration package (for example `langchain-openai`) and set that provider's key.
The agent-host images of 04 and 12 need the package in `images/agent-host/pyproject.toml` before you build them;
example 12 passes `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY` and
`COHERE_API_KEY` into its hosts. Example 02 runs dcode, which takes its model from `-M`, not from these variables.

**MicroVMs.** `DO_MICROVM_TRANSPORT=local` (sandboxes) and
`DO_MICROVM_HOST_MODE=local` (agent hosts, the default) run everything locally
with no DigitalOcean account. For real MicroVMs, see the root
[`README.md`](../README.md#configuration).

Variables set in the shell win over `.env`, so
`DO_MICROVM_TRANSPORT=local uv run examples/...` works without editing it.
