# Examples

What DigitalOcean MicroVMs add as sandboxes for Deep Agents, from a single sandbox to whole teams of agents.

01, 04, 05 and 07 are ports of labs of the [Deep Agents course](https://github.com/langchain-ai/lca-deepagents) that use sandboxes; each one's README names the lab. The others were written for this package: 02, 03 and 06 each show one feature (the `dcode` provider, subagent sandboxes, templates), and 08-12 are larger applications built on several of them.

Each example has its own model setup, run commands and README; the README has the details the table leaves out.


| Example | Min. MicroVMs | Also needed for the real run | Shows |
| --- | --- | --- | --- |
| [`01_quickstart`](01_quickstart) | 1 | sandbox image, model key | A deep agent in a MicroVM; pause and resume keep its files |
| [`02_deepagents_code`](02_deepagents_code) | 1 | sandbox image, `dcode` (Python 3.12+) | `dcode --sandbox microvm` (Deep Agents Code) |
| [`03_subagent_sandboxes`](03_subagent_sandboxes) | 2 per thread | sandbox image, model key | One MicroVM per synchronous subagent, created in parallel |
| [`04_async_subagents`](04_async_subagents) | 2 | agent-host image (`uv`, `crane`, your registry) | Async subagents inside MicroVMs behind an embedded gateway (experimental) |
| [`05_data_analyst`](05_data_analyst) | 1 | sandbox image, model key | SQL and charts in a MicroVM; the paused sandbox keeps the database |
| [`06_warm_start`](06_warm_start) | 1 | sandbox image, model key | Sandbox templates: checkpoint once, clone in seconds |
| [`07_sales_assistant`](07_sales_assistant) | 2 (1 per thread once the template exists) | sandbox image, model key; optional Tavily key | One MicroVM per conversation thread, paused between turns |
| [`08_board_pack`](08_board_pack) | 7 (6 once the template exists) | sandbox image, model key | An orchestrator and six specialists, one MicroVM each; live dashboard |
| [`09_incident_response`](09_incident_response) | 5 (4 per thread once the template exists) | sandbox image, model key, Action Gateway token, a public GitHub repository you own, a small billed droplet | An incident team of MicroVMs; the restore is an Action Gateway approval; live dashboard |
| [`10_erasure`](10_erasure) | 5 (4 per thread once the template exists) | sandbox image, model key | An erasure request: four locator MicroVMs; Action Gateway approvals; evidence report; live dashboard |
| [`11_vulnerability_response`](11_vulnerability_response) | 6 | sandbox image, model key | A security lead and one analyst MicroVM per repository; Action Gateway approvals; live dashboard |
| [`12_incident_response_agents`](12_incident_response_agents) | 6 per incident | two images in your registry, everything example 09 needs, a public HTTPS URL for the relay's broker port (a tunnel is enough) | Example 09's incident handled by specialist agents, each in a MicroVM agent host of its own; a human decides in the console or on an approvals page |

The incident of examples 09 and 12 (the droplet, its service, the simulated world) is in [`incident_common`](incident_common).

The minimum is the team limit needed to run the example on real MicroVMs: the most MicroVMs a run holds at once, including the temporary MicroVM that builds a template (the number in brackets applies once the template exists). Paused MicroVMs count against it, so delete finished sandboxes. Local mode needs none.

## Try them in local mode

No DigitalOcean account; needs a model (see Model below). The agent's shell commands run on your machine.

```bash
git clone https://github.com/digitalocean-labs/langchain-digitalocean-microvm
cd langchain-digitalocean-microvm && uv sync --all-extras
cp .env.example .env          # set DO_MODEL_ACCESS_KEY, or EXAMPLES_MODEL and that provider's key
uv run examples/tour.py
```

The tour runs example 11 with its live dashboard (about 2 minutes): the security lead, one sandbox per analysed repository, a row per repository and every approval and refusal. On macOS the dashboard opens in a window of its own. Sandboxes and the store live in a temporary directory that is removed at the end.

```bash
uv run examples/tour.py --same-terminal   # keep the dashboard in this terminal
uv run examples/tour.py --full            # all five steps: 01, 03, 06, 10, 11 (about 10 minutes)
uv run examples/tour.py --only 2,3        # pick steps by number (1 to 5)
uv run examples/tour.py --verbose         # stream everything, no dashboard
```

## Real sessions

A real run bills MicroVM time and model tokens; delete what a run leaves. Paused MicroVMs count against the team limit.

Check what is ready or missing, per example. The check is read-only:

```bash
uv run python scripts/check_setup.py             # reads .env: ready, ready* (also needs the setup named) or what is missing
uv run python scripts/check_setup.py --offline   # variables only
uv run python scripts/check_setup.py --local     # local mode
uv run python scripts/check_setup.py --deep      # creates, uses and deletes one throwaway MicroVM: checks token, scopes, image, access token and model key
```

The output also shows the team's MicroVM limit, how many are free and the hourly price of a running MicroVM of your size as the API states it.

Set up what is missing: tokens (typed in, never echoed, saved to `.env`), the model key, the sandbox image and, for the examples that need them, the Action Gateway actor, the demo repository, the droplet and the agent hosts' repositories:

```bash
uv run python scripts/setup_examples.py                      # the base for 01, 03, 05, 06, 08, 10 and 11
uv run python scripts/setup_examples.py --examples 09,12     # the examples' own steps
uv run python scripts/setup_examples.py --examples 09,12 --dry-run   # read-only checks
```

Run an example and clean up:

```bash
uv run examples/01_quickstart/quickstart.py    # each example's README has its commands
uv run scripts/delete_all.py                   # lists the MicroVMs you created; deletes nothing
uv run scripts/delete_all.py --yes             # deletes them (asks first)
```

01, 03 and 06 need only a sandbox image and a model key. Examples 04, 09 and 12 need setup of your own (the last column of the table). Build the sandbox image once with `scripts/build_sandbox_image.py` and push it to your own container registry. Agent-host images (04, 12) contain this package's code, so use a private registry. Local mode runs the agent's commands on your machine, so keep it to scratch checkouts.

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
