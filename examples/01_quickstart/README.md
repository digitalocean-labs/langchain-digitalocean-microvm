# 01 · Quickstart: a deep agent in a MicroVM, across a pause/resume

**Requires** (real MicroVMs): a team limit of at least 1 MicroVM.

> Port of the sandbox lesson of the
> [Deep Agents course](https://github.com/langchain-ai/lca-deepagents)
> (`python/m2/m2.3_sandbox_agent.py`), which runs a LangSmith sandbox through a
> create → use → delete cycle.

**Quick run** (local mode: no DigitalOcean account, and the agent's commands run on this machine; needs a model key):

```bash
uv sync --all-extras    # from the repository root
cp -n .env.example .env # then set a model key in .env, one of:
                        #   DO_MODEL_ACCESS_KEY: DigitalOcean Serverless Inference, with access to
                        #     anthropic-claude-haiku-4.5
                        #   ANTHROPIC_API_KEY: Anthropic (other providers: EXAMPLES_MODEL, see ../README.md)
DO_MICROVM_TRANSPORT=local uv run examples/01_quickstart/quickstart.py
```

Real MicroVMs: see Run below.

`quickstart.py` gives a deep agent a MicroVM as its filesystem and shell
(`MicroVMSandbox`, a deepagents `BaseSandbox`), then shows what MicroVMs add:
the sandbox is **paused** (memory, files and processes kept) and **resumed**,
and the agent finds both the seeded file and the one it wrote.

```python
from deepagents import create_deep_agent
from langchain_digitalocean_microvm import MicroVMLifecycle

lifecycle = MicroVMLifecycle()                       # DO_MICROVM_* from the environment
backend = lifecycle.get_or_create(sandbox_id="quickstart", seed_files=[("/notes.md", b"hi\n")])
agent = create_deep_agent(model=model, backend=backend)
...
lifecycle.pause(backend)                             # later: get_or_create() resumes it
```

## Run (local mode, no account)

From the repository root, with a model configured (see [`../README.md`](../README.md)). The sandbox is a local directory and the agent's commands run on this machine:

```bash
DO_MICROVM_TRANSPORT=local uv run examples/01_quickstart/quickstart.py
```

## Run (real MicroVMs)

A real MicroVM needs these in `.env`; the image is a `linux/amd64` image that keeps running and ships `sh`, coreutils and
`python3` ([`scripts/build_sandbox_image.py`](../../scripts/build_sandbox_image.py) builds one):

```bash
# .env
DO_MICROVM_API_TOKEN=...      # microvm:create/read/update/delete
DO_MICROVM_ACCESS_TOKEN=...   # microvm:access only (the sandboxes' HTTPS endpoints)
DO_MICROVM_IMAGE=registry.digitalocean.com/<registry>/microvm-sandbox:<tag>
DO_MODEL_ACCESS_KEY=...       # or EXAMPLES_MODEL plus that provider's key
```

To set them up step by step:

```bash
uv run python scripts/setup_examples.py
uv run python scripts/setup_examples.py --dry-run   # checks only
uv run python scripts/check_setup.py --examples 01   # ready, ready* (also needs the setup named) or what is missing
```

Then:

```bash
uv run examples/01_quickstart/quickstart.py            # a real MicroVM
uv run examples/01_quickstart/quickstart.py --delete   # delete the sandbox
uv run examples/01_quickstart/pause_all.py             # pause every running MicroVM you created
```

## LangSmith traces

Optional. Set these in `.env` and run the example as above:

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=quickstart      # unset: the project is `default`
```

To read the traces, open <https://smith.langchain.com>, sign in and open the project named in `LANGSMITH_PROJECT`. Each run of the agent is a trace with its model calls and tool calls.
