# 02 · Deep Agents Code (`dcode`) in a MicroVM

**Requires** (real MicroVMs): a team limit of at least 1 MicroVM.

**The use case.** [Deep Agents Code](https://docs.langchain.com/oss/deepagents/code/remote-sandboxes) (`dcode`) is a
coding agent for the terminal, built on Deep Agents. By default it edits files and runs commands on your machine. With
this package its files and shell live in a MicroVM instead, so it can install packages and run code without touching
your computer, and a sandbox id lets you come back later to the same, paused machine.

Deep Agents Code discovers third-party sandbox providers through the
`deepagents_code.sandbox_providers` entry point. This package registers
`microvm`, so installing it next to `deepagents-code` is all it takes:

```bash
pip install deepagents-code "langchain-digitalocean-microvm @ git+https://github.com/digitalocean-labs/langchain-digitalocean-microvm"
export DO_MICROVM_API_TOKEN=...      # microvm:create/read/update/delete
export DO_MICROVM_ACCESS_TOKEN=...   # microvm:access only (the sandbox's HTTPS endpoint)
export DO_MICROVM_IMAGE=...          # see ../01_quickstart

dcode --sandbox microvm                       # a new MicroVM for this session
dcode --sandbox microvm --sandbox-id <id>     # reuse, resume or create sandbox <id>
dcode --sandbox microvm -n "run the tests and summarize failures"   # non-interactive
```

From a checkout of this repository: `uv run dcode --sandbox microvm`; `uv run python scripts/setup_examples.py --examples 02`
sets up the tokens, the image and the `dcode` CLI step by step (`--dry-run` only checks).
`DO_MICROVM_TRANSPORT=local` runs the sandbox as a local directory instead.

To use DigitalOcean Serverless Inference as dcode's model, point its Anthropic
client at it:

```bash
ANTHROPIC_API_KEY=$DO_MODEL_ACCESS_KEY ANTHROPIC_BASE_URL=https://inference.do-ai.run \
  dcode --sandbox microvm -M anthropic:anthropic-claude-haiku-4.5 -n "uname -a"
```

> **Clean-up.** dcode deletes a sandbox it created from an `atexit` hook in its
> server subprocess. With `-n`, that hook does not
> run: the MicroVM stays `running` until it pauses itself after
> `DO_MICROVM_IDLE_TIMEOUT`, and paused MicroVMs still count against your team's
> limit. Reuse one sandbox with `--sandbox-id <id>` (dcode never deletes those),
> and remove leftovers with `uv run scripts/delete_all.py --yes`.

The provider (`langchain_digitalocean_microvm.integrations.deepagents_code`)
accepts dcode's sandbox id or the MicroVM id, and supports `seed_files` and
`tags`. It does not support snapshot names; MicroVM checkpoints are not exposed
through dcode yet.
