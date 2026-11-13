# AGENTS.md

Guidance for coding agents working in this repository. Humans: see [`CONTRIBUTING.md`](CONTRIBUTING.md),
which this file summarizes and does not replace.

This package gives LangChain Deep Agents DigitalOcean MicroVMs as sandboxes. Read
[`README.md`](README.md) first, then [`docs/design.md`](docs/design.md) (how it works) and
[`docs/reference.md`](docs/reference.md) (modules and service limits).

## Commands

```bash
uv sync --all-extras
uv run pytest              # unit tests in local mode; no DigitalOcean account needed
uv run ruff check .
uv run pyright             # src/ only
```

Do your work in local mode. `uv run pytest -m e2e` and `-m slow` are local and safe to run;
`-m live` is not (see below).

## Rules

- **Do not run live tests, examples against a real account, or `scripts/delete_all.py` unless the
  person you are working for asked you to.** They create real MicroVMs that cost money and count
  against a team limit, and `delete_all.py` deletes the MicroVMs and checkpoints you created with this package in one namespace
  of the team (`--all-owners`: other users' in that namespace too; `--all-namespaces`: everyone's).
- Never read, print or commit `.env` or any credential. `.env.example` lists the variables.
- In local mode (`DO_MICROVM_TRANSPORT=local`) an agent's commands run on this machine. Do not let an
  agent in local mode run `pip install` or anything destructive.
- Public API changes (anything exported from `langchain_digitalocean_microvm`) beyond small additions:
  open an issue first.
- One change per pull request, with tests. Add a `CHANGELOG.md` line for user-visible changes.

## Things the code does not tell you

- All DigitalOcean API specifics live in `client.py`; everything above it works in local mode.
  Clocks, clients and transports are injected, so tests need no network.
- The `deepagents` range is `>=0.7.0,<0.8.0`. `subagents.py` uses private deepagents hooks, guarded
  by a contract test in `tests/unit_tests/test_subagents.py`. Run it if you touch the range.
- `glob` can return absolute paths, depending on the deepagents version; handle both.
- Known failures of LangChain's standard sandbox suite are marked xfail, with the reason, in
  `tests/standard_suite_gaps.py`. Do not add an xfail without a reason.
- Changing `images/agent-host/` means the agent-host image must be rebuilt (see its `README.md`);
  tests in local mode do not cover that.
- Each example under `examples/` is self-contained and has its own `make_model()`. If you change how
  many MicroVMs an example creates, update the "Requires" line in its README and the table in
  [`examples/README.md`](examples/README.md).

## Documentation style

Write for a reader who has just cloned the repository. Describe service limits as current behaviour,
not as a history of what changed or failed, and leave dated run results out.
