# Contributing

Thanks for helping. This package gives LangChain Deep Agents DigitalOcean MicroVMs as sandboxes;
see [`README.md`](README.md) for what it does and [`docs/design.md`](docs/design.md) for how.
If you work with a coding agent, point it at [`AGENTS.md`](AGENTS.md).

## Setup

```bash
uv sync --all-extras
uv run pytest            # unit tests, local mode, about 1-2 minutes
uv run ruff check .
```

Local mode (`DO_MICROVM_TRANSPORT=local`, `DO_MICROVM_HOST_MODE=local`) runs everything locally with no
DigitalOcean account, so most changes need nothing else.

## Tests

| Command | What it runs |
| --- | --- |
| `uv run pytest` | Unit tests in local mode |
| `uv run pytest -m e2e` | End-to-end tests against real local `langgraph dev` processes |
| `uv run pytest -m slow` | LangChain's standard sandbox suite, offline, about 5 minutes |
| `uv run pytest -m live` | The standard suite on a real MicroVM |
| `uv run ruff check .`, `uv run pyright` | Lint and type check (`src/`) |
| `scripts/grep_gate.sh` | Wording gate: no dates, release-stage words or planning labels; patterns in an untracked `.grep_gate_extra` add to it |
| `gitleaks git --log-opts="--all" --redact .` | Secret scan of the history (version pinned in `.github/workflows/ci.yml`; config in `.gitleaks.toml`) |

Live runs need DigitalOcean credentials in a `.env` at the repository root (copy `.env.example`;
`.env` is gitignored) and create real MicroVMs. Check that the team has room first, and clean up
afterwards:

```bash
uv run scripts/delete_all.py         # dry run: lists what it would delete
uv run scripts/delete_all.py --yes   # deletes, after asking you to confirm
```

`delete_all.py` deletes the MicroVMs and checkpoints you created with this package in your namespace (`DO_MICROVM_NAMESPACE`;
`--all-owners`: other users' in that namespace too; `--all-namespaces`: every user's on the team);
sandbox templates are shared and are kept unless `--include-templates` (which also needs `--all-owners`). Paused MicroVMs count against the
team's limit.

In local mode an agent's commands run on your machine: do not let an agent in local mode run
`pip install`.

`scripts/check.sh` runs lint, types, the wording gate, the secret scan (if `gitleaks` is installed) and the
unit tests in one go; `--quick` skips the tests.

Cases from the standard suite that are known to fail upstream are marked xfail in
`tests/standard_suite_gaps.py`, with the reason.

## Code

- All DigitalOcean API specifics live in `client.py`; everything above it works in local mode.
- Frozen dataclasses; a sync API with `asyncio.to_thread` twins; clocks, clients and transports
  injected so tests need no network.
- The `deepagents` range is `>=0.7.0,<0.8.0`. `subagents.py` uses private deepagents hooks, guarded
  by a contract test in `tests/unit_tests/test_subagents.py`: if you widen the range, run it.
- Public API changes (anything exported from `langchain_digitalocean_microvm`) beyond small
  additions: open an issue first.

## Examples

Each example under `examples/` is self-contained: its own `make_model()`, the repository `.env`
loaded with python-dotenv (variables set in the shell win), and a README that states the
minimum number of MicroVMs it needs and describes where the time goes (model turns, boot, clone,
resume) instead of quoting point timings, which go stale. A port of a
[Deep Agents course](https://github.com/langchain-ai/lca-deepagents) lab names the lab it ports.
Keep the "Requires" line in the README and the table in [`examples/README.md`](examples/README.md)
current when you change how many MicroVMs an example creates.

## Documentation

Write for a reader who has just cloned the repository: what the package does, what it needs to
run, how it behaves. Describe service limits as current behaviour ("the API has no update endpoint,
so X is fixed at create"), not as a history of what changed or failed, and leave dated run
results out.

## Pull requests

No write access is needed: report bugs and ideas as issues (templates are provided), and send code
by forking the repository and opening a pull request from your fork. Contributions are licensed
under the Apache License 2.0, the project's licence (section 5 of [LICENSE](LICENSE)). Changes reach `main` only
through a pull request reviewed by a maintainer.

```bash
# fork the repository in the web interface, then:
git clone <your fork> && cd langchain-digitalocean-microvm
git switch -c my-change
uv sync --all-extras
# ... change, test, commit ...
git push -u origin my-change     # then open the pull request against `main`
```

CI (`.github/workflows/ci.yml`) runs on every pull request and on each push to `main`: lint, types, the
unit, end-to-end and standard-suite tests on Python 3.11 and 3.13, and a hygiene job with the wording gate
and the secret scan. Run `uv run ruff check .` and `uv run pytest` before you push.

- One change per pull request, with tests (local mode where possible).
- `uv run ruff check .` and `uv run pytest` pass.
- Add a line to `CHANGELOG.md` for user-visible changes, under an "Unreleased" heading at the top (add the heading if it is missing).
