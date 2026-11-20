# 05 · Data analyst: SQL and charts in a MicroVM

**Requires** (real MicroVMs): a team limit of at least 1 MicroVM.

> Port of two lessons of the
> [Deep Agents course](https://github.com/langchain-ai/lca-deepagents):
> `python/m2/m2.3_sales_agent.py` (Chinook in a LangSmith sandbox, a donut chart)
> and `python/m2/m2.4_interpreter_agent.py` (chained SQL questions in an
> in-process QuickJS interpreter).

**Quick run** (a real MicroVM; needs a model, see [`../README.md`](../README.md)):

```bash
uv run python scripts/setup_examples.py
uv run examples/05_data_analyst/analyst.py
```

Setup and options: see Run below.

A deep agent analyses the Chinook music-store database inside a MicroVM: it
writes Python (`sqlite3`, `matplotlib`), runs it with `execute`, and saves
charts under `/outputs/`, which the script copies to `outputs/` here. The
database is uploaded when the sandbox is created; matplotlib is installed by
the agent the first time it needs it. Between runs the sandbox is **paused**,
so the next run resumes it with the database, matplotlib and earlier charts in
place.

## Run (local mode)

Local mode runs the agent's commands, `pip install` included, on this machine: use a scratch checkout.

```bash
DO_MICROVM_TRANSPORT=local uv run examples/05_data_analyst/analyst.py
```

## Run (real MicroVMs)

From the repository root, with these in `.env` (the model: see [`../README.md`](../README.md)):

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
uv run python scripts/check_setup.py --examples 05   # ready, ready* (also needs the setup named) or what is missing
```

Then:

```bash
uv run examples/05_data_analyst/analyst.py                  # chained questions + donut chart
uv run examples/05_data_analyst/analyst.py --ask "Top 5 customers by spend, as a bar chart"
uv run examples/05_data_analyst/analyst.py --template       # a new sandbox cloned from 06_warm_start's template (after --delete)
uv run examples/05_data_analyst/analyst.py --delete
```

The image must let the agent `pip install` (outbound network, a writable
site-packages): [`scripts/build_sandbox_image.py`](../../scripts/build_sandbox_image.py)'s
image does.

## LangSmith traces

Optional. Set these in `.env` and run the example as above:

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=data-analyst      # unset: the project is `default`
```

To read the traces, open <https://smith.langchain.com>, sign in and open the project named in `LANGSMITH_PROJECT`. Each question is a trace with the agent's model calls and tool calls.

## Where the time goes

- The first run creates the MicroVM and uploads the 1 MB database; that boot is its longest wait.
- A later run resumes the paused sandbox in a couple of seconds, with matplotlib, the database and the chart still there.
- `--template` after `--delete` clones 06's template (database and matplotlib in place) instead of booting.
- Each question is dominated by the model's turns; the first chart also pays for `pip install matplotlib` unless a template has it.

The Chinook database (`data/chinook.db`) is
[lerocha/chinook-database](https://github.com/lerocha/chinook-database), MIT license.
