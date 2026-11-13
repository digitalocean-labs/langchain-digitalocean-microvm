# 05 · Data analyst: SQL and charts in a MicroVM

**Requires** (real MicroVMs): a team limit of at least 1 MicroVM.

> Port of two lessons of the
> [Deep Agents course](https://github.com/langchain-ai/lca-deepagents):
> `python/m2/m2.3_sales_agent.py` (Chinook in a LangSmith sandbox, a donut chart)
> and `python/m2/m2.4_interpreter_agent.py` (chained SQL questions in an
> in-process QuickJS interpreter).

A deep agent analyses the Chinook music-store database inside a MicroVM: it
writes Python (`sqlite3`, `matplotlib`), runs it with `execute`, and saves
charts under `/outputs/`, which the script copies to `outputs/` here. The
database is uploaded when the sandbox is created; matplotlib is installed by
the agent the first time it needs it. Between runs the sandbox is **paused**,
so the next run resumes it with the database, matplotlib and earlier charts in
place.

## Run

From the repository root, with a model configured (see [`../README.md`](../README.md))
and `DO_MICROVM_API_TOKEN`, `DO_MICROVM_ACCESS_TOKEN` + `DO_MICROVM_IMAGE`:

```bash
uv run examples/05_data_analyst/analyst.py                  # chained questions + donut chart
uv run examples/05_data_analyst/analyst.py --ask "Top 5 customers by spend, as a bar chart"
uv run examples/05_data_analyst/analyst.py --template       # a new sandbox cloned from 06_warm_start's template (after --delete)
uv run examples/05_data_analyst/analyst.py --delete
```

The image must let the agent `pip install` (outbound network, a writable
site-packages): [`scripts/build_sandbox_image.py`](../../scripts/build_sandbox_image.py)'s
image does. Local mode (`DO_MICROVM_TRANSPORT=local`) works but runs the agent's
commands, `pip install` included, on this machine.

## Where the time goes

- The first run creates the MicroVM and uploads the 1 MB database; that boot is its longest wait.
- A later run resumes the paused sandbox in a couple of seconds, with matplotlib, the database and the chart still there.
- `--template` after `--delete` clones 06's template (database and matplotlib in place) instead of booting.
- Each question is dominated by the model's turns; the first chart also pays for `pip install matplotlib` unless a template has it.

The Chinook database (`data/chinook.db`) is
[lerocha/chinook-database](https://github.com/lerocha/chinook-database), MIT license.
