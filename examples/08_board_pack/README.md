# 08 · Board pack: a team of Deep Agents, one MicroVM per specialist

**Requires** (real MicroVMs): a team limit of at least 6 MicroVMs, one per specialist (four analysts, auditor, publisher), plus 1 temporary MicroVM while the template is built. The orchestrator runs in your process.

**Quick run** (local mode: no DigitalOcean account, and the agent's commands run on this machine; needs a model, see [`../README.md`](../README.md)):

```bash
DO_MICROVM_TRANSPORT=local uv run --with matplotlib examples/08_board_pack/run_script.py
```

The live dashboard and real MicroVMs: see Run below.

## Overview

**The scenario.** A **board pack** is the document a company's leadership sends its board of
directors before a board meeting: the headline numbers, how each part of the business performed,
what changed and why, and the questions the board should discuss. Analysts usually spend days on
it, by region, before someone checks the figures and someone else assembles the deck. Here a team
of agents builds the FY2025 board pack of Chinook, a fictional online music store, from its sales
database in about two minutes.

**The team.** An **orchestrator** (the main deep agent) plans the work and delegates it; it cannot
run commands itself. Four regional **analysts** query the database and draw charts, each for its
own region. An **auditor** recomputes every headline figure from its own copy of the data, and a
**publisher** assembles the final HTML document. You can then ask follow-up questions in the same
conversation.

**What it teaches about Deep Agents.**
- **Subagents and the `task` tool.** The orchestrator hands each job to a specialist subagent with
  its own prompt and tools, and gets back only the result, which keeps its own context small.
- **Parallel, then sequential delegation.** The four analysts work at the same time; the auditor
  and the publisher wait for them.
- **A backend per subagent.** Normally a subagent works in the main agent's backend. Here each
  specialist's filesystem and shell are a MicroVM of its own, so a mistake (even `rm -rf /`) in one
  cannot touch another. Files move between them only when the orchestrator copies them.
- **Threads.** A follow-up question continues the same conversation and finds each specialist's
  files where it left them.

**What MicroVMs add.** Each specialist's MicroVM is cloned from a prepared **template** (database,
helper scripts and Python packages already installed) in seconds, only when that specialist gets
work, and pauses with its files intact when the run ends. A live dashboard shows every MicroVM
starting, working and pausing.

## How it works

Every specialist is a Deep Agents subagent whose **filesystem and shell are a DigitalOcean MicroVM
of its own**, cloned from one template the moment it gets work:

```
 you ── Studio / agent-chat-ui / run_script.py
          │
 langgraph dev ── board-pack-orchestrator (Sonnet, no shell) ─────────── LangSmith traces
          │   task ×4, in parallel                     handoff_files (VM → VM)
          ├──► analyst-north-america ─► MicroVM ┐
          ├──► analyst-europe        ─► MicroVM ├─ cloned on demand from the "board-pack" template:
          ├──► analyst-latin-america ─► MicroVM │  chinook.db, SQL/chart/render helpers, matplotlib
          ├──► analyst-asia-pacific  ─► MicroVM ┘
          │   then
          ├──► auditor   ─► MicroVM   re-computes every headline figure from its own database copy
          │   then
          └──► publisher ─► MicroVM   renders /outputs/board-pack.html ─► export_board_pack ─► browser

 dashboard.py ── polls the MicroVMs API + outputs/events.jsonl: one tile per VM, live
```

| | What it shows |
| --- | --- |
| **One VM per agent** | Each specialist runs its SQL, charts and scripts in its own VM with its own copy of the data. The orchestrator cannot run anything; files move between VMs only through `handoff_files`. `chaos.py` wipes one VM with `rm -rf /` mid-demo: nothing else notices. |
| **On demand, in parallel and in series** | `asubagent_sandboxes(..., on_demand=True)`: no VM exists until its specialist acts. The four analysts' VMs start together when the orchestrator fans out; the auditor's and publisher's only when their turn comes. A follow-up about one region wakes one VM. |
| **Templates** | `templates={...}`: every VM is a clone of a checkpoint with the database, the helper scripts and matplotlib already in place (ready in seconds) instead of boot + upload + `pip install`. A `TemplateCache` builds it once per process and rebuilds it if its checkpoint was deleted while the server runs. |
| **Pause / resume** | When a run ends, the VMs it used pause with memory and files intact; the next question resumes them in a couple of seconds. |
| **Observability** | LangSmith shows the orchestrator, each subagent as a nested run, every tool call and a `MicroVM clone` / `MicroVM resume` span where a VM started. |


## Files

| File | |
| --- | --- |
| `agent.py` | The team (prompts, tools per specialist), the template, the per-thread graph factory `make_graph` |
| `tools.py` | Specialist tools (run helper scripts in the VM) and the orchestrator's `handoff_files` / `export_board_pack` |
| `sandbox/` | Helper scripts seeded into the template as `/tools/`: read-only SQL, charts, the HTML render |
| `events.py` | `DemoLifecycle`: logs every clone/resume/pause to `outputs/events.jsonl` and LangSmith |
| `dashboard.py` | The live MicroVM dashboard for the second screen |
| `run_script.py` | Headless driver: streams every tool call, labelled by agent (rehearsals, recordings, fallback) |
| `prepare.py` | Pre-flight: configuration, template, model warm-up, `--clean` for an empty dashboard and a fresh dev server |
| `chaos.py` | The isolation moment (real MicroVMs only) |

The Chinook database is read from [`../07_sales_assistant/data/`](../07_sales_assistant/data)
([lerocha/chinook-database](https://github.com/lerocha/chinook-database), MIT).

## Run (local mode, no account)

From the repository root, after `uv sync --all-extras`, with a model configured (see
[`../README.md`](../README.md)). Sandboxes are local directories and the helpers run with this process's Python, so charts need
matplotlib; `chaos.py` refuses to run in local mode.

```bash
DO_MICROVM_TRANSPORT=local uv run --with matplotlib examples/08_board_pack/run_script.py
```

## Run (real MicroVMs)

From the repository root, after `uv sync --all-extras`, with these in `.env` (the image is the root sandbox image from
`scripts/build_sandbox_image.py`):

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
uv run python scripts/check_setup.py --examples 08   # ready, ready* (also needs the setup named) or what is missing
```

Then:

```bash
# in terminal #1 run (pre-flight, template, empty dashboard; then the VM tiles):
uv run examples/08_board_pack/prepare.py --clean
uv run examples/08_board_pack/dashboard.py

# in terminal #2 run (board pack + a follow-up):
uv run examples/08_board_pack/run_script.py
uv run examples/08_board_pack/run_script.py --delete      # or: delete the thread's VMs at the end
```

Interactive, with LangGraph Studio (opens in the browser) or agent-chat-ui:

```bash
cd examples/08_board_pack && uv run langgraph dev           # graph id: board_pack
```

- **Studio**: `langgraph dev` prints the Studio URL
  (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`); it draws the graph and
  shows each subagent's steps live.
- **agent-chat-ui**: open <https://agentchat.vercel.app> (or run it locally:
  `git clone https://github.com/langchain-ai/agent-chat-ui && cd agent-chat-ui && pnpm install && pnpm dev`,
  then <http://localhost:3000>), Deployment URL `http://localhost:2024`, Assistant / Graph ID
  `board_pack`, no API key for a local server.

Both talk to the same server at the same time, so you can pick one on the day.

**LangSmith**: set the variables in `.env`:

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=board-pack-demo
```

- Open <https://smith.langchain.com>, sign in and open the project named in `LANGSMITH_PROJECT` (`default` when it is unset).
- A board pack run is one trace: the orchestrator, four parallel `task` runs named after the analysts, then the auditor and the publisher, each with its tools and a `MicroVM clone` span.

**Opening the result**: `export_board_pack` copies the HTML to
`outputs/<thread>/board-pack.html` and opens it in the browser (`BOARD_PACK_OPEN=0` to turn that off).

**Clean-up**: `run_script.py --delete`, `prepare.py --clean` (keeps the template), or
`scripts/delete_all.py --yes` (everything else; add `--include-templates --all-owners` for the template too).

`langgraph dev` keeps unfinished runs in `.langgraph_api/` and re-queues them when it starts, so a run stopped
mid-way would recreate its VMs after a clean-up and can block new requests. `prepare.py --clean` therefore also
deletes `.langgraph_api/` (saved threads included), but only when no `langgraph dev` is running: stop it first.
The other two commands delete MicroVMs only.

## Where the time goes

Orchestrator and publisher run on `anthropic-claude-4.6-sonnet`, analysts and auditor on `anthropic-claude-haiku-4.5`,
on DigitalOcean Serverless Inference.

- **Model time dominates.** The board pack runs in phases: the orchestrator's fan-out (4 `task` calls), the analysts,
  then the auditor, then the publisher, which exports the pack. The follow-up on one region resumes that region's VM and
  leaves the other five untouched; its length depends on how many queries the analyst decides to run.
- **Each specialist's VM costs one clone per conversation** (a few seconds, and they are cloned in parallel), and a pause
  when the run ends.
- **Every tool call that touches a VM** is a request to the sandbox server over the VM's HTTPS endpoint, a fraction of a
  second; the whole session sent 6 exec requests (one per VM, to set its server's token) and none while the agents ran.
  With `DO_MICROVM_TRANSPORT=exec` a clone is faster, but every such tool call is a round trip to the MicroVMs exec API
  (126 in one board pack), which is slower per call.
- **Local mode** (no VMs) takes about as long as the VM run, which confirms the model is the bottleneck.

`langgraph dev` rejects blocking calls (disk I/O, sleeps) on the event loop; `run_script.py` does
not, so rehearse once under the server too. `tests/unit_tests/test_example_board_pack.py` runs the
tools with the same guard (blockbuster).
