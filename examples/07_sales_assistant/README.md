# 07 · Sales assistant: one MicroVM per conversation thread

**Requires** (real MicroVMs): a team limit of at least 1 MicroVM per conversation thread, plus 1 temporary MicroVM while the template is built.

> Port of the capstone of the
> [Deep Agents course](https://github.com/langchain-ai/lca-deepagents):
> [`python/m5/sales_assistant_sandbox`](https://github.com/langchain-ai/lca-deepagents/tree/main/python/m5/sales_assistant_sandbox)
> (the sales assistant with a per-thread sandbox), with a DigitalOcean MicroVM
> as each thread's sandbox.

**Quick run** (a real MicroVM; needs a model, see [`../README.md`](../README.md)):

```bash
uv run python scripts/setup_examples.py --examples 07
uv run examples/07_sales_assistant/run_script.py
```

Setup and options: see Run below.

The Chinook Sales Assistant works Jane Peacock's book of business with skills (territory
report, RFQ quote, weekly newsletter), specialist subagents (database analyst, inbox manager,
quote reviewer), human approvals for the two writes (a new customer, a saved draft), and a
mock mail MCP server. Its **entire filesystem is a MicroVM**: `/AGENTS.md`, `/skills/`,
`/data/chinook.db`, matplotlib, and everything the agent writes under `/outputs/`. The model
and the tools run in the agent process; only files and commands go to the MicroVM.

```
agent process (langgraph dev or run_script.py) ──────────────────────────────┐
  assistant ── task ──► chinook-analyst   SQL tools ─ execute ──────┐         │
            ── task ──► inbox-manager     mail MCP tools (host) ──► mail_server (:5002)
            ── task ──► quote-reviewer                              │         │
            ── execute / write_file / skills / memory ──────────────┤         │
                                                                    ▼         │
thread T:  MicroVM sales-T  /AGENTS.md /skills /data/chinook.db /tools /outputs
└──────────────────────────────────────────────────────────────────────────────┘
```

## Per run

`agent.py` is a per-thread graph factory (`make_graph`). `langgraph dev` enters it for every
run of a thread, and `threads.asandbox(thread_id)` (`ThreadSandboxes`):

1. **gets or creates `sales-<thread id>`**: a thread's first run clones the template, a later
   run resumes the paused MicroVM with its files and charts where they were left
   (`MicroVMLifecycle.get_or_create`; at most four threads do this at once per process);
2. runs the agent against it (a run that stops at an approval also ends here, and the answer
   starts the next run);
3. **pauses it** when the run ends.

With real MicroVMs each run also records the thread's use in the gateway store
(`gateway.sqlite`, or `gateway-<namespace>.sqlite` with `DO_MICROVM_NAMESPACE`, in the package's state directory, docs/reference.md "Where state lives"), so
[`scripts/gc_microvms.py --sandbox-ttl 24h`](../../scripts/gc_microvms.py) deletes threads
nobody has used for a day.

Introspection calls (Studio drawing the graph) get the same graph on an in-memory backend,
without a MicroVM.

### The template

Everything a thread needs is prepared once, as a sandbox template
(see [`06_warm_start`](../06_warm_start)): the seeded files plus
`pip install matplotlib`, checkpointed. The first run in a process builds it or finds
its checkpoint; a new thread then clones it in seconds with matplotlib and the files in
place, instead of booting, uploading and installing (06 measures the gap). Editing
`AGENTS.md`, a skill, the database or `sandbox/chinook_sql.py` changes the template's name, so
the next run builds a new one.

The template is cached in the server process (`TemplateCache`). If its checkpoint is
deleted while the server runs (`scripts/delete_all.py --include-templates --all-owners` deletes it), the API refuses the next
clone with a `404`; the cache then rebuilds the template once and retries that thread, so the
new thread waits for the rebuild instead of failing, and threads that fail at
the same moment share one rebuild.

## What changed from the course

| Course (LangSmith sandbox) | Here (MicroVM) |
| --- | --- |
| `_lookup_or_create`: list sandboxes by name, restart stopped ones, wait out transitional ones | `MicroVMLifecycle.get_or_create`: one call, all of that included |
| Seeds `/AGENTS.md` and `/skills` into every new sandbox (`upload_files`) | Seeds are in the template; nothing is uploaded per thread |
| `pip install matplotlib` by the agent, in every sandbox | matplotlib is in the template |
| SQL tools open `chinook.db` on the agent server's disk | `tools/sql.py` runs `sandbox/chinook_sql.py` **in the thread's MicroVM**: the database is `/data/chinook.db`, one copy per thread (a customer added in one thread does not exist in another) |
| `idle_ttl_seconds` / `delete_after_stop_seconds` bound the cost | Paused when the run ends; `DO_MICROVM_IDLE_TIMEOUT` pauses a MicroVM that is left running. Paused MicroVMs count against the team's limit: `scripts/gc_microvms.py --sandbox-ttl` deletes unused threads |
| `stop_sandboxes.py` on shutdown | `run_script.py --delete`, [`01_quickstart/pause_all.py`](../01_quickstart/pause_all.py), [`scripts/delete_all.py`](../../scripts/delete_all.py) (`--include-templates --all-owners --yes` also deletes the template) |
| `langchain-mcp-adapters` loads the mail tools | `tools/mail.py` bridges them with FastMCP's client: `langchain-mcp-adapters` needs `mcp<2`, the `deepagents-code` of this repo's test group needs `mcp>=2` |
| Mail server and chat UI started by `start.sh` | Start the mail server yourself (below); `run_script.py` starts it if nothing listens |

Unchanged: the skills (except that matplotlib is already installed), `AGENTS.md`, the
subagents and their approvals, the mock mailbox, the Tavily newsletter subagent (below).

The agent also has `execute`, so it *could* edit the database in its own sandbox: the SQL
tools enforce read-only queries and an approved insert, but that boundary is the tools', not
the sandbox's. Commands run in the thread's own MicroVM, not on your machine.

## Run (local mode, no account)

Local mode (`DO_MICROVM_TRANSPORT=local`) needs no DigitalOcean account; the sandbox is a local directory, the template's
setup is skipped and the prompt tells the agent not to install packages (its commands run on this machine), so charts need
matplotlib installed locally.

```bash
DO_MICROVM_TRANSPORT=local uv run examples/07_sales_assistant/run_script.py
```

## Run (real MicroVMs)

From the repository root, with a model configured (see [`../README.md`](../README.md)) and
`DO_MICROVM_API_TOKEN`, `DO_MICROVM_ACCESS_TOKEN` + `DO_MICROVM_IMAGE` (an image the agent can `pip install` into, as in
[`05_data_analyst`](../05_data_analyst)). To set them up step by step, with the optional `TAVILY_API_KEY` (below):

```bash
uv run python scripts/setup_examples.py --examples 07
uv run python scripts/setup_examples.py --examples 07 --dry-run   # checks only
```

Then:

```bash
uv run examples/07_sales_assistant/run_script.py             # territory report, then the RFQ (one thread, two turns)
uv run examples/07_sales_assistant/run_script.py --ask "Which genre earns the most in my territory?"
uv run examples/07_sales_assistant/run_script.py --thread lab-1234 --ask "..."   # continue a thread's sandbox
uv run examples/07_sales_assistant/run_script.py --delete --save outputs   # copy /outputs/ to outputs/<thread>/, delete the sandbox
```

`run_script.py` does what the factory does for each run (and each resume after an approval),
approves the two gated writes and prints what it approved, and prints per run how long the
sandbox took to get ready, the agent to run and the pause, with the MicroVMs API requests
each sent. It starts the mock mail server with a fresh mailbox unless one is already
listening.

With `langgraph dev` (approvals in Studio or [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui)):

```bash
uv run examples/07_sales_assistant/mail_server/mock_mail_server.py &
cd examples/07_sales_assistant && uv run langgraph dev --n-jobs-per-worker 10
uv run examples/07_sales_assistant/mail_server/send_to_inbox.py --reset   # a fresh RFQ in the inbox
```

### Weekly newsletter (optional)

With `TAVILY_API_KEY` set, the assistant also launches `newsletter-agent` (the second graph in
`langgraph.json`) as an async subagent, as in the course. It runs inside `langgraph dev`, next
to the main graph, and does not use the sandbox; it needs the server, not `run_script.py`.
Without the key the subagent is not registered.

## Notes

- **One MicroVM per thread, paused ones included.** Delete finished threads with `run_script.py --delete`,
  expire unused ones with [`scripts/gc_microvms.py`](../../scripts/gc_microvms.py) (`--sandbox-ttl`), or
  delete everything with [`scripts/delete_all.py`](../../scripts/delete_all.py). Threads can run at the same time
  (see below).
- The agent's commands and file operations go to the sandbox server over the MicroVM's HTTPS
  endpoint, so they are not MicroVMs API calls and do not use the team's exec budget. With
  `DO_MICROVM_TRANSPORT=exec` all threads share the team's exec budget, and the client backs
  off on 429.
- A run that is cancelled or crashes still pauses the sandbox (`finally`); a killed agent
  process does not, and the MicroVM pauses itself after `DO_MICROVM_IDLE_TIMEOUT`.

## Where the time goes

Agent: `anthropic-claude-4.6-sonnet` (assistant, reviewer) and `anthropic-claude-haiku-4.5` (analyst, inbox
manager) on DigitalOcean Serverless Inference. `run_script.py` runs one thread for two turns.

- **The model dominates.** Each turn is the agent's own work (a report with a chart, or the RFQ with its quote, review,
  draft, ledger and two approvals); the sandbox is a small part of it.
- **A new thread clones the template** instead of booting, uploading and installing. The template is built once
  (temporary MicroVM, upload, `pip install matplotlib`, checkpoint, delete).
- **Turn 2 resumes the paused sandbox** in a couple of seconds; the later runs of a turn find it running.
  A run ends with a pause.
- **MicroVMs API requests for a whole session:** 2 creates (build VM, thread VM), 1 checkpoint, 2 execs (one per VM, to
  set its server's token), 5 pauses, 2 resumes (requests: a pause is sent again while the MicroVM still reads running,
  and a run that finds it running needs no resume); none while the agent runs.
- **With `DO_MICROVM_TRANSPORT=exec`** the same session sends about 40 exec requests per thread; three threads at once
  (three `run_script.py --thread … --delete` processes started together, template prebuilt) were not refused (no 429).
- **Under `langgraph dev`** introspection needs no MicroVM, and a new thread's first run (clone, list the skills, count
  customers) is quicker than a full turn; the thread's sandbox is paused afterwards.
- **The weekly newsletter** (Tavily, Rock, Jazz, Latin) is launched as a background task and takes a few minutes; it is
  saved as `/outputs/newsletter-<timestamp>-<task id>.html` (3 genre sections, no `<script>`).

The chart, report and ledger of that run are what `--save` copies to `outputs/`
(not committed). The Chinook database is
[lerocha/chinook-database](https://github.com/lerocha/chinook-database), MIT license.
