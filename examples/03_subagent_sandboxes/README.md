# 03 · Subagent sandboxes: one MicroVM per subagent

**Requires** (real MicroVMs): a team limit of at least 2 MicroVMs (one per subagent) for each thread.

**Quick run** (local mode: no DigitalOcean account, and the agent's commands run on this machine; needs a model, see [`../README.md`](../README.md)):

```bash
DO_MICROVM_TRANSPORT=local uv run examples/03_subagent_sandboxes/run_script.py --delete
```

Real MicroVMs: see Real MicroVMs below.

**The use case.** A supervisor agent has two helpers: an **analyst** that computes revenue per region from a sales
CSV file, and a **reviewer** that checks the analyst's numbers. In Deep Agents a subagent normally works in the main
agent's backend, so every subagent sees and can change the same files. Here each subagent gets **its own
DigitalOcean MicroVM as its filesystem and shell**: the reviewer cannot see or alter the analyst's workspace, and the
supervisor decides what passes between them by copying a file.

These are ordinary synchronous subagents (the `task` tool): their model calls run in the supervisor's process, and only
their files and commands go to the MicroVMs. [`04_async_subagents`](../04_async_subagents) goes further and moves whole
subagents *into* MicroVMs. Design: [`docs/design.md`](../../docs/design.md) §5.

```
supervisor process (one thread) ───────────────────────────────┐
  supervisor: state + /subagents/<name>/ mounts (files only)   │
  ├── task → analyst   (execute, files) ──────────► MicroVM A  │  /data/sales.csv, /outputs/...
  └── task → reviewer  (execute, files) ──────────► MicroVM B  │  /inputs/..., /outputs/review.md
───────────────────────────────────────────────────────────────┘
```

| | Sees | Can run commands |
| --- | --- | --- |
| `analyst` | only its VM (seeded with `/data/sales.csv`) | yes, in its VM |
| `reviewer` | only its VM (`/inputs/` is where work arrives) | yes, in its VM |
| supervisor | its state, plus `/subagents/analyst/...` and `/subagents/reviewer/...` (each VM's `/`) | no |

Handing the analysis to the reviewer is a file copy by the supervisor:
read `/subagents/analyst/outputs/summary.md`, write
`/subagents/reviewer/inputs/summary.md`, then delegate.

## How it is wired

`agent.py` is a per-thread graph factory (`make_graph`). For every run,
`asubagent_sandboxes(...)` from `langchain_digitalocean_microvm.subagents`
(steps 1, 2 and 4) and the graph factory (step 3):

1. gets or creates the thread's two sandboxes **in parallel**, off the event
   loop (created on the thread's first run and seeded with `SEED_FILES`;
   resumed on later runs);
2. rewrites the subagent specs so each one's `FilesystemMiddleware`,
   summarization and skills middleware use its own sandbox
   (`per_subagent_specs`);
3. `sandboxes.mount(StateBackend())` gives the supervisor a `CompositeBackend` with
   each sandbox under `/subagents/<name>/` (`execute` never goes through a mount);
4. pauses both sandboxes when the run ends.

Studio's introspection calls get the same graph without sandboxes.

`langchain_digitalocean_microvm.subagents` relies on deepagents internals
(middleware names and two private hooks), checked by a contract test for every
supported deepagents version. Treat it as experimental.

## Run it without a DigitalOcean account (local mode)

From the repository root, with a model configured (see [`../README.md`](../README.md)):

```bash
DO_MICROVM_TRANSPORT=local uv run examples/03_subagent_sandboxes/run_script.py --delete
```

Two turns: the analyst computes revenue per region and month-over-month growth
into `/outputs/summary.md`; then the supervisor copies it to the reviewer, which
checks the numbers and writes `/outputs/review.md`. After each turn the script
lists the files in each sandbox. Local sandboxes are directories under
`$TMPDIR/microvm_sandbox_local/<thread>-<subagent>/`.

Or with `langgraph dev`, in LangGraph Studio or agent-chat-ui:

```bash
cd examples/03_subagent_sandboxes
uv run langgraph dev     # graph id: agent
```

- **Studio**: open the URL `langgraph dev` prints (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`);
  it asks you to sign in to LangSmith in the browser.
- **agent-chat-ui**: open <https://agentchat.vercel.app>, Deployment URL `http://localhost:2024`, Assistant / Graph ID
  `agent`, no API key for a local server.

`langgraph dev` loads the repository's `.env` (`env` in `langgraph.json`) and its values replace the shell's, so set
`DO_MICROVM_TRANSPORT=local` in `.env` (a variable set on the command line has no effect there), or point `env` at
another file.

## Real MicroVMs

Set up the variables below step by step:

```bash
uv run python scripts/setup_examples.py
uv run python scripts/setup_examples.py --dry-run   # checks only
uv run python scripts/check_setup.py --examples 03   # ready, ready* (also needs the setup named) or what is missing
```

By hand:

```bash
# .env
DO_MICROVM_API_TOKEN=...                             # microvm:create/read/update/delete
DO_MICROVM_ACCESS_TOKEN=...                          # microvm:access only (the sandboxes' HTTPS endpoints)
DO_MICROVM_IMAGE=registry.digitalocean.com/<registry>/microvm-sandbox:<tag>
```

The image must keep running, ship `sh`, coreutils and `python3`, and let its
user write to `/` (the subagents write `/outputs/...` and whatever absolute
paths the model picks; deepagents offloads large results to
`/large_tool_results/`). The agent-host image runs as uid 1000, so it does not
qualify. Build one that does (root, `python:3.12-slim`, port 8080, no Docker
needed) and push it to your registry:

```bash
uv run scripts/build_sandbox_image.py --repository registry.digitalocean.com/<registry>/microvm-sandbox
```

Then:

```bash
uv run examples/03_subagent_sandboxes/run_script.py --delete
```

Each turn prints how long provisioning, the agent run and the pause took, and
how many MicroVMs API requests each sent (`langchain_digitalocean_microvm.client.api_calls()`).

A subagent's files live only in its MicroVM, so `--delete` deletes them too. To
keep them, add `--save DIR`: after every turn, while the sandboxes are still
running, the script copies each subagent's `/outputs/` to
`DIR/<thread>/<subagent>/outputs/` (later turns overwrite earlier copies):

```bash
uv run examples/03_subagent_sandboxes/run_script.py --delete --save outputs
cat outputs/lab-*/reviewer/outputs/review.md
```

- **Two MicroVMs per thread.** Paused MicroVMs count against your team's limit,
  so delete a thread's sandboxes when you are done with it: `run_script.py --delete`, or
  `scripts/delete_all.py --yes`.
- **Where the time goes.** The model's turns dominate and vary most. The sandboxes add one boot per thread (both
  created in parallel, the seeds uploaded) and a resume on each later run, and each run ends with a pause. No
  MicroVMs API request is made while the agent runs. A whole script sends 2 exec requests (one per sandbox, to set
  its server's token), 2 creates, 4-5 pauses, 1-2 resumes and 2 deletes.
- **Request budget.** The subagents' commands and file operations go to a small
  server in each MicroVM over its HTTPS endpoint (`DO_MICROVM_TRANSPORT=endpoint`, the
  default), so they are not MicroVMs API calls and do not use the team's exec budget;
  only create, resume, pause and delete are API calls. With `DO_MICROVM_TRANSPORT=exec` every file
  operation is a call to the MicroVMs exec API instead (8-15 per turn in this
  example), and the client backs off and retries when it is rate limited
  (counted as `retry:429`).
- **Clean-up.** With real MicroVMs, each run records its sandboxes' last use in
  `gateway.sqlite` (`gateway-<namespace>.sqlite` with `DO_MICROVM_NAMESPACE`) in the package's state directory (docs/reference.md, "Where state lives"); `scripts/gc_microvms.py`
  (dry run unless `--yes`) deletes sandboxes unused for `--sandbox-ttl` (default 24 h).
- **Permissions.** deepagents cannot enforce `permissions` rules on a backend
  that runs commands, so a sandboxed subagent must not have (or inherit) any;
  the MicroVM is the boundary. `per_subagent_specs` says so if it happens.

## LangSmith traces

Optional. Set these in `.env` and run the example as above:

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=subagent-sandboxes      # unset: the project is `default`
```

To read the traces, open <https://smith.langchain.com>, sign in and open the project named in `LANGSMITH_PROJECT`. A trace shows the supervisor and each subagent's `task` as a nested run with its tool calls.
