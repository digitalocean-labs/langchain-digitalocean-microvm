# 04 · Async subagents, each in its own MicroVM

**Requires** (real MicroVMs): a team limit of at least 2 MicroVMs (one shared host per subagent type: `echo` and `researcher`; 1 without the researcher).

> Port of the async subagents lab of the
> [Deep Agents course](https://github.com/langchain-ai/lca-deepagents)
> (`python/m5/async_lab`), whose supervisor delegates to an `AsyncSubAgent` on a
> deployment you start by hand.

This supervisor is the same kind of deep agent, but its specs point at an
embedded **SubAgentGateway** (`langchain_digitalocean_microvm.agent_host`,
experimental, installed with the `agent-host` extra) that runs every task in an **agent host**: a DigitalOcean MicroVM running the
in-memory LangGraph Agent Server (one shared host per subagent type), or, by default, a local
`langgraph dev` process per task standing in for one. Design and decisions:
[`docs/design.md`](../../docs/design.md) §4.

```
you ── Studio ── supervisor (langgraph dev) ── embedded gateway ──┬── host for task A  (echo)
                 start/check/update/cancel/list   127.0.0.1:port  ├── host for task B  (echo)
                                                                  └── host for task C  (researcher)
```

| Subagent | What it is | Needs |
| --- | --- | --- |
| `echo` | Model-free; answers `echo: <message>`. Prefix the message with `sleep N:` to keep it busy N seconds, and/or `save NAME:` to write the rest of the message to `/outputs/NAME` | nothing |
| `researcher` | Deep agent with a shell and filesystem rooted in its own task directory inside the host (`/work/tasks/<task id>`, which it sees as `/`; results under `/outputs`) | `DO_MODEL_ACCESS_KEY` (passed into the host by `agent.py`) |

## Setup

```bash
uv sync --all-extras    # the agent-host extra: langgraph-cli[inmem], langgraph-sdk, starlette, uvicorn
```

The supervisor needs a model (see [`../README.md`](../README.md)); the
`researcher` subagent needs `DO_MODEL_ACCESS_KEY`. Everything else has a default:

```bash
# .env
DO_MICROVM_HOST_MODE=local        # default; `microvm` uses real MicroVMs
# DO_MICROVM_GATEWAY_TOKEN=      # random per process when unset
# LANGGRAPH_API_KEY=               # set to the gateway token for you when unset (see below)
```

Local mode spawns one `langgraph dev` per task from
[`images/agent-host`](../../images/agent-host) using this repository's
virtualenv (override the template with `DO_MICROVM_SUBAGENT_TEMPLATE`).

## Run it

```bash
cd examples/04_async_subagents
uv run langgraph dev
```

Studio opens on the `agent` graph. The gateway is not running yet: it starts on
your first message (`LazyGateway` in `agent.py`), and the server log prints
`Subagent gateway up at http://127.0.0.1:<port> (local hosts, ...)`.

### Walkthrough

1. **Two tasks at once.** *"Start two echo tasks at once, each taking about
   60 seconds: one saying 'ping one', one saying 'ping two'. Report both task
   ids."* Two `start_async_task` calls go out in one turn; each one makes the
   gateway boot a host (a few seconds locally) and forward the run. Two
   `langgraph dev` processes now live under `$TMPDIR/microvm_agent_local/<task id>/`.
2. **Check while running.** *"Check on the ping-one task."* The gateway proxies
   `runs.get` to the host: status `running`.
3. **Update one.** *"Tell the ping-one task 'pong' instead."* `update_async_task`
   creates a new run on the same thread with `multitask_strategy=interrupt`; the
   sleeping run is interrupted and `pong` is echoed straight away. Check it again:
   `echo: pong`.
4. **Cancel the other.** *"Cancel the ping-two task."* Proxied to the host while
   the run is alive. (Cancelling an already-finished run is answered by the
   gateway itself; the Agent Server would return 404 and the tool would report a
   failed cancel.)
5. **List.** *"List all tasks."* Terminal statuses come from the supervisor's
   state; a running task is asked live.
6. **Kill a host.** Start one more long task (*"Start an echo task that takes
   two minutes saying 'gone'"*), then remove its host. From another terminal the
   host process is the one whose config path carries the task id:

   ```bash
   pkill -f "microvm_agent_local/<task id>"
   ```

   Now *"Check on that task."* The gateway finds the host unreachable and the
   host manager reports it failed, so the tool gets `{"status": "error",
   "error": "agent host ... unreachable ..."}` and the task is marked `error`.
   Without the gateway the deepagents middleware would keep saying `running`.

`run_script.py` does the same walkthrough programmatically, deleting the third
task's host through the host manager itself:

```bash
uv run examples/04_async_subagents/run_script.py
```

### Try the researcher

*"Launch a researcher task: write a Python script under /outputs that prints
the first 20 Fibonacci numbers, run it, and summarize the output."* The host
gets `DO_MODEL_ACCESS_KEY` from the supervisor's environment via
`HostSpec(env=...)`; its files live under `$TMPDIR/microvm_sandbox_local/agenthost-<task id>/`
in local mode. Then: *"Fetch that task's files and show me the script."* The
supervisor calls `fetch_subagent_artifacts`, which copies the task's
`/outputs/` into its own filesystem as `/subagents/<task id>/fib.py`, and reads
it with `read_file`.

### Fetching files (artifacts)

Files stay in the host that produced them. Each task writes to its own
directory there (`/work/tasks/<task id>/outputs/` on a MicroVM), so tasks that
share a pooled host never see each other's files. The supervisor's extra tool:

- `fetch_subagent_artifacts(task_id)` copies every file of the task into the
  supervisor's state at `/subagents/<task id>/<path>`; `paths=[...]` picks some
  (`"/outputs/report.md"` or `"report.md"`). It works while the task runs and
  after it finished; a paused host is woken for it and paused again later.
- Up to 10 MiB per file (gateway) and 8 MiB per call (tool); symlinks are
  never followed. Binary files are stored base64-encoded in the supervisor's
  state.
- If the host is gone (deleted, or evicted to make room at the MicroVM limit),
  the tool answers that the task's files are gone: there is no copy elsewhere.

Without a model: *"Start an echo task with the message 'save hello.md: hi',
then fetch its files and read them."*

The same through the gateway's HTTP API (token in `Authorization: Bearer`):
`GET /artifacts/<task id>` lists, `GET /artifacts/<task id>/files/<path>` returns
the bytes (410 when the host is gone).

## What runs where

| Thing | Location |
| --- | --- |
| Gateway store (hosts, threads, cached results) | `gateway-local.sqlite` in the package's state directory (`gateway.sqlite` in microvm mode; docs/reference.md, "Where state lives") |
| local host working dir, `langgraph.json`, `.env`, `server.log` | `$TMPDIR/microvm_agent_local/<task id>/` |
| local host filesystem (`/work`; a task's outputs under `tasks/<task id>/outputs/`) | `$TMPDIR/microvm_sandbox_local/agenthost-<host id>/` |
| Gateway inventory | `curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:<port>/gateway/status` |

Idle hosts are paused (`SIGSTOP` locally) a minute after their last run and
resumed on the next `update`; finished results are served from the store, so a
`check` never wakes a paused host.

## Notes

- **`LANGGRAPH_API_KEY`.** The LangGraph SDK always sends an `x-api-key`
  header, taken from `LANGGRAPH_API_KEY`, else `LANGSMITH_API_KEY`. The gateway sets
  `LANGGRAPH_API_KEY` to the gateway token when it is unset, so that header
  authenticates and your LangSmith key never leaves the process. If you set it
  yourself to another value, requests still authenticate by the spec's
  `Authorization: Bearer` header, but that value is sent to the gateway as `x-api-key`.
- **Reloads and restarts.** `langgraph dev` re-imports `agent.py` on file
  changes; the gateway starts again on the next turn. Every local host runs
  under a small guard process that shuts it down as soon as the supervisor
  process is gone (uvicorn ends a reload worker by re-raising SIGTERM, so no
  exit hook inside it can be relied on). The next gateway marks the previous
  process's hosts as failed on its first poll, and an old task checked
  afterwards reports an error, not `running`. The one gap: a host that was
  *paused* (SIGSTOP) when the supervisor was killed hard stays frozen;
  `pkill -CONT -f microvm_agent_local; pkill -f microvm_agent_local` clears it.

## Real MicroVMs (`DO_MICROVM_HOST_MODE=microvm`)

Set up what follows step by step: the tokens, the model key, `DO_MICROVM_HOST_MODE=microvm` and the agent-host image
(rebuilt when its sources changed):

```bash
uv run python scripts/setup_examples.py --examples 04
uv run python scripts/setup_examples.py --examples 04 --dry-run   # checks only
```

By hand:

```bash
# .env (in addition to the model settings)
DO_MICROVM_HOST_MODE=microvm
DO_MICROVM_API_TOKEN=...        # microvm:create/read/update/delete
DO_MICROVM_ACCESS_TOKEN=...     # microvm:access only; never the control token
DO_MICROVM_AGENT_IMAGE=registry.digitalocean.com/<registry>/microvm-subagent:<tag>
```

Build the image once with [`images/agent-host/build_image.py`](../../images/agent-host)
(see its README). Then run the same walkthrough, in Studio or with:

```bash
DO_MICROVM_HOST_MODE=microvm uv run examples/04_async_subagents/run_script.py --with-researcher --delete
```

What is different from local mode:

- **One shared host per subagent type.** To keep the fleet at two MicroVMs
  (paused ones count against the team's limit), `echo` tasks share one host and `researcher`
  tasks another (`reuse="pooled"`); each task is its own thread on that host.
  The first task of each type waits for its MicroVM to boot (image pull plus
  server start); later ones start at once.
- **Killing a host** means deleting the MicroVM: `doctl compute microvm delete <id>`
  (ids in `doctl compute microvm list`, names start with `microvm-agenthost-`).
  Every running task on that host then reports an error. Finished tasks keep
  answering `check` from the gateway's cache.
- **Pausing.** A MicroVM pauses after 10 minutes without inbound requests,
  even if its subagent is still working. The gateway polls every running task
  every 2 seconds, which keeps the host awake; when nothing is running it
  pauses the host itself after a minute, and the next task resumes it in
  under a second.
- **At the limit.** If a new host is needed while the team is at its MicroVM limit, the
  gateway deletes an idle host whose results are cached once it has been unused for a minute
  (`eviction_grace`); until then, or if every host is busy, launching fails with a clear
  "MicroVM limit" message.
- **Warm start (optional).** With `DO_MICROVM_WARM_START=1`, hosts start from a checkpoint
  of an already-booted host, which skips the image pull, after a one-time build of the
  checkpoint. Worth it when hosts come and go (deleted, evicted); with one
  shared host per type that mostly means across sessions.
- **State across runs.** Hosts and their tokens are remembered in
  `gateway.sqlite` in the package's state directory, so the next session reuses them.
- **Lifecycle.** The gateway deletes a host it has not used for an hour (per-task) or a
  day (pooled), and its files with it: copy what you need with `fetch_subagent_artifacts`
  soon after a task finishes. `check` keeps working afterwards; `update` asks for a new
  task. Hosts left behind by a supervisor that is no longer running:

  ```bash
  uv run scripts/gc_microvms.py        # list
  uv run scripts/gc_microvms.py --yes  # delete
  ```
- **Clean up** (paused MicroVMs count against the limit). `run_script.py` leaves its hosts paused for
  the next run, and says so; `--delete` removes every live host in the gateway's store at its end,
  also after a failed run. Otherwise:

  ```bash
  uv run scripts/delete_all.py        # list
  uv run scripts/delete_all.py --yes  # delete
  ```
