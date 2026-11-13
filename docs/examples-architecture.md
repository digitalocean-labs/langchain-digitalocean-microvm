# What runs in the MicroVM, per example

The examples use the two patterns of [`design.md`](design.md) (and of LangChain's
[Sandboxes for Deep Agents](https://docs.langchain.com/oss/python/deepagents/sandboxes)):

| Pattern | Where the agent loop and model calls run | What the MicroVM receives |
| --- | --- | --- |
| **Sandbox as a tool** (design §5) | In your process | `execute` and file operations, sent to a small server in the MicroVM over its HTTPS endpoint |
| **Agent in sandbox** (design §4) | In the MicroVM: a whole deep agent served by a LangGraph Agent Server (an *agent host*) | Runs forwarded by a gateway in the supervisor, which creates, pauses and deletes the hosts |

## Sandbox as a tool

| Example | What runs inside the MicroVM |
| --- | --- |
| [01 quickstart](../examples/01_quickstart) | Nothing is installed. The agent's own shell commands and file operations run there; the seeded `/notes.md` and the agent's `/fact.txt` survive a pause and resume. |
| [02 dcode](../examples/02_deepagents_code) | The file edits and shell commands (package installs, test runs) of the `dcode` coding agent, which itself runs on your machine. |
| [03 subagent sandboxes](../examples/03_subagent_sandboxes) | Two MicroVMs per thread, one for the analyst and one for the reviewer, each with its own shell and filesystem. The analyst runs commands on the seeded `/data/sales.csv` and writes `/outputs/summary.md`; the reviewer works on a copy of that file. The supervisor has no shell and reads and writes files through `/subagents/<name>/` mounts. |
| [05 data analyst](../examples/05_data_analyst) | Python the model writes (`sqlite3` queries on the uploaded Chinook database, `matplotlib` charts into `/outputs/`) and the agent's `pip install matplotlib`. |
| [06 warm start](../examples/06_warm_start) | The setup commands (`pip install matplotlib`, with the database seeded) run once in a temporary MicroVM, which is checkpointed as a template and cloned for each new sandbox. The analyst of 05 then works in a clone. |
| [07 sales assistant](../examples/07_sales_assistant) | One MicroVM per thread, cloned from a template: `/AGENTS.md`, `/skills/`, `/data/chinook.db`, matplotlib and the agent's outputs. The SQL tools run `python3 /tools/chinook_sql.py` there (standard library only, read-only `SELECT`, plus one parameterized `INSERT` into `Customer`). The model, the subagents and the mail MCP tools stay in the agent process. |
| [08 board pack](../examples/08_board_pack) | Six MicroVMs cloned from one template, one per specialist (four regional analysts, an auditor, a publisher). Fixed scripts run in them: `board_sql.py` (read-only SQL on `chinook.db`), `board_chart.py` (matplotlib charts), `board_render.py` (the publisher builds one HTML file), besides each specialist's own files and shell. The orchestrator has no shell. |
| [09 incident response](../examples/09_incident_response) | Four MicroVMs per thread (three investigators and a verifier), cloned from an `incident` template. Evidence fetched through the Action Gateway is written to `/data/`, and the scripts of [`incident_common/sandbox`](../examples/incident_common) run on it: `series.py` (when the errors started), `timeline.py` (the commit before the onset), `verdict.py` (healthy or not). The investigators have file tools and tools that call these fixed scripts, but no shell. The remediator has no MicroVM. |
| [10 erasure](../examples/10_erasure) | Four locator MicroVMs. `classify.py` reads `/data/request.json` and `/data/records.json`, prints a manifest of ids, categories and proposed actions, then deletes the raw records, so the models only see manifests. |
| [11 vulnerability response](../examples/11_vulnerability_response) | One analyst MicroVM per repository, plus fresh MicroVMs to re-test patches. `analyse.py` pins the version, checks it against the advisory, runs the repository's unittest suite, runs a static reachability check and a harmless proof of concept, and tests the fixed version from a local package index. It writes `verdict.json` and, when the tests pass, `fix.patch`. |

## Agent in sandbox

| Example | What runs inside the MicroVM |
| --- | --- |
| [04 async subagents](../examples/04_async_subagents) | One shared host per subagent type, from `images/agent-host`. `echo` is a model-free graph that replies `echo: <message>`. `researcher` is a deep agent with its own shell and filesystem under `/work/tasks/<task id>`, given a model key through the host's environment. Local mode runs a `langgraph dev` process per task instead. |
| [12 incident agents](../examples/12_incident_response_agents) | Six MicroVMs per incident, each a separate agent host. The commander's host runs the commander graph (`host/graph.py`) with an embedded gateway, which creates one host per specialist from the specialists' image (Metrics, Change, Infra, Remediator, Verifier), each a deep agent with its own model. Specialists run the scripts of `incident_common/sandbox` through their tools (below). Each holds a model key and a role-scoped broker token, and no Action Gateway credential. The relay, the approvals and the Action Gateway session stay on your machine. |

### How an agent with no shell runs a script

In examples 09 and 12 an agent runs analysis scripts without having a shell. The agent gets no `execute` tool, because
deepagents offers `execute` only when its backend can run commands:

- **Example 09** wraps the sandbox in `FilesOnly` (`tools.py`), a backend that cannot run commands. The tools call
  `execute` on the MicroVM themselves.
- **Example 12** runs the agent inside the MicroVM with a `FilesystemBackend` rooted at the task directory
  (`specialists/graphs.py`).

In both, the script is run by a custom `@tool` function, not by the model. In 12 the tools (`find_onset`,
`build_timeline`, `judge_health`) call `run_script`, which starts `python <task dir>/tools/<name>` with
`create_subprocess_exec`. The model picks only a tool's parameters (such as `minutes` or `onset`); it never chooses the
program or writes a command line. Evidence reaches the scripts as files: the fetch tools write `/data/*.json`, the script
reads them and prints JSON back to the tool. The Verifier's verdict is computed by the image's own `verdict.py`, not by a
copy in the task directory that the agent could edit.
