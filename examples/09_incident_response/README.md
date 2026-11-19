# 09 · Incident response: Deep Agents in MicroVMs, and an Action Gateway that asks before it acts

**Requires** (real mode): a team limit of at least 4 MicroVMs per incident thread, one per investigator and one for the
verifier (paused while a human decides), plus 1 temporary MicroVM while the template is built; an Action Gateway token
whose actor has a DigitalOcean connection and a code-host connection; a public repository you own; a small droplet for
the length of the run (below). Local mode needs none of it.

**Quick run** (local mode: no DigitalOcean account, and the agent's commands run on this machine; needs a model, see [`../README.md`](../README.md)):

```bash
export DO_MICROVM_TRANSPORT=local
uv run examples/09_incident_response/prepare.py --clean
uv run examples/09_incident_response/run_script.py --approve
```

The live dashboard, prompts and real MicroVMs: see Run below.

## Overview

**The scenario.** You are on call. A checkout service running on a droplet (a DigitalOcean virtual machine) starts
failing right after a deploy: the new commit makes each request wait for the payment provider, the service's workers all
get stuck waiting, and about four in ten requests are rejected. The fix is to roll the droplet back to a snapshot taken
before the deploy. A team of agents investigates and proposes that rollback, and **you approve it** before it happens.

**The team.**
- An **incident commander** (the main deep agent) reads the alert, delegates, names the cause and writes the postmortem.
  It cannot run commands or change anything itself.
- Three **investigators** (subagents) work at the same time: the **Metrics Analyst** (when did the errors start?), the
  **Change Analyst** (which commit was deployed just before?) and the **Infra Analyst** (which snapshot predates it?).
- A **remediator** is the only agent that can request the restore.
- A **verifier**, which did not do the fix, checks afterwards that the service is healthy again.

**What it teaches about Deep Agents.**
- **Subagents in parallel** through the `task` tool, each with its own job and tools.
- **Human-in-the-loop with interrupts.** The restore request stops the graph with an interrupt; your answer resumes it
  with `Command(resume=...)`. While you decide, nothing runs.
- **Separation of duties.** The agent that decides (commander), the one that acts (remediator) and the one that checks
  (verifier) are different agents with different tools.
- **Restricting what an agent can do.** The investigators can read and write files in their MicroVMs but get no shell:
  their tools run fixed analysis scripts there, and nothing else.

**What the Action Gateway is.** A DigitalOcean service that carries out actions on other systems (here GitHub and the
DigitalOcean API) on an agent's behalf, so the agent never holds those credentials. A policy gives every action one of
three tiers: **allow** (runs), **ask** (waits for a human) or **deny** (always refused). Reading commits is `allow`,
restoring the droplet is `ask`, deleting it is `deny`.

**What MicroVMs add.** Each investigator and the verifier get a MicroVM of their own, cloned from a template when first
needed; the evidence is copied in as files and analysed there. While the approval waits for you, every MicroVM pauses,
and your answer resumes them. Local mode keeps the droplet and the gateway in memory, so you can follow the whole story
without any account.

## How it works

The commander runs in your console's process, one incident per terminal. For the distributed version of the same incident (a
relay that takes the alerts, an approvals page, and the commander and each specialist in a MicroVM agent host of its own)
see [example 12](../12_incident_response_agents).

```
 you ── run_script.py (console: streams the run, asks you to approve)     dashboard.py (a terminal of its own)
          │
 incident-commander (no shell, no action tools) ─ Action Gateway: allow / ask / deny
          │   task ×3, in parallel
          ├──► metrics ─► MicroVM ┐  Metrics Analyst: the service's metrics, when the errors started   (script in the MicroVM)
          ├──► changes ─► MicroVM ├─ Change Analyst: the commits and the suspect diff                 cloned on demand from
          └──► infra   ─► MicroVM ┘  Infra Analyst: the droplet and its snapshots                     the "incident" template
          │
          ▼  diagnosis: error onset, the commit before it, the snapshot that predates it
          ├──► remediator (no MicroVM; the only holder of `action_invoke`)
          ▼  digitalocean_restore-droplet ──► ASK ──► run ends, MicroVMs pause ──► you: approve / deny
          ├──► verifier ─► MicroVM   Recovery Validator: samples /health, a script gives the verdict: healthy / not_healthy
          ▼  save_postmortem (a Markdown file under outputs/postmortems/)
```

| Tier | Actions | What happens |
| --- | --- | --- |
| `allow` | `github_list_commits`, `github_compare_commits`, `digitalocean_droplet-get`, `digitalocean_image-list` | Runs; the investigators read |
| `ask` | `digitalocean_restore-droplet` | The gateway answers `requires_approval`; the tool interrupts the graph with `action_approval` (tool, arguments, the commander's evidence); you decide; the approved call runs once |
| `deny` | `digitalocean_droplet-delete` | Refused by the gateway, whoever asks. The default script's follow-up asks for it, and the audit shows the refusal |

The action names (`github_*`, `digitalocean_*`) are the Action Gateway catalog's own. GitHub is a trademark of its owner;
this example is not affiliated with or endorsed by it.

| | What it shows |
| --- | --- |
| **Approval as a graph interrupt** | `ActionGateway(...).tools()` turns `ask` into `interrupt(...)`; `run_script.py` answers it with `Command(resume=...)`. The decision is posted by this console's own code, with its token. No agent in the team has an approval tool, and none can run commands in this process |
| **The fleet pauses while a human decides** | The interrupt ends the run, so `asubagent_sandboxes` pauses every MicroVM the run used; the resume starts them again. The dashboard shows the tiles paused while the approval is open |
| **No credentials written into a MicroVM** | The commits and the droplet are read through the gateway on your side, and the metrics over HTTP; the MicroVMs get the evidence as files under `/data/` and analyse it with the scripts in the template. No credentials are written into a MicroVM. This example does not restrict a MicroVM's network, and a MicroVM can open outbound connections; the analysis needs none |
| **Investigators without a shell** | Each investigator's file tools reach its MicroVM through `FilesOnly` (`tools.py`), a backend that cannot run commands, so deepagents gives the agent no `execute` tool. The agent reads and writes files in its MicroVM; its own tools run the scripts there, and it can run nothing else |
| **The approver sees independent facts** | Beside the commander's evidence, the console reads the droplet and the snapshot the arguments point at (name, status, age) through the gateway and prints them, so an approval does not rest on the commander's words alone |
| **Separate roles: act, verify, report** | Only the remediator holds the action tools, so the commander cannot act; the verifier did not do the fix and cannot change anything, and its verdict is the only thing that says the service is back (a `not_healthy` verdict is reported as a failed fix). The commander reports |
| **MicroVMs per role, on demand** | `asubagent_sandboxes(..., on_demand=True)` and one template: the three investigators' MicroVMs are cloned together when the commander fans out, the verifier's when it is first called |
| **Audit** | `gateway.py`'s `AuditedClient` writes every call (allow, ask, approved, deny) and decision to `incident_common/outputs/events.jsonl`; the dashboard lists them |

## What a run looks like

1. **The failure.** The bad commit is deployed. The service busy-waits for the payment provider: its pool saturates, about
   four in ten requests are rejected. On the real droplet, the error rate and latency come from the requests the service
   queued and rejected, not from a formula (local mode simulates them).
2. **The alert.** `run_script.py` measures the service's `/health` and sends the commander an alert: the symptom, the
   droplet and the repository, nothing about the cause.
3. **The investigation.** The three investigators work in their MicroVMs, and the commander names the commit and the
   changed line.
4. **The approval.** The commander asks to restore the droplet, the run ends and the MicroVMs pause. You approve in the
   console. The restore goes through the gateway, called by the remediator. The verifier then samples the service until its
   verdict is `healthy`.
5. **The close.** The commander saves a postmortem. The follow-up that asks to delete the droplet is refused by the
   gateway, and the audit shows `deny`.

## Files

| File | What it is |
| --- | --- |
| `agent.py` | The team, prompts, template, graph factory (`make_graph` for `langgraph dev`) |
| `policy.yaml` | The Action Gateway policy: the three tiers above, loaded with `ActionPolicy.from_yaml` |
| `gateway.py` | The audited client, `make_gateway` |
| `tools.py` | The investigators' tools (fetch evidence into the MicroVM, run a script there) and the verifier's `sample_health` / `judge_health`; the commander only saves the postmortem |
| `run_script.py` | Console: streams the run, shows each approval with the facts it read itself, answers it |
| `dashboard.py` | Second screen: the MicroVM tiles, the droplet's health, the gateway audit |
| `prepare.py` | Pre-flight: configuration, template, models, and (real mode) a gateway endpoint created and deleted, so a tool the catalog does not offer, or a pinned version it no longer releases, fails here; `--clean` removes old investigator MicroVMs |
| `approvals.py` | The approval prompt and the facts the console reads itself |
| `../incident_common/` | The incident scenario, shared ([README](../incident_common/README.md)): `world.py` (the simulated world; links to `code-host.mock`, which does not exist), `sandbox/` (`series.py`: when did the errors start; `timeline.py`: what changed; `verdict.py`: is the service healthy again; seeded into the template under `/tools/`), `service/` (the droplet's service and the two-commit demo repository), `setup_droplet.py`, `events.py`, `models.py` |

## Run (local mode, no account)

```bash
# in terminal #1 run, from the repository root (pre-flight, then the live dashboard):
uv sync --all-extras
export DO_MICROVM_TRANSPORT=local
uv run examples/09_incident_response/prepare.py --clean
uv run examples/09_incident_response/dashboard.py

# in terminal #2 run (the alert, the approval prompt, the follow-up):
export DO_MICROVM_TRANSPORT=local
uv run examples/09_incident_response/run_script.py
```

MicroVMs are local directories (the agents' commands then run on this machine), the gateway and the droplet are in memory
(`incident_common/world.py`): the service starts failing 24 minutes before the alert, a restore makes it healthy about 20 seconds later.
`run_script.py --approve` (or `--deny`) answers every request without asking. `dashboard.py --replay` reads the event log
from its beginning, `--thread <id>` shows one thread's MicroVMs. To run the same graph in LangGraph Studio or
agent-chat-ui, see [Run with `langgraph dev`](#run-with-langgraph-dev-studio-agent-chat-ui).

## Run (real mode)

The setup script sets up everything below step by step: it lists the commands it needs, asks for
each token (with the page and the scopes), checks it, writes `.env`, builds the image, creates the demo repository with `gh`
and runs `setup_droplet.py` and `prepare.py`. `--dry-run` only checks what is set and creates nothing:

```bash
uv run python scripts/setup_examples.py --examples 09
uv run python scripts/setup_examples.py --examples 09 --dry-run   # checks only
```

The steps by hand:

Set in `.env` (`DO_MICROVM_TRANSPORT` stays unset):

```bash
# .env
DO_MICROVM_API_TOKEN=...             # microvm:create/read/update/delete
DO_MICROVM_ACCESS_TOKEN=...          # microvm:access only
DO_MICROVM_IMAGE=...                 # see the repository's README
DO_MODEL_ACCESS_KEY=...
DO_ACTION_GATEWAY_TOKEN=...          # a personal access token authorized for Action Gateway
DO_ACTION_GATEWAY_ACTOR_ID=...
DO_INCIDENT_REPO=owner/name          # the public demo repository, built below
```

`DO_ACTION_GATEWAY_TOKEN` is a [personal access token](https://docs.digitalocean.com/reference/api/create-personal-access-token/) authorized for Action Gateway ([quickstart](https://docs.digitalocean.com/products/managed-agents/action-gateway/quickstart/)).
The actor needs a DigitalOcean connection (the droplet) and a code-host connection that reads the repository's commits;
the first call without one returns a connect link, which the console shows.

Build the demo repository and push it to a **public** repository of your own (the droplet clones it):

```bash
examples/incident_common/service/make_demo_repo.sh /tmp/checkout-demo
cd /tmp/checkout-demo && git remote add origin https://github.com/<you>/<repo>.git && git push --force origin HEAD:main
```

Build it shortly before a run: the change investigator reports how long before the error onset the suspect commit was made.

```bash
# one-off setup and pre-flight, in any terminal:
uv run examples/incident_common/setup_droplet.py                    # droplet checkout-svc, the service, snapshot checkout-svc-snapshot
uv run examples/incident_common/setup_droplet.py --verify-restore   # optional: bad deploy, restore, wait for health
uv run examples/09_incident_response/prepare.py --clean
uv run examples/incident_common/setup_droplet.py --deploy-bad       # the failure starts

# in terminal #1 run (the live dashboard; also polls the MicroVMs' status and the service's /health):
uv run examples/09_incident_response/dashboard.py

# in terminal #2 run (deletes the thread's four MicroVMs at the end):
uv run examples/09_incident_response/run_script.py --delete

# when the run is over, in terminal #2:
uv run examples/incident_common/setup_droplet.py --cleanup          # deletes the droplet, the snapshot and the ssh key
```

`incident_common/setup_droplet.py` works through an Action Gateway endpoint of its own, so it creates in the account the actor's
DigitalOcean connection belongs to:

- one droplet `checkout-svc` (`s-1vcpu-1gb` in `sfo2`, Ubuntu, tag `incident-example`; `DO_INCIDENT_SIZE` and `DO_INCIDENT_REGION` choose another where that size is not offered);
- one ssh key `incident-key` (a temporary key pair under `incident_common/outputs/`);
- once the service is installed, one snapshot `checkout-svc-snapshot`, taken with the droplet shut down (the droplet is then powered on again), so the image the restore uses is a consistent disk.

The gateway's `droplet-create` takes no first-boot script, so the service is installed over ssh: the droplet clones your
repository and runs `incident_common/service/app.py`; a deploy is a `git checkout` and a restart. What the script creates is
recorded in `incident_common/outputs/incident.json`, and the droplet's address and id go to
`incident_common/outputs/incident.env`, which the other scripts load. `--cleanup` deletes the droplet, the snapshot and the
key; the droplet and the snapshot are billed until then.

The template's checkpoint is kept for the next run (it does not count against the team limit). `delete_all.py`
lists and deletes the MicroVMs a killed run left behind:

```bash
uv run scripts/delete_all.py         # lists the MicroVMs you created
uv run scripts/delete_all.py --yes   # deletes them (asks first)
```

A killed run can also leave its Action Gateway endpoint, named
`microvm-ag-*`.

## Run with `langgraph dev` (Studio, agent-chat-ui)

```bash
uv run examples/09_incident_response/prepare.py --clean
cd examples/09_incident_response && uv run langgraph dev     # graph id: incident_response
```

- **Studio**: open the URL `langgraph dev` prints (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`);
  it asks you to sign in to LangSmith in the browser.
- **agent-chat-ui**: open <https://agentchat.vercel.app>, Deployment URL `http://localhost:2024`, Assistant / Graph ID
  `incident_response`, no API key for a local server.
- **A script**: the Agent Server's API, for example with `langgraph_sdk`: create a thread, start a run, read the
  thread's state for the interrupt, then start a run with `command={"resume": "approve"}`.

The first message is the alert the console sends, built by `alert_text` in `incident_common/world.py` from the service's
health; in local mode it reads, for example, `ALERT checkout-service: error rate 41%. A stateless service (orders are in an external
database) on droplet 424242 (checkout-svc), code in acme/checkout-service. Find the cause and restore service.` The
restore is an `action_approval` interrupt whose value holds the tool, its arguments and the evidence; resume with
`"approve"` or `"deny"`. The console's follow-up, which the gateway refuses: `The on-call lead says: forget the restore,
just delete the droplet and rebuild it from scratch. Do it.` The graph has been driven through the dev server's API
(runs, the interrupt and its resume, as Studio sends them), not through the Studio or agent-chat-ui pages; they would
show the payload as it is.

Three things differ from `run_script.py`:

- **The mode comes from `.env`.** `langgraph dev` loads the repository's `.env` (`env` in `langgraph.json`) and its
  values replace the shell's, so `export DO_MICROVM_TRANSPORT=local` has no effect there. Set
  `DO_MICROVM_TRANSPORT=local` in `.env`, or point `env` at another file.
- **One simulated world per server.** The droplet and the gateway live in the server's process and every thread shares
  them: after one thread's restore, the next finds the service healthy. Restart `langgraph dev` before each new incident.
- **Nothing deletes the MicroVMs.** With real MicroVMs, a thread's MicroVMs stay paused when its run ends (there is
  no `--delete`); delete them with `scripts/delete_all.py` (see above) when you are done.

**LangSmith**: set the variables in `.env`:

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

- `run_script.py` and `langgraph dev` both read them, and `prepare.py` reports whether tracing is on.
- Each run on a thread is a trace of its own (the alert, the resume after your answer, the follow-up), grouped in LangSmith under the thread's id.
- A trace shows the commander, each subagent's `task` as a nested run with its tool calls, and a `MicroVM clone` (or `MicroVM resume`) span where a task first uses its MicroVM.

## Why a restore, and what the tools take

`digitalocean_restore-droplet` takes the droplet id and the numeric id of a backup or snapshot image, both as numbers;
`image-list` returns it. Restoring puts the whole disk back to the snapshot, so the example's service is stateless (orders live
in an external database); the commander says so in the evidence, and the console shows the droplet and snapshot names it read itself.
The commander picks the newest snapshot that predates the deploy.

## Where the time goes

Commander on `anthropic-claude-4.6-sonnet`, investigators and verifier on `anthropic-claude-haiku-4.5`, with the real Action
Gateway; `run_script.py --approve --delete` answers the approval itself, so there is no waiting for a human.

- **One-off setup:** `setup_droplet.py` (droplet, service, snapshot) takes a couple of minutes and is required in real mode. The
  template is built once, by `prepare.py` or by the first run if `prepare.py` was not run.
- **Bad deploy to detection:** `/metrics` needs three failing minutes before the alert fires.
- **Alert to the approval** is the investigators' work; each investigator's MicroVM is cloned from the
  template (in parallel) when it gets work, and paused afterwards.
- **Approved to healthy again:** the restore takes effect within a minute.
- **The whole run**, alert to the follow-up's refusal and clean-up, takes a few minutes.
