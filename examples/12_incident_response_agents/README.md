# 12 · Incident response with specialist agents, each in a MicroVM agent host of its own

**Requires** (real mode): a team limit of at least 6 MicroVMs per incident (the commander's agent host and one agent host
per specialist); the commander's and the specialists' images (below); an Action Gateway token whose actor has a
DigitalOcean connection and a code-host connection; the droplet of [`../incident_common/`](../incident_common); a public
HTTPS URL for the relay's broker port (a tunnel is enough for a demo). Local mode needs none of it.

## Overview

**The scenario.** The same incident as [example 09](../09_incident_response): a checkout service on a droplet fails after
a deploy, and restoring the droplet from a snapshot fixes it, with a human's approval. Read 09 first: this example
changes the architecture, not the story.

**What changes.** In example 09 every agent runs in one process on your machine, and only their files live in MicroVMs.
Here each team member is a **whole, separate deep agent running in a MicroVM of its own** (an **agent host**: a MicroVM
running a LangGraph Agent Server), with its own model and only the actions its role needs. That is closer to how a team
of agents would run in production: each agent's tools, files and credentials stay in its own MicroVM.

**What it teaches about Deep Agents.**
- **Async subagents** (as in [example 04](../04_async_subagents)): the commander starts a specialist as a background
  task, does not wait for it, and checks its result later with `check_async_task`. Because a subagent on another server
  cannot wake its parent, a small server on your machine, the **relay**, starts a new turn for the commander when its
  specialists finish.
- **A model per task**: Sonnet where the reading is hard (the commander, the code diff), Haiku for the rest.
- **Least privilege.** The commander has no action tools at all. Each specialist can call only its own role's actions,
  through the relay, and holds no gateway credential.
- **Human-in-the-loop across processes.** The Remediator's restore request interrupts *its own* thread, in its own
  MicroVM. The relay shows it to you in the console or on a web page, and runs the restore itself once you approve.

**The pieces.**
- **The relay** (`relay.py`, on your machine) receives the alert, wakes the commander, keeps the Action Gateway token,
  and shows the approvals. Its **broker** passes each specialist's permitted actions on to the gateway.
- **The commander's agent host** runs the incident commander.
- **One agent host per specialist** task: Metrics, Change and Infra Analysts, the Remediator and the Recovery Validator.

## How it works

```
 alert ─► relay.py (approvals API; wakes the commander; the broker; local mode: the simulated world)
            │                                                   ▲
            ▼                                                   │ broker port (https): a role's token, its actions only
 commander's agent host incident-<id>   Sonnet 4.6 · start / check async sub-agents · save_postmortem
            │  embedded gateway: one agent host per task, with that role's model and environment
   ┌────────┼───────────┬─────────────┬───────────────────┐
 metrics  changes     infra       remediator          verifier
 Haiku    Sonnet      Haiku       Haiku               Haiku
 /metrics commits     droplet,    restore: ASK        /health,
          (broker)    snapshots   delete:  DENY       droplet (broker)
                      (broker)    (broker; the relay
                                  decides and runs it)
```

| Specialist | Model | Action Gateway actions | Credential in its MicroVM |
| --- | --- | --- | --- |
| Metrics Analyst | `anthropic-claude-haiku-4.5` | none: reads the service's `/metrics`, `series.py` in its host | the model key |
| Change Analyst | `anthropic-claude-4.6-sonnet` | `github_list_commits`, `github_compare_commits`, through the relay's broker | the model key, its broker token |
| Infra Analyst | `anthropic-claude-haiku-4.5` | `digitalocean_droplet-get`, `digitalocean_image-list`, through the relay's broker | the model key, its broker token |
| Remediator | `anthropic-claude-haiku-4.5` | `action_search` and `action_invoke` on its incident's session in the relay: `digitalocean_restore-droplet` **ask**, `digitalocean_droplet-delete` **deny** | the model key, its broker token |
| Recovery Validator | `anthropic-claude-haiku-4.5` | `digitalocean_droplet-get` through the relay's broker, plus the service's `/health` and `verdict.py` | the model key, its broker token |

`DO_INCIDENT_MODEL_<ROLE>` (for example `DO_INCIDENT_MODEL_METRICS`) and `DO_INCIDENT_MODEL_COMMANDER` pick other models.
The action names (`github_*`, `digitalocean_*`) are the Action Gateway catalog's own. GitHub is a trademark of its owner;
this example is not affiliated with or endorsed by it.

| | What it shows |
| --- | --- |
| **Sub-agents as whole agents in MicroVMs** | Each specialist is a deep agent served by an Agent Server in a MicroVM of its own (`specialists/`), created per task by the commander's embedded `SubAgentGateway` (the package's `agent_host`) with that role's `HostSpec`: the specialists' image and the role's environment |
| **Least privilege per agent** | Every specialist reaches its actions through the relay's broker, which holds the gateway token and every gateway session: one per read-only role that allows that role's actions and nothing else, and one per incident for the Remediator. A specialist's MicroVM holds only a token for its own role in its own incident; an action it does not hold is not offered to it, and the broker refuses it |
| **The agent proposes, the relay acts** | The Remediator's request for the restore runs nothing: its run ends with an `action_approval` interrupt in **its** thread. The relay shows it with the droplet and the snapshot it read itself, posts the human's decision to the gateway with its own token and, once approved, runs the restore itself; then it resumes the Remediator's thread, which reports the outcome. No credential in any MicroVM can approve an action or open a gateway session, and the commander is not involved |
| **A model per task** | Sonnet where the reading is hard (the commander, the diff), Haiku for the rest |
| **The commander is woken** | A sub-agent on another server cannot wake its parent. The relay reads the tasks while an incident has work open and starts a short turn on the commander's thread when specialists finished |

## How a run goes

1. **Alert.** The relay gets the commander's host (`host/`) and starts its first run with the alert.
2. **Investigate.** In one turn the commander starts `metrics`, `changes` and `infra`, then stops. The gateway creates an agent
   host for each; they work in parallel. When they are done, the relay starts a turn on the commander's thread ("Update from
   your sub-agents..."); the commander reads them with `check_async_task` and names the cause.
3. **Restore.** It starts the `remediator` with the action and its arguments. The Remediator's `action_invoke` asks the
   broker; the incident's gateway session (the relay's) answers `requires_approval`, and nothing runs. Its run ends, waiting
   for a human. The task is `waiting`; the approval appears in the console (or on the
   [approvals page](#the-approvals-page)). On the decision the relay posts it to the gateway and, if approved, runs the
   restore itself; then it resumes the Remediator's thread through the commander's gateway, and the Remediator reports what
   happened.
4. **Verify.** Woken again, the commander starts the `verifier`, which samples `/health` and judges it in its host; only its
   verdict says the service is back. The verdict comes from the host's own samples and the image's `verdict.py`, not from files
   the agent can edit. When the verifier reports, the relay also reads the service's `/health` itself (real mode) and the page
   shows whether that reading agrees, and flags it when it does not.
5. **Postmortem.** The commander saves its narrative (summary, root cause, what was done, follow-ups; no timeline: it has no
   clock). The relay adds the timeline and both readings of the service from what it recorded. A follow-up asking to delete the
   droplet goes to the Remediator, and its session's policy refuses it. When the incident closes, the relay writes the postmortem
   to `outputs/postmortems/<incident>.md` (`DO_INCIDENT_POSTMORTEMS` picks another directory) before any host is deleted: a
   close that cannot write it deletes nothing.

## Who holds what

- The **relay** holds the Action Gateway token (every session: the read-only roles', the Remediator's, and a read-only session
  of its own for the approver's facts), the hosts' tokens, the operator token, the signing secret and the broker key. None of
  them reaches a host, except as noted below.
- The **commander's host** holds the MicroVMs tokens (its gateway creates, pauses and deletes the specialists' MicroVMs, and the
  control token can create and delete any MicroVM of the team), the model key, and what it passes to the specialists' hosts
  when it creates them: its incident's broker tokens (one per role). It holds no gateway token. The commander agent
  has no tool that reads them: no shell and no action tools. Use a team and tokens for this example alone, and rotate them
  after it.
- **Each specialist** holds the model key and, if it has actions, a broker token for its own role in its own incident. The
  Remediator's token can request the restore but not approve it. No specialist holds a gateway token or a MicroVMs token. The
  specialists have no shell (file tools only, in their task's directory); this example does not restrict a MicroVM's network.

## The broker

The relay serves the broker on a port of its own (`--broker-port`, default 8788, on 127.0.0.1); it is the only port the
specialists' MicroVMs need to reach, so it is the only one to expose. `POST /incidents/{id}/actions/{role}` takes one of:

| Body | Answer |
| --- | --- |
| `{"op": "list"}` | The role's actions: name, description, input schema |
| `{"op": "search", "query", "limit"}` | The role's actions that fit a description |
| `{"op": "invoke", "tool", "arguments", "rationale"}` | `{"text": ...}`: what the gateway answered |
| Remediator: `{"op": "invoke", "tool", "arguments", "rationale", "call"}` | `{"text": ...}`, or for an action that asks a human `{"approval": {"proposal", "tool", "arguments", "rationale", "expires_at"}}` until the relay has decided it; the same `call` (the tool call's id) then answers the outcome. The Remediator has `search` and `invoke`, no `list`, and no request that decides |

- **Tokens.** A role's token for an incident is an HMAC of the incident and the role under the broker key
  (`DO_INCIDENT_BROKER_KEY`, else a random key the relay creates in `outputs/broker.key`, readable by its owner only), so a
  restarted relay accepts the same tokens. The key is not derived from the alerts' signing secret: whoever can sign an
  alert cannot mint a broker token.
  A token is refused for another role (401), another incident (401), and once its incident is closed (403).
- **Allow lists.** One gateway session per role (`roles.py`): `changes` allows `github_list_commits@5` and
  `github_compare_commits@4`, `infra` allows `digitalocean_droplet-get@3` and `digitalocean_image-list@2`, `verifier` allows
  `digitalocean_droplet-get@3`; everything else is denied by the session's policy, and the broker refuses any other tool
  before it reaches the gateway (403). The Remediator's session, one per incident, asks before `digitalocean_restore-droplet@3`
  and denies `digitalocean_droplet-delete@3`. `metrics` has no actions: 404.
- **Revocation.** `DELETE /incidents/{id}` revokes the incident's tokens first, then the commander's host deletes the
  specialists' hosts, then the relay deletes the commander's host and the incident's Remediator session. A close refused (a
  run in flight) revokes nothing.
- **Limits.** At most 60 requests a minute per role and incident (429 beyond), a request body of at most 64 KiB, arguments a
  JSON object, a rationale of at most 500 characters, 30 seconds for the gateway's answer.

In local mode the broker serves only the Remediator, on 127.0.0.1 (`DO_INCIDENT_BROKER_PORT`, default 8788), over the
simulated world, so approvals go the same way as in real mode. The read-only specialists reach the simulated world the relay
serves (`incident_common/world_server.py`) through the package's local gateway, with the same allow lists.

## Files

| File | What it is |
| --- | --- |
| `roles.py` | The team: one row per specialist (model, actions, what the commander reads about it) |
| `agent.py` | The commander: its async sub-agents behind the embedded gateway, one `HostSpec` per role, `save_postmortem` |
| `specialists/` | The specialists' host app: `graphs.py` (one deep agent per role, rooted in its task's directory), `broker_client.py` (the role's actions, and the Remediator's, through the broker), `langgraph.json`, and `image.json` / `pyproject.toml` for `build_image.py --spec` |
| `host/` | The commander's host app: `graph.py`, `routes.py` (`/incident/tasks`, `/incident/decision`, `/incident/events`, `/incident/cleanup`), `langgraph.json`, `image.json`, `pyproject.toml` |
| `relay.py` | Alerts, the approvals API, waking the commander, the broker, deciding approvals and running approved actions, resuming a specialist |
| `console.py` | `demo` (a relay on a local port, a simulated alert, follow it; `--web`: decide on the page), `list`, `close` |
| `approvals.py` | The approval prompt and the facts the relay reads itself |
| `postmortem.py` | The postmortem the relay writes: the timeline and the verdict from its record, then the commander's narrative |
| `web/v2`, `web/v3` | The approvals page the relay serves: the same script, two stylesheets |

The droplet, the simulated world and the analysis scripts are the shared scenario in [`../incident_common/`](../incident_common);
in local mode the relay serves its world to every host (`incident_common/world_server.py`), so every specialist sees the same
droplet.

## Run (local mode, no account)

```bash
uv sync --all-extras          # from the repository root
DO_MICROVM_TRANSPORT=local uv run examples/12_incident_response_agents/console.py demo            # approve in the console
DO_MICROVM_TRANSPORT=local uv run examples/12_incident_response_agents/console.py demo --approve  # approve every request
DO_MICROVM_TRANSPORT=local uv run examples/12_incident_response_agents/console.py demo --web      # decide on the approvals page
```

Every host is a local `langgraph dev` (the commander's serving `host/`, each specialist's serving `specialists/`). With
`DO_MODEL_ACCESS_KEY` the agents use DigitalOcean Serverless Inference; without it, `EXAMPLES_MODEL` (see
[`../README.md`](../README.md)). The demo deletes every host it started when it ends.

You do not start `langgraph dev` yourself here: the relay and the commander start the hosts, and `host/` and
`specialists/` are their templates, not a graph to serve by hand.

**LangSmith**: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in `.env`. In local mode each
host is a process of this machine that inherits the console's environment, so the commander and every specialist are
traced, each host's runs as traces of their own. In real mode a host is a MicroVM whose environment holds only the
variables the relay and the commander set (`host_env` in `relay.py`, `role_env` in `agent.py`), and those leave the
LangSmith variables out: nothing in the MicroVMs is traced. `DO_MICROVM_SUBAGENT_ENV_PASSTHROUGH=LANGSMITH_TRACING,LANGSMITH_API_KEY,LANGSMITH_PROJECT`
passes them to the commander's host, which puts the key in that MicroVM; the specialists' hosts do not get them.

## Run (real mode)

Set in `.env`: `DO_MICROVM_API_TOKEN`, `DO_MICROVM_ACCESS_TOKEN` (see the repository's README), `DO_MODEL_ACCESS_KEY`,
`DO_ACTION_GATEWAY_TOKEN` (a [personal access token](https://docs.digitalocean.com/reference/api/create-personal-access-token/) authorized for Action Gateway; [quickstart](https://docs.digitalocean.com/products/managed-agents/action-gateway/quickstart/)) and `DO_ACTION_GATEWAY_ACTOR_ID` (the actor needs a DigitalOcean
connection and a code-host connection that reads the repository's commits), and `DO_INCIDENT_REPO=owner/name`; leave
`DO_MICROVM_TRANSPORT` unset. Build the demo repository and the droplet as [example 09](../09_incident_response#run-real-mode)
does.

**The images.** The commander's host runs `host/` from an image of its own, and each specialist's host runs `specialists/`
from another. Build both, and again after changing this package, the example's modules or `incident_common`:

```bash
python3 images/agent-host/build_image.py --spec examples/12_incident_response_agents/host/image.json \
  --repository registry.digitalocean.com/<registry>/microvm-incident-agents-commander
python3 images/agent-host/build_image.py --spec examples/12_incident_response_agents/specialists/image.json \
  --repository registry.digitalocean.com/<registry>/microvm-incident-specialists
```

and set `DO_INCIDENT_HOST_IMAGE` and `DO_INCIDENT_SPECIALISTS_IMAGE` to the references they print.

**The broker's URL.** The specialists' MicroVMs reach the broker over HTTPS. For a demo, a tunnel to the broker port is
enough, for example a Cloudflare quick tunnel (no account; the URL changes each time):

```bash
cloudflared tunnel --url http://127.0.0.1:8788        # prints https://<random>.trycloudflare.com
```

Set `DO_INCIDENT_BROKER_URL` to the URL it prints. The tunnel exposes the broker port only: the approvals API stays on
127.0.0.1, and every broker request needs a role's token. The demo traffic (the specialists' action calls and their
answers) then passes through the tunnel's provider. For anything longer-lived, run the relay where it has a public HTTPS
address of its own, and treat that place like one that holds credentials (it holds the gateway token).

```bash
uv run examples/incident_common/setup_droplet.py --deploy-bad                 # the failure starts
uv run examples/12_incident_response_agents/console.py demo --web --delete       # or --approve: no human
uv run examples/12_incident_response_agents/relay.py serve                    # or a relay for other approvers (DO_INCIDENT_OPERATOR_TOKEN, DO_INCIDENT_RELAY_SECRET)
uv run examples/incident_common/setup_droplet.py --cleanup
```

`scripts/delete_all.py` lists and deletes what a killed run left behind (the commander's host and the specialists' hosts).

The relay opens one Remediator session per incident, named `microvm-ag-<incident>-remediator-<hex>`, and deletes it when it
closes the incident, after the hosts. When it starts, it also deletes those of every incident whose commander host its store
records as deleted (a run killed while closing, or hosts deleted by `scripts/delete_all.py`). A killed relay leaves its
sessions: the open incidents' Remediator sessions, and its own, named `microvm-ag-<hex>` (one for the approver's facts and
one per role the broker served). A Remediator session of an incident that was never closed stays until a relay closes that
incident. Delete leftovers through the Action Gateway API (`GET /v2/action-gateway/sessions`, `DELETE /v2/action-gateway/sessions/{sessionUrn}`).

## The relay's routes

`POST /alerts` takes an Insights webhook delivery signed with `DO_INCIDENT_RELAY_SECRET`
(`X-DigitalOcean-Signature: t=<unix>,v1=<HMAC-SHA256 of "<unix>.<body>">`, at most 5 minutes old) and answers 202 at once.
The incident is named after the body, so a retried delivery starts nothing new. Everything under `/incidents`, `/service`
and `/policy` needs `Authorization: Bearer <DO_INCIDENT_OPERATOR_TOKEN>` (whoever holds it can approve):

| Route | What it does |
| --- | --- |
| `GET /` | The [approvals page](#the-approvals-page) (`web/v2`, or `--page` / `DO_INCIDENT_PAGE`); its files under `/static`. No token needed to load it: the page asks for one |
| `GET /incidents` | The commander hosts and what the relay last saw of each, its specialists' hosts among it, as far as it passed through the relay (no host is woken up) |
| `GET /incidents/{id}` | The incident: its pending approvals (the interrupts its specialists wait on, each with its `task` and `role`) with the facts the relay reads itself, the last message, its tasks and its story |
| `DELETE /incidents/{id}` | Closes the incident: revokes its broker tokens, the commander's host deletes the specialists' hosts its gateway created, the relay deletes the commander's host, then the incident's Remediator session; answers with what it deleted; 409 while a run is in flight |
| `POST /incidents/{id}/decision` | `{"interrupt_id", "decision": "approve" \| "deny", "reason"?}`: resumes the specialist's thread and returns its run; 409 when the approval is no longer waiting (a second approver) |
| `POST /incidents/{id}/messages` | `{"text"}`: another turn, when the commander is idle |
| `GET /incidents/{id}/runs/{run_id}/events` | Server-sent events: the run's tool calls and the story as it changes, then its end |
| `GET /incidents/{id}/log?after=<offset>` | The incident's event log, read from its host: the Remediator's gateway calls and the decisions |
| `GET /incidents/{id}/postmortem` | The postmortem as it stands: the recorded timeline and verdict, then the commander's narrative (`{"markdown"}`) |
| `GET /service` | The service's `/health` as the relay reads it (real mode; `null` in local mode) |
| `GET /policy` | The actions each role holds, and the Remediator's policy |

**Hosts while they work.** A MicroVM is idle when it gets no inbound requests, even while it works, and is then paused.
While an incident has work open, the relay reads its tasks through the commander's host every 3 seconds, and its gateway
polls the specialists' hosts, so neither is paused mid-run; a specialist's host pauses a minute after its run ends (the
Remediator's while a human decides), and the decision resumes it. The relay records the commander hosts, their tokens and
each incident's latest run in the package's gateway store, so a restarted relay picks the open incidents up again.
`scripts/gc_microvms.py` knows the commander hosts too.

## The approvals page

`console.py demo --web` prints the page's address with the operator token in the URL fragment
(`http://127.0.0.1:<port>/#token=...`): the browser never sends a fragment, and the page moves the token into the tab's
session storage and out of the address bar. A relay started with `relay.py serve` serves the same page at `/`; it asks for
the token. Two approvers can have it open at once: the first decision wins, the second gets "already decided".

**Who approves.** Anyone with `DO_INCIDENT_OPERATOR_TOKEN` can decide, and the relay's log records the decision and its
reason, not who made it; the gateway records the relay's token as the approver. That is enough for a demo. For real use, put
the relay's port behind your single sign-on (a reverse proxy that authenticates each approver and passes their identity), keep
the operator token out of people's hands, and for destructive actions require a second approver before the relay runs it.

- **Incident view.** A rail of the open incidents; for the selected one, its steps (alert, investigate, diagnose, approve,
  restore, verify, report), the commander's latest words, the decision card, its agent hosts in one lane (the commander's,
  then one per specialist: running, paused, not started), and four tabs: the story (each specialist's report, every decision
  and policy check, newest first), every tool call, the Action Gateway's policy and calls, and the postmortem.
- **The decision card.** The restore in plain words ("Restore droplet ... from snapshot ...?"), the commander's reason, the
  droplet and snapshot the relay read itself, when the approval expires, and whether another open incident waits on the same
  action. Approve or Deny (with an optional reason): the relay posts the decision to the gateway, runs an approved restore
  itself and resumes the Remediator's thread; the console, if one follows the incident, sees the decision and goes on.
- **Fleet view** (`#fleet`, for a second screen). One lane per incident, every agent host side by side.

The page polls the relay's own list every 3 seconds, which wakes no host; it reads an incident (its commander's host) only
when the list shows it changed, when it is selected, or while one of its runs works, so a host waiting for a decision can
pause. Every value is shown as text, never as HTML, and the relay serves it with a strict Content-Security-Policy (no inline
script or style, nothing from another origin, no framing). `web/v3` is the same page in a quiet light style:
`--page v3`, or `DO_INCIDENT_PAGE=v3`.

## Where the time goes

`console.py demo --approve --delete`: the commander's host and five specialists' hosts start from their images (cold) in
MicroVMs; the commander and the Change Analyst run on `anthropic-claude-4.6-sonnet`, the others on
`anthropic-claude-haiku-4.5`, on DigitalOcean Serverless Inference; the real Action Gateway; the broker behind a Cloudflare
quick tunnel; the approval answered by the console (no waiting for a human).

- **One-off setup:** `setup_droplet.py` (droplet, service, snapshot) takes a couple of minutes.
- **The relay answers the alert at once** (a `202`); everything after that is background work.
- **Hosts boot from their images**, so the commander's host and each specialist's host add a boot before their first run.
- **The investigators run in parallel**: the Metrics and Infra Analysts answer well before the Change Analyst.
- **Alert to the approval** takes a few minutes, and the restore starts shortly after the approval; the Remediator's
  thread resumes to report, and the verifier confirms `healthy` about a minute later.
- **Alert to the postmortem and to the follow-up's refusal** takes several minutes in MicroVMs. In local mode, with the
  same models, there are no hosts to boot and the alert to the postmortem takes about a third of that.

The read-only specialists made 9 broker calls in that run, each one of its role's actions; the Remediator made two requests
through the broker (the restore, held for the approval, and the delete, refused by its session's policy).

With `console.py demo --web --delete`, the restore is approved by a human on the approvals page, and the times from the
approval on include the human's.
