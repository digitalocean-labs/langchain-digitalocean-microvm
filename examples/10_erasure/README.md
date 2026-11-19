# 10 · Erasure request: Deep Agents in MicroVMs, one human decision per irreversible step

**Requires** (real MicroVMs): a team limit of at least 4 MicroVMs per request thread, one per locator (paused while a
human decides), plus 1 temporary MicroVM while the template is built. Local mode needs none of it. The five systems the
request touches are **mocks** in every mode (below).

## Overview

**The scenario.** Privacy laws such as the GDPR give people the right to have their personal data erased. A customer,
Jane Doe <jane.doe@example.com>, asks to be forgotten. Her data is spread over five systems: a CRM, a support desk,
billing, email marketing and product analytics. Someone has to find every record, erase what may be erased, keep what
the law requires, and prove it afterwards. Erasure is not "delete everything":

- **another Jane Doe** (a different address) is in the CRM, and must stay;
- **billing keeps its invoices** (tax records), so the billing customer is *redacted* (her details are cleared), not
  deleted, and deleting it is refused by the policy, whoever asks.

**The team.** An **erasure coordinator** (the main deep agent, no shell) sends four **locators** (subagents: CRM,
support, billing, marketing) to search their systems at the same time. From their findings it makes a plan, record by
record (erase, redact, keep or skip), carries it out one record at a time with **your approval for each**, checks that
the data is really gone (`verify_erasure`) and writes an **evidence report**.

**What it teaches about Deep Agents.**
- **Subagents in parallel** through the `task` tool, each searching one system.
- **Keeping sensitive data away from the model.** A locator's tool fetches the person's records straight into the
  locator's MicroVM, and a script classifies them there. The model sees only a manifest (record ids, categories, the
  proposed action), never a name or an address.
- **Human-in-the-loop with interrupts**: one decision per irreversible step, five in the default run.
- **Tools that can't be misused**: the locators get file tools but no shell, so they can run only the classifier.

**What the Action Gateway is.** A service that carries out actions on other systems on an agent's behalf, under a
policy: **allow** (runs), **ask** (waits for a human) or **deny** (always refused). Here searches are `allow`, each
erasure or redaction is `ask`, and deleting the billing customer is `deny`.

**Mocks.** The five systems are **in-memory mocks** with the same action names and arguments as the real ones, so no
accounts are needed; only the MicroVMs have a real mode.

## How it works

```
 you ── run_script.py (console: streams the run, asks you to approve each erasure)     dashboard.py (second terminal)
          │
 erasure-coordinator (no shell) ────────────────── Action Gateway: allow / ask / deny
          │   task ×4, in parallel
          ├──► crm       ─► MicroVM ┐  each fetches the person's records through the gateway (on your side) into its
          ├──► support   ─► MicroVM ├─ MicroVM, classifies them there (/tools/classify.py) and returns a manifest:
          ├──► billing   ─► MicroVM │  ids, identity match, categories, the proposed action and its call; no values
          └──► marketing ─► MicroVM ┘  cloned on demand from the "erasure" template
          │
          ▼  plan: erase / redact / retain / skip, per record
          ▼  action_invoke, one record at a time ──► ASK ──► run ends, MicroVMs pause ──► you: approve / deny
          ▼  verify_erasure (every record re-read, every system searched again) ──► save_report (ids and a hash)
```

| Tier | Actions | What happens |
| --- | --- | --- |
| `allow` | searches and reads: `hubspot_crm_contacts_search`, `hubspot_crm_contacts_get_by_contactid`, `intercom_search_contacts`, `intercom_get_a_contact`, `stripe_get_customers`, `stripe_get_customers_customer`, `stripe_get_invoices`, `mailchimp_search_members`, `mixpanel_engage_query` | Runs; the locators and `verify_erasure` read |
| `ask` | `hubspot_crm_contacts_gdpr_delete_create`, `intercom_delete_a_contact`, `mailchimp_forget_contact`, `mixpanel_delete_profile` (erasures) and `stripe_post_customers_customer` (the billing redaction) | The gateway answers `requires_approval`; the tool interrupts the graph with `action_approval`; you decide; the approved call runs once. One decision per record: five in the default run |
| `deny` | `stripe_delete_customers_customer`, `hubspot_files_gdpr_delete_by_fileid` | Refused by the gateway, whoever asks. The default script's follow-up asks to delete the billing customer, and the audit shows the refusal |

The action names (`hubspot_*`, `intercom_*`, `stripe_*`, `mailchimp_*`, `mixpanel_*`) are the Action Gateway catalog's own. HubSpot, Intercom, Stripe, Mailchimp and Mixpanel are trademarks of their respective owners; this example is not affiliated with or endorsed by them and calls only in-memory mocks.

The tiers are written out by hand rather than taken from the catalog's hints: the catalog marks `intercom_search_contacts`
as not read-only though it only reads, and `mailchimp_forget_contact` as not destructive though it cannot be undone.

| | What it shows |
| --- | --- |
| **The models never see the personal data** | The locators' tools fetch through the gateway on your side and write the records into the locator's MicroVM; `classify.py` runs there and prints a manifest of ids, categories and counts, then deletes the raw records. The models see manifests, never a name or an address. No credentials are written into a MicroVM. This example does not restrict a MicroVM's network, and a MicroVM can open outbound connections, so it shows data minimisation toward the models, not containment |
| **Locators without a shell** | Each locator's file tools reach its MicroVM through `FilesOnly` (`tools.py`), a backend that cannot run commands, so deepagents gives the agent no `execute` tool. Its own tools run the classifier there, and it can run nothing else |
| **Identity matching** | The CRM locator searches by email address and by name; the classifier marks the second Jane Doe `different_person` and the plan skips her. `verify_erasure` and the report show her record still present, as it should be |
| **A refusal that is correct** | Billing's invoices are retained, so the customer record is redacted (an `ask`), and deleting it is `deny` |
| **The approver sees independent facts** | Beside the coordinator's rationale, the console reads the record the arguments point at through the gateway and prints its system, id, age and whether its email address is the request's (without printing it); for the redaction, which fields the call clears and how many invoices stay |
| **The fleet pauses while a human decides** | Each `ask` ends the run, so `asubagent_sandboxes` pauses every MicroVM the run used; the resume starts them again. The dashboard shows the tiles paused while an approval is open |
| **Every record, live** | `dashboard.py` shows one column per system and one line per record: its plan, then awaiting approval, erased, redacted, kept, skipped, denied or a refused delete, and `verify_erasure`'s check. It reads the event log, which holds ids and statuses only |
| **A report with no personal data** | `save_report` builds `outputs/erasure-report.md` from the verification and the gateway's audit: a hash of the email address, record ids, the plan, each decision, what `verify_erasure` found. The coordinator's notes are added with email addresses and the requester's name removed |

## The systems are mocks

`world.py` holds a CRM, a support desk, payments (billing), email marketing and product analytics in memory, behind
the package's local Action Gateway. Their tool names, versions and argument shapes follow the Action Gateway's catalog (the CRM's search needs `after`, `filter_groups`, `limit`, `properties` and `sorts`,
with filters on `propertyName`; email marketing's `forget_contact` takes `audience_id` and `contact_id`; product analytics'
`delete_profile` takes `project_id` and `updates`), and the mocks check the required arguments, so a wrong call fails as
it would against the real tool. Connecting the real systems needs an account in each with an app the gateway can use;
this example does not do that. Only the MicroVMs have a real mode.

## One run

1. **The request.** The coordinator sends the email address and the name to the four locators at once; their MicroVMs
   are cloned from the template on first use.
2. **The manifests.** CRM: the requester's contact (erase) and the other Jane Doe (skip). Support: one contact (erase).
   Billing: the customer (redact) and three invoices (retain). Marketing: one list member and one analytics profile (erase).
3. **Five approvals**, one record at a time, each with the facts the console read. Approve all, or deny one: the record
   stays, and the report says it was denied and is still present.
4. **The check and the report.** `verify_erasure` re-reads every record of the plan and searches every system for the
   email address again (anything found that the plan missed is listed); `save_report` writes the evidence report.
5. **The follow-up**: "delete her billing customer entirely". The coordinator asks the gateway, which refuses (`deny`).

## Files

| File | What it is |
| --- | --- |
| `agent.py` | The team, prompts, template, graph factory (`make_graph` for `langgraph dev`) |
| `policy.yaml` | The Action Gateway policy: the three tiers above, loaded with `ActionPolicy.from_yaml` |
| `gateway.py` | The audited client, which record each changing action targets, `make_gateway` |
| `world.py` | The five mock systems and their gateway actions |
| `tools.py` | The locators' tools (fetch into the MicroVM, classify there), the coordinator's `verify_erasure` and `save_report` |
| `sandbox/classify.py` | Runs in each locator's MicroVM: identity match, categories, the proposed action and its call; seeded into the template under `/tools/` |
| `run_script.py` | Console: streams the run, shows each approval with independent facts, answers it |
| `dashboard.py` | Second screen: the locator MicroVMs, every record of the five systems, the gateway's changing calls and decisions |
| `prepare.py` | Pre-flight: configuration, template, models; `--clean` removes old MicroVMs and the old report |

## Run (local mode, no account)

```bash
uv sync --all-extras          # from the repository root
export DO_MICROVM_TRANSPORT=local
uv run examples/10_erasure/prepare.py --clean      # pre-flight
uv run examples/10_erasure/dashboard.py            # terminal 2
uv run examples/10_erasure/run_script.py           # terminal 1: the request, five approval prompts, the follow-up, the report
uv run examples/10_erasure/run_script.py --approve --deny-system marketing   # no prompts: all approved but the email-marketing erasure
```

In local mode the MicroVMs are local directories (the agent's commands then run on this machine). `dashboard.py` shows what happens from the moment it starts (`--replay` reads the
event log from its beginning, `--thread <id>` shows one thread's MicroVMs). `run_script.py --approve` (or `--deny`) answers every request without asking;
`--deny-system <system>` (crm, support, billing, marketing, analytics; repeatable) denies that one. To run the same
graph in LangGraph Studio or agent-chat-ui, see [Run with `langgraph dev`](#run-with-langgraph-dev-studio-agent-chat-ui).

## Run (real MicroVMs)

Set `DO_MICROVM_API_TOKEN`, `DO_MICROVM_IMAGE` (see the repository's README) and `DO_MODEL_ACCESS_KEY` in `.env` and leave `DO_MICROVM_TRANSPORT` unset:

```bash
uv run examples/10_erasure/prepare.py --clean
uv run examples/10_erasure/dashboard.py               # terminal 2: also polls the MicroVMs' status from the API
uv run examples/10_erasure/run_script.py --delete     # deletes the thread's four MicroVMs at the end
```

The template's checkpoint is kept for the next run (it does not count against the team limit); `scripts/delete_all.py`
lists and deletes the MicroVMs a killed run left behind.

## Run with `langgraph dev` (Studio, agent-chat-ui)

```bash
uv run examples/10_erasure/prepare.py --clean
cd examples/10_erasure && uv run langgraph dev     # graph id: erasure
```

- **Studio**: open the URL `langgraph dev` prints (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`);
  it asks you to sign in to LangSmith in the browser.
- **agent-chat-ui**: open <https://agentchat.vercel.app>, Deployment URL `http://localhost:2024`, Assistant / Graph ID
  `erasure`, no API key for a local server.
- **A script**: the Agent Server's API, for example with `langgraph_sdk`: create a thread, start a run, read the
  thread's state for the interrupt, then start a run with `command={"resume": "approve"}`.

The console's request: `Erasure request from Jane Doe <jane.doe@example.com>, identity verified by the privacy team:
erase her personal data from every system that holds it, keep only what the law requires us to keep, and report.` Each
of the five approvals is an `action_approval` interrupt of its own (one per run) whose value holds the tool, its
arguments and the rationale; resume with `"approve"` or `"deny"`. The console's follow-up, which the gateway refuses:
`The billing lead says: delete her billing customer entirely as well, so nothing is left. Do it.` The graph has been
driven through the dev server's API (runs, the interrupts and their resumes, as Studio sends them), in local mode and
with real MicroVMs, not through the Studio or agent-chat-ui pages; they would show the payload as it is.

Three things differ from `run_script.py`:

- **The mode comes from `.env`.** `langgraph dev` loads the repository's `.env` (`env` in `langgraph.json`) and its
  values replace the shell's, so `export DO_MICROVM_TRANSPORT=local` has no effect there. Set
  `DO_MICROVM_TRANSPORT=local` in `.env`, or point `env` at another file; leave it unset for real MicroVMs.
- **One set of systems per server.** The five mock systems live in the server's process and every thread shares them
  (in every mode): once a thread has erased the records, the next finds nothing to erase. Restart `langgraph dev`
  before each new request.
- **Nothing deletes the MicroVMs.** With real MicroVMs, a thread's four MicroVMs stay paused when its run ends (there
  is no `--delete`); delete them with `uv run scripts/delete_all.py` (it lists them first) when you are done.

**LangSmith**: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in `.env`; `run_script.py`
and `langgraph dev` both read them, and `prepare.py` reports whether tracing is on. Each run on a thread is a trace
of its own (the request, the resume after each answer, the follow-up), grouped in LangSmith under the thread's id. A
trace shows the coordinator, each locator's `task` as a nested run with its tool calls, and a `MicroVM clone` (or
`MicroVM resume`) span where a locator first uses its MicroVM.

## Kept, redacted, skipped

A finalized invoice cannot be edited and is kept for the tax retention period, so the classifier marks the invoices
`retain` and their customer `redact`: `stripe_post_customers_customer` clears the customer's name, email, phone and
address and keeps the record the invoices point at. Deleting the customer would break that link; the policy denies it.
A record that matches the name but not the email address belongs to someone else and is skipped. The report records
each of these with its reason, so it can be shown without exposing the person it is about. It is written when the
coordinator calls `save_report`; a refusal in a later turn (the follow-up) is in the console's audit, not in the report.

## Where the time goes

Coordinator on `anthropic-claude-4.6-sonnet`, locators on `anthropic-claude-haiku-4.5`; `run_script.py --approve --delete`
answers the approvals itself, so there is no waiting for a human.

- **One-off setup:** the template is built once, by `prepare.py` or by the first run if `prepare.py` was not run (the first run then takes longer).
- **The locators** each get a MicroVM cloned from the template (in parallel) and are paused when done; this is the first
  phase of the run.
- **The rest is the coordinator's work:** five approvals, verification and the report. With real approvers the wait for
  a human adds to it.
- **The whole run**, request to the follow-up's refusal and clean-up, is about a minute or two.
