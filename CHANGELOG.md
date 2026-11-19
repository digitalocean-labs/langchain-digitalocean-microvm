# Changelog

## Unreleased

**Changed**

- `examples/tour.py` runs example 11 with its live dashboard by default (about 2 minutes): on macOS the dashboard opens in a new Terminal or iTerm window and stays open after the run, elsewhere or with `--same-terminal` it runs full screen in this terminal. The output ends with the gateway audit and the report's path. `--full` adds examples 01, 03, 06 and 10.
- The dashboards of examples 08, 09, 10 and 11 say in their header what the run is about, and their bottom panel is titled "MicroVM lifecycle (cloned, resumed, paused, deleted)".
- Action Gateway: the client sends an endpoint's calls to the `mcp_url` host only when it is an `https` host on `do-ai.run` or `do-ai-test.run`. With `DO_ACTION_GATEWAY_DATA_URL` (or `data_url=`) set, every call goes to that host and the `mcp_url` is not checked.

**Fixed**

- `scripts/check_setup.py`: the per-example table's first column is as wide as the longest example label.

## 0.2.0

Supports `deepagents>=0.7.0,<0.8.0`. The Action Gateway tools use endpoints (breaking: names below); `scripts/setup_examples.py` sets up and checks the examples' real runs.

**Breaking**

- Action Gateway tools use endpoints instead of sessions: `ActionGateway` creates an endpoint with its tools and policy (`POST /v2/action-gateway/endpoints`) and calls it with `X-Endpoint-Id` on the data plane the endpoint's `mcp_url` names (an `https` host on `do-ai.run`; `DO_ACTION_GATEWAY_DATA_URL` or `data_url=` overrides). Renamed: `session_prefix=` to `endpoint_prefix=`, `session_id` to `endpoint_id`, `SESSION_PREFIX` to `ENDPOINT_PREFIX`, `ActionGatewayClient.list_sessions` / `delete_session` to `list_endpoints` / `delete_endpoint`, and `GatewayClient.create_session` / `delete_session` to `create_endpoint` / `delete_endpoint` (custom clients must rename them). `LocalActionGatewayClient` follows.

**Added**

- `scripts/setup_examples.py [--examples 09,07]`: sets up what the selected examples need for a real run, step by step, and runs `check_setup.py` for them at the end. Without `--examples`, the common base: the MicroVMs control and access tokens, the model access key, the sandbox image and room on the team (examples 01, 03, 05, 06, 08, 10 and 11). Per example: 09 the Action Gateway token and actor, the public demo repository (`gh`), the droplet (`setup_droplet.py`) and `prepare.py --clean`; 12 the same incident steps, plus `DO_MICROVM_HOST_MODE=microvm`, the commander's and the specialists' images (built with `build_image.py --spec`, and checked against their sources by the content hash their tag carries) and the broker URL (the tunnel command is printed, the URL asked for); 04 `DO_MICROVM_HOST_MODE=microvm` and the default agent-host image (`DO_MICROVM_AGENT_IMAGE`, checked against its sources the same way); 02 the `dcode` CLI; 07 the optional Tavily key. It lists the commands it needs, asks for each token that is missing or refused (input hidden), checks it and writes it to `.env` (mode 0600); a DigitalOcean token with scopes the setup does not need (Full Access, Read Only, extra scopes) is reported, and a pasted one is saved only if you confirm. `--dry-run` runs only read-only checks of what is set (tokens, team limit, image manifest; for 09 and 12 the actor's DigitalOcean and GitHub connections, the repository and the droplet's health; for 12 the broker URL (one GET: a name that does not resolve, no answer or a 530 fails; any other answer passes); for 04 and 12 the agent-host images against their sources; for 12 the `microvm-ag-*` endpoints left by earlier runs) and exits 0 when they pass.

**Changed**

- Examples 09 and 12: the incident's droplet is named `checkout-svc` and its snapshot `checkout-svc-snapshot` (`incident_common/world.py`, `setup_droplet.py`); the snapshot's name does not say which state it holds, so the Infra Analyst picks it by its date. A droplet and snapshot created with the earlier names keep them until `setup_droplet.py --cleanup` and a new setup.
- `setup_droplet.py --deploy-bad` and `--verify-restore` exit with a message to run the setup first when its state file is missing. The example 12 README lists the setup step before `--deploy-bad`; the READMEs of examples 09, 10 and 11 say the first run builds the template when `prepare.py` was not run.
- Tool slugs need no version: the package sends a slug without `@version` as it is, and `name@version` pins one. The examples' policies and tool lists name tools without a version.
- `scripts/check_setup.py --deep` calls one read-only action through a throwaway `microvm-ag-check-*` endpoint and deletes it.
- `scripts/smoke_action_gateway.py --slug-version` is optional (default: no pin).
- `scripts/grep_gate.sh` reads extra patterns from an untracked `.grep_gate_extra` (or `GREP_GATE_EXTRA`) and checks them in addition to its built-in ones.
- Example 12's broker logs the arguments, the exception type and the last call frames when a gateway call raises, so a 502 "needs the relay's gateway actor to connect an account first" can be traced to the call that raised it.
- Example 12's relay logs one `settle` event when it stops reading an incident whose specialists' hosts were still awake: `paused` (every host read as paused) or `limit` (the read window ended first), with each role's tile status and `since`.
- Example 12's broker calls a read-only role's action with `ActionGateway.request` and sends it again once, after 3 s, when the gateway answers `Not connected`. The specialist receives the answer's text, where an interrupt outside a graph used to end in a 502.

**Fixed**

- `scripts/setup_examples.py`: before it asks to replace the demo repository's commits, it checks that the `gh` login can push to the repository, and a failed push shows git's reason. A rebuild that does not happen (no push access, a confirmation that does not match, a failed push) keeps a repository that passes its check, and the run continues. A step the run does not reach because an earlier one did not finish (the droplet, the final `check_setup.py`) is reported as skipped, with the step that stopped it, and added to the to-do list.
- Example 12 approvals page: one log read per incident at a time, and a gateway or decision row already shown is not added again. An approval whose decision is in flight keeps its buttons disabled through redraws; they are enabled again only when the decision was refused or failed.
- Example 12 relay: an incident read between its commander's host coming up and the thread being created answers with status `starting` (`GET /incidents/<id>`) or a 409 (a message), not a 500.
- Example 12 relay: when the commander's host answer to a decision is lost (the connection drops), the relay reads the sub-agent's task again. If the host took the decision, the decision stands and its run id is returned; if the approval still waits, `POST /incidents/<id>/decision` answers 502 and the approval is offered again (the broker returns its recorded outcome, so a second decision runs nothing again).

## 0.1.0

First release. Supports `deepagents>=0.7.0,<0.8.0`.

**Sandboxes**

- `MicroVMSandbox` (a deepagents `BaseSandbox`) and `MicroVMLifecycle`: get-or-create, seed, pause, resume, delete.
- Transports (`DO_MICROVM_TRANSPORT`): `endpoint` (the default; commands and files over the MicroVM's HTTPS endpoint, needs an access token), `exec` (the MicroVMs exec API) and `local` (a local directory, no credentials).
- Sandbox templates (`build_template`, `TemplateCache`): prepare a sandbox once, checkpoint it and clone new sandboxes from it.
- `ThreadSandboxes`: one sandbox per conversation thread, paused after each run.
- `DO_MICROVM_NAMESPACE` and owner tags keep one user's or deployment's MicroVMs apart on a shared team; `scripts/delete_all.py` and `scripts/gc_microvms.py` clean up only what you created.

**Subagents and agent hosts** (experimental)

- One MicroVM per synchronous subagent, per thread or per task.
- Async subagents in MicroVM agent hosts behind an embedded Agent Protocol gateway (the `[agent-host]` extra), including runs that pause for a human (`interrupt()`).

**Integrations**

- `dcode --sandbox microvm`: a Deep Agents Code sandbox provider.
- Action Gateway tools for a supervisor agent (`integrations.action_gateway`): an allow / ask / deny `ActionPolicy` (loadable from YAML), where `ask` becomes a LangGraph interrupt a human approves or denies, and an in-memory local gateway that needs no credentials.

**Examples and scripts**

- Twelve examples in `examples/`, from one agent in one MicroVM to an incident-response team; 01, 04, 05 and 07 port labs of the LangChain Deep Agents course.
- `examples/tour.py` runs examples in local mode; `scripts/check_setup.py` reports what each example still needs.
