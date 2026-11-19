# Changelog

## Unreleased

**Breaking**

- Action Gateway tools use endpoints instead of sessions: `ActionGateway` creates an endpoint with its tools and policy (`POST /v2/action-gateway/endpoints`) and calls it with `X-Endpoint-Id` on the data plane the endpoint's `mcp_url` names (only `https` hosts on `do-ai.run` / `do-ai-test.run`; `DO_ACTION_GATEWAY_DATA_URL` or `data_url=` overrides). Renamed: `session_prefix=` to `endpoint_prefix=`, `session_id` to `endpoint_id`, `SESSION_PREFIX` to `ENDPOINT_PREFIX`, `ActionGatewayClient.list_sessions` / `delete_session` to `list_endpoints` / `delete_endpoint`, and `GatewayClient.create_session` / `delete_session` to `create_endpoint` / `delete_endpoint` (custom clients must rename them). `LocalActionGatewayClient` follows.

**Added**

- `scripts/setup_example_09.py`: sets up example 09's real mode step by step. It lists the commands it needs, asks for each token that is missing or refused (input hidden), checks it and writes it to `.env` (mode 0600); DigitalOcean tokens with scopes the setup does not need (Full Access, Read Only, extra scopes) are refused. It then builds the sandbox image, creates the public demo repository with `gh`, and runs `setup_droplet.py`, `prepare.py --clean` and `check_setup.py --examples 09`. `--dry-run` runs only read-only checks of what is set (tokens, team limit, image manifest, repository, droplet health) and exits 0 when they pass.

**Changed**

- `setup_droplet.py --deploy-bad` and `--verify-restore` exit with a message to run the setup first when its state file is missing. The example 12 README lists the setup step before `--deploy-bad`; the READMEs of examples 09, 10 and 11 say the first run builds the template when `prepare.py` was not run.
- Tool slugs need no version: a slug without `@version` gets the released one. The examples' policies and tool lists no longer pin versions; `name@version` still pins one.
- `scripts/check_setup.py --deep` calls one read-only action through a throwaway `microvm-ag-check-*` endpoint and deletes it.
- `scripts/smoke_action_gateway.py --slug-version` is optional (default: no pin).
- `scripts/grep_gate.sh` reads extra patterns from an untracked `.grep_gate_extra` (or `GREP_GATE_EXTRA`) and checks them in addition to its built-in ones.
- Example 12's broker logs the arguments, the exception type and the last call frames when a gateway call raises, so a 502 "needs the relay's gateway actor to connect an account first" can be traced to the call that raised it.
- Example 12's relay logs one `settle` event when it stops reading an incident whose specialists' hosts were still awake: `paused` (every host read as paused) or `limit` (the read window ended first), with each role's tile status and `since`.

**Fixed**

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
