# Changelog

## Unreleased

**Breaking**

- Action Gateway tools use endpoints instead of sessions: `ActionGateway` creates an endpoint with its tools and policy (`POST /v2/action-gateway/endpoints`) and calls it with `X-Endpoint-Id` on the data plane the endpoint's `mcp_url` names (only `https` hosts on `do-ai.run` / `do-ai-test.run`; `DO_ACTION_GATEWAY_DATA_URL` or `data_url=` overrides). Renamed: `session_prefix=` to `endpoint_prefix=`, `session_id` to `endpoint_id`, `SESSION_PREFIX` to `ENDPOINT_PREFIX`, `ActionGatewayClient.list_sessions` / `delete_session` to `list_endpoints` / `delete_endpoint`, and `GatewayClient.create_session` / `delete_session` to `create_endpoint` / `delete_endpoint` (custom clients must rename them). `LocalActionGatewayClient` follows.

**Changed**

- Tool slugs need no version: a slug without `@version` gets the released one. The examples' policies and tool lists no longer pin versions; `name@version` still pins one.
- `scripts/check_setup.py --deep` calls one read-only action through a throwaway `microvm-ag-check-*` endpoint and deletes it.
- `scripts/smoke_action_gateway.py --slug-version` is optional (default: no pin).
- `scripts/grep_gate.sh` reads extra patterns from an untracked `.grep_gate_extra` (or `GREP_GATE_EXTRA`) and checks them in addition to its built-in ones.

**Fixed**

- Example 12 relay: an incident read between its commander's host coming up and the thread being created answers with status `starting` (`GET /incidents/<id>`) or a 409 (a message), not a 500.

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
