# Agent-host image

The deployment that runs **inside** a MicroVM agent host: an in-memory
LangGraph Agent Server serving two graphs behind a per-host token.

| Graph | What it is | Needs |
| --- | --- | --- |
| `echo` | Model-free `StateGraph` answering `echo: <your message>`; a `sleep N:` prefix delays the answer N seconds (max 300) so a task can be observed while running; then a `save NAME:` prefix writes the rest of the message to the task's `outputs/NAME` | nothing |
| `researcher` | Deep agent on a `LocalShellBackend` rooted at its task's directory (the VM is the sandbox) | `DO_MODEL_ACCESS_KEY` at run time |

Every task gets its own directory, keyed by the run's thread id (the
supervisor's task id; the gateway creates the thread on the host with that id):
`$SUBAGENT_WORKDIR/tasks/<thread id>/`, which the researcher sees as `/`, with
the files the supervisor can fetch under `outputs/`. The Agent Server calls the
`researcher` factory once per run with that run's config, which is where the
thread id comes from. A pooled host serving several tasks therefore keeps
their files apart. `langchain_digitalocean_microvm.agent_host.artifacts` reads the same
layout (`TASKS_DIR`, `OUTPUTS_DIR`); change both together.

`auth.py` rejects any Agent Protocol request whose `x-api-key` does not equal
the host token: `SUBAGENT_HOST_TOKEN`, or the token file `$SUBAGENT_HOST_TOKEN_FILE` when it exists. `langgraph.json` sets `"disable_studio_auth": true`,
which is required: without it any request carrying `x-auth-scheme: langsmith`
(the header the deepagents middleware sends) would bypass the check under
`langgraph dev`. The meta routes `/ok`, `/`, `/info`, `/metrics`, `/docs` and `/openapi.json` are open by
design; see `auth.py`'s docstring.

`artifacts_app.py` adds two routes to the same server (`"http"` in
`langgraph.json`, with `enable_custom_route_auth`, so `auth.py` checks them
too, and the app checks the token again): `GET /microvm/artifacts/<task id>`
lists the regular files under the task's `outputs/`, and
`GET /microvm/artifacts/<task id>/files/<path>` returns one, with `Range`
support. Nothing there follows a symlink. The gateway's `/artifacts` routes
read through them.

## Run it locally

```bash
cd images/agent-host
uv sync
SUBAGENT_HOST_TOKEN=dev-token uv run langgraph dev --port 2024 --no-browser --no-reload \
  --n-jobs-per-worker 2 --allow-blocking
curl -H "x-api-key: dev-token" http://127.0.0.1:2024/ok
```

`langgraph.json` has no `env` file: the server reads its variables from the
process environment (your shell locally; the container's environment, set at
create time, on a MicroVM).
`langchain_digitalocean_microvm.agent_host.LocalDevServerHost` automates all of this for
tests and examples (own working directory per host, generated `langgraph.json`
with absolute paths, `./.env`, `SIGSTOP`/`SIGCONT` for pause/resume).
The package's wheel ships a copy of `agent.py`, `auth.py`, `artifacts_app.py` and
`langgraph.json` (as `agent_host/_template/`, built from this directory), which
`LocalDevServerHost` serves when no source checkout is around; this directory stays
the only source copy.

Design: [`docs/design.md`](../../docs/design.md) (in-MicroVM runtime, auth, per-task directories and
artifacts).

## Build the image (MicroVMs run `linux/amd64` containers)

Rebuild the image after changing any file in this directory; the new tag goes in
`DO_MICROVM_AGENT_IMAGE`.

Without Docker (needs `uv` and [`crane`](https://github.com/google/go-containerregistry)):

```bash
python3 images/agent-host/build_image.py \
  --repository registry.digitalocean.com/<registry>/microvm-subagent
```

It installs the locked dependencies as `linux/amd64` wheels, appends them and
the four source files as one layer on `python:3.12-slim`, sets the
entrypoint (`python -m langgraph_cli dev --host 0.0.0.0 --port 2024 ...`,
user 1000, `HOME=/work`) and pushes. The tag is a hash of the layer's
contents. Registry credentials come from `DO_REGISTRY_TOKEN` in `.env`
and live only in a temporary directory during the push. With Docker,
`Dockerfile` builds the same image:

```bash
docker buildx build --platform linux/amd64 -t registry.digitalocean.com/<registry>/microvm-subagent:<tag> --push images/agent-host
```

On a MicroVM the variables (`SUBAGENT_HOST_TOKEN`, `DO_MODEL_ACCESS_KEY`,
`N_JOBS_PER_WORKER`) arrive as the container's environment, set once by
`MicroVMAgentHost` when it creates the VM.

### Another agent host's image (`--spec`)

The same script builds an image for other graphs on the same base, entrypoint and
`auth.py`. A JSON spec names a uv project (its locked dependencies; a path
dependency such as this package is built into a wheel) and the files to copy, which
land under `/opt/agent/` laid out as in the repository, so the relative paths in its
`langgraph.json` hold both locally and in the image. The server runs from that
`langgraph.json`'s directory: the Agent Server reads the file's relative paths from
its working directory.

```bash
python3 images/agent-host/build_image.py --spec examples/12_incident_response_agents/host/image.json \
  --repository registry.digitalocean.com/<registry>/microvm-incident-agents-commander
```

The tag is a hash of everything in the layer, the dependencies included, so rebuild
after changing any file the spec lists or the package itself.

Design: [`docs/design.md`](../../docs/design.md) (real API, runtime, auth).

