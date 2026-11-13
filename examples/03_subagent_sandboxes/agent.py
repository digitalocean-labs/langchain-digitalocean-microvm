"""Ordinary (synchronous) subagents, each with its own DigitalOcean MicroVM.

THE IDEA
../04_async_subagents moves whole subagents INTO MicroVMs (each runs its own
Agent Server there; the supervisor talks to them over HTTP). This lab keeps the
classic synchronous `task` tool: the subagents' LLM loops run right here, in
the supervisor's process, but every subagent gets a MicroVM of its own as its
filesystem and shell. Whatever `analyst` runs or writes happens in its VM;
`reviewer` cannot see it, and neither can the supervisor, except through the
files-only mount below.

- analyst   has a shell (`execute`) and a seeded dataset (/data/sales.csv);
            writes scripts and results under /outputs/.
- reviewer  reviews what it finds under /inputs/ and writes /outputs/review.md.
- the supervisor's own filesystem is its state, plus one CompositeBackend
  route per subagent: /subagents/analyst/... IS the analyst VM's /..., and the
  same for reviewer. So handing the analysis to the reviewer is: read
  /subagents/analyst/outputs/x, write /subagents/reviewer/inputs/x. The
  supervisor has no `execute`: commands only ever run in a subagent's VM.

PER TURN (the graph factory below; design doc §5.1)
`langgraph dev` enters `make_graph` for every run of a thread. It gets or
creates the thread's two sandboxes in parallel (resuming them if paused),
builds the agent against them, and pauses both when the run ends. Each thread
has its own pair; a new thread means two new MicroVMs. Introspection calls
(Studio drawing the graph) get the same graph with no sandboxes at all.

MODES (.env)
- DO_MICROVM_TRANSPORT=local: sandboxes are local directories under
  $TMPDIR/microvm_sandbox_local/; no credentials, no network.
- DO_MICROVM_TRANSPORT=endpoint (default when unset) or exec: real MicroVMs from
  DO_MICROVM_IMAGE: a linux/amd64 image that keeps running, has sh, coreutils
  and python3, and whose user can write to `/` (subagents write /outputs/...,
  deepagents offloads to /large_tool_results/): the image
  scripts/build_sandbox_image.py pushes
  (python:3.12-slim as root, port 8080). The microvm-subagent image runs as
  uid 1000 and cannot write there. DO_MICROVM_* set in the shell win over
  .env: `DO_MICROVM_TRANSPORT=exec uv run ...`. The endpoint transport also
  needs DO_MICROVM_ACCESS_TOKEN.
  Paused MicroVMs count against your team's MicroVM limit: delete a thread's
  pair when done (run_script.py --delete, scripts/delete_all.py, or
  scripts/gc_microvms.py once unused).

RUN (from the repository root)
    cd examples/03_subagent_sandboxes && uv run langgraph dev
or  uv run examples/03_subagent_sandboxes/run_script.py
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime

from langchain_digitalocean_microvm import MicroVMLifecycle
from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    default_db_path,
)
from langchain_digitalocean_microvm.subagents import (
    SubagentSandboxes,
    asubagent_sandboxes,
)


def make_model():
    """DigitalOcean Serverless Inference with DO_MODEL_ACCESS_KEY, else EXAMPLES_MODEL (any init_chat_model id)."""
    if os.environ.get("DO_MODEL_ACCESS_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("DO_INFERENCE_MODEL", "anthropic-claude-haiku-4.5"),
            base_url="https://inference.do-ai.run",
            api_key=os.environ["DO_MODEL_ACCESS_KEY"],
        )
    from langchain.chat_models import init_chat_model

    return init_chat_model(os.environ.get("EXAMPLES_MODEL", "anthropic:claude-haiku-4-5"))


model = make_model()  # built at import so every subagent and the supervisor share one client

# Subagent specs in the plain deepagents format. asubagent_sandboxes() later returns
# copies of these with a per-thread sandbox backend attached to each.
SUBAGENTS: list[dict[str, Any]] = [
    {
        "name": "analyst",
        "description": (
            "A data analyst with its own Linux VM (shell + python3). It has /data/sales.csv. "
            "Give it a self-contained question; it writes scripts and results under /outputs/ "
            "in its VM and reports the paths."
        ),
        "system_prompt": (
            "You are a data analyst working alone in your own Linux VM. Use `execute` to run "
            "python3 (standard library only, no network) and keep every result you produce "
            "under /outputs/. Finish with a short summary listing the files you wrote."
        ),
    },
    {
        "name": "reviewer",
        "description": (
            "A careful reviewer with its own VM. Put the material to review under its /inputs/ "
            "first (as the supervisor: /subagents/reviewer/inputs/...); it writes /outputs/review.md."
        ),
        "system_prompt": (
            "You review analyses. Read everything under /inputs/, check the numbers where you "
            "can (you may run python3 with `execute`), and write a short review to "
            "/outputs/review.md. Answer with the review's main points."
        ),
    },
]

SALES_CSV = b"""month,region,units,revenue
2026-07,emea,120,14400
2026-07,amer,180,23400
2026-08,emea,95,11400
2026-08,amer,210,27300
2026-09,emea,140,16800
2026-09,amer,160,20800
"""
# Uploaded only when a sandbox is created, never on resume.
SEED_FILES = {
    "analyst": [("/data/sales.csv", SALES_CSV)],
    "reviewer": [("/inputs/README.md", b"Material to review is placed here by the supervisor.\n")],
}

SYSTEM_PROMPT = (
    "You coordinate two subagents through the `task` tool. Each has its own isolated VM; "
    "you cannot run commands yourself. You can see each subagent's files: "
    "/subagents/analyst/ is the analyst's filesystem and /subagents/reviewer/ the "
    "reviewer's. To pass work from one to the other, read the file from the first and "
    "write it into the second (e.g. /subagents/reviewer/inputs/...) before delegating."
)

lifecycle = MicroVMLifecycle()  # DO_MICROVM_* from the environment (.env: langgraph.json or run_script.py)
# Real MicroVMs: record each sandbox's last use where scripts/gc_microvms.py looks.
store = None if lifecycle.config.transport == "local" else GatewayStore(default_db_path("microvm", lifecycle.config.namespace))


def build_supervisor(sandboxes: SubagentSandboxes | None, **kwargs: Any):
    """The supervisor for one turn; without sandboxes (introspection) the subagents use its state."""
    if sandboxes is None:
        # Introspection only (Studio drawing the graph): nothing is provisioned, nothing runs.
        return create_deep_agent(model=model, system_prompt=SYSTEM_PROMPT, subagents=SUBAGENTS, **kwargs)
    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        subagents=list(sandboxes.subagents),
        backend=sandboxes.mount(StateBackend()),  # /subagents/<name>/ -> that subagent's VM, files only
        **kwargs,
    )


@contextlib.asynccontextmanager
async def sandboxes_for(thread_id: str):
    """This thread's sandboxes for one turn: provisioned in parallel, paused on exit."""
    async with asubagent_sandboxes(
        lifecycle, SUBAGENTS, thread_id=thread_id, default_model=model, seed_files=SEED_FILES, store=store,
    ) as sandboxes:
        yield sandboxes


# https://docs.langchain.com/langsmith/graph-rebuild#context-manager-factory
@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    if not runtime.execution_runtime:
        yield build_supervisor(None)
        return
    thread_id = config.get("configurable", {}).get("thread_id")
    async with sandboxes_for(thread_id) as sandboxes:
        yield build_supervisor(sandboxes)
