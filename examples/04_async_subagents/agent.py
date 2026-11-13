"""A supervisor whose async subagents each run in their own DigitalOcean MicroVM.

Port of the Deep Agents course's Module 5 async subagents lab
(python/m5/async_lab in https://github.com/langchain-ai/lca-deepagents).

THE IDEA
The course lab points its AsyncSubAgent at ONE deployment you start by hand
(port 2025). This example keeps the supervisor the same in spirit -- same five
tools (start / check / update / cancel / list_async_task), same AsyncSubAgent
spec -- but points every spec at an embedded SubAgentGateway from
langchain_digitalocean_microvm.agent_host, which runs the subagents in agent hosts:

- DO_MICROVM_HOST_MODE=local (default): one local `langgraph dev` per TASK,
  standing in for a VM. No DigitalOcean credentials, no network.
- DO_MICROVM_HOST_MODE=microvm: real DigitalOcean MicroVMs running the
  agent-host image (images/agent-host/, DO_MICROVM_AGENT_IMAGE). To keep the
  number of MicroVMs low, each subagent TYPE gets one shared host
  (`reuse="pooled"`) and its tasks run side by side on it as separate threads.

Either way, finished runs are cached so checking never wakes a paused host,
and a host that dies turns its task into an error instead of a task that
says "running" forever. Files a subagent saves under /outputs/ stay in its
host; the supervisor's extra tool `fetch_subagent_artifacts` copies them into
its own filesystem (/subagents/<task id>/...) where `read_file` can see them.

The gateway starts on the supervisor's FIRST TURN, not at import: `langgraph
dev` re-imports this module on every reload and an imported-but-idle graph
should not spawn servers or open ports. It tears itself (and its local hosts)
down when the process exits.

Two subagent types, both served from images/agent-host/:
- echo        no model; answers "echo: <message>". Prefix the message with
              "sleep N:" to keep it busy for N seconds -- the way to watch a
              task run, update it, cancel it, or lose its host, for free --
              and/or "save NAME:" to write the rest to /outputs/NAME.
- researcher  a deep agent whose filesystem IS the VM: its own task directory
              (/work/tasks/<task id>, seen as "/"), keeping results under
              /outputs/. Needs DO_MODEL_ACCESS_KEY inside the host; passed via
              HostSpec(env=...).

RUN (from the repository root; needs the agent-host extra: uv sync --all-extras)
    cd examples/04_async_subagents && uv run langgraph dev
Then follow README.md: two echo tasks at once, check one while it runs,
update it, cancel the other, kill a host and watch the error arrive.
"""

import os

from deepagents import create_deep_agent

from langchain_digitalocean_microvm.agent_host import (
    GatewaySettings,
    HostSpec,
    LazyGateway,
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


model = make_model()


def _researcher_env() -> dict[str, str]:
    """Only the inference key crosses into a host (docs §4.8).

    The example reuses the supervisor's key so it works out of the box; anything
    closer to production should hand hosts a dedicated, rate-limited Model Access
    Key -- generated code inside the VM can read it.
    """
    key = os.environ.get("DO_MODEL_ACCESS_KEY")
    return {"DO_MODEL_ACCESS_KEY": key} if key else {}


settings = GatewaySettings.from_env()  # DO_MICROVM_HOST_MODE etc.
# Real MicroVMs: one shared host per subagent type keeps the example at two
# MicroVMs (echo + researcher). n_jobs_per_worker is how
# many of that type's tasks can run at once on the shared host.
pooled = settings.host_mode == "microvm"

gateway = LazyGateway(
    {
        "echo": HostSpec(graph_id="echo", reuse="pooled" if pooled else "per-task", n_jobs_per_worker=4 if pooled else 2),
        "researcher": HostSpec(
            graph_id="researcher", env=_researcher_env(), reuse="pooled" if pooled else "per-task"
        ),
    },
    settings=settings,
    # The gateway's own poll loop is what notices a finished run and caches it;
    # the default 15 s is fine for real MicroVMs, too sleepy for a demo.
    poll_interval=2.0,
    # An idle host is paused (SIGSTOP locally, checkpoint on a MicroVM) this
    # long after its last run; `update` wakes it again.
    pause_grace=60.0,
)

subagents = [
    gateway.subagent(
        name="echo",
        description=(
            "A model-free test subagent that replies 'echo: <your message>'. It costs "
            "nothing, so use it to demonstrate the task lifecycle. To make a task take "
            "time (so it can be checked while running, updated, or cancelled), start the "
            "message with 'sleep N:' where N is seconds, e.g. 'sleep 45: ping one'. To make "
            "it save a file, write 'save NAME: <text>' (after any sleep prefix); the text is "
            "stored as /outputs/NAME in its VM. Launch one task per message the user wants echoed."
        ),
    ),
    gateway.subagent(
        name="researcher",
        description=(
            "A deep agent that works alone in its own VM with a real shell and filesystem "
            "(no web access). Give it a self-contained brief; it saves anything worth keeping "
            "under /outputs/ and its final summary lists those files. Slow (minutes), so launch "
            "it as a background task and check later; use fetch_subagent_artifacts to copy its "
            "files into your own filesystem."
        ),
    ),
]


def build_graph(**kwargs):
    """The supervisor. `langgraph dev` serves `graph`; run_script.py builds one with a checkpointer."""
    return create_deep_agent(
        model=model,
        subagents=subagents,
        middleware=[gateway.middleware()],  # starts the gateway on the first turn
        tools=[gateway.artifacts_tool()],  # fetch_subagent_artifacts -> /subagents/<task id>/...
        **kwargs,
    )


# langgraph.json points at this module-level variable: "./agent.py:graph"
graph = build_graph()
