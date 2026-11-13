# images/agent-host/agent.py
"""The graphs served INSIDE a MicroVM agent host.

This package is what the baked agent image (and, in local mode,
langchain_digitalocean_microvm.agent_host.LocalDevServerHost) runs under the in-memory
LangGraph Agent Server. It is deliberately a separate, self-contained
deployment -- own pyproject.toml, `"dependencies": ["."]` -- modeled on the
Deep Agents course's async_lab/specialized_agent. See docs/design.md.

Two graphs are registered in langgraph.json:

- `echo`     : a model-free StateGraph that answers with "echo: <last human
               message>". Needs no API key; used by tests and health checks.
               A message starting with "sleep N:" is answered after N seconds
               (capped at MAX_ECHO_DELAY) so a supervisor can watch a task
               while it is still running, update it, cancel it, or lose its
               host mid-flight -- the same role the deliberate sleep in
               the course async_lab's analyst plays, without a model. Then
               "save NAME:" writes the rest of the message to the task's
               outputs/NAME, a model-free artifact for tests and smoke runs:
               "sleep 5: save notes.md: hello".
- `researcher`: a deep agent built by the `make_graph` factory at RUN time,
               so importing this module never requires DO_MODEL_ACCESS_KEY.
               The VM is the sandbox, so it uses a plain LocalShellBackend
               rooted at the task's directory (below).

Per-task directories. A pooled host serves several tasks of the
same type, so each task works in its own directory, keyed by the run's
LangGraph thread id (the supervisor's task id; the gateway creates the thread
on the host with the same id):

    $SUBAGENT_WORKDIR/tasks/<thread_id>/            the researcher's "/"
    $SUBAGENT_WORKDIR/tasks/<thread_id>/outputs/    what the supervisor can fetch

The Agent Server calls the `researcher` factory once per run with that run's
config, so the thread id comes from `config["configurable"]["thread_id"]`;
`echo` reads the same key from its node config. The layout is part of the
contract with langchain_digitalocean_microvm.agent_host.artifacts (TASKS_DIR, OUTPUTS_DIR).

Environment contract (the container's environment, set when MicroVMAgentHost
creates the MicroVM, or ./.env written by LocalDevServerHost in local mode):

    SUBAGENT_HOST_TOKEN          checked by auth.py on every request (a token file wins, see auth.py)
    DO_MODEL_ACCESS_KEY          DO Gradient inference key (researcher only)
    SUBAGENT_MODEL               model id, default anthropic-claude-haiku-4.5
    SUBAGENT_INFERENCE_BASE_URL  default https://inference.do-ai.run
    SUBAGENT_WORKDIR             default /work
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

DEFAULT_MODEL = "anthropic-claude-haiku-4.5"
DEFAULT_INFERENCE_BASE_URL = "https://inference.do-ai.run"
DEFAULT_WORKDIR = "/work"
MAX_ECHO_DELAY = 300.0
TASKS_DIR = "tasks"
OUTPUTS_DIR = "outputs"
_SLEEP_PREFIX = re.compile(r"^\s*sleep\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?\s*[:;,-]?\s*", re.IGNORECASE)
_SAVE_PREFIX = re.compile(r"^\s*save\s+([^\s:]+)\s*:\s?", re.IGNORECASE)
_SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

SYSTEM_PROMPT = """You are a research subagent running in your own isolated VM.
Work in the filesystem you have been given; put anything the caller should keep
under /outputs/ (in shell commands that directory is ./outputs, relative to the
current directory). Finish with a short summary and list every file you produced
under /outputs/ (full paths), because that summary is all the caller sees; the
caller can fetch those files afterwards."""


# ----------------------------------------------------------------------------
# echo: no model, no filesystem -- proves the server, auth, and protocol work.
# ----------------------------------------------------------------------------


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal content blocks
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def parse_echo_request(text: str) -> tuple[float, str]:
    """Split an optional "sleep N:" prefix off an echo request.

    "sleep 20: ping" -> (20.0, "ping"); "ping" -> (0.0, "ping"). The delay is
    capped at MAX_ECHO_DELAY so a typo cannot park a host for an hour.
    """
    match = _SLEEP_PREFIX.match(text)
    if match is None:
        return 0.0, text
    return min(float(match.group(1)), MAX_ECHO_DELAY), text[match.end() :]


def parse_save_request(text: str) -> tuple[str | None, str]:
    """Split an optional "save NAME:" prefix off an echo request.

    "save notes.md: hi" -> ("notes.md", "hi"); "hi" -> (None, "hi"). NAME is
    a relative path under the task's outputs/ directory; anything that could
    leave it (absolute, "..") raises ValueError.
    """
    match = _SAVE_PREFIX.match(text)
    if match is None:
        return None, text
    name = match.group(1)
    parts = Path(name).parts
    if name.startswith("/") or not parts or any(part in ("..", ".") for part in parts):
        raise ValueError(f"save: {name!r} must be a relative path inside outputs/")
    return name, text[match.end() :]


def thread_id_of(config: RunnableConfig | None) -> str | None:
    """The run's thread id (= the supervisor's task id), if the server passed one."""
    value = ((config or {}).get("configurable") or {}).get("thread_id")
    return str(value) if value else None


def task_dir(thread_id: str | None) -> Path:
    """The task's working directory, created with its outputs/ on first use.

    Without a thread id (the server also builds graphs for schema and
    assistant calls) this is the shared workdir, and nothing is created.
    """
    root = Path(os.environ.get("SUBAGENT_WORKDIR", DEFAULT_WORKDIR))
    if thread_id is None:
        return root
    if not _SAFE_THREAD_ID.match(thread_id) or thread_id in (".", ".."):
        raise ValueError(f"unexpected thread id {thread_id!r}")
    path = root / TASKS_DIR / thread_id
    (path / OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    return path


def _save_output(thread_id: str | None, name: str, text: str) -> str:
    if thread_id is None:
        raise ValueError("save: this run has no thread id, so it has no outputs/ directory")
    target = task_dir(thread_id) / OUTPUTS_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return f"/{OUTPUTS_DIR}/{name}"


async def _echo_node(state: MessagesState, config: RunnableConfig) -> dict:
    last = state["messages"][-1] if state["messages"] else ""
    delay, text = parse_echo_request(_message_text(last))
    if delay > 0:
        await asyncio.sleep(delay)  # async so `runs.cancel` interrupts it immediately
    try:
        name, text = parse_save_request(text)
        saved = await asyncio.to_thread(_save_output, thread_id_of(config), name, text) if name else None
    except (ValueError, OSError) as exc:
        return {"messages": [AIMessage(content=f"echo: error: {exc}")]}
    suffix = f" (saved {saved})" if saved else ""
    return {"messages": [AIMessage(content=f"echo: {text}{suffix}")]}


echo = (
    StateGraph(MessagesState)
    .add_node("echo", _echo_node)
    .add_edge(START, "echo")
    .add_edge("echo", END)
    .compile()
)


# ----------------------------------------------------------------------------
# researcher: a deep agent, built lazily so a missing model key fails the RUN,
# not the server start.
# ----------------------------------------------------------------------------


def make_graph(config: RunnableConfig):
    """Graph factory registered as `researcher` in langgraph.json; called once per run.

    The backend is rooted at this run's task directory, so the agent's "/"
    (and its /outputs/) belongs to this task alone, even on a pooled host.
    """
    from deepagents import (
        create_deep_agent,  # noqa: PLC0415 - keep import cost off the echo path
    )
    from deepagents.backends import LocalShellBackend  # noqa: PLC0415
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    api_key = os.environ.get("DO_MODEL_ACCESS_KEY")
    if not api_key:
        raise RuntimeError(
            "DO_MODEL_ACCESS_KEY is not set inside this agent host; the bootstrap must "
            "write it to the env file (HostSpec.env or DO_MICROVM_SUBAGENT_ENV_PASSTHROUGH)."
        )
    model = ChatAnthropic(
        model=os.environ.get("SUBAGENT_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("SUBAGENT_INFERENCE_BASE_URL", DEFAULT_INFERENCE_BASE_URL),
        api_key=api_key,
    )
    return create_deep_agent(
        model=model,
        backend=LocalShellBackend(root_dir=task_dir(thread_id_of(config)), virtual_mode=True),
        system_prompt=SYSTEM_PROMPT,
    )
