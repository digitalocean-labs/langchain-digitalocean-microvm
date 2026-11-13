"""Subagent-sandbox graph factory served by a real `langgraph dev` in test_subagents_e2e.py.

No model key: `PolicyModel` decides each step from the conversation. The parent
delegates to `reviewer`, which writes a note in its own sandbox; on a later
turn the parent reads the note back through /subagents/reviewer/. Fake
sandboxes (no network); the store path comes from PHASE2_E2E_DB.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle
from langchain_digitalocean_microvm.agent_host.store import GatewayStore
from langchain_digitalocean_microvm.subagents import asubagent_sandboxes

NOTE = "/notes/n.md"


def _call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{uuid.uuid4().hex[:8]}", "type": "tool_call"}])


class PolicyModel(BaseChatModel):
    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        system = " ".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        since_human: list[Any] = []
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                human = str(m.content)
                break
            since_human.insert(0, m)
        tools = [m for m in since_human if isinstance(m, ToolMessage)]
        if "SUBAGENT" in system:  # the reviewer, inside its sandbox
            step = _call("write_file", file_path=NOTE, content=human) if not tools else AIMessage(content="saved")
        elif human.startswith("save "):
            step = _call("task", subagent_type="reviewer", description=human[5:]) if not tools else AIMessage(content="delegated")
        elif not tools:
            step = _call("read_file", file_path=f"/subagents/reviewer{NOTE}")
        else:
            step = AIMessage(content=f"note: {tools[-1].content}")
        return ChatResult(generations=[ChatGeneration(message=step)])

    @property
    def _llm_type(self) -> str:
        return "policy"


model = PolicyModel()
SPECS = [{"name": "reviewer", "description": "keeps notes", "system_prompt": "SUBAGENT: write notes."}]
lifecycle = MicroVMLifecycle(MicroVMConfig(transport="local"))
store = GatewayStore(os.environ["PHASE2_E2E_DB"])


@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):  # type: ignore[no-untyped-def]
    if not runtime.execution_runtime:
        yield create_deep_agent(model=model, system_prompt="PARENT", subagents=SPECS)
        return
    thread_id = config["configurable"]["thread_id"]
    async with asubagent_sandboxes(lifecycle, SPECS, thread_id=thread_id, default_model=model, store=store) as sb:
        yield create_deep_agent(model=model, system_prompt="PARENT", subagents=list(sb.subagents),
                                backend=sb.mount(StateBackend()))
