"""A chat model that replays a script instead of calling an LLM.

deepagents' AsyncSubAgentMiddleware is driven entirely by the model's tool
calls (start / check / update / cancel / list_async_task). Feeding it a
scripted model lets a test exercise those tools end to end -- the real
middleware, the real LangGraph SDK client, the real gateway -- with no model
key and deterministic tool arguments. A step is either a ready AIMessage or a
callable that builds one from the prompt, for arguments (a task_id) that only
exist once an earlier tool has answered.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]
_TASK_ID = re.compile(r"task_id: (\S+)")


class ScriptedChatModel(BaseChatModel):
    """Replays `steps` one model call at a time; fails loudly when the script runs out."""

    steps: list[Any] = Field(default_factory=list)
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: set[str] = Field(default_factory=set)

    def script(self, *steps: Step) -> ScriptedChatModel:
        self.steps.extend(steps)
        return self

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        for tool in tools:
            name = _tool_name(tool)
            if name:
                self.bound_tools.add(name)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        if not self.steps:
            raise AssertionError(f"Scripted model ran out of steps; last message was: {messages[-1]!r}")
        step = self.steps.pop(0)
        message = step(messages) if callable(step) else step
        unknown = [call["name"] for call in message.tool_calls if self.bound_tools and call["name"] not in self.bound_tools]
        if unknown:
            raise AssertionError(f"Script calls tools the agent does not have: {unknown}; available: {sorted(self.bound_tools)}")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def _tool_name(tool: Any) -> str | None:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(tool, dict):
        if isinstance(tool.get("name"), str):
            return tool["name"]
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return None


# ---------------------------------------------------------------------------
# step builders and prompt readers
# ---------------------------------------------------------------------------


def tool_calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    """One AIMessage carrying several tool calls (the model launching tasks in parallel)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"} for name, args in calls
        ],
    )


def tool_call(name: str, **args: Any) -> AIMessage:
    return tool_calls((name, args))


def say(text: str) -> AIMessage:
    """A final answer with no tool calls: ends the agent's turn."""
    return AIMessage(content=text)


def tool_messages(messages: Iterable[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def last_tool_message(messages: Iterable[BaseMessage]) -> ToolMessage:
    found = tool_messages(messages)
    if not found:
        raise AssertionError("no ToolMessage in the conversation")
    return found[-1]


def task_ids(messages: Iterable[BaseMessage]) -> list[str]:
    """Every `task_id: ...` the async tools reported, first occurrence first (what a real model would read)."""
    seen: dict[str, None] = {}
    for message in tool_messages(messages):
        for match in _TASK_ID.finditer(str(message.content)):
            seen.setdefault(match.group(1), None)
    return list(seen)


def check_result(message: ToolMessage) -> dict[str, Any]:
    """`check_async_task` answers with a JSON object: {"status", "thread_id", "result" | "error"}."""
    return json.loads(str(message.content))
