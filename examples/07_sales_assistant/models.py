"""The two models of the assistant: DigitalOcean Serverless Inference with DO_MODEL_ACCESS_KEY, else EXAMPLES_MODEL.

`model` (fast) runs the specialists and `strong_model` the orchestrating agents, as in the course.
"""

from __future__ import annotations

import os


def make_model(strong: bool = False):
    """DigitalOcean Serverless Inference with DO_MODEL_ACCESS_KEY, else EXAMPLES_MODEL (any init_chat_model id)."""
    if os.environ.get("DO_MODEL_ACCESS_KEY"):
        from langchain_anthropic import ChatAnthropic

        name = os.environ.get("DO_INFERENCE_MODEL", "anthropic-claude-haiku-4.5")
        if strong:
            name = os.environ.get("DO_INFERENCE_STRONG_MODEL", "anthropic-claude-4.6-sonnet")
        return ChatAnthropic(model=name, base_url="https://inference.do-ai.run", api_key=os.environ["DO_MODEL_ACCESS_KEY"])
    from langchain.chat_models import init_chat_model

    name = os.environ.get("EXAMPLES_MODEL", "anthropic:claude-haiku-4-5")
    if strong:
        name = os.environ.get("EXAMPLES_STRONG_MODEL", name)
    return init_chat_model(name)


model = make_model()
strong_model = make_model(strong=True)
