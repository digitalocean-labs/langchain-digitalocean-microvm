"""A deep agent whose filesystem and shell are a DigitalOcean MicroVM, across a pause/resume.

Port of the Deep Agents course's sandbox lesson (python/m2/m2.3_sandbox_agent.py
in https://github.com/langchain-ai/lca-deepagents), which uses a LangSmith
sandbox for a create -> use -> delete cycle. This version shows what MicroVMs
add: the sandbox is created once, seeded with a file, used by an agent,
PAUSED (memory, files and processes kept; no compute billed), then RESUMED,
and a second turn shows both files survived.

    uv run examples/01_quickstart/quickstart.py            # real MicroVM (DO_MICROVM_* in .env)
    DO_MICROVM_TRANSPORT=local uv run examples/01_quickstart/quickstart.py   # no DigitalOcean account

The sandbox is left paused so a re-run reuses it; delete it with
`uv run examples/01_quickstart/quickstart.py --delete`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv

from langchain_digitalocean_microvm import MicroVMLifecycle, SandboxNotFoundError

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # the shell wins over .env

# Stable id: get_or_create() maps it to the same MicroVM on every run, so a re-run reuses the paused one.
SANDBOX_ID = "quickstart"


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


def ask(backend, prompt: str) -> None:
    # create_deep_agent binds the backend at build time, so a resumed sandbox
    # (a new transport connection) needs a new agent.
    agent = create_deep_agent(
        model=make_model(),
        backend=backend,
        system_prompt="You are a coding assistant. Quote file contents exactly in your final answer.",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    print(result["messages"][-1].content)


def main() -> None:
    lifecycle = MicroVMLifecycle()  # DO_MICROVM_* from the environment
    if "--delete" in sys.argv:
        try:
            lifecycle.delete(sandbox_id=SANDBOX_ID)
            print(f"deleted sandbox {SANDBOX_ID}")
        except SandboxNotFoundError:
            print(f"no sandbox {SANDBOX_ID} to delete (local sandboxes live only as long as the process)")
        return

    # Seed files are uploaded only when this call creates the MicroVM.
    backend = lifecycle.get_or_create(sandbox_id=SANDBOX_ID, seed_files=[("/notes.md", b"seeded before the first run\n")])
    print(f"sandbox {SANDBOX_ID} -> MicroVM {backend.id} ({lifecycle.config.transport})")
    try:
        ask(backend, "Write a short fun fact about DigitalOcean to /fact.txt, then read it back.")

        print("\npausing ...")
        lifecycle.pause(backend)
        print("resuming ...")
        backend = lifecycle.get_or_create(sandbox_id=SANDBOX_ID)  # resumes the paused MicroVM

        ask(backend, "Read /notes.md and /fact.txt and confirm both survived a pause/resume cycle.")
    finally:
        # Pause even after a failure: a paused MicroVM bills no compute, a running one does.
        lifecycle.pause(backend)
        print("\nsandbox left paused; delete it with: uv run examples/01_quickstart/quickstart.py --delete")


if __name__ == "__main__":
    main()
