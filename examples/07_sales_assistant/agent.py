"""The Chinook Sales Assistant with one DigitalOcean MicroVM per conversation thread.

Port of the Deep Agents course capstone (https://github.com/langchain-ai/lca-deepagents),
python/m5/sales_assistant_sandbox/agent.py, which gave every thread a LangSmith sandbox.

THE IDEA
The assistant's whole filesystem lives in the thread's MicroVM: the operating manual
(/AGENTS.md), the skills (/skills/), the Chinook database (/data/chinook.db), its SQL helper
(/tools/chinook_sql.py), matplotlib, and everything the agent writes (/outputs/). The model
and the tools run in this process; only files and commands (`execute`, and the analyst's SQL)
go to the MicroVM. There is no local filesystem for the agent to read or write.

PER RUN (`make_graph`; design doc §5.1)
`langgraph dev` enters `make_graph` for every run of a thread, so:

1. the thread's sandbox is gotten or created (ThreadSandboxes): a thread's first run clones
   the template, a later run resumes the paused MicroVM, with its files and charts where they
   were left;
2. the agent is built against it and runs (a run that stops at a human approval ends too);
3. the sandbox is paused when the run ends. Paused MicroVMs keep memory and files, and still
   count against the team's MicroVM limit: delete a thread's with run_script.py --delete, or
   let scripts/gc_microvms.py delete threads unused past --sandbox-ttl (each run's use is
   recorded in the gateway store).

THE TEMPLATE
Everything every thread needs is prepared once, as a sandbox template (see ../06_warm_start):
`pip install matplotlib` plus the seeded files, checkpointed. The first run in a process
builds it (about 30 s), or reuses an existing checkpoint; each new thread then clones it in
seconds instead of booting, uploading and installing. If the checkpoint was deleted meanwhile
(scripts/delete_all.py --include-templates), the first clone that fails rebuilds it once and retries
(TemplateCache).

MODES (.env)
- DO_MICROVM_TRANSPORT=local: sandboxes are local directories; no DigitalOcean account. The
  agent's commands run on this machine, so the prompt forbids installing packages.
- otherwise real MicroVMs: DO_MICROVM_API_TOKEN, DO_MICROVM_ACCESS_TOKEN and a sandbox image that
  lets the agent `pip install` (scripts/build_sandbox_image.py), see ../../README.md.

RUN (from the repository root)
    uv run examples/07_sales_assistant/mail_server/mock_mail_server.py &   # the mock mail MCP server
    cd examples/07_sales_assistant && uv run langgraph dev --n-jobs-per-worker 10
or  uv run examples/07_sales_assistant/run_script.py
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")  # the shell wins over .env; before the modules below read the environment

from async_research import build_async_research_middleware
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.runnables import RunnableConfig
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph_sdk.runtime import ServerRuntime
from models import strong_model
from subagents import build_subagents
from tools.html import markdown_to_html
from tools.mail import load_mail_tools

from langchain_digitalocean_microvm import (
    MicroVMLifecycle,
    TemplateCache,
    ThreadSandboxes,
)
from langchain_digitalocean_microvm.agent_host.store import (
    GatewayStore,
    default_db_path,
)

logger = logging.getLogger(__name__)

lifecycle = MicroVMLifecycle()  # DO_MICROVM_* from the environment
LOCAL = lifecycle.config.transport == "local"

# Prepared once into the template; a thread's sandbox starts with all of it.
SETUP = ["pip install --quiet --no-cache-dir matplotlib"]
MAIL_URL = os.environ.get("MAIL_MCP_URL", "http://127.0.0.1:5002/mcp")

_enable_search = bool(os.environ.get("TAVILY_API_KEY"))
if not _enable_search:
    logger.info("TAVILY_API_KEY not set: newsletter research subagent disabled.")

SYSTEM_PROMPT = (
    "You are a sales assistant for Jane Peacock, a Sales Support Agent at "
    "Chinook, an online music distributor. Follow your operating manual (loaded "
    "from your memory) and use the matching playbook from /skills/ for each task.\n\n"
    "Your entire filesystem (skills, memory, the database, and anything you write) lives "
    "inside an isolated sandbox; there is no separate local filesystem. To produce a chart, "
    "write a Python script with write_file and run it with execute, saving the image under "
    "/outputs/. "
    + (
        "Never install packages (pip, apt): matplotlib may be missing; say so instead of installing it."
        if LOCAL
        else "matplotlib is already installed; do not pip install anything."
    )
)


def seed_files() -> list[tuple[str, bytes]]:
    """What a thread's sandbox starts with: the manual, the skills, the database and its SQL helper."""
    files = [
        ("/AGENTS.md", (HERE / "AGENTS.md").read_bytes()),
        ("/tools/chinook_sql.py", (HERE / "sandbox" / "chinook_sql.py").read_bytes()),
        ("/data/chinook.db", (HERE / "data" / "chinook.db").read_bytes()),
    ]
    for path in sorted((HERE / "skills").rglob("*")):
        if path.is_file():
            files.append((f"/skills/{path.relative_to(HERE / 'skills').as_posix()}", path.read_bytes()))
    return files


# Process-wide cache: langgraph dev runs many threads concurrently, but the template is built once
# (and rebuilt once if its checkpoint is deleted while the server runs).
templates = TemplateCache(lambda: lifecycle.build_template("sales-assistant", setup=SETUP, seed_files=seed_files()))
# Real MicroVMs: record each thread's last use where scripts/gc_microvms.py looks.
store = None if LOCAL else GatewayStore(default_db_path("microvm", lifecycle.config.namespace))
# Thread T's sandbox is sales-T; at most 4 threads create or resume their MicroVM at once.
threads = ThreadSandboxes(lifecycle, prefix="sales", template=templates, store=store)


async def build_agent(backend, **kwargs):
    """The assistant for one run, with `backend` (the thread's sandbox, or StateBackend for introspection)."""
    try:
        mail_tools = await load_mail_tools(MAIL_URL)
    except Exception as exc:  # the mail server is a separate process: run without an inbox-manager
        logger.warning("mail server %s not reachable (%s): running without mail tools", MAIL_URL, exc)
        mail_tools = []
    # The code interpreter lets the model chain several execute/write_file calls in one
    # step (programmatic tool calling) instead of one model round trip per call.
    middleware = [CodeInterpreterMiddleware(ptc=["execute", "write_file"])]
    if _enable_search:
        middleware.append(build_async_research_middleware())
    return create_deep_agent(
        model=strong_model,
        tools=[markdown_to_html, *mail_tools],
        system_prompt=SYSTEM_PROMPT,
        subagents=build_subagents(backend, mail_tools=mail_tools),
        skills=["/skills"],  # paths inside the sandbox, seeded by seed_files()
        memory=["/AGENTS.md"],  # always loaded into the system prompt
        backend=backend,
        middleware=middleware,
        name="chinook-sales-assistant",
        **kwargs,
    )


# https://docs.langchain.com/langsmith/graph-rebuild#context-manager-factory
#
# The factory accepts ServerRuntime so the server can say whether it is processing a real run
# (execution_runtime is set) or introspecting (get_schema, get_graph, ...). Introspection gets
# the same topology on an in-memory backend, without touching a MicroVM.
@contextlib.asynccontextmanager
async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    if not runtime.execution_runtime:
        yield await build_agent(StateBackend())
        return
    thread_id = config.get("configurable", {}).get("thread_id")
    async with threads.asandbox(thread_id) as backend:
        yield await build_agent(backend)
