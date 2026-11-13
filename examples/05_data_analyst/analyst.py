"""A data analyst agent that runs SQL and plotting code inside a DigitalOcean MicroVM.

Port of two Deep Agents course lessons (https://github.com/langchain-ai/lca-deepagents):
python/m2/m2.3_sales_agent.py (Chinook in a LangSmith sandbox, a donut chart) and
python/m2/m2.4_interpreter_agent.py (chained SQL questions in an in-process
QuickJS interpreter). Here the database, the generated Python and the
installed packages all live in one MicroVM, which is paused between runs:
the next run finds the database, matplotlib and earlier charts where it left them.

    uv run examples/05_data_analyst/analyst.py                 # both tasks
    uv run examples/05_data_analyst/analyst.py --ask "..."     # your own question
    uv run examples/05_data_analyst/analyst.py --delete        # delete the sandbox
    uv run examples/05_data_analyst/analyst.py --template      # start from 06_warm_start's template

Charts the agent saves under /outputs/ are copied to examples/05_data_analyst/outputs/.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv

from langchain_digitalocean_microvm import MicroVMLifecycle, SandboxNotFoundError

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")  # the shell wins over .env

SANDBOX_ID = "data-analyst"
DATABASE = ("/data/chinook.db", (HERE / "data" / "chinook.db").read_bytes())

SYSTEM_PROMPT = (
    "You are a sales data analyst for Chinook, an online music store. Its SQLite database is at "
    "/data/chinook.db (tables: Artist, Album, Track, Genre, Invoice, InvoiceLine, Customer, ...). "
    "Work in Python with the sqlite3 module: write a script with write_file, run it with execute, "
    "and inspect the schema first if you are unsure. For charts use matplotlib "
    "(`python3 -c 'import matplotlib' || pip install --quiet matplotlib` first), save them under "
    "/outputs/, and confirm the file exists. Answer with the numbers you found."
)

CHAINED = (
    "Who is our top-selling artist by revenue, what is their best-selling album, what is the most "
    "purchased track on that album, and how many distinct customers bought that track? Each answer "
    "depends on the previous one."
)
CHART = (
    "Make a donut chart of each genre's share of total revenue. Group genres under 3% into 'Other', "
    "label each slice with the genre and percentage, avoid overlapping labels, and save it to "
    "/outputs/genre_revenue.png."
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


def ask(backend, question: str) -> None:
    # Build the agent per question: the backend is bound at build time (see 01_quickstart).
    agent = create_deep_agent(model=make_model(), backend=backend, system_prompt=SYSTEM_PROMPT)
    started = time.monotonic()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\n>>> {question}\n\n{result['messages'][-1].content}\n[{time.monotonic() - started:.0f} s]")


def save_outputs(backend) -> None:
    """Copy every file under the sandbox's /outputs/ to examples/05_data_analyst/outputs/."""
    if backend.execute("test -d /outputs").exit_code != 0:
        return  # nothing saved yet
    # "**" matches recursively; directories are skipped below so only files are downloaded.
    found = backend.glob("**", path="/outputs").matches or []
    # deepagents >= 0.7.21 returns absolute paths; earlier 0.7.x, paths relative to `path`
    paths = [m["path"] if m["path"].startswith("/") else f"/outputs/{m['path']}" for m in found if not m.get("is_dir")]
    for response in backend.download_files(paths) if paths else []:
        if response.content is None:
            continue
        target = HERE / "outputs" / response.path.removeprefix("/outputs/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        print(f"saved {response.path} -> {target.relative_to(Path.cwd()) if target.is_relative_to(Path.cwd()) else target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ask", help="ask this instead of the two built-in tasks")
    parser.add_argument("--delete", action="store_true", help="delete the sandbox and exit")
    parser.add_argument("--template", action="store_true", help="create the sandbox from 06_warm_start's template")
    args = parser.parse_args()

    lifecycle = MicroVMLifecycle()  # DO_MICROVM_* from the environment
    if args.delete:
        try:
            lifecycle.delete(sandbox_id=SANDBOX_ID)
            print(f"deleted sandbox {SANDBOX_ID}")
        except SandboxNotFoundError:
            print(f"no sandbox {SANDBOX_ID} to delete")
        return

    template = None
    if args.template:  # same inputs as 06_warm_start, so this reuses its checkpoint
        template = lifecycle.build_template(
            "data-analyst", setup=["pip install --quiet --no-cache-dir matplotlib"], seed_files=[DATABASE]
        )

    started = time.monotonic()
    # The database is uploaded only when this call creates the sandbox.
    backend = lifecycle.get_or_create(sandbox_id=SANDBOX_ID, seed_files=[DATABASE], template=template)
    print(f"sandbox {SANDBOX_ID} -> MicroVM {backend.id} ({lifecycle.config.transport}), ready in {time.monotonic() - started:.1f} s")
    try:
        for question in [args.ask] if args.ask else [CHAINED, CHART]:
            ask(backend, question)
        save_outputs(backend)
    finally:
        lifecycle.pause(backend)
        print("\nsandbox paused; the next run resumes it. Delete it with: uv run examples/05_data_analyst/analyst.py --delete")


if __name__ == "__main__":
    main()
