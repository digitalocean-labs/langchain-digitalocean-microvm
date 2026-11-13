"""Sandbox templates: prepare a sandbox once, checkpoint it, start new sandboxes from it warm.

MicroVM checkpoints capture memory and disk. `build_template` boots a MicroVM,
uploads the Chinook database, installs matplotlib, checkpoints it and deletes
the VM; `get_or_create(..., template=...)` then clones new sandboxes from that
checkpoint, with the database and packages already there.

The script times both ways of getting a sandbox that can plot a chart:

- cold: boot from the image, upload the database, `pip install matplotlib`;
- warm: clone the template.

then asks the data analyst of 05_data_analyst one question in the warm sandbox.

    uv run examples/06_warm_start/warm_start.py                     # build (or reuse) the template, compare, ask
    uv run examples/06_warm_start/warm_start.py --skip-cold         # only the warm path
    uv run examples/06_warm_start/warm_start.py --delete-template   # delete the template's checkpoint and exit

Sandboxes are deleted at the end; the template is kept (checkpoints do not
count as MicroVMs) so later runs, and `05_data_analyst/analyst.py --template`,
reuse it.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv

from langchain_digitalocean_microvm import MicroVMLifecycle

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")  # the shell wins over .env

DATABASE = ("/data/chinook.db", (HERE.parent / "05_data_analyst" / "data" / "chinook.db").read_bytes())
SETUP = ["pip install --quiet --no-cache-dir matplotlib"]  # same inputs as 05_data_analyst --template
# Proves the template really has both halves: matplotlib importable and the database queryable.
READY = "python3 -c 'import matplotlib, sqlite3; sqlite3.connect(\"/data/chinook.db\").execute(\"select 1 from Track\")'"


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


def timed(label: str, fn):
    """Run fn(), print how long it took under `label`, and return its result."""
    started = time.monotonic()
    result = fn()
    print(f"{label:<44} {time.monotonic() - started:6.1f} s")
    return result


def check_ready(backend) -> None:
    result = backend.execute(READY)
    if result.exit_code != 0:
        raise SystemExit(f"sandbox not ready to plot: {result.output.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-cold", action="store_true", help="skip the cold comparison")
    parser.add_argument("--delete-template", action="store_true", help="delete the template's checkpoint and exit")
    args = parser.parse_args()

    lifecycle = MicroVMLifecycle()  # DO_MICROVM_* from the environment
    local = lifecycle.config.transport == "local"
    if args.delete_template:  # look it up only: never build a template just to delete it
        found = lifecycle.find_template("data-analyst", setup=SETUP, seed_files=[DATABASE])
        if found is None:
            print("no template to delete")
            return
        lifecycle.delete_template(found)
        print(f"template {found.checkpoint_name} deleted")
        return
    template = timed("template (built once, then reused)", lambda: lifecycle.build_template("data-analyst", setup=SETUP, seed_files=[DATABASE]))
    print(f"  {'built' if template.built else 'reused'} {template.checkpoint_name} ({template.checkpoint_id})")
    run = uuid.uuid4().hex[:6]  # unique sandbox ids per run, so concurrent runs never share a VM

    if not args.skip_cold and not local:  # local mode cannot pip install (it would install on this machine)
        cold_id = f"cold-{run}"
        try:
            backend = timed("cold: boot from the image", lambda: lifecycle.get_or_create(sandbox_id=cold_id, seed_files=[DATABASE]))
            timed("cold: pip install matplotlib", lambda: backend.execute(SETUP[0], timeout=900))
            timed("cold: ready check", lambda: check_ready(backend))
        finally:
            # Delete the cold sandbox before the warm create, so only one MicroVM exists at a time.
            lifecycle.delete(sandbox_id=cold_id)

    warm_id = f"warm-{run}"
    try:
        backend = timed("warm: clone the template", lambda: lifecycle.get_or_create(sandbox_id=warm_id, template=template))
        if not local:
            timed("warm: ready check", lambda: check_ready(backend))
        agent = create_deep_agent(
            model=make_model(),
            backend=backend,
            system_prompt=(
                "You are a sales data analyst. The Chinook SQLite database is at /data/chinook.db and "
                "matplotlib is installed. Write Python scripts with write_file, run them with execute."
            ),
        )
        question = "Which 5 countries generate the most revenue? Give the amounts, and save a bar chart to /outputs/top_countries.png."
        result = timed("warm: agent answers one question", lambda: agent.invoke({"messages": [{"role": "user", "content": question}]}))
        print(f"\n>>> {question}\n\n{result['messages'][-1].content}")
    finally:
        lifecycle.delete(sandbox_id=warm_id)
        print(f"\ndeleted sandbox {warm_id}; template {template.checkpoint_name} kept")


if __name__ == "__main__":
    main()
