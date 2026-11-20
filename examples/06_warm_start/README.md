# 06 · Warm start: sandbox templates from checkpoints

**Requires** (real MicroVMs): a team limit of at least 1 MicroVM at a time (the cold sandbox is deleted before the warm clone). The template is a checkpoint, not a MicroVM.

**Quick run** (a real MicroVM; needs a model, see [`../README.md`](../README.md)):

```bash
uv run python scripts/setup_examples.py
uv run examples/06_warm_start/warm_start.py
```

Setup and options: see below.

**The use case.** An agent's sandbox often needs preparing before the agent can work: data uploaded, packages
installed. Doing that for every new user or conversation costs time before the agent's first
answer. A template does it once and gives every new sandbox a ready copy in a few seconds.

A MicroVM checkpoint captures memory and disk; a MicroVM created from one is
running in seconds, with the files, installed packages and processes of the
original. `MicroVMLifecycle.build_template` turns that into reusable sandbox
templates:

```python
from langchain_digitalocean_microvm import MicroVMLifecycle

lifecycle = MicroVMLifecycle()
template = lifecycle.build_template(
    "data-analyst",
    setup=["pip install --quiet --no-cache-dir matplotlib"],   # run once, in a temporary MicroVM
    seed_files=[("/data/chinook.db", db_bytes)],
)                                                               # checkpoint kept, temporary MicroVM deleted
backend = lifecycle.get_or_create(sandbox_id="user-42", template=template)  # cloned, ready in seconds
...
lifecycle.delete_template(template)                             # when you no longer need it
```

- The checkpoint name includes a hash of the image, size, port, setup commands
  and seed files: the same inputs reuse it, any change builds a new one
  (`rebuild=True` forces it).
- A failing setup command raises, and the temporary MicroVM is deleted either way.
- Templates are checkpoints, not MicroVMs, so they do not count against the
  team's MicroVM limit. `scripts/delete_all.py --include-templates --all-owners --yes` deletes them too, with every other user's MicroVMs (add `--all-namespaces` where namespaces are in use; without `--include-templates` it keeps them).
- Local mode copies a directory and does not run `setup` (it would run on this machine).

## Run (local mode)

Local mode copies a directory as the template and does not run `setup`; the agent's commands run on this machine.

```bash
DO_MICROVM_TRANSPORT=local uv run examples/06_warm_start/warm_start.py
```

## Run (real MicroVMs)

`warm_start.py` builds (or reuses) a template with the Chinook database and
matplotlib, times a cold sandbox (boot + upload + `pip install`) against a
clone of the template, and asks the analyst of
[`05_data_analyst`](../05_data_analyst) one question in the warm sandbox. It needs the variables of
[`05_data_analyst`](../05_data_analyst), in `.env`:

```bash
# .env
DO_MICROVM_API_TOKEN=...      # microvm:create/read/update/delete
DO_MICROVM_ACCESS_TOKEN=...   # microvm:access only (the sandboxes' HTTPS endpoints)
DO_MICROVM_IMAGE=registry.digitalocean.com/<registry>/microvm-sandbox:<tag>
DO_MODEL_ACCESS_KEY=...       # or EXAMPLES_MODEL plus that provider's key
```

To set them up step by step:

```bash
uv run python scripts/setup_examples.py
uv run python scripts/setup_examples.py --dry-run   # checks only
uv run python scripts/check_setup.py --examples 06   # ready, ready* (also needs the setup named) or what is missing
```

Then:

```bash
uv run examples/06_warm_start/warm_start.py                    # build/reuse the template, compare, ask
uv run examples/06_warm_start/warm_start.py --skip-cold        # warm path only
uv run examples/06_warm_start/warm_start.py --delete-template
uv run examples/05_data_analyst/analyst.py --template          # the analyst, started from the same template
```

## Where the time goes

| | Cold | Warm (clone of the template) |
| --- | --- | --- |
| Get the sandbox | boot, then upload the 1 MB database | clone of the checkpoint |
| `pip install matplotlib` | every time | none (in the checkpoint) |
| First command (ready check: `import matplotlib`, open the database) | fast | a little slower: the clone's endpoint can take a moment to answer |
| Ready to plot | boot + upload + install | **two to three times sooner** |

Getting a warm sandbox includes connecting to the sandbox server in the clone: the
server still holds the template's token, so the transport sets the clone's own over the exec API
(one call). Later commands are as fast as in a cold sandbox.

Building the template is a one-off cost (temporary MicroVM, setup, checkpoint, delete); reusing it later
costs one checkpoint lookup. The gap grows with heavier setup (more packages,
larger datasets, a warmed-up service).
