"""The incident scenario the incident examples share: the droplet and its service, the simulated world, the models,
the event log and the investigators' sandbox scripts. An example adds `examples/` to `sys.path` and imports
`incident_common.<module>`; the setup script is run from here (README.md).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"  # setup_droplet.py's incident.json / incident.env, events.jsonl
SANDBOX = ROOT / "sandbox"  # uploaded to the investigators' MicroVMs as /tools/<name>.py
