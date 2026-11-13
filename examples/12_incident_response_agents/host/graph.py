"""The incident commander as the graph of an agent host: one host per incident, started by relay.py.

The commander's async sub-agents run in agent hosts of their own, which its embedded gateway creates (agent.py). The host's
token check is the agent-host image's `auth.py`; routes.py serves the relay's /incident/* routes.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]
if str(EXAMPLE) not in sys.path:  # agent.py imports roles.py by name
    sys.path.insert(0, str(EXAMPLE))

from agent import build_commander  # noqa: E402

graph = build_commander()
