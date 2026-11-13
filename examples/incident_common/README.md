# The incident scenario

Not an example of its own: the incident that [`09_incident_response`](../09_incident_response) and
[`12_incident_response_agents`](../12_incident_response_agents) respond to. A stateless
checkout service on a droplet starts failing after a deploy; a restore from a snapshot taken before the deploy fixes it.

| File | What it is |
| --- | --- |
| `setup_droplet.py` | Real mode: create the droplet, snapshot it, start the incident (`--deploy-bad`), clean up |
| `world.py` | The simulated world (in-memory droplet, commits, snapshot, metrics) and `HttpService` for the real one |
| `world_server.py` | Local mode, agents in separate processes: one process serves the simulated world over HTTP, the others reach it |
| `events.py` | `DemoLifecycle` and `log_event`: the event log (`outputs/events.jsonl`) |
| `models.py` | The investigators' model and the commander's |
| `sandbox/` | `series.py`, `timeline.py`, `verdict.py`: the investigators' scripts, uploaded under `/tools/` |
| `service/` | `app.py`, the service the droplet runs, and `make_demo_repo.sh`, which builds its two-commit repository |

State goes to `outputs/` here: `incident.json` and `incident.env` (`setup_droplet.py`, loaded by the examples) and
`events.jsonl`. How to run it all: the example's README.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # examples/
from incident_common.world import make_service
```
