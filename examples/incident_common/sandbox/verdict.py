"""Runs INSIDE the verifier's MicroVM (seeded into the template as /tools/verdict.py): is the service healthy again?

Reads /data/health.json, the samples of the service's `/health` taken after the fix ({"t", "status", "version",
"error_rate"}, oldest first), and prints one JSON document with a verdict: "healthy" when the last HEALTHY_RUN samples are
all `healthy` with an error rate under MAX_ERROR_RATE, otherwise "not_healthy" with the reason. Standard library only.

    python3 /tools/verdict.py
"""

from __future__ import annotations

import json
from pathlib import Path

HEALTHY_RUN = 3  # consecutive healthy samples that count as recovered
MAX_ERROR_RATE = 0.05
DATA = Path(__file__).resolve().parents[1] / "data" / "health.json"


def is_healthy(sample: dict) -> bool:
    """One sample counts as healthy: status `healthy` and a numeric error rate under MAX_ERROR_RATE (none, or a text, is not)."""
    rate = sample.get("error_rate")
    return sample.get("status") == "healthy" and isinstance(rate, (int, float)) and rate < MAX_ERROR_RATE


def judge(samples: list[dict]) -> dict:
    if not samples:
        return {"verdict": "not_healthy", "reason": "no samples", "samples": 0}
    latest = samples[-1]
    out = {"samples": len(samples), "latest": latest, "first": samples[0]}
    recent = samples[-HEALTHY_RUN:]
    if len(recent) < HEALTHY_RUN:
        return {**out, "verdict": "not_healthy", "reason": f"only {len(recent)} sample(s): need {HEALTHY_RUN} in a row"}
    bad = [s for s in recent if not is_healthy(s)]
    if bad:
        return {**out, "verdict": "not_healthy",
                "reason": f"{len(bad)} of the last {HEALTHY_RUN} samples are failing (latest: {latest.get('status')}, "
                          f"error rate {latest.get('error_rate')})"}
    return {**out, "verdict": "healthy", "version": latest.get("version"),
            "reason": f"the last {HEALTHY_RUN} samples are healthy (error rate {latest.get('error_rate')})"}


if __name__ == "__main__":
    try:
        print(json.dumps(judge(json.loads(DATA.read_text()))))
    except (OSError, ValueError) as exc:
        print(json.dumps({"verdict": "not_healthy", "reason": f"cannot read {DATA.name}: {exc}"}))
