"""Runs INSIDE a specialist's MicroVM (seeded into the template as /tools/series.py): where did the errors start?

Reads /data/metrics.json, the service's per-minute points ({"t", "version", "error_rate", "p95_ms"}), and prints
one JSON document: the baseline, the onset (the first minute the error rate stays above the threshold),
the peak, and which version served the traffic at the onset, the one before it, and when the version changed. Standard library only; paths are
relative to this file, so it works at / in a MicroVM and in local mode.

    python3 /tools/series.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


DATA = Path(__file__).resolve().parents[1] / "data" / "metrics.json"


def analyse(points: list[dict]) -> dict:
    if len(points) < 6:
        return {"error": "need at least 6 points"}
    head = sorted(p["error_rate"] for p in points[: max(len(points) // 3, 3)])
    baseline = head[len(head) // 2]
    threshold = max(0.05, baseline * 5)
    onset = next((i for i in range(len(points)) if all(p["error_rate"] >= threshold for p in points[i:i + 3])), None)
    peak = max(points, key=lambda p: p["error_rate"])
    out = {"points": len(points), "from": points[0]["t"], "to": points[-1]["t"], "baseline_error_rate": round(baseline, 3),
           "threshold": round(threshold, 3), "peak_error_rate": peak["error_rate"], "peak_p95_ms": max(p["p95_ms"] for p in points),
           "latest": points[-1]}
    if onset is None:
        return {**out, "onset": None}
    versions = []
    for p in points:
        if not versions or versions[-1]["version"] != p["version"]:
            versions.append({"version": p["version"], "first_seen": p["t"]})
    current = next(v for v in reversed(versions) if v["first_seen"] <= points[onset]["t"])
    earlier = versions[versions.index(current) - 1]["version"] if versions.index(current) else None
    minutes = round((_time(points[onset]["t"]) - _time(current["first_seen"])).total_seconds() / 60, 1)
    return {**out, "onset": points[onset]["t"], "version_at_onset": current["version"], "previous_version": earlier,
            "version_first_seen": current["first_seen"], "minutes_from_version_change_to_onset": minutes,
            "versions": versions}


if __name__ == "__main__":
    try:
        print(json.dumps(analyse(json.loads(DATA.read_text()))))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": f"cannot read {DATA.name}: {exc}"}))
