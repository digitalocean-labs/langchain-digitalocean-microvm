"""Runs INSIDE a subagent's MicroVM (seeded as /tools/board_chart.py): one chart, in the board pack's style.

    python3 /tools/board_chart.py <b64 {"kind": "bar"|"line"|"barh", "title": ..., "labels": [...],
                                        "series": {"name": [numbers]}, "filename": "slug", "unit": "$"}>

Writes /outputs/<filename>.png (relative to this file's root) with matplotlib, which the sandbox
template installs. Prints {"path": "/outputs/<filename>.png"} or {"error": ...}.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#0069FF", "#00C2A8", "#FFB000", "#E5484D", "#8E4EC6"]


def render(spec: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display in a VM
    import matplotlib.pyplot as plt

    labels = [str(label) for label in spec["labels"]]
    series = {name: [float(v) for v in values] for name, values in spec["series"].items()}
    if any(len(values) != len(labels) for values in series.values()):
        return {"error": "every series needs one value per label"}
    kind = spec.get("kind", "bar")
    unit = spec.get("unit", "$")
    # Slugify: the model chooses the name and it becomes a path, so keep it to [a-z0-9-].
    filename = re.sub(r"[^a-z0-9-]+", "-", str(spec.get("filename", "chart")).lower()).strip("-") or "chart"

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    width = 0.8 / max(1, len(series))  # grouped bars: split 80% of each slot among the series
    for i, (name, values) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        if kind == "line":
            ax.plot(labels, values, marker="o", linewidth=2.5, color=color, label=name)
        elif kind == "barh":
            ax.barh([y + i * width for y in range(len(labels))], values, height=width, color=color, label=name)
            ax.set_yticks([y + width * (len(series) - 1) / 2 for y in range(len(labels))], labels)
        else:
            bars = ax.bar([x + i * width for x in range(len(labels))], values, width=width, color=color, label=name)
            ax.set_xticks([x + width * (len(series) - 1) / 2 for x in range(len(labels))], labels)
            if len(series) == 1:
                ax.bar_label(bars, labels=[f"{unit}{v:,.0f}" for v in values], padding=3, fontsize=8, color="#333")
    ax.set_title(spec.get("title", ""), loc="left", fontsize=12, fontweight="bold", color="#031B4E")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if kind == "barh" else "y", color="#E5E8ED", linewidth=0.8)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = ROOT / "outputs" / f"{filename}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {"path": f"/outputs/{filename}.png"}


if __name__ == "__main__":
    try:
        result = render(json.loads(base64.b64decode(sys.argv[1])))
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
