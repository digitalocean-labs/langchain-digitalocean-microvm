"""Runs INSIDE a specialist's MicroVM (seeded into the template as /tools/timeline.py): what changed, and when?

Reads /data/commits.json (the commit list the code host's commit tool returned) and prints the changes oldest first,
and, given the onset of the errors, the last commit before it and the minutes between the two.
Standard library only.

    python3 /tools/timeline.py [<onset, an ISO time such as the "onset" find_onset prints>]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "commits.json"


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build(commits: list[dict], onset: str | None) -> dict:
    # The list arrives newest first. Commits made in the same second keep that order: the newer one sorts last.
    indexed = sorted(
        enumerate(commits),
        key=lambda item: (parse(item[1]["commit"]["author"]["date"]), -item[0]),
    )
    rows = [{"sha": c["sha"][:7], "at": c["commit"]["author"]["date"], "message": c["commit"]["message"].splitlines()[0]}
            for _, c in indexed]
    out: dict = {"commits": rows}
    if onset:
        before = [r for r in rows if parse(r["at"]) <= parse(onset)]
        if before:
            last = before[-1]
            out["last_before_onset"] = {**last, "minutes_before_onset": round((parse(onset) - parse(last["at"])).total_seconds() / 60, 1)}
    return out


if __name__ == "__main__":
    try:
        print(json.dumps(build(json.loads(DATA.read_text()), sys.argv[1] if len(sys.argv) > 1 else None)))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": f"cannot read {DATA.name}: {exc}"}))
