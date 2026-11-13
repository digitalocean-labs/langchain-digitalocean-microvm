"""The incident's postmortem as the relay writes it: the timeline and the verdict from what was recorded, the narrative from the
commander.

The commander writes its postmortem from memory, so its clock times are guesses. The relay holds the record: when the alert came,
what the commander and the specialists did, every broker call, the Remediator's gateway calls, each decision, the validator's
verdict and the relay's own reading of the service. This module turns that record into the timeline and the verdict of the
document, puts the commander's narrative (its summary, root cause, what was done, follow-ups) after them, and writes it where the
relay runs, outside the MicroVMs that are deleted when the incident closes.

Pure functions, no network: relay.py gathers the inputs.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# What a Remediator's gateway call says, by the status the commander's host logged for it (host/routes.py gateway_calls).
GATEWAY_TEXT = {"ask": "held for a human's approval", "approved": "reported it ran after a human approved it", "allow": "ran",
                "denied": "did not run: a human denied it", "deny": "refused by the Action Gateway's policy", "error": "failed"}
APPROVER = "the holder of the operator token (no individual identity is recorded)"


def read_events(path: Path, incident: str, thread: str) -> list[dict[str, Any]]:
    """The incident's events in the event log at `path`: those naming the incident (the relay's) or its thread (a host's)."""
    if not path.exists():
        return []
    found = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and (event.get("incident") == incident or event.get("thread") == thread):
            found.append(event)
    return found


def _clock(at: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(at))


def _alert_text(alarm: dict[str, Any]) -> str:
    value = alarm.get("value")
    reading = f" at {value:.0f}" if isinstance(value, (int, float)) else ""
    return f"{alarm.get('rule')} is {alarm.get('state')} ({alarm.get('severity')}): {alarm.get('metric')}{reading} on {alarm.get('resource')}"


def timeline(alarm: dict[str, Any] | None, story: dict[str, Any], events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows {"at", "who", "what"}, oldest first, from the alert, the story the relay kept and the events. Events seen twice (a
    local host and the relay share one log) count once."""
    rows: list[dict[str, Any]] = []

    def add(at: Any, who: str, what: str) -> None:
        if isinstance(at, (int, float)):
            rows.append({"at": float(at), "who": who, "what": what})

    if alarm:
        add(alarm.get("at"), "Insights", f"Alert received: {_alert_text(alarm)}")
    for sent in (story.get("sent") or {}).values():
        add(sent.get("at"), "Commander", f"Started the {sent.get('role')} sub-agent")
    for finding in (story.get("findings") or {}).values():
        add(finding.get("at"), str(finding.get("role")).capitalize(), "Reported (as the relay first read it)")
    verdict = story.get("verdict") or {}
    if verdict.get("verdict"):
        add(verdict.get("at"), "Recovery Validator", f"Verdict: {verdict['verdict']}")
        check = verdict.get("relay_check")
        if check:
            add(check.get("at"), "Relay", f"Read the service itself: {check['verdict']} "
                f"({'agrees with' if check.get('agrees') else 'DIFFERS from'} the validator)")
    if story.get("report"):
        add(story["report"].get("at"), "Commander", f"Saved its narrative: {story['report'].get('title')}")
    seen: set[str] = set()
    for event in events:
        key = json.dumps(event, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        at, kind = event.get("ts"), event.get("event")
        if kind == "incident" and event.get("run") and not event.get("error"):
            add(at, "Relay", "Started the commander's first run")
        elif kind == "relay" and event.get("woke"):
            add(at, "Relay", f"Woke the commander: {len(event['woke'])} finished task(s)")
        elif kind == "broker" and event.get("role") == "remediator":
            # The Remediator's requests: the commander's host logs each as a gateway call (below); the relay adds what only
            # it did, running an approved action with its own token, and a failure of its own.
            if event.get("status") == "approved":
                add(at, "Relay", f"Ran {event.get('tool')} after a human approved it")
            elif event.get("status") == "error":
                add(at, "Relay", f"{event.get('tool')} failed: {event.get('error') or 'no detail'}")
        elif kind == "broker":
            add(at, "Relay", f"{event.get('role')} read through the broker: {event.get('tool')} ({event.get('status')})")
        elif kind == "gateway":
            status = str(event.get("status"))
            add(at, "Remediator", f"{event.get('tool')} {GATEWAY_TEXT.get(status, status)}")
        elif kind == "approver" and event.get("decision") == "refused":
            add(at, "Relay", f"Refused an approval: {'; '.join(event.get('problems') or [])}")
        elif kind == "approver":
            add(at, "Approver", f"Decided {event.get('decision')} ({APPROVER})")
    return sorted(rows, key=lambda row: row["at"])


def render(*, incident: str, alarm: dict[str, Any] | None, story: dict[str, Any], events: Iterable[dict[str, Any]],
           notes: Iterable[str] = (), now: float | None = None) -> str:
    """The postmortem as Markdown: verdict and timeline from the record, then the commander's narrative."""
    report = story.get("report") or {}
    verdict = story.get("verdict") or {}
    check = verdict.get("relay_check")
    lines = [f"# {report.get('title') or f'Incident {incident}'}", "",
             f"Incident `{incident}`"
             + (f" · alert received {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(alarm['at']))} UTC" if alarm and alarm.get("at") else "")
             + f" · written {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() if now is None else now))} UTC", "",
             "## Outcome (from the record)", ""]
    if verdict.get("verdict"):
        lines.append(f"- Recovery Validator's verdict: **{verdict['verdict']}**")
    else:
        lines.append("- No verdict was recorded.")
    if check:
        lines.append(f"- The relay's own reading of the service: **{check['verdict']}** (status {check.get('status')}, version "
                     f"{check.get('version')}), which {'agrees with' if check.get('agrees') else '**differs from**'} the verdict.")
    rows = timeline(alarm, story, events)
    lines += ["", "## Timeline (recorded)", "", "| Time (UTC) | Who | What |", "| --- | --- | --- |"]
    lines += [f"| {_clock(row['at'])} | {row['who']} | {row['what'].replace('|', '/')} |" for row in rows]
    lines += ["", "Each time is when the relay or a host recorded the event: a specialist's report is timed when the relay first read it."]
    for note in notes:
        lines.append(f"\n_{note}_")
    lines += ["", "## Narrative (the commander's, from its sub-agents' reports; not checked against the record)", ""]
    lines.append(str(report["body"]).strip() if report.get("body") else "_The commander saved no narrative._")
    return "\n".join(lines) + "\n"


def write(directory: Path, incident: str, text: str) -> Path:
    """`<directory>/<incident>.md`, written whole or not at all."""
    if not incident.isalnum():
        raise ValueError(f"unexpected incident id {incident!r}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{incident}.md"
    partial = path.with_suffix(".md.part")
    partial.write_text(text)
    os.replace(partial, path)
    return path
