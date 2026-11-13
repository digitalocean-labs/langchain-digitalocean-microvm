"""Runs INSIDE the publisher's MicroVM (seeded as /tools/board_render.py): renders the board pack as one HTML file.

    python3 /tools/board_render.py <b64 spec>

spec = {
  "title": "...", "subtitle": "...", "summary": "<markdown>",
  "kpis": [{"label": "...", "value": "...", "note": "...", "tone": "good" | "bad" | "neutral"}],
  "sections": [{"title": "...", "markdown_path": "/inputs/...md", "chart_path": "/inputs/...png", "agent": "..."}],
  "audit_path": "/inputs/auditor/audit.md",
}

Markdown files and charts are read from this VM (paths relative to this file's root) and embedded,
so the result is a single self-contained /outputs/board-pack.html. `markdown` is installed by the
sandbox template. Prints {"path": ...} or {"error": ...}.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "/outputs/board-pack.html"

CSS = """
:root { --navy:#031B4E; --blue:#0069FF; --ink:#1B2333; --muted:#5B6478; --line:#E5E8ED; --good:#00A88F; --bad:#E5484D; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--ink); background:#F5F7FA; }
header { background:linear-gradient(120deg,var(--navy),#0B3A8C 60%,var(--blue)); color:#fff; padding:48px 56px 40px; }
header .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:12px; opacity:.8; }
header h1 { margin:8px 0 6px; font-size:34px; }
header p { margin:0; opacity:.85; }
main { max-width:1080px; margin:-24px auto 48px; padding:0 24px; }
.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:28px 32px; margin-bottom:20px; box-shadow:0 1px 2px rgba(3,27,78,.04); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-bottom:20px; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px; border-top:4px solid var(--blue); }
.kpi.good { border-top-color:var(--good); } .kpi.bad { border-top-color:var(--bad); }
.kpi .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
.kpi .value { font-size:28px; font-weight:700; margin:6px 0 2px; color:var(--navy); }
.kpi .note { font-size:13px; color:var(--muted); }
h2 { color:var(--navy); margin:0 0 12px; font-size:22px; }
.section { display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:start; }
.section img { width:100%; border:1px solid var(--line); border-radius:10px; }
.agent { display:inline-block; font-size:11px; color:var(--blue); background:#EBF2FF; border-radius:999px; padding:3px 10px; margin-bottom:10px; }
table { border-collapse:collapse; width:100%; font-size:14px; margin:10px 0; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
th { color:var(--muted); font-weight:600; }
.audit { border-left:5px solid var(--good); }
footer { text-align:center; color:var(--muted); font-size:12px; padding:0 0 40px; }
@media (max-width:820px) { .section { grid-template-columns:1fr; } }
"""


def vm(path: str) -> Path:
    """A path as the agent sees it ('/inputs/x.md') -> where it is on disk (ROOT is / in a MicroVM)."""
    return ROOT / path.lstrip("/")


def md(text: str) -> str:
    """Markdown -> HTML; falls back to an escaped <pre> block if the `markdown` package is missing."""
    try:
        import markdown
    except ImportError:
        return f"<pre>{html.escape(text)}</pre>"
    return markdown.markdown(text, extensions=["tables"])


def read_md(path: str | None) -> str:
    if not path:
        return ""
    file = vm(path)
    return md(file.read_text()) if file.is_file() else f"<p><em>missing: {html.escape(path)}</em></p>"


def image(path: str | None) -> str:
    """The PNG inlined as a data: URI, so the board pack stays one file."""
    if not path or not vm(path).is_file():
        return ""
    data = base64.b64encode(vm(path).read_bytes()).decode()
    return f'<img alt="chart" src="data:image/png;base64,{data}">'


def render(spec: dict) -> dict:
    # Every model-supplied string is html.escape()d before it goes into the page.
    kpis = "".join(
        f'<div class="kpi {html.escape(k.get("tone", "neutral"))}"><div class="label">{html.escape(str(k.get("label", "")))}</div>'
        f'<div class="value">{html.escape(str(k.get("value", "")))}</div><div class="note">{html.escape(str(k.get("note", "")))}</div></div>'
        for k in spec.get("kpis", [])
    )
    sections = "".join(
        f'<div class="card"><h2>{html.escape(s.get("title", ""))}</h2>'
        + (f'<span class="agent">prepared by {html.escape(s["agent"])} in its own MicroVM</span>' if s.get("agent") else "")
        + f'<div class="section"><div>{read_md(s.get("markdown_path"))}</div><div>{image(s.get("chart_path"))}</div></div></div>'
        for s in spec.get("sections", [])
    )
    audit = spec.get("audit_path")
    audit_html = (
        f'<div class="card audit"><h2>Independent audit</h2><span class="agent">verified by auditor in its own MicroVM</span>{read_md(audit)}</div>'
        if audit else ""
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(spec.get("title", "Board pack"))}</title><style>{CSS}</style></head><body>
<header><div class="eyebrow">Board pack · confidential</div><h1>{html.escape(spec.get("title", "Board pack"))}</h1>
<p>{html.escape(spec.get("subtitle", ""))}</p></header>
<main><div class="kpis">{kpis}</div>
<div class="card"><h2>Executive summary</h2>{md(spec.get("summary", ""))}</div>
{sections}{audit_html}</main>
<footer>Generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC by a team of Deep Agents, each specialist in its own DigitalOcean MicroVM.</footer>
</body></html>"""
    out = vm(OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return {"path": OUTPUT, "bytes": len(page)}


if __name__ == "__main__":
    try:
        result = render(json.loads(base64.b64decode(sys.argv[1])))
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
