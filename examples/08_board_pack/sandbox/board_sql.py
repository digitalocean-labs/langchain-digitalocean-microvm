"""Runs INSIDE a subagent's MicroVM (seeded into the template as /tools/board_sql.py): read-only SQL on Chinook.

tools.py calls it through the sandbox's `execute`, passing the query as one base64 JSON blob so
model-written SQL never goes through the shell's quoting. Standard library only; prints one JSON
document. Paths are relative to this file, so it works at / in a MicroVM and in local mode.

    python3 /tools/board_sql.py query <b64 {"sql": ...}>
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chinook.db"
MAX_ROWS = 200

# Defense in depth on top of the read-only connection below; matched as whole words.
FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace", "truncate", "attach", "detach", "pragma", "vacuum")


def query(sql: str) -> object:
    stripped = sql.strip().rstrip(";").strip()
    lowered = stripped.lower()
    if not lowered.startswith(("select", "with")):
        return {"error": "Only SELECT queries are allowed."}
    if ";" in stripped:
        return {"error": "Only a single statement is allowed."}
    if any(f" {word} " in f" {lowered} " for word in FORBIDDEN):
        return {"error": "Query contains a forbidden (write) keyword."}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)  # SQLite itself refuses writes
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(stripped).fetchmany(MAX_ROWS + 1)  # one extra row tells us it was truncated
    except sqlite3.Error as exc:
        return {"error": f"SQL error: {exc}"}
    finally:
        conn.close()
    out: object = [dict(row) for row in rows[:MAX_ROWS]]
    if len(rows) > MAX_ROWS:
        out = {"rows": out, "note": f"truncated to {MAX_ROWS} rows; aggregate in SQL"}
    return out


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    args = json.loads(base64.b64decode(sys.argv[2])) if len(sys.argv) > 2 else {}
    result = query(args["sql"]) if command == "query" else {"error": f"unknown command {command!r}"}
    print(json.dumps(result, default=str))
