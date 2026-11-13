"""Runs INSIDE the sandbox (seeded as /tools/chinook_sql.py): the Chinook database access of the sales assistant.

tools/sql.py calls it through the sandbox's `execute`, passing its arguments as one base64 JSON
blob so model-written SQL never goes through the shell's quoting. Standard library only. Prints one
JSON document.

    python3 /tools/chinook_sql.py query <b64 {"sql": ...}>
    python3 /tools/chinook_sql.py schema
    python3 /tools/chinook_sql.py add-customer <b64 {"first_name": ..., ...}>

The trust boundary is the one of the course's host-side tools: queries open the database in SQLite
read-only mode and must be a single SELECT; the only write is a parameterized INSERT into Customer,
assigned to Jane (employee 3), reached through the human-approved `add_customer` tool.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys

DB_PATH = "/data/chinook.db"
REP_EMPLOYEE_ID = 3  # Jane Peacock; "my customers" = SupportRepId

# Not allowed in a read-only query, as defense in depth on top of the read-only connection.
FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace", "truncate", "attach", "detach", "pragma", "vacuum")


def read_only() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str) -> object:
    stripped = sql.strip().rstrip(";").strip()
    lowered = stripped.lower()
    if not lowered.startswith(("select", "with")):
        return {"error": "Only SELECT queries are allowed."}
    if ";" in stripped:
        return {"error": "Only a single statement is allowed."}
    # Padding with spaces matches whole words only (so a column named "created_at" is fine).
    if any(f" {word} " in f" {lowered} " for word in FORBIDDEN):
        return {"error": "Query contains a forbidden (write) keyword."}
    conn = read_only()
    try:
        return [dict(row) for row in conn.execute(stripped).fetchall()]
    except sqlite3.Error as exc:
        return {"error": f"SQL error: {exc}"}
    finally:
        conn.close()


def schema() -> object:
    conn = read_only()
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        return {"schema": "\n\n".join(row["sql"] for row in rows if row["sql"])}
    finally:
        conn.close()


def add_customer(first_name: str, last_name: str, email: str, company: str = "", city: str = "", state: str = "", country: str = "", phone: str = "") -> object:
    if not email or "@" not in email:
        return {"error": "A valid email is required."}
    conn = sqlite3.connect(DB_PATH)  # the only read-write connection in this file
    try:
        # The Customer table has no unique constraint on Email, so check it here.
        existing = conn.execute("SELECT CustomerId FROM Customer WHERE lower(Email) = lower(?)", (email,)).fetchone()
        if existing:
            return {"error": f"Customer with email {email} already exists (CustomerId {existing[0]})."}
        cursor = conn.execute(
            "INSERT INTO Customer (FirstName, LastName, Company, City, State, Country, Phone, Email, SupportRepId) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (first_name, last_name, company or None, city or None, state or None, country or None, phone or None, email, REP_EMPLOYEE_ID),
        )
        conn.commit()
        return {"status": "created", "customer_id": cursor.lastrowid, "name": f"{first_name} {last_name}", "email": email}
    except sqlite3.Error as exc:
        return {"error": f"SQL error: {exc}"}
    finally:
        conn.close()


def main(argv: list[str]) -> object:
    command = argv[0] if argv else ""
    args = json.loads(base64.b64decode(argv[1])) if len(argv) > 1 else {}
    if command == "query":
        return query(args["sql"])
    if command == "schema":
        return schema()
    if command == "add-customer":
        return add_customer(**args)
    return {"error": f"unknown command {command!r}"}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:]), default=str))
