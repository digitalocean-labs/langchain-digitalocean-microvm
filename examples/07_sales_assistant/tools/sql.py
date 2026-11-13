"""SQL tools for the chinook-analyst: the database lives in the thread's MicroVM.

In the course (python/m5/tools/sql.py) these tools opened a database file on the
agent server's disk. Here the database is /data/chinook.db inside the sandbox
(seeded into the template), so each tool runs sandbox/chinook_sql.py there with the
sandbox's `execute` and returns its JSON. Every thread has its own copy of the
database: a customer added in one conversation does not exist in another.

The three tools keep the course's trust boundary: read-only SELECTs, the schema, and
one parameterized INSERT into Customer behind a human approval (configured where the
subagent is built).
"""

from __future__ import annotations

import base64
import json
import shlex

from langchain.tools import tool

SCRIPT = "/tools/chinook_sql.py"


def make_sql_tools(backend) -> list:
    """The analyst's three tools, bound to this thread's sandbox."""

    def run(command: str, args: dict | None = None) -> str:
        # Arguments travel as base64 JSON, shell-quoted, so model-written SQL is never parsed by the shell.
        arg = f" {shlex.quote(base64.b64encode(json.dumps(args).encode()).decode())}" if args is not None else ""
        result = backend.execute(f"python3 {SCRIPT} {command}{arg}", timeout=60)
        if result.exit_code != 0:
            return json.dumps({"error": f"sandbox command failed ({result.exit_code}): {result.output.strip()[-500:]}"})
        return result.output.strip()

    @tool
    def query_chinook(sql: str) -> str:
        """Run a read-only SQL SELECT against the Chinook database.

        Returns a JSON array of row objects. Only a single SELECT statement is
        allowed; any attempt to modify the database is rejected. Use this for all
        lookups: catalogue prices, a customer's purchase history, territory
        metrics, and so on.
        """
        return run("query", {"sql": sql})

    @tool
    def introspect_schema() -> str:
        """Return the full database schema (CREATE statements for every table).

        Call this once to learn the schema, then record it in your memory so you
        don't have to rediscover it on every task.
        """
        output = run("schema")
        try:
            return json.loads(output)["schema"]
        except (ValueError, KeyError, TypeError):
            return output

    @tool
    def add_customer(
        first_name: str,
        last_name: str,
        email: str,
        company: str = "",
        city: str = "",
        state: str = "",
        country: str = "",
        phone: str = "",
    ) -> str:
        """Add a NEW customer to the database, assigned to the current sales rep.

        Use this only after confirming the customer is not already in the system
        (search by email or name first). A human approves this write before it
        runs. Returns the new CustomerId on success.
        """
        return run(
            "add-customer",
            {"first_name": first_name, "last_name": last_name, "email": email, "company": company,
             "city": city, "state": state, "country": country, "phone": phone},
        )

    return [query_chinook, introspect_schema, add_customer]
