"""The erasure team's tools: the locators' run in their own MicroVMs, the coordinator's verify and report.

Locator tools are bound to one subagent's sandbox. `fetch_records` searches that locator's systems through the Action
Gateway on this side (credentials never enter a MicroVM) and writes the raw answers into the MicroVM's /data/;
`classify_records` runs /tools/classify.py there (sandbox/, seeded into the template), which prints a manifest of ids,
categories and proposed actions. The models see the manifest, never the records.

The coordinator's `verify_erasure` re-reads every record of the plan through the gateway and searches every system
for the request's email again; `save_report` writes the evidence report from the verification and the gateway's audit,
with a hash of the email and record ids only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from events import log_event
from langchain.tools import tool
from world import MIXPANEL_PROJECT, MIXPANEL_TOKEN

from langchain_digitalocean_microvm.transport.local import root_for_host

HERE = Path(__file__).resolve().parent
REPORT = HERE / "outputs" / "erasure-report.md"
SYSTEMS = {"crm": "CRM", "support": "support desk", "billing": "payments", "marketing": "email marketing", "analytics": "product analytics"}
SYSTEM_OF = {"hubspot": "crm", "intercom": "support", "stripe": "billing", "mailchimp": "marketing", "mixpanel": "analytics"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


def request_hash(email: str) -> str:
    """How the report names the requester: the normalized email address, hashed."""
    return "sha256:" + hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def _json(answer: str) -> Any:
    try:
        return json.loads(answer)
    except ValueError:
        raise RuntimeError(answer[:300]) from None  # the gateway's error text


# ── Fetching, on this side: what each locator's systems hold for an email address and a name ──────

def _crm(gateway: Any, email: str, name: str) -> dict[str, list]:
    first, _, last = name.partition(" ")
    groups = [{"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}]
    if first and last:  # a second group (ORed): the same name, whatever the address
        groups.append({"filters": [{"propertyName": "firstname", "operator": "EQ", "value": first},
                                   {"propertyName": "lastname", "operator": "EQ", "value": last}]})
    found = _json(gateway.call("hubspot_crm_contacts_search", {
        "filter_groups": groups, "properties": ["email", "firstname", "lastname", "phone", "city"],
        "sorts": ["createdate"], "after": "0", "limit": 20}))
    contacts = [_json(gateway.call("hubspot_crm_contacts_get_by_contactid", {
        "contact_id": c["id"], "properties": ["email", "firstname", "lastname", "phone", "city"], "associations": ["notes"]}))
        for c in found.get("results", [])]
    return {"hubspot_contacts": contacts}


def _support(gateway: Any, email: str, name: str) -> dict[str, list]:
    query = {"operator": "OR", "value": [{"field": "email", "operator": "=", "value": email},
                                         {"field": "name", "operator": "=", "value": name}]}
    return {"intercom_contacts": _json(gateway.call("intercom_search_contacts", {"query": query})).get("data", [])}


def _billing(gateway: Any, email: str, name: str) -> dict[str, list]:
    customers = _json(gateway.call("stripe_get_customers", {"email": email, "limit": 10})).get("data", [])
    invoices = [v for c in customers
                for v in _json(gateway.call("stripe_get_invoices", {"customer": c["id"], "limit": 100})).get("data", [])]
    return {"stripe_customers": customers, "stripe_invoices": invoices}


def _marketing(gateway: Any, email: str, name: str) -> dict[str, list]:
    found = _json(gateway.call("mailchimp_search_members", {"query": email}))
    members = [*found.get("exact_matches", {}).get("members", []), *found.get("full_search", {}).get("members", [])]
    profiles = _json(gateway.call("mixpanel_engage_query", {
        "project_id": MIXPANEL_PROJECT, "where": f'properties["$email"] == "{email}"'})).get("results", [])
    # the project the profiles live in, which the delete call names
    profiles = [{**p, "project_id": MIXPANEL_PROJECT, "project_token": MIXPANEL_TOKEN} for p in profiles]
    return {"mailchimp_members": members, "mixpanel_profiles": profiles}


FETCHERS: dict[str, Callable[[Any, str, str], dict[str, list]]] = {
    "crm": _crm, "support": _support, "billing": _billing, "marketing": _marketing,
}


# ── Locators: fetch into the MicroVM, classify there ───────────────────────────────────────────────

class FilesOnly(BackendProtocol):
    """A sandbox's files without its shell: what an investigator's file tools get.

    deepagents offers the model `execute` only on a backend that can run commands; this one cannot, so the agent has
    read, write, ls, grep and glob in its MicroVM and no shell. Its tools hold the sandbox itself and run their scripts.
    """

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def ls(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.ls(*args, **kwargs)

    async def als(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.als(*args, **kwargs)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.read(*args, **kwargs)

    async def aread(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aread(*args, **kwargs)

    def grep(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.grep(*args, **kwargs)

    async def agrep(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.agrep(*args, **kwargs)

    def glob(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.glob(*args, **kwargs)

    async def aglob(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aglob(*args, **kwargs)

    def write(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.write(*args, **kwargs)

    async def awrite(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.awrite(*args, **kwargs)

    def edit(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.edit(*args, **kwargs)

    async def aedit(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aedit(*args, **kwargs)

    def upload_files(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.upload_files(*args, **kwargs)

    async def aupload_files(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.aupload_files(*args, **kwargs)

    def download_files(self, *args: Any, **kwargs: Any) -> Any:
        return self.sandbox.download_files(*args, **kwargs)

    async def adownload_files(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sandbox.adownload_files(*args, **kwargs)


async def _started(backend: Any) -> Any:
    """The MicroVM behind an on-demand sandbox, started if it is not yet (a local root must exist before use)."""
    return await backend.aget() if hasattr(backend, "aget") else backend


async def _put(backend: Any, files: list[tuple[str, str]]) -> None:
    real = await _started(backend)
    uploads = await asyncio.to_thread(real.upload_files, [(path, content.encode()) for path, content in files])
    failed = [u for u in uploads if u.error]
    if failed:
        raise RuntimeError(f"could not write {failed[0].path}: {failed[0].error}")


async def _script(backend: Any, name: str, args: list[str], *, local: bool) -> str:
    """Run /tools/<name> in the MicroVM; its JSON output, or a JSON error."""
    try:
        real = await _started(backend)
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        return json.dumps({"error": f"MicroVM could not be started: {exc}"})
    # local mode: this process's interpreter, without the local transport's path shim; the scripts resolve their paths
    python = f"env -u MICROVM_LOCAL_ROOT {shlex.quote(sys.executable)}" if local else "python3"
    path = await asyncio.to_thread(lambda: f"{root_for_host(real.id)}/tools/{name}") if local else f"/tools/{name}"
    result = await real.aexecute(" ".join([python, shlex.quote(path), *map(shlex.quote, args)]))
    if result.exit_code != 0:
        return json.dumps({"error": f"command failed ({result.exit_code}): {result.output.strip()[-600:]}"})
    return result.output.strip()


def locator_tools(role: str, backend: Any, gateway: Any, *, local: bool) -> list:
    """`fetch_records` and `classify_records`, bound to one locator's MicroVM."""
    fetch = FETCHERS[role]

    @tool
    async def fetch_records(email: str, name: str) -> str:
        """Search your systems for the requester (by email address, and by name to catch look-alikes) and write what they
        hold into your MicroVM at /data/records.json. Returns how many records were written, not their contents."""
        try:
            sources = await asyncio.to_thread(fetch, gateway, email, name)
        except RuntimeError as exc:
            return f"Error: {exc}"
        await _put(backend, [("/data/request.json", json.dumps({"email": email, "name": name})),
                             ("/data/records.json", json.dumps({"system": role, "sources": sources}))])
        counts = ", ".join(f"{len(v)} {k}" for k, v in sources.items())
        return f"Wrote {counts} to /data/records.json."

    @tool
    async def classify_records() -> str:
        """Classify /data/records.json in your MicroVM: per record its id, identity match, personal-data categories, the
        proposed action and the call that performs it. Holds no personal data."""
        answer = await _script(backend, "classify.py", [], local=local)
        try:
            manifest = json.loads(answer)
        except ValueError:
            return answer
        # the dashboard's record list: ids and planned actions only
        records = [{"system": SYSTEM_OF.get(r.get("source", ""), role), "kind": r.get("kind"), "id": r.get("id"),
                    "action": r.get("action"), "identity": r.get("identity")} for r in manifest.get("records", [])]
        await asyncio.to_thread(log_event, event="manifest", locator=role, records=records)
        return answer

    return [fetch_records, classify_records]


# ── The coordinator: verify, report ────────────────────────────────────────────────────────────────

class Ledger:
    """What the report is built from: the request (kept in memory only), the last verification, the gateway's audit."""

    def __init__(self, audit: list[dict[str, Any]]) -> None:
        self.audit = audit
        self.email = ""
        self.name = ""
        self.started = time.time()
        self.verification: dict[str, Any] | None = None

    def scrub(self, text: str) -> str:
        """Text without email addresses and without the requester's name."""
        text = EMAIL.sub("[email removed]", text)
        for part in filter(None, [self.name, *self.name.split()]):
            text = re.sub(rf"\b{re.escape(part)}\b", "[name removed]", text, flags=re.IGNORECASE)
        return text


def _present(gateway: Any, system: str, kind: str, rid: str, email: str) -> dict[str, Any]:
    """Re-read one record through the gateway: {"exists": bool, "personal": bool}."""
    if system == "crm":
        found = _json(gateway.call("hubspot_crm_contacts_get_by_contactid", {"contact_id": rid, "properties": ["email", "firstname", "lastname", "phone"]}))
        exists = found.get("id") == rid
        return {"exists": exists, "personal": exists and any(found.get("properties", {}).get(k) for k in ("email", "firstname", "lastname", "phone"))}
    if system == "support":
        found = _json(gateway.call("intercom_get_a_contact", {"contact_id": rid}))
        return {"exists": found.get("id") == rid, "personal": found.get("id") == rid}
    if system == "billing" and kind == "invoice":
        invoices = _json(gateway.call("stripe_get_invoices", {"limit": 100})).get("data", [])
        return {"exists": any(v.get("id") == rid for v in invoices), "personal": True}
    if system == "billing":
        found = _json(gateway.call("stripe_get_customers_customer", {"customer": rid}))
        exists = found.get("id") == rid and not found.get("deleted")
        return {"exists": exists, "personal": exists and any(found.get(k) for k in ("name", "email", "phone", "address"))}
    if system == "marketing":
        found = _json(gateway.call("mailchimp_search_members", {"query": email}))
        members = [*found.get("exact_matches", {}).get("members", []), *found.get("full_search", {}).get("members", [])]
        exists = any(m.get("contact_id") == rid for m in members)
        return {"exists": exists, "personal": exists}
    if system == "analytics":
        found = _json(gateway.call("mixpanel_engage_query", {"project_id": MIXPANEL_PROJECT, "where": f'properties["$email"] == "{email}"'}))
        exists = any(p.get("$distinct_id") == rid for p in found.get("results", []))
        return {"exists": exists, "personal": exists}
    raise ValueError(f"unknown system {system!r}")


EXPECTED = {"erase": "erased", "redact": "redacted", "retain": "retained", "skip": "present"}


def _status(action: str, seen: dict[str, Any]) -> str:
    if not seen["exists"]:
        return "erased"
    if action == "redact" and not seen["personal"]:
        return "redacted"
    if action == "retain":
        return "retained"
    if action == "skip":
        return "present"
    return "still present"


def _by_email(gateway: Any, email: str) -> set[tuple[str, str, str]]:
    """Every record any system still finds under the email address: (system, kind, id)."""
    found: set[tuple[str, str, str]] = set()
    for c in _crm(gateway, email, "")["hubspot_contacts"]:
        found.add(("crm", "contact", str(c["id"])))
    for c in _support(gateway, email, "")["intercom_contacts"]:
        if str(c.get("email", "")).lower() == email.lower():
            found.add(("support", "contact", str(c["id"])))
    for c in _billing(gateway, email, "")["stripe_customers"]:
        found.add(("billing", "customer", c["id"]))
    marketing = _marketing(gateway, email, "")
    for m in marketing["mailchimp_members"]:
        if m.get("email_address", "").lower() == email.lower():
            found.add(("marketing", "member", m["contact_id"]))
    for p in marketing["mixpanel_profiles"]:
        found.add(("analytics", "profile", p["$distinct_id"]))
    return found


def verify(gateway: Any, ledger: Ledger, email: str, name: str, records: list[dict[str, str]]) -> dict[str, Any]:
    """Re-read each planned record and search every system for the email again; ids and statuses only."""
    ledger.email, ledger.name = email, name
    checked = []
    for r in records:
        system, kind, rid, action = r.get("system", ""), r.get("kind", ""), str(r.get("id", "")), r.get("action", "erase")
        try:
            status = _status(action, _present(gateway, system, kind, rid, email))
        except (RuntimeError, ValueError) as exc:
            status = f"unknown ({exc})"
        checked.append({"system": system, "kind": kind, "id": rid, "action": action, "status": status,
                        "ok": status == EXPECTED.get(action)})
    planned = {(c["system"], c["kind"], c["id"]) for c in checked}
    missed = [{"system": s, "kind": k, "id": i} for s, k, i in sorted(_by_email(gateway, email) - planned)]
    result = {"request": request_hash(email), "records": checked, "missed": missed,
              "complete": all(c["ok"] for c in checked) and not missed}
    ledger.verification = result
    log_event(event="verified", complete=result["complete"], missed=missed,
              records=[{k: c[k] for k in ("system", "kind", "id", "status", "ok")} for c in checked])
    return result


def _decision(audit: list[dict[str, Any]], system: str, kind: str, rid: str) -> str:
    """What happened to the erasure of one record, from the audit: approved, denied, refused, or not requested."""
    outcome = "not requested"
    for e in audit:
        if e.get("target") != (system, kind, rid):
            continue
        if e["event"] == "decision":
            outcome = "approved" if e["decision"] == "approve" else "denied by the approver"
        elif e["status"] == "deny":
            outcome = "refused by the policy"
        elif e["status"] == "ask" and outcome == "not requested":
            outcome = "waiting for a decision"
    return outcome


def render_report(ledger: Ledger, notes: str) -> str:
    """The evidence report, in Markdown: ids, statuses and decisions; a hash for the requester; no personal data."""
    v = ledger.verification or {"request": None, "records": [], "missed": [], "complete": False}
    stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))  # noqa: E731
    open_items = sum(1 for r in v["records"] if not r["ok"]) + len(v["missed"])
    lines = [
        "# Erasure report",
        "",
        f"- Request: `{v['request']}` (the requester's email address, hashed; this report holds no personal data)" if v["request"]
        else "- Request: unknown (no verification ran; this report holds no personal data)",
        f"- Started: {stamp(ledger.started)}; report written: {stamp(time.time())}",
        f"- Outcome: **{'COMPLETE' if v['complete'] else f'INCOMPLETE ({open_items} open)'}**"
        + ("" if ledger.verification else " (no verification ran)"),
        "- Systems: mocks (world.py), reached through the Action Gateway",
        "",
        "| System | Record | Plan | Decision | Verified |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in v["records"]:
        decision = _decision(ledger.audit, r["system"], r["kind"], r["id"]) if r["action"] in ("erase", "redact") else "-"
        lines.append(f"| {r['system']} ({SYSTEMS.get(r['system'], '?')}) | {r['kind']} `{r['id']}` | {r['action']} | {decision} | "
                     f"{r['status']} {'✔' if r['ok'] else '✘'} |")
    for m in v["missed"]:
        lines.append(f"| {m['system']} ({SYSTEMS.get(m['system'], '?')}) | {m['kind']} `{m['id']}` | not in the plan | - | still present ✘ |")
    refused = [e for e in ledger.audit if e["event"] == "gateway" and e["status"] == "deny"]
    if refused:
        lines += ["", "Refused by the gateway's policy (`deny`), whoever asked:", ""]
        lines += [f"- `{e['tool']}`" + (f" on {e['target'][0]} {e['target'][1]} `{e['target'][2]}`" if e.get("target") else "") for e in refused]
    lines += ["", "Kept, and why: a `retain` record is kept by law (billing's invoices are tax records); a `redact` record stays with its "
              "personal fields cleared so the retained records still point at it; a `skip` record belongs to someone else.", ""]
    if notes.strip():
        lines += ["## Notes (the coordinator's)", "", ledger.scrub(notes.strip()), ""]
    return "\n".join(lines)


def coordinator_tools(gateway: Any, ledger: Ledger) -> list:
    """`verify_erasure` and `save_report`."""

    @tool
    async def verify_erasure(email: str, name: str, records: list[dict[str, str]]) -> str:
        """Check the outcome (`email` and `name` as in the request): re-read every record of the plan (each {"system", "kind", "id", "action"}, action one of
        erase / redact / retain / skip; skipped and retained ones too) and search every system for the email address
        again. Returns each record's status and anything still found that the plan missed."""
        return json.dumps(await asyncio.to_thread(verify, gateway, ledger, email, name, records))

    @tool
    async def save_report(notes: str) -> str:
        """Write the evidence report (from the last verification and the gateway's audit) to a file on this machine.
        `notes`: at most 4 lines, no personal data (email addresses and the requester's name are removed anyway)."""
        text = render_report(ledger, notes)

        def write() -> None:
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(text)

        await asyncio.to_thread(write)
        return f"Saved the report to {REPORT}."

    return [verify_erasure, save_report]
