"""Runs INSIDE a locator's MicroVM (seeded into the template as /tools/classify.py): whose data is this, and what may go?

Reads /data/request.json ({"email", "name"}) and /data/records.json ({"system", "sources": {<source>: [raw API
records]}}), which the locator's fetch tool wrote, and prints one JSON manifest: for every record its id, whether it
is the requester's (`match`: the email address is the same), another person's (`different_person`: same name, a
different address) or unrelated, which personal-data categories it holds, the proposed action (erase / redact /
retain / skip) with the reason, and the call that performs the action. The manifest holds ids, categories and
counts, never a value. The records never leave this MicroVM, and once the manifest is printed the script deletes
them (`raw_deleted`): an erasure must not leave copies behind in the paused MicroVM. Standard library only; paths are
relative to this file, so it works at / in a MicroVM and in local mode.

    python3 /tools/classify.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RETENTION = "tax records: finalized invoices are kept for the retention period and cannot be edited"


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _date(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%d")
    return str(value)[:10] if value else None


def _fields(record: dict, names: dict[str, str]) -> list[str]:
    """The categories (email, name, phone, address, location) of the fields that hold a value."""
    return sorted({category for field, category in names.items() if record.get(field)})


def records(sources: dict) -> list[dict]:
    """Every raw record, normalized: source, kind, id, the email and name (for matching only), categories, related, call."""
    out = []
    for c in sources.get("hubspot_contacts", []):
        p = c.get("properties", {})
        out.append({"source": "hubspot", "kind": "contact", "id": str(c["id"]), "email": p.get("email"),
                    "name": f"{p.get('firstname', '')} {p.get('lastname', '')}".strip(), "created": _date(p.get("createdate")),
                    "fields": _fields(p, {"email": "email", "firstname": "name", "lastname": "name", "phone": "phone", "city": "location"}),
                    "related": {"notes": len(c.get("associations", {}).get("notes", {}).get("results", []))},
                    "call": {"tool": "hubspot_crm_contacts_gdpr_delete_create", "arguments": {"object_id": str(c["id"])}}})
    for c in sources.get("intercom_contacts", []):
        out.append({"source": "intercom", "kind": "contact", "id": str(c["id"]), "email": c.get("email"), "name": c.get("name"),
                    "created": _date(c.get("created_at")),
                    "fields": _fields({**c, "city": (c.get("location") or {}).get("city")},
                                      {"email": "email", "name": "name", "phone": "phone", "city": "location"}),
                    "related": {"notes": (c.get("notes") or {}).get("total_count", 0)},
                    "call": {"tool": "intercom_delete_a_contact", "arguments": {"contact_id": str(c["id"])}}})
    invoices = sources.get("stripe_invoices", [])
    for c in sources.get("stripe_customers", []):
        kept = [v for v in invoices if v.get("customer") == c["id"]]
        out.append({"source": "stripe", "kind": "customer", "id": c["id"], "email": c.get("email"), "name": c.get("name"),
                    "created": _date(c.get("created")),
                    "fields": _fields(c, {"email": "email", "name": "name", "phone": "phone", "address": "address"}),
                    "related": {"invoices": len(kept)}, "retained_by": [v["id"] for v in kept],
                    "call": {"tool": "stripe_post_customers_customer",
                             "arguments": {"customer": c["id"], "name": "", "email": "", "phone": "",
                                           "address": {"line1": "", "line2": "", "city": "", "state": "", "postal_code": "", "country": ""}}}})
    for v in invoices:
        out.append({"source": "stripe", "kind": "invoice", "id": v["id"], "email": v.get("customer_email"), "name": v.get("customer_name"),
                    "created": _date(v.get("created")), "customer": v.get("customer"),
                    "fields": _fields(v, {"customer_email": "email", "customer_name": "name"}), "related": {}})
    for m in sources.get("mailchimp_members", []):
        out.append({"source": "mailchimp", "kind": "member", "id": m["contact_id"], "email": m.get("email_address"),
                    "name": m.get("full_name"), "created": _date(m.get("timestamp_signup")),
                    "fields": _fields(m, {"email_address": "email", "full_name": "name"}), "related": {"audience": m.get("list_id")},
                    "call": {"tool": "mailchimp_forget_contact", "arguments": {"audience_id": m["list_id"], "contact_id": m["contact_id"]}}})
    for p in sources.get("mixpanel_profiles", []):
        props = p.get("$properties", {})
        out.append({"source": "mixpanel", "kind": "profile", "id": p["$distinct_id"], "email": props.get("$email"),
                    "name": props.get("$name"), "created": _date(props.get("$created")),
                    "fields": _fields(props, {"$email": "email", "$name": "name", "$city": "location"}), "related": {},
                    "call": {"tool": "mixpanel_delete_profile",
                             "arguments": {"project_id": p.get("project_id"),
                                           "updates": [{"$token": p.get("project_token"), "$distinct_id": p["$distinct_id"], "$delete": ""}]}}})
    return out


def classify(request: dict, raw: dict) -> dict:
    email, name = _norm(request.get("email")), _norm(request.get("name"))
    found = records(raw.get("sources", {}))
    matched_customers = {r["id"] for r in found if r["kind"] == "customer" and _norm(r["email"]) == email}
    manifest = []
    for r in found:
        if _norm(r["email"]) == email or r.get("customer") in matched_customers:
            identity = "match"
        elif name and _norm(r["name"]) == name:
            identity = "different_person"
        else:
            identity = "unrelated"
        entry = {"source": r["source"], "kind": r["kind"], "id": r["id"], "identity": identity, "personal_fields": r["fields"],
                 "related": r["related"], "created": r["created"]}
        if identity == "different_person":
            entry.update(action="skip", reason="same name, a different email address: another person")
        elif identity == "unrelated":
            entry.update(action="skip", reason="not the requester")
        elif r["kind"] == "invoice":
            entry.update(action="retain", reason=RETENTION)
        elif r["kind"] == "customer" and r.get("retained_by"):
            entry.update(action="redact", reason=f"its {len(r['retained_by'])} invoices are retained ({RETENTION}): clear the "
                                                  "customer's personal fields, keep the record the invoices point at",
                         call=r["call"])
        else:
            entry.update(action="erase", reason="the requester's record (the email address is the same)", call=r["call"])
        manifest.append(entry)
    counts = {a: sum(1 for e in manifest if e["action"] == a) for a in ("erase", "redact", "retain", "skip")}
    return {"system": raw.get("system"), "request": "sha256:" + hashlib.sha256(email.encode()).hexdigest()[:16],
            "records": manifest, "counts": counts}


def main() -> dict:
    inputs = [DATA / "request.json", DATA / "records.json"]
    try:
        manifest = classify(json.loads(inputs[0].read_text()), json.loads(inputs[1].read_text()))
    except (OSError, ValueError, KeyError) as exc:
        return {"error": f"cannot classify: {exc}"}
    for path in inputs:  # the manifest is all that is needed from here on
        path.unlink(missing_ok=True)
    return {**manifest, "raw_deleted": True}


if __name__ == "__main__":
    print(json.dumps(main()))
