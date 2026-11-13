"""The erasure request's world: five MOCK systems that hold one customer's personal data.

Everything here is a mock, in memory: a CRM, a support desk, payments (billing), email marketing and
product analytics. `actions()` gives the local Action Gateway one handler per tool,
named and shaped as the gateway's catalog has them; required arguments are checked, so a wrong call fails as it
would against the real tool. What an erasure changes is written to the event log.

The data:
- the requester, Jane Doe <jane.doe@example.com>, in all five systems;
- a different Jane Doe <jane.doe@example.org> in the CRM, who must not be touched;
- three paid invoices in billing, which are kept (tax records) while the customer record is redacted.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from events import log_event

from langchain_digitalocean_microvm.integrations.action_gateway_local import (
    LocalAction,
)

REQUEST_EMAIL, REQUEST_NAME = "jane.doe@example.com", "Jane Doe"
DECOY_EMAIL = "jane.doe@example.org"
CRM_ID, DECOY_ID, OTHER_ID = "51", "87", "12"
SUPPORT_ID = "6650a1f3c2e4"
BILLING_ID = "cus_Q8mJd2LkT0aZ"
INVOICE_IDS = ["in_1PaQ7m", "in_1PbR2x", "in_1PcS9k"]
AUDIENCE_ID, MAILCHIMP_CONTACT = "a1b2c3d4e5", "mc_7f3e21"
MIXPANEL_PROJECT, MIXPANEL_TOKEN, MIXPANEL_ID = 31337, "demo-project-token", "u-5c1e9a"  # a project token is not a secret
DAY = 86400.0


def _iso(moment: float) -> str:
    return datetime.fromtimestamp(moment, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _missing(arguments: Mapping[str, Any], *names: str) -> dict[str, Any] | None:
    """The tool's error for a missing required argument, as the gateway validates them."""
    absent = [n for n in names if n not in arguments]
    return {"error": f"Request is invalid: missing required argument(s) {', '.join(absent)}."} if absent else None


def _schema(required: list[str], **props: str) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": required}


class SimulatedSystems:
    """The five systems of one erasure request. `now` is injectable so tests can fix time."""

    def __init__(self, now: Callable[[], float] = time.time) -> None:
        t = now()
        self._lock = threading.Lock()
        self.crm: dict[str, dict[str, Any]] = {
            CRM_ID: {"properties": {"email": REQUEST_EMAIL, "firstname": "Jane", "lastname": "Doe", "phone": "+1 415 555 0134",
                                    "city": "Oakland", "createdate": _iso(t - 610 * DAY)}, "notes": ["n-9001", "n-9002"]},
            DECOY_ID: {"properties": {"email": DECOY_EMAIL, "firstname": "Jane", "lastname": "Doe", "phone": "+44 20 7946 0321",
                                      "city": "Leeds", "createdate": _iso(t - 95 * DAY)}, "notes": ["n-9107"]},
            OTHER_ID: {"properties": {"email": "ana.lima@example.net", "firstname": "Ana", "lastname": "Lima", "phone": "",
                                      "city": "Porto", "createdate": _iso(t - 400 * DAY)}, "notes": []},
        }
        self.support: dict[str, dict[str, Any]] = {
            SUPPORT_ID: {"type": "contact", "id": SUPPORT_ID, "role": "user", "email": REQUEST_EMAIL, "name": REQUEST_NAME,
                         "phone": "+1 415 555 0134", "created_at": int(t - 540 * DAY),
                         "location": {"city": "Oakland", "country": "United States"},
                         "notes": {"type": "list", "data": [], "total_count": 2}},
        }
        self.customers: dict[str, dict[str, Any]] = {
            BILLING_ID: {"id": BILLING_ID, "object": "customer", "email": REQUEST_EMAIL, "name": REQUEST_NAME,
                         "phone": "+1 415 555 0134", "created": int(t - 600 * DAY), "metadata": {},
                         "address": {"line1": "1200 Broadway", "city": "Oakland", "postal_code": "94612", "country": "US"}},
        }
        self.invoices: dict[str, dict[str, Any]] = {
            iid: {"id": iid, "object": "invoice", "number": f"ACME-{1041 + i}", "customer": BILLING_ID, "status": "paid",
                  "customer_email": REQUEST_EMAIL, "customer_name": REQUEST_NAME, "amount_paid": 4900, "currency": "usd",
                  "created": int(t - (90 - 30 * i) * DAY)}
            for i, iid in enumerate(INVOICE_IDS)
        }
        self.members: dict[str, dict[str, Any]] = {
            MAILCHIMP_CONTACT: {"id": hashlib.md5(REQUEST_EMAIL.encode()).hexdigest(), "contact_id": MAILCHIMP_CONTACT,  # noqa: S324 - the email-marketing API's subscriber hash
                                "email_address": REQUEST_EMAIL, "full_name": REQUEST_NAME, "status": "subscribed",
                                "list_id": AUDIENCE_ID, "timestamp_signup": _iso(t - 500 * DAY),
                                "merge_fields": {"FNAME": "Jane", "LNAME": "Doe"}},
        }
        self.profiles: dict[str, dict[str, Any]] = {
            MIXPANEL_ID: {"$distinct_id": MIXPANEL_ID,
                          "$properties": {"$email": REQUEST_EMAIL, "$name": REQUEST_NAME, "$city": "Oakland",
                                          "$created": _iso(t - 530 * DAY), "$last_seen": _iso(t - 2 * DAY)}},
        }

    def _changed(self, system: str, kind: str, record: str, what: str) -> None:
        log_event(event="world", system=system, kind=kind, record=record, what=what)

    # -- the gateway's tools -------------------------------------------------------------------

    def actions(self) -> dict[str, LocalAction]:
        """The local gateway's actions, named and shaped like the catalog's (labelled as mocks)."""
        arr, obj, s, i = "array", "object", "string", "integer"
        return {
            "hubspot_crm_contacts_search": LocalAction(
                "MOCK CRM: search for contacts by filtering on properties.", self._crm_search,
                _schema(["after", "filter_groups", "limit", "properties", "sorts"], after=s, filter_groups=arr, limit=i, properties=arr, sorts=arr)),
            "hubspot_crm_contacts_get_by_contactid": LocalAction(
                "MOCK CRM: retrieve a contact by its ID.", self._crm_get,
                _schema(["contact_id"], contact_id=s, properties=arr, associations=arr)),
            "hubspot_crm_contacts_gdpr_delete_create": LocalAction(
                "MOCK CRM: permanently delete a contact and all associated content (GDPR).", self._crm_gdpr_delete,
                _schema(["object_id"], object_id=s, id_property=s)),
            "hubspot_files_gdpr_delete_by_fileid": LocalAction(
                "MOCK CRM: GDPR-delete a file.", lambda arguments: {"deleted": arguments.get("file_id")}, _schema(["file_id"], file_id=s)),
            "intercom_search_contacts": LocalAction(
                "MOCK support desk: find contacts matching conditions on their attributes.", self._support_search, _schema(["query"], query=obj)),
            "intercom_get_a_contact": LocalAction(
                "MOCK support desk: return one contact's profile by its contact id.", self._support_get, _schema(["contact_id"], contact_id=s)),
            "intercom_delete_a_contact": LocalAction(
                "MOCK support desk: permanently remove a contact and their data.", self._support_delete, _schema(["contact_id"], contact_id=s)),
            "stripe_get_customers": LocalAction(
                "MOCK payments: list customers, optionally by email.", self._customers_list, _schema([], email=s, limit=i)),
            "stripe_get_customers_customer": LocalAction(
                "MOCK payments: retrieve a customer.", self._customer_get, _schema(["customer"], customer=s)),
            "stripe_get_invoices": LocalAction(
                "MOCK payments: list invoices, optionally of one customer.", self._invoices_list, _schema([], customer=s, limit=i)),
            "stripe_post_customers_customer": LocalAction(
                "MOCK payments: update a customer.", self._customer_update,
                _schema(["customer"], customer=s, name=s, email=s, phone=s, address=obj, metadata=obj)),
            "stripe_delete_customers_customer": LocalAction(
                "MOCK payments: delete a customer.", self._customer_delete, _schema(["customer"], customer=s)),
            "mailchimp_search_members": LocalAction(
                "MOCK email marketing: search subscribers by email or name across every audience.", self._members_search,
                _schema(["query"], query=s, list_id=s)),
            "mailchimp_forget_contact": LocalAction(
                "MOCK email marketing: erase a contact and its personal data from the audience (privacy request).", self._member_forget,
                _schema(["audience_id", "contact_id"], audience_id=s, contact_id=s)),
            "mixpanel_engage_query": LocalAction(
                "MOCK product analytics: query user profiles that match a filter.", self._profiles_query, _schema(["project_id"], project_id=i, where=s)),
            "mixpanel_delete_profile": LocalAction(
                "MOCK product analytics: delete a user profile.", self._profile_delete, _schema(["project_id", "updates"], project_id=i, updates=arr)),
        }

    # CRM

    def _contact(self, cid: str, properties: list[str] | None = None, associations: list[str] | None = None) -> dict[str, Any]:
        record = self.crm[cid]
        props = {k: v for k, v in record["properties"].items() if properties is None or k in properties or k == "createdate"}
        out = {"id": cid, "properties": {**props, "hs_object_id": cid}, "createdAt": record["properties"]["createdate"], "archived": False}
        if associations and "notes" in associations:
            out["associations"] = {"notes": {"results": [{"id": n, "type": "contact_to_note"} for n in record["notes"]]}}
        return out

    @staticmethod
    def _matches(props: Mapping[str, Any], flt: Mapping[str, Any]) -> bool | None:
        value = str(props.get(flt.get("propertyName", ""), "") or "").lower()
        wanted = str(flt.get("value", "")).lower()
        if flt.get("operator") == "EQ":
            return value == wanted
        if flt.get("operator") == "CONTAINS_TOKEN":
            return wanted in value.split()
        return None

    def _crm_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "after", "filter_groups", "limit", "properties", "sorts"):
            return error
        found = []
        for cid, record in self.crm.items():
            groups = []
            for group in arguments["filter_groups"]:
                results = [self._matches(record["properties"], f) for f in group.get("filters", [])]
                if None in results:
                    return {"status": "error", "message": "This mock supports the EQ and CONTAINS_TOKEN operators only."}
                groups.append(all(results))
            if any(groups):  # groups are ORed, the filters in a group ANDed
                found.append(self._contact(cid, list(arguments["properties"])))
        return {"total": len(found), "results": found[: int(arguments["limit"])]}

    def _crm_get(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "contact_id"):
            return error
        cid = str(arguments["contact_id"])
        if cid not in self.crm:
            return {"status": "error", "category": "OBJECT_NOT_FOUND", "message": "resource not found"}
        return self._contact(cid, arguments.get("properties"), arguments.get("associations"))

    def _crm_gdpr_delete(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "object_id"):
            return error
        with self._lock:
            if self.crm.pop(str(arguments["object_id"]), None) is None:
                return {"status": "error", "category": "OBJECT_NOT_FOUND", "message": "resource not found"}
        self._changed("crm", "contact", str(arguments["object_id"]), "erased")
        return {"deleted": str(arguments["object_id"]), "associated_content": "deleted"}

    # support desk

    @staticmethod
    def _support_match(contact: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
        if query.get("operator") in ("AND", "OR"):
            results = [SimulatedSystems._support_match(contact, q) for q in query.get("value", [])]
            return all(results) if query["operator"] == "AND" else any(results)
        value, wanted = str(contact.get(query.get("field", ""), "") or "").lower(), str(query.get("value", "")).lower()
        return value == wanted if query.get("operator") == "=" else (wanted in value if query.get("operator") == "~" else False)

    def _support_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "query"):
            return error
        data = [c for c in self.support.values() if self._support_match(c, arguments["query"])]
        return {"type": "list", "data": data, "total_count": len(data), "pages": {"type": "pages", "page": 1, "total_pages": 1}}

    def _support_get(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "contact_id"):
            return error
        contact = self.support.get(str(arguments["contact_id"]))
        return contact or {"type": "error.list", "errors": [{"code": "not_found", "message": "User Not Found"}]}

    def _support_delete(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "contact_id"):
            return error
        with self._lock:
            if self.support.pop(str(arguments["contact_id"]), None) is None:
                return {"type": "error.list", "errors": [{"code": "not_found", "message": "User Not Found"}]}
        self._changed("support", "contact", str(arguments["contact_id"]), "erased")
        return {"id": arguments["contact_id"], "object": "contact", "deleted": True}

    # payments

    def _customers_list(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        email = arguments.get("email")
        data = [c for c in self.customers.values() if email is None or c.get("email") == email]  # case-sensitive, as the real API's
        return {"object": "list", "data": data[: int(arguments.get("limit", 10))], "has_more": False, "url": "/v1/customers"}

    def _customer_get(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "customer"):
            return error
        customer = self.customers.get(str(arguments["customer"]))
        return customer or {"error": {"type": "invalid_request_error", "code": "resource_missing",
                                      "message": f"No such customer: '{arguments['customer']}'"}}

    def _invoices_list(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        customer = arguments.get("customer")
        data = [v for v in self.invoices.values() if customer is None or v["customer"] == customer]
        return {"object": "list", "data": data[: int(arguments.get("limit", 10))], "has_more": False, "url": "/v1/invoices"}

    def _customer_update(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "customer"):
            return error
        with self._lock:
            customer = self.customers.get(str(arguments["customer"]))
            if customer is None:
                return self._customer_get(arguments)
            for field in ("name", "email", "phone", "description"):
                if field in arguments:
                    customer[field] = arguments[field] or None  # an empty string unsets the field, as in the real API
            if isinstance(arguments.get("address"), Mapping):
                address = {k: v for k, v in {**(customer.get("address") or {}), **arguments["address"]}.items() if v}
                customer["address"] = address or None
            if isinstance(arguments.get("metadata"), Mapping):
                customer["metadata"] = {**customer["metadata"], **arguments["metadata"]}
            redacted = not any(customer.get(f) for f in ("name", "email", "phone", "address"))
        self._changed("billing", "customer", customer["id"], "redacted" if redacted else "updated")
        return customer

    def _customer_delete(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.customers.pop(str(arguments.get("customer")), None)  # unreachable: the policy denies it
        return {"id": arguments.get("customer"), "object": "customer", "deleted": True}

    # email marketing

    def _members_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "query"):
            return error
        query = str(arguments["query"]).strip().lower()
        members = [m for m in self.members.values() if not arguments.get("list_id") or m["list_id"] == arguments["list_id"]]
        exact = [m for m in members if m["email_address"].lower() == query]
        full = [m for m in members if m not in exact and query in m["full_name"].lower()]
        return {"exact_matches": {"members": exact, "total_items": len(exact)}, "full_search": {"members": full, "total_items": len(full)}}

    def _member_forget(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "audience_id", "contact_id"):
            return error
        with self._lock:
            member = self.members.get(str(arguments["contact_id"]))
            if member is None or member["list_id"] != arguments["audience_id"]:
                return {"status": 404, "title": "Resource Not Found", "detail": "The requested contact could not be found."}
            del self.members[member["contact_id"]]
        self._changed("marketing", "member", member["contact_id"], "erased")
        return {"audience_id": arguments["audience_id"], "contact_id": arguments["contact_id"], "forgotten": True}

    # product analytics

    def _profiles_query(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "project_id"):
            return error
        if arguments["project_id"] != MIXPANEL_PROJECT:
            return {"status": "error", "error": "unknown project"}
        results = list(self.profiles.values())
        if where := arguments.get("where"):
            match = re.fullmatch(r'\s*properties\["([^"]+)"\]\s*==\s*"([^"]*)"\s*', str(where))
            if not match:
                return {"status": "error", "error": 'This mock supports where = properties["<name>"] == "<value>" only.'}
            key, value = match.groups()
            results = [p for p in results if str(p["$properties"].get(key, "")).lower() == value.lower()]
        return {"page": 0, "page_size": 1000, "results": results, "session_id": "1700000000-mock", "status": "ok", "total": len(results)}

    def _profile_delete(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if error := _missing(arguments, "project_id", "updates"):
            return error
        updates = arguments["updates"]
        if arguments["project_id"] != MIXPANEL_PROJECT or not isinstance(updates, list) or not updates:
            return {"status": 0, "error": "Request is invalid: a project id and at least one update are required."}
        bad = [u for u in updates if not isinstance(u, Mapping) or {"$token", "$distinct_id", "$delete"} - set(u)]
        if bad or any(u["$token"] != MIXPANEL_TOKEN for u in updates):
            return {"status": 0, "error": "Request is invalid: each update needs $token (the project token), $distinct_id and $delete."}
        gone = []
        with self._lock:
            for update in updates:
                if self.profiles.pop(str(update["$distinct_id"]), None) is not None:
                    gone.append(str(update["$distinct_id"]))
        for distinct_id in gone:
            self._changed("analytics", "profile", distinct_id, "erased")
        return {"status": 1, "error": None, "num_records_deleted": len(gone)}
