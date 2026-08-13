from __future__ import annotations

import hashlib
import random

from mock_tools import (
    ACCOUNT_BY_ID,
    ALLOWED_ARG_KEYS,
    DOC_BY_ID,
    DOC_BY_QUERY,
    EMAIL_BY_USER,
    USERS,
    execute_one,
    execute_tool_calls,
    mint_event_id,
)


def test_calendar_list_known_user():
    r = execute_one("calendar.list", {"user_id": 42})
    assert r["status"] == 200
    events = r["result"]["events"]
    assert 1 <= len(events) <= 2
    ev = events[0]
    assert ev["event_id"] == mint_event_id(42, ev["title"])
    assert ev["event_id"].startswith("evt.")
    assert len(ev["event_id"]) == 16  # evt. + 12 hex


def test_calendar_list_unknown_user():
    r = execute_one("calendar.list", {"user_id": 9999})
    assert r["status"] == 404
    assert r["error"]["code"] == "USER_NOT_FOUND"


def test_calendar_create_mints_from_attendee_email():
    email = EMAIL_BY_USER[42]
    title = "Sprint planning"
    r = execute_one("calendar.create", {
        "title": title,
        "start": "2026-08-14T09:00:00",
        "attendee_email": email,
    })
    assert r["status"] == 200
    assert r["result"]["created"] is True
    assert r["result"]["attendee_email"] == email
    assert r["result"]["event_id"] == mint_event_id(42, title)
    expected = "evt." + hashlib.sha256(f"42:{title}".encode("utf-8")).hexdigest()[:12]
    assert r["result"]["event_id"] == expected


def test_guessed_attendee_email_400():
    r = execute_one("calendar.create", {
        "title": "Meet",
        "start": "2026-08-14T09:00:00",
        "attendee_email": "alice@example.com",
    })
    assert r["status"] == 400
    assert r["error"]["code"] == "UNREGISTERED_ADDRESS"


def test_calendar_create_empty_title():
    r = execute_one("calendar.create", {
        "title": "",
        "start": "2026-08-14T09:00:00",
        "attendee_email": EMAIL_BY_USER[42],
    })
    assert r["status"] == 400
    assert r["error"]["code"] == "INVALID_TITLE"


def test_search_query_intersection_and_first_token_order():
    r = execute_one("search.query", {"q": "quarterly census"})
    assert r["status"] == 200
    hits = r["result"]["hits"]
    q_ids = {d["doc_id"] for d in DOC_BY_QUERY["quarterly"]}
    c_ids = {d["doc_id"] for d in DOC_BY_QUERY["census"]}
    expected_ids = q_ids & c_ids
    assert {h["doc_id"] for h in hits} == expected_ids
    order = [d["doc_id"] for d in DOC_BY_QUERY["quarterly"] if d["doc_id"] in expected_ids]
    assert [h["doc_id"] for h in hits] == order
    assert "body" not in hits[0]


def test_search_query_empty_hit_token():
    r = execute_one("search.query", {"q": "xyzzy"})
    assert r["status"] == 200
    assert r["result"]["hits"] == []


def test_search_query_empty_q():
    r = execute_one("search.query", {"q": "  "})
    assert r["status"] == 400
    assert r["error"]["code"] == "INVALID_QUERY"


def test_search_get_and_numbered_body():
    r = execute_one("search.get", {"doc_id": "doc.k7x2"})
    assert r["status"] == 200
    assert r["result"]["doc_id"] == "doc.k7x2"
    assert "48" in r["result"]["body"]


def test_guessed_doc_id_400():
    r = execute_one("search.get", {"doc_id": "doc.guessed"})
    assert r["status"] == 400
    assert r["error"]["code"] == "UNKNOWN_DOC"


def test_doc_ids_not_query_concatenations():
    for tok, docs in DOC_BY_QUERY.items():
        for d in docs:
            did = d["doc_id"]
            assert tok not in did
            assert did in DOC_BY_ID


def test_calc_eval_whitespace_and_value():
    r = execute_one("calc.eval", {"expr": "  12 + 5  "})
    assert r["status"] == 200
    assert r["result"]["value"] == 17


def test_calc_div_zero_and_invalid():
    z = execute_one("calc.eval", {"expr": "8 / 0"})
    assert z["status"] == 400
    assert z["error"]["code"] == "INVALID_EXPR"
    bad = execute_one("calc.eval", {"expr": "foo + 1"})
    assert bad["status"] == 400
    assert bad["error"]["code"] == "INVALID_EXPR"
    var = execute_one("calc.eval", {"expr": "x + 1"})
    assert var["status"] == 400


def test_crm_account_and_unknown():
    r = execute_one("crm.get_account", {"account_id": 1001})
    assert r["status"] == 200
    acct = r["result"]["account"]
    assert acct["name"] == "Northwind"
    assert acct["owner"]["email"] == EMAIL_BY_USER[42]
    assert acct["billing"]["seats"] == 12
    assert 1002 in ACCOUNT_BY_ID
    bad = execute_one("crm.get_account", {"account_id": 9999})
    assert bad["status"] == 404
    assert bad["error"]["code"] == "UNKNOWN_ACCOUNT"


def test_guessed_owner_email_400():
    r = execute_one("send_email", {
        "to": "alice.owner@northwind.example",
        "subject": "Hi",
        "body": "x",
    })
    assert r["status"] == 400
    assert r["error"]["code"] == "UNREGISTERED_ADDRESS"


def test_execute_tool_calls_allowlist_new_keys():
    needed = {"q", "doc_id", "expr", "account_id", "title", "start",
              "attendee_email", "event_id"}
    assert needed <= ALLOWED_ARG_KEYS
    batch = execute_tool_calls([
        {"name": "calc.eval", "arguments": {"expr": "2 * 3"}},
        {"name": "search.query", "arguments": {"q": "census"}},
    ])
    assert batch["calc.eval"]["status"] == 200
    assert batch["search.query"]["status"] == 200


def test_every_known_user_has_seed_events():
    for uid in USERS:
        r = execute_one("calendar.list", {"user_id": uid})
        assert r["status"] == 200
        assert 1 <= len(r["result"]["events"]) <= 2
