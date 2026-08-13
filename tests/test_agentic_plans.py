from __future__ import annotations

import random

import pytest

from agentic_plans import (
    PLANS,
    SKILLS,
    VAR_POOLS,
    _resolve_path,
    _resolve_ref,
    build_chain,
    validate_chain,
)
from mock_tools import DOC_BY_ID, EMAIL_BY_USER, execute_one
from prose_writer import _key_facts_from_payload

EXISTING_IDS = [
    "user-email-welcome", "user-email-notify", "user-email-get-address",
    "user-email-account", "user-plan-count", "user-plan-list",
    "weather-then-user-email", "weather-report-email", "db-count-email",
    "file-status-email", "user-file-status", "bad-id-stop", "bad-id-stop-v2",
    "bad-city-only-if", "two-user-notify", "two-user-subject",
    "join-two-users-plan-email", "file-missing-no-email",
    "get-user-str-retry-int", "fanout-partial-fail", "plan-filter-email-count",
    "user-then-weather-email", "idempotent-reuse-email", "empty-subject-retry",
    "weather-db-join-email", "guess-email-then-correct", "five-step-digest",
    "two-candidate-id", "stop-after-welcome",
]
EXISTING_SKILLS = {
    "user-email-welcome": ["multi_hop", "stop"],
    "user-email-notify": ["multi_hop", "stop"],
    "user-email-get-address": ["multi_hop", "stop"],
    "user-email-account": ["multi_hop", "stop"],
    "user-plan-count": ["multi_hop"],
    "user-plan-list": ["multi_hop"],
    "weather-then-user-email": ["multi_hop", "reorder"],
    "weather-report-email": ["multi_hop", "reorder"],
    "db-count-email": ["multi_hop"],
    "file-status-email": ["multi_hop"],
    "user-file-status": ["multi_hop", "digest"],
    "bad-id-stop": ["branch"],
    "bad-id-stop-v2": ["branch"],
    "bad-city-only-if": ["branch"],
    "two-user-notify": ["fanout", "multi_hop"],
    "two-user-subject": ["fanout", "multi_hop"],
    "join-two-users-plan-email": ["join", "multi_hop"],
    "file-missing-no-email": ["branch"],
    "get-user-str-retry-int": ["recovery"],
    "fanout-partial-fail": ["fanout", "branch"],
    "plan-filter-email-count": ["multi_hop"],
    "user-then-weather-email": ["reorder", "multi_hop"],
    "idempotent-reuse-email": ["idempotent", "recovery", "schema"],
    "empty-subject-retry": ["schema", "recovery"],
    "weather-db-join-email": ["join", "multi_hop"],
    "guess-email-then-correct": ["recovery"],
    "five-step-digest": ["digest", "multi_hop"],
    "two-candidate-id": ["disambiguate", "recovery"],
    "stop-after-welcome": ["stop", "multi_hop"],
}
NEW_IDS = [
    "search-get-email", "search-miss-stop", "search-then-calc-email",
    "search-reorder-user-first", "cal-list-email", "cal-create-learned",
    "cal-create-guess-then-correct", "cal-fanout-two", "cal-stop-after-create",
    "calc-then-email", "calc-bad-expr-retry", "crm-owner-email",
    "crm-seats-calc-email", "crm-unknown-stop", "nested-schema-retry",
    "search-disambiguate", "digest-search-cal-crm", "idempotent-cal-reuse-email",
]
TIERS = {"easy", "medium", "hard", "expert"}


def _by_id():
    return {p["id"]: (i, p) for i, p in enumerate(PLANS)}


def test_len_plans_and_skills():
    assert len(PLANS) >= 45
    assert len(PLANS) == 47
    assert len(SKILLS) >= 14
    for tag in ("search", "calendar", "arithmetic", "nested"):
        assert tag in SKILLS


def test_every_plan_has_tier():
    for p in PLANS:
        assert p.get("tier") in TIERS, p["id"]


def test_existing_ids_skills_stable():
    ids = [p["id"] for p in PLANS]
    assert ids[:29] == EXISTING_IDS
    by = _by_id()
    for pid, skills in EXISTING_SKILLS.items():
        assert by[pid][1]["skills"] == skills


def test_300_chains_invalid_zero():
    rng = random.Random(0)
    invalid = 0
    for _ in range(300):
        traj = build_chain(rng)
        err = validate_chain(traj["steps"])
        if err:
            invalid += 1
        assert traj.get("tier") in TIERS
    assert invalid == 0


def test_every_new_plan_builds():
    rng = random.Random(1)
    by = _by_id()
    for pid in NEW_IDS:
        assert pid in by, pid
        traj = build_chain(rng, plan_index=by[pid][0])
        assert traj["plan_id"] == pid
        assert traj["tier"] in TIERS
        err = validate_chain(traj["steps"])
        assert err is None, f"{pid}: {err}"


def test_nested_ref_owner_email():
    payload = execute_one("crm.get_account", {"account_id": 1001})
    steps = [{"result": payload}]
    got = _resolve_ref("$0.result.account.owner.email", steps)
    assert got == EMAIL_BY_USER[42]
    seats = _resolve_ref("$0.result.account.billing.seats", steps)
    assert seats == 12


def test_nested_hits_and_first_number():
    q = execute_one("search.query", {"q": "census"})
    steps = [{"result": q}]
    doc_id = _resolve_ref("$0.result.hits[0].doc_id", steps)
    got = execute_one("search.get", {"doc_id": doc_id})
    steps.append({"result": got})
    n = _resolve_ref("$1.result.body|first_number", steps)
    assert n == 48
    walked = _resolve_path(got["result"], ".body|first_number")
    assert walked == 48


def test_inline_nested_seats_expr():
    rng = random.Random(2)
    by = _by_id()
    traj = build_chain(rng, plan_index=by["crm-seats-calc-email"][0])
    calc = traj["steps"][1]
    assert calc["tool"] == "calc.eval"
    assert calc["result"]["status"] == 200
    seats = traj["steps"][0]["result"]["result"]["account"]["billing"]["seats"]
    price = traj["vars"]["price"]
    assert calc["result"]["result"]["value"] == seats * price


def test_search_then_calc_uses_first_number():
    rng = random.Random(3)
    by = _by_id()
    traj = build_chain(rng, plan_index=by["search-then-calc-email"][0])
    body = traj["steps"][1]["result"]["result"]["body"]
    assert "48" in body or any(ch.isdigit() for ch in body)
    assert traj["steps"][2]["tool"] == "calc.eval"
    assert traj["steps"][2]["result"]["status"] == 200


def test_pools_have_no_opaque_ids():
    blob = str(VAR_POOLS)
    for email in EMAIL_BY_USER.values():
        assert email not in blob
    for did in DOC_BY_ID:
        assert did not in blob
    assert "evt." not in blob


def test_build_chain_tier_and_exclude():
    rng = random.Random(4)
    traj = build_chain(rng, tier="easy")
    assert traj["tier"] == "easy"
    keep = PLANS[0]["id"]
    exclude = {p["id"] for p in PLANS} - {keep}
    for _ in range(15):
        t = build_chain(random.Random(_), exclude_ids=exclude)
        assert t["plan_id"] == keep


def test_plan_space_exhausted():
    rng = random.Random(0)
    all_ids = {p["id"] for p in PLANS}
    with pytest.raises(RuntimeError, match="plan space exhausted"):
        build_chain(rng, exclude_ids=all_ids)


def test_key_facts_nested_crm_and_hits():
    crm = execute_one("crm.get_account", {"account_id": 1001})
    facts = _key_facts_from_payload(crm)
    assert EMAIL_BY_USER[42] in facts
    assert "pro" in facts
    assert "12" in facts
    q = execute_one("search.query", {"q": "census"})
    hf = _key_facts_from_payload(q)
    assert any(f.startswith("doc.") for f in hf)
    calc = execute_one("calc.eval", {"expr": "20 * 3"})
    assert "60" in _key_facts_from_payload(calc)
