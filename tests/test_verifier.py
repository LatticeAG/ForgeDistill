from __future__ import annotations

import random

from agentic_plans import build_chain
from prose_writer import validate_answer_grounding
from verifier import (
    classify_grounding_error,
    deterministic_repair,
    parse_verify_reply,
    should_llm_verify,
)


def _welcome_traj():
    return build_chain(random.Random(0), plan_index=0)


def test_classify_kinds():
    assert classify_grounding_error(
        "final missing key fact from last result (expected one of ['x'])"
    ) == "missing_fact"
    assert classify_grounding_error("ungrounded email: a@b.c") == "fabrication"
    assert classify_grounding_error("ungrounded number: 99") == "fabrication"
    assert classify_grounding_error("empty final answer") == "empty"
    assert classify_grounding_error("no steps to ground against") == "other"


def test_missing_fact_repair_then_grounding_passes():
    traj = _welcome_traj()
    weak = "I finished the user's request."
    gerr = validate_answer_grounding(traj, weak)
    assert gerr and gerr.startswith("final missing key fact")
    fixed = deterministic_repair(traj, weak, gerr)
    assert fixed is not None
    assert validate_answer_grounding(traj, fixed) is None
    assert "Recorded facts:" in fixed


def test_ungrounded_email_is_not_repaired():
    traj = _welcome_traj()
    lie = "I emailed zed.q.0000@nowhere.invalid about the account."
    gerr = validate_answer_grounding(traj, lie)
    assert gerr and gerr.startswith("ungrounded ")
    assert deterministic_repair(traj, lie, gerr) is None
    assert validate_answer_grounding(traj, lie) is not None


def test_should_llm_verify_true_for_repaired_even_at_rate_zero():
    rng = random.Random(0)
    assert should_llm_verify(True, rng, 0.0) is True
    rng = random.Random(0)
    assert should_llm_verify(False, rng, 0.0) is False


def test_parse_verify_reply():
    assert parse_verify_reply("PASS") == "pass"
    assert parse_verify_reply("FAIL: invented email") == "fail"
    assert parse_verify_reply("") is None
    assert parse_verify_reply("maybe") is None
