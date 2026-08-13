from __future__ import annotations

from mock_tools import DOC_BY_ID, EMAIL_BY_USER, execute_one
from prose_writer import validate_answer_grounding
from verifier import _substantial_facts, classify_grounding_error, deterministic_repair


def _cal_traj():
    attendee = EMAIL_BY_USER[42]
    title = "Sprint planning"
    args = {"title": title, "start": "2026-08-14T09:00:00", "attendee_email": attendee}
    result = execute_one("calendar.create", args)
    assert result.get("status") == 200
    return {
        "prompt": f"Create {title} with {attendee}",
        "steps": [{
            "tool": "calendar.create",
            "args": args,
            "result": result,
            "expect": "success",
        }],
    }, title, attendee, result["result"]["event_id"]


def _search_traj():
    doc_id = "doc.k7x2"
    assert doc_id in DOC_BY_ID
    args = {"doc_id": doc_id}
    result = execute_one("search.get", args)
    assert result.get("status") == 200
    inner = result["result"]
    return {
        "prompt": "Fetch the census document",
        "steps": [{
            "tool": "search.get",
            "args": args,
            "result": result,
            "expect": "success",
        }],
    }, inner["title"], inner["body"], doc_id


def test_calendar_create_omitting_event_id_is_ok():
    traj, title, attendee, _eid = _cal_traj()
    final = f"Created {title} for {attendee}."
    assert validate_answer_grounding(traj, final) is None


def test_invented_evt_id_fails():
    traj, title, attendee, _eid = _cal_traj()
    final = f"Created {title} for {attendee} as evt.deadbeef."
    err = validate_answer_grounding(traj, final)
    assert err is not None
    assert err.startswith("ungrounded opaque id")


def test_search_get_omitting_doc_id_is_ok():
    traj, title, body, _doc = _search_traj()
    final = f"Document {title}: {body}"
    assert validate_answer_grounding(traj, final) is None


def test_invented_doc_id_fails():
    traj, title, body, _doc = _search_traj()
    final = f"{title}: {body} (see doc.zzzz)"
    err = validate_answer_grounding(traj, final)
    assert err is not None
    assert err.startswith("ungrounded opaque id")


def test_ungrounded_opaque_id_is_not_repaired():
    traj, title, attendee, _eid = _cal_traj()
    final = f"Created {title} for {attendee} as evt.deadbeef."
    gerr = validate_answer_grounding(traj, final)
    assert gerr and gerr.startswith("ungrounded opaque id")
    assert classify_grounding_error(gerr) == "fabrication"
    assert deterministic_repair(traj, final, gerr) is None


def test_substantial_facts_skip_opaque_ids():
    traj, title, attendee, eid = _cal_traj()
    facts = _substantial_facts(traj["steps"][0], limit=8)
    assert eid not in facts
    assert any(title in f or attendee in f for f in facts)
    weak = "I finished the user's request."
    gerr = validate_answer_grounding(traj, weak)
    assert gerr and gerr.startswith("final missing key fact")
    fixed = deterministic_repair(traj, weak, gerr)
    assert fixed is not None
    assert eid not in fixed
    assert validate_answer_grounding(traj, fixed) is None
