"""Format-contract tests for the prose lane (0.5.1).

Two axes are accepted as teacher INPUT: canonical <thought> tags, and bare
labelled prose ("THOUGHTS:" / "FINAL_ANSWER:"). Both must produce the SAME
exported trace, because assemble_trace re-emits canonical tags itself. The
literal-syntax reminder must be present in the prompt, and the label parser
must stay strict enough to reject terse or malformed replies.
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_plans import build_chain, validate_chain  # noqa: E402
from prose_writer import (  # noqa: E402
    LITERAL_TAG_REMINDER,
    assemble_trace,
    build_prose_prompt,
    parse_teacher_output,
)


@pytest.fixture()
def traj():
    t = build_chain(random.Random(11))
    assert validate_chain(t["steps"]) is None
    return t


def test_reminder_present_in_prompt(traj):
    prompt = build_prose_prompt(traj)
    assert LITERAL_TAG_REMINDER in prompt
    assert "LITERAL SYNTAX" in prompt
    # the pre-existing contract must remain untouched
    assert "<thought>reasoning for step 1</thought>" in prompt
    assert "FINAL_ANSWER:" in prompt


def test_tag_reply_reports_tag_axis(traj):
    n = len(traj["steps"])
    content = "\n".join(
        [f"<thought>step {i+1} reasoning about the call and why.</thought>" for i in range(n)]
        + ["FINAL_ANSWER: done, real values used."]
    )
    parsed = parse_teacher_output(content, n)
    assert parsed is not None
    assert parsed["format"] == "tags"
    assert len(parsed["thoughts"]) == n


def test_label_reply_recovered(traj):
    n = len(traj["steps"])
    content = (
        "THOUGHTS:\n"
        + "\n".join(f"I will call tool number {i+1} because the prior result requires it." for i in range(n))
        + "\n\nFINAL_ANSWER:\nDone using the real values from the results.\n"
        + "\nREASONING:\ninternal scratch that must not leak\n"
    )
    parsed = parse_teacher_output(content, n)
    assert parsed is not None
    assert parsed["format"] == "labels"
    assert len(parsed["thoughts"]) == n
    assert "REASONING" not in parsed["final"]
    assert "internal scratch" not in parsed["final"]


def test_label_axis_rejected_when_too_few_thoughts(traj):
    n = len(traj["steps"])
    if n < 2:
        pytest.skip("chain too short to test the floor")
    content = (
        "THOUGHTS:\nI will call one tool because of the request.\n\n"
        "FINAL_ANSWER:\nDone.\n"
    )
    assert parse_teacher_output(content, n) is None


def test_no_thoughts_and_no_labels_rejected():
    assert parse_teacher_output("FINAL_ANSWER: just an answer", 2) is None
    assert parse_teacher_output("", 1) is None
    assert parse_teacher_output("THOUGHTS:\n\nFINAL_ANSWER:\n", 1) is None


def test_tags_win_when_both_shapes_present(traj):
    n = len(traj["steps"])
    content = (
        "THOUGHTS:\nlabelled line that should be ignored when tags exist\n\n"
        + "\n".join(f"<thought>tagged thought {i+1} with enough length.</thought>" for i in range(n))
        + "\nFINAL_ANSWER: tagged final answer.\n"
    )
    parsed = parse_teacher_output(content, n)
    assert parsed is not None
    assert parsed["format"] == "tags"
    assert all("<thought>" not in t for t in parsed["thoughts"])


def test_exported_trace_identical_across_axes(traj):
    """The student must see the same thing whichever axis the teacher used."""
    n = len(traj["steps"])
    steps = traj["steps"]
    tag_content = "\n".join(
        [f"<thought>reasoning sentence {i+1} about the call.</thought>" for i in range(n)]
        + ["FINAL_ANSWER: grounded answer."]
    )
    label_content = "\n".join(
        ["THOUGHTS:"]
        + [f"reasoning sentence {i+1} about the call." for i in range(n)]
        + ["", "FINAL_ANSWER:", "grounded answer."]
    )
    meta = {"teacher": "unit-test", "mode": "thinking"}
    sys_prompt = "system contract"
    tag_parsed = parse_teacher_output(tag_content, n)
    label_parsed = parse_teacher_output(label_content, n)
    assert tag_parsed is not None and label_parsed is not None
    tag_trace, tag_err = assemble_trace(traj, meta, tag_parsed, sys_prompt)
    label_trace, label_err = assemble_trace(traj, meta, label_parsed, sys_prompt)
    assert tag_err is None and label_err is None
    assert tag_trace is not None and label_trace is not None
    assert tag_trace["messages"] == label_trace["messages"]
    assert tag_trace["prose_format"] == "tags"
    assert label_trace["prose_format"] == "labels"
    assistant = [m for m in tag_trace["messages"] if m.get("role") == "assistant"]
    assert assistant, "trace must contain assistant turns"
    # one assistant turn per step, each carrying exactly one canonical thought block
    with_thoughts = [m for m in assistant if "<thought>" in m["content"]]
    assert len(with_thoughts) == n
    assert all(m["content"].count("<thought>") == 1 for m in with_thoughts)