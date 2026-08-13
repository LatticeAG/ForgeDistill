"""Deterministic grounding repair plus optional LLM PASS/FAIL gate.

Repair runs on every missing-fact grounding miss. Fabrications are dropped,
never edited. The LLM verifier (roster roles.verifier) does not rewrite prose.
"""
from __future__ import annotations

from typing import Literal

from prose_writer import _facts_from_step, _is_opaque_id, validate_answer_grounding

VerifyKind = Literal["missing_fact", "fabrication", "empty", "other"]
VerifyVerdict = Literal["pass", "fail"]


def classify_grounding_error(gerr: str) -> VerifyKind:
    s = (gerr or "").strip()
    if s.startswith("final missing key fact"):
        return "missing_fact"
    if s.startswith("ungrounded "):
        return "fabrication"
    if s == "empty final answer" or s.startswith("empty final answer"):
        return "empty"
    return "other"


def _substantial_facts(step: dict, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for f in _facts_from_step(step):
        fl = f.lower()
        if fl in ("true", "false"):
            continue
        if len(f) < 2 or fl in seen:
            continue
        if _is_opaque_id(f):
            continue
        seen.add(fl)
        out.append(f)
        if len(out) >= limit:
            break
    return out


def deterministic_repair(traj, final, gerr) -> str | None:
    """Append up to 4 last-success facts. Only for missing_fact. Else None."""
    if classify_grounding_error(gerr) != "missing_fact":
        return None
    steps = traj.get("steps") or []
    last_success = None
    for s in reversed(steps):
        if s.get("result", {}).get("status") == 200:
            last_success = s
            break
    if last_success is None and steps:
        last_success = steps[-1]
    if last_success is None:
        return None
    facts = _substantial_facts(last_success, 4)
    if not facts:
        return None
    sentence = " Recorded facts: " + "; ".join(facts) + "."
    repaired = (final or "").rstrip() + sentence
    if validate_answer_grounding(traj, repaired) is not None:
        return None
    return repaired


def build_verify_prompt(traj, final) -> str:
    """Ask for a single line PASS or FAIL: <reason>. Do not rewrite prose."""
    lines = [
        "You are verifying a final answer against a tool execution record.",
        "Reply with exactly one line: PASS  or  FAIL: <short reason>.",
        "Do not rewrite the answer. Do not emit tool calls.",
        "",
        "USER: " + str(traj.get("prompt") or ""),
        "",
        "TOOL STEPS:",
    ]
    for i, s in enumerate(traj.get("steps") or []):
        lines.append(
            f"  {i}: {s.get('tool')} args={s.get('args')} result={s.get('result')}"
        )
    lines += [
        "",
        "FINAL_ANSWER:",
        str(final or ""),
        "",
        "PASS if the final answer uses only values from the record and mentions "
        "a key fact from the last successful result. FAIL if it fabricates values "
        "or omits the outcome.",
    ]
    return "\n".join(lines)


def parse_verify_reply(content) -> VerifyVerdict | None:
    if not content or not str(content).strip():
        return None
    line = str(content).strip().splitlines()[0].strip()
    up = line.upper()
    if up == "PASS" or up.startswith("PASS ") or up.startswith("PASS:"):
        return "pass"
    if up == "FAIL" or up.startswith("FAIL:") or up.startswith("FAIL "):
        return "fail"
    return None


def should_llm_verify(repaired: bool, rng, sample_rate: float) -> bool:
    if repaired:
        return True
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return rng.random() < rate
