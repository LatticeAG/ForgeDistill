from __future__ import annotations

import asyncio
import random
from pathlib import Path

from agentic_plans import build_chain
from distill_tools import Distiller
from prose_writer import _facts_from_step, validate_answer_grounding

MCONF = {"mode": "concise", "max_tokens": 64, "weight": 1}


def _roster(pin=True, two=True):
    roster = {
        "p1": {
            "base_url": "http://127.0.0.1:9/v1",
            "key_env": "",
            "concurrency": 1,
            "models": {"m1": dict(MCONF)},
        },
    }
    if two:
        roster["p2"] = {
            "base_url": "http://127.0.0.1:9/v1",
            "key_env": "",
            "concurrency": 1,
            "models": {"m2": dict(MCONF)},
        }
    if pin:
        roster["roles"] = {"answer": {"provider": "p2" if two else "p1", "model": "m2" if two else "m1"}}
    return roster


def _grounded_final(traj: dict) -> str:
    last = None
    for s in reversed(traj["steps"]):
        if s.get("result", {}).get("status") == 200:
            last = s
            break
    facts = _facts_from_step(last or traj["steps"][-1])
    text = "Completed. " + " ".join(facts[:8])
    err = validate_answer_grounding(traj, text)
    assert err is None, err
    return text


def _thoughts_blob(n: int, final: str) -> str:
    blocks = "\n".join(f"<thought>reason for step {i}</thought>" for i in range(n))
    return f"THOUGHTS:\n{blocks}\nFINAL_ANSWER:\n{final}"


def _run(d: Distiller, prov: str, model: str, traj: dict):
    return asyncio.run(
        d.generate_agentic_trace(prov, model, MCONF, traj=traj, rng=random.Random(0))
    )


def test_pin_used_when_set(tmp_path: Path, monkeypatch):
    traj = build_chain(random.Random(0), plan_index=0)
    n = len(traj["steps"])
    final = _grounded_final(traj)
    seen = []

    async def fake(self, prov, model, conf, content):
        seen.append((prov, model, "answer" if "generating the FINAL ANSWER" in content else "thoughts"))
        if "generating the FINAL ANSWER" in content:
            return {"ok": True, "content": f"FINAL_ANSWER:\n{final}"}
        return {"ok": True, "content": _thoughts_blob(n, final)}

    monkeypatch.setattr(Distiller, "_chat_user", fake)
    d = Distiller(_roster(pin=True, two=True), tmp_path, cross_teacher=True,
                  cross_teacher_rate=1.0, no_verify=True, holdout_frac=0)
    res = _run(d, "p1", "m1", traj)
    assert res.get("ok"), res
    trace = res["trace"]
    assert trace["teacher_thoughts"] == "p1/m1"
    assert trace["teacher_answer"] == "p2/m2"
    assert trace["eval"]["cross_teacher"] is True
    assert "cross_teacher_fallback" not in trace["eval"]
    assert ("p2", "m2", "answer") in seen


def test_pin_equal_to_thought_samples_other(tmp_path: Path, monkeypatch):
    traj = build_chain(random.Random(1), plan_index=0)
    n = len(traj["steps"])
    final = _grounded_final(traj)
    seen = []

    async def fake(self, prov, model, conf, content):
        seen.append((prov, model))
        if "generating the FINAL ANSWER" in content:
            return {"ok": True, "content": f"FINAL_ANSWER:\n{final}"}
        return {"ok": True, "content": _thoughts_blob(n, final)}

    monkeypatch.setattr(Distiller, "_chat_user", fake)
    roster = _roster(pin=False, two=True)
    roster["roles"] = {"answer": {"provider": "p1", "model": "m1"}}
    d = Distiller(roster, tmp_path, cross_teacher=True, cross_teacher_rate=1.0,
                  no_verify=True, holdout_frac=0)
    res = _run(d, "p1", "m1", traj)
    assert res.get("ok"), res
    trace = res["trace"]
    assert trace["teacher_thoughts"] == "p1/m1"
    assert trace["teacher_answer"] == "p2/m2"
    assert ("p2", "m2") in seen


def test_single_route_disables_split(tmp_path: Path, monkeypatch):
    traj = build_chain(random.Random(2), plan_index=0)
    n = len(traj["steps"])
    final = _grounded_final(traj)

    async def fake(self, prov, model, conf, content):
        return {"ok": True, "content": _thoughts_blob(n, final)}

    monkeypatch.setattr(Distiller, "_chat_user", fake)
    d = Distiller(_roster(pin=False, two=False), tmp_path, cross_teacher=True,
                  cross_teacher_rate=1.0, no_verify=True, holdout_frac=0)
    res = _run(d, "p1", "m1", traj)
    assert res.get("ok"), res
    trace = res["trace"]
    assert "teacher_thoughts" not in trace
    assert "teacher_answer" not in trace
    assert trace["eval"]["cross_teacher"] is False
    assert "cross_teacher_fallback" not in trace["eval"]


def test_fallback_omits_fields_and_stamps(tmp_path: Path, monkeypatch):
    traj = build_chain(random.Random(3), plan_index=0)
    n = len(traj["steps"])
    final = _grounded_final(traj)

    async def fake(self, prov, model, conf, content):
        is_answer = "generating the FINAL ANSWER" in content
        if is_answer and (prov, model) == ("p2", "m2"):
            return {"ok": True, "content": "not parseable as a final"}
        if is_answer:
            return {"ok": True, "content": f"FINAL_ANSWER:\n{final}"}
        return {"ok": True, "content": _thoughts_blob(n, final)}

    monkeypatch.setattr(Distiller, "_chat_user", fake)
    d = Distiller(_roster(pin=True, two=True), tmp_path, cross_teacher=True,
                  cross_teacher_rate=1.0, no_verify=True, holdout_frac=0)
    res = _run(d, "p1", "m1", traj)
    assert res.get("ok"), res
    trace = res["trace"]
    assert "teacher_thoughts" not in trace
    assert "teacher_answer" not in trace
    assert trace["eval"]["cross_teacher"] is False
    assert trace["eval"]["cross_teacher_fallback"] is True


def test_select_answer_route_exists():
    assert hasattr(Distiller, "_select_answer_route")


def test_valid_pin_never_calls_rng_choice(tmp_path: Path):
    d = Distiller(_roster(pin=True, two=True), tmp_path, holdout_frac=0)

    class Boom:
        def choice(self, seq):
            raise AssertionError("sampling must not override a valid pin")

        def random(self):
            return 0.0

    route = d._select_answer_route("p1", "m1", Boom())
    assert route is not None
    assert route[0] == "p2" and route[1] == "m2"


def test_unused_pin_logs_once(tmp_path: Path, capsys):
    roster = _roster(pin=False, two=True)
    roster["roles"] = {"answer": {"provider": "missing", "model": "nope"}}
    d = Distiller(roster, tmp_path, holdout_frac=0)
    rng = random.Random(0)
    a = d._select_answer_route("p1", "m1", rng)
    b = d._select_answer_route("p1", "m1", rng)
    assert a is not None and b is not None
    out = capsys.readouterr().out
    assert out.count("roles.answer pin unused; sampling") == 1


def test_rate_zero_does_not_split(tmp_path: Path, monkeypatch):
    traj = build_chain(random.Random(4), plan_index=0)
    n = len(traj["steps"])
    final = _grounded_final(traj)

    async def fake(self, prov, model, conf, content):
        return {"ok": True, "content": _thoughts_blob(n, final)}

    monkeypatch.setattr(Distiller, "_chat_user", fake)
    d = Distiller(_roster(pin=True, two=True), tmp_path, cross_teacher=True,
                  cross_teacher_rate=0.0, no_verify=True, holdout_frac=0)
    res = _run(d, "p1", "m1", traj)
    assert res.get("ok"), res
    trace = res["trace"]
    assert "teacher_thoughts" not in trace
    assert "teacher_answer" not in trace
    assert trace["eval"]["cross_teacher"] is False
    assert "cross_teacher_fallback" not in trace["eval"]


def test_single_route_logs_disabled_once(tmp_path: Path, monkeypatch, capsys):
    traj = build_chain(random.Random(5), plan_index=0)
    n = len(traj["steps"])
    final = _grounded_final(traj)

    async def fake(self, prov, model, conf, content):
        return {"ok": True, "content": _thoughts_blob(n, final)}

    monkeypatch.setattr(Distiller, "_chat_user", fake)
    d = Distiller(_roster(pin=False, two=False), tmp_path, cross_teacher=True,
                  cross_teacher_rate=1.0, no_verify=True, holdout_frac=0)
    r1 = _run(d, "p1", "m1", traj)
    r2 = _run(d, "p1", "m1", traj)
    assert r1.get("ok") and r2.get("ok")
    out = capsys.readouterr().out
    assert out.count("only one route in worklist; split disabled") == 1
