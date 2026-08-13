from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from agentic_plans import PLANS, build_chain
from curriculum import split_plan_ids as curriculum_split_plan_ids
from eval_live import (
    LABEL,
    LiveRunner,
    ReplayStudent,
    WrongArgStudent,
    aggregate_report,
    split_plan_ids,
)
import eval_live

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
EVAL = str(REPO / "src" / "eval_live.py")

# Ten plan_index values spanning multi_hop, branch, fanout, recovery, search,
# calendar, and a 6-step digest. None is the 1-tool simple plan.
REPLAY_INDICES = [0, 4, 11, 14, 18, 25, 29, 34, 36, 45]


def _run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [PY, EVAL, *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env=env,
    )


def test_split_plan_ids_imported_from_curriculum():
    assert eval_live.split_plan_ids is curriculum_split_plan_ids
    train, hold = split_plan_ids(PLANS, 0.15, 42)
    assert set(train).isdisjoint(set(hold))
    assert set(train) | set(hold) == {p["id"] for p in PLANS}


def test_replay_student_scores_one_on_ten_built_chains():
    runner = LiveRunner()
    scores = []
    for idx in REPLAY_INDICES:
        rng = random.Random(1000 + idx)
        traj = build_chain(rng, plan_index=idx)
        student = ReplayStudent(traj)
        rollout = runner.run_one(student, traj["prompt"])
        scored = runner.score_rollout(traj, rollout)
        scores.append(scored)
        assert scored["name_sequence_match"] == 1.0, (idx, scored)
        assert scored["dependency_arg_match"] == 1.0, (idx, scored)
        assert scored["format_ok"] == 1.0, (idx, scored)
        assert scored["grounding_ok"] == 1.0, (idx, scored)
        assert scored["n_gold_steps"] == len(traj["steps"])
        assert scored["n_student_steps"] == len(traj["steps"])
    assert len(scores) == 10
    assert {s["plan_index"] for s in scores} == set(REPLAY_INDICES)


def test_score_plan_binds_replay_student():
    runner = LiveRunner()
    student = ReplayStudent()
    for idx in REPLAY_INDICES[:3]:
        scored = runner.score_plan(student, PLANS[idx]["id"], random.Random(idx))
        assert scored["name_sequence_match"] == 1.0
        assert scored["dependency_arg_match"] == 1.0
        assert scored["format_ok"] == 1.0
        assert scored["grounding_ok"] == 1.0


def test_wrong_arg_student_dependency_arg_match_below_one():
    runner = LiveRunner()
    rng = random.Random(7)
    traj = build_chain(rng, plan_index=0)
    student = WrongArgStudent(traj)
    rollout = runner.run_one(student, traj["prompt"])
    scored = runner.score_rollout(traj, rollout)
    assert scored["name_sequence_match"] == 1.0
    assert scored["dependency_arg_match"] < 1.0
    assert scored["format_ok"] == 1.0


def test_replay_cli_scores_one_on_ten_built_chains(tmp_path: Path):
    holdout_ids = [PLANS[i]["id"] for i in REPLAY_INDICES]
    hold_path = tmp_path / "holdout_plan_ids.json"
    hold_path.write_text(json.dumps(holdout_ids), encoding="utf-8")
    out_path = tmp_path / "live_eval.json"
    r = _run_cli([
        "--replay",
        "--holdout", str(hold_path),
        "--n", "10",
        "--out", str(out_path),
        "--seed", "0",
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    assert LABEL in r.stdout
    assert "BFCL v3" not in r.stdout.lower() or "not a BFCL" in r.stdout
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["label"] == LABEL
    assert report["n"] == 10
    assert len(report["per_plan"]) == 10
    s = report["scores"]
    assert s["name_sequence_match"] == 1.0
    assert s["dependency_arg_match"] == 1.0
    assert s["grounding_ok"] == 1.0
    assert s["format_ok"] == 1.0
    for row in report["per_plan"]:
        assert row["name_sequence_match"] == 1.0
        assert row["dependency_arg_match"] == 1.0
        assert row["grounding_ok"] == 1.0
        assert row["format_ok"] == 1.0


def test_replay_cli_does_not_hit_network(tmp_path: Path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("Forge Live Tool Eval --replay must not use HTTP")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    holdout_ids = [PLANS[i]["id"] for i in REPLAY_INDICES]
    hold_path = tmp_path / "holdout_plan_ids.json"
    hold_path.write_text(json.dumps(holdout_ids), encoding="utf-8")
    out_path = tmp_path / "live_eval.json"
    rc = eval_live.main([
        "--replay",
        "--holdout", str(hold_path),
        "--n", "10",
        "--out", str(out_path),
        "--seed", "1",
    ])
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["scores"]["name_sequence_match"] == 1.0


def test_irrelevance_category_present_but_unscored():
    runner = LiveRunner()
    student = ReplayStudent()
    report = runner.run(student, [PLANS[0]["id"]], 2, random.Random(3))
    cats = report["categories"]
    assert "irrelevance" in cats
    irr = cats["irrelevance"]
    assert irr["scored"] is False
    assert irr["n"] == 0
    assert irr["name_sequence_match"] is None
    assert irr["dependency_arg_match"] is None
    assert irr["grounding_ok"] is None
    assert irr["format_ok"] is None
    simple = cats["simple"]
    assert simple["scored"] is False
    assert simple["name_sequence_match"] is None
    assert "multiple" in cats
    assert "parallel" in cats
    assert "multi_turn" in cats
    assert cats["multi_turn"]["scored"] is True


def test_simple_category_unscored_even_when_replay_is_perfect():
    runner = LiveRunner()
    rng = random.Random(9)
    # crm-unknown-stop is the 1-tool plan
    idx = next(i for i, p in enumerate(PLANS) if p["id"] == "crm-unknown-stop")
    traj = build_chain(rng, plan_index=idx)
    assert len(traj["steps"]) == 1
    student = ReplayStudent(traj)
    scored = runner.score_rollout(traj, runner.run_one(student, traj["prompt"]))
    assert scored["category"] == "simple"
    assert scored["name_sequence_match"] == 1.0
    report = aggregate_report([scored], holdout_plan_ids=[traj["plan_id"]], n=1)
    assert report["categories"]["simple"]["scored"] is False
    assert report["categories"]["simple"]["name_sequence_match"] is None
    assert report["categories"]["irrelevance"]["scored"] is False


def test_parallel_maps_fanout_plans():
    runner = LiveRunner()
    rng = random.Random(11)
    idx = next(i for i, p in enumerate(PLANS) if p["id"] == "two-user-notify")
    traj = build_chain(rng, plan_index=idx)
    scored = runner.score_rollout(traj, runner.run_one(ReplayStudent(traj), traj["prompt"]))
    assert "fanout" in traj["skills"]
    assert scored["category"] == "parallel"
    assert scored["name_sequence_match"] == 1.0
