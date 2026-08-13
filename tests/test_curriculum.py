from __future__ import annotations

import json
import random

import pytest

from agentic_plans import PLANS
from curriculum import (
    TIERS,
    mix_for_progress,
    pick_plan_index,
    plans_by_tier,
    split_plan_ids,
)


def test_linear_mix_endpoints_and_sum():
    m0 = mix_for_progress(0.0, "linear")
    assert abs(m0["easy"] - 0.50) < 1e-6
    assert abs(m0["medium"] - 0.30) < 1e-6
    assert abs(m0["hard"] - 0.15) < 1e-6
    assert abs(m0["expert"] - 0.05) < 1e-6
    assert abs(sum(m0.values()) - 1.0) < 1e-6
    m1 = mix_for_progress(1.0, "linear")
    assert abs(m1["easy"] - 0.10) < 1e-6
    assert abs(m1["medium"] - 0.20) < 1e-6
    assert abs(m1["hard"] - 0.40) < 1e-6
    assert abs(m1["expert"] - 0.30) < 1e-6
    assert abs(sum(m1.values()) - 1.0) < 1e-6
    mid = mix_for_progress(0.5, "linear")
    assert abs(sum(mid.values()) - 1.0) < 1e-6


def test_uniform_and_off():
    u = mix_for_progress(0.3, "uniform")
    assert u == {t: 0.25 for t in TIERS}
    assert abs(sum(u.values()) - 1.0) < 1e-6
    assert mix_for_progress(0.5, "off") is None


def test_plans_by_tier_covers_all():
    buckets = plans_by_tier(PLANS)
    seen = []
    for t in TIERS:
        seen.extend(buckets[t])
    assert sorted(seen) == list(range(len(PLANS)))


def test_holdout_disjoint_and_frac_zero():
    train, hold = split_plan_ids(PLANS, 0.15, seed=42)
    assert set(train).isdisjoint(set(hold))
    assert set(train) | set(hold) == {p["id"] for p in PLANS}
    assert len(hold) == round(0.15 * len(PLANS))
    train0, hold0 = split_plan_ids(PLANS, 0, seed=1)
    assert hold0 == []
    assert len(train0) == len(PLANS)
    train_neg, hold_neg = split_plan_ids(PLANS, -0.2, seed=2)
    assert hold_neg == []
    assert len(train_neg) == len(PLANS)


def test_excluded_id_never_returned():
    exclude = {PLANS[0]["id"]}
    rng = random.Random(0)
    for _ in range(200):
        i = pick_plan_index(rng, PLANS, "uniform", 0.5, exclude)
        assert PLANS[i]["id"] not in exclude
        assert 0 <= i < len(PLANS)


def test_pick_plan_space_exhausted():
    rng = random.Random(0)
    all_ids = {p["id"] for p in PLANS}
    with pytest.raises(RuntimeError, match="plan space exhausted"):
        pick_plan_index(rng, PLANS, "uniform", 0.0, all_ids)


def test_pick_off_mode_still_returns_index():
    rng = random.Random(7)
    i = pick_plan_index(rng, PLANS, "off", 0.0, set())
    assert 0 <= i < len(PLANS)


def test_distiller_writes_holdout_file(tmp_path):
    from distill_tools import Distiller

    roster = {
        "p": {
            "base_url": "http://127.0.0.1:9/v1",
            "key_env": "",
            "models": {"m": {"mode": "concise", "max_tokens": 8, "weight": 1}},
        }
    }
    d = Distiller(roster, tmp_path, seed=42, holdout_frac=0.15)
    hold_path = tmp_path / "holdout_plan_ids.json"
    assert hold_path.is_file()
    ids = json.loads(hold_path.read_text(encoding="utf-8"))
    assert isinstance(ids, list) and ids
    assert set(ids) == d.holdout_ids
    assert set(ids).issubset({p["id"] for p in PLANS})
    d0 = Distiller(roster, tmp_path / "off", seed=1, holdout_frac=0)
    assert d0.holdout_ids == set()
