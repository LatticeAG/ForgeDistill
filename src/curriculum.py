"""Curriculum sampling: tier mix, holdout split, plan picking."""
from __future__ import annotations
import random

TIERS = ("easy", "medium", "hard", "expert")

_LINEAR_START = {"easy": 0.50, "medium": 0.30, "hard": 0.15, "expert": 0.05}
_LINEAR_END = {"easy": 0.10, "medium": 0.20, "hard": 0.40, "expert": 0.30}


def plans_by_tier(plans) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {t: [] for t in TIERS}
    for i, p in enumerate(plans):
        t = p.get("tier")
        if t in buckets:
            buckets[t].append(i)
        elif t:
            buckets.setdefault(t, []).append(i)
    return buckets


def mix_for_progress(progress: float, mode: str) -> dict[str, float] | None:
    """Return per-tier sampling weights, or None for mode 'off'."""
    if mode == "off":
        return None
    if mode == "uniform":
        return {t: 0.25 for t in TIERS}
    if mode == "linear":
        p = float(progress)
        return {
            t: _LINEAR_START[t] + p * (_LINEAR_END[t] - _LINEAR_START[t])
            for t in TIERS
        }
    raise ValueError(f"unknown curriculum mode: {mode}")


def pick_plan_index(rng, plans, mode, progress, exclude_ids) -> int:
    """Sample a plan index under a curriculum mix, honoring exclude_ids.

    Resample up to 40 times then raise RuntimeError("plan space exhausted").
    """
    exclude_ids = exclude_ids or set()
    mix = mix_for_progress(progress, mode)
    buckets = plans_by_tier(plans)
    for _ in range(40):
        if mix is None:
            candidates = [i for i, p in enumerate(plans) if p.get("id") not in exclude_ids]
        else:
            weights = [mix[t] for t in TIERS]
            tier = rng.choices(list(TIERS), weights=weights, k=1)[0]
            candidates = [
                i for i in buckets.get(tier, [])
                if plans[i].get("id") not in exclude_ids
            ]
        if candidates:
            return rng.choice(candidates)
    raise RuntimeError("plan space exhausted")


def split_plan_ids(plans, frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle with Random(seed); last round(frac*n) ids are holdout.

    Disjoint. frac<=0 => holdout empty, train is all ids.
    """
    ids = [p["id"] for p in plans]
    n = len(ids)
    if frac <= 0 or n == 0:
        return list(ids), []
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_hold = round(frac * n)
    if n_hold <= 0:
        return shuffled, []
    holdout_ids = shuffled[-n_hold:]
    train_ids = shuffled[:-n_hold]
    return train_ids, holdout_ids
