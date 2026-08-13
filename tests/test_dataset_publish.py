"""test_dataset_publish.py - public bundle must never leak internal fields."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dataset_publish as dp


def _trace(teacher: str = "lexgf/crow-grok-4.3") -> dict:
    return {
        "seed_class": "agentic",
        "prompt": "Email user 42",
        "teacher": teacher,
        "teacher_mode": "thinking",
        "plan_template": 44,
        "plan_id": "plan_user-email-welcome",
        "plan_tier": "easy",
        "plan_skills": ["multi_hop"],
        "chain_steps": [{"tool": "get_user", "args": {"user_id": 42}}],
        "vars": {"uid": 42},
        "distill_version": "reversed-v2",
        "forge_spec": "0.2",
        "traj_hash": "abc123",
        "eval": {"format_ok": True, "n_rounds": 2},
        "messages": [{"role": "user", "content": "hi"}],
    }


def _pair() -> dict:
    t = _trace()
    return {
        "pair_id": "p1",
        "forge_spec": "0.2",
        "distill_version": "reversed-v2",
        "mutation": "fabricated_final",
        "chosen": t,
        "rejected": {**t, "traj_hash": "def456"},
        "gates": {"chosen_grounding": True, "rejected_grounding": False},
    }


def test_scrub_trace_drops_teacher_and_internals():
    out = dp.scrub_trace(_trace())
    assert "teacher" not in out
    assert "teacher_mode" not in out
    assert "plan_template" not in out
    assert "chain_steps" not in out
    assert "vars" not in out
    assert "seed_class" not in out
    assert "distill_version" not in out
    assert "forge_spec" not in out
    assert "plan_skills" not in out
    # public fields survive
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["plan_id"] == "plan_user-email-welcome"
    assert out["plan_tier"] == "easy"
    assert out["traj_hash"] == "abc123"
    assert out["eval"]["n_rounds"] == 2


def test_scrub_pair_drops_teacher_from_both_sides():
    out = dp.scrub_pair(_pair())
    assert "teacher" not in json.dumps(out)
    assert "teacher_mode" not in json.dumps(out)
    assert "chain_steps" not in json.dumps(out)
    assert out["pair_id"] == "p1"
    assert out["mutation"] == "fabricated_final"
    assert out["gates"]["rejected_grounding"] is False
    assert out["chosen"]["traj_hash"] == "abc123"
    assert out["rejected"]["traj_hash"] == "def456"


def test_assert_clean_raises_on_leak():
    import pytest

    leaked = [dp.scrub_trace(_trace()), _trace()]  # second one still has teacher
    with pytest.raises(AssertionError):
        dp.assert_clean(leaked, "sft")


def test_publish_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traces_prov.jsonl").write_text(
        json.dumps(_trace()) + "\n" + json.dumps(_trace("lexzm/deepseek/deepseek-v4-pro")) + "\n",
        encoding="utf-8",
    )
    (raw / "dpo_pairs_prov.jsonl").write_text(
        json.dumps(_pair()) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "bundle"
    dp.publish(raw, out)
    sft = [json.loads(l) for l in (out / "sft" / "train.jsonl").read_text().splitlines()]
    dpo = [json.loads(l) for l in (out / "dpo" / "train.jsonl").read_text().splitlines()]
    assert len(sft) == 2
    assert len(dpo) == 1
    dp.assert_clean(sft, "sft")
    dp.assert_clean(dpo, "dpo")
    assert "teacher" not in (out / "sft" / "train.jsonl").read_text()
    assert "teacher" not in (out / "dpo" / "train.jsonl").read_text()
