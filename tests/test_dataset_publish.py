"""test_dataset_publish.py - public bundle must never leak internal fields."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dataset_publish as dp


def _trace(teacher: str = "example-provider/example-model-thinking") -> dict:
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


def test_assert_clean_internal_key_re_guards_allowlist(monkeypatch):
    import pytest

    rec = dp.scrub_trace(_trace())
    rec["teacher_mode"] = "thinking"
    monkeypatch.setattr(dp, "PUBLIC_TRACE_KEYS", dp.PUBLIC_TRACE_KEYS + ("teacher_mode",))
    with pytest.raises(AssertionError, match="INTERNAL_KEY_RE"):
        dp.assert_clean([rec], "sft")


def test_assert_clean_current_sft_does_not_trip_internal_re():
    dp.assert_clean([dp.scrub_trace(_trace())], "sft")


def test_assert_clean_raises_on_leak():
    import pytest

    leaked = [dp.scrub_trace(_trace()), _trace()]  # second one still has teacher
    with pytest.raises(AssertionError):
        dp.assert_clean(leaked, "sft")


def test_publish_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traces_prov.jsonl").write_text(
        json.dumps(_trace()) + "\n" + json.dumps(_trace("local-router/local-model-a")) + "\n",
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
    assert (out / "lineage" / "train.jsonl").is_file()
    assert (out / "lineage" / "train.jsonl").read_text(encoding="utf-8") == ""


def test_publish_fixture_traces_have_no_teacher_substring(tmp_path):
    src = Path(__file__).resolve().parent / "fixtures"
    out = tmp_path / "bundle"
    dp.publish(src, out)
    sft = (out / "sft" / "train.jsonl").read_text(encoding="utf-8")
    lin = (out / "lineage" / "train.jsonl").read_text(encoding="utf-8")
    assert "teacher" not in sft
    assert "teacher" not in lin
    assert sft.strip()


def test_publish_scrubs_lineage_teacher(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traces_prov.jsonl").write_text(json.dumps(_trace()) + "\n", encoding="utf-8")
    row = {
        "lineage_spec": "1.0",
        "lineage_id": "lid1",
        "traj_hash": "abc123",
        "plan_id": "plan_user-email-welcome",
        "plan_tier": "easy",
        "plan_skills": ["multi_hop"],
        "teacher": "example-provider/example-model-thinking",
        "teacher_thoughts": "example-provider/example-model-thinking",
        "teacher_answer": "example-provider/example-model-thinking",
        "teacher_mode": "thinking",
        "tokens_in": 11,
        "tokens_out": 5,
        "eval": {"format_ok": True},
        "kept": True,
    }
    (raw / "lineage_prov.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "bundle"
    summary = dp.publish(raw, out)
    assert summary["lineage"] == 1
    text = (out / "lineage" / "train.jsonl").read_text(encoding="utf-8")
    assert "teacher" not in text
    rec = json.loads(text.splitlines()[0])
    assert rec["lineage_id"] == "lid1"
    assert rec["traj_hash"] == "abc123"
    assert rec["kept"] is True
    assert "teacher_mode" not in rec
    dp.assert_clean([rec], "lineage")
