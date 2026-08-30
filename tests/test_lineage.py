from __future__ import annotations

import json
from pathlib import Path

import yaml

from distill_tools import Distiller
from lineage import (
    LINEAGE_SPEC,
    kept_record,
    lineage_id,
    reject_record,
    summarize,
)
from prose_writer import DISTILL_VERSION

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "configs" / "roster.example.yaml"


def test_lineage_id_stable_and_changes_with_teacher():
    a = lineage_id("th", "p/m1", "reversed-v2", "0.2")
    b = lineage_id("th", "p/m1", "reversed-v2", "0.2")
    c = lineage_id("th", "p/m2", "reversed-v2", "0.2")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_kept_record_never_contains_messages_or_sk():
    trace = {
        "traj_hash": "abc",
        "teacher": "prov/model",
        "distill_version": DISTILL_VERSION,
        "forge_spec": "0.2",
        "plan_id": "plan_x",
        "plan_tier": "easy",
        "plan_skills": ["search"],
        "teacher_mode": "concise",
        "messages": [{"role": "user", "content": "sk-secret"}],
        "chain_steps": [{"tool": "get_user"}],
        "vars": {"uid": 1},
        "eval": {"format_ok": True, "repaired": False},
        "dpo_pair_id": None,
    }
    rec = kept_record(trace, tokens_in=3, tokens_out=7, curriculum_mode="uniform", seed=42)
    blob = json.dumps(rec)
    assert "messages" not in rec
    assert "chain_steps" not in rec
    assert "vars" not in rec
    assert "sk-" not in blob
    assert rec["kept"] is True
    assert rec["lineage_spec"] == LINEAGE_SPEC
    assert rec["tokens_in"] == 3
    assert rec["tokens_out"] == 7
    assert rec["lineage_id"] == lineage_id("abc", "prov/model", DISTILL_VERSION, "0.2")


def test_append_lineage_matches_trace_id(tmp_path: Path):
    roster = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    d = Distiller(roster, tmp_path / "raw", holdout_frac=0)
    lid = lineage_id("hash1", "example-provider/m", DISTILL_VERSION, "0.2")
    trace = {
        "traj_hash": "hash1",
        "teacher": "example-provider/m",
        "distill_version": DISTILL_VERSION,
        "forge_spec": "0.2",
        "plan_id": "plan_x",
        "plan_tier": "easy",
        "plan_skills": [],
        "eval": {"format_ok": True},
        "lineage_id": lid,
    }
    rec = kept_record(trace, tokens_in=0, tokens_out=0, curriculum_mode="off", seed=42)
    d._append_lineage("example-provider", rec)
    path = tmp_path / "raw" / "lineage_example-provider.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["lineage_id"] == trace["lineage_id"]
    assert row["lineage_id"] == lid
    assert "messages" not in row


def test_reject_record_distinct_attempts():
    a = reject_record(
        prov="p", model="m", plan_id="x", traj_hash="h",
        reason="http:429", http=429, error="rate", attempt_seq=1,
    )
    b = reject_record(
        prov="p", model="m", plan_id="x", traj_hash="h",
        reason="http:429", http=429, error="rate", attempt_seq=2,
    )
    assert a["lineage_id"] != b["lineage_id"]
    assert a["attempt_seq"] == 1
    assert b["attempt_seq"] == 2
    assert a["kept"] is False
    assert "ts" in a
    assert "messages" not in a


def test_compute_card_includes_observability():
    from eval_card import compute_card, load_traces

    mini = REPO / "tests" / "fixtures" / "mini_traces.jsonl"
    traces = load_traces([mini])
    card = compute_card(traces, extra={"input_paths": [str(mini)]})
    assert "observability" in card
    obs = card["observability"]
    assert obs["n_rejects_files"] == 0
    assert obs["lineage_spec"] == "1.0"
    assert obs["n_lineage_ids"] == 0
    assert obs["n_kept"] == len(traces)


def test_summarize_counts_reasons():
    kept = [{"lineage_id": "a"}, {"lineage_id": "b"}]
    rejected = [
        {"reason": "format"},
        {"reason": "format"},
        {"reason": "http:429"},
    ]
    s = summarize(kept, rejected, {"input": 10, "output": 4})
    assert s["n_kept"] == 2
    assert s["n_rejects_files"] == 3
    assert s["reject_reasons"]["format"] == 2
    assert s["reject_reasons"]["http:429"] == 1
    assert s["tokens_per_kept_trace"]["input"] == 5
    assert s["n_lineage_ids"] == 2
