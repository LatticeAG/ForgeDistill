from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from distill_tools import (
    Distiller,
    merge_token_usage,
    parse_mp,
    parse_shard,
    run_mp,
    shard_keeps,
    split_counts,
)

ROSTER = {
    "p": {
        "base_url": "http://127.0.0.1:9/v1",
        "key_env": "",
        "concurrency": 1,
        "models": {"m": {"mode": "concise", "max_tokens": 8, "weight": 1}},
    }
}


def test_locked_append_two_threads(tmp_path: Path):
    d = Distiller(ROSTER, tmp_path)
    path = tmp_path / "traces_p.jsonl"

    def worker():
        for i in range(100):
            d._locked_append(path, json.dumps({"i": i}))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 200
    for ln in lines:
        json.loads(ln)


def test_shard_keeps_partitions_known_hashes():
    hashes = [f"{i:08x}deadbeef" for i in range(32)]
    a = [h for h in hashes if shard_keeps(h, 0, 2)]
    b = [h for h in hashes if shard_keeps(h, 1, 2)]
    assert set(a) & set(b) == set()
    assert set(a) | set(b) == set(hashes)
    assert shard_keeps("abcdefgh", 0, 1) is True


def test_merge_token_usage_sums_by_route(tmp_path: Path):
    (tmp_path / ".token_usage_0.json").write_text(
        json.dumps({
            "input": 10,
            "output": 2,
            "by_route": {"a/m": {"input": 10, "output": 2}},
        }),
        encoding="utf-8",
    )
    (tmp_path / ".token_usage_1.json").write_text(
        json.dumps({
            "input": 6,
            "output": 4,
            "by_route": {
                "a/m": {"input": 5, "output": 3},
                "b/n": {"input": 1, "output": 1},
            },
        }),
        encoding="utf-8",
    )
    merged = merge_token_usage(tmp_path)
    assert merged["input"] == 16
    assert merged["output"] == 6
    assert merged["by_route"]["a/m"] == {"input": 15, "output": 5}
    assert merged["by_route"]["b/n"] == {"input": 1, "output": 1}


def test_split_counts_and_parse_shard():
    assert split_counts(21, 2) == [11, 10]
    assert split_counts(20, 2) == [10, 10]
    assert parse_shard("1/4") == (1, 4)
    with pytest.raises(SystemExit) as exc:
        parse_shard("4/4")
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        parse_shard("1")
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        parse_mp(0)
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        parse_mp(-1)
    assert exc.value.code == 2
    assert parse_mp(1) == 1


def test_merge_token_usage_empty_dir_is_zeros(tmp_path: Path):
    merged = merge_token_usage(tmp_path)
    assert merged == {"input": 0, "output": 0, "by_route": {}}


def _run_mp_kwargs(out_dir: Path, **overrides) -> dict:
    kw = dict(
        roster_path="/dev/null",
        out_dir=out_dir,
        count=0,
        nproc=2,
        seed=1,
        holdout_frac=0.0,
        provider_filter=None,
        model_filter=None,
        curriculum_mode="uniform",
        write_eval_card=False,
        stamp_trace_eval=True,
        verify_sample=0.0,
        no_verify=True,
        cross_teacher=False,
        cross_teacher_rate=0.3,
        dpo_enabled=False,
        dpo_rate=1.0,
        pilot=False,
    )
    kw.update(overrides)
    return kw


def test_run_mp_rejects_holdout_file(tmp_path: Path, capsys):
    (tmp_path / "holdout_plan_ids.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_mp(**_run_mp_kwargs(tmp_path))
    assert exc.value.code == 1
    assert "archive or wipe before --mp" in capsys.readouterr().out


def test_run_mp_rejects_traces_jsonl(tmp_path: Path, capsys):
    (tmp_path / "traces_p.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_mp(**_run_mp_kwargs(tmp_path))
    assert exc.value.code == 1
    assert "archive or wipe before --mp" in capsys.readouterr().out


def test_run_mp_rejects_eval_card(tmp_path: Path, capsys):
    (tmp_path / "eval_card.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_mp(**_run_mp_kwargs(tmp_path))
    assert exc.value.code == 1
    assert "archive or wipe before --mp" in capsys.readouterr().out


def test_mp_fcntl_none_exits_2(tmp_path: Path, monkeypatch, capsys):
    import sys

    import distill_tools as dt

    example = Path(__file__).resolve().parent.parent / "configs" / "roster.example.yaml"
    monkeypatch.setattr(dt, "fcntl", None)
    monkeypatch.setattr(sys, "argv", [
        "distill",
        "--mp", "2",
        "--count", "0",
        "--out-dir", str(tmp_path),
        "--roster", str(example),
        "--holdout-frac", "0",
    ])
    with pytest.raises(SystemExit) as exc:
        dt.main()
    assert exc.value.code == 2
    assert "fcntl missing" in capsys.readouterr().out


def test_second_distiller_loads_holdout_file(tmp_path: Path, monkeypatch):
    import distill_tools

    calls = {"n": 0}
    real = distill_tools.split_plan_ids

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(distill_tools, "split_plan_ids", wrapped)
    d1 = Distiller(ROSTER, tmp_path, seed=1, holdout_frac=0.15)
    assert calls["n"] == 1
    assert d1.holdout_ids
    d2 = Distiller(ROSTER, tmp_path, seed=999, holdout_frac=0.15)
    assert calls["n"] == 1
    assert d2.holdout_ids == d1.holdout_ids
    d3 = Distiller(ROSTER, tmp_path, seed=0, holdout_frac=0)
    assert calls["n"] == 1
    assert d3.holdout_ids == d1.holdout_ids
