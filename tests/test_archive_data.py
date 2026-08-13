from __future__ import annotations

from pathlib import Path

import archive_data


def test_root_is_relative_to_package():
    expected = Path(archive_data.__file__).resolve().parent.parent
    assert archive_data.ROOT == expected
    assert archive_data.ROOT.name  # not the old hardcoded absolute-only layout


def test_move_not_copy_and_dest_exists(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    traces = raw / "traces_example-provider.jsonl"
    traces.write_text('{"prompt": "x"}\n', encoding="utf-8")
    ckpt = raw / "checkpoint_example-provider.json"
    ckpt.write_text("1", encoding="utf-8")
    (raw / "eval_card.json").write_text("{}", encoding="utf-8")
    (raw / "holdout_plan_ids.json").write_text("[]", encoding="utf-8")
    dpo = raw / "dpo_pairs_x.jsonl"
    dpo.write_text('{"pair_id": "p1"}\n', encoding="utf-8")
    token_sidecar = raw / ".token_usage_0.json"
    token_sidecar.write_text('{"input": 1, "output": 2}', encoding="utf-8")

    dest = archive_data.archive_raw(
        raw_dir=raw,
        archive_dir=tmp_path / "data" / "archive",
        label="unit",
    )
    assert dest.exists()
    assert dest.is_dir()
    assert "unit" in dest.name
    assert not traces.exists()
    assert not ckpt.exists()
    assert (dest / "traces_example-provider.jsonl").is_file()
    assert (dest / "checkpoint_example-provider.json").is_file()
    assert (dest / "eval_card.json").is_file()
    assert (dest / "holdout_plan_ids.json").is_file()
    assert (dest / "dpo_pairs_x.jsonl").is_file()
    assert (dest / ".token_usage_0.json").is_file()
    assert not dpo.exists()
    assert not token_sidecar.exists()
    # Source files were moved, not copied.
    assert traces.read_text() if traces.exists() else True
    assert list(raw.glob("traces_*.jsonl")) == []


def test_nothing_to_archive_still_returns_dest(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    dest = archive_data.archive_raw(raw_dir=raw, archive_dir=tmp_path / "data" / "archive")
    assert dest.exists()


def test_default_archive_dir_sibling():
    raw = Path("/tmp/fake/data/raw")
    assert archive_data.default_archive_dir(raw) == Path("/tmp/fake/data/archive")
