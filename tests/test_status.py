from __future__ import annotations

from pathlib import Path

import status


def test_status_counts_given_raw_dir(tmp_path: Path, capsys):
    (tmp_path / "traces_a.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    rc = status.main(["--raw-dir", str(tmp_path)])
    assert rc == 0
    assert "TOTAL: 2" in capsys.readouterr().out


def test_status_missing_raw_dir_exits_2(tmp_path: Path, capsys):
    missing = tmp_path / "nope"
    rc = status.main(["--raw-dir", str(missing)])
    assert rc == 2
    err = capsys.readouterr().err
    assert f"raw dir not found: {missing}" in err
