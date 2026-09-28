from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(REPO / "src")


def _help(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=ENV,
    )


def _console_or_src(name: str, src_rel: str, extra: list[str] | None = None) -> list[str]:
    cmd = shutil.which(name)
    if cmd:
        return [cmd, *(extra or ["--help"])]
    return [PY, str(REPO / src_rel), *(extra or ["--help"])]


def test_distill_help_flags():
    r = _help(_console_or_src("distill", "src/distill_tools.py"))
    assert r.returncode == 0, r.stderr
    text = r.stdout
    flags = [
        "--count",
        "--pilot",
        "--seed",
        "--roster",
        "--providers",
        "--models",
        "--wipe",
        "--out-dir",
        "--legacy-v1",
        "--no-eval-card",
        "--no-trace-eval",
        "--curriculum",
        "--holdout-frac",
        "--verify-sample",
        "--no-verify",
        "--cross-teacher",
        "--cross-teacher-rate",
        "--dpo",
        "--dpo-rate",
        "--mp",
        "--shard",
    ]
    for flag in flags:
        assert flag in text, f"missing {flag} in distill --help"
    for phrase in (
        "Teacher writes prose only",
        "seed + i * 7919",
        "key_env",
        "linear interpolates",
        "0 disables holdout",
        "0-indexed",
        "checked before teacher HTTP",
    ):
        assert phrase in text, f"missing help phrase: {phrase}"


def test_eval_card_help_flags():
    r = _help(_console_or_src("eval_card", "src/eval_card.py"))
    assert r.returncode == 0, r.stderr
    text = r.stdout
    for flag in ("--input", "--out", "--require-gates"):
        assert flag in text
    assert "malformed_tool_call_rate" in text


def test_export_sft_help_flags():
    r = _help(_console_or_src("export_sft", "src/export_sft.py"))
    assert r.returncode == 0, r.stderr
    text = r.stdout
    for flag in ("--format", "--check-mask", "--check-tokenizer"):
        assert flag in text
    assert "trainable_roles" in text


def test_dpo_pairs_help_flags():
    r = _help(_console_or_src("dpo_pairs", "src/dpo_pairs.py"))
    assert r.returncode == 0, r.stderr
    assert "--dpo-rate" in r.stdout
    assert "--dpo-rewrite-prose" not in r.stdout


def test_eval_live_help_flags():
    r = _help(_console_or_src("eval_live", "src/eval_live.py"))
    assert r.returncode == 0, r.stderr
    text = r.stdout
    for flag in ("--replay", "--max-turns", "--key-env", "--seed"):
        assert flag in text


def test_dataset_publish_help_flags():
    r = _help(_console_or_src("dataset_publish", "src/dataset_publish.py"))
    assert r.returncode == 0, r.stderr
    text = r.stdout
    for flag in ("--input", "--out", "--check"):
        assert flag in text


def test_archive_data_help_flags():
    r = _help(_console_or_src("archive_data", "src/archive_data.py"))
    assert r.returncode == 0, r.stderr
    assert "--label" in r.stdout


def test_forge_status_help_flags():
    r = _help(_console_or_src("forge-status", "src/status.py"))
    assert r.returncode == 0, r.stderr
    text = r.stdout + r.stderr
    assert "--watch" in text
    assert "--raw-dir" in text
    assert "TOTAL:" not in text
    assert "usage:" in text.lower() or "Count traces" in text
