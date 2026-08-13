#!/usr/bin/env python3
"""archive_data.py - move current raw traces to data/archive/ with a
timestamped folder. NEVER deletes anything. Run before any fresh
distillation run that would overwrite/clear data/raw.

Usage: python src/archive_data.py [--label agentic-v3]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ARCHIVE = ROOT / "data" / "archive"

# Extra artifacts that must leave a fresh raw dir (never ingest as traces).
_EXTRA_NAMES = ("eval_card.json", "holdout_plan_ids.json")


def archive_dest(archive_dir: Path, label: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{label}" if label else ""
    return archive_dir / f"{ts}{suffix}"


def files_to_archive(raw_dir: Path) -> list[Path]:
    """Traces, checkpoints, eval card, holdout ids. Never returns directories."""
    out: list[Path] = []
    if not raw_dir.is_dir():
        return out
    out.extend(sorted(raw_dir.glob("traces_*.jsonl")))
    out.extend(sorted(raw_dir.glob("checkpoint_*.json")))
    out.extend(sorted(raw_dir.glob("dpo_pairs_*.jsonl")))
    for name in _EXTRA_NAMES:
        p = raw_dir / name
        if p.is_file():
            out.append(p)
    return out


def archive_raw(
    raw_dir: Path | None = None,
    archive_dir: Path | None = None,
    label: str = "",
) -> Path:
    """Move archiveable files from raw_dir into a new timestamped folder.

    Never unlinks. If a move fails, the exception propagates so callers
    can exit 1 without deleting leftovers.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW
    archive_dir = Path(archive_dir) if archive_dir is not None else ARCHIVE
    dest = archive_dest(archive_dir, label=label)
    dest.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in files_to_archive(raw_dir):
        shutil.move(str(f), str(dest / f.name))
        moved += 1

    if moved:
        n = sum(
            1
            for f in dest.glob("traces_*.jsonl")
            for line in f.open(encoding="utf-8")
            if line.strip()
        )
        print(f"[archive] moved {moved} files -> {dest}")
        print(f"[archive] {n} traces archived")
    else:
        print("[archive] nothing to archive (raw dir is empty)")
    return dest


def default_archive_dir(out_dir: Path) -> Path:
    """Sibling archive folder: .../data/raw -> .../data/archive."""
    out_dir = Path(out_dir)
    if out_dir.name == "raw":
        return out_dir.parent / "archive"
    return out_dir.parent / "archive"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Archive data/raw into data/archive. Never deletes.")
    ap.add_argument("--label", default="", help="optional folder suffix")
    ap.add_argument("--raw-dir", default="", help="override raw directory")
    ap.add_argument("--archive-dir", default="", help="override archive parent directory")
    args = ap.parse_args(argv)
    label = (args.label or "").replace("/", "-").replace(" ", "_")
    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW
    archive_dir = Path(args.archive_dir) if args.archive_dir else ARCHIVE
    archive_raw(raw_dir=raw_dir, archive_dir=archive_dir, label=label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
