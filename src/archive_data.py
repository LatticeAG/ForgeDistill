#!/usr/bin/env python3
"""archive_data.py - move current raw traces to data/archive/ with a
timestamped folder. NEVER deletes anything. Run before any fresh
distillation run that would overwrite/clear data/raw.

Usage: python src/archive_data.py [--label agentic-v3]
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/ubuntu/nanbeige-agentic")
RAW = ROOT / "data" / "raw"
ARCHIVE = ROOT / "data" / "archive"

label = ""
if "--label" in sys.argv:
    label = sys.argv[sys.argv.index("--label") + 1].replace("/", "-").replace(" ", "_")

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
dest = ARCHIVE / f"{ts}{'_' + label if label else ''}"
dest.mkdir(parents=True, exist_ok=True)

moved = 0
for f in sorted(RAW.glob("traces_*.jsonl")) + sorted(RAW.glob("checkpoint_*.json")):
    shutil.move(str(f), str(dest / f.name))
    moved += 1

if moved:
    print(f"[archive] moved {moved} files -> {dest}")
    # quick summary of what was archived
    n = sum(1 for f in dest.glob("traces_*.jsonl") for _ in f.open(encoding="utf-8") if _.strip())
    print(f"[archive] {n} traces archived")
else:
    print("[archive] nothing to archive (data/raw is empty)")
