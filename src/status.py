#!/usr/bin/env python3
"""status.py - check distillation progress at any time.

Usage: python src/status.py [--watch]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEEDS = ROOT / "data" / "seeds"


def main(watch: bool = False):
    while True:
        print("=" * 60)
        print(f"{time.strftime('%H:%M:%S')}  distillation status")
        total = 0
        for p in sorted(RAW.glob("traces_*.jsonl")):
            n = sum(1 for _ in p.open())
            total += n
            print(f"  {p.name.replace('traces_','').replace('.jsonl',''):12s} {n:5d} traces")
        ext = SEEDS / "external.jsonl"
        if ext.exists():
            n_ext = sum(1 for _ in ext.open())
            print(f"  external-seeds {n_ext:5d} prompts")
        print(f"  TOTAL: {total}")
        if not watch:
            break
        time.sleep(10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Count traces_*.jsonl lines under data/raw")
    ap.add_argument("--watch", action="store_true", help="reprint every 10s")
    args = ap.parse_args()
    main(args.watch)