#!/usr/bin/env python3
"""status.py - check distillation progress at any time.

Usage: forge-status [--watch] [--raw-dir DIR]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path


def report(raw_dir: Path) -> None:
    print("=" * 60)
    print(f"{time.strftime('%H:%M:%S')}  distillation status")
    total = 0
    for p in sorted(raw_dir.glob("traces_*.jsonl")):
        n = sum(1 for _ in p.open())
        total += n
        print(f"  {p.name.replace('traces_','').replace('.jsonl',''):12s} {n:5d} traces")
    print(f"  TOTAL: {total}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Count traces_*.jsonl lines under data/raw")
    ap.add_argument("--watch", action="store_true", help="reprint every 10s")
    ap.add_argument("--raw-dir", type=str, default="data/raw",
                    help="directory of traces_*.jsonl (default: data/raw)")
    args = ap.parse_args(argv)
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"raw dir not found: {args.raw_dir}", file=sys.stderr)
        return 2
    while True:
        report(raw_dir)
        if not args.watch:
            break
        time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
