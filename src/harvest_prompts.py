#!/usr/bin/env python3
"""Harvest real user prompts from open tool-calling datasets.

Uses the HF datasets-server API (no datasets lib needed). Extracts
user-turn prompts, filters to tool-callable / actionable queries,
dedupes, and writes to data/seeds/external.jsonl for the distiller.
"""
from __future__ import annotations
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "seeds" / "external.jsonl"

SOURCES = [
    # (dataset, config, split, max_rows_to_scan)
    ("glaiveai/glaive-function-calling-v2", "default", "train", 6000),
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", 2000),
    ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", 2000),
    ("NousResearch/hermes-function-calling-v1", "json_mode_agentic", "train", 1000),
]

URL = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"

BAD_KEYWORDS = [
    "image", "picture", "photo", "video", "audio", "voice", "speech", "song",
    "draw", "paint", "generate an image", "jailbreak", "bomb", "explosive",
    "illegal drug", "credit card numbers", "ssn", "password of",
]


def fetch_rows(ds, cfg, split, offset, length=100):
    url = URL.format(ds=ds, cfg=cfg, split=split, off=offset, length=length)
    req = urllib.request.Request(url, headers={"User-Agent": "latticeag-forge-distill/0.4"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["rows"]


def extract_user_prompts(row):
    """Tolerant extraction: try conversations/dialogue/turns arrays, JSON-string
    chat columns, else any string cols."""
    prompts = []
    r = row.get("row", row)
    # Glaive style: 'chat' is a JSON string of [{role, content}]
    for key in ("chat", "conversation"):
        v = r.get(key)
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                arr = json.loads(v)
                for m in arr:
                    if isinstance(m, dict) and m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, str) and c.strip():
                            prompts.append(c.strip())
                if prompts:
                    return prompts
            except Exception:
                pass
    for key in ("conversations", "dialogue", "turns", "messages"):
        v = r.get(key)
        if isinstance(v, list):
            for m in v:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        prompts.append(c.strip())
            if prompts:
                return prompts
    # xlam style: "query" column
    for key in ("query", "prompt", "instruction", "input", "question"):
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            prompts.append(v.strip())
            return prompts
    return prompts


def is_good_prompt(p: str) -> bool:
    if len(p) < 8 or len(p) > 500:
        return False
    low = p.lower()
    if any(b in low for b in BAD_KEYWORDS):
        return False
    # Must look actionable (verb-ish) - reject pure chit-chat
    starters = ("what", "how", "why", "when", "where", "can you", "could you", "please", "find", "check", "get", "fetch", "look", "show", "tell", "send", "create", "list", "retrieve", "query", "search", "calculate", "convert", "compare", "is there", "does", "do you", "i need", "give me", "run")
    return low.startswith(starters)


def main():
    seen = set()
    out = []
    for ds, cfg, split, max_scan in SOURCES:
        scanned = 0
        offset = 0
        print(f"[harvest] {ds}", file=sys.stderr)
        while scanned < max_scan:
            try:
                rows = fetch_rows(ds, cfg, split, offset)
            except Exception as e:
                print(f"  stop at offset {offset}: {e}", file=sys.stderr)
                break
            if not rows:
                break
            for row in rows:
                scanned += 1
                for p in extract_user_prompts(row):
                    if p in seen:
                        continue
                    seen.add(p)
                    if is_good_prompt(p):
                        out.append(p)
            offset += len(rows)
            if scanned % 1000 == 0:
                print(f"  scanned {scanned}, kept {len(out)}", file=sys.stderr)
    rng = random.Random(7)
    rng.shuffle(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"[harvest] wrote {len(out)} prompts -> {OUT}")


if __name__ == "__main__":
    main()