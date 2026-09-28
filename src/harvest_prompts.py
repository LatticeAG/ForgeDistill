#!/usr/bin/env python3
"""Harvest real user prompts from open tool-calling datasets.

Uses the HF datasets-server API (no datasets lib needed). Extracts
user-turn prompts, filters to tool-callable / actionable queries,
dedupes, and writes to data/seeds/external.jsonl for the distiller.

This is the PROMPT lane only: harvested prompts are not wired into
generation. Injecting an arbitrary external prompt over an unrelated
deterministic chain would break the grounding gate; the chain lane
(src/external_chains.py) is the axis that widens the task space.
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("data/seeds/external.jsonl")

SOURCES = [
    # (dataset, config, split, max_rows_to_scan)
    ("glaiveai/glaive-function-calling-v2", "default", "train", 6000),
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", 2000),
    ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", 2000),
    ("NousResearch/hermes-function-calling-v1", "json_mode_agentic", "train", 1000),
    ("Team-ACE/ToolACE", "default", "train", 2000),
    ("lockon/xlam-function-calling-60k", "dataset", "train", 2000),
]

URL = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"

BAD_KEYWORDS = [
    "image", "picture", "photo", "video", "audio", "voice", "speech", "song",
    "draw", "paint", "generate an image", "jailbreak", "bomb", "explosive",
    "illegal drug", "credit card numbers", "ssn", "password of",
]

_USER_ROLES = ("user", "human")
# Glaive plain-text chat: "SYSTEM: ... USER: ... ASSISTANT: ... <|endoftext|>"
_TURN_RE = re.compile(r"(?:^|\n)\s*(SYSTEM|USER|HUMAN|ASSISTANT|GPT|TOOL)\s*:", re.I)


def fetch_rows(ds, cfg, split, offset, length=100, retries=5):
    url = URL.format(ds=ds, cfg=cfg, split=split, off=offset, length=length)
    req = urllib.request.Request(url, headers={"User-Agent": "latticeag-forge-distill/0.5"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())["rows"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503) and attempt + 1 < retries:
                ra = e.headers.get("Retry-After") if e.headers else None
                delay = float(ra) if ra and str(ra).isdigit() else 2.0 * (2 ** attempt)
                time.sleep(min(delay, 30.0))
                continue
            raise


def _msg_role(m: dict) -> str:
    return str(m.get("role") or m.get("from") or "").lower()


def _msg_content(m: dict):
    c = m.get("content")
    if isinstance(c, str):
        return c
    v = m.get("value")
    return v if isinstance(v, str) else None


def _collect_msgs(arr, prompts: list) -> None:
    for m in arr:
        if not isinstance(m, dict):
            continue
        if _msg_role(m) not in _USER_ROLES:
            continue
        c = _msg_content(m)
        if isinstance(c, str) and c.strip():
            prompts.append(c.strip())


def _extract_text_turns(text: str) -> list[str]:
    """USER: ... segments out of a plain-text chat blob (glaive shape)."""
    marks = list(_TURN_RE.finditer(text))
    prompts = []
    for i, m in enumerate(marks):
        if m.group(1).upper() not in ("USER", "HUMAN"):
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end]
        body = re.sub(r"<\|endoftext\|>.*", "", body, flags=re.S).strip()
        if body:
            prompts.append(body)
    return prompts


def extract_user_prompts(row):
    """Tolerant extraction: JSON-array chat columns, ShareGPT from/value
    lists, plain-text USER:/ASSISTANT: blobs, else any string cols."""
    prompts = []
    r = row.get("row", row)
    # Glaive variant: 'chat'/'conversation' is a JSON string of messages
    for key in ("chat", "conversation"):
        v = r.get(key)
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                arr = json.loads(v)
                if isinstance(arr, list):
                    _collect_msgs(arr, prompts)
                if prompts:
                    return prompts
            except Exception:
                pass
    # ShareGPT style: list columns of {from, value} or {role, content}
    for key in ("conversations", "dialogue", "turns", "messages"):
        v = r.get(key)
        if isinstance(v, list):
            _collect_msgs(v, prompts)
            if prompts:
                return prompts
    # Glaive plain-text chat: "USER: ... ASSISTANT: ... <|endoftext|>"
    for key in ("chat", "conversation", "text"):
        v = r.get(key)
        if isinstance(v, str) and _TURN_RE.search(v):
            prompts.extend(_extract_text_turns(v))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Harvest user prompts from open tool-calling corpora "
                    "into data/seeds/external.jsonl (prompt lane only - not "
                    "wired into generation).")
    ap.add_argument("--out", default=str(OUT),
                    help=f"output JSONL (default: {OUT})")
    args = ap.parse_args(argv)
    out_path = Path(args.out)

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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"[harvest] wrote {len(out)} prompts -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
