#!/usr/bin/env python3
"""Dataset dedup + filter pass.

Rules:
  1. Group traces by prompt; keep only the BEST trace per prompt
     (score: valid tool calls > thought blocks > more turns > no truncation)
  2. Drop traces with malformed <tool_call> JSON
  3. Drop traces with truncated assistant turns (abrupt endings)
  4. Keep class balance report
"""
import json, re, collections, random
from pathlib import Path

RAW = Path("/home/ubuntu/nanbeige-agentic/data/raw")
OUT = Path("/home/ubuntu/nanbeige-agentic/data/validated")
OUT.mkdir(parents=True, exist_ok=True)

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

GOOD_ENDINGS = (".", "?", "!", '"', "</tool_call>", ">", "```", ")", "]", "}", ":")


def parse_tool_calls(content):
    m = TOOL_CALL_RE.search(content)
    if not m:
        return None
    try:
        c = json.loads(m.group(1).strip())
        return c if isinstance(c, list) else ([c] if isinstance(c, dict) else None)
    except Exception:
        return None


def is_truncated(m):
    c = (m.get("content") or "").strip()
    if not c:
        return False
    return not c.endswith(GOOD_ENDINGS)


def score_trace(t):
    """Higher = better. Prefer: valid tool calls, thoughts, more turns, clean endings."""
    s = 0
    n_tool = 0
    n_bad_tool = 0
    n_thought = 0
    n_trunc = 0
    for m in t["messages"]:
        c = m.get("content") or ""
        if m["role"] == "assistant":
            if "<tool_call>" in c:
                if parse_tool_calls(c):
                    n_tool += 1
                else:
                    n_bad_tool += 1
            if THOUGHT_RE.search(c):
                n_thought += 1
            if is_truncated(m):
                n_trunc += 1
    if n_bad_tool:
        return -1000  # hard reject
    s += n_tool * 50
    s += n_thought * 10
    s += len(t["messages"]) * 2
    s -= n_trunc * 100
    return s


# Load all traces
traces = []
for p in RAW.glob("traces_*.jsonl"):
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                traces.append(json.loads(line))
            except Exception:
                pass

print(f"Loaded: {len(traces)}")

# Group by prompt, keep best
by_prompt = collections.defaultdict(list)
for t in traces:
    by_prompt[t["prompt"]].append(t)

kept = []
dropped_reasons = collections.Counter()
for prompt, group in by_prompt.items():
    scored = [(score_trace(t), t) for t in group]
    best_score, best = max(scored, key=lambda x: x[0])
    if best_score < -500:
        dropped_reasons["malformed_tool_call"] += 1
        continue
    kept.append(best)

# Filter truncated-only traces
final = []
for t in kept:
    truncs = [m for m in t["messages"] if m["role"] == "assistant" and is_truncated(m)]
    if truncs and score_trace(t) < 30:
        dropped_reasons["truncated"] += 1
        continue
    final.append(t)

# Shuffle deterministically
random.Random(42).shuffle(final)

# Write
out_file = OUT / "train_clean.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for t in final:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

# Report
classes = collections.Counter(t["seed_class"] for t in final)
teachers = collections.Counter(t["teacher"] for t in final)
tool_traces = sum(1 for t in final if any("<tool_call>" in m.get("content", "") for m in t["messages"]))
thought_traces = sum(1 for t in final if any("<thought>" in m.get("content", "") for m in t["messages"]))

print(f"\nAfter dedup (best per prompt): {len(kept)}")
print(f"After truncation filter: {len(final)}")
print(f"Dropped: {dict(dropped_reasons)}")
print(f"\nClass distribution: {dict(classes)}")
print(f"Teacher distribution:")
for k, v in teachers.most_common():
    print(f"  {k:45s} {v}")
print(f"\nTool-call traces: {tool_traces} ({100*tool_traces/len(final):.1f}%)")
print(f"Thought-block traces: {thought_traces} ({100*thought_traces/len(final):.1f}%)")
print(f"\nWrote: {out_file}")
