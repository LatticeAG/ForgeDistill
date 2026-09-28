#!/usr/bin/env python3
"""external_chains.py - public tool-calling corpora -> chain records.

Fetches rows from the HF datasets-server API (stdlib urllib only, no
datasets lib) and converts them into chain records that the distiller
consumes via distill --external-chains / --external-frac. This is the
axis that widens the task space: the corpus supplies the tool calls,
the teacher still writes prose only.

Sources:
  toolace  Team-ACE/ToolACE (Apache-2.0): multi-turn dialogues with
           RECORDED tool results (result_source="recorded"). One record
           per call segment, not per row.
  xlam     lockon/xlam-function-calling-60k (CC-BY-4.0): queries with
           calls but no results; results are synthesized deterministically
           (result_source="synthesized"). The Salesforce original is
           gated and 401s anonymous requests - do not switch back.

Record schema (one JSON object per line):
  {source, record_id, prompt, result_source, tool_schemas,
   steps: [{tool, args, result: {status, result}, expect, round}]}
`round` is the index of the assistant call turn within the segment; it
lets downstream code infer fanout without reordering steps.

CLI:
  python src/external_chains.py --source both --max-rows 300
  python src/external_chains.py --check data/seeds/external_chains.jsonl
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agentic_plans import validate_external_chain  # noqa: E402

OUT = Path("data/seeds/external_chains.jsonl")

SOURCES = {
    "toolace": ("Team-ACE/ToolACE", "default", "train"),
    "xlam": ("lockon/xlam-function-calling-60k", "dataset", "train"),
}

URL = ("https://datasets-server.huggingface.co/rows"
       "?dataset={ds}&config={cfg}&split={split}&offset={off}&length={n}")

MAX_STEPS = 6
PROMPT_MIN = 8
PROMPT_MAX = 400
PAGE = 100
PACE_S = 0.15


def fetch_rows(ds: str, cfg: str, split: str, offset: int, length: int = PAGE,
               retries: int = 5):
    """Same network pattern as harvest_prompts, plus 429/5xx backoff:
    datasets-server rate-limits anonymous callers, so retry with the
    Retry-After hint (or exponential backoff) before giving up."""
    url = URL.format(ds=ds, cfg=cfg, split=split, off=offset, n=length)
    req = urllib.request.Request(
        url, headers={"User-Agent": "latticeag-forge-distill/0.5"})
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


# ----------------------------------------------------------------------
# Call-block parsing (ToolACE assistant turns look like
#   [Market Trends API(trend_type="MARKET_INDEXES", country="us")]
# Accept only when every comma-item is a full Name(args) call covering the
# whole bracketed block; otherwise the turn is prose.
# ----------------------------------------------------------------------
_CALL_ITEM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.\- ]*?)\s*\((.*)\)", re.S)
_ARG_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")


def _split_top_level(s: str) -> list[str]:
    """Split on commas outside quotes and any () [] {} nesting."""
    parts, buf = [], []
    depth = 0
    quote = None
    esc = False
    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if quote:
            buf.append(ch)
            if ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_value(v: str):
    """key="str" / key=123 / key=true / key=None / JSON-ish / bare string."""
    if not v:
        return ""
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        try:
            return json.loads(v)
        except Exception:
            return v[1:-1]
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        return v[1:-1].replace("\\'", "'")
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return json.loads(v)
    except Exception:
        return v


def _parse_args(s: str) -> dict | None:
    """All args must be key=value. Positional args -> None (never guess)."""
    args: dict = {}
    s = s.strip()
    if not s:
        return args
    for part in _split_top_level(s):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            return None
        k, _, v = part.partition("=")
        k = k.strip()
        if not _ARG_NAME_RE.fullmatch(k):
            return None
        args[k] = _parse_value(v.strip())
    return args


def _parse_call_block(text: str) -> list[tuple[str, dict]] | None:
    """[Name(a=1), Other()] -> [(name, args)] or None when the bracketed
    block is not fully covered by Name(...) calls (trailing junk = prose)."""
    t = (text or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    inner = t[1:-1].strip()
    if not inner:
        return None
    calls = []
    for item in _split_top_level(inner):
        m = _CALL_ITEM_RE.fullmatch(item.strip())
        if not m:
            return None
        name = m.group(1).strip()
        args = _parse_args(m.group(2))
        if not name or args is None:
            return None
        calls.append((name, args))
    return calls


def _parse_result_block(text: str) -> list[dict] | None:
    """Tool turn: JSON array of {name, results} entries, else None."""
    t = (text or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    try:
        arr = json.loads(t)
    except Exception:
        return None
    if not isinstance(arr, list):
        return None
    return [e for e in arr
            if isinstance(e, dict) and "name" in e and "results" in e]


def _payload_has_image(payload) -> bool:
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return "base64" in blob or "data:image" in blob


# ----------------------------------------------------------------------
# ToolACE: one record per (user turn -> call turns -> tool turns) segment.
# ----------------------------------------------------------------------
def records_from_toolace_row(row: dict, row_idx: int):
    """-> (records, drops Counter, n_segments_without_calls)."""
    r = row.get("row", row) if isinstance(row, dict) else {}
    convs = r.get("conversations")
    if not isinstance(convs, list):
        return [], Counter({"bad_row": 1}), 0
    records: list[dict] = []
    drops: Counter = Counter()
    no_calls = 0
    seg_ord = -1

    prompt: str | None = None
    pending: list[tuple[str, dict, int]] = []  # calls awaiting results
    steps: list[dict] = []
    rounds = 0
    bad: str | None = None

    def flush() -> None:
        nonlocal prompt, pending, steps, rounds, bad, no_calls
        if prompt is None:
            return
        p = prompt.strip()
        if bad is not None:
            drops[bad] += 1
        elif pending:
            drops["unmatched_result"] += 1
        elif not steps:
            no_calls += 1
        elif len(steps) > MAX_STEPS:
            drops["too_many_steps"] += 1
        elif not (PROMPT_MIN <= len(p) <= PROMPT_MAX):
            drops["bad_prompt_len"] += 1
        else:
            records.append({
                "source": "toolace",
                "record_id": f"toolace-{row_idx:06d}-{seg_ord}",
                "prompt": p,
                "result_source": "recorded",
                "tool_schemas": [],
                "steps": steps,
            })
        prompt, pending, steps, rounds, bad = None, [], [], 0, None

    for turn in convs:
        if not isinstance(turn, dict):
            continue
        frm = str(turn.get("from") or turn.get("role") or "").lower()
        val = turn.get("value")
        if not isinstance(val, str):
            val = turn.get("content")
        if not isinstance(val, str):
            val = ""
        if frm in ("user", "human"):
            flush()
            prompt = val
            seg_ord += 1
        elif prompt is None:
            continue
        elif frm in ("assistant", "gpt"):
            calls = _parse_call_block(val)
            if calls is None:
                continue
            if pending:
                bad = bad or "unmatched_result"
                pending = []
            for name, args in calls:
                pending.append((name, args, rounds))
            rounds += 1
        elif frm in ("tool", "function"):
            if not pending:
                continue
            entries = _parse_result_block(val)
            if entries is None:
                bad = bad or "result_not_json"
                pending = []
                continue
            by_name: dict[str, list] = {}
            for e in entries:
                by_name.setdefault(str(e.get("name")), []).append(e)
            for name, args, rd in pending:
                queue = by_name.get(name)
                if not queue:
                    bad = bad or "unmatched_result"
                    continue
                payload = queue.pop(0).get("results")
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        bad = bad or "result_not_json"
                        continue
                if _payload_has_image(payload):
                    bad = bad or "base64_payload"
                    continue
                steps.append({
                    "tool": name,
                    "args": args,
                    "result": {"status": 200, "result": payload},
                    "expect": "success",
                    "round": rd,
                })
            pending = []
        # system / unknown roles do not break a segment
    flush()
    return records, drops, no_calls


# ----------------------------------------------------------------------
# xLAM: calls without results -> deterministic synthesized results.
# ----------------------------------------------------------------------
def _synth_value(name: str, args: dict, prop: str, ptype):
    digest = hashlib.sha256(
        f"{name}|{json.dumps(args, sort_keys=True)}|{prop}".encode("utf-8")
    ).hexdigest()
    h = int(digest[:8], 16)
    if ptype in ("integer", "number", "float", "int"):
        pool = [0, 1, 3, 7, 12, 42, 100, 365]
        return pool[h % len(pool)]
    if ptype in ("boolean", "bool"):
        return bool(h & 1)
    if ptype in ("array", "list"):
        return []
    if ptype in ("object", "dict"):
        return {}
    pool = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "ok", "ready"]
    return pool[h % len(pool)]


def _schema_props(schema: dict | None) -> dict:
    if not isinstance(schema, dict):
        return {}
    fn = schema.get("function")
    fn = fn if isinstance(fn, dict) else schema
    params = fn.get("parameters")
    if not isinstance(params, dict):
        return {}
    props = params.get("properties")
    return props if isinstance(props, dict) else {}


def _schema_name(schema: dict) -> str | None:
    if not isinstance(schema, dict):
        return None
    if isinstance(schema.get("name"), str):
        return schema["name"]
    fn = schema.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return None


def synthesize_result(name: str, args: dict, schema: dict | None) -> dict:
    """Deterministic payload: one value per declared schema property.

    Declared args echo the call's arguments; the rest are drawn from a
    small typed pool keyed by sha256(tool|args_json|prop). Byte-identical
    across runs - that determinism is the point.
    """
    data = {}
    for prop, pspec in _schema_props(schema).items():
        if prop in args:
            data[prop] = args[prop]
        else:
            ptype = pspec.get("type") if isinstance(pspec, dict) else None
            data[prop] = _synth_value(name, args, prop, ptype)
    return {"status": 200,
            "result": {"ok": True, "tool": name, "args": args, "data": data}}


def _load_jsonish(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v


def records_from_xlam_row(row: dict, row_idx: int):
    """-> (records, drops Counter, n_rows_without_calls)."""
    r = row.get("row", row) if isinstance(row, dict) else {}
    prompt = r.get("query")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not (PROMPT_MIN <= len(prompt) <= PROMPT_MAX):
        return [], Counter({"bad_prompt_len": 1}), 0
    raw_ans = r.get("answers")
    answers = _load_jsonish(raw_ans)
    if raw_ans is not None and answers is None:
        return [], Counter({"bad_answers": 1}), 0
    if not answers:
        return [], Counter(), 1
    if not isinstance(answers, list):
        return [], Counter({"bad_answers": 1}), 0
    calls: list[tuple[str, dict]] = []
    for a in answers:
        name = a.get("name") if isinstance(a, dict) else None
        args = _load_jsonish(a.get("arguments")) if isinstance(a, dict) else None
        if not isinstance(name, str) or not name.strip():
            return [], Counter({"bad_answers": 1}), 0
        if not isinstance(args, dict):
            return [], Counter({"bad_answers": 1}), 0
        calls.append((name.strip(), args))
    if not calls:
        return [], Counter(), 1
    if len(calls) > MAX_STEPS:
        return [], Counter({"too_many_steps": 1}), 0
    schemas = _load_jsonish(r.get("tools"))
    if not isinstance(schemas, list):
        schemas = []
    by_name = {}
    for s in schemas:
        nm = _schema_name(s)
        if nm:
            by_name[nm] = s
    steps = [{
        "tool": n,
        "args": a,
        "result": synthesize_result(n, a, by_name.get(n)),
        "expect": "success",
        "round": 0,
    } for n, a in calls]
    rec = {
        "source": "xlam",
        "record_id": f"xlam-{row_idx:06d}-0",
        "prompt": prompt,
        "result_source": "synthesized",
        "tool_schemas": schemas,
        "steps": steps,
    }
    return [rec], Counter(), 0


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def harvest_source(name: str, max_rows: int):
    ds, cfg, split = SOURCES[name]
    convert = records_from_toolace_row if name == "toolace" else records_from_xlam_row
    scanned = kept = no_calls = 0
    drops: Counter = Counter()
    seen_ids: set[str] = set()
    out: list[dict] = []
    offset = 0
    print(f"[{name}] fetching {ds} ({cfg}/{split})", file=sys.stderr)
    while scanned < max_rows:
        try:
            rows = fetch_rows(ds, cfg, split, offset, min(PAGE, max_rows - scanned))
        except Exception as e:
            print(f"  stop at offset {offset}: {e}", file=sys.stderr)
            break
        if not rows:
            break
        for i, item in enumerate(rows):
            scanned += 1
            row_idx = item.get("row_idx", offset + i) if isinstance(item, dict) else offset + i
            try:
                recs, d, nc = convert(item, row_idx)
            except Exception:
                recs, d, nc = [], Counter({"convert_error": 1}), 0
            drops.update(d)
            no_calls += nc
            for rec in recs:
                if rec["record_id"] in seen_ids:
                    drops["dup_record_id"] += 1
                    continue
                seen_ids.add(rec["record_id"])
                out.append(rec)
                kept += 1
            if scanned >= max_rows:
                break
        offset += len(rows)
        time.sleep(PACE_S)
    return out, scanned, kept, no_calls, drops


def check_file(path: Path) -> int:
    """Validate a chains file: per-source counts + duplicate record_ids."""
    path = Path(path)
    if not path.exists():
        print(f"[check] {path}: not found", file=sys.stderr)
        return 2
    counts: Counter = Counter()
    ids: Counter = Counter()
    bad_lines = invalid = 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            bad_lines += 1
            continue
        counts[str(rec.get("source") or "?")] += 1
        rid = rec.get("record_id")
        if rid:
            ids[str(rid)] += 1
        else:
            invalid += 1
        if validate_external_chain(rec.get("steps") or []):
            invalid += 1
    dupes = {k: v for k, v in ids.items() if v > 1}
    for src, n in sorted(counts.items()):
        print(f"[check] {src}: {n} records")
    print(f"[check] total={sum(counts.values())} "
          f"bad_lines={bad_lines} invalid={invalid} dup_record_ids={len(dupes)}")
    for k in sorted(dupes):
        print(f"  dup {k} x{dupes[k]}")
    return 1 if (dupes or bad_lines or invalid) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert public tool-calling corpora into ForgeDistill "
                    "chain records (data/seeds/external_chains.jsonl).")
    ap.add_argument("--source", choices=("toolace", "xlam", "both"),
                    default="both", help="which corpus to fetch (default: both)")
    ap.add_argument("--max-rows", type=int, default=2000,
                    help="rows to scan per source (default: 2000)")
    ap.add_argument("--out", default=str(OUT),
                    help=f"output JSONL (default: {OUT})")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of overwriting")
    ap.add_argument("--check", metavar="PATH", default=None,
                    help="validate a chains file (counts + duplicate ids) and exit")
    args = ap.parse_args(argv)

    if args.check:
        return check_file(Path(args.check))

    names = ["toolace", "xlam"] if args.source == "both" else [args.source]
    all_records: list[dict] = []
    for name in names:
        recs, scanned, kept, no_calls, drops = harvest_source(name, args.max_rows)
        all_records.extend(recs)
        why = ", ".join(f"{k}={v}" for k, v in sorted(drops.items())) or "-"
        print(f"[{name}] rows_scanned={scanned} kept={kept} "
              f"no_chain_segments={no_calls} dropped={sum(drops.values())} [{why}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[done] wrote {len(all_records)} records -> {out_path}"
          + (" (append)" if args.append else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
