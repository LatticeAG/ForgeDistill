#!/usr/bin/env python3
"""eval_card.py - dataset quality card for reversed-v2 traces.

Re-runs validate_prose_trace, validate_answer_grounding, validate_chain,
and gate_dependency_fidelity over traces_*.jsonl. Writes eval_card.json
and stamps a compact per-trace eval block.

CLI:
  python src/eval_card.py --input data/raw --out data/raw/eval_card.json --require-gates
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_plans import (
    PLANS,
    SKILLS,
    trajectory_hash,
    validate_chain,
    validate_external_chain,
)
from lineage import summarize
from mock_tools import EMAIL_BY_USER, execute_one
from prose_writer import (
    DISTILL_VERSION,
    NUDGE_TEXT,
    validate_answer_grounding,
    validate_prose_trace,
)

FORGE_SPEC = "0.2"
COVERAGE_NOTE = (
    "skills.coverage is computed against train_ids only; holdout_plan_ids lists "
    "excluded ids so readers can reproduce. Tier coverage assertions in CI run "
    "with --holdout-frac 0 fixtures."
)
CMD_EVAL_CARD = (
    "python src/eval_card.py --input data/raw --out data/raw/eval_card.json --require-gates"
)
CMD_STRESS_300 = (
    'python -c "import sys,random; sys.path.insert(0,\'src\'); '
    "from agentic_plans import build_chain, validate_chain; "
    "r=random.Random(0); "
    "print(sum(1 for _ in range(300) if validate_chain(build_chain(r)['steps'])))\""
)
KEY_FIELDS = (
    "email", "to", "plan", "temp_c", "condition", "count", "exists",
    "path", "value", "doc_id", "event_id",
)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TIERS = ("easy", "medium", "hard", "expert")
HIST_KEYS = ("2", "3", "4", "5")


def _expand_input(path: Path) -> list[Path]:
    """DIR -> DIR/traces_*.jsonl only. A file path is used as given."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.glob("traces_*.jsonl") if p.is_file())
    if path.is_file():
        return [path]
    return []


def load_traces(paths: list[Path]) -> list[dict]:
    """Load jsonl traces. Directories expand to traces_*.jsonl only."""
    traces: list[dict] = []
    seen_files: list[Path] = []
    for raw in paths:
        for p in _expand_input(Path(raw)):
            if p in seen_files:
                continue
            seen_files.append(p)
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        traces.append(obj)
    return traces


def _merge_token_sidecars(out_dir: Path, acc: dict | None = None) -> dict:
    """Sum .token_usage_*.json under a directory into acc. Zeros if none exist."""
    merged = acc or {"input": 0, "output": 0, "by_route": {}}
    merged.setdefault("input", 0)
    merged.setdefault("output", 0)
    merged.setdefault("by_route", {})
    if not out_dir.is_dir():
        return merged
    for p in sorted(out_dir.glob(".token_usage_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        merged["input"] += int(data.get("input", 0) or 0)
        merged["output"] += int(data.get("output", 0) or 0)
        by_route = data.get("by_route") or {}
        if not isinstance(by_route, dict):
            continue
        for route, usage in by_route.items():
            slot = merged["by_route"].setdefault(str(route), {"input": 0, "output": 0})
            if not isinstance(usage, dict):
                continue
            slot["input"] += int(usage.get("input", 0) or 0)
            slot["output"] += int(usage.get("output", 0) or 0)
    return merged


def _parse_tool_calls(content: str) -> list[dict] | None:
    """Parse <tool_call> JSON to a non-empty list of {name, arguments}."""
    m = TOOL_CALL_RE.search(content or "")
    if not m:
        return None
    try:
        calls = json.loads(m.group(1).strip())
    except Exception:
        return None
    if not (isinstance(calls, list) and calls):
        return None
    out: list[dict] = []
    for c in calls:
        if not (isinstance(c, dict) and "name" in c and "arguments" in c):
            return None
        out.append(c)
    return out


def _parse_tool_payload(content: str) -> dict | None:
    try:
        obj = json.loads(content or "")
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _as_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _inner_result(payload: dict) -> dict:
    inner = payload.get("result") if isinstance(payload, dict) else None
    return inner if isinstance(inner, dict) else {}


def reconstruct_steps(trace: dict) -> list[dict]:
    """Return chain_steps if present, else rebuild from messages."""
    cs = trace.get("chain_steps")
    if isinstance(cs, list) and cs:
        return list(cs)

    steps: list[dict] = []
    msgs = trace.get("messages") or []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<tool_call>" not in content:
            continue
        calls = _parse_tool_calls(content)
        if not calls:
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        stored = None
        if nxt and nxt.get("role") == "tool":
            stored = _parse_tool_payload(nxt.get("content") or "")
        if stored is None:
            stored = {}
        for call in calls:
            name = call.get("name")
            args = _as_args(call.get("arguments"))
            result = stored
            if (
                isinstance(stored, dict)
                and name in stored
                and isinstance(stored[name], dict)
                and "status" in stored[name]
            ):
                result = stored[name]
            status = result.get("status") if isinstance(result, dict) else None
            expect = "success" if status == 200 else "error"
            steps.append({
                "tool": name,
                "args": args,
                "result": result,
                "expect": expect,
            })
    return steps


def gate_prose(trace) -> str | None:
    return validate_prose_trace(trace)


def gate_grounding(trace, steps) -> str | None:
    msgs = trace.get("messages") or []
    if not msgs:
        return "no messages"
    final = (msgs[-1] or {}).get("content") or ""
    traj = {"prompt": trace.get("prompt", ""), "steps": steps or []}
    return validate_answer_grounding(traj, final)


def _is_external(trace) -> bool:
    """External chains carry ext-<source>-<record_id> plan ids."""
    return str(trace.get("plan_id") or "").startswith("ext-")


def gate_chain(steps, trace=None) -> str | None:
    """Chain gate: corpus chains get the generic structural rules only.

    The internal laws in validate_chain (registered opaque addresses,
    calendar attendees, recovery tagging) are properties of the mock tool
    surface; corpus tool names are arbitrary.
    """
    if trace is not None and _is_external(trace):
        return validate_external_chain(steps or [])
    return validate_chain(steps or [])


def _external_fidelity(msgs: list[dict], steps: list[dict]) -> str | None:
    """External chains have no mock executor to re-run; fidelity is that the
    assistant tool_calls match the recorded chain in order and content."""
    cursor = 0
    for i, m in enumerate(msgs):
        content = m.get("content") or ""
        if m.get("role") != "assistant" or "<tool_call>" not in content:
            continue
        calls = _parse_tool_calls(content) or []
        for call in calls:
            if cursor >= len(steps):
                return f"message {i}: more tool calls than chain steps"
            s = steps[cursor]
            if call.get("name") != s.get("tool"):
                return (f"call {cursor}: name {call.get('name')!r} "
                        f"!= chain tool {s.get('tool')!r}")
            if _as_args(call.get("arguments")) != _as_args(s.get("args")):
                return f"call {cursor}: arguments differ from chain record"
            cursor += 1
    if cursor != len(steps):
        return f"{len(steps) - cursor} chain steps missing from messages"
    return None


def _collect_key_fields(obj: object, acc: dict | None = None) -> dict:
    acc = acc if acc is not None else {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in KEY_FIELDS and k not in acc:
                acc[k] = v
            _collect_key_fields(v, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_key_fields(item, acc)
    return acc


def _prior_result_blob(prior_steps: list[dict]) -> str:
    parts = []
    for s in prior_steps:
        parts.append(json.dumps(s.get("result"), ensure_ascii=False, default=str))
    return "\n".join(parts)


def _emails_from_get_user(prior_steps: list[dict]) -> set:
    out: set = set()
    for s in prior_steps:
        if s.get("tool") != "get_user":
            continue
        payload = s.get("result") or {}
        if payload.get("status") != 200:
            continue
        email = _inner_result(payload).get("email")
        if email:
            out.add(email)
    return out


def _doc_ids_from_search_query(prior_steps: list[dict]) -> set:
    out: set = set()
    for s in prior_steps:
        if s.get("tool") != "search.query":
            continue
        payload = s.get("result") or {}
        if payload.get("status") != 200:
            continue
        hits = _inner_result(payload).get("hits")
        if not isinstance(hits, list):
            continue
        for h in hits:
            if isinstance(h, dict) and "doc_id" in h:
                out.add(h["doc_id"])
    return out


def _nudge_in_messages(trace: dict) -> bool:
    for m in trace.get("messages") or []:
        if NUDGE_TEXT in (m.get("content") or ""):
            return True
    return False


def _trace_has_malformed_tool_call(trace: dict) -> bool:
    for m in trace.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<tool_call>" not in content:
            continue
        if _parse_tool_calls(content) is None:
            return True
    return False


def _n_rounds(trace: dict) -> int:
    n = 0
    for m in trace.get("messages") or []:
        if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or ""):
            n += 1
    return n


def _n_tool_calls(trace: dict, steps: list[dict]) -> int:
    total = 0
    found = False
    for m in trace.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<tool_call>" not in content:
            continue
        found = True
        calls = _parse_tool_calls(content)
        total += len(calls) if calls else 1
    if found:
        return total
    return len(steps)


def _successful_send_tos(steps: list[dict]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for i, s in enumerate(steps):
        if s.get("tool") != "send_email":
            continue
        payload = s.get("result") or {}
        if payload.get("status") != 200:
            continue
        to = _inner_result(payload).get("to")
        if to is None:
            to = _as_args(s.get("args")).get("to")
        if isinstance(to, str) and to:
            found.append((i, to))
    return found


def _send_email_tos_learned(steps: list[dict]) -> bool:
    """True if every successful send_email to appeared in an earlier tool payload."""
    for i, to in _successful_send_tos(steps):
        blob = _prior_result_blob(steps[:i])
        if to not in blob:
            return False
    return True


def gate_dependency_fidelity(trace, steps) -> str | None:
    """Structural + re-execution dependency law. Error string or None."""
    msgs = list(trace.get("messages") or [])
    seq = list(steps or [])
    if not seq:
        seq = reconstruct_steps(trace)

    for i, m in enumerate(msgs):
        content = m.get("content") or ""
        if NUDGE_TEXT in content:
            return "final-answer nudge leaked into training messages"
        if m.get("role") != "assistant" or "<tool_call>" not in content:
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if not nxt or nxt.get("role") != "tool":
            return f"assistant tool_call at {i} not followed by a tool message"
        if _parse_tool_calls(content) is None:
            return f"malformed <tool_call> JSON at message {i}"

    if _is_external(trace):
        # Corpus tools cannot be re-executed against the mock executor;
        # fidelity for an external chain is the messages <-> chain match.
        return _external_fidelity(msgs, seq)

    registered = set(EMAIL_BY_USER.values())
    has_calendar = any(s.get("tool") == "calendar.create" for s in seq)
    has_search_get = any(s.get("tool") == "search.get" for s in seq)

    for i, s in enumerate(seq):
        name = s.get("tool") or s.get("name")
        args = _as_args(s.get("args") if "args" in s else s.get("arguments"))
        stored = s.get("result") if isinstance(s.get("result"), dict) else {}
        if not name:
            return f"step {i}: missing tool name"
        live = execute_one(name, args)
        stored_status = stored.get("status")
        live_status = live.get("status")
        if live_status != stored_status:
            return (
                f"step {i} {name}: re-exec status {live_status} != stored {stored_status}"
            )
        if stored_status == 200:
            stored_fields = _collect_key_fields(_inner_result(stored))
            live_fields = _collect_key_fields(_inner_result(live))
            for k, v in stored_fields.items():
                if k not in live_fields or live_fields[k] != v:
                    return f"step {i} {name}: key field {k!r} mismatch"

        prior = seq[:i]
        if name == "send_email" and stored_status == 200:
            to = _inner_result(stored).get("to")
            if to is None:
                to = args.get("to")
            if to not in registered:
                return f"step {i}: send_email to unregistered address {to!r}"
            if not isinstance(to, str) or to not in _prior_result_blob(prior):
                return f"step {i}: send_email to {to!r} was not learned from an earlier tool payload"

        if has_calendar and name == "calendar.create" and stored_status == 200:
            att = _inner_result(stored).get("attendee_email")
            if att is None:
                att = args.get("attendee_email")
            learned = _emails_from_get_user(prior)
            if att not in registered:
                return f"step {i}: calendar.create attendee_email {att!r} is not registered"
            if att not in learned:
                return (
                    f"step {i}: calendar.create attendee_email {att!r} "
                    "was not returned by an earlier get_user"
                )

        if has_search_get and name == "search.get" and stored_status == 200:
            doc_id = _inner_result(stored).get("doc_id")
            if doc_id is None:
                doc_id = args.get("doc_id")
            known = _doc_ids_from_search_query(prior)
            if doc_id not in known:
                return (
                    f"step {i}: search.get doc_id {doc_id!r} "
                    "was not in an earlier search.query hits list"
                )

    return None


def _frac(num: int, den: int, *, empty: float) -> float:
    if den == 0:
        return float(empty)
    return num / den


def _traj_hash_of(trace: dict, steps: list[dict]) -> str:
    if trace.get("traj_hash"):
        return str(trace["traj_hash"])
    return trajectory_hash({
        "plan_id": trace.get("plan_id", trace.get("plan_template")),
        "vars": trace.get("vars") or {},
        "steps": steps,
    })


def _tier_of(trace: dict) -> str | None:
    t = trace.get("plan_tier") or (trace.get("eval") or {}).get("tier")
    if t in TIERS:
        return t
    return None


def _is_cross_teacher(trace: dict) -> bool:
    th = trace.get("teacher_thoughts")
    ta = trace.get("teacher_answer")
    return th is not None and ta is not None and th != ta


def _load_jsonl_glob(directory: Path, pattern: str) -> list[dict]:
    rows: list[dict] = []
    if not directory.is_dir():
        return rows
    for p in sorted(directory.glob(pattern)):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _observability(n: int, token_usage: dict, lineage_rows: list, reject_rows: list) -> dict:
    obs = summarize(list(lineage_rows or []), list(reject_rows or []), token_usage)
    obs["n_kept"] = n
    inp = int(token_usage.get("input") or 0)
    outp = int(token_usage.get("output") or 0)
    obs["tokens_per_kept_trace"] = {
        "input": (inp / n) if n else 0,
        "output": (outp / n) if n else 0,
    }
    return obs


def compute_card(traces, token_usage=None, extra=None) -> dict:
    extra = extra or {}
    token_usage = token_usage or {}
    traces = list(traces or [])
    n = len(traces)

    n_prose = n_ground = n_chain = n_fid = 0
    n_nudge = n_malformed = 0
    n_multi = 0
    n_repaired = 0
    n_cross = 0
    n_fallback = 0
    n_send = 0
    n_send_learned = 0
    hist: Counter = Counter()
    prompts: set[str] = set()
    hashes: set[str] = set()
    skill_counts: Counter = Counter()
    plan_ids: set[str] = set()
    ext_plan_ids: set[str] = set()
    tier_counts = {t: 0 for t in TIERS}
    routes: Counter = Counter()

    for trace in traces:
        steps = reconstruct_steps(trace)
        if gate_prose(trace) is None:
            n_prose += 1
        if gate_grounding(trace, steps) is None:
            n_ground += 1
        if gate_chain(steps, trace) is None:
            n_chain += 1
        if gate_dependency_fidelity(trace, steps) is None:
            n_fid += 1
        if _nudge_in_messages(trace):
            n_nudge += 1
        if _trace_has_malformed_tool_call(trace):
            n_malformed += 1
        rounds = _n_rounds(trace)
        hist[str(rounds)] += 1
        if rounds >= 2:
            n_multi += 1
        prompts.add(trace.get("prompt") or "")
        hashes.add(_traj_hash_of(trace, steps))
        if (trace.get("eval") or {}).get("repaired"):
            n_repaired += 1
        if _is_cross_teacher(trace):
            n_cross += 1
        if (trace.get("eval") or {}).get("cross_teacher_fallback"):
            n_fallback += 1
        tos = _successful_send_tos(steps)
        if tos:
            n_send += 1
            if _send_email_tos_learned(steps):
                n_send_learned += 1
        for sk in (trace.get("plan_skills") or (trace.get("eval") or {}).get("skills") or []):
            skill_counts[sk] += 1
        pid = trace.get("plan_id")
        if pid:
            if str(pid).startswith("ext-"):
                ext_plan_ids.add(str(pid))
            else:
                plan_ids.add(pid)
        tier = _tier_of(trace)
        if tier:
            tier_counts[tier] += 1
        teacher = trace.get("teacher")
        if teacher:
            routes[str(teacher)] += 1

    present = set(skill_counts)
    missing = [s for s in SKILLS if s not in present]
    round_depth = {k: int(hist.get(k, 0)) for k in HIST_KEYS}
    for k, v in hist.items():
        if k not in round_depth:
            round_depth[k] = int(v)

    input_paths = extra.get("input_paths", [])
    input_paths = [str(p) for p in input_paths]
    holdout = extra.get("holdout_plan_ids", [])
    if not isinstance(holdout, list):
        holdout = list(holdout)

    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "forge_spec": FORGE_SPEC,
        "distill_version": DISTILL_VERSION,
        "created_at": created,
        "input_paths": input_paths,
        "n_traces": n,
        "gates": {
            "validate_prose_trace_pass": _frac(n_prose, n, empty=1.0),
            "validate_answer_grounding_pass": _frac(n_ground, n, empty=1.0),
            "validate_chain_pass": _frac(n_chain, n, empty=1.0),
            "dependency_fidelity_pass": _frac(n_fid, n, empty=1.0),
            "nudge_leak_rate": _frac(n_nudge, n, empty=0.0),
            "malformed_tool_call_rate": _frac(n_malformed, n, empty=0.0),
        },
        "structure": {
            "multi_round_rate": _frac(n_multi, n, empty=0.0),
            "round_depth_histogram": round_depth,
            "unique_prompt_rate": _frac(len(prompts), n, empty=0.0),
            "unique_traj_hash_rate": _frac(len(hashes), n, empty=0.0),
            "send_email_learned_address_rate": _frac(n_send_learned, n_send, empty=1.0),
            "repaired_rate": _frac(n_repaired, n, empty=0.0),
            "cross_teacher_split_rate": _frac(n_cross, n, empty=0.0),
            "cross_teacher_fallback_rate": _frac(n_fallback, n, empty=0.0),
        },
        "skills": {
            "counts": dict(skill_counts),
            "n_tags_present": len(present),
            "n_tags_defined": len(SKILLS),
            "missing_tags": missing,
            "n_templates_used": len(plan_ids),
            "n_external_templates_used": len(ext_plan_ids),
            "n_templates_defined": len(PLANS),
            "tier_counts": tier_counts,
            "coverage_note": COVERAGE_NOTE,
        },
        "teachers": {
            "routes": dict(routes),
            "cross_teacher_rate": _frac(n_cross, n, empty=0.0),
        },
        "holdout_plan_ids": holdout,
        "tokens": {
            "input": int(token_usage.get("input", 0) or 0),
            "output": int(token_usage.get("output", 0) or 0),
            "by_route": dict(token_usage.get("by_route") or {}),
        },
        "commands": {
            "eval_card": CMD_EVAL_CARD,
            "stress_300": CMD_STRESS_300,
        },
        "observability": _observability(
            n,
            token_usage,
            extra.get("lineage_rows") or [],
            extra.get("reject_rows") or [],
        ),
    }


def stamp_eval(trace, traj, repaired=False, cross_teacher=False,
               cross_teacher_fallback=False) -> None:
    """Mutate trace with forge_spec, plan_tier, and the compact eval block."""
    traj = traj or {}
    steps = list(traj.get("steps") or []) or reconstruct_steps(trace)
    if not trace.get("plan_tier") and traj.get("tier"):
        trace["plan_tier"] = traj["tier"]
    skills = list(traj.get("skills") or trace.get("plan_skills") or [])
    tier = traj.get("tier") or trace.get("plan_tier")
    trace["forge_spec"] = FORGE_SPEC
    trace["eval"] = {
        "format_ok": gate_prose(trace) is None,
        "grounding_ok": gate_grounding(trace, steps) is None,
        "chain_ok": gate_chain(steps, trace) is None,
        "dependency_ok": gate_dependency_fidelity(trace, steps) is None,
        "n_rounds": _n_rounds(trace),
        "n_tool_calls": _n_tool_calls(trace, steps),
        "skills": skills,
        "tier": tier,
        "repaired": bool(repaired),
        "cross_teacher": bool(cross_teacher),
    }
    if cross_teacher_fallback:
        trace["eval"]["cross_teacher_fallback"] = True


def gates_required_ok(card: dict) -> bool:
    g = card.get("gates") or {}
    return (
        g.get("validate_prose_trace_pass") == 1.0
        and g.get("validate_answer_grounding_pass") == 1.0
        and g.get("validate_chain_pass") == 1.0
        and g.get("dependency_fidelity_pass") == 1.0
        and g.get("nudge_leak_rate") == 0.0
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute a reversed-v2 eval card over traces_*.jsonl"
    )
    ap.add_argument(
        "--input", action="append", required=True, type=Path,
        help="trace jsonl file, or a directory expanded to traces_*.jsonl only",
    )
    ap.add_argument("--out", type=Path, default=None, help="write eval_card JSON here")
    ap.add_argument(
        "--require-gates", action="store_true",
        help="exit 1 unless prose/grounding/chain/fidelity are 1.0 and nudge_leak_rate is 0.0. "
             "malformed_tool_call_rate is reported in JSON but is not part of the exit predicate",
    )
    args = ap.parse_args(argv)

    files: list[Path] = []
    for p in args.input:
        files.extend(_expand_input(p))
    traces = load_traces(list(args.input))
    holdout: list = []
    token_usage = {"input": 0, "output": 0, "by_route": {}}
    for p in args.input:
        if not p.is_dir():
            continue
        hp = p / "holdout_plan_ids.json"
        if hp.is_file():
            try:
                loaded = json.loads(hp.read_text(encoding="utf-8"))
            except Exception:
                loaded = []
            if isinstance(loaded, list):
                holdout = loaded
        token_usage = _merge_token_sidecars(p, token_usage)
    lineage_rows: list[dict] = []
    reject_rows: list[dict] = []
    for p in args.input:
        if p.is_dir():
            lineage_rows.extend(_load_jsonl_glob(p, "lineage_*.jsonl"))
            reject_rows.extend(_load_jsonl_glob(p, "rejects_*.jsonl"))
    card = compute_card(
        traces,
        token_usage=token_usage,
        extra={
            "input_paths": [str(p) for p in files],
            "holdout_plan_ids": holdout,
            "lineage_rows": lineage_rows,
            "reject_rows": reject_rows,
        },
    )

    text = json.dumps(card, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} n_traces={card['n_traces']}")
    else:
        sys.stdout.write(text)

    if args.require_gates and not gates_required_ok(card):
        g = card["gates"]
        print(
            "REQUIRE-GATES failed: "
            f"prose={g['validate_prose_trace_pass']} "
            f"grounding={g['validate_answer_grounding_pass']} "
            f"chain={g['validate_chain_pass']} "
            f"fidelity={g['dependency_fidelity_pass']} "
            f"nudge_leak={g['nudge_leak_rate']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
