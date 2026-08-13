#!/usr/bin/env python3
"""eval_live.py - Forge Live Tool Eval (student hook).

Drives a student through held-out Forge plans against execute_one.
Scores name_sequence_match, dependency_arg_match, grounding_ok, format_ok.

This is not a BFCL v3 run. Category names (multiple, parallel, multi_turn)
exist so a later adapter can map rows; v0.2 reports Forge Live Tool Eval.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_plans import PLANS, build_chain
from curriculum import split_plan_ids
from mock_tools import TOOL_DEFINITIONS, execute_one
from prose_writer import _key_facts_from_payload, validate_answer_grounding

LABEL = "Forge Live Tool Eval"
FORGE_SPEC = "0.2"
NOTE = (
    "Held-out Forge plans against execute_one. "
    "This is not a BFCL v3 run."
)

# Same patterns as Distiller._parse_turn (copied; do not import Distiller).
THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

OPAQUE_KEYS = frozenset({"to", "doc_id", "attendee_email", "email"})
PLAN_INDEX = {p["id"]: i for i, p in enumerate(PLANS)}

CONTRACT = (
    "When you need to use tools, first output a short <thought> block "
    "(1-2 sentences explaining which tools you will call and why), then a "
    "<tool_call> block containing a JSON array of tool calls:\n"
    "<thought>\nBrief reasoning here.\n</thought>\n"
    '<tool_call>\n[{"name": "tool_name", "arguments": {"param": "value"}}]\n</tool_call>\n'
    "If a tool call fails, read the error payload and retry with corrected "
    "arguments in your next turn. If no tool is needed, answer directly without "
    "any tags."
)

SCORE_KEYS = (
    "name_sequence_match",
    "dependency_arg_match",
    "grounding_ok",
    "format_ok",
)


def default_system_prompt(tools=None) -> str:
    schemas = json.dumps(tools if tools is not None else TOOL_DEFINITIONS)
    return (
        "You are a tool-calling agent with access to these tools:\n"
        f"{schemas}\n"
        f"{CONTRACT}\n"
        "These tasks often need multiple dependent steps. When a tool result "
        "gives you a value needed for the next call (an email address, a plan, "
        "a document id), use that exact value. Do not invent opaque identifiers."
    )


def parse_turn(content: str) -> tuple[str | None, list[dict] | None]:
    """Extract <thought> and <tool_call> from assistant content.

    Copied from Distiller._parse_turn so eval_live does not import Distiller.
    """
    thought_m = THOUGHT_RE.search(content or "")
    thought = thought_m.group(1).strip() if thought_m else None
    tool_m = TOOL_CALL_RE.search(content or "")
    calls = None
    if tool_m:
        try:
            calls = json.loads(tool_m.group(1).strip())
            if isinstance(calls, dict):
                calls = [calls]
            if not isinstance(calls, list):
                calls = None
            else:
                for c in calls:
                    if not isinstance(c, dict) or "name" not in c:
                        raise ValueError("missing name")
                    if not isinstance(c.get("arguments"), dict):
                        c["arguments"] = (
                            json.loads(c.get("arguments", "{}"))
                            if isinstance(c.get("arguments"), str)
                            else {}
                        )
        except Exception:
            calls = None
    return thought, calls


def _student_complete(student, messages: list[dict]) -> str:
    if hasattr(student, "complete"):
        out = student.complete(messages)
    elif callable(student):
        out = student(messages)
    else:
        raise TypeError("student must provide complete(messages) or be callable")
    if out is None:
        return ""
    return out if isinstance(out, str) else str(out)


def _collect_learned(obj, learned: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_learned(v, learned)
    elif isinstance(obj, list):
        for v in obj:
            _collect_learned(v, learned)
    elif isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 3:
            learned.add(s)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        learned.add(str(obj))


def _is_opaque_or_learned(key: str, value, learned: set[str]) -> bool:
    if key in OPAQUE_KEYS:
        return True
    if isinstance(value, str) and EMAIL_RE.fullmatch(value.strip()):
        return True
    if value is None or isinstance(value, bool):
        return False
    s = str(value).strip()
    return bool(s) and s in learned


def _values_equal(gold, student) -> bool:
    if gold == student:
        return True
    if gold is None or student is None:
        return False
    return str(gold) == str(student)


def dependency_arg_match(gold_steps: list[dict], student_steps: list[dict]) -> float:
    """Fraction of opaque/learned gold args that equal the student args."""
    checks: list[bool] = []
    learned: set[str] = set()
    for i, gold in enumerate(gold_steps):
        stud = student_steps[i] if i < len(student_steps) else {}
        gargs = dict(gold.get("args") or {})
        sargs = dict(stud.get("args") or stud.get("arguments") or {})
        for key, gv in gargs.items():
            if not _is_opaque_or_learned(key, gv, learned):
                continue
            checks.append(_values_equal(gv, sargs.get(key)))
        _collect_learned(gold.get("result"), learned)
    if not checks:
        return 1.0
    return sum(1.0 for ok in checks if ok) / len(checks)


def plan_primary_category(plan_id: str | None, n_tools: int, skills: list[str] | None) -> str:
    skills = list(skills or [])
    if n_tools <= 1:
        return "simple"
    if "fanout" in skills:
        return "parallel"
    return "multiple"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _unscored_category(n: int, note: str) -> dict:
    row = {"scored": False, "n": n, "note": note}
    for k in SCORE_KEYS:
        row[k] = None
    return row


def _scored_category(rows: list[dict]) -> dict:
    out = {"scored": True, "n": len(rows)}
    for k in SCORE_KEYS:
        out[k] = _mean([float(r[k]) for r in rows])
    return out


def load_holdout_ids(path: Path) -> list[str]:
    """Holdout file is a JSON list of plan_id strings."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("holdout must be a JSON list of plan_id strings")


def _guess_opaque(key: str, gold) -> str:
    if key in ("to", "attendee_email", "email") or (
        isinstance(gold, str) and EMAIL_RE.fullmatch(gold.strip())
    ):
        guessed = "nobody@guessed.invalid"
        if str(gold) != guessed:
            return guessed
        return "other@guessed.invalid"
    if key == "doc_id":
        guessed = "doc.guessed"
        if str(gold) != guessed:
            return guessed
        return "doc.guessed2"
    return "GUESSED_OPAQUE"


def _find_opaque_mutation(traj: dict) -> tuple[int, str, object] | None:
    """Return (step_index, arg_key, guessed_value) for one opaque/learned arg."""
    learned: set[str] = set()
    for i, step in enumerate(traj.get("steps") or []):
        args = dict(step.get("args") or {})
        for key, gv in args.items():
            if _is_opaque_or_learned(key, gv, learned):
                return i, key, _guess_opaque(key, gv)
        _collect_learned(step.get("result"), learned)
    return None


def _emit_tool_turn(tool: str, args: dict, thought: str | None = None) -> str:
    call = {"name": tool, "arguments": args}
    body = json.dumps([call], ensure_ascii=False)
    reason = thought or f"Calling {tool} next."
    return (
        f"<thought>\n{reason}\n</thought>\n"
        f"<tool_call>\n{body}\n</tool_call>"
    )


def _final_from_traj(traj: dict) -> str:
    """Final answer that contains key facts from the last successful result."""
    steps = list(traj.get("steps") or [])
    if not steps:
        return "No tools were required."
    last_success = None
    for step in reversed(steps):
        if (step.get("result") or {}).get("status") == 200:
            last_success = step
            break
    source = last_success if last_success is not None else steps[-1]
    facts = _key_facts_from_payload(source.get("result") or {})
    for v in (source.get("args") or {}).values():
        if isinstance(v, str) and len(v.strip()) >= 3:
            facts.append(v.strip())
        elif not isinstance(v, bool) and v is not None:
            sv = str(v)
            if len(sv) >= 2:
                facts.append(sv)
    seen: set[str] = set()
    unique: list[str] = []
    for f in facts:
        fl = f.lower()
        if fl in ("true", "false"):
            continue
        if len(f) >= 2 and fl not in seen:
            seen.add(fl)
            unique.append(f)
    if not unique:
        return "Task finished."
    return "Completed the task. Key facts: " + " ".join(unique[:12])


class ReplayStudent:
    """Emits the next gold tool call, then a grounded final answer.

    Given a build_chain trajectory (constructor or bind_traj).
    """

    def __init__(self, traj: dict | None = None):
        self.traj = traj

    def bind_traj(self, traj: dict) -> None:
        self.traj = traj

    def complete(self, messages: list[dict]) -> str:
        steps = list((self.traj or {}).get("steps") or [])
        n_asst = sum(1 for m in messages if m.get("role") == "assistant")
        if n_asst < len(steps):
            step = steps[n_asst]
            return _emit_tool_turn(step["tool"], dict(step.get("args") or {}))
        return _final_from_traj(self.traj or {})


class WrongArgStudent(ReplayStudent):
    """Replay gold calls but mutate one opaque/learned argument."""

    def complete(self, messages: list[dict]) -> str:
        steps = list((self.traj or {}).get("steps") or [])
        n_asst = sum(1 for m in messages if m.get("role") == "assistant")
        if n_asst < len(steps):
            step = steps[n_asst]
            args = dict(step.get("args") or {})
            mut = _find_opaque_mutation(self.traj or {})
            if mut is not None:
                idx, key, guessed = mut
                if n_asst == idx:
                    args[key] = guessed
            return _emit_tool_turn(
                step["tool"], args, thought=f"Calling {step['tool']} with a guessed argument.",
            )
        return _final_from_traj(self.traj or {})


class HttpStudent:
    """OpenAI-compatible student: POST {endpoint}/chat/completions.

    Bearer token is read from the named env var only. No fallback key.
    """

    def __init__(self, endpoint: str, model: str, *, key_env: str = "", timeout: float = 120.0):
        self.endpoint = (endpoint or "").rstrip("/")
        self.model = model
        self.key_env = key_env or ""
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        url = self.endpoint + "/chat/completions"
        body = {"model": self.model, "messages": messages, "stream": False}
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key_env:
            key = os.environ.get(self.key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"student HTTP error: {exc}") from exc
        payload = json.loads(raw)
        msg = (payload.get("choices") or [{}])[0].get("message") or {}
        return msg.get("content") or ""


class LiveRunner:
    def __init__(self, executor=execute_one, tools=TOOL_DEFINITIONS, max_turns: int = 8):
        self.executor = executor
        self.tools = tools
        self.max_turns = int(max_turns)

    def run_one(self, student, prompt, system_prompt=None) -> dict:
        sys_p = (
            system_prompt
            if system_prompt is not None
            else default_system_prompt(self.tools)
        )
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": prompt},
        ]
        steps: list[dict] = []
        format_ok = True
        final = ""
        for _ in range(self.max_turns):
            try:
                content = _student_complete(student, messages)
            except Exception:
                format_ok = False
                break
            _thought, calls = parse_turn(content)
            messages.append({"role": "assistant", "content": content})
            if "<tool_call>" in content and calls is None:
                format_ok = False
                break
            if not calls:
                final = content
                break
            results = []
            for call in calls:
                name = call["name"]
                args = call.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
                result = self.executor(name, args)
                steps.append({
                    "tool": name,
                    "args": args,
                    "result": result,
                    "step_index": len(steps),
                })
                results.append(result)
            payload = results[0] if len(results) == 1 else results
            messages.append({
                "role": "tool",
                "content": json.dumps(payload, ensure_ascii=False),
            })
        else:
            last = messages[-1] if messages else {}
            if last.get("role") == "assistant" and "<tool_call>" not in (last.get("content") or ""):
                final = last.get("content") or ""
            else:
                format_ok = False

        last = messages[-1] if messages else {}
        if last.get("role") != "assistant":
            format_ok = False
        elif "<tool_call>" in (last.get("content") or ""):
            format_ok = False
        if not (final or "").strip() and last.get("role") == "assistant":
            final = last.get("content") or ""

        return {
            "messages": messages,
            "steps": steps,
            "final": final,
            "format_ok": format_ok,
            "n_turns": sum(1 for m in messages if m.get("role") == "assistant"),
        }

    def score_rollout(self, traj: dict, rollout: dict) -> dict:
        gold_steps = list(traj.get("steps") or [])
        stud_steps = list(rollout.get("steps") or [])
        gold_names = [s["tool"] for s in gold_steps]
        stud_names = [s["tool"] for s in stud_steps]
        name_match = 1.0 if gold_names == stud_names else 0.0
        dep = dependency_arg_match(gold_steps, stud_steps)
        final = rollout.get("final") or ""
        gerr = validate_answer_grounding(traj, final)
        grounding = 1.0 if gerr is None else 0.0
        format_ok = 1.0 if rollout.get("format_ok") else 0.0
        skills = list(traj.get("skills") or [])
        category = plan_primary_category(traj.get("plan_id"), len(gold_names), skills)
        return {
            "plan_id": traj.get("plan_id"),
            "plan_index": traj.get("plan_index"),
            "category": category,
            "skills": skills,
            "name_sequence_match": name_match,
            "dependency_arg_match": dep,
            "grounding_ok": grounding,
            "format_ok": format_ok,
            "n_gold_steps": len(gold_names),
            "n_student_steps": len(stud_names),
        }

    def score_plan(self, student, plan_id, rng) -> dict:
        idx = PLAN_INDEX.get(plan_id)
        if idx is None:
            return {
                "plan_id": plan_id,
                "plan_index": None,
                "category": None,
                "skills": [],
                "name_sequence_match": 0.0,
                "dependency_arg_match": 0.0,
                "grounding_ok": 0.0,
                "format_ok": 0.0,
                "n_gold_steps": 0,
                "n_student_steps": 0,
                "error": "unknown plan_id",
            }
        traj = build_chain(rng, plan_index=idx)
        bind = getattr(student, "bind_traj", None)
        if callable(bind):
            bind(traj)
        rollout = self.run_one(student, traj["prompt"])
        return self.score_rollout(traj, rollout)

    def run(self, student, plan_ids, n, rng) -> dict:
        ids = [str(p) for p in (plan_ids or [])]
        if not ids:
            _train, hold = split_plan_ids(PLANS, 0.0, 0)
            ids = list(_train)
        n = int(n)
        rows: list[dict] = []
        for i in range(max(n, 0)):
            pid = ids[i % len(ids)]
            rows.append(self.score_plan(student, pid, rng))
        return aggregate_report(rows, holdout_plan_ids=list(plan_ids or ids), n=n)


def aggregate_report(rows: list[dict], holdout_plan_ids: list[str], n: int) -> dict:
    simple_n = sum(1 for r in rows if r.get("category") == "simple")
    multiple = [r for r in rows if r.get("category") == "multiple"]
    parallel = [r for r in rows if r.get("category") == "parallel"]
    scored_all = [r for r in rows if r.get("category") is not None]
    overall = {k: _mean([float(r[k]) for r in scored_all]) for k in SCORE_KEYS}
    return {
        "label": LABEL,
        "forge_spec": FORGE_SPEC,
        "note": NOTE,
        "n": n,
        "n_rows": len(rows),
        "holdout_plan_ids": list(holdout_plan_ids),
        "scores": overall,
        "categories": {
            "simple": _unscored_category(
                simple_n,
                "1 tool; unscored in Forge Live Tool Eval v0.2",
            ),
            "multiple": _scored_category(multiple),
            "parallel": _scored_category(parallel),
            "multi_turn": _scored_category(scored_all),
            "irrelevance": _unscored_category(0, "not scored in v0.2"),
        },
        "per_plan": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Forge Live Tool Eval: score a student on held-out Forge plans. "
            "This is not a BFCL v3 run."
        )
    )
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL")
    ap.add_argument("--model", default=None, help="student model name")
    ap.add_argument(
        "--holdout",
        type=Path,
        required=True,
        help="JSON list of plan_id strings",
    )
    ap.add_argument("--n", type=int, default=50, help="number of scored chains")
    ap.add_argument("--out", type=Path, required=True, help="write live_eval JSON")
    ap.add_argument(
        "--replay",
        action="store_true",
        help="use ReplayStudent (no HTTP); for CI",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--key-env",
        default="",
        help="env var name holding the bearer token; never pass a raw key",
    )
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args(argv)

    if not args.replay and (not args.endpoint or not args.model):
        print("Forge Live Tool Eval requires --endpoint and --model unless --replay",
              file=sys.stderr)
        return 2

    try:
        holdout = load_holdout_ids(args.holdout)
    except Exception as exc:
        print(f"holdout load failed: {exc}", file=sys.stderr)
        return 2
    if not holdout:
        _train, hold = split_plan_ids(PLANS, 0.15, args.seed)
        holdout = hold or _train

    runner = LiveRunner(executor=execute_one, tools=TOOL_DEFINITIONS, max_turns=args.max_turns)
    rng = random.Random(args.seed)
    if args.replay:
        student = ReplayStudent()
    else:
        student = HttpStudent(args.endpoint, args.model, key_env=args.key_env)

    report = runner.run(student, holdout, args.n, rng)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    s = report["scores"]
    print(
        f"{LABEL}  n={report['n']}  "
        f"name_sequence_match={s.get('name_sequence_match')}  "
        f"dependency_arg_match={s.get('dependency_arg_match')}  "
        f"grounding_ok={s.get('grounding_ok')}  "
        f"format_ok={s.get('format_ok')}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
