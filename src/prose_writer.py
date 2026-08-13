"""prose_writer.py - Phase 2 of reversed distillation.

Given a guaranteed-correct trajectory (prompt + steps with real tool
results), the teacher writes ONLY the language layer:
  - a <thought> before each tool-call step (why/what to call)
  - a recovery thought when a step failed (error correction reasoning)
  - the final answer to the user (grounded in the tool results)

The harness assembles the final trace: teacher's thoughts + our exact
tool calls (never teacher-generated) + real tool results. This makes
malformed tool calls, missed dependencies, and lazy chains IMPOSSIBLE.

Teacher contract is thoughts + FINAL_ANSWER only. Tool-call JSON is
injected by assemble_trace, never parsed from the teacher.
"""
from __future__ import annotations
import json
import re

from mock_tools import EMAIL_BY_USER, USERS

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.S)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
OPAQUE_ID_RE = re.compile(r"\b(?:evt|doc)\.[A-Za-z0-9]+\b")
NUDGE_TEXT = "Now provide the final answer to the user's original request."
DISTILL_VERSION = "reversed-v2"

KNOWN_NAMES = {u["name"] for u in USERS.values()}
KNOWN_EMAILS = set(EMAIL_BY_USER.values())


def _is_opaque_id(s: str) -> bool:
    return bool(OPAQUE_ID_RE.fullmatch((s or "").strip()))


def build_prose_prompt(traj: dict) -> str:
    """Build the teacher prompt for one trajectory.

    The teacher emits N <thought> blocks and FINAL_ANSWER only.
    Do NOT ask for <tool_call> - those are injected from the trajectory.
    """
    n = len(traj["steps"])
    lines = [
        "You are generating training data for a tool-calling model. Below is a",
        "conversation a USER started, and the tool calls that were made along with",
        "their EXACT results. Your job: write the assistant's internal thoughts",
        "and the final answer ONLY.",
        "Do not emit tool calls. Do not invent tools, arguments, or result values.",
        "",
        "USER: " + traj["prompt"],
        "",
        "TOOL EXECUTION RECORD:",
    ]
    for i, s in enumerate(traj["steps"]):
        payload = json.dumps(s["result"], ensure_ascii=False)
        lines.append(f"  Step {i+1}: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})")
        lines.append(f"    -> {payload}")
    lines += [
        "",
        f"Write exactly {n} thought blocks, one per tool step, then the final answer.",
        "Each thought is 1-2 sentences: which tool you are about to call and why,",
        "using the task plus any prior results. If the previous call failed, say",
        "what went wrong and how you are correcting it (or why you are stopping).",
        "",
        "After the thoughts, write the final answer to the user as plain text.",
        "The final answer MUST use REAL values from the tool results above",
        "(exact numbers, names, addresses, counts, conditions). Do not invent values.",
        "If a call failed, say so; do not pretend a substitute succeeded.",
        "Opaque ids (event_id, doc_id) may be quoted only if they appear in the tool results above. "
        "Do not invent them. You are not required to recite them. "
        "Prefer email, title, name, plan, value, error code.",
        "",
        "Format your reply EXACTLY as:",
        "THOUGHTS:",
        "<thought>reasoning for step 1</thought>",
        "<thought>reasoning for step 2</thought>",
        f"(... {n} thought blocks total ...)",
        "FINAL_ANSWER:",
        "plain-text answer grounded in the tool results",
        "",
        f"You must output {n} <thought>...</thought> blocks. No <tool_call> blocks.",
        "No extra user turns. Do not write a line like",
        f"'{NUDGE_TEXT}'",
    ]
    return "\n".join(lines)


def parse_thoughts(content: str, n_steps: int) -> list[str] | None:
    """Same <thought> rules as today. FINAL_ANSWER is optional and ignored."""
    if not content or n_steps < 1:
        return None
    thoughts = [t.strip() for t in THOUGHT_RE.findall(content) if t.strip()]
    if len(thoughts) < n_steps:
        return None
    return thoughts[:n_steps]


def parse_final(content: str) -> str | None:
    """Extract FINAL_ANSWER body, stripping leftover tags. None if missing/empty."""
    if not content:
        return None
    m = re.search(r"FINAL_ANSWER:\s*(.*)$", content, re.S | re.I)
    if not m:
        return None
    final = m.group(1).strip()
    final = re.sub(r"<tool_call>.*?</tool_call>", "", final, flags=re.S).strip()
    final = re.sub(r"<thought>.*?</thought>", "", final, flags=re.S).strip()
    if not final:
        return None
    return final


def parse_teacher_output(content: str, n_steps: int) -> dict | None:
    """Parse teacher reply. Returns {"thoughts": [...n_steps], "final": str} or None.

    Wrapper around parse_thoughts + parse_final. Single-teacher path stays one blob.
    """
    thoughts = parse_thoughts(content, n_steps)
    final = parse_final(content)
    if not thoughts or not final:
        return None
    return {"thoughts": thoughts, "final": final}


def build_answer_prompt(traj: dict, thoughts: list[str]) -> str:
    """Answer-teacher prompt: execution record plus thought blocks. FINAL_ANSWER only."""
    n = len(traj["steps"])
    lines = [
        "You are generating the FINAL ANSWER for a tool-calling training trace.",
        "Another teacher already wrote the internal thoughts. You see the user",
        "request, the exact tool execution record, and those thoughts.",
        "Write ONLY the final answer. Do not emit tool calls. Do not invent values.",
        "",
        "USER: " + traj["prompt"],
        "",
        "TOOL EXECUTION RECORD:",
    ]
    for i, s in enumerate(traj["steps"]):
        payload = json.dumps(s["result"], ensure_ascii=False)
        lines.append(f"  Step {i+1}: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})")
        lines.append(f"    -> {payload}")
    lines += ["", "THOUGHTS:"]
    for i, t in enumerate(thoughts[:n]):
        lines.append(f"<thought>{t}</thought>")
    lines += [
        "",
        "Write the final answer to the user as plain text.",
        "The final answer MUST use REAL values from the tool results above",
        "(exact numbers, names, addresses, counts, conditions). Do not invent values.",
        "If a call failed, say so; do not pretend a substitute succeeded.",
        "Opaque ids (event_id, doc_id) may be quoted only if they appear in the tool results above. "
        "Do not invent them. You are not required to recite them. "
        "Prefer email, title, name, plan, value, error code.",
        "",
        "Format your reply EXACTLY as:",
        "FINAL_ANSWER:",
        "plain-text answer grounded in the tool results",
        "",
        "No <tool_call> blocks. Do not write a line like",
        f"'{NUDGE_TEXT}'",
    ]
    return "\n".join(lines)


def _corpus(traj: dict) -> str:
    parts = [traj.get("prompt") or ""]
    for s in traj.get("steps") or []:
        parts.append(json.dumps(s.get("args", {}), ensure_ascii=False, default=str))
        parts.append(json.dumps(s.get("result", {}), ensure_ascii=False, default=str))
    return "\n".join(parts)


def _significant_number(tok: str) -> bool:
    try:
        v = float(tok)
    except ValueError:
        return False
    if "." in tok:
        return True
    return abs(v) >= 10


def _key_facts_from_payload(payload: dict) -> list[str]:
    """Extract strings the final answer should mention."""
    facts: list[str] = []
    if not isinstance(payload, dict):
        return facts
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        if result is not None:
            facts.append(str(result))
        return facts
    keys = ("email", "name", "plan", "to", "subject", "condition", "temp_c",
            "count", "path", "city", "value", "doc_id", "event_id", "seats")

    def _collect(obj: dict) -> None:
        for k in keys:
            if k in obj and obj[k] is not None and not isinstance(obj[k], bool):
                s = str(obj[k]).strip()
                if s:
                    facts.append(s)

    _collect(result)
    # Walk dicts one extra level (and one more under those) so owner.email
    # and billing.plan on crm payloads reach grounding.
    for v in result.values():
        if isinstance(v, dict):
            _collect(v)
            for v2 in v.values():
                if isinstance(v2, dict):
                    _collect(v2)
    if "exists" in result and isinstance(result["exists"], bool):
        facts.append("exists" if result["exists"] else "does not exist")
    rows = result.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                for v in row.values():
                    if v is not None and not isinstance(v, bool):
                        facts.append(str(v))
    for list_key in ("hits", "events"):
        items = result.get(list_key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for k in ("doc_id", "event_id", "title"):
                        if k in item and item[k] is not None:
                            s = str(item[k]).strip()
                            if s:
                                facts.append(s)
    err = payload.get("error")
    if isinstance(err, dict):
        if err.get("code"):
            facts.append(str(err["code"]))
        if err.get("message"):
            facts.append(str(err["message"]))
    return facts


def _facts_from_step(step: dict) -> list[str]:
    """Key facts from a step payload plus argument strings."""
    out = _key_facts_from_payload(step.get("result") or {})
    for v in (step.get("args") or {}).values():
        if isinstance(v, str) and len(v.strip()) >= 3:
            out.append(v.strip())
        elif not isinstance(v, bool) and v is not None:
            sv = str(v)
            if len(sv) >= 2:
                out.append(sv)
    return out


def validate_answer_grounding(traj: dict, final: str) -> str | None:
    """Reject finals that invent emails/numbers/names not in tool results or prompt.

    Also require at least one key fact from the last successful tool result
    (or from the last step if the chain ended on an expected error).
    """
    if not (final or "").strip():
        return "empty final answer"
    corpus = _corpus(traj)
    corpus_l = corpus.lower()
    final_l = final.lower()

    for email in EMAIL_RE.findall(final):
        if email.lower() not in corpus_l:
            return f"ungrounded email: {email}"

    corpus_nums = set(NUMBER_RE.findall(corpus))
    for num in NUMBER_RE.findall(final):
        if _significant_number(num) and num not in corpus_nums:
            return f"ungrounded number: {num}"

    for name in KNOWN_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", final) and name.lower() not in corpus_l:
            return f"ungrounded name: {name}"

    for email in KNOWN_EMAILS:
        if email.lower() in final_l and email.lower() not in corpus_l:
            return f"ungrounded registry email: {email}"

    for oid in OPAQUE_ID_RE.findall(final):
        if oid not in corpus:
            return f"ungrounded opaque id: {oid}"

    steps = traj.get("steps") or []
    if not steps:
        return "no steps to ground against"
    last_success = None
    for s in reversed(steps):
        if s.get("result", {}).get("status") == 200:
            last_success = s
            break

    facts: list[str] = []
    if last_success is not None:
        facts.extend(_facts_from_step(last_success))
    facts.extend(_facts_from_step(steps[-1]))
    substantial = []
    seen = set()
    for f in facts:
        fl = f.lower()
        if fl in ("true", "false"):
            continue
        if len(f) >= 2 and fl not in seen:
            seen.add(fl)
            substantial.append(f)
    if not substantial:
        return None
    filtered = [f for f in substantial if not _is_opaque_id(f)]
    if not filtered:
        return None
    matched = False
    for f in filtered:
        fl = f.lower()
        if fl in final_l:
            matched = True
            break
        if f == "does not exist" and any(
            p in final_l for p in ("not exist", "doesn't exist", "missing", "absent", "not present")
        ):
            matched = True
            break
        if f == "exists" and any(p in final_l for p in ("exists", "present", "found", "is on disk")):
            matched = True
            break
    if not matched:
        return f"final missing key fact from last result (expected one of {filtered[:6]})"
    return None


def assemble_trace(traj: dict, teacher_meta: dict, parsed: dict,
                   sys_prompt: str) -> tuple[dict | None, str | None]:
    """Assemble the final training trace.

    Returns (trace, None) or (None, error). Missing thoughts are rejected,
    never padded. The 'Now provide the final answer...' user turn is NOT
    written into exported messages (it will not exist at inference).
    """
    n = len(traj["steps"])
    thoughts = list(parsed.get("thoughts") or [])
    if len(thoughts) < n:
        return None, f"missing thoughts: got {len(thoughts)} need {n}"
    thoughts = thoughts[:n]

    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": traj["prompt"]}]

    for i, s in enumerate(traj["steps"]):
        call = {"name": s["tool"], "arguments": s["args"]}
        call_json = json.dumps([call], ensure_ascii=False)
        content = f"<thought>\n{thoughts[i]}\n</thought>\n<tool_call>\n{call_json}\n</tool_call>"
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "tool", "content": json.dumps(s["result"], ensure_ascii=False)})

    # Final assistant turn immediately after the last tool result.
    msgs.append({"role": "assistant", "content": parsed["final"]})

    trace = {
        "seed_class": "agentic",
        "prompt": traj["prompt"],
        "teacher": teacher_meta["teacher"],
        "teacher_mode": teacher_meta["mode"],
        "plan_template": traj.get("plan_index", -1),
        "plan_id": traj.get("plan_id"),
        "plan_skills": traj.get("skills") or [],
        "plan_tier": traj.get("tier"),
        "vars": traj.get("vars") or {},
        "distill_version": DISTILL_VERSION,
        "forge_spec": "0.2",
        "messages": msgs,
        "chain_steps": [
            {"tool": s["tool"], "args": s["args"], "result": s["result"],
             "expect": s.get("expect", "success")}
            for s in traj["steps"]
        ],
    }
    return trace, None


def validate_prose_trace(trace: dict) -> str | None:
    """Final gate on the assembled trace. Returns error string or None."""
    msgs = trace.get("messages") or []
    if not msgs or msgs[0]["role"] != "system":
        return "no system message"
    if trace.get("distill_version") != DISTILL_VERSION:
        return f"missing or wrong distill_version (want {DISTILL_VERSION})"
    prev = None
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content") or ""
        if role not in ("system", "user", "assistant", "tool"):
            return f"bad role at {i}"
        if NUDGE_TEXT in content:
            return "final-answer nudge leaked into training messages"
        if prev == "assistant" and role not in ("tool", "user", "assistant"):
            return f"assistant not followed by tool/user at {i}"
        if prev == "tool" and role not in ("user", "assistant", "tool"):
            return f"tool not followed by user/assistant at {i}"
        prev = role
    # every assistant tool turn must have valid JSON calls
    n_tool_turns = 0
    for m in msgs:
        if m["role"] == "assistant" and "<tool_call>" in m["content"]:
            n_tool_turns += 1
            tm = re.search(r"<tool_call>(.*?)</tool_call>", m["content"], re.S)
            if not tm:
                return "assistant has <tool_call> without closing tag"
            try:
                calls = json.loads(tm.group(1).strip())
            except Exception:
                return "malformed <tool_call> JSON"
            if not (isinstance(calls, list) and calls and isinstance(calls[0], dict) and "name" in calls[0]):
                return "tool_call not a list of {name, arguments}"
            if "<thought>" not in m["content"]:
                return "tool turn missing <thought>"
    if n_tool_turns < 1:
        return "no tool-calling assistant turns"
    # every tool message follows an assistant with tool_call
    for i, m in enumerate(msgs):
        if m["role"] == "tool" and i > 0 and msgs[i-1]["role"] != "assistant":
            return "tool message not preceded by assistant"
    if msgs[-1]["role"] != "assistant":
        return "trace does not end with assistant final answer"
    if "<tool_call>" in msgs[-1]["content"]:
        return "final assistant turn still contains tool_call"
    return None
