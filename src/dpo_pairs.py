#!/usr/bin/env python3
"""dpo_pairs.py - preference pairs from already-assembled chosen traces.

Mutate AFTER the chosen trace is assembled. No second teacher call.
Rejected thoughts are reused; only the action or the final claim changes.

CLI:
  python src/dpo_pairs.py --input data/raw --out data/raw/dpo_pairs_offline.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_plans import GUESSED_EMAILS, PLANS, VAR_POOLS, trajectory_hash
from eval_card import (
    FORGE_SPEC,
    _as_args,
    _parse_tool_calls,
    _parse_tool_payload,
    gate_chain,
    gate_dependency_fidelity,
    gate_grounding,
    gate_prose,
    load_traces,
    reconstruct_steps,
)
from mock_tools import execute_one
from prose_writer import DISTILL_VERSION, KNOWN_EMAILS

MUTATIONS = ("unregistered_recipient", "fabricated_final")

# Reserved sentinels outside ALL pools. Static invariant, asserted at import:
# DPO_FAKE_EMAIL not in KNOWN_EMAILS and not in any VAR_POOLS value;
# DPO_FAKE_NUMBER not in any plan prompt or pool. They can never collide
# with corpus content by construction. No runtime probing, no fallback.
DPO_FAKE_EMAIL = "no.such.user.0000@invalid.invalid"
DPO_FAKE_NUMBER = "424243"

_FABRICATED_SENTENCE = (
    f" Also notify {DPO_FAKE_EMAIL} quoting reference {DPO_FAKE_NUMBER}."
)


def _flatten_pool_values(pools: dict) -> list:
    out: list = []
    for v in pools.values():
        if isinstance(v, (list, tuple, set)):
            for item in v:
                if isinstance(item, (list, tuple, set)):
                    out.extend(item)
                else:
                    out.append(item)
        else:
            out.append(v)
    return out


def _assert_sentinels_outside_pools() -> None:
    pools = _flatten_pool_values(VAR_POOLS)
    pool_strs = [str(v) for v in pools]
    assert DPO_FAKE_EMAIL not in KNOWN_EMAILS
    assert DPO_FAKE_EMAIL not in GUESSED_EMAILS
    assert DPO_FAKE_EMAIL not in pools
    assert DPO_FAKE_EMAIL not in pool_strs
    for s in pool_strs:
        assert DPO_FAKE_NUMBER not in s
        assert DPO_FAKE_EMAIL not in s
    for plan in PLANS:
        prompt = plan.get("prompt") or ""
        blob = json.dumps(plan, ensure_ascii=False, default=str)
        assert DPO_FAKE_EMAIL not in prompt
        assert DPO_FAKE_EMAIL not in blob
        assert DPO_FAKE_NUMBER not in prompt
        assert DPO_FAKE_NUMBER not in blob


_assert_sentinels_outside_pools()


def _replace_tool_call(content: str, calls: list[dict]) -> str:
    m = re.search(r"<tool_call>(.*?)</tool_call>", content or "", re.S)
    if not m:
        return content
    new = json.dumps(calls, ensure_ascii=False)
    return content[: m.start()] + f"<tool_call>\n{new}\n</tool_call>" + content[m.end():]


def _steps_for(trace: dict, traj: dict | None = None) -> list[dict]:
    if traj and traj.get("steps"):
        return list(traj["steps"])
    return reconstruct_steps(trace)


def _four_gates(trace: dict, steps: list[dict] | None = None) -> dict:
    seq = list(steps or []) or reconstruct_steps(trace)
    return {
        "prose": gate_prose(trace),
        "grounding": gate_grounding(trace, seq),
        "chain": gate_chain(seq),
        "fidelity": gate_dependency_fidelity(trace, seq),
    }


def _all_pass(gates: dict) -> bool:
    return all(gates.get(k) is None for k in ("prose", "grounding", "chain", "fidelity"))


def _has_successful_send_email(trace: dict) -> bool:
    for s in reconstruct_steps(trace):
        if s.get("tool") != "send_email":
            continue
        if (s.get("result") or {}).get("status") == 200:
            return True
    msgs = trace.get("messages") or []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<tool_call>" not in content:
            continue
        calls = _parse_tool_calls(content)
        if not calls or not any(c.get("name") == "send_email" for c in calls):
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if not nxt or nxt.get("role") != "tool":
            continue
        payload = _parse_tool_payload(nxt.get("content") or "") or {}
        if payload.get("status") == 200:
            return True
    return False


def _chosen_hash(chosen: dict, traj: dict) -> str:
    if chosen.get("traj_hash"):
        return str(chosen["traj_hash"])
    return trajectory_hash({
        "plan_id": chosen.get("plan_id", chosen.get("plan_template")),
        "vars": chosen.get("vars") or traj.get("vars") or {},
        "steps": _steps_for(chosen, traj),
    })


def _stamp_traj_hash(trace: dict) -> None:
    steps = reconstruct_steps(trace)
    trace["traj_hash"] = trajectory_hash({
        "plan_id": trace.get("plan_id", trace.get("plan_template")),
        "vars": trace.get("vars") or {},
        "steps": steps,
    })


def pair_id(chosen_hash, mutation) -> str:
    return hashlib.sha256(f"{chosen_hash}:{mutation}".encode("utf-8")).hexdigest()


def mutate_unregistered_recipient(trace: dict) -> dict | None:
    """Deep-copy. Last successful send_email goes to GUESSED_EMAILS[0].

    Tool payload becomes the real execute_one error (UNREGISTERED_ADDRESS).
    Thoughts stay unchanged. chain_steps for that send_email are updated.
    Returns None if there is no successful send_email.
    """
    out = copy.deepcopy(trace)
    msgs = out.get("messages") or []
    last_i = None
    last_calls = None
    last_args = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<tool_call>" not in content:
            continue
        calls = _parse_tool_calls(content)
        if not calls:
            continue
        send = None
        for c in calls:
            if c.get("name") == "send_email":
                send = c
        if send is None:
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if not nxt or nxt.get("role") != "tool":
            continue
        payload = _parse_tool_payload(nxt.get("content") or "") or {}
        if payload.get("status") != 200:
            continue
        last_i = i
        last_calls = copy.deepcopy(calls)
        last_args = dict(_as_args(send.get("arguments")))

    if last_i is None:
        return None

    mutated_args = dict(last_args)
    mutated_args["to"] = GUESSED_EMAILS[0]
    err_payload = execute_one("send_email", mutated_args)

    new_calls = []
    for c in last_calls:
        c = copy.deepcopy(c)
        if c.get("name") == "send_email":
            args = dict(_as_args(c.get("arguments")))
            args["to"] = GUESSED_EMAILS[0]
            c["arguments"] = args
        new_calls.append(c)
    msgs[last_i]["content"] = _replace_tool_call(msgs[last_i]["content"], new_calls)
    msgs[last_i + 1]["content"] = json.dumps(err_payload, ensure_ascii=False)

    cs = out.get("chain_steps")
    updated = False
    if isinstance(cs, list):
        last_j = None
        for j, s in enumerate(cs):
            if s.get("tool") != "send_email":
                continue
            if (s.get("result") or {}).get("status") == 200:
                last_j = j
        if last_j is not None:
            step = cs[last_j]
            args = dict(_as_args(step.get("args")))
            args["to"] = GUESSED_EMAILS[0]
            step["args"] = args
            step["result"] = err_payload
            # Keep original expect (success). An error with expect=success
            # is what makes validate_chain fail. Do not retag as error.
            updated = True
    if not updated:
        rebuilt = reconstruct_steps({k: v for k, v in out.items() if k != "chain_steps"})
        for s in reversed(rebuilt):
            if s.get("tool") == "send_email":
                s["expect"] = "success"
                break
        out["chain_steps"] = rebuilt

    _stamp_traj_hash(out)
    return out


def mutate_fabricated_final(trace: dict, traj: dict) -> dict:
    """Deep-copy. Append one sentence with DPO_FAKE_EMAIL and DPO_FAKE_NUMBER.

    Grounding fails: neither sentinel is in the corpus.
    """
    out = copy.deepcopy(trace)
    _ = traj
    msgs = out.get("messages") or []
    if not msgs:
        return out
    last = msgs[-1]
    last["content"] = (last.get("content") or "") + _FABRICATED_SENTENCE
    return out


def _traj_from_trace(trace: dict) -> dict:
    return {
        "prompt": trace.get("prompt", ""),
        "steps": reconstruct_steps(trace),
        "plan_id": trace.get("plan_id"),
        "vars": trace.get("vars") or {},
    }


def build_pair(chosen: dict, traj: dict, rng) -> dict | None:
    """Prefer unregistered_recipient when a successful send_email exists.

    Else fabricated_final. Return None if chosen fails a gate or if the
    rejected side still passes all four.
    """
    _ = rng
    traj = traj or _traj_from_trace(chosen)
    chosen_steps = _steps_for(chosen, traj)
    chosen_gates = _four_gates(chosen, chosen_steps)
    if not _all_pass(chosen_gates):
        return None

    mutation = None
    rejected = None
    if _has_successful_send_email(chosen):
        rejected = mutate_unregistered_recipient(chosen)
        mutation = "unregistered_recipient"
        if rejected is not None:
            rej_gates = _four_gates(rejected)
            if _all_pass(rej_gates):
                rejected = None
    if rejected is None:
        rejected = mutate_fabricated_final(chosen, traj)
        mutation = "fabricated_final"

    rej_gates = _four_gates(rejected)
    if _all_pass(rej_gates):
        return None

    h = _chosen_hash(chosen, traj)
    chosen_out = copy.deepcopy(chosen)
    if not chosen_out.get("traj_hash"):
        chosen_out["traj_hash"] = h
    if not rejected.get("traj_hash"):
        _stamp_traj_hash(rejected)

    return {
        "pair_id": pair_id(h, mutation),
        "forge_spec": FORGE_SPEC,
        "distill_version": DISTILL_VERSION,
        "mutation": mutation,
        "chosen": chosen_out,
        "rejected": rejected,
        "gates": {
            "chosen_prose": chosen_gates["prose"],
            "chosen_grounding": chosen_gates["grounding"],
            "chosen_chain": chosen_gates["chain"],
            "chosen_fidelity": chosen_gates["fidelity"],
            "rejected_prose": rej_gates["prose"],
            "rejected_grounding": rej_gates["grounding"],
            "rejected_chain": rej_gates["chain"],
            "rejected_fidelity": rej_gates["fidelity"],
        },
    }


def _rejected_send_tos(trace: dict) -> list[str]:
    tos: list[str] = []
    for s in reconstruct_steps(trace):
        if s.get("tool") != "send_email":
            continue
        to = _as_args(s.get("args")).get("to")
        if isinstance(to, str) and to:
            tos.append(to)
    return tos


def sft_paths_would_not_contain_rejected(out_dir, rejected: dict | None = None) -> bool:
    """True iff traces_*.jsonl under out_dir do not contain the rejected trace.

    load_traces expands a directory to traces_*.jsonl only, so dpo_pairs_*.jsonl
    in the same folder is ignored. Distinctive rejected markers (DPO fake
    email/number, guessed address as the only send) must not appear there.
    """
    traces = load_traces([Path(out_dir)])
    if rejected is None:
        return True
    rej_msgs = rejected.get("messages")
    rej_blob = json.dumps(rejected, ensure_ascii=False)
    fake_in_rej = DPO_FAKE_EMAIL in rej_blob or DPO_FAKE_NUMBER in rej_blob
    rej_tos = _rejected_send_tos(rejected)
    guessed_only = bool(rej_tos) and all(t == GUESSED_EMAILS[0] for t in rej_tos)
    for t in traces:
        if t.get("messages") == rej_msgs:
            return False
        tblob = json.dumps(t, ensure_ascii=False)
        if fake_in_rej and (DPO_FAKE_EMAIL in tblob or DPO_FAKE_NUMBER in tblob):
            return False
        if guessed_only:
            tos = _rejected_send_tos(t)
            if tos and all(x == GUESSED_EMAILS[0] for x in tos):
                return False
    return True


def pairs_from_traces(traces: list[dict], rng: random.Random, dpo_rate: float = 1.0) -> list[dict]:
    out: list[dict] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        if dpo_rate < 1.0 and rng.random() >= dpo_rate:
            continue
        traj = _traj_from_trace(trace)
        pair = build_pair(trace, traj, rng)
        if pair is None:
            continue
        out.append(pair)
    return out


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--dpo-rewrite-prose" in raw:
        print("--dpo-rewrite-prose is out of v0.2", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(
        description="Build DPO pairs from traces_*.jsonl by mutating assembled chosen traces"
    )
    ap.add_argument(
        "--input", action="append", required=True, type=Path,
        help="trace jsonl file, or a directory expanded to traces_*.jsonl only",
    )
    ap.add_argument("--out", required=True, type=Path, help="write dpo pair jsonl here")
    ap.add_argument(
        "--dpo-rate", type=float, default=1.0,
        help="probability of building a pair per chosen trace (default 1.0)",
    )
    args = ap.parse_args(raw)

    traces = load_traces(list(args.input))
    rng = random.Random(0)
    pairs = pairs_from_traces(traces, rng, dpo_rate=args.dpo_rate)

    dest = args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {dest} n_pairs={len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
