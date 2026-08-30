"""lineage.py - run lineage ledger for Distillation Studio (sidecar JSONL).

Kept traces get one lineage row; dropped attempts get one reject row.
No messages, chain_steps, vars, or API keys in either sidecar.
"""
from __future__ import annotations

import hashlib
import time

LINEAGE_SPEC = "1.0"

PUBLIC_LINEAGE_KEYS = (
    "lineage_id",
    "traj_hash",
    "plan_id",
    "plan_tier",
    "plan_skills",
    "tokens_in",
    "tokens_out",
    "eval",
    "lineage_spec",
    "kept",
)


def lineage_id(traj_hash: str, teacher: str, distill_version: str, forge_spec: str) -> str:
    """sha256 hex of those four fields joined by '\\n', utf-8."""
    payload = "\n".join(
        [str(traj_hash or ""), str(teacher or ""), str(distill_version or ""), str(forge_spec or "")]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reject_lineage_id(traj_hash: str, route: str, reason: str, attempt_seq: int) -> str:
    """sha256 of traj_hash+route+reason+attempt_seq, newline-joined."""
    payload = "\n".join(
        [str(traj_hash or ""), str(route or ""), str(reason or ""), str(int(attempt_seq))]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def kept_record(
    trace: dict,
    *,
    tokens_in: int,
    tokens_out: int,
    curriculum_mode: str,
    seed: int,
) -> dict:
    """Sidecar row for a written trace."""
    teacher = str(trace.get("teacher") or "")
    traj_hash = str(trace.get("traj_hash") or "")
    distill_version = str(trace.get("distill_version") or "")
    forge_spec = str(trace.get("forge_spec") or "")
    lid = trace.get("lineage_id") or lineage_id(
        traj_hash, teacher, distill_version, forge_spec
    )
    ev = trace.get("eval")
    return {
        "lineage_spec": LINEAGE_SPEC,
        "lineage_id": lid,
        "traj_hash": traj_hash,
        "plan_id": trace.get("plan_id"),
        "plan_tier": trace.get("plan_tier"),
        "plan_skills": list(trace.get("plan_skills") or []),
        "teacher": teacher,
        "teacher_thoughts": trace.get("teacher_thoughts"),
        "teacher_answer": trace.get("teacher_answer"),
        "teacher_mode": trace.get("teacher_mode"),
        "distill_version": distill_version,
        "forge_spec": forge_spec,
        "curriculum_mode": curriculum_mode,
        "seed": int(seed),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "eval": dict(ev) if isinstance(ev, dict) else None,
        "repaired": bool(isinstance(ev, dict) and ev.get("repaired")),
        "dpo_pair_id": trace.get("dpo_pair_id"),
        "kept": True,
    }


def reject_record(
    *,
    prov: str,
    model: str,
    plan_id,
    traj_hash,
    reason: str,
    http,
    error: str,
    attempt_seq: int,
) -> dict:
    """Sidecar row for a dropped attempt. No messages, no chain_steps, no API keys.

    One row per attempt; never deduplicated. lineage_id includes attempt_seq
    so retries of the same traj/route/reason stay distinct.
    """
    teacher = f"{prov}/{model}"
    seq = int(attempt_seq)
    return {
        "lineage_spec": LINEAGE_SPEC,
        "lineage_id": reject_lineage_id(str(traj_hash or ""), teacher, reason, seq),
        "traj_hash": traj_hash,
        "plan_id": plan_id,
        "teacher": teacher,
        "reason": reason,
        "http": http,
        "error": (error or "")[:200],
        "attempt_seq": seq,
        "ts": time.time(),
        "kept": False,
    }


def summarize(
    kept: list[dict],
    rejected: list[dict],
    token_usage: dict | None = None,
) -> dict:
    """Aggregates for eval_card['observability']."""
    token_usage = token_usage or {}
    n = len(kept)
    inp = int(token_usage.get("input") or 0)
    outp = int(token_usage.get("output") or 0)
    reasons: dict[str, int] = {}
    for row in rejected:
        key = str(row.get("reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    lids = {row.get("lineage_id") for row in kept if row.get("lineage_id")}
    return {
        "n_kept": n,
        "n_rejects_files": len(rejected),
        "reject_reasons": reasons,
        "tokens_per_kept_trace": {
            "input": (inp / n) if n else 0,
            "output": (outp / n) if n else 0,
        },
        "lineage_spec": LINEAGE_SPEC,
        "n_lineage_ids": len(lids),
    }


def map_reject_reason(res: dict, *, gate_err: str | None = None) -> str:
    """Map Distiller result / append failure to a reject reason string."""
    if gate_err is not None:
        return f"gate:{gate_err}"
    http = res.get("http")
    if http in ("FORMAT", "PLAN", "GROUNDING", "VERIFY"):
        return str(http).lower()
    if isinstance(http, int):
        return f"http:{http}"
    return "transport"
