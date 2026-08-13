from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from agentic_plans import GUESSED_EMAILS, PLANS, build_chain, trajectory_hash
from eval_card import (
    gate_chain,
    gate_dependency_fidelity,
    gate_grounding,
    gate_prose,
    load_traces,
    reconstruct_steps,
    stamp_eval,
)
from prose_writer import KNOWN_EMAILS, assemble_trace, validate_answer_grounding

from dpo_pairs import (
    DPO_FAKE_EMAIL,
    DPO_FAKE_NUMBER,
    MUTATIONS,
    build_pair,
    mutate_fabricated_final,
    mutate_unregistered_recipient,
    pair_id,
    sft_paths_would_not_contain_rejected,
)

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
DPO = str(REPO / "src" / "dpo_pairs.py")
FIXTURES = REPO / "tests" / "fixtures"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [PY, DPO, *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env=env,
    )


def _plan_index(plan_id: str) -> int:
    for i, p in enumerate(PLANS):
        if p["id"] == plan_id:
            return i
    raise KeyError(plan_id)


def _canned_parsed(traj: dict) -> dict:
    thoughts = [
        f"I will call {s['tool']} using the task and any prior tool results."
        for s in traj["steps"]
    ]
    last_success = None
    for s in reversed(traj["steps"]):
        if (s.get("result") or {}).get("status") == 200:
            last_success = s
            break
    last = last_success or traj["steps"][-1]
    final = "Completed the request. Recorded result: " + json.dumps(
        last.get("result"), ensure_ascii=False
    )
    return {"thoughts": thoughts, "final": final}


def _assemble(plan_id: str, seed: int = 0) -> tuple[dict, dict]:
    traj = build_chain(random.Random(seed), plan_index=_plan_index(plan_id))
    parsed = _canned_parsed(traj)
    gerr = validate_answer_grounding(traj, parsed["final"])
    assert gerr is None, gerr
    trace, err = assemble_trace(
        traj,
        {"teacher": "fixture-local/dpo-test", "mode": "concise"},
        parsed,
        "You are a tool-calling agent.",
    )
    assert err is None and trace is not None
    trace["traj_hash"] = trajectory_hash(traj)
    stamp_eval(trace, traj)
    ev = trace["eval"]
    assert ev["format_ok"] is True
    assert ev["grounding_ok"] is True
    assert ev["chain_ok"] is True
    assert ev["dependency_ok"] is True
    return trace, traj


def _thoughts_of(trace: dict) -> list[str]:
    out = []
    for m in trace.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if "<thought>" not in content:
            continue
        start = content.find("<thought>")
        end = content.find("</thought>")
        if start >= 0 and end > start:
            out.append(content[start:end + len("</thought>")])
    return out


def _assert_four_gate_split(pair: dict) -> None:
    g = pair["gates"]
    assert g["chosen_prose"] is None
    assert g["chosen_grounding"] is None
    assert g["chosen_chain"] is None
    assert g["chosen_fidelity"] is None
    rejected_hits = [
        g["rejected_prose"],
        g["rejected_grounding"],
        g["rejected_chain"],
        g["rejected_fidelity"],
    ]
    assert any(e is not None for e in rejected_hits)

    chosen = pair["chosen"]
    rejected = pair["rejected"]
    c_steps = reconstruct_steps(chosen)
    r_steps = reconstruct_steps(rejected)
    assert gate_prose(chosen) is None
    assert gate_grounding(chosen, c_steps) is None
    assert gate_chain(c_steps) is None
    assert gate_dependency_fidelity(chosen, c_steps) is None
    r_fail = [
        gate_prose(rejected),
        gate_grounding(rejected, r_steps),
        gate_chain(r_steps),
        gate_dependency_fidelity(rejected, r_steps),
    ]
    assert any(e is not None for e in r_fail)


def test_import_sentinels_outside_known_emails():
    assert DPO_FAKE_EMAIL not in KNOWN_EMAILS
    assert DPO_FAKE_EMAIL not in GUESSED_EMAILS
    assert DPO_FAKE_NUMBER == "424243"
    assert MUTATIONS == ("unregistered_recipient", "fabricated_final")


def test_unregistered_recipient_on_welcome():
    chosen, traj = _assemble("user-email-welcome")
    pair = build_pair(chosen, traj, random.Random(0))
    assert pair is not None
    assert pair["mutation"] == "unregistered_recipient"
    assert pair["forge_spec"] == "0.2"
    assert pair["distill_version"] == "reversed-v2"
    assert pair["chosen"]["seed_class"] == "agentic"
    assert pair["rejected"]["seed_class"] == "agentic"
    assert pair["chosen"]["messages"]
    assert pair["rejected"]["messages"]
    _assert_four_gate_split(pair)
    assert _thoughts_of(pair["chosen"]) == _thoughts_of(pair["rejected"])
    guessed = GUESSED_EMAILS[0]
    rej_blob = json.dumps(pair["rejected"], ensure_ascii=False)
    ch_blob = json.dumps(pair["chosen"], ensure_ascii=False)
    assert guessed in rej_blob
    send_tos = []
    for s in reconstruct_steps(pair["rejected"]):
        if s.get("tool") == "send_email":
            send_tos.append((s.get("args") or {}).get("to"))
            assert (s.get("result") or {}).get("status") != 200
            err = (s.get("result") or {}).get("error") or {}
            assert err.get("code") == "UNREGISTERED_ADDRESS"
    assert guessed in send_tos
    assert guessed not in ch_blob or guessed not in json.dumps(
        [s.get("args") for s in reconstruct_steps(pair["chosen"]) if s.get("tool") == "send_email"],
        ensure_ascii=False,
    )


def test_fabricated_final_on_bad_id_stop():
    chosen, traj = _assemble("bad-id-stop")
    assert mutate_unregistered_recipient(chosen) is None
    pair = build_pair(chosen, traj, random.Random(0))
    assert pair is not None
    assert pair["mutation"] == "fabricated_final"
    _assert_four_gate_split(pair)
    final = pair["rejected"]["messages"][-1]["content"]
    assert DPO_FAKE_EMAIL in final
    assert DPO_FAKE_NUMBER in final
    assert DPO_FAKE_EMAIL not in pair["chosen"]["messages"][-1]["content"]
    assert pair["gates"]["rejected_grounding"] is not None
    forced = mutate_fabricated_final(chosen, traj)
    assert DPO_FAKE_EMAIL in forced["messages"][-1]["content"]
    assert DPO_FAKE_NUMBER in forced["messages"][-1]["content"]


def test_rejected_never_lands_in_sft_jsonl(tmp_path: Path):
    chosen, traj = _assemble("user-email-welcome")
    pair = build_pair(chosen, traj, random.Random(0))
    assert pair is not None
    (tmp_path / "traces_x.jsonl").write_text(
        json.dumps(chosen, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "dpo_pairs_x.jsonl").write_text(
        json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert sft_paths_would_not_contain_rejected(tmp_path, pair["rejected"]) is True
    traces = load_traces([tmp_path])
    assert len(traces) == 1
    blob = json.dumps(traces, ensure_ascii=False)
    assert GUESSED_EMAILS[0] not in blob or not _only_guessed_sends(traces[0])
    assert DPO_FAKE_EMAIL not in blob
    assert DPO_FAKE_NUMBER not in blob
    assert traces[0].get("messages") != pair["rejected"].get("messages")

    chosen2, traj2 = _assemble("bad-id-stop")
    pair2 = build_pair(chosen2, traj2, random.Random(0))
    assert pair2 is not None
    (tmp_path / "traces_y.jsonl").write_text(
        json.dumps(chosen2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert sft_paths_would_not_contain_rejected(tmp_path, pair2["rejected"]) is True
    sft_blob = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(tmp_path.glob("traces_*.jsonl"))
    )
    assert DPO_FAKE_EMAIL not in sft_blob
    assert DPO_FAKE_NUMBER not in sft_blob


def _only_guessed_sends(trace: dict) -> bool:
    tos = []
    for s in reconstruct_steps(trace):
        if s.get("tool") == "send_email":
            tos.append((s.get("args") or {}).get("to"))
    return bool(tos) and all(t == GUESSED_EMAILS[0] for t in tos)


def test_pair_id_sha256():
    h = "abc"
    mutation = "fabricated_final"
    assert pair_id(h, mutation) == hashlib.sha256(
        f"{h}:{mutation}".encode("utf-8")
    ).hexdigest()


def test_offline_cli_over_fixtures(tmp_path: Path):
    out = tmp_path / "dpo_pairs_offline.jsonl"
    r = _run(["--input", str(FIXTURES), "--out", str(out), "--dpo-rate", "1.0"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.is_file()
    pairs = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = load_traces([FIXTURES])
    assert traces, "fixtures must expose traces_*.jsonl"
    assert len(pairs) >= 1
    assert len(pairs) <= len(traces)
    muts = {p["mutation"] for p in pairs}
    assert "unregistered_recipient" in muts
    assert "fabricated_final" in muts
    for p in pairs:
        _assert_four_gate_split(p)
        assert p["forge_spec"] == "0.2"
        assert p["distill_version"] == "reversed-v2"
        assert "pair_id" in p
        assert p["chosen"]["messages"]
        assert p["rejected"]["messages"]


def test_dpo_rewrite_prose_exits_2(tmp_path: Path):
    r = _run([
        "--dpo-rewrite-prose",
        "--input", str(FIXTURES),
        "--out", str(tmp_path / "nope.jsonl"),
    ])
    assert r.returncode == 2
    msg = (r.stderr or "") + (r.stdout or "")
    assert "out of v0.2" in msg
