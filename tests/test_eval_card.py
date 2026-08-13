from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mock_tools import EMAIL_BY_USER, execute_one
from eval_card import (
    CMD_EVAL_CARD,
    CMD_STRESS_300,
    COVERAGE_NOTE,
    compute_card,
    gate_dependency_fidelity,
    load_traces,
    reconstruct_steps,
    stamp_eval,
)

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
EVAL = str(REPO / "src" / "eval_card.py")
FIXTURES = REPO / "tests" / "fixtures"
MINI = FIXTURES / "mini_traces.jsonl"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [PY, EVAL, *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env=env,
    )


def _copy_fixture_line(i: int = 0) -> dict:
    lines = MINI.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[i])


def test_load_traces_dir_ignores_non_trace_files(tmp_path: Path):
    good = {"prompt": "only-this", "messages": []}
    (tmp_path / "traces_ok.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")
    (tmp_path / "checkpoint_x.json").write_text("{}", encoding="utf-8")
    (tmp_path / "dpo_pairs_x.jsonl").write_text(json.dumps({"chosen": {}}) + "\n", encoding="utf-8")
    (tmp_path / "eval_card.json").write_text("{}", encoding="utf-8")
    (tmp_path / "holdout_plan_ids.json").write_text("[]", encoding="utf-8")
    (tmp_path / "mini_traces.jsonl").write_text(json.dumps({"prompt": "nope"}) + "\n", encoding="utf-8")
    traces = load_traces([tmp_path])
    assert len(traces) == 1
    assert traces[0]["prompt"] == "only-this"


def test_compute_card_gates_clean_on_mini_traces():
    traces = load_traces([FIXTURES])
    assert len(traces) >= 3
    card = compute_card(traces, extra={"input_paths": [str(FIXTURES)]})
    g = card["gates"]
    assert g["validate_prose_trace_pass"] == 1.0
    assert g["validate_answer_grounding_pass"] == 1.0
    assert g["validate_chain_pass"] == 1.0
    assert g["dependency_fidelity_pass"] == 1.0
    assert g["nudge_leak_rate"] == 0.0
    assert g["malformed_tool_call_rate"] == 0.0
    assert card["forge_spec"] == "0.2"
    assert card["distill_version"] == "reversed-v2"
    assert card["n_traces"] == len(traces)
    assert set(card["structure"]["round_depth_histogram"]) >= {"2", "3", "4", "5"}
    assert card["skills"]["coverage_note"] == COVERAGE_NOTE
    assert card["commands"]["eval_card"] == CMD_EVAL_CARD
    assert card["commands"]["stress_300"] == CMD_STRESS_300
    assert set(card.keys()) == {
        "forge_spec", "distill_version", "created_at", "input_paths", "n_traces",
        "gates", "structure", "skills", "teachers", "holdout_plan_ids", "tokens",
        "commands",
    }


def test_reconstruct_steps_without_chain_steps():
    original = _copy_fixture_line(0)
    assert original.get("chain_steps")
    expected = [
        (s["tool"], s["args"], s["result"])
        for s in original["chain_steps"]
    ]
    stripped = json.loads(json.dumps(original))
    del stripped["chain_steps"]
    rebuilt = reconstruct_steps(stripped)
    got = [(s["tool"], s["args"], s["result"]) for s in rebuilt]
    assert got == expected
    assert all(s.get("expect") in ("success", "error") for s in rebuilt)


def test_fidelity_fails_on_guessed_send_email_to():
    alice = execute_one("get_user", {"user_id": 42})
    bob_email = EMAIL_BY_USER[7]
    send_args = {"to": bob_email, "subject": "Hi", "body": "Hello."}
    send = execute_one("send_email", send_args)
    assert send.get("status") == 200
    trace = {
        "prompt": "Look up user 42 and email Bob.",
        "distill_version": "reversed-v2",
        "messages": [
            {"role": "system", "content": "You are a tool-calling agent."},
            {"role": "user", "content": "Look up user 42 and email Bob."},
            {
                "role": "assistant",
                "content": (
                    "<thought>\nLook up user 42.\n</thought>\n"
                    '<tool_call>\n[{"name": "get_user", "arguments": {"user_id": 42}}]\n</tool_call>'
                ),
            },
            {"role": "tool", "content": json.dumps(alice, ensure_ascii=False)},
            {
                "role": "assistant",
                "content": (
                    "<thought>\nSend to a guessed registered address.\n</thought>\n"
                    '<tool_call>\n'
                    + json.dumps([{"name": "send_email", "arguments": send_args}])
                    + "\n</tool_call>"
                ),
            },
            {"role": "tool", "content": json.dumps(send, ensure_ascii=False)},
            {"role": "assistant", "content": f"Sent to {bob_email}."},
        ],
    }
    steps = reconstruct_steps(trace)
    err = gate_dependency_fidelity(trace, steps)
    assert err is not None
    assert "not learned" in err or "guess" in err.lower() or "earlier" in err


def test_cli_fixtures_require_gates_exits_0():
    r = _run(["--input", str(FIXTURES), "--require-gates"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_stamp_eval_writes_forge_spec_and_eval():
    trace = _copy_fixture_line(2)
    trace.pop("eval", None)
    trace.pop("forge_spec", None)
    trace.pop("plan_tier", None)
    traj = {
        "prompt": trace.get("prompt", ""),
        "steps": trace.get("chain_steps") or [],
        "skills": trace.get("plan_skills") or ["branch"],
        "tier": "hard",
    }
    stamp_eval(trace, traj, repaired=False, cross_teacher=False)
    assert trace["forge_spec"] == "0.2"
    assert trace["plan_tier"] == "hard"
    ev = trace["eval"]
    assert ev["format_ok"] is True
    assert ev["grounding_ok"] is True
    assert ev["chain_ok"] is True
    assert ev["dependency_ok"] is True
    assert ev["n_rounds"] >= 2
    assert ev["n_tool_calls"] >= 2
    assert ev["skills"] == ["branch"]
    assert ev["tier"] == "hard"
    assert ev["repaired"] is False
    assert ev["cross_teacher"] is False


def test_require_gates_fails_on_nudge_leak(tmp_path: Path):
    t = _copy_fixture_line(0)
    t["messages"].append({
        "role": "user",
        "content": "Now provide the final answer to the user's original request.",
    })
    p = tmp_path / "traces_leaky.jsonl"
    p.write_text(json.dumps(t, ensure_ascii=False) + "\n", encoding="utf-8")
    r = _run(["--input", str(p), "--require-gates"])
    assert r.returncode == 1, r.stdout + r.stderr


def test_empty_traces_pass_rates_are_one():
    card = compute_card([])
    assert card["n_traces"] == 0
    assert card["gates"]["validate_prose_trace_pass"] == 1.0
    assert card["gates"]["validate_answer_grounding_pass"] == 1.0
    assert card["gates"]["validate_chain_pass"] == 1.0
    assert card["gates"]["dependency_fidelity_pass"] == 1.0
    assert card["gates"]["nudge_leak_rate"] == 0.0
    assert card["gates"]["malformed_tool_call_rate"] == 0.0
    assert card["structure"]["send_email_learned_address_rate"] == 1.0
