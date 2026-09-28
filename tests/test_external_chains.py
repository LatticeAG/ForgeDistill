"""Offline tests for src/external_chains.py + agentic_plans external lane.

No network: corpus rows are inline fixtures matching the live shapes
(ToolACE conversations list; xLAM query/tools/answers JSON strings).
"""
from __future__ import annotations

import json
import random

import pytest

from agentic_plans import (
    SKILLS,
    build_chain,
    build_external_chain,
    external_skills,
    external_tier,
    trajectory_hash,
    validate_chain,
    validate_external_chain,
)
from external_chains import (
    _parse_args,
    _parse_call_block,
    check_file,
    records_from_toolace_row,
    records_from_xlam_row,
)
from prose_writer import build_prose_prompt


TOOLACE_ROW = {
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "user", "value": "What are the current market trends in the US?"},
        {"from": "assistant",
         "value": '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]'},
        {"from": "tool",
         "value": '[{"name": "Market Trends API", "results": {"trends": ["up"]}}]'},
        {"from": "assistant", "value": "The trend is up."},
        {"from": "user",
         "value": "Get the filings for AAPL and mobility data for 2025-05-17."},
        {"from": "assistant",
         "value": '[SEC Filings(identifier="AAPL"), '
                  'United States Away from Home Mobility API(string="2025-05-17")]'},
        {"from": "tool",
         "value": '[{"name": "SEC Filings", "results": {"filing": "10-K"}}, '
                  '{"name": "United States Away from Home Mobility API", '
                  '"results": {"mobility": 42}}]'},
        {"from": "user", "value": "List all competitions please."},
        {"from": "assistant", "value": "[GetCompetitions()]"},
        {"from": "tool",
         "value": '[{"name": "GetCompetitions", "results": {"count": 3}}]'},
        {"from": "user", "value": "Thanks, that is helpful."},
        {"from": "assistant", "value": "You are welcome."},
    ]
}

XLAM_ROW = {
    "query": "Find the current price of AAPL and convert it to EUR.",
    "tools": json.dumps([
        {"name": "get_price",
         "parameters": {"type": "object", "properties": {
             "symbol": {"type": "string"},
             "currency": {"type": "string"},
             "verbose": {"type": "boolean"},
             "retries": {"type": "integer"}}}},
    ]),
    "answers": json.dumps([
        {"name": "get_price", "arguments": {"symbol": "AAPL", "currency": "USD"}}
    ]),
}


def _row(user_text, *turns):
    return {"conversations": [{"from": "user", "value": user_text}, *turns]}


def test_call_block_single_multi_and_noarg():
    assert _parse_call_block('[A(x=1)]') == [("A", {"x": 1})]
    calls = _parse_call_block('[A(x=1), B(y="s")]')
    assert calls == [("A", {"x": 1}), ("B", {"y": "s"})]
    assert _parse_call_block('[GetCompetitions()]') == [("GetCompetitions", {})]


def test_call_block_arg_types():
    calls = _parse_call_block('[F(a="x, y", b=123, c=true, d=None, e=1.5)]')
    assert calls == [("F", {"a": "x, y", "b": 123, "c": True, "d": None, "e": 1.5})]


def test_call_block_rejects_trailing_junk_and_positional():
    assert _parse_call_block('[A(x=1)] then tell me more') is None
    assert _parse_call_block('[A("positional")]') is None
    assert _parse_call_block('Sure, here is the call: [A(x=1)]') is None
    assert _parse_call_block('[{"name": "A", "results": {}}]') is None
    assert _parse_call_block('plain prose turn') is None
    assert _parse_args('a=1, stray') is None


def test_toolace_segment_splitting_and_rounds():
    recs, drops, no_calls = records_from_toolace_row(TOOLACE_ROW, 7)
    assert not drops
    assert no_calls == 1  # the prose-only segment
    assert [r["record_id"] for r in recs] == [
        "toolace-000007-0", "toolace-000007-1", "toolace-000007-2"]
    assert all(r["source"] == "toolace" for r in recs)
    assert all(r["result_source"] == "recorded" for r in recs)
    seg0, seg1, seg2 = recs
    assert seg0["steps"][0]["tool"] == "Market Trends API"
    assert seg0["steps"][0]["args"] == {
        "trend_type": "MARKET_INDEXES", "country": "us"}
    assert seg0["steps"][0]["result"] == {
        "status": 200, "result": {"trends": ["up"]}}
    assert [s["round"] for s in seg1["steps"]] == [0, 0]  # fanout turn
    assert [s["tool"] for s in seg1["steps"]] == [
        "SEC Filings", "United States Away from Home Mobility API"]
    assert seg1["steps"][1]["args"] == {"string": "2025-05-17"}
    assert seg2["steps"][0]["args"] == {}


def test_toolace_multi_round_segment():
    row = _row(
        "First get competitions, then search the archive for them.",
        {"from": "assistant", "value": "[GetCompetitions()]"},
        {"from": "tool",
         "value": '[{"name": "GetCompetitions", "results": {"count": 2}}]'},
        {"from": "assistant", "value": '[ArchiveSearch(q="cup")]'},
        {"from": "tool",
         "value": '[{"name": "ArchiveSearch", "results": {"hits": []}}]'},
    )
    recs, drops, _ = records_from_toolace_row(row, 0)
    assert not drops and len(recs) == 1
    assert [s["round"] for s in recs[0]["steps"]] == [0, 1]


def test_toolace_drops_unmatched_result():
    row = _row(
        "Check the flux capacitor reading now.",
        {"from": "assistant", "value": '[FluxCapacitor(reading="now")]'},
        {"from": "assistant", "value": "It returned something odd."},
    )
    recs, drops, _ = records_from_toolace_row(row, 0)
    assert recs == [] and drops == {"unmatched_result": 1}


def test_toolace_drops_non_json_result():
    row = _row(
        "Check the flux capacitor reading now.",
        {"from": "assistant", "value": "[FluxCapacitor()]"},
        {"from": "tool", "value": "flux capacitor is humming"},
    )
    recs, drops, _ = records_from_toolace_row(row, 0)
    assert recs == [] and drops == {"result_not_json": 1}


def test_toolace_drops_base64_result():
    row = _row(
        "Render the chart image for the report.",
        {"from": "assistant", "value": '[ChartAPI(kind="bar")]'},
        {"from": "tool",
         "value": '[{"name": "ChartAPI", "results": '
                  '{"png": "data:image/png;base64,iVBORw0KGgo="}}]'},
    )
    recs, drops, _ = records_from_toolace_row(row, 0)
    assert recs == [] and drops == {"base64_payload": 1}


def test_toolace_drops_bad_prompt_lengths():
    short = _row(
        "hi",
        {"from": "assistant", "value": "[Ping()]"},
        {"from": "tool", "value": '[{"name": "Ping", "results": {}}]'},
    )
    long = _row(
        "run " + "x" * 400,
        {"from": "assistant", "value": "[Ping()]"},
        {"from": "tool", "value": '[{"name": "Ping", "results": {}}]'},
    )
    for row in (short, long):
        recs, drops, _ = records_from_toolace_row(row, 0)
        assert recs == [] and drops == {"bad_prompt_len": 1}


def test_toolace_drops_too_many_steps():
    turns = []
    for i in range(7):
        turns.append({"from": "assistant", "value": f"[Step{i}(x={i})]"})
        turns.append({"from": "tool",
                      "value": f'[{{"name": "Step{i}", "results": {{}}}}]'})
    recs, drops, _ = records_from_toolace_row(_row("Do all seven steps.", *turns), 0)
    assert recs == [] and drops == {"too_many_steps": 1}


def test_xlam_record_and_deterministic_synthesis():
    recs_a, drops_a, _ = records_from_xlam_row(XLAM_ROW, 3)
    recs_b, drops_b, _ = records_from_xlam_row(XLAM_ROW, 3)
    assert not drops_a and len(recs_a) == 1
    blob_a = json.dumps(recs_a[0], sort_keys=True)
    blob_b = json.dumps(recs_b[0], sort_keys=True)
    assert blob_a == blob_b  # byte-identical across runs
    rec = recs_a[0]
    assert rec["record_id"] == "xlam-000003-0"
    assert rec["result_source"] == "synthesized"
    assert len(rec["tool_schemas"]) == 1
    step = rec["steps"][0]
    assert step["tool"] == "get_price"
    assert step["result"]["status"] == 200
    inner = step["result"]["result"]
    assert inner["ok"] is True and inner["tool"] == "get_price"
    # declared args echo through; undeclared props are synthesized typed values
    assert inner["data"]["symbol"] == "AAPL"
    assert inner["data"]["currency"] == "USD"
    assert isinstance(inner["data"]["verbose"], bool)
    assert isinstance(inner["data"]["retries"], int)


def test_xlam_drops_bad_answers_and_prompt():
    bad = dict(XLAM_ROW, answers="not json [")
    recs, drops, _ = records_from_xlam_row(bad, 0)
    assert recs == [] and drops == {"bad_answers": 1}
    short = dict(XLAM_ROW, query="hi")
    recs, drops, _ = records_from_xlam_row(short, 0)
    assert recs == [] and drops == {"bad_prompt_len": 1}
    no_ans = dict(XLAM_ROW, answers=json.dumps([]))
    recs, drops, no_calls = records_from_xlam_row(no_ans, 0)
    assert recs == [] and not drops and no_calls == 1


def test_check_file_counts_and_dup_ids(tmp_path, capsys):
    good = {
        "source": "xlam", "record_id": "xlam-000000-0", "prompt": "p" * 10,
        "steps": [{"tool": "t", "args": {},
                   "result": {"status": 200, "result": {}}, "expect": "success"}],
    }
    p = tmp_path / "chains.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n",
                 encoding="utf-8")
    assert check_file(p) == 1
    out = capsys.readouterr().out
    assert "xlam-000000-0" in out and "xlam: 2" in out
    p.write_text(json.dumps(good) + "\n", encoding="utf-8")
    assert check_file(p) == 0
    assert check_file(tmp_path / "missing.jsonl") == 2


def test_validate_external_chain_rules():
    assert validate_external_chain([]) == "empty chain"
    ok = [{"tool": "T", "args": {}, "result": {"status": 200, "result": {}},
           "expect": "success"}]
    assert validate_external_chain(ok) is None
    assert validate_external_chain(
        [{"tool": "", "args": {}, "result": {"status": 200}}])
    assert validate_external_chain(
        [{"tool": "T", "args": "x", "result": {"status": 200}}])
    assert validate_external_chain(
        [{"tool": "T", "args": {}, "result": {"nope": 1}}])
    err_last = [{"tool": "T", "args": {},
                 "result": {"status": 500, "error": {"code": "X"}},
                 "expect": "success"}]
    assert validate_external_chain(err_last) == (
        "chain does not complete (final step failed)")
    tagged = [{"tool": "T", "args": {},
               "result": {"status": 500, "error": {"code": "X"}},
               "expect": "error"}]
    assert validate_external_chain(tagged) is None


def test_external_tier_buckets():
    def steps(n):
        return [{"tool": f"T{i}", "args": {},
                 "result": {"status": 200, "result": {}}} for i in range(n)]
    assert external_tier(steps(1)) == "easy"
    assert external_tier(steps(2)) == "medium"
    assert external_tier(steps(3)) == "hard"
    assert external_tier(steps(4)) == "hard"
    assert external_tier(steps(5)) == "expert"
    assert external_tier(steps(9)) == "expert"


def test_external_skills_only_known_tags():
    def step(tool, round_=0):
        return {"tool": tool, "args": {},
                "result": {"status": 200, "result": {}}, "round": round_}
    assert external_skills([step("A")]) == []
    assert external_skills([step("A"), step("B", 1)]) == ["multi_hop"]
    assert external_skills(
        [step("A"), step("B")]) == ["multi_hop", "fanout"]
    tags = external_skills([step("Web Search"), step("price_convert")])
    assert "search" in tags and "arithmetic" in tags and "multi_hop" in tags
    rec = external_skills([step("Retry"), step("Retry", 1)])
    assert "recovery" in rec
    for t in tags + rec:
        assert t in SKILLS


def _ext_record(steps, **kw):
    rec = {
        "source": "toolace", "record_id": "toolace-000001-0",
        "prompt": "Send an email to the ops team about the deploy.",
        "result_source": "recorded", "tool_schemas": [],
        "steps": steps,
    }
    rec.update(kw)
    return rec


def test_build_external_chain_end_to_end():
    steps = [
        {"tool": "send_email",
         "args": {"to": "ops@corp.example", "subject": "deploy"},
         "result": {"status": 200,
                    "result": {"to": "ops@corp.example", "sent": True}},
         "expect": "success", "round": 0},
    ]
    traj = build_external_chain(_ext_record(steps))
    assert traj["plan_id"] == "ext-toolace-toolace-000001-0"
    assert traj["plan_index"] == -1
    assert traj["seed_class"] == "agentic"
    assert traj["source"] == "external"
    assert traj["result_source"] == "recorded"
    assert traj["vars"] == {}
    # The generic gate passes; the internal gate must reject (unregistered
    # send_email address is a mock-surface law, not a corpus law).
    assert validate_external_chain(traj["steps"]) is None
    assert validate_chain(traj["steps"]) is not None
    # Teacher prompt renders over corpus steps.
    prompt = build_prose_prompt(traj)
    assert "TOOL EXECUTION RECORD" in prompt and "send_email" in prompt
    # traj_hash: stable, and distinct from a plan-built chain.
    assert trajectory_hash(traj) == trajectory_hash(traj)
    plan_traj = build_chain(random.Random(0))
    assert trajectory_hash(traj) != trajectory_hash(plan_traj)


def test_build_external_chain_raises_on_invalid():
    with pytest.raises(ValueError):
        build_external_chain(_ext_record([]))
    with pytest.raises(ValueError):
        build_external_chain(_ext_record(
            [{"tool": "T", "args": {}, "result": {"status": 500},
              "expect": "success"}]))


def test_distiller_loads_pool_and_validates_path(tmp_path):
    import yaml
    from distill_tools import Distiller
    from pathlib import Path as P
    roster = yaml.safe_load(
        (P(__file__).resolve().parent.parent
         / "configs" / "roster.example.yaml").read_text(encoding="utf-8"))
    pool = tmp_path / "chains.jsonl"
    recs, _, _ = records_from_xlam_row(XLAM_ROW, 0)
    pool.write_text("\n".join(json.dumps(r) for r in recs) + "\n",
                    encoding="utf-8")
    d = Distiller(roster, tmp_path / "raw",
                  external_chains_path=str(pool), external_frac=0.5)
    assert len(d.external_chains) == 1
    assert d.external_frac == 0.5
    with pytest.raises(SystemExit):
        Distiller(roster, tmp_path / "raw2",
                  external_chains_path=str(tmp_path / "nope.jsonl"))
