from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from prose_writer import NUDGE_TEXT

from export_sft import (
    check_loss_mask,
    export_path,
    load_template,
    message_loss_spans,
    render_messages,
    tokenize_and_mask,
    trainable_roles,
)

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
EXPORT = str(REPO / "src" / "export_sft.py")
FIXTURES = REPO / "tests" / "fixtures"
MINI_TRACES = FIXTURES / "mini_traces.jsonl"
TEMPLATES = REPO / "configs" / "templates"
NANBEIGE = TEMPLATES / "nanbeige.json"
CHATML = TEMPLATES / "chatml.json"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [PY, EXPORT, *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env=env,
    )


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _fixture_traces() -> list[dict]:
    return _load_jsonl(MINI_TRACES)


def test_template_files_assistant_only_loss():
    for path in (NANBEIGE, CHATML):
        tmpl = load_template(path)
        roles = tmpl["roles"]
        assert roles["assistant"]["loss"] is True
        for name, spec in roles.items():
            if name != "assistant":
                assert spec["loss"] is False, name
        assert trainable_roles(tmpl) == ["assistant"]
        assert "<tool_response>" in roles["tool"]["prefix"]


def test_nanbeige_template_contract():
    tmpl = load_template(NANBEIGE)
    assert tmpl["name"] == "nanbeige-chatml-v1"
    assert tmpl["bos"] == ""
    assert tmpl["eos"] == "<|im_end|>"
    tool = tmpl["roles"]["tool"]
    assert tool["prefix"] == "<|im_start|>user\n<tool_response>\n"
    assert tool["suffix"] == "\n</tool_response><|im_end|>\n"
    assert tool["loss"] is False


def test_chatml_name_and_tool_response():
    tmpl = load_template(CHATML)
    assert tmpl["name"] == "chatml-v1"
    assert "<tool_response>" in tmpl["roles"]["tool"]["prefix"]
    assert tmpl["roles"]["assistant"]["loss"] is True


def test_tool_turns_in_rendered_chatml_contain_tool_response():
    tmpl = load_template(CHATML)
    traces = _fixture_traces()
    assert traces
    saw_tool = False
    for trace in traces:
        messages = trace["messages"]
        text = render_messages(messages, tmpl)
        spans = message_loss_spans(messages, tmpl)
        for msg, (start, end, trainable) in zip(messages, spans):
            chunk = text[start:end]
            spec = tmpl["roles"][msg["role"]]
            expected = spec["prefix"] + (msg.get("content") or "") + spec["suffix"]
            assert chunk == expected
            assert trainable is spec["loss"]
            if msg["role"] == "tool":
                saw_tool = True
                assert "<tool_response>" in chunk
                assert "</tool_response>" in chunk
        assert NUDGE_TEXT not in text
    assert saw_tool


def test_nudge_absent_in_fixtures_and_export(tmp_path: Path):
    tmpl = load_template(NANBEIGE)
    for trace in _fixture_traces():
        blob = json.dumps(trace, ensure_ascii=False)
        assert NUDGE_TEXT not in blob
        for msg in trace["messages"]:
            assert NUDGE_TEXT not in (msg.get("content") or "")
        rec = {
            "messages": trace["messages"],
            "trainable_roles": ["assistant"],
            "template": tmpl["name"],
            "_template": tmpl,
        }
        check_loss_mask(rec, None)
    out = tmp_path / "sft.jsonl"
    records = export_path(MINI_TRACES, out, tmpl, fmt="messages")
    for rec in records:
        assert NUDGE_TEXT not in json.dumps(rec, ensure_ascii=False)
        check_loss_mask({**rec, "_template": tmpl}, None)


def test_assistant_only_trainable_spans():
    for path in (NANBEIGE, CHATML):
        tmpl = load_template(path)
        for trace in _fixture_traces():
            messages = trace["messages"]
            spans = message_loss_spans(messages, tmpl)
            assert len(spans) == len(messages)
            assert any(tr for _, _, tr in spans)
            for msg, (_, _, tr) in zip(messages, spans):
                if msg["role"] == "assistant":
                    assert tr is True
                else:
                    assert tr is False


def test_tokenize_and_mask_whitespace_assistant_only():
    tmpl = load_template(NANBEIGE)
    trace = _fixture_traces()[0]
    messages = trace["messages"]
    text = render_messages(messages, tmpl)
    spans = message_loss_spans(messages, tmpl)
    packed = tokenize_and_mask(text, spans, None)
    assert packed["input_ids"]
    assert len(packed["input_ids"]) == len(packed["labels"])
    assert any(lab != -100 for lab in packed["labels"])
    check_loss_mask(
        {
            "messages": messages,
            "trainable_roles": ["assistant"],
            "_template": tmpl,
            "text": text,
            "loss_char_spans": spans,
            **packed,
        },
        None,
    )
    from export_sft import WhitespaceTokenizer

    tok = WhitespaceTokenizer()
    check_loss_mask(
        {
            "messages": messages,
            "trainable_roles": ["assistant"],
            "_template": tmpl,
            "text": text,
            "loss_char_spans": spans,
            **packed,
        },
        tok,
    )


def test_cli_check_mask_exits_0_on_fixtures(tmp_path: Path):
    out = tmp_path / "nanbeige.jsonl"
    r = _run([
        "--input", str(MINI_TRACES),
        "--out", str(out),
        "--template", str(NANBEIGE),
        "--format", "messages",
        "--check-mask",
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    rows = _load_jsonl(out)
    assert len(rows) == len(_fixture_traces())
    for rec in rows:
        assert rec["trainable_roles"] == ["assistant"]
        assert rec["template"] == "nanbeige-chatml-v1"
        assert rec.get("plan_id")
        assert rec.get("traj_hash")
        assert rec["messages"]
        assert "text" not in rec
    audit = tmp_path / "mask_audit.txt"
    assert audit.is_file() and audit.stat().st_size > 0
    audit_text = audit.read_text(encoding="utf-8")
    for rec in rows[:3]:
        assert rec["traj_hash"] in audit_text
    assert "trainable_spans=" in audit_text


def test_cli_omits_template_uses_packaged_nanbeige(tmp_path: Path):
    out = tmp_path / "default.jsonl"
    r = _run([
        "--input", str(MINI_TRACES),
        "--out", str(out),
        "--check-mask",
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    rows = _load_jsonl(out)
    assert rows
    assert rows[0]["template"] == "nanbeige-chatml-v1"
    assert out.is_file() and out.stat().st_size > 0


def test_export_does_not_ingest_dpo_pairs(tmp_path: Path):
    traces = _fixture_traces()
    (tmp_path / "traces_ok.jsonl").write_text(
        json.dumps(traces[0], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dummy = {
        "chosen": {
            "messages": [{"role": "assistant", "content": "DPO_SENTINEL"}],
            "plan_id": "dpo-dummy",
        }
    }
    (tmp_path / "dpo_pairs_x.jsonl").write_text(
        json.dumps(dummy, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out = tmp_path / "export.jsonl"
    tmpl = load_template(NANBEIGE)
    records = export_path(tmp_path, out, tmpl, fmt="messages")
    assert len(records) == 1
    blob = out.read_text(encoding="utf-8")
    assert "DPO_SENTINEL" not in blob
    assert records[0]["plan_id"] == traces[0]["plan_id"]


def test_rendered_chatml_writes_jinja_next_to_out(tmp_path: Path):
    out = tmp_path / "ov.jsonl"
    tmpl = load_template(CHATML)
    records = export_path(MINI_TRACES, out, tmpl, fmt="rendered")
    jinja = tmp_path / "chat_template.jinja"
    assert jinja.is_file()
    body = jinja.read_text(encoding="utf-8")
    assert "for message in messages" in body
    assert "<tool_response>" in body
    assert records
    assert "text" in records[0]
    assert "loss_char_spans" in records[0]
    assert "<tool_response>" in records[0]["text"]
    r = _run([
        "--input", str(MINI_TRACES),
        "--out", str(tmp_path / "cli.jsonl"),
        "--template", str(CHATML),
        "--format", "rendered",
        "--check-mask",
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "chat_template.jinja").is_file()
