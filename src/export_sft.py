#!/usr/bin/env python3
"""export_sft.py - render traces to SFT jsonl with assistant-only loss masks.

Render is data-driven from configs/templates/*.json. Do not special-case
Nanbeige in Python; if a checkpoint chat_template disagrees, edit the JSON.

CLI:
  python src/export_sft.py --input data/raw --out data/export/nanbeige.jsonl \\
      --template configs/templates/nanbeige.json --format messages --check-mask
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agentic_plans import trajectory_hash
from eval_card import load_traces
from prose_writer import NUDGE_TEXT

LABEL_IGNORE = -100


class WhitespaceTokenizer:
    """Split on non-whitespace. Used when no HuggingFace tokenizer is given."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids, offsets = self._tokenize(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids, _ = self._tokenize(text)
        return ids

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(str(i) for i in ids)

    def _tokenize(self, text: str):
        ids = []
        offsets = []
        for m in re.finditer(r"\S+", text or ""):
            ids.append(_stable_id(m.group()))
            offsets.append((m.start(), m.end()))
        return ids, offsets


def _stable_id(piece: str) -> int:
    n = 0
    for c in piece:
        n = (n * 16777619 + ord(c)) & 0x7FFFFFFF
    return n or 1


def load_template(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "name" not in data or "roles" not in data:
        raise ValueError(f"template missing name or roles: {path}")
    if not isinstance(data["roles"], dict):
        raise ValueError(f"template roles must be an object: {path}")
    return data


def _content_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _role_spec(template: dict, role) -> dict:
    roles = template.get("roles") or {}
    if role not in roles:
        raise ValueError(f"unknown role: {role}")
    spec = roles[role]
    if not isinstance(spec, dict):
        raise ValueError(f"invalid role spec for {role}")
    return spec


def _iter_message_chunks(messages, template: dict):
    bos = template.get("bos") or ""
    pos = len(bos)
    chunks = []
    for msg in messages or []:
        role = msg.get("role")
        spec = _role_spec(template, role)
        content = _content_str(msg.get("content"))
        piece = f"{spec.get('prefix') or ''}{content}{spec.get('suffix') or ''}"
        start = pos
        end = pos + len(piece)
        chunks.append((piece, start, end, bool(spec.get("loss")), role))
        pos = end
    return bos, chunks


def render_messages(messages, template) -> str:
    """Full conversation string: bos + prefix + content + suffix per message."""
    bos, chunks = _iter_message_chunks(messages, template)
    return bos + "".join(piece for piece, *_ in chunks)


def message_loss_spans(messages, template) -> list[tuple[int, int, bool]]:
    """Character spans (start, end, trainable) into the rendered string. end is exclusive."""
    _, chunks = _iter_message_chunks(messages, template)
    return [(start, end, trainable) for _, start, end, trainable, _ in chunks]


def trainable_roles(template) -> list[str]:
    roles = template.get("roles") or {}
    return [name for name, spec in roles.items() if isinstance(spec, dict) and spec.get("loss")]


def _norm_spans(spans) -> list[tuple[int, int, bool]]:
    out = []
    for item in spans or []:
        start, end, trainable = item[0], item[1], item[2]
        out.append((int(start), int(end), bool(trainable)))
    return out


def _as_1d(seq):
    if seq is None:
        return []
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    seq = list(seq)
    if seq and isinstance(seq[0], (list, tuple)):
        seq = list(seq[0])
    return seq


def _as_offsets(raw) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    seq = list(raw)
    if not seq:
        return []
    first = seq[0]
    # Batched: [ [(s, e), ...], ... ]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        seq = list(first)
        first = seq[0]
    if isinstance(first, (list, tuple)) and len(first) == 2 and not isinstance(first[0], (list, tuple)):
        return [(int(s), int(e)) for s, e in seq]
    return None


def _token_trainable(start: int, end: int, spans) -> bool:
    if start == end:
        return False
    for s, e, tr in spans:
        if tr and start < e and end > s:
            return True
    return False


def _token_fully_non_trainable(start: int, end: int, spans) -> bool:
    if start == end:
        return True
    for s, e, tr in spans:
        if start < e and end > s and tr:
            return False
    return True


def _get_ids_and_offsets(tokenizer, text: str):
    if tokenizer is None:
        tokenizer = WhitespaceTokenizer()
    out = None
    for kw in (
        {"add_special_tokens": False, "return_offsets_mapping": True},
        {"return_offsets_mapping": True},
        {"add_special_tokens": False},
    ):
        try:
            out = tokenizer(text, **kw)
            break
        except TypeError:
            continue
    if out is None:
        if hasattr(tokenizer, "encode"):
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = tokenizer.encode(text)
            return [int(x) for x in _as_1d(ids)], None
        raise TypeError("tokenizer must be callable or have encode()")

    if isinstance(out, dict):
        ids = _as_1d(out.get("input_ids"))
        offsets = _as_offsets(out.get("offset_mapping")) if "offset_mapping" in out else None
    else:
        ids = _as_1d(getattr(out, "input_ids", None))
        offsets = None
        if hasattr(out, "offset_mapping"):
            offsets = _as_offsets(out.offset_mapping)
        elif hasattr(out, "encodings") and out.encodings:
            offsets = [(int(s), int(e)) for s, e in out.encodings[0].offsets]

    ids = [int(x) for x in ids]
    return ids, offsets


def _labels_without_offsets(text: str, spans, tokenizer, ids: list[int]) -> list[int]:
    labels = [LABEL_IGNORE] * len(ids)
    decode = getattr(tokenizer, "decode", None)
    if decode is None or not text:
        return labels
    pos = 0
    for i, tid in enumerate(ids):
        try:
            piece = decode([tid], skip_special_tokens=False)
        except TypeError:
            try:
                piece = decode([tid])
            except Exception:
                continue
        except Exception:
            continue
        if not piece:
            continue
        idx = text.find(piece, pos)
        used = piece
        if idx < 0:
            stripped = piece.lstrip()
            idx = text.find(stripped, pos) if stripped else -1
            used = stripped
        if idx < 0:
            continue
        start, end = idx, idx + len(used)
        pos = end
        if _token_trainable(start, end, spans):
            labels[i] = int(tid)
    return labels


def tokenize_and_mask(text, spans, tokenizer) -> dict:
    """Encode full text. labels[i] = -100 outside trainable char spans."""
    spans = _norm_spans(spans)
    ids, offsets = _get_ids_and_offsets(tokenizer, text or "")
    if offsets is None or len(offsets) != len(ids):
        labels = _labels_without_offsets(text or "", spans, tokenizer or WhitespaceTokenizer(), ids)
        return {"input_ids": ids, "labels": labels}
    labels = []
    for tid, (start, end) in zip(ids, offsets):
        if _token_trainable(int(start), int(end), spans):
            labels.append(int(tid))
        else:
            labels.append(LABEL_IGNORE)
    return {"input_ids": ids, "labels": labels}


def _example_blob(example: dict) -> str:
    parts = []
    if example.get("text"):
        parts.append(str(example["text"]))
    for m in example.get("messages") or []:
        parts.append(_content_str(m.get("content")))
    return "\n".join(parts)


def _template_from_example(example: dict) -> dict | None:
    tmpl = example.get("_template")
    if isinstance(tmpl, dict) and "roles" in tmpl:
        return tmpl
    t = example.get("template")
    if isinstance(t, dict) and "roles" in t:
        return t
    return None


def check_loss_mask(example, tokenizer) -> None:
    """Assert assistant-only loss, no NUDGE_TEXT, and token labels when present."""
    assert NUDGE_TEXT not in _example_blob(example), "NUDGE_TEXT leaked into export"
    tmpl = _template_from_example(example)
    messages = example.get("messages")
    if tmpl:
        for role, spec in (tmpl.get("roles") or {}).items():
            if not isinstance(spec, dict):
                continue
            if role == "assistant":
                assert spec.get("loss") is True, "assistant loss must be true"
            else:
                assert spec.get("loss") is False, f"{role} loss must be false"
        if messages:
            spans = message_loss_spans(messages, tmpl)
            assert any(tr for _, _, tr in spans), "at least one assistant span must be trainable"
            for msg, (start, end, tr) in zip(messages, spans):
                expected = bool(_role_spec(tmpl, msg.get("role")).get("loss"))
                assert tr is expected, f"trainable mismatch for role {msg.get('role')}"
                piece = render_messages([msg], {**tmpl, "bos": ""})
                rendered = example.get("text")
                if rendered:
                    assert rendered[start:end] == piece
    if example.get("trainable_roles") is not None:
        assert list(example["trainable_roles"]) == ["assistant"], example.get("trainable_roles")
    if messages:
        assert any(m.get("role") == "assistant" for m in messages), "no assistant turn"
        for m in messages:
            role = m.get("role")
            if tmpl:
                loss = bool(_role_spec(tmpl, role).get("loss"))
                if role != "assistant":
                    assert loss is False
                else:
                    assert loss is True
    spans = example.get("loss_char_spans")
    if spans is not None:
        spans = _norm_spans(spans)
        assert any(tr for _, _, tr in spans), "no trainable char span"
        if tmpl and messages:
            expected = message_loss_spans(messages, tmpl)
            assert [(s, e, t) for s, e, t in spans] == expected

    if tokenizer is not None and "input_ids" in example and "labels" in example:
        ids = [int(x) for x in example["input_ids"]]
        labels = [int(x) for x in example["labels"]]
        assert len(ids) == len(labels), "input_ids/labels length mismatch"
        assert any(lab != LABEL_IGNORE for lab in labels), "no assistant token has a real label"
        text = example.get("text")
        if text is None and tmpl and messages:
            text = render_messages(messages, tmpl)
        use_spans = _norm_spans(example.get("loss_char_spans") or [])
        if not use_spans and tmpl and messages:
            use_spans = message_loss_spans(messages, tmpl)
        if text is not None and use_spans:
            _, offsets = _get_ids_and_offsets(tokenizer, text)
            if offsets and len(offsets) == len(labels):
                saw_assistant = False
                for lab, (start, end) in zip(labels, offsets):
                    start, end = int(start), int(end)
                    if _token_fully_non_trainable(start, end, use_spans):
                        assert lab == LABEL_IGNORE, "non-assistant token has a real label"
                    if _token_trainable(start, end, use_spans):
                        assert lab != LABEL_IGNORE, "assistant token was masked"
                        saw_assistant = True
                assert saw_assistant, "no assistant token has a real label"


def _trace_hash(trace: dict) -> str:
    if trace.get("traj_hash"):
        return str(trace["traj_hash"])
    steps = trace.get("chain_steps") or trace.get("steps") or []
    return trajectory_hash({
        "plan_id": trace.get("plan_id", trace.get("plan_template")),
        "vars": trace.get("vars") or {},
        "steps": steps,
    })


def _slim_messages(messages) -> list[dict]:
    out = []
    for m in messages or []:
        out.append({"role": m.get("role"), "content": _content_str(m.get("content"))})
    return out


def _is_chatml_template(template: dict) -> bool:
    name = template.get("name") or ""
    return name == "chatml-v1" or name.startswith("chatml-")


def chat_template_jinja(template: dict) -> str:
    """Jinja chat template that loops messages and emits prefix/content/suffix."""
    lines = []
    bos = template.get("bos") or ""
    if bos:
        lines.append("{{ " + json.dumps(bos) + " }}")
    lines.append("{%- for message in messages -%}")
    roles = list((template.get("roles") or {}).items())
    for i, (role, spec) in enumerate(roles):
        spec = spec if isinstance(spec, dict) else {}
        kw = "if" if i == 0 else "elif"
        prefix = spec.get("prefix") or ""
        suffix = spec.get("suffix") or ""
        lines.append(f"{{%- {kw} message['role'] == {json.dumps(role)} -%}}")
        lines.append(
            "{{ " + json.dumps(prefix) + " + message['content'] + " + json.dumps(suffix) + " }}"
        )
    if roles:
        lines.append("{%- endif -%}")
    lines.append("{%- endfor -%}")
    return "\n".join(lines) + "\n"


def write_chat_template_jinja(template: dict, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(chat_template_jinja(template), encoding="utf-8")


def _input_paths(input_glob) -> list[Path]:
    if input_glob is None:
        return []
    if isinstance(input_glob, (list, tuple)):
        paths: list[Path] = []
        for item in input_glob:
            paths.extend(_input_paths(item))
        return paths
    return [Path(input_glob)]


def build_record(trace: dict, template: dict, fmt: str) -> dict:
    messages = _slim_messages(trace.get("messages") or [])
    rec = {
        "messages": messages,
        "trainable_roles": trainable_roles(template) or ["assistant"],
        "template": template.get("name"),
        "traj_hash": _trace_hash(trace),
        "plan_id": trace.get("plan_id"),
    }
    if fmt == "rendered":
        rec["text"] = render_messages(messages, template)
        rec["loss_char_spans"] = [
            [s, e, t] for s, e, t in message_loss_spans(messages, template)
        ]
    return rec


def export_path(input_glob, out_path, template, tokenizer=None, *, fmt="messages"):
    """Load traces_*.jsonl from a dir (or a file as given) and write SFT jsonl."""
    if not isinstance(template, dict):
        template = load_template(template)
    traces = load_traces(_input_paths(input_glob))
    records = [build_record(t, template, fmt) for t in traces if t.get("messages")]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if fmt == "rendered" and _is_chatml_template(template):
        write_chat_template_jinja(template, out_path.parent / "chat_template.jinja")
    if tokenizer is not None and records:
        rec = records[0]
        text = rec.get("text") or render_messages(rec["messages"], template)
        spans = rec.get("loss_char_spans") or message_loss_spans(rec["messages"], template)
        tm = tokenize_and_mask(text, spans, tokenizer)
        check_loss_mask(
            {**rec, **tm, "_template": template, "text": text, "loss_char_spans": spans},
            tokenizer,
        )
    return records


def _load_hf_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is not installed; pip install transformers", file=sys.stderr)
        raise SystemExit(2)
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export traces_*.jsonl to SFT jsonl with assistant-only loss masks"
    )
    ap.add_argument(
        "--input", action="append", required=True, type=Path,
        help="trace jsonl file, or a directory expanded to traces_*.jsonl only",
    )
    ap.add_argument("--out", required=True, type=Path, help="write SFT jsonl here")
    ap.add_argument("--template", required=True, type=Path, help="template JSON path")
    ap.add_argument(
        "--format", dest="fmt", choices=("messages", "rendered"), default="messages",
    )
    ap.add_argument("--check-mask", action="store_true", help="assert assistant-only loss")
    ap.add_argument(
        "--tokenizer", default=None,
        help="HuggingFace tokenizer path; enables --check-tokenizer",
    )
    ap.add_argument(
        "--check-tokenizer", action="store_true",
        help="tokenize N examples and assert token-level labels",
    )
    ap.add_argument("--n-check", type=int, default=8, help="examples to tokenize (default 8)")
    args = ap.parse_args(argv)

    template = load_template(args.template)
    do_tok = bool(args.check_tokenizer or args.tokenizer)
    tok = None
    if do_tok:
        if not args.tokenizer:
            print("--check-tokenizer requires --tokenizer PATH", file=sys.stderr)
            return 2
        tok = _load_hf_tokenizer(args.tokenizer)

    try:
        records = export_path(args.input, args.out, template, tokenizer=None, fmt=args.fmt)
        if args.check_mask:
            for rec in records:
                check_loss_mask({**rec, "_template": template}, None)
        if do_tok:
            n = max(0, args.n_check)
            for rec in records[:n]:
                text = rec.get("text") or render_messages(rec["messages"], template)
                spans = rec.get("loss_char_spans") or message_loss_spans(
                    rec["messages"], template
                )
                tm = tokenize_and_mask(text, spans, tok)
                check_loss_mask(
                    {
                        **rec,
                        **tm,
                        "_template": template,
                        "text": text,
                        "loss_char_spans": spans,
                    },
                    tok,
                )
    except AssertionError as exc:
        print(f"check-mask failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.out} n={len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
