"""dataset_publish.py - build a PUBLIC-safe dataset bundle from raw traces.

Only what downstream consumers need, nothing else. Strips every internal
field: teacher identity, teacher mode, plan template index, chain_steps,
vars, seed class, and any provider/route identifiers.

Public SFT record:  messages, prompt, plan_id, plan_tier, traj_hash, eval
Public DPO record:  pair_id, mutation, gates, chosen/rejected (slimmed)
                    + plan_id, plan_tier, traj_hash on each side

Usage:
    python src/dataset_publish.py --input data/raw --out /tmp/hf-bundle
    python src/dataset_publish.py --input data/raw --out /tmp/hf-bundle --check
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

PUBLIC_TRACE_KEYS = ("messages", "prompt", "plan_id", "plan_tier", "traj_hash", "eval")
PUBLIC_PAIR_KEYS = ("pair_id", "mutation", "gates")
PUBLIC_SIDE_KEYS = ("messages", "plan_id", "plan_tier", "traj_hash")

# Anything that smells like a teacher identity / route / internal field.
INTERNAL_KEY_RE = re.compile(
    r"teacher|provider|route|key_env|base_url|model_id|seed_class|"
    r"plan_template|chain_steps|\bvars\b|distill_version|forge_spec",
    re.IGNORECASE,
)


def scrub_trace(trace: dict) -> dict:
    """Return a public-safe copy of an SFT trace (slim, no internal keys)."""
    out = {k: copy.deepcopy(trace[k]) for k in PUBLIC_TRACE_KEYS if k in trace}
    return out


def scrub_side(side: dict) -> dict:
    """Slim one side (chosen/rejected) of a DPO pair."""
    return {k: copy.deepcopy(side[k]) for k in PUBLIC_SIDE_KEYS if k in side}


def scrub_pair(pair: dict) -> dict:
    """Return a public-safe copy of a DPO pair line."""
    out = {k: copy.deepcopy(pair[k]) for k in PUBLIC_PAIR_KEYS if k in pair}
    if isinstance(pair.get("chosen"), dict):
        out["chosen"] = scrub_side(pair["chosen"])
    if isinstance(pair.get("rejected"), dict):
        out["rejected"] = scrub_side(pair["rejected"])
    return out


def load_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_traces(input_dir: Path):
    for p in sorted(input_dir.glob("traces_*.jsonl")):
        yield from load_lines(p)


def iter_pairs(input_dir: Path):
    for p in sorted(input_dir.glob("dpo_pairs_*.jsonl")):
        yield from load_lines(p)


def _keys_of(rec: dict) -> set[str]:
    keys = set(rec.keys())
    if isinstance(rec.get("chosen"), dict):
        keys |= {"chosen." + k for k in rec["chosen"]}
    if isinstance(rec.get("rejected"), dict):
        keys |= {"rejected." + k for k in rec["rejected"]}
    return keys


def assert_clean(records: list[dict], kind: str) -> None:
    """Fail loudly if any top-level or side field is not in the public set."""
    allowed = set(PUBLIC_TRACE_KEYS) if kind == "sft" else set(PUBLIC_PAIR_KEYS)
    allowed |= {"chosen", "rejected"}  # the side containers themselves
    allowed |= {"chosen." + k for k in PUBLIC_SIDE_KEYS}
    allowed |= {"rejected." + k for k in PUBLIC_SIDE_KEYS}
    bad: list[str] = []
    for i, rec in enumerate(records):
        extra = _keys_of(rec) - allowed
        if extra:
            bad.append(f"#{i}:{sorted(extra)}")
    if bad:
        raise AssertionError(
            f"{kind}: {len(bad)} record(s) have non-public fields: "
            + ", ".join(bad[:5])
        )


def publish(input_dir: Path, out_dir: Path, check: bool = False) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "sft").mkdir(parents=True, exist_ok=True)
    (out_dir / "dpo").mkdir(parents=True, exist_ok=True)

    sft = [scrub_trace(t) for t in iter_traces(input_dir)]
    pairs = [scrub_pair(p) for p in iter_pairs(input_dir)]

    assert_clean(sft, "sft")
    assert_clean(pairs, "dpo")

    with (out_dir / "sft" / "train.jsonl").open("w", encoding="utf-8") as fh:
        for rec in sft:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "dpo" / "train.jsonl").open("w", encoding="utf-8") as fh:
        for rec in pairs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    card = input_dir / "eval_card.json"
    if card.exists():
        import shutil

        shutil.copyfile(card, out_dir / "eval_card.json")

    summary = {"sft": len(sft), "dpo": len(pairs)}
    if check:
        print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="data/raw dir with traces_*.jsonl")
    ap.add_argument("--out", required=True, help="bundle dir (sft/, dpo/ created)")
    ap.add_argument("--check", action="store_true", help="print counts after writing")
    args = ap.parse_args(argv)
    publish(Path(args.input), Path(args.out), check=args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
