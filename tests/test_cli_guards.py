from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

import distill_tools
from distill_tools import Distiller, _resolve_out_dir, iter_provider_items

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
DISTILL = str(REPO / "src" / "distill_tools.py")
EXAMPLE = REPO / "configs" / "roster.example.yaml"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [PY, DISTILL, *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env=env,
    )


def test_example_roster_yields_real_provider_route(tmp_path: Path):
    roster = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    d = Distiller(roster, tmp_path)
    routes = d._providers_with_models()
    assert routes, "roster.example.yaml must unwrap to at least one provider route"
    names = {p for p, _, _ in routes}
    assert "example-provider" in names or "local-router" in names
    assert "providers" not in names
    assert "roles" not in names


def test_iter_provider_items_skips_reserved_keys():
    roster = {
        "providers": {
            "p1": {"base_url": "http://x", "key_env": "", "models": {"m": {}}},
        },
        "roles": {"verifier": {"provider": "p1", "model": "m"}},
        "curriculum": {"mode": "uniform"},
        "verify": {"sample_rate": 0.2},
        "dpo": {"enabled": False},
        "backoff": {"cap": 60},
    }
    items = list(iter_provider_items(roster))
    assert [p for p, _ in items] == ["p1"]


def test_strict_tool_protocol_comes_from_roster(tmp_path: Path):
    roster = {
        "strict-prov": {
            "base_url": "http://x",
            "key_env": "",
            "strict_tool_protocol": True,
            "models": {"m": {"mode": "concise", "max_tokens": 8, "weight": 1}},
        },
        "loose-prov": {
            "base_url": "http://y",
            "key_env": "",
            "strict_tool_protocol": False,
            "models": {"m": {"mode": "concise", "max_tokens": 8, "weight": 1}},
        },
    }
    d = Distiller(roster, tmp_path)
    assert d.strict_tool_ids["strict-prov"] is True
    assert d.strict_tool_ids["loose-prov"] is False


def test_existing_trace_guard_exits_1_without_wipe(tmp_raw: Path, tmp_path: Path):
    (tmp_raw / "traces_example-provider.jsonl").write_text("{}\n", encoding="utf-8")
    r = _run([
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
    ])
    assert r.returncode == 1, r.stdout + r.stderr
    assert "GUARD" in r.stdout or "GUARD" in r.stderr
    assert (tmp_raw / "traces_example-provider.jsonl").exists()


def test_wipe_archives_then_continues(tmp_raw: Path, tmp_path: Path):
    src = tmp_raw / "traces_example-provider.jsonl"
    src.write_text('{"prompt": "keep-me"}\n', encoding="utf-8")
    r = _run([
        "--count", "0",
        "--wipe",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not src.exists()
    archived = list((tmp_path / "data" / "archive").glob("*/traces_example-provider.jsonl"))
    assert archived, "wipe must move traces into data/archive, never unlink"
    assert "keep-me" in archived[0].read_text(encoding="utf-8")


def test_legacy_v1_exits_2(tmp_raw: Path):
    r = _run([
        "--legacy-v1",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
    ])
    assert r.returncode == 2, r.stdout + r.stderr
    msg = (r.stdout + r.stderr).lower()
    assert "legacy-v1 is frozen" in msg
    assert "reversed-v2" in msg


def test_pilot_defaults_to_first_roster_provider_and_model(tmp_raw: Path):
    r = _run([
        "--pilot",
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
    ])
    assert r.returncode == 0, r.stdout + r.stderr
    combined = r.stdout + r.stderr
    assert "example-provider" in combined
    assert "example-model-thinking" in combined
    # First provider only: local-router must not be selected when filters are empty.
    assert "local-router" not in combined or "1 model routes" in combined or "1 model route" in combined


def test_example_roster_documents_roles():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "roles:" in text
    assert "verifier:" in text
    assert "answer:" in text


def test_mp_zero_exits_2(tmp_raw: Path):
    r = _run([
        "--mp", "0",
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
        "--holdout-frac", "0",
    ])
    assert r.returncode == 2, r.stdout + r.stderr


def test_mp_plus_explicit_shard_exits_2(tmp_raw: Path):
    r = _run([
        "--mp", "2",
        "--shard", "1/2",
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
        "--holdout-frac", "0",
    ])
    assert r.returncode == 2, r.stdout + r.stderr
    msg = r.stdout + r.stderr
    assert "do not combine --mp with explicit --shard" in msg


def test_mp_rejects_holdout_only_dir(tmp_raw: Path):
    (tmp_raw / "holdout_plan_ids.json").write_text("[]\n", encoding="utf-8")
    r = _run([
        "--mp", "2",
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(EXAMPLE),
        "--holdout-frac", "0",
    ])
    assert r.returncode == 1, r.stdout + r.stderr
    assert "archive or wipe before --mp" in (r.stdout + r.stderr)


def test_missing_roster_exits_2(tmp_raw: Path, tmp_path: Path):
    missing = tmp_path / "no-such-roster.yaml"
    r = _run([
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(missing),
    ])
    assert r.returncode == 2, r.stdout + r.stderr
    combined = r.stdout + r.stderr
    assert "roster not found" in combined
    assert "configs/roster.example.yaml" in combined


def test_empty_roster_exits_2(tmp_raw: Path, tmp_path: Path):
    roster = tmp_path / "reserved-only.yaml"
    roster.write_text("roles: {}\ncurriculum: {}\nverify: {}\ndpo: {}\nbackoff: {}\n", encoding="utf-8")
    r = _run([
        "--count", "0",
        "--out-dir", str(tmp_raw),
        "--roster", str(roster),
        "--holdout-frac", "0",
    ])
    assert r.returncode == 2, r.stdout + r.stderr
    assert "No models in roster" in (r.stdout + r.stderr)


def test_non_agentic_seed_class_rejects_not_raises(tmp_path: Path):
    """Synthetic-caller guard: build_chain never sets seed_class on traj.
    This injects seed_class='irrelevant' to lock the defensive reject.
    The raise is dead on the reversed-v2 worker path today.
    """
    import asyncio
    roster = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    d = Distiller(roster, tmp_path / "raw")
    traj = {"seed_class": "irrelevant", "prompt": "x", "steps": []}
    res = asyncio.run(d.generate_agentic_trace(
        "example-provider", "example-model-thinking", {"mode": "concise"}, traj=traj,
    ))
    assert res["ok"] is False
    assert res["http"] == "PLAN"
    assert "frozen" in res["error"]


def test_relative_out_dir_resolves_against_cwd(tmp_path: Path):
    r = _run(
        [
            "--count", "0",
            "--out-dir", "rel/raw",
            "--roster", str(EXAMPLE),
            "--holdout-frac", "0",
        ],
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "rel" / "raw").is_dir()
    assert not (REPO / "rel").exists()


def test_default_paths_are_cwd_relative(tmp_path: Path, monkeypatch):
    assert distill_tools.ROSTER_PATH == Path("configs/roster.yaml")
    assert distill_tools.OUT_DIR == Path("data/raw")
    assert not hasattr(distill_tools, "ROOT")
    monkeypatch.chdir(tmp_path)
    assert _resolve_out_dir("rel/raw") == (tmp_path / "rel" / "raw").resolve()
