from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from distill_tools import Distiller, iter_provider_items

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
