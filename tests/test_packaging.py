from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_pyproject_metadata():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "forge-distill"
    assert project["version"] == "0.3.0"
    deps = project["dependencies"]
    assert any("httpx" in d for d in deps)
    assert any("PyYAML" in d for d in deps)
    dev = project["optional-dependencies"]["dev"]
    assert any("pytest" in d for d in dev)
    scripts = project["scripts"]
    assert set(scripts.keys()) == {"distill", "eval_card", "export_sft"}


def test_no_src_init_py():
    assert not (REPO / "src" / "__init__.py").exists()


def test_py_modules_exclude_dedup_and_harvest():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    py_modules = data["tool"]["setuptools"]["py-modules"]
    assert "dedup_filter" not in py_modules
    assert "harvest_prompts" not in py_modules


def test_no_transformers_in_install_metadata():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"]).lower()
    assert "transformers" not in deps
    extras = data["project"]["optional-dependencies"]
    extra_blob = " ".join(x for group in extras.values() for x in group).lower()
    assert "transformers" not in extra_blob


def test_traces_fixture_matches_mini_traces():
    mini = (REPO / "tests" / "fixtures" / "mini_traces.jsonl").read_bytes()
    card = (REPO / "tests" / "fixtures" / "traces_fixture.jsonl").read_bytes()
    assert mini == card
