from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_pyproject_metadata():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "latticeag-forge-distill"
    # The version is the release's single source of truth, so it is asserted
    # structurally: semver shape plus a matching CHANGELOG heading. A hardcoded
    # literal here fails the suite on every bump without catching anything.
    version = project["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog, f"CHANGELOG.md has no heading for {version}"
    deps = project["dependencies"]
    assert any("httpx" in d for d in deps)
    assert any("PyYAML" in d for d in deps)
    dev = project["optional-dependencies"]["dev"]
    assert any("pytest" in d for d in dev)
    scripts = project["scripts"]
    assert set(scripts.keys()) == {
        "distill",
        "eval_card",
        "export_sft",
        "dpo_pairs",
        "eval_live",
        "archive_data",
        "dataset_publish",
        "forge-status",
    }
    py_modules = data["tool"]["setuptools"]["py-modules"]
    assert "dataset_publish" in py_modules
    assert "lineage" in py_modules
    assert "dedup_filter" not in py_modules
    assert "harvest_prompts" not in py_modules
    assert "forge_assets" in data["tool"]["setuptools"]["packages"]


def test_no_src_init_py():
    assert not (REPO / "src" / "__init__.py").exists()


def test_py_modules_exclude_dedup_and_harvest():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    py_modules = data["tool"]["setuptools"]["py-modules"]
    assert "dedup_filter" not in py_modules
    assert "harvest_prompts" not in py_modules
    assert "external_chains" not in py_modules


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


def test_pyproject_urls():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    urls = data["project"]["urls"]
    assert urls["Homepage"] == "https://github.com/LatticeAG/ForgeDistill"
    assert urls["Repository"] == "https://github.com/LatticeAG/ForgeDistill"
    assert urls["Issues"] == "https://github.com/LatticeAG/ForgeDistill/issues"


def test_security_md_exists():
    path = REPO / "SECURITY.md"
    text = path.read_text(encoding="utf-8")
    assert path.is_file()
    assert "key_env" in text
    assert "GitHub Security Advisories" in text


def test_agent_md_exists():
    path = REPO / "AGENT.md"
    text = path.read_text(encoding="utf-8")
    assert path.is_file()
    assert "len(PLANS)==47" in text
    assert "sys.path" in text


def test_assets_match_configs():
    pairs = [
        ("configs/templates/nanbeige.json", "src/forge_assets/templates/nanbeige.json"),
        ("configs/templates/chatml.json", "src/forge_assets/templates/chatml.json"),
        ("configs/roster.example.yaml", "src/forge_assets/roster.example.yaml"),
    ]
    for a, b in pairs:
        left = (REPO / a).read_bytes()
        right = (REPO / b).read_bytes()
        assert left == right, a


def test_forge_assets_importable():
    from importlib.resources import files

    blob = (files("forge_assets") / "templates" / "nanbeige.json").read_bytes()
    assert blob == (REPO / "configs" / "templates" / "nanbeige.json").read_bytes()


def test_manifest_excludes_operator_loops():
    text = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    assert "prod_loop.sh" not in text
    assert "loop_watcher.sh" not in text


def test_requirements_dev_mirrors_pyproject():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extra = set(data["project"]["optional-dependencies"]["dev"])
    pins = {
        line.strip()
        for line in (REPO / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert pins == extra

