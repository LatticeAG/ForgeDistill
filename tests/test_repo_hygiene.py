from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )


def _require_git_checkout() -> None:
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    r = _git("rev-parse", "--is-inside-work-tree")
    if r.returncode != 0:
        pytest.skip("not a git checkout")


TEXT_SUFFIXES = (
    ".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".in", ".sh", ".json",
)


def _iter_text_files():
    """Yield (rel_path, text) for every tracked text file (fallback: tree walk).

    Aliases leaked into any published file (docs, changelog, tests) are the
    failure mode; scanning the whole tracked set is the only way to catch one
    that lands outside tests/src/configs/README.
    """
    _require_git_checkout()
    listed = _git("ls-files", "-z")
    rels = [p for p in listed.stdout.split("\0") if p]
    if not rels:
        rels = [
            str(f.relative_to(REPO))
            for f in REPO.rglob("*")
            if f.is_file()
            and ".git" not in f.parts
            and "__pycache__" not in f.parts
            and ".venv" not in f.parts
        ]
    for rel in rels:
        p = REPO / rel
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield rel, p.read_text(encoding="utf-8", errors="replace")


def test_operator_loops_untracked():
    _require_git_checkout()
    r = _git("ls-files", "prod_loop.sh", "loop_watcher.sh")
    assert r.stdout.strip() == ""


def test_no_home_ubuntu_paths_in_tracked_files():
    _require_git_checkout()
    needle = "/" + "home" + "/" + "ubuntu"
    listed = _git("ls-files", "-z")
    files = [p for p in listed.stdout.split("\0") if p]
    hits = []
    for rel in files:
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        if needle in text:
            hits.append(rel)
    assert hits == [], hits


def test_no_internal_route_aliases():
    aliases = ["lex" + "gf", "lex" + "zm", "nvd" + "acf", "kim" + "cf"]
    pat = re.compile("|".join(re.escape(a) for a in aliases))
    hits = []
    for rel, text in _iter_text_files():
        if pat.search(text):
            hits.append(rel)
    assert hits == [], hits
