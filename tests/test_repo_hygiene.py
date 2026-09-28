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
    pat = "|".join(re.escape(a) for a in aliases)
    blob_parts: list[str] = []
    for rel in ("tests", "src", "configs", "README.md"):
        p = REPO / rel
        if p.is_file():
            blob_parts.append(p.read_text(encoding="utf-8", errors="replace"))
            continue
        for f in p.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                blob_parts.append(f.read_text(encoding="utf-8", errors="replace"))
    blob = "".join(blob_parts)
    assert re.search(pat, blob) is None
