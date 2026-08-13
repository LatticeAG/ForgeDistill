"""Pytest bootstrap: put src/ on sys.path and give tests a tmp raw dir."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def tmp_raw(tmp_path: Path) -> Path:
    """Isolated data/raw under a fake repo root. Never touches live data/raw."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (tmp_path / "data" / "archive").mkdir(parents=True)
    return raw


@pytest.fixture
def repo_root() -> Path:
    return REPO
