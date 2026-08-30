"""Packaged Distillation Studio assets (templates + example roster)."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def template_path(name: str) -> Path:
    """Return the packaged template JSON path (nanbeige.json or chatml.json)."""
    p = _ROOT / "templates" / name
    if not p.is_file():
        raise FileNotFoundError(f"forge_assets template not found: {name}")
    return p


def roster_example_path() -> Path:
    return _ROOT / "roster.example.yaml"
