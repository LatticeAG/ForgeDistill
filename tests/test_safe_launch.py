from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "safe_launch.sh"


def test_bash_n_safe_launch():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_hermes_path():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "/home/" + "ubuntu/.hermes" not in text
    assert "hermes-agent" not in text
    assert "FORGE_PYTHON" in text
