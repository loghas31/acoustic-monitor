"""
Runs the dashboard's chart-data transforms (T1.11) inside the normal pytest
suite by shelling out to node.

Why bother: the severity chart is the commercial deliverable of T1.7/T1.11,
and its correctness lives in `byRegime` / `hasDisplayIndex` / `sparklineCaption`
— pure functions that decide whether a customer sees one honest line per
operating regime or one misleading sawtooth. The repo has no JS test runner and
adding vitest+jsdom is a dependency and a build change for four functions, so
they are plain ESM with a plain node test. This wrapper means `pytest tests/`
is still the single command that checks everything.

Skips (does not fail) when node is unavailable, so the Python suite stays
runnable on a machine without it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "frontend" / "src" / "lib" / "trend.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_trend_transforms():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    r = subprocess.run(["node", str(SCRIPT)], capture_output=True, text=True, timeout=120)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
