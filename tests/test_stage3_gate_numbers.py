"""
T1.14 / F20 — pin the STAGE 3 GATE's numbers, not just its PASS/FAIL.

WHY THIS FILE EXISTS
--------------------
On 2026-08-23 the per-machine `crest_floor` (T1.13) moved
`ml/evaluate.py`'s `deployed_threshold_fpr` from **0.000 to 0.107** — three of
28 healthy windows crossing threshold — and **every check stayed green**. The
gate keys on AUC, regime false alarms and the gating counts, all of which were
unchanged; the 30-minute persistence gate absorbed the extra windows; no test
asserted the rate. A proven measurement regressed by ten percentage points in
silence.

That is the same class of failure as F11, where the README claimed two
different test counts because nothing checked them. The remedy is the same:
**make the number a tested claim.**

These are deliberately loose bounds, not exact equality. The point is to catch
a regression of the size F20 was (0.00 -> 0.107), not to fail on the third
decimal place when scipy changes a default. If a change genuinely improves
these, the test should be updated — with the new measurement written into the
commit message, which is exactly the conversation that did not happen for F20.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "firmware" / "baseline.npz"


@pytest.fixture(scope="module")
def gate() -> dict:
    """Run the real `ml/evaluate.py` and parse its JSON report.

    A subprocess, not an import: evaluate.py is the artefact whose output gets
    quoted in DOC_STATUS and in funding applications, so what is pinned here
    must be what that script actually prints.
    """
    if not BASELINE.exists():
        pytest.skip(f"no baseline at {BASELINE} — run "
                    f"`python firmware/baseline.py --simulate --windows 48 "
                    f"--out firmware/baseline.npz --db /tmp/state.db` first")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "ml" / "evaluate.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "TMPDIR": "/tmp"})
    out = proc.stdout
    if "{" not in out:
        pytest.fail(f"evaluate.py printed no JSON.\nstdout:\n{out[-2000:]}\n"
                    f"stderr:\n{proc.stderr[-2000:]}")
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


def test_gate_passes(gate):
    assert gate["auc"] == pytest.approx(1.0, abs=0.02)


def test_false_alarm_rate_has_not_regressed(gate):
    """THE ONE F20 NEEDED. Measured 0.000 before T1.13, 0.107 after it, and
    0.000 again after T1.14 raised CREST_FLOOR_MARGIN to 0.7.

    the project's risk assessment (not in this public copy) names alarm fatigue as churn risk #1 against a target
    of <=1 false alarm per node-week. A per-window rate above a few percent
    makes that target unreachable no matter how good the persistence gate is,
    so this is the number that decides whether the product is usable.
    """
    fpr = gate["deployed_threshold_fpr"]
    assert fpr <= 0.02, (
        f"deployed_threshold_fpr is {fpr:.4f}; it was 0.000 when this test was "
        f"written. F20 is repeating — something has made healthy windows cross "
        f"threshold. Check the demodulation band first: an unstable band moves "
        f"the feature vector for reasons unrelated to the machine.")


def test_detection_rate_has_not_regressed(gate):
    """The other side. A change that drives FPR to zero by refusing to detect
    anything must fail too — that is how the pre-T1.13 constant 'passed'."""
    assert gate["deployed_threshold_tpr"] >= 0.95, (
        f"TPR {gate['deployed_threshold_tpr']:.3f}; was 1.000. The detector "
        f"has stopped finding faults it used to find.")


def test_regime_switching_still_causes_no_false_alarms(gate):
    """A machine changing speed must not look like a fault. This is the
    property `baseline.choose_k` exists for and the one a customer notices
    first, because their machine changes state every day."""
    assert gate["regime_switch_false_alarms"] == 0


def test_the_gate_still_suppresses_a_transient_and_catches_a_persistent_fault(gate):
    """The persistence gate is the false-alarm defence `DOC_ALERTING.md` calls
    the real product. One loud window must not alert; a sustained fault must,
    exactly once."""
    assert gate["gating_alerts_transient"] == 0
    assert gate["gating_alerts_persistent"] == 1
