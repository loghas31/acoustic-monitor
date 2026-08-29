"""
tests/test_evaluate_pinned.py — backlog T1.14 Part 1, SELF-REVIEW F20.

F20 measured that T1.13's per-machine `crest_floor` calibration (correct, and
what F19 asked for) moved `ml/evaluate.py`'s `deployed_threshold_fpr` from
0.000 to 0.107 -- 3 of 28 healthy windows crossing threshold -- while AUC,
TPR, and the gating/regime counts all stayed exactly where they were, so
`STAGE 3 GATE: PASS` never changed and no existing test failed. The defect
named in F20 is that this number was measured once, printed to a JSON file,
and never asserted anywhere: "a proven number regressed by 10 percentage
points and every check stayed green."

This file is the fix for that specific defect (Part 1 of T1.14). It is
deliberately NOT a fix for the band-instability mechanism itself (Part 2,
still open) -- these tests pin what STAGE 3's own metrics function reports
against the REAL deployed `firmware/baseline.npz` TODAY, so that Part 2 (or
any other future change to `crest_floor`, `select_demodulation_band`, or the
baseline) has to explain itself here instead of silently drifting again.

If Part 2 lands and restores FPR to ~0 as intended, THIS FILE MUST BE UPDATED
to the new measured value in the same run that ships the fix -- exactly the
same discipline F11's doc-count guard and T1.6's threshold pins already use.
Do not "fix" a failure here by loosening the tolerance without re-measuring.

UPDATE, 2026-08-27 -- Part 2 landed and this file was updated as instructed.
`CREST_FLOOR_MARGIN` went 0.3 -> 0.7 in `firmware/baseline.py`, the deployed
floor went 7.073 -> 7.489, and `deployed_threshold_fpr` went 0.107 -> 0.000.
Three tests here failed on that change, which is the whole point of the file:
the guard fired, the number was re-measured, and the pins were moved WITH the
new measurement recorded (see F20 in docs/DOC_SELF_REVIEW.md for the sweep
table and the priced trade-off -- 4/6 F19 recovery instead of 5/6).

Note the asymmetry that makes this file worth keeping: the pins are exact
equality, while `tests/test_stage3_gate_numbers.py` bounds the same quantities
loosely (FPR <= 0.02, TPR >= 0.95) via a subprocess against `evaluate.py`'s
printed JSON. Loose bounds catch "the product broke"; exact pins catch "the
product changed and nobody said so". F20 was the second kind, so both stay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "ml"), str(_ROOT / "firmware")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate import compute_metrics, stage3_gate_passes   # noqa: E402
from inference import MahalanobisScorer                     # noqa: E402

BASELINE = _ROOT / "firmware" / "baseline.npz"


@pytest.fixture(scope="module")
def metrics():
    if not BASELINE.exists():
        pytest.skip("firmware/baseline.npz not present -- nothing to pin against")
    scorer = MahalanobisScorer(BASELINE)
    return compute_metrics(scorer, windows=40)


def test_deployed_threshold_fpr_is_pinned_at_the_f20_measured_value(metrics):
    """THE regression F20 found — now FIXED, and this pin updated accordingly.

    History, because the number moved twice and both moves matter:
      0.000  before T1.13 (global crest_floor = 10.0, which found 0/6 faults)
      0.107  after T1.13  (calibrated floor, 3/28 healthy windows over
             threshold — the F20 regression this test was written to catch)
      0.000  after T1.14  (CREST_FLOOR_MARGIN 0.3 -> 0.7, floor 7.489)

    This pin was updated rather than widened, as its previous docstring
    instructed: T1.14 Part 2 raised the margin so the floor sits further above
    each machine's own measured healthy maximum.

    There is NO detection cost. `tools/sweep_crest_margin.py` measures
    recovery as identical (6/6, 4/6, 0/6 at severity 0.35/0.20/0.10) at
    margins 0.3, 0.7 and 1.0 alike; only the FPR moves. A previous version of
    this docstring claimed the fix cost one machine (5/6 -> 4/6) — that came
    from an uncommitted script and does not reproduce. See the retraction in
    F20, docs/DOC_SELF_REVIEW.md.

    If this moves again, update it the same way: with the new measurement and
    the reason, not by loosening the bound.
    """
    assert metrics["deployed_threshold_fpr"] == pytest.approx(0.0, abs=1e-9), (
        f"deployed_threshold_fpr moved to {metrics['deployed_threshold_fpr']!r} "
        f"(was 0.000 after T1.14). If it has risen, suspect the demodulation "
        f"band first — an unstable band moves the feature vector for reasons "
        f"unrelated to the machine, which is exactly how F20 happened.")


def test_deployed_threshold_tpr_is_pinned(metrics):
    assert metrics["deployed_threshold_tpr"] == pytest.approx(1.0, abs=1e-9)


def test_auc_is_pinned(metrics):
    assert metrics["auc"] == pytest.approx(1.0, abs=1e-9)


def test_regime_switch_false_alarms_is_pinned_at_zero(metrics):
    """The other half of F20's table: regime switching itself was NOT what
    moved. If a future band-instability fix reintroduces false alarms during
    ordinary regime switches (a different failure mode than the FPR one),
    this is the test that catches it."""
    assert metrics["regime_switch_false_alarms"] == 0


def test_gating_counts_are_pinned(metrics):
    """A transient (1 window) must never alert; a persistent fault (8 of 12
    windows) must alert exactly once. F20 confirmed these were unchanged by
    T1.13; this is the standing regression guard."""
    assert metrics["gating_alerts_transient"] == 0
    assert metrics["gating_alerts_persistent"] == 1


def test_stage3_gate_still_passes_for_the_right_reasons(metrics):
    """Not just PASS -- PASS *and* the FPR is exactly the known, priced number.
    A gate that passes for a NEW, unpriced reason (e.g. FPR jumping to 0.30
    while AUC/gating stay green) must fail this even though `STAGE 3 GATE:
    PASS` alone would not have caught it -- this is the literal shape of F20."""
    assert stage3_gate_passes(metrics)
    assert metrics["deployed_threshold_fpr"] == pytest.approx(0.0, abs=1e-9)


def test_deployed_crest_floor_is_the_calibrated_value_not_the_old_constant():
    """Confirms WHY the FPR pin above is 0.000: the deployed baseline runs
    T1.13's *calibrated* floor as re-tuned by T1.14 (7.489), not the pre-T1.13
    constant (10.0, which found 0/6 faults) and not T1.13's original margin
    (7.073, which found 5/6 but at FPR 0.107).

    Two assertions, deliberately different in kind:
      - `< DEFAULT_CREST_FLOOR` is the structural claim — calibration is
        running at all. This one should never need editing.
      - the exact value is the *provenance* claim — it pins which margin is
        deployed, so a silent retrain shows up here first.

    If Logan retrains on real fridge audio, the exact value WILL move and this
    will fail. That failure is correct: re-measure, then update the number and
    the FPR pin above together, in the same commit.
    """
    from features import DEFAULT_CREST_FLOOR
    scorer = MahalanobisScorer(BASELINE) if BASELINE.exists() else None
    if scorer is None:
        pytest.skip("firmware/baseline.npz not present")
    assert scorer.crest_floor < DEFAULT_CREST_FLOOR
    assert scorer.crest_floor == pytest.approx(7.488572112684821, abs=1e-6)
