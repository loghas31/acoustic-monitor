"""
tests/test_baseline_fingerprint.py — backlog T3.7, fingerprint the baseline
against the code that produced it.

WHY THIS EXISTS
----------------------------------------------------------------------------
T1.8 (2026-08-18) measured a real bug: changing only the simulated
accelerometer model — with the feature vector still 37-dimensional — made
100% of fresh HEALTHY windows score a median 138.4x their threshold against
the deployed `baseline.npz`. T1.5's dimension-mismatch check (baseline was
fit on N dims, firmware now produces M dims) passed the whole time, because
the dimension did not change — only the DISTRIBUTION underneath it did.
Before this file, `firmware/main.py`'s ordinary persistence gate could not
tell that apart from a real fault appearing in the unit's very first
windows: both look like "score above threshold, repeatedly". Eventually it
would raise a normal-looking `ALERT #1`, and a customer whose firmware
silently drifted from its baseline would be told their machine is failing.

`fit_baseline` (firmware/baseline.py) now stores two extra scalars —
`startup_ratio_median`/`startup_ratio_p95`, a summary of the learn period's
own pooled (held-out where cross-validation ran) score/threshold ratio
distribution. `MahalanobisScorer.score()` itself stays a pure function with
no hidden side effect — this suite calls it directly against curated
windows for unrelated reasons (feedback retraining, threshold behaviour,
sensitivity sweeps) and none of that may raise. Instead, a new explicit
method, `check_startup_fingerprint(ratios)`, is called exactly once by
`firmware/main.py`'s real startup loop after it has collected the unit's
first `STARTUP_CHECK_WINDOWS` (8) real score/threshold ratios: if nearly
all of them (>=75%) are anomalous AND their median ratio is implausibly
large relative to what the learn period itself produced (>=5x its own
p95), it raises `BaselineMismatchError` instead of letting the persistence
gate treat it as an ordinary fault.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline import fit_baseline, save_baseline               # noqa: E402
from features import FEATURE_NAMES                             # noqa: E402
from inference import BaselineMismatchError, MahalanobisScorer  # noqa: E402

D = len(FEATURE_NAMES)
ACCEL_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("accel_")]
assert len(ACCEL_COLS) >= 10, "test assumption: plenty of accel-derived columns to shift"


def _learn_set(rng: np.random.Generator, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """A single-regime, independently-constructed synthetic learn period —
    not routed through ml/simulate.py, so this test exercises fit_baseline/
    MahalanobisScorer against inputs it did not itself generate."""
    X = rng.normal(0.0, 1.0, size=(n, D))
    X[:, 0] += 0.3 * X[:, 1]         # a little real correlation, not pure diagonal
    X[:, 10] += 0.2 * X[:, 11]
    fr = rng.normal(50.0, 0.3, n)
    audio_rms = rng.normal(-3.0, 0.05, n)
    accel_rms = rng.normal(-2.0, 0.05, n)
    OP = np.column_stack([fr, audio_rms, accel_rms])
    return X, OP


def _write_baseline(tmp_path: Path, rng: np.random.Generator, n: int = 48,
                    drop_fingerprint: bool = False) -> Path:
    X, OP = _learn_set(rng, n)
    b = fit_baseline(X, OP, list(FEATURE_NAMES))
    if drop_fingerprint:
        del b["startup_ratio_median"]
        del b["startup_ratio_p95"]
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    return path


# ----------------------------------------------------------------------------
# 1. fit_baseline stores a sane fingerprint
# ----------------------------------------------------------------------------

def test_fit_baseline_stores_a_pooled_learn_ratio_fingerprint():
    rng = np.random.default_rng(0)
    X, OP = _learn_set(rng)
    b = fit_baseline(X, OP, list(FEATURE_NAMES))
    assert "startup_ratio_median" in b
    assert "startup_ratio_p95" in b
    # by construction the threshold sits near the learn period's own tail,
    # so the learn period's OWN median ratio should be comfortably under 1
    assert 0.0 <= b["startup_ratio_median"] < 1.0
    assert b["startup_ratio_median"] <= b["startup_ratio_p95"]


# ----------------------------------------------------------------------------
# 2. Must not cry wolf
# ----------------------------------------------------------------------------

def _score_ratios(scorer: MahalanobisScorer, X: np.ndarray, OP: np.ndarray) -> list[float]:
    """Score every row through the real, side-effect-free score() and return
    the score/threshold ratios in order — what main.py itself accumulates
    before calling check_startup_fingerprint."""
    ratios = []
    for i in range(len(X)):
        out = scorer.score(X[i], OP[i])
        ratios.append(out["score"] / out["threshold"] if out["threshold"] > 0 else float("inf"))
    return ratios


def test_score_itself_never_raises_no_matter_what_it_is_fed(tmp_path):
    """score() must stay a pure function with no hidden one-shot side
    effect — this codebase's own test suite (feedback retraining, threshold
    behaviour, sensitivity sweeps) calls it against curated, sometimes very
    anomalous windows for reasons that have nothing to do with a unit's
    real startup, and none of that may raise BaselineMismatchError."""
    rng = np.random.default_rng(1)
    path = _write_baseline(tmp_path, rng)
    scorer = MahalanobisScorer(path)
    rng2 = np.random.default_rng(2)
    Xnew, OPnew = _learn_set(rng2, n=20)
    Xnew[:, ACCEL_COLS] += 50.0       # wildly anomalous, every window
    for i in range(20):
        scorer.score(Xnew[i], OPnew[i])   # must not raise, ever


def test_genuinely_healthy_startup_windows_do_not_trip_the_check(tmp_path):
    rng = np.random.default_rng(1)
    path = _write_baseline(tmp_path, rng)
    scorer = MahalanobisScorer(path)
    rng2 = np.random.default_rng(2)
    Xnew, OPnew = _learn_set(rng2, n=8)      # same distribution, fresh draw
    ratios = _score_ratios(scorer, Xnew, OPnew)
    scorer.check_startup_fingerprint(ratios)  # must not raise


def test_a_single_early_transient_does_not_trip_the_check(tmp_path):
    """One loud window among the first 8 is exactly the false-positive shape
    T1.6's contamination guard was built to distrust elsewhere in this
    codebase — the startup check must show the same restraint."""
    rng = np.random.default_rng(3)
    path = _write_baseline(tmp_path, rng)
    scorer = MahalanobisScorer(path)
    rng2 = np.random.default_rng(4)
    Xnew, OPnew = _learn_set(rng2, n=8)
    Xnew[2, ACCEL_COLS] += 8.0
    ratios = _score_ratios(scorer, Xnew, OPnew)
    scorer.check_startup_fingerprint(ratios)  # must not raise


def test_a_real_fault_scored_normally_is_unaffected_by_the_startup_check(tmp_path):
    """check_startup_fingerprint is only ever invoked by main.py once, on
    its own explicitly-collected list — calling it (or not calling it) has
    no bearing on what score() itself returns for a later, genuinely
    anomalous window. A real developing fault must still be able to fire
    the ordinary persistence gate."""
    rng = np.random.default_rng(5)
    path = _write_baseline(tmp_path, rng)
    scorer = MahalanobisScorer(path)
    rng2 = np.random.default_rng(6)
    Xnew, OPnew = _learn_set(rng2, n=8)
    ratios = _score_ratios(scorer, Xnew, OPnew)
    scorer.check_startup_fingerprint(ratios)   # clears cleanly, healthy startup
    Xfault, OPfault = _learn_set(rng2, n=1)
    Xfault[0, ACCEL_COLS] += 6.0
    out = scorer.score(Xfault[0], OPfault[0])  # must NOT raise
    assert out["anomalous"] is True


def test_a_baseline_saved_before_this_feature_existed_skips_the_check(tmp_path):
    """Backward compatibility: a baseline.npz already on a deployed unit
    from before T3.7 has no startup_ratio_* fields. check_startup_fingerprint
    must be a no-op against it — not crash on a missing key, and not apply
    a check it has no data to support."""
    path = _write_baseline(tmp_path, np.random.default_rng(7), drop_fingerprint=True)
    scorer = MahalanobisScorer(path)
    assert scorer.startup_ratio_p95 is None
    rng2 = np.random.default_rng(8)
    Xnew, OPnew = _learn_set(rng2, n=8)
    Xnew[:, ACCEL_COLS] += 10.0
    ratios = _score_ratios(scorer, Xnew, OPnew)
    scorer.check_startup_fingerprint(ratios)  # must not raise — nothing to check against


# ----------------------------------------------------------------------------
# 3. The bug itself: a dimension-preserving feature-contract shift
# ----------------------------------------------------------------------------

def test_a_systematically_shifted_feature_contract_now_refuses_loudly_instead_of_scoring_silently(tmp_path):
    """Reproduces the T1.8-class bug directly: a firmware change that alters
    how a SUBSET of features are generated, with the vector's dimension
    unchanged (so T1.5's separate dimension check does not fire), used to
    be indistinguishable from a real fault — every one of the first 8
    windows scores as anomalous, at a magnitude the learn period itself
    never produced. Must now raise BaselineMismatchError, not silently
    proceed to what would eventually become an ordinary persistence alert."""
    rng = np.random.default_rng(9)
    path = _write_baseline(tmp_path, rng)
    scorer = MahalanobisScorer(path)
    rng2 = np.random.default_rng(10)
    Xnew, OPnew = _learn_set(rng2, n=8)
    Xnew[:, ACCEL_COLS] += 10.0    # systematic, every window, same columns
    ratios = _score_ratios(scorer, Xnew, OPnew)  # score() itself must not raise

    with pytest.raises(BaselineMismatchError, match="different feature-generation contract"):
        scorer.check_startup_fingerprint(ratios)


# ----------------------------------------------------------------------------
# 4. End to end through the real main.py CLI
# ----------------------------------------------------------------------------

def test_main_py_refuses_a_mismatched_baseline_legibly_end_to_end(tmp_path):
    """Fits a REAL baseline via the real `firmware/baseline.py --simulate`
    CLI, shifts its stored accel-related means (the same shape of change
    T1.8 made to the underlying signal generator) and saves it back out
    with the real save_baseline, then runs the real `firmware/main.py
    --simulate` CLI against the shifted file. Must exit non-zero with a
    named "baseline mismatch" message on stderr, not a bare traceback, and
    must exit before any windows had a chance to accumulate into an
    ordinary ALERT."""
    good = tmp_path / "good.npz"
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"),
         "--simulate", "--windows", "48", "--out", str(good),
         "--db", str(tmp_path / "learn_state.db")],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr

    import numpy as _np
    z = _np.load(good, allow_pickle=True)
    b = {k: z[k] for k in z.files}
    b["means"] = b["means"].copy()
    b["means"][:, ACCEL_COLS] += 10.0   # emulate a T1.8-shaped firmware drift
    bad = tmp_path / "bad.npz"
    save_baseline(bad, b)

    cfg_path = ROOT / "firmware" / "config.yaml"
    r2 = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "main.py"),
         "--config", str(cfg_path), "--simulate", "--no-mqtt", "--fast",
         "--minutes", "5", "--baseline", str(bad),
         "--db", str(tmp_path / "run_state.db")],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r2.returncode != 0
    assert "baseline mismatch" in r2.stderr
    assert "different feature-generation contract" in r2.stderr
    assert "ALERT #" not in r2.stdout
