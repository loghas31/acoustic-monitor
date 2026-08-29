"""
tests/test_crest_calibration.py — backlog T1.13, SELF-REVIEW F19.

F19 measured `select_demodulation_band`'s fixed `crest_floor = 10.0` rejecting
13 of 14 realistic (pink-noise) synthetic faults at severity 0.20, falling
back to DEFAULT_BAND while a real 4.8x BPFO peak sat unexamined in the band
the resonance actually lives in. The obvious fix (lower the constant) does
not work: healthy crest across 14 machines spans 5.56-7.33, severity-0.20
fault crest spans 6.56-10.21 — they OVERLAP, so no single global constant
can separate every machine.

This file pins the fix that was actually shipped: calibrate the floor PER
MACHINE from its own learn period, exactly as `baseline.cv_threshold`
calibrates the anomaly threshold from the same learn period. A machine whose
own healthy crest sits low gets a low floor; a machine whose own healthy
crest sits high keeps something close to the old constant.

Written BEFORE the implementation, per the frozen-file rule in the task backlog (not in this public copy)
(firmware/features.py and firmware/baseline.py are both frozen): the two
tests marked FAILS-ON-OLD-CODE below reproduce F19's miss directly and must
fail against the pre-T1.13 code, then pass once `baseline.calibrate_crest_floor`
and the `crest_floor` plumbing exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "firmware"), str(_ROOT / "ml" / "realdata")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features import DEFAULT_BAND, DEFAULT_CREST_FLOOR, extract_features, select_demodulation_band  # noqa: E402
from synth_phone_recording import make_pair  # noqa: E402

FS = 16000.0    # the repo's real audio sample rate (not the phone-file default)


def _healthy_learn_crests(n: int = 48, seed0: int = 1) -> np.ndarray:
    """One machine's own healthy learn period: n independent 30 s draws of
    the SAME machine (rpm/bearing/resonance held fixed, only the noise
    realisation changes) — the within-machine analogue of F19's 14-machine
    table. Uses the realistic PINK generator, same reason synth_phone_recording
    exists at all (see its own module docstring): white noise never reaches
    crest_floor in the first place, so it cannot exercise this bug."""
    crests = []
    for seed in range(seed0, seed0 + n):
        pair = make_pair(seed=seed, duration_s=30.0, fs=FS, severity=0.0)
        _, c = select_demodulation_band(pair["healthy"], FS)
        crests.append(c)
    return np.array(crests)


# ----------------------------------------------------------------------------
# FAILS-ON-OLD-CODE: reproduces F19's miss directly
# ----------------------------------------------------------------------------

def test_default_floor_misses_the_f19_fault():
    """Ground truth for the bug this task fixes: at the DEFAULT crest_floor
    (10.0, unchanged), a severity-0.20 fault whose resonance sits at 1600 Hz
    (outside DEFAULT_BAND = 3-6 kHz) is missed — the selector falls back to
    the default band instead of the band the fault is actually in. This
    passed on old code too (it is measuring the bug, not the fix) and must
    keep passing after the fix, since the fix is calibration, not raising
    detection at the OLD floor."""
    pair = make_pair(seed=42, duration_s=30.0, fs=FS, severity=0.20)
    band, crest = select_demodulation_band(pair["faulty"], FS, crest_floor=DEFAULT_CREST_FLOOR)
    assert band == DEFAULT_BAND, (
        f"expected the default floor to still miss this fault (band {band}) — "
        f"if this now finds the true band, DEFAULT_CREST_FLOOR itself changed, "
        f"which is not what T1.13 does")
    assert crest < DEFAULT_CREST_FLOOR


def test_calibrated_floor_recovers_the_f19_fault():
    """THE FIX, pinned directly: using a floor calibrated from this machine's
    OWN healthy learn period (not the global constant), the same severity-0.20
    fault crosses the floor and the selector finds the TRUE band, containing
    the 1600 Hz resonance — not DEFAULT_BAND. This must fail against
    pre-T1.13 code, where `calibrate_crest_floor` does not exist."""
    from baseline import calibrate_crest_floor

    learn_crest = _healthy_learn_crests(n=48, seed0=1)
    floor = calibrate_crest_floor(learn_crest)
    assert floor < DEFAULT_CREST_FLOOR, (
        "calibration should tighten the floor below the old constant for a "
        "machine whose own healthy crest sits well under 10")

    pair = make_pair(seed=42, duration_s=30.0, fs=FS, severity=0.20)
    band, crest = select_demodulation_band(pair["faulty"], FS, crest_floor=floor)
    assert band != DEFAULT_BAND, (
        f"calibrated floor {floor:.3f} still fell back to DEFAULT_BAND "
        f"(crest {crest:.3f}) — the fault should now be found")
    resonance_hz = pair["resonance_hz"]
    assert band[0] <= resonance_hz <= band[1], (
        f"chose {band}, resonance at {resonance_hz} Hz")


# ----------------------------------------------------------------------------
# calibrate_crest_floor: unit behaviour
# ----------------------------------------------------------------------------

def test_calibration_never_exceeds_the_old_default():
    """Safety direction, same philosophy as T3.7/the threshold `min()` rule:
    calibration may only make the gate MORE willing to look at a candidate
    band, never less — so it must never exceed DEFAULT_CREST_FLOOR, even for
    an unusually loud healthy learn period."""
    from baseline import calibrate_crest_floor

    loud_healthy = np.full(48, 50.0)   # pathological: "healthy" crest of 50
    assert calibrate_crest_floor(loud_healthy) <= DEFAULT_CREST_FLOOR


def test_calibration_has_a_hard_lower_bound():
    """F19's own sweep measured floor=6.0 causing 12/14 healthy machines to
    wrongly pick a band. Calibration must not be able to reach that regime
    even from a pathologically quiet learn period."""
    from baseline import calibrate_crest_floor, MIN_CREST_FLOOR

    quiet_healthy = np.full(48, 0.1)
    assert calibrate_crest_floor(quiet_healthy) >= MIN_CREST_FLOOR


def test_calibration_falls_back_to_default_when_data_is_sparse_or_degenerate():
    """T3.7 pattern: a caller with too little or non-finite data gets the
    old, well-tested constant rather than a value fitted to noise."""
    from baseline import calibrate_crest_floor

    assert calibrate_crest_floor(np.array([6.0, 6.1, 6.2])) == DEFAULT_CREST_FLOOR   # n < 8
    assert calibrate_crest_floor(np.array([])) == DEFAULT_CREST_FLOOR
    degenerate = np.array([6.0] * 40 + [np.nan, np.inf, -np.inf])
    # still >= 8 finite samples once the bad ones are dropped -> should NOT
    # silently fall back; but must never raise or propagate a NaN floor
    floor = calibrate_crest_floor(degenerate)
    assert np.isfinite(floor)


def test_calibration_is_monotone_in_the_learn_periods_own_ceiling():
    """A machine whose learn period is quieter should get a lower (or equal)
    floor than one whose learn period is louder — calibration tracks the
    machine, it does not invert it."""
    from baseline import calibrate_crest_floor

    quiet = np.random.default_rng(0).normal(5.5, 0.1, 48)
    loud = np.random.default_rng(0).normal(7.0, 0.1, 48)
    assert calibrate_crest_floor(quiet) <= calibrate_crest_floor(loud)


# ----------------------------------------------------------------------------
# extract_features: crest_floor plumbing, default unchanged
# ----------------------------------------------------------------------------

def test_extract_features_default_crest_floor_is_unchanged():
    """No caller that omits crest_floor should see ANY behaviour change —
    this is what makes the change safe to ship without re-verifying every
    existing DOC_STATUS row that used extract_features implicitly."""
    pair = make_pair(seed=7, duration_s=30.0, fs=FS, severity=0.20)
    accel3 = np.zeros((len(pair["faulty"]), 3))   # mic-only stand-in, dead accel channel
    out_default = extract_features(pair["faulty"], FS, accel3, FS)
    out_explicit = extract_features(pair["faulty"], FS, accel3, FS,
                                    crest_floor=DEFAULT_CREST_FLOOR)
    assert out_default["band"] == out_explicit["band"]
    assert out_default["band_crest"] == out_explicit["band_crest"]
    assert np.array_equal(out_default["vector"], out_explicit["vector"])


def test_extract_features_accepts_a_lower_crest_floor_and_it_changes_the_band():
    pair = make_pair(seed=42, duration_s=30.0, fs=FS, severity=0.20)
    accel3 = np.zeros((len(pair["faulty"]), 3))
    out_default = extract_features(pair["faulty"], FS, accel3, FS)
    out_calibrated = extract_features(pair["faulty"], FS, accel3, FS, crest_floor=7.5)
    assert out_default["band"] == DEFAULT_BAND
    assert out_calibrated["band"] != DEFAULT_BAND


# ----------------------------------------------------------------------------
# baseline.py plumbing: learn crest is collected, stored, and reloaded
# ----------------------------------------------------------------------------

def _tiny_learn_set(n=32, seed0=0):
    """A minimal healthy+regime learn set built the same way test_baseline.py
    builds its fixtures — real simulate.py windows, not hand-built vectors."""
    import simulate as sim
    from baseline import operating_point

    rows, ops, crests = [], [], []
    for i in range(n):
        cfg = sim.SimConfig()
        rng = np.random.default_rng(seed0 + i)
        audio = sim.normal_signal(cfg, cfg.fs_audio, rng)
        accel = sim.normal_signal(cfg, cfg.fs_accel, np.random.default_rng(seed0 + i + 1000))
        out = extract_features(audio, cfg.fs_audio, accel[:, None], cfg.fs_accel)
        rows.append(out["vector"])
        ops.append(operating_point(out["vector"], cfg.fr))
        crests.append(out["band_crest"])
    return np.array(rows), np.array(ops), np.array(crests)


def test_fit_baseline_stores_a_calibrated_crest_floor():
    from baseline import fit_baseline

    X, OP, crests = _tiny_learn_set()
    b = fit_baseline(X, OP, learn_crest=crests)
    assert "crest_floor" in b
    assert 0 < float(b["crest_floor"]) <= DEFAULT_CREST_FLOOR


def test_fit_baseline_without_learn_crest_defaults_to_the_old_constant():
    """Backward compatibility for every existing caller of fit_baseline that
    does not pass learn_crest (there are many — tools/, ml/, most of
    tests/test_baseline.py etc.) — none of them should change behaviour."""
    from baseline import fit_baseline

    X, OP, _ = _tiny_learn_set()
    b = fit_baseline(X, OP)
    assert float(b["crest_floor"]) == DEFAULT_CREST_FLOOR


def test_scorer_loads_crest_floor_with_backward_compatible_default(tmp_path):
    from baseline import fit_baseline, save_baseline
    from inference import MahalanobisScorer

    X, OP, crests = _tiny_learn_set()
    b = fit_baseline(X, OP, learn_crest=crests)
    p = tmp_path / "b.npz"
    save_baseline(p, b)
    scorer = MahalanobisScorer(p)
    assert scorer.crest_floor == pytest.approx(float(b["crest_floor"]))

    # simulate a PRE-T1.13 baseline: strip the field the way an old file
    # written before this change would never have had it
    b_old = {k: v for k, v in b.items() if k != "crest_floor"}
    p_old = tmp_path / "b_old.npz"
    save_baseline(p_old, b_old)
    scorer_old = MahalanobisScorer(p_old)
    assert scorer_old.crest_floor == DEFAULT_CREST_FLOOR
