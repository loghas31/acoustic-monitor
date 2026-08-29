"""
tests/test_severity_calibration.py — backlog T1.12 (self-review F18).

Pins the properties of `ml/sensitivity/calibrate_severity_scales.py` that the
T1.12 write-up in `docs/DOC_SENSITIVITY.md` depends on, so a future change to
either signal generator can't silently invalidate that finding without a test
failing first. Full detection sweeps are expensive (baseline fit + score per
point), so this file uses small n_learn/n_test — enough to prove the wiring
and the qualitative finding, not to re-measure the full report (that is
recorded, with its own larger sample sizes, in the docs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "firmware"), str(_ROOT / "ml"),
          str(_ROOT / "ml" / "realdata"), str(_ROOT / "ml" / "sensitivity")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calibrate_severity_scales import (band_rms_db, detect_phone,          # noqa: E402
                                       detect_simulate, invert_curve,
                                       phone_curve, simulate_curve)
from simulate import SimConfig, normal_signal                              # noqa: E402
from synth_phone_recording import make_pair                                # noqa: E402


# ----------------------------------------------------------------------------
# band_rms_db itself
# ----------------------------------------------------------------------------

def test_band_rms_db_is_zero_for_identical_signals():
    cfg = SimConfig(duration_s=2.0)
    rng = np.random.default_rng(3)
    x = normal_signal(cfg, cfg.fs_audio, rng)
    assert band_rms_db(x, x, cfg.fs_audio, cfg.resonance_hz, cfg.resonance_q) == pytest.approx(0.0, abs=1e-9)


def test_band_rms_db_increases_with_added_in_band_energy():
    """A pure tone added inside the band must raise the measured dB; the same
    tone added far outside the band must not (checks the filter is actually
    selective, not just measuring total signal power)."""
    fs = 16000.0
    n = int(2.0 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    healthy = 0.05 * rng.standard_normal(n)
    in_band = healthy + 0.5 * np.sin(2 * np.pi * 4500 * t)      # inside (4500, q=30)
    out_of_band = healthy + 0.5 * np.sin(2 * np.pi * 200 * t)   # far outside

    db_in = band_rms_db(healthy, in_band, fs, 4500.0, 30.0)
    db_out = band_rms_db(healthy, out_of_band, fs, 4500.0, 30.0)
    assert db_in > 10.0
    assert db_out < db_in - 10.0


# ----------------------------------------------------------------------------
# Calibration curves: monotone in severity, and NOT simply proportional to
# each other (the F18 "same scale" hypothesis is not what's actually going on)
# ----------------------------------------------------------------------------

def test_simulate_curve_is_monotone_increasing_in_severity():
    curve = simulate_curve([0.02, 0.05, 0.10, 0.20], n_seeds=1)
    dbs = [r["band_rms_db"] for r in curve]
    assert dbs == sorted(dbs)
    assert dbs[-1] - dbs[0] > 10.0   # a real, large swing across the range


def test_phone_curve_is_monotone_increasing_in_severity():
    curve = phone_curve([0.05, 0.10, 0.20, 1.0, 2.0], n_seeds=1)
    dbs = [r["band_rms_db"] for r in curve]
    assert dbs == sorted(dbs)


def test_the_two_severity_scales_are_not_related_by_a_constant_factor():
    """F18 posed two hypotheses: (1) the scales are simply ~10x apart, or
    (2) pink noise masks faults a white floor does not. This test pins the
    measurement that rules out (1): if `phone_severity = k * sim_severity`
    for ANY constant k, then band_rms_db(phone, k*s) should track
    band_rms_db(sim, s) with a roughly constant OFFSET (dB scales
    logarithmically, so a fixed multiplicative severity ratio shows up as a
    fixed dB offset). It does not — the offset itself changes by tens of dB
    across the tested range, because the phone generator's own healthy
    reference already carries resonance-band energy (the deliberate
    `shared_knock_ring`, see synth_phone_recording.py's docstring) that
    ml/simulate.py's healthy signal has none of. See docs/DOC_SENSITIVITY.md
    T1.12 for the full measurement and interpretation."""
    sim_low = simulate_curve([0.02], n_seeds=1)[0]["band_rms_db"]     # ~3.9 dB
    sim_high = simulate_curve([0.20], n_seeds=1)[0]["band_rms_db"]    # ~21.6 dB
    # phone severities chosen once (by inspection) to be a plausible "k*s" pair
    # under a naive 10x hypothesis (sim 0.02 -> phone 0.2, sim 0.20 -> phone 2.0)
    phone_low = phone_curve([0.20], n_seeds=1)[0]["band_rms_db"]      # ~0.5-0.6 dB
    phone_high = phone_curve([2.0], n_seeds=1)[0]["band_rms_db"]      # ~11-12 dB

    offset_low = sim_low - phone_low
    offset_high = sim_high - phone_high
    # If it were a constant multiplicative rescaling the two offsets would be
    # nearly equal. Measured (2026-08-26): offset_low ~3.4 dB, offset_high
    # ~9.7 dB, a ~6.3 dB drift - not the same number, so a constant factor is
    # not what is happening. Threshold set below the measured drift with
    # margin, not at it, so this doesn't flap on ordinary rng noise.
    assert abs(offset_low - offset_high) > 5.0


def test_invert_curve_interpolates_within_range_and_refuses_outside_it():
    curve = simulate_curve([0.02, 0.05, 0.10, 0.20], n_seeds=1)
    lo, hi = curve[0]["band_rms_db"], curve[-1]["band_rms_db"]
    mid_target = (lo + hi) / 2
    got = invert_curve(curve, mid_target)
    assert got is not None
    assert curve[0]["severity"] <= got <= curve[-1]["severity"]
    assert invert_curve(curve, hi + 100.0) is None
    assert invert_curve(curve, lo - 100.0) is None


# ----------------------------------------------------------------------------
# Detection: the phone generator recovers strong detection ABOVE its own
# knock floor, and is not systematically worse than the white generator once
# matched at the same physical band_rms_db - pins the T1.12 headline finding.
# ----------------------------------------------------------------------------

def test_phone_detection_is_near_chance_below_its_own_knock_floor():
    """Severity 0.05-0.10 sits well below the ~0.15 knock-ring amplitude
    `synth_phone_recording.make_pair` adds to BOTH healthy and faulty signals
    - the fault should not be reliably separable from healthy yet."""
    r = detect_phone(0.05, n_learn=6, n_test_healthy=3, n_test_fault=3, seed_base=500)
    assert r["band_rms_db"] < 1.0
    assert r["fault_ratio_median"] < 2.0   # nowhere near the 10x+ margins seen once past the knock floor


def test_phone_detection_recovers_above_its_own_knock_floor():
    r = detect_phone(2.0, n_learn=6, n_test_healthy=3, n_test_fault=3, seed_base=500)
    assert r["tpr"] == 1.0
    assert r["fault_ratio_median"] > 10.0


def test_matched_band_rms_db_phone_is_not_worse_than_simulate():
    """The actual T1.12 answer: at the SAME physical band_rms_db (not the
    same severity index), the pink/phone generator is not systematically
    harder to detect than the white/simulate one - contradicting a naive
    reading of hypothesis 2 ('pink floor masks faults'). Both calls are seeded
    so this is a real, reproducible measurement, not a coin flip; a wide
    margin (2x) is asserted rather than a tight one, since both are small-n
    detection runs with real sampling noise."""
    sim_r = detect_simulate(0.02, n_learn=6, n_test_healthy=3, n_test_fault=3, seed_base=700)
    phone_r = detect_phone(1.0, n_learn=6, n_test_healthy=3, n_test_fault=3, seed_base=700)
    # band_rms_db of the two points is deliberately close (a few dB apart)
    assert abs(sim_r["band_rms_db"] - phone_r["band_rms_db"]) < 5.0
    assert phone_r["fault_ratio_median"] > 0.5 * sim_r["fault_ratio_median"]


def test_detect_functions_return_the_documented_keys():
    r = detect_phone(1.0, n_learn=6, n_test_healthy=2, n_test_fault=2, seed_base=900)
    for key in ("severity", "band_rms_db", "fpr", "tpr", "auc",
               "healthy_ratio_median", "fault_ratio_median", "k_regimes", "contaminated"):
        assert key in r
