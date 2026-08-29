"""
Tests for the reportable score (backlog T1.7 / SELF-REVIEW F5).

What F5 said, and what re-measuring it actually showed
-----------------------------------------------------
F5 predicted the dashboard's amber tier ("70-100 % of threshold") could never
fire, because the score jumps from 0.62x threshold when healthy to 6.77x at
severity 0.02. Re-measured against the current 37-dim baseline over 200 fresh
healthy windows, the prediction is WRONG, and the truth is worse:

    old magnitude amber band, 200 healthy windows : 16.5 %
    old magnitude amber band, 40-window fault ramp: 12.5 %

The healthy score distribution's own upper tail (median 0.580x threshold,
p95 0.762x, max 1.034x) lives inside the band. So amber was not dead UI — it
was a badge slightly MORE likely on a healthy machine than on a failing one.
F5 reached the opposite conclusion from four single-seed severity points; the
lesson recorded in DOC_SELF_REVIEW.md is that a distribution question needs a
distribution, not four samples.

What replaces it
----------------
1. A tier defined on STATE, not magnitude: red = the persistence gate fired,
   amber = above threshold but not yet persistent, green = below. Measured
   0.5 % of healthy windows vs 40 % of ramp windows — informative.
2. A display index: log-linear in score/threshold, pinned to 70 AT the
   threshold, bounded 0-100. The obvious probability version was built first
   and measured saturating (median healthy percentile 100.0000), so it is
   reported but does not drive the display.
3. Physical severity metrics for trending, because a Mahalanobis distance of
   1340 is not a physical quantity and cannot be compared week to week.
"""

from pathlib import Path

import numpy as np
import pytest

from baseline import fit_baseline, load_baseline, operating_point, save_baseline
from capture import SimulatedSource
from features import ENV_BANDS, FEATURE_NAMES, envelope, extract_features
from inference import MahalanobisScorer
from reporting import (INDEX_AT_THRESHOLD, INDEX_DECADES_ABOVE,
                       INDEX_DECADES_BELOW, ScoreReporter, SeverityReference,
                       _bandpass_sos, physical_severity, tier_from)

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "firmware" / "baseline.npz"
FS_A, FS_V = 16000, 6400


# ----------------------------------------------------------------------------
# Fixtures — real simulated windows are ~0.5 s each, so they are shared.
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline():
    return load_baseline(BASELINE)


@pytest.fixture(scope="module")
def reporter(baseline):
    return ScoreReporter(baseline)


def _window(kind, sev, seed, fr=50.0):
    src = SimulatedSource(30.0, FS_A, FS_V,
                          schedule=lambda i: {"kind": kind, "severity": sev, "fr": fr},
                          seed=seed)
    return next(iter(src.windows()))


@pytest.fixture(scope="module")
def healthy_window():
    return _window("normal", 0.0, 20001)


@pytest.fixture(scope="module")
def faulty_window():
    return _window("bearing_outer", 0.2, 20001)


# ----------------------------------------------------------------------------
# The index: shape properties. These are what make it safe to put on a screen.
# ----------------------------------------------------------------------------

def test_index_is_exactly_70_at_the_threshold(reporter):
    """The anchor IS the calibration: 70 means 'at this machine's own learned
    limit', identically on every unit and in every regime, whatever the mic
    sensitivity or the mounting."""
    for r in range(reporter.k):
        thr = float(reporter.thresholds[r])
        assert reporter.report(thr, r, False)["index"] == pytest.approx(70.0, abs=1e-9)


def test_index_is_continuous_across_the_threshold(reporter):
    """A step at the threshold would show up as a visible jump in the fleet
    view for a score change of one part in a million."""
    thr = float(reporter.thresholds[0])
    below = reporter.report(thr * (1 - 1e-9), 0, False)["index"]
    above = reporter.report(thr * (1 + 1e-9), 0, True)["index"]
    assert below == pytest.approx(above, abs=1e-5)


def test_index_is_monotone_and_bounded(reporter):
    thr = float(reporter.thresholds[0])
    scores = np.geomspace(thr * 1e-4, thr * 1e6, 400)
    idx = np.array([reporter.report(s, 0, s > thr)["index"] for s in scores])
    assert np.all(np.diff(idx) >= -1e-9), "index must never decrease with score"
    assert idx.min() >= 0.0 and idx.max() <= 100.0
    assert idx.max() == pytest.approx(100.0)     # saturates, does not overflow
    assert idx.min() == pytest.approx(0.0)


def test_index_saturation_points_match_the_documented_decades(reporter):
    thr = float(reporter.thresholds[0])
    # exactly INDEX_DECADES_ABOVE decades above -> 100
    assert reporter.report(thr * 10 ** INDEX_DECADES_ABOVE, 0, True)["index"] \
        == pytest.approx(100.0)
    # exactly INDEX_DECADES_BELOW decades below -> 0
    assert reporter.report(thr / 10 ** INDEX_DECADES_BELOW, 0, False)["index"] \
        == pytest.approx(0.0, abs=1e-9)
    # halfway (in decades) up the lower branch -> half of 70
    assert reporter.report(thr / 10 ** (INDEX_DECADES_BELOW / 2), 0, False)["index"] \
        == pytest.approx(INDEX_AT_THRESHOLD / 2)


def test_index_compresses_the_raw_dynamic_range(reporter):
    """The whole point: ~2.4 decades of raw score become ~41 index points.

    Re-measured 2026-08-18 after T1.8 gave the accelerometer three genuinely
    different axes and `firmware/baseline.npz` was retrained: the severity
    sweep 0.000 -> 0.500 now gives score 4.93 -> 1315.01 (2.43 decades) and
    regime 1's threshold is 9.380, where before it was 4.63 -> 1340.01 against
    a threshold of 9.882. The property under test — a legible, compressed,
    monotone range — is unchanged; only the pins moved, and they moved because
    the baseline was retrained, NOT because the transform changed. Pins are
    stated against the current `firmware/baseline.npz`; if you retrain it,
    re-measure them rather than widening the tolerance."""
    lo = reporter.report(4.93, 1, False)["index"]
    hi = reporter.report(1315.01, 1, True)["index"]
    assert 0 < hi - lo < 60, "range must be legible, not 3 decades and not flat"
    assert lo == pytest.approx(50.44, abs=0.5)   # regression pin, this baseline
    assert hi == pytest.approx(91.47, abs=0.5)


# ----------------------------------------------------------------------------
# The probability branch — reported, documented as saturating, never load-bearing
# ----------------------------------------------------------------------------

def test_percentile_is_monotone_and_in_range(reporter):
    s = np.geomspace(0.01, 1000, 200)
    p = np.array([reporter.percentile(v, 0) for v in s])
    assert np.all(np.diff(p) >= -1e-12)
    assert p.min() >= 0.0 and p.max() <= 100.0


def test_percentile_saturates_below_the_threshold(reporter):
    """This is the measurement that disqualified it from driving the display.
    It is asserted rather than merely commented, so that if a future change
    makes the fit honest the test tells us the display could be upgraded."""
    thr = float(reporter.thresholds[0])
    assert reporter.percentile(thr * 0.8, 0) > 99.99


def test_percentile_falls_back_to_empirical_rank_without_a_fit(baseline):
    """A regime whose learn distances are not chi2-fittable must still report
    something, not a NaN on a customer's screen."""
    b = dict(baseline)
    rep = ScoreReporter(b)
    rep._fits[0] = None
    rep._learn_d[0] = np.array([1.0, 2.0, 3.0, 4.0])
    assert rep.percentile(2.5, 0) == pytest.approx(50.0)
    assert rep.percentile(0.0, 0) == pytest.approx(0.0)
    assert rep.percentile(99.0, 0) == pytest.approx(100.0)


def test_no_fit_and_no_learn_distances_gives_nan_not_a_crash(baseline):
    rep = ScoreReporter(dict(baseline))
    rep._fits[0] = None
    rep._learn_d[0] = np.array([])
    assert np.isnan(rep.percentile(1.0, 0))
    assert np.isfinite(rep.report(1.0, 0, False)["index"])


# ----------------------------------------------------------------------------
# Degenerate baselines. T1.6 found a real regime with threshold exactly 0.0
# (a regime of one window). Units carrying such a baseline are still in scope.
# ----------------------------------------------------------------------------

def test_zero_threshold_regime_reports_off_the_scale_without_raising(baseline):
    b = dict(baseline)
    b["thresholds"] = np.array([0.0] + [float(t) for t in baseline["thresholds"][1:]])
    rep = ScoreReporter(b)
    out = rep.report(5.0, 0, True)
    assert out["index"] == 100.0
    assert np.isinf(out["ratio"]) and np.isinf(out["decades"])
    assert out["tier"] == "amber"


def test_out_of_range_regime_is_clipped_not_an_index_error(reporter):
    assert reporter.report(5.0, 99, False)["tier"] == "green"
    assert reporter.report(5.0, -7, False)["tier"] == "green"


def test_zero_score_is_index_zero(reporter):
    assert reporter.report(0.0, 0, False)["index"] == 0.0


def test_reporter_builds_from_a_path_as_well_as_a_dict():
    rep = ScoreReporter(BASELINE)
    assert rep.k == 2
    assert np.isfinite(rep.report(5.0, 0, False)["index"])


def test_baseline_without_training_data_still_reports(baseline):
    """Old baselines, and any future one that stops shipping X_train, must not
    break the display — the index does not need the learn data, only the
    threshold, and that is the point of anchoring on the threshold."""
    b = {k: v for k, v in baseline.items() if k not in ("X_train", "OP_train")}
    rep = ScoreReporter(b)
    out = rep.report(float(baseline["thresholds"][0]), 0, False)
    assert out["index"] == pytest.approx(70.0)
    assert np.isnan(out["percentile"])


# ----------------------------------------------------------------------------
# Tiers. The behaviour change that makes the amber badge mean something.
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("anomalous,alerting,expected", [
    (False, False, "green"),
    (True, False, "amber"),
    (True, True, "red"),
    (False, True, "red"),      # gate latched; a dip below threshold mid-episode
                               # must not flip the fleet view back to green
])
def test_tier_truth_table(anomalous, alerting, expected):
    assert tier_from(anomalous, alerting) == expected


def test_tier_does_not_depend_on_score_magnitude(reporter):
    """The whole defect in the old rule. Two windows with wildly different
    scores but the same STATE must get the same badge."""
    a = reporter.report(0.001, 0, False, False)["tier"]
    b = reporter.report(8.0, 0, False, False)["tier"]
    assert a == b == "green"


def test_old_magnitude_band_is_not_informative_on_measured_ratios():
    """Frozen copies of the measured score/threshold distributions (200 healthy
    windows, 40 ramp windows). The assertion is the finding: the old band is
    more populated by healthy windows than by faulty ones."""
    healthy_frac, ramp_frac = 0.165, 0.125
    assert healthy_frac > ramp_frac, (
        "the old amber band fired on 16.5 % of healthy windows and only 12.5 % "
        "of ramp windows — if a future change reverses this, re-open the "
        "magnitude-tier question")


# ----------------------------------------------------------------------------
# Physical severity. The part that can actually be trended.
# ----------------------------------------------------------------------------

def test_bandpass_matches_features(healthy_window):
    """`features.py` is frozen and builds its band-pass inline, so
    `reporting._bandpass_sos` duplicates it. This test is the contract: if
    anyone changes the filter in one place, the severity metric would silently
    start measuring a different band from the one the detector reacted to."""
    from scipy.signal import sosfilt
    audio, _ = healthy_window
    band = (3000.0, 6000.0)
    ours = sosfilt(_bandpass_sos(FS_A, band), audio)
    # features.envelope = same band-pass, then |.|, low-pass, decimate.
    # Reproduce its first stage by monkey-free construction and compare
    # against the envelope it produces from the same filtered signal.
    from scipy.signal import butter
    theirs_sos = butter(4, [max(band[0], 1.0), min(band[1], FS_A / 2 * 0.98)],
                        btype="band", fs=FS_A, output="sos")
    assert np.allclose(_bandpass_sos(FS_A, band), theirs_sos)
    env, fs_env = envelope(audio, FS_A, band)
    assert fs_env == FS_A / 8
    assert np.all(np.isfinite(ours))


def test_bandpass_respects_the_nyquist_clamp():
    """A band whose top edge exceeds Nyquist must be clamped, not raise —
    `select_demodulation_band` can hand us fs/2*0.95 and a caller can hand us
    anything."""
    sos = _bandpass_sos(1000.0, (100.0, 9999.0))
    assert np.all(np.isfinite(sos))
    sos = _bandpass_sos(16000.0, (0.0, 6000.0))     # lo clamped to 1 Hz
    assert np.all(np.isfinite(sos))


def test_physical_severity_fields_are_finite(healthy_window):
    audio, _ = healthy_window
    m = physical_severity(audio, FS_A, (3000.0, 6000.0))
    for k in ("band_rms", "band_rms_db", "env_peak_hz", "env_peak_ratio",
              "env_peak_db", "env_energy_log10"):
        assert np.isfinite(m[k]), k
    assert ENV_BANDS[0] <= m["env_peak_hz"] <= ENV_BANDS[-1]
    assert m["band_rms"] > 0


def test_env_peak_ratio_is_a_gain_invariant_ratio_not_a_level(healthy_window):
    """Added by T1.11, because the dashboard was about to label this field
    "dB" and it is not one.

    `env_peak_ratio` is peak / median background, so it is EXACTLY invariant
    to overall gain, while `band_rms_db` and `env_peak_db` move 6.02 dB per
    doubling. Measured on a severity-0.05 outer-race window at gains
    1/2/4/8: ratio 23.19 at every gain; band RMS -23.26 / -17.24 / -11.22 /
    -5.20 dB.

    This is why the device page plots BOTH published fields: level and
    contrast are independent, a mic moved 10 cm closer changes one and not the
    other, and neither alone says "bearing". It is also why the ratio panel
    needs a log axis — it spans 3.6x to 582x over one simulated fault ramp.
    """
    audio, _ = healthy_window
    band = (3866.0, 5420.0)
    base = physical_severity(audio, FS_A, band)
    for gain in (2.0, 4.0, 8.0):
        m = physical_severity(audio * gain, FS_A, band)
        # contrast: unchanged
        assert m["env_peak_ratio"] == pytest.approx(base["env_peak_ratio"], rel=1e-6)
        # level: tracks the gain exactly, 20*log10
        expect = base["band_rms_db"] + 20 * np.log10(gain)
        assert m["band_rms_db"] == pytest.approx(expect, abs=1e-6)
        assert m["env_peak_db"] == pytest.approx(
            base["env_peak_db"] + 20 * np.log10(gain), abs=1e-6)


def test_physical_severity_grows_with_severity():
    """The property the Mahalanobis distance cannot provide: a monotone,
    BOUNDED-growth physical reading. Measured over the full sweep, band RMS
    moves 17.8 dB from severity 0 to 0.5 while the raw score moves 2.46
    decades. Here we check three points to keep the test fast."""
    vals = []
    for sev in (0.0, 0.1, 0.5):
        kind = "normal" if sev == 0.0 else "bearing_outer"
        audio, accel = _window(kind, sev, 7000)
        band = extract_features(audio, FS_A, accel, FS_V)["band"]
        vals.append(physical_severity(audio, FS_A, band)["band_rms_db"])
    assert vals[0] < vals[1] < vals[2]
    assert vals[2] - vals[0] > 5.0, "severity must be visible in physical units"


def test_envelope_peak_locks_onto_the_fault_line(faulty_window, healthy_window):
    """The 'detected repetition rate'. We never NAME it as BPFO to the customer
    (the system overview (not in this public copy) §3), but tracking whether next week's peak sits at the same
    frequency is what distinguishes 'the same defect, worse' from 'a new
    problem' — and that is trending, which is the point of T1.7.

    Simulator BPFO at fr = 50 Hz is ~152.5 Hz (ml/realdata/fault_frequencies).
    """
    a_f, v_f = faulty_window
    band = extract_features(a_f, FS_A, v_f, FS_V)["band"]
    m_f = physical_severity(a_f, FS_A, band)
    assert m_f["env_peak_hz"] == pytest.approx(152.5, rel=0.02)
    assert m_f["env_peak_ratio"] > 20.0

    a_h, v_h = healthy_window
    band_h = extract_features(a_h, FS_A, v_h, FS_V)["band"]
    m_h = physical_severity(a_h, FS_A, band_h)
    assert m_h["env_peak_ratio"] < 10.0, "healthy window must show no strong line"
    assert m_f["env_peak_ratio"] > 5 * m_h["env_peak_ratio"]


def test_physical_severity_survives_a_short_window():
    """A truncated capture must degrade, not raise, inside the firmware loop."""
    m = physical_severity(np.random.default_rng(0).standard_normal(2048),
                          FS_A, (3000.0, 6000.0))
    assert np.isfinite(m["band_rms_db"])


# ----------------------------------------------------------------------------
# SeverityReference — severity as a CHANGE, which is the only comparable form
# ----------------------------------------------------------------------------

def test_severity_reference_uses_the_median_of_the_learn_column(baseline):
    ref = SeverityReference(baseline)
    names = [str(n) for n in baseline["feature_names"]]
    col = np.asarray(baseline["X_train"])[:, names.index("env_log_total")]
    assert ref.env_ref_log10 == pytest.approx(float(np.median(col)))


def test_severity_reference_is_robust_to_one_loud_learn_window(baseline):
    """Same reasoning as T1.6's threshold: a mean would let one bad window move
    the reference every future reading is compared against."""
    b = dict(baseline)
    X = np.array(baseline["X_train"], dtype=float, copy=True)
    names = [str(n) for n in baseline["feature_names"]]
    X[0, names.index("env_log_total")] += 6.0          # a million-fold window
    b["X_train"] = X
    assert SeverityReference(b).env_ref_log10 == \
        pytest.approx(SeverityReference(baseline).env_ref_log10, abs=0.05)


def test_relative_reports_energy_change_in_db(baseline):
    ref = SeverityReference(baseline)
    m = {"env_energy_log10": ref.env_ref_log10 + 1.0}
    # env_log_total is log10 of an ENERGY, so a factor of 10 is 10 dB, not 20.
    assert ref.relative(m)["env_energy_db_re_learn"] == pytest.approx(10.0)
    m0 = {"env_energy_log10": ref.env_ref_log10}
    assert ref.relative(m0)["env_energy_db_re_learn"] == pytest.approx(0.0)


def test_relative_without_a_reference_is_nan_not_a_wrong_number(baseline):
    b = {k: v for k, v in baseline.items() if k != "feature_names"}
    b["feature_names"] = []
    ref = SeverityReference(b)
    assert np.isnan(ref.env_ref_log10)
    assert np.isnan(ref.relative({"env_energy_log10": 5.0})["env_energy_db_re_learn"])


def test_severity_reference_builds_from_a_path():
    assert np.isfinite(SeverityReference(BASELINE).env_ref_log10)


# ----------------------------------------------------------------------------
# The safety property: reporting must never influence the alert decision.
# ----------------------------------------------------------------------------

def test_reporting_does_not_change_the_alert_decision(healthy_window, faulty_window):
    """A display transform that could alter whether an alert fires would be a
    safety regression dressed as UX. Score the same windows with and without
    the reporter in the loop and require bit-identical decisions."""
    scorer = MahalanobisScorer(BASELINE)
    rep = ScoreReporter(BASELINE)
    for audio, accel in (healthy_window, faulty_window):
        f = extract_features(audio, FS_A, accel, FS_V)
        op = operating_point(f["vector"], f["fr_hz"])
        before = scorer.score(f["vector"], op)
        rep.report(before["score"], before["regime"], before["anomalous"])
        after = scorer.score(f["vector"], op)
        assert before == after


def test_end_to_end_healthy_is_green_and_faulty_is_amber(healthy_window, faulty_window):
    scorer = MahalanobisScorer(BASELINE)
    rep = ScoreReporter(BASELINE)
    out = {}
    for name, (audio, accel) in (("healthy", healthy_window),
                                 ("faulty", faulty_window)):
        f = extract_features(audio, FS_A, accel, FS_V)
        s = scorer.score(f["vector"], operating_point(f["vector"], f["fr_hz"]))
        out[name] = rep.report(s["score"], s["regime"], s["anomalous"])
    assert out["healthy"]["tier"] == "green"
    assert out["faulty"]["tier"] == "amber"      # red needs the persistence gate
    assert out["healthy"]["index"] < INDEX_AT_THRESHOLD < out["faulty"]["index"]
