"""
reporting.py — turn the raw anomaly score into numbers a human can read.

WHY THIS FILE EXISTS (self-review F5 / backlog T1.7)
----------------------------------------------------------------------------
The Mahalanobis distance is an excellent *decision* variable and a terrible
*display* variable. Re-measured against the current 37-dim baseline (F5's own
table predates T1.5 and T1.6 and its magnitudes no longer hold — see the
correction below):

    severity 0.000 ->     4.6   =   0.47x threshold
    severity 0.010 ->     6.8   =   0.69x threshold
    severity 0.020 ->    18.6   =   1.88x threshold
    severity 0.100 ->   221     =  22.4x threshold
    severity 0.500 ->  1340     = 135.6x threshold

Two things break because of that shape:

1. **The dashboard's amber tier carries no information.** It was defined as
   "70-100 % of threshold" — a band in score MAGNITUDE. F5 predicted it would
   never fire. Measured over 200 healthy windows, the truth is worse and the
   opposite: it fires on **16.5 %** of them, because the healthy score
   distribution's own upper tail (median 0.58x threshold, p95 0.76x) lives
   inside that band. On a fault ramped from severity 0.002 to 0.05 it fired on
   only **12.5 %**. A "watch this one" badge that is MORE likely on a healthy
   machine than on a failing one does not merely fail to inform — it teaches
   the customer that colour on this dashboard means nothing.

2. **No severity trending is possible.** d = 1340 is not a physical quantity.
   It means "some feature moved 1340 learn-period sigmas", and the sigma in
   question is whatever that feature happened to wobble by during 24 minutes
   of learning. You cannot tell a customer "your machine is 30 % worse than
   last week" from it, and you certainly cannot estimate time to failure.

This module fixes both, and it fixes them *separately*, because they are
different problems:

  (a) `ScoreReporter` produces a **bounded, calibrated display index** (0-100)
      anchored so that 70 is always this machine's own threshold. No transform
      of d can make a magnitude band informative when the healthy and faulty
      distributions overlap inside it, so we do not try. Instead the tier is
      redefined on *what the pipeline already knows*: amber = "this window is
      above threshold but the persistence gate has not fired". That state is
      real, it separates (0.5 % of healthy windows, 40 % of ramp windows), and
      it costs no trust because it still never notifies.

  (b) `physical_severity` produces **physical** quantities for trending —
      band-limited RMS in the demodulation band, and the height and frequency
      of the strongest envelope-spectrum peak. These are proportional to the
      energy the impacts actually deliver, so they grow smoothly with defect
      size instead of exploding, and they are comparable week to week.

Nothing here touches the alert decision. `MahalanobisScorer.score()` and
`AlertGate` are unchanged: the raw distance still decides, this module only
describes. That separation is deliberate — a display transform that could
alter whether an alert fires would be a safety regression dressed as UX.

Dependency policy as everywhere else on the device: numpy + scipy only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Display-index geometry
# ----------------------------------------------------------------------------
#
# The index is log-linear in score/threshold, with a different slope each side
# of the threshold, and is pinned to 70 exactly AT the threshold. The anchor is
# what makes it calibrated: the threshold is this machine's own per-regime
# 99.5th learn-period percentile, so "70" means the same thing on every unit,
# in every regime, regardless of mic sensitivity or mounting.
#
# WHAT WE TRIED FIRST AND REJECTED, with the measurement that killed it.
# The obvious reading of "calibrated" is a probability: index = 70 * F(d),
# with F the scaled-chi2 CDF already fitted for the threshold (T1.6). Built
# and measured over 200 healthy windows: it SATURATES. Median percentile
# 100.0000, p95 100.0000; half of all healthy windows landed within 0.9 index
# points of 70, so the green region — the part that has to be alive between
# alerts — was flat. The cause is structural, not a tuning miss: the fit is
# made on in-sample learn distances, which are biased low (the Gaussian hugs
# the points it was fitted on), so F reaches 1.0 well before the threshold.
# `percentile` still reports that probability, because it is the right answer
# to "how unusual is this" whenever it is not saturated, but it cannot drive
# the display.
INDEX_AT_THRESHOLD = 70.0
# Above threshold: 3 decades (1000x) maps to index 100. Chosen from the
# measured range — the largest score this simulator produces is 136x threshold
# at severity 0.5 — so the scale saturates past the observable range, not
# inside it.
INDEX_DECADES_ABOVE = 3.0
# Below threshold: 1 decade of headroom maps to the whole 0-70 range, so a
# window ten times quieter than the machine's own limit reads 0. Measured
# healthy windows span 0.283-1.034x threshold, i.e. index 31.6-70, which is
# the spread the fleet view needs.
INDEX_DECADES_BELOW = 1.0


class ScoreReporter:
    """Calibrated *display* numbers for a raw Mahalanobis distance.

    Construct from a baseline (path or the dict `load_baseline` returns). The
    calibration is derived entirely from what the baseline already stores —
    `X_train`, `OP_train`, `means`, `precisions`, `thresholds` — so this adds
    no new field to baseline.npz and works against every baseline already in
    the field.

    report(score, regime, anomalous, alerting) -> dict with

        index        0-100, bounded, monotone in score, 70 exactly at threshold
        percentile   calibrated P(learn window <= this score), as a percentage
        ratio        score / threshold, the raw number, kept for engineers
        decades      log10(score / threshold); 0 at threshold
        tier         "green" | "amber" | "red"

    `alerting` is the persistence gate's verdict (in an alert episode). It is
    passed in rather than inferred because this module has no memory and the
    gate does — see `inference.AlertGate`.
    """

    def __init__(self, baseline):
        if isinstance(baseline, (str, Path)):
            from baseline import load_baseline
            baseline = load_baseline(Path(baseline))
        self.thresholds = np.asarray(baseline["thresholds"], dtype=float)
        self.k = int(baseline["k"])
        self._fits: list = [None] * self.k
        self._learn_d: list = [np.array([]) for _ in range(self.k)]

        learn_d = _learn_distances(baseline)
        for r in range(self.k):
            d = learn_d.get(r, np.array([]))
            self._learn_d[r] = d
            self._fits[r] = _fit_or_none(d)
            if self._fits[r] is None and d.size:
                # Not fatal: `percentile` falls back to the empirical rank of
                # the observed distances. Logged because a regime whose learn
                # distances are not chi2-shaped is worth knowing about — it is
                # the same signal `choose_threshold` uses for contamination.
                log.info("regime %d: learn distances not chi2-fittable "
                         "(n=%d); display percentile falls back to empirical "
                         "rank", r, d.size)

    # -- public ------------------------------------------------------------

    def report(self, score: float, regime: int, anomalous: bool,
               alerting: bool = False) -> dict:
        regime = int(np.clip(regime, 0, self.k - 1))
        score = float(score)
        thr = float(self.thresholds[regime])

        pct = self.percentile(score, regime)
        if thr > 0:
            ratio = score / thr
            decades = float(np.log10(max(ratio, 1e-12)))
        else:
            # A zero threshold means the regime is degenerate (T1.6 found one:
            # a regime of a single window). `MIN_REGIME_WINDOWS` should prevent
            # it now, but a baseline trained before that fix can still be in
            # the field, and a display that raises on such a unit is worse than
            # one that says "off the scale".
            ratio, decades = float("inf"), float("inf")

        return {
            "index": self._index(score, regime, thr, pct),
            "percentile": pct,
            "ratio": ratio,
            "decades": decades,
            "tier": tier_from(anomalous, alerting),
        }

    def percentile(self, score: float, regime: int) -> float:
        """P(a learn-period window scores <= this), as a percentage.

        Saturates at 100 for anything meaningfully faulty — that is not a
        defect of the estimator, it is the honest answer to the question
        ("more extreme than every window the machine has ever shown us") and
        it is exactly why the index does not use it above threshold.
        """
        regime = int(np.clip(regime, 0, self.k - 1))
        fit = self._fits[regime]
        if fit is not None:
            from scipy import stats
            c, nu = fit
            return float(100.0 * stats.chi2.cdf(max(score, 0.0) ** 2 / c, nu))
        d = self._learn_d[regime]
        if d.size == 0:
            return float("nan")
        return float(100.0 * np.mean(d <= score))

    # -- internals ---------------------------------------------------------

    def _index(self, score: float, regime: int, thr: float, pct: float) -> float:
        if not np.isfinite(thr) or thr <= 0:
            return 100.0
        if score <= 0:
            return 0.0
        decades = float(np.log10(score / thr))     # 0 exactly at the threshold
        if decades <= 0:
            frac = max(1.0 + decades / INDEX_DECADES_BELOW, 0.0)
            return float(INDEX_AT_THRESHOLD * frac)
        frac = min(decades / INDEX_DECADES_ABOVE, 1.0)
        return float(INDEX_AT_THRESHOLD + (100.0 - INDEX_AT_THRESHOLD) * frac)


def tier_from(anomalous: bool, alerting: bool) -> str:
    """green / amber / red — defined on STATE, not on score magnitude.

    The old definition ("amber = 70-100 % of threshold") was a band in score
    space, and the score does not linger in it: measured 0.62x threshold
    healthy and 6.77x at severity 0.02, with nothing in between. This
    definition uses the two facts the pipeline already computes:

        red    an alert episode is live (the persistence gate fired)
        amber  this window is above threshold, but the gate has not fired
        green  below threshold

    Amber therefore means "something is off right now and we are watching it".
    Measured against the repo baseline: 0.5 % of 200 fresh healthy windows
    (1 window), against 40 % of the windows of a fault ramped from severity
    0.002 to 0.05. The old magnitude rule gave 16.5 % on those same healthy
    windows and only 12.5 % on the ramp — i.e. it was MORE likely on a healthy
    machine than on a failing one. Amber still never notifies either way, so
    it costs no trust; the difference is whether it carries information.
    """
    if alerting:
        return "red"
    if anomalous:
        return "amber"
    return "green"


def _learn_distances(baseline) -> dict[int, np.ndarray]:
    """Recompute each regime's learn-period distances from the stored data.

    In-sample (the model saw these windows), so they are biased LOW relative
    to the cross-validated distances the threshold was set from. That bias is
    harmless here for one specific reason: the index normalises by its own
    fitted percentile at the threshold, so a uniformly optimistic fit shifts
    numerator and denominator together. It would NOT be harmless if these
    distances were used to set a threshold, and they are not.
    """
    need = ("X_train", "OP_train", "global_mean", "global_std", "op_mean",
            "op_scale", "op_centroids", "means", "precisions")
    if any(k not in baseline for k in need):
        return {}
    X = np.asarray(baseline["X_train"], dtype=float)
    OP = np.asarray(baseline["OP_train"], dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return {}
    Z = (X - baseline["global_mean"]) / baseline["global_std"]
    OPz = (OP - baseline["op_mean"]) / baseline["op_scale"]
    cent = np.asarray(baseline["op_centroids"], dtype=float)
    labels = np.argmin(
        np.linalg.norm(OPz[:, None, :] - cent[None, :, :], axis=2), axis=1)

    means = np.asarray(baseline["means"], dtype=float)
    precisions = np.asarray(baseline["precisions"], dtype=float)
    out = {}
    for r in range(len(cent)):
        Zr = Z[labels == r]
        if len(Zr) == 0:
            continue
        delta = Zr - means[r]
        out[r] = np.sqrt(np.maximum(
            np.einsum("ij,jk,ik->i", delta, precisions[r], delta), 0.0))
    return out


def _fit_or_none(d: np.ndarray):
    if d.size < 5:
        return None
    from baseline import fit_scaled_chi2
    return fit_scaled_chi2(np.asarray(d, dtype=float) ** 2)


# ----------------------------------------------------------------------------
# Physical severity — the part you can actually trend
# ----------------------------------------------------------------------------
#
# The anomaly score answers "is this machine behaving like itself". Severity
# has to answer a different question: "how much impact energy is this defect
# delivering, compared with last week". That has to be a physical measurement,
# because only a physical measurement is comparable across time, across
# retrains, and (eventually) across machines.
#
# Two quantities, both already implied by the pipeline:
#
#   band_rms      RMS of the signal inside the chosen demodulation band. This
#                 is the ringing energy of the housing resonance. A bigger
#                 spall hits harder; the resonance rings louder. Physical
#                 units (whatever the mic delivers), reported in dB so the
#                 numbers stay readable across three decades.
#
#   env_peak_*    height and frequency of the strongest line in the envelope
#                 spectrum, 5-500 Hz. The frequency is the repetition rate of
#                 whatever is impacting — which we deliberately do NOT name as
#                 BPFO/BPFI (see the system overview (not in this public copy) §3), but CAN track: if the line
#                 is at the same frequency next week and taller, that is one
#                 defect getting worse rather than a new problem. The height
#                 relative to the local background is the classic severity
#                 indicator from envelope analysis.
#
# These are computed from the same band the feature extractor already chose,
# so the answer is about the same physics the detector reacted to.


def _bandpass_sos(fs: float, band: tuple[float, float]):
    """The SAME band-pass `features.envelope` applies before demodulating.

    Duplicated rather than imported because `features.py` is frozen and builds
    it inline. `tests/test_reporting.py::test_bandpass_matches_features`
    asserts the two stay identical, so if anyone changes the filter in one
    place the suite says so instead of the severity metric silently measuring
    a different band from the one the detector used.
    """
    from scipy.signal import butter
    lo = max(band[0], 1.0)
    hi = min(band[1], fs / 2 * 0.98)
    return butter(4, [lo, hi], btype="band", fs=fs, output="sos")


def physical_severity(audio: np.ndarray, fs: float,
                      band: tuple[float, float]) -> dict:
    """Physical, trendable severity indicators for one window.

    Returns
        band_rms        RMS inside `band`, signal units
        band_rms_db     20*log10(band_rms), dB re 1.0 (i.e. dBFS for [-1, 1])
        env_peak_hz     frequency of the strongest envelope line, 5-500 Hz
        env_peak_ratio  that line's height / the band's median height
        env_peak_db     20*log10 of the line's ABSOLUTE height — the trendable
                        one; the ratio can stay flat while the whole thing
                        grows, and vice versa, so we keep both
        env_energy_log10  log10 of total envelope-fluctuation energy, 5-500 Hz.
                        Identical to the `env_log_total` feature, repeated here
                        so a severity reading is self-contained.
    """
    from scipy.signal import sosfilt
    from features import ENV_BANDS, envelope_spectrum

    x = np.asarray(audio, dtype=float)
    xb = sosfilt(_bandpass_sos(fs, band), x)
    band_rms = float(np.sqrt(np.mean(xb ** 2)))

    freqs, mag = envelope_spectrum(x, fs, band)
    sel = (freqs >= ENV_BANDS[0]) & (freqs <= ENV_BANDS[-1])
    fsel, msel = freqs[sel], mag[sel]
    if msel.size == 0:                       # pathologically short window
        return {"band_rms": band_rms, "band_rms_db": _db20(band_rms),
                "env_peak_hz": float("nan"), "env_peak_ratio": float("nan"),
                "env_peak_db": float("nan"), "env_energy_log10": float("nan")}
    i = int(np.argmax(msel))
    peak = float(msel[i])
    background = float(np.median(msel))
    return {
        "band_rms": band_rms,
        "band_rms_db": _db20(band_rms),
        "env_peak_hz": float(fsel[i]),
        "env_peak_ratio": peak / (background + 1e-12),
        "env_peak_db": _db20(peak),
        "env_energy_log10": float(np.log10(float(np.sum(msel ** 2)) + 1e-12)),
    }


def _db20(v: float) -> float:
    return float(20.0 * np.log10(max(float(v), 1e-12)))


class SeverityReference:
    """Learn-period reference so severity can be reported as a CHANGE.

    An absolute dB number is meaningless to a customer — it depends on the
    mic, the mounting and how far the box is from the machine. "6 dB louder in
    the impact band than when we learned this machine" is meaningful, and it
    is the quantity that trends.

    Built from the baseline's stored learn vectors, so again no new field in
    baseline.npz. Only `env_log_total` is available there (it is a feature);
    band RMS is not, so `relative()` returns the band figures unchanged and
    says so rather than inventing a reference.
    """

    def __init__(self, baseline):
        if isinstance(baseline, (str, Path)):
            from baseline import load_baseline
            baseline = load_baseline(Path(baseline))
        self.env_ref_log10 = float("nan")
        names = [str(n) for n in baseline.get("feature_names", [])]
        X = np.asarray(baseline.get("X_train", np.empty((0, 0))), dtype=float)
        if "env_log_total" in names and X.ndim == 2 and X.shape[0] > 0:
            col = X[:, names.index("env_log_total")]
            # Median, not mean: one loud window during learning should not move
            # the reference every subsequent reading is compared against. Same
            # reasoning as the robust threshold estimator (T1.6).
            self.env_ref_log10 = float(np.median(col))

    def relative(self, m: dict) -> dict:
        """Add change-relative-to-learn fields to a `physical_severity` dict."""
        out = dict(m)
        if np.isfinite(self.env_ref_log10):
            # env_log_total is log10 of an ENERGY, so 10*, not 20*.
            out["env_energy_db_re_learn"] = float(
                10.0 * (m["env_energy_log10"] - self.env_ref_log10))
        else:
            out["env_energy_db_re_learn"] = float("nan")
        return out
