"""
baseline.py — on-device "learn normal" (v2): regime clustering + per-regime
Gaussian + Mahalanobis thresholds. Pure NumPy/scikit-learn. No TensorFlow —
this runs comfortably in the Pi Zero 2W's 512 MB, unlike any training of a
neural model.

Why regimes (the single biggest false-alarm defence):
A machine that runs idle / loaded / high-speed occupies several distinct
'normal' islands in feature space. One Gaussian over all of them is a blob
whose centre may sit in empty space — every legitimate regime switch then
looks anomalous, the customer gets spammed, and the unit gets unplugged.
Churn risk #1 is false alarms, not missed faults. So: cluster the learn-period
windows (k-means, k chosen 1–4 by silhouette), fit one Gaussian per cluster,
and score each new window against its NEAREST regime.

Why Mahalanobis distance:
For a Gaussian baseline, the Mahalanobis distance d(x) = sqrt((x-mu)^T
Sigma^-1 (x-mu)) is the statistically correct "how many standard deviations
away is this window, accounting for feature correlations". Features that
naturally wobble together (e.g. RMS and band energies) don't double-count.

Why Ledoit-Wolf shrinkage:
With ~50-100 learn windows and 40 features, the sample covariance is
ill-conditioned (n is close to d) and its inverse explodes. Ledoit-Wolf
shrinks toward a scaled identity with an analytically optimal weight —
standard fix, no hyperparameter to tune on-device.

Threshold: the 99.5th percentile of cross-validated distances, per regime —
but NOT read off the sample. See §Threshold estimation below; taking the
sample percentile at n ~ 24 means taking the maximum, which hands the
calibration of a deployed device to its single worst learn window.

Feedback loop: windows the customer marked "this was normal" (via the
dashboard) are stored by main.py in SQLite; `--retrain` folds them back in
and refits, so one noisy week doesn't poison the unit forever.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("baseline")

K_RANGE = (2, 3, 4)
SILHOUETTE_MIN = 0.5       # in 3 physical dims a real regime split scores ~0.8+

# ...but the silhouette's NULL distribution depends on how many directions the
# operating-point cloud actually varies in, and 0.5 is only safe in 2 or more.
# Measured on single-cluster Gaussian noise — no regimes present at all — 1500
# trials, n = 48, using this module's own candidate loop:
#
#     directions of variance   median   p95     p99     max
#         1                     0.584   0.637   0.664   0.694
#         2                     0.382   0.433   0.455   0.512
#         3                     0.283   0.326   0.348   0.401
#         2 but collinear       0.584   0.641   0.660   0.702
#
# In one effective dimension the noise floor sits ABOVE 0.5, so the shipped
# threshold split pure noise in 98 % of trials. Raising it to 0.75 puts it
# above the measured null maximum (0.702) with room to spare.
#
# What that costs, stated honestly: in one dimension a GENUINE split whose
# centroid gap is 1-4x the within-regime scatter also scores 0.61-0.70 — it is
# not separable from noise by this statistic at any threshold. Clearing 0.75
# needs a gap of roughly 6x the scatter. So on a mic-only fixed-speed machine
# we will merge two operating points that overlap substantially, and that is
# the correct trade: merging costs a slightly wide Gaussian, splitting costs
# 6.3x the false alarms (measured, see MIN_REGIME_SEPARATION below).
SILHOUETTE_MIN_1D = 0.75

# A regime is an OPERATING MODE the machine spends time in, so it must have
# enough learn windows to fit a Gaussian to. Without this floor, one outlying
# learn window (a lorry outside during the learn period) forms its own
# cluster of size 1: LedoitWolf is then fitted to a single sample, its own
# distance to its own mean is exactly 0, and the regime's threshold comes out
# at **0.0** — after which every window ever assigned to that regime alarms
# unconditionally. Found by executing the T1.6 contamination case end to end;
# it is the same failure mode as F3 arriving through the clustering rather
# than the threshold. 8 matches the intent of the existing len(OPz) >= k*8
# guard, which constrained the average per regime but not the minimum.
MIN_REGIME_WINDOWS = 8

# A SPLIT MUST BE PHYSICALLY REAL, NOT MERELY WELL-SHAPED  (backlog T1.9 /
# SELF-REVIEW F7).
#
# THE BUG. Silhouette measures the *shape* of a partition, not its *size*, and
# its null distribution depends on how many dimensions actually carry variance.
# Measured on pure single-cluster Gaussian noise, 48 points, 400 trials, using
# this module's own choose_k:
#
#     live dims   P(k>1)   median best silhouette
#         1        0.980            0.586
#         2        0.000            0.381
#         3        0.000            0.283
#
# So SILHOUETTE_MIN = 0.5 — tuned in 3 dims, where it is very safe — splits
# pure noise 98 % of the time when only ONE dimension varies. A mic-only build
# on a fixed-speed machine is exactly that case: fr is constant, the accel
# log-RMS is the dead-channel sentinel -9.0, and only the audio level moves.
# Measured end to end on 48 healthy mic-only windows of one unchanging machine:
# k = 2 with counts [30, 18], i.e. a regime boundary drawn through sensor noise
# 0.0002 decades wide. Over 100 bootstrap learn periods it chose k>1 EVERY
# time (k=2 66x, k=3 32x, k=4 2x) and cost 6.3x the false alarms:
# held-out healthy FPR 0.1358 +/- 0.1445 against 0.0217 +/- 0.0290 at k=1,
# because each spurious regime fits a 37-dim Gaussian to ~24 windows instead
# of 48. It also fired T1.6's learn-period-contamination warning on 14 of 200
# perfectly clean fits.
#
# THE FIX. Require the split to be big in PHYSICAL units as well as clean in
# shape. `op_scale` already defines the unit: 1.0 in OPz means "speed moved
# 5 %" or "level moved 0.1 decade", which is this file's own definition of a
# regime change. A noise split cannot manufacture that separation; a real
# idle/loaded transition clears it by an order of magnitude (measured: the
# repo's two-speed learn schedule separates its centroids by 10.0).
#
# WHAT IT COSTS. Two genuine regimes closer together than ~0.1 decade of level
# (at constant speed) are now merged into one Gaussian. That is the safe
# direction — one Gaussian over two nearly-coincident operating points is
# slightly wide, whereas two Gaussians over one operating point is a
# false-alarm machine — and the measured recovery threshold is stated in
# tests/test_regimes_miconly.py.
#
# TWO CRITERIA, BECAUSE THEY CATCH DIFFERENT THINGS. The separation gate is
# ABSOLUTE (is the machine measurably somewhere else?) and the dimension-aware
# silhouette floor is RELATIVE (is it further away than this machine normally
# wanders?). The deployed mic-only failure passes the relative test and fails
# the absolute one — its split is 1.5 sigma of the sensor noise, which looks
# convincing, but that sigma is 0.0001 decades. A machine that genuinely
# wanders 0.1 decade within one operating mode is the mirror image: the split
# passes the absolute test and fails the relative one. Neither criterion alone
# is enough, and each was added only after a measurement showed the other
# missing a case.
#
# NOT the fix: F7 proposed dropping dead dimensions before clustering. That is
# provably a no-op. A dimension that is constant contributes exactly 0 to every
# pairwise distance after `(OP - op_mean) / op_scale`, so k-means, its
# centroids and the silhouette are bit-identical with the dead column present
# or removed (pinned by test_dropping_the_dead_dimension_is_a_no_op). The
# problem was never the extra column; it was that one *live* dimension makes
# silhouette easy to satisfy.
MIN_REGIME_SEPARATION = 1.0

# Regime = OPERATING POINT, not "any cluster the data shows". We cluster in a
# 3-dim physical subspace — (fr, audio log-RMS, accel log-RMS) — because that
# is what actually changes when a machine goes idle/loaded/high-speed.
#
# We MEASURED the alternative failing: k-means on all 40 standardised dims
# recovers true regimes, but every model-selection statistic we tried
# (silhouette 0.118; resampling-stability ARI 0.36 vs 0.34 for one regime)
# cannot tell 2 regimes from 1, because 35 noise dimensions get z-scored up
# to unit scale and bury the signal. Choosing k is fragile in 40-d with ~50
# samples; it is trivial in 3 physical dims. Domain knowledge beats statistics
# you can't estimate.
OP_INDICES = (0, 4)        # audio_stat_logrms, accel_x_logrms (fr appended separately)


def operating_point(vector: np.ndarray, fr_hz: float) -> np.ndarray:
    return np.array([fr_hz, vector[OP_INDICES[0]], vector[OP_INDICES[1]]])


# ============================================================================
# Threshold estimation  (backlog T1.6 / SELF-REVIEW F3)
# ============================================================================
#
# THE PROBLEM. A learn period is 24-96 windows. `np.percentile(d, 99.5)` on
# 24 numbers is, arithmetically, the largest of them (measured p99.5/max =
# 0.989 at n=24, 0.964 even at n=480). So the alert threshold of every
# deployed unit was set by whichever single learn window happened to be
# worst. Measured on the frozen implementation, one contaminated window out
# of 24 moved the threshold by 1.47x at 12 sigma, 2.16x at 25 sigma and
# 3.30x at 50 sigma — i.e. it scales with the size of the outlier, without
# limit. One lorry reversing outside during the learn period permanently
# desensitises that unit and nothing reports it.
#
# WHAT DOES NOT WORK. The obvious analytic fix — squared Mahalanobis is
# chi-square with p degrees of freedom, so use `chi2.ppf(0.995, p)` — is
# wrong here, and we measured it rather than assuming it. Ledoit-Wolf shrinks
# the covariance toward a scaled identity, which deflates the distances:
# over 480 simulated healthy windows, mean(d^2) = 28.5 against p = 37, and a
# KS test rejects chi2_p at p = 3e-90. Deployed on 24-window learn periods it
# produced a held-out false-alarm rate of 11.0 %, against 3.8 % for the
# estimator it was meant to replace. Churn risk #1 is false alarms, so that
# "fix" would have been a serious regression.
#
# WHAT WORKS. Keep the chi-square SHAPE but fit its scale and its degrees of
# freedom from the data, using two quantiles taken from the BODY of the
# distribution (median and 75th percentile). Nothing in the tail is used, so
# no amount of tail contamination can move the fit — that is the whole
# robustness argument, and it is why the estimator is indifferent to whether
# the bad window is 12 or 50 sigma out. Over the same 480 windows the fitted
# model is not rejected (KS p = 0.62) and reproduces the pooled 99.5th
# percentile to 0.7 % (6.866 fitted vs 6.914 measured).
#
# WHAT IS DEPLOYED. min(empirical, fitted). `min` is deliberate: it
# guarantees this change can never make a device LESS sensitive than the code
# it replaces, so it cannot cause a miss that the old estimator would have
# caught. On clean learn periods the empirical value is the lower one and
# behaviour is unchanged; under contamination the fitted value takes over and
# caps the damage. Measured at n=24: with one 12-sigma window present, the
# old estimator's detection rate on a severity-0.01 bearing fault collapsed
# from 0.525 to 0.000, while this rule held it at 0.195.
#
# THE DISAGREEMENT IS THE DIAGNOSTIC. empirical/fitted > 1.25 means the
# sample tail is far heavier than the fitted body implies, which is what a
# contaminated learn period looks like from the inside. Measured false-flag
# rate on clean learn periods 1.0 % (n=24) / 0.5 % (n=48); catches 99.2 % of
# 12-sigma contamination and 100 % of 25-sigma. It does not catch 6-sigma
# contamination, and does not need to: at 6 sigma the empirical threshold
# only moved 1.08x.

QUANTILE = 0.995            # the spec's alert quantile, in one place
CONTAMINATION_RATIO = 1.25  # empirical/fitted above this => flag the learn period


def fit_scaled_chi2(d2: np.ndarray, ql: float = 0.5, qh: float = 0.75):
    """Fit d^2 ~ c * chi2_nu using only two low quantiles. Returns (c, nu).

    The ratio of two quantiles of chi2_nu is free of the scale c and is
    monotone in nu (as nu grows the distribution becomes symmetric and the
    ratio tends to 1), so nu is recoverable from the observed ratio by
    bisection; c then follows from either anchor.

    Returns None when the shape is not estimable — a constant or degenerate
    sample. The caller must fall back rather than propagate a NaN threshold
    onto a device.
    """
    from scipy import stats
    from scipy.optimize import brentq

    d2 = np.asarray(d2, dtype=float)
    if d2.size < 5 or not np.all(np.isfinite(d2)):
        return None
    lo, hi = np.quantile(d2, [ql, qh])
    if lo <= 0 or hi <= lo:
        return None                      # constant, zero, or inverted sample

    target = lo / hi

    def f(nu):
        return stats.chi2.ppf(ql, nu) / stats.chi2.ppf(qh, nu) - target

    a, b = 1.0, 5000.0
    if f(a) * f(b) > 0:
        return None                      # ratio outside the family's range
    nu = brentq(f, a, b, xtol=1e-3)
    return float(lo / stats.chi2.ppf(ql, nu)), float(nu)


def analytic_threshold(d: np.ndarray, q: float = QUANTILE) -> float:
    """The q-quantile of |d| extrapolated from a robust scaled-chi2 fit."""
    from scipy import stats

    fit = fit_scaled_chi2(np.asarray(d, dtype=float) ** 2)
    if fit is None:
        return float(np.percentile(d, q * 100))   # nothing better available
    c, nu = fit
    return float(np.sqrt(c * stats.chi2.ppf(q, nu)))


def choose_threshold(d: np.ndarray, q: float = QUANTILE) -> tuple[float, dict]:
    """Compute both estimates, deploy the safer, and report the disagreement.

    Returns (threshold, diagnostics). `diagnostics["contaminated"]` True means
    the learn period contains windows the fitted body cannot account for —
    the machine was probably not running normally throughout, and the honest
    response is to relearn, not to trust the number.
    """
    emp = float(np.percentile(d, q * 100))
    ana = analytic_threshold(d, q)
    ratio = emp / ana if ana > 0 else float("inf")
    return min(emp, ana), {
        "empirical": emp,
        "analytic": ana,
        "ratio": ratio,
        "contaminated": bool(ratio > CONTAMINATION_RATIO),
    }


def centroid_separation(OPz: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Smallest distance between any two regime centroids, in OPz units.

    OPz units are physical by construction (see `fit_baseline`'s `op_scale`):
    1.0 == 5 % of shaft speed, or 0.1 decade of level, or the Pythagorean
    combination. This is the number that says whether a candidate split
    corresponds to the machine doing something different, as opposed to a
    tidy-looking cut through sensor noise. Returns inf for k == 1.
    """
    C = np.array([OPz[labels == r].mean(axis=0) for r in range(k)])
    gaps = [float(np.linalg.norm(C[i] - C[j]))
            for i in range(k) for j in range(i + 1, k)]
    return min(gaps) if gaps else float("inf")


def effective_dims(OPz: np.ndarray, frac: float = 0.05) -> int:
    """How many directions the operating-point cloud actually varies in.

    Counts singular values of the centred cloud that reach `frac` of the
    largest. This is deliberately NOT a count of non-constant columns: two
    columns that move together (audio and accelerometer level on a machine
    whose load changes) span one direction, and behave exactly like one
    dimension for the silhouette null — measured: median best silhouette
    0.584 for a genuinely 1-D cloud and 0.584 for a collinear 2-column one.
    A direction holding under 5 % of the leading spread cannot host a regime
    split worth honouring, so it is not counted.
    """
    OPz = np.asarray(OPz, dtype=float)
    if OPz.ndim != 2 or len(OPz) < 2:
        return 0
    s = np.linalg.svd(OPz - OPz.mean(axis=0), compute_uv=False)
    return int((s >= frac * s.max()).sum()) if s.max() > 0 else 0


def silhouette_floor(OPz: np.ndarray) -> float:
    """The silhouette a split must reach, given the cloud's dimensionality."""
    return SILHOUETTE_MIN_1D if effective_dims(OPz) <= 1 else SILHOUETTE_MIN


def choose_k(OPz: np.ndarray) -> tuple[int, np.ndarray]:
    """Pick k in 1..4 by silhouette on standardised operating points."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_s, best_labels = 1, -1.0, np.zeros(len(OPz), dtype=int)
    for k in K_RANGE:
        if len(OPz) < k * 8:               # need a minimum of windows per regime
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(OPz)
        if len(set(km.labels_)) < 2:
            continue        # degenerate fit (e.g. constant operating point):
                            # silhouette is undefined and k>1 is meaningless
        if np.bincount(km.labels_, minlength=k).min() < MIN_REGIME_WINDOWS:
            continue        # a cluster this small is an outlier, not a regime;
                            # see MIN_REGIME_WINDOWS
        if centroid_separation(OPz, km.labels_, k) < MIN_REGIME_SEPARATION:
            continue        # well-shaped but physically nothing happened;
                            # see MIN_REGIME_SEPARATION
        s = silhouette_score(OPz, km.labels_)
        if s > best_s:
            best_k, best_s, best_labels = k, s, km.labels_
    if best_s < silhouette_floor(OPz):
        return 1, np.zeros(len(OPz), dtype=int)
    return best_k, best_labels


# ============================================================================
# Per-machine crest_floor calibration  (backlog T1.13 / SELF-REVIEW F19)
# ============================================================================
#
# THE PROBLEM. `features.select_demodulation_band`'s `crest_floor = 10.0` is a
# global constant that decides whether the protrugram trusts its own best
# candidate band or falls back to DEFAULT_BAND (3-6 kHz). Measured on a
# realistic PINK noise floor (ml/realdata/synth_phone_recording, deliberately
# resonant OUTSIDE DEFAULT_BAND): the constant rejects 13 of 14 synthetic
# machines' severity-0.20 faults, which then get demodulated in the wrong
# band and vanish. `ml/simulate.py`'s WHITE noise floor never surfaces this —
# its healthy crest tops out around 6.7 and its faults jump to 17+ by
# severity 0.05 — which is why the bug went unnoticed until F18/F19 built a
# realistic generator.
#
# THE OBVIOUS FIX DOES NOT WORK. Lowering the constant helps but cannot be
# right for every machine: healthy crest across 14 independent synthetic
# machines spans 5.56-7.33, severity-0.20 fault crest spans 6.56-10.21 —
# they OVERLAP, so no single global floor separates all of them (F19).
#
# THE FIX. Calibrate the floor PER MACHINE from its own learn period, exactly
# as `choose_threshold` already calibrates the anomaly threshold from the
# same learn period instead of a fixed constant. Measured on one machine's
# own 48-window learn period (pink noise, this module's synth generator, the
# repo's real 16 kHz audio rate):
#
#     healthy crest (n=48): min 5.38  p50 6.23  p95 7.26  p99 7.70  max 7.73
#
# `p99(healthy) + CREST_FLOOR_MARGIN` with a margin of 0.3 gives a floor of
# ~8.0 for this machine. Measured against 30 held-out fault trials at the
# SAME severity (0.20) and 30 held-out healthy trials, both disjoint seeds
# from calibration:
#
#     held-out healthy false-band-pick rate:  0.000  (n=30)
#     severity-0.20 faults caught:             0.833  (vs ~0.07 at the old
#                                                        fixed floor of 10.0)
#
# and on `ml/simulate.py`'s WHITE noise (the deployed baseline's own
# generator, so this is the "did we break what already worked" check):
#
#     calibrated floor:                        ~7.0 (vs fixed 10.0)
#     held-out healthy false-band-pick rate:    0.01  (n=100)
#     severity 0.02 detection (band-pick only): 0.13  (vs 0.00 at floor=10)
#     severity 0.05+ detection:                 1.00  either way (unchanged)
#
# i.e. the calibrated floor is a strict improvement on the realistic (pink)
# generator and a small, measured, non-zero-cost improvement on the
# already-tested white-noise path — it does not regress anything severity
# 0.05 and up already caught.
#
# WHAT IS SHIPPED. `min(DEFAULT_CREST_FLOOR, ...)` — same "can never regress"
# direction as T3.7/the threshold `min()` rule: calibration can only make the
# gate MORE willing to look at a non-default band, never less, so a machine
# whose own noise floor happens to be loud keeps exactly today's behaviour.
# `max(MIN_CREST_FLOOR, ...)` bounds the other side: F19's own sweep measured
# floor=6.0 causing 12/14 healthy machines to wrongly pick a band, so
# calibration is not allowed to reach that regime even from a pathologically
# quiet learn period.
#
# AN EARLIER VERSION OF THIS FIX LEFT X_train AT THE OLD FLOOR AND REGRESSED.
# The first attempt calibrated the floor from the learn period but kept
# `X_train` (and therefore the fitted means/covariances) extracted at the
# unchanged DEFAULT_CREST_FLOOR, reasoning that "almost every learn window
# stays in DEFAULT_BAND at either floor, so the mismatch is cheap". Running
# the actual STAGE 3 GATE (`ml/evaluate.py`) instead of trusting that
# reasoning found it was wrong: `deployed_threshold_fpr` moved from the
# documented 0.0 (docs/RUN_IT.md) to **0.107**. The mechanism is exactly what
# the crest_floor gate exists to prevent — "band-hopping would inflate the
# baseline covariance for no information gain" — except the inflation now
# ran the other way: a handful of held-out healthy windows crossed the LOWER
# calibrated floor at score time and picked a real (non-default) band, whose
# envelope features the fitted Gaussian had never seen, because every
# training window was forced through DEFAULT_BAND at floor=10 regardless.
# Train-time and score-time were silently using different floors.
#
# THE FIX. `collect_features` now runs two passes: the first measures each
# learn window's raw crest cheaply (band selection only, not the full 37-dim
# vector) to compute the calibration; the second re-extracts every learn
# window's FEATURE VECTOR at that calibrated floor, so the Gaussian this
# baseline fits is trained on exactly the band-selection behaviour it will
# be scored against. This is why fit_baseline's docstring warns the deployed
# thresholds WILL move on retrain (unlike, say, T1.8's axis-0 change, which
# was provably bit-identical) — a few learn windows now legitimately pick a
# non-default band that they did not before.
CREST_FLOOR_MARGIN = 0.7    # T1.14 / F20: raised from 0.3, which left the
                            # STAGE 3 deployed_threshold_fpr at 0.107. At 0.7
                            # it is 0.000 and fault recovery is UNCHANGED at
                            # every severity tested (6/6, 4/6, 0/6 at severity
                            # 0.35/0.20/0.10 — identical at margins 0.3, 0.7
                            # and 1.0). Reproduce with:
                            #   python tools/sweep_crest_margin.py \
                            #     --margins 0.3 0.7 1.0 --severity 0.35 0.20 0.10
                            # An earlier comment here claimed this cost one
                            # machine's recovery (5/6 -> 4/6). That was from an
                            # uncommitted script and DOES NOT REPRODUCE; there
                            # is no trade. See F20's retraction in
                            # docs/DOC_SELF_REVIEW.md. Pinned by
                            # tests/test_evaluate_pinned.py.
                            # majority of same-severity faults on pink noise
MIN_CREST_FLOOR = 6.5       # F19: floor=6.0 already gives 12/14 false picks
MIN_CREST_SAMPLES = 8       # below this, fall back rather than fit noise


def calibrate_crest_floor(learn_crest: np.ndarray) -> float:
    """Per-machine crest_floor for `features.select_demodulation_band`.

    `learn_crest`: the raw best-candidate-band crest measured on each learn
    window (`extract_features(...)["band_crest"]`), BEFORE any floor is
    applied — `select_demodulation_band` always returns the true best crest
    regardless of `crest_floor`; the floor only decides which band gets used.

    Falls back to `features.DEFAULT_CREST_FLOOR` (the pre-T1.13 constant) if
    there is too little data or none of it is finite — the T3.7 pattern:
    absence of a good calibration is not evidence the default is wrong."""
    from features import DEFAULT_CREST_FLOOR

    lc = np.asarray(learn_crest, dtype=float)
    lc = lc[np.isfinite(lc)]
    if lc.size < MIN_CREST_SAMPLES:
        return DEFAULT_CREST_FLOOR
    # MAX, not percentile(99) — T1.14 / SELF-REVIEW F20.
    #
    # The first version used p99 + margin. With a 48-window learn period the
    # 99th percentile sits BELOW the largest sample, so the loudest healthy
    # window the machine actually produced clears its own floor. Measured on
    # evaluate.py's 28 healthy windows: crest max 7.41 against a p99-derived
    # floor of 7.073, so 1 window selected a different band (1000-1402 Hz
    # instead of 3000-6000). That one window is enough — it shifts the feature
    # vector for reasons unrelated to the machine, and the deployed-threshold
    # false-alarm rate went 0.000 -> 0.107.
    #
    # The floor must sit above everything the machine did while healthy, not
    # above 99% of it.
    #
    # ⚠ THE MAX RULE ALONE DID NOT FIX IT, and that is the interesting part.
    # It moved the floor 7.073 -> 7.089 and the FPR stayed at 0.107, because
    # the learn period's max (6.79) cannot bound windows it never saw — the
    # SCORED healthy windows reached 7.41. No floor fitted to a finite sample
    # ever bounds the next sample. The margin is what absorbs that, so it is
    # doing the real work and `CREST_FLOOR_MARGIN` was raised 0.3 -> 0.7.
    #
    # Measured trade-off, sweeping the stored floor against `ml/evaluate.py`'s
    # FPR and F19's 6-machine resonance recovery:
    #
    #     floor   FPR     F19 recovery
    #     7.089   0.107   5/6      <- p99/max rule with the old 0.3 margin
    #     7.500   0.000   4/6      <- max + 0.7, what ships
    #     8.500   0.000   1/6
    #    10.000   0.000   0/6      <- the pre-T1.13 constant
    #
    # 4/6 recovery at zero false alarms beats 5/6 at a 10.7 % false-alarm
    # rate: the project's risk assessment (not in this public copy) names alarm fatigue as churn risk #1, and a
    # detector nobody trusts catches nothing at all. The margin is still a
    # global constant, but it sits above a PER-MACHINE measured maximum, which
    # is what F19 showed a global floor could never do.
    calibrated = float(np.max(lc) + CREST_FLOOR_MARGIN)
    return float(min(DEFAULT_CREST_FLOOR, max(MIN_CREST_FLOOR, calibrated)))


def fit_baseline(X: np.ndarray, OP: np.ndarray,
                 feature_names: list[str] | None = None,
                 learn_crest: np.ndarray | None = None) -> dict:
    """X: (n, 40) feature vectors; OP: (n, 3) operating points, both from an
    asserted-healthy learn period. `learn_crest` (T1.13, optional): per-window
    raw envelope-band crest from the same learn period, used to calibrate
    `select_demodulation_band`'s floor for future scoring — see
    `calibrate_crest_floor`. Omitting it (every pre-T1.13 caller) deploys
    the old fixed constant, unchanged."""
    from sklearn.covariance import LedoitWolf

    from features import DEFAULT_CREST_FLOOR

    crest_floor = (calibrate_crest_floor(learn_crest) if learn_crest is not None
                  else DEFAULT_CREST_FLOOR)

    g_mean, g_std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - g_mean) / g_std

    # Operating points are scaled by PHYSICAL units, not sample std. Sample-std
    # scaling inflates whichever dims are constant (e.g. RMS when only speed
    # changes) into unit-scale noise that buries the real split — we measured
    # silhouette dropping from ~0.99 to 0.37 exactly because of this. A regime
    # is "speed moved >~5 %" or "level moved >~0.1 decade"; those constants ARE
    # the right scale.
    op_mean = OP.mean(axis=0)
    op_scale = np.array([0.05 * max(op_mean[0], 1.0), 0.1, 0.1])
    OPz = (OP - op_mean) / op_scale

    k, labels = choose_k(OPz)
    centroids = np.array([OPz[labels == r].mean(axis=0) for r in range(k)])

    def cv_distances(Zr: np.ndarray, n_folds: int = 5) -> np.ndarray:
        """Threshold from CROSS-VALIDATED distances, not in-sample ones.

        In-sample Mahalanobis distances are biased low — the fitted Gaussian
        hugs the very points it was fitted on, and with n (~24-50) close to
        d (40) the bias is severe. We measured the consequence: an in-sample
        99.5th-percentile threshold flagged 79 % of HELD-OUT healthy windows.
        Out-of-fold scoring asks the only fair question: 'how far away does a
        normal window look to a model that has never seen it?'

        Returns the distances themselves, not a percentile of them: the
        percentile was the bug (see §Threshold estimation)."""
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(Zr))
        folds = np.array_split(idx, n_folds)
        oof = []
        for f in range(n_folds):
            tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
            lw_f = LedoitWolf().fit(Zr[tr])
            mu_f = Zr[tr].mean(axis=0)
            delta = Zr[folds[f]] - mu_f
            oof.extend(np.sqrt(np.maximum(
                np.einsum("ij,jk,ik->i", delta, lw_f.precision_, delta), 0.0)))
        return np.array(oof)

    means, precisions, counts = [], [], []
    thresholds, thr_emp, thr_ana, thr_ratio, thr_flag = [], [], [], [], []
    startup_ratios: list[float] = []   # T3.7: pooled d/threshold, one summary
                                       # per learn window, held-out where CV ran
    for r in range(k):
        Zr = Z[labels == r]
        lw = LedoitWolf().fit(Zr)          # final model: fit on ALL windows
        means.append(Zr.mean(axis=0))
        precisions.append(lw.precision_)
        if len(Zr) >= 15:
            d, inflate = cv_distances(Zr), 1.0
        else:
            # Too few windows for CV. In-sample distances are biased low, so
            # inflate by 1.5 — but estimate the quantile the same robust way,
            # because a rushed learn period is exactly where one bad window
            # does the most damage.
            d = np.sqrt(np.maximum(np.einsum(
                "ij,jk,ik->i", Zr - means[-1], lw.precision_, Zr - means[-1]), 0.0))
            inflate = 1.5
        thr, info = choose_threshold(d)
        final_thr = thr * inflate
        thresholds.append(final_thr)
        thr_emp.append(info["empirical"] * inflate)
        thr_ana.append(info["analytic"] * inflate)
        thr_ratio.append(info["ratio"])
        thr_flag.append(info["contaminated"])
        startup_ratios.extend((d / max(final_thr, 1e-9)).tolist())
        if info["contaminated"]:
            log.warning(
                "regime %d: learn period looks CONTAMINATED — the empirical "
                "99.5th percentile (%.3f) is %.2fx the robust fit (%.3f). "
                "Something abnormal was running during the learn period; the "
                "safer (lower) threshold has been deployed, but the honest "
                "fix is to relearn on a quiet machine.",
                r, info["empirical"], info["ratio"], info["analytic"])
        counts.append(int(len(Zr)))

    return {
        "created": time.time(),
        "n_windows": int(len(X)),
        "k": int(k),
        "global_mean": g_mean, "global_std": g_std,
        "op_mean": op_mean, "op_scale": op_scale, "op_centroids": centroids,
        "means": np.array(means), "precisions": np.array(precisions),
        "thresholds": np.array(thresholds), "counts": np.array(counts),
        # Diagnostics, not inputs to scoring. Kept so that a unit whose alerts
        # look wrong months later can be asked "was the learn period clean?"
        # without re-running it.
        "thresholds_empirical": np.array(thr_emp),
        "thresholds_analytic": np.array(thr_ana),
        "threshold_ratios": np.array(thr_ratio),
        "threshold_contaminated": np.array(thr_flag, dtype=bool),
        "feature_names": list(feature_names or []),
        "X_train": X, "OP_train": OP,   # kept so --retrain can refit with feedback
        # T1.13: per-machine crest_floor for select_demodulation_band, applied
        # to windows scored AFTER this baseline is fit (see MahalanobisScorer
        # and main.py). Equal to features.DEFAULT_CREST_FLOOR whenever
        # learn_crest was not supplied — every caller predating T1.13 deploys
        # unchanged.
        "crest_floor": crest_floor,
        # T3.7: a fingerprint of the learn period's own score/threshold
        # distribution, pooled across regimes. Not used to compute scores —
        # used by MahalanobisScorer at startup to sanity-check that INCOMING
        # windows still look like they were generated by the same feature
        # contract this baseline was fit under (see docstring on
        # BaselineMismatchError in inference.py for why: T1.8 measured a
        # firmware change that left the feature vector's DIMENSION unchanged
        # but silently invalidated a deployed baseline — 100% of fresh
        # healthy windows scored a median 138.4x their threshold, and
        # nothing distinguished that from a real fault appearing in the
        # unit's very first windows).
        "startup_ratio_median": float(np.median(startup_ratios)),
        "startup_ratio_p95": float(np.percentile(startup_ratios, 95)),
    }


def save_baseline(path: Path, b: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in b.items() if k != "feature_names"},
                        feature_names=np.array(b["feature_names"], dtype=object))


def load_baseline(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------
# CLI: initial learn (simulated or hardware) and feedback retrain
# ----------------------------------------------------------------------------

def collect_features(source, fs_audio: int, fs_accel: int, n_windows: int,
                     progress=log.info) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, OP, learn_crest). `learn_crest` (T1.13) is the raw
    envelope-band crest measured on each learn window; pass it to
    `fit_baseline(..., learn_crest=...)` so the SAME calibrated floor gets
    recorded in the saved baseline for use at score time.

    Two passes, not one — see the "AN EARLIER VERSION OF THIS FIX... REGRESSED"
    comment above `calibrate_crest_floor`: the first pass measures crest only
    (cheap — band selection, not the full 37-dim vector) to compute the
    calibration; the second re-extracts every learn window's feature vector
    AT that calibrated floor, so `X` (and the Gaussian fit_baseline fits to
    it) reflects the exact band-selection behaviour that will be used to
    score future windows. Extracting at two different floors for train vs.
    score measurably regressed the STAGE 3 GATE false-positive rate; this is
    the fix for that, not an unrelated refactor."""
    from features import extract_features, select_demodulation_band

    windows = []
    crests = []
    for i, (audio, accel) in enumerate(source.windows()):
        if i >= n_windows:
            break
        windows.append((audio, accel))
        _, c = select_demodulation_band(audio, fs_audio)
        crests.append(c)
        progress(f"learn window {i + 1}/{n_windows} (pass 1/2: crest)")
    crests = np.array(crests)
    crest_floor = calibrate_crest_floor(crests)

    rows, ops = [], []
    for i, (audio, accel) in enumerate(windows):
        out = extract_features(audio, fs_audio, accel, fs_accel, crest_floor=crest_floor)
        rows.append(out["vector"])
        ops.append(operating_point(out["vector"], out["fr_hz"]))
        progress(f"learn window {i + 1}/{n_windows} (pass 2/2: features @ floor {crest_floor:.2f})")
    return np.array(rows), np.array(ops), crests


def main() -> None:
    from capture import make_source
    from config_schema import ConfigError, load_config
    from features import FEATURE_NAMES
    from state import StateDB

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--windows", type=int, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "baseline.npz")
    p.add_argument("--retrain", action="store_true",
                   help="refit folding in customer 'this was normal' feedback windows")
    p.add_argument("--db", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(1)
    fs_a, fs_v = cfg["audio"]["sample_rate"], cfg["accelerometer"]["sample_rate"]

    retrain_crest_floor = None   # T1.13: set below only on the --retrain path
    if args.retrain:
        old = load_baseline(args.out)
        db = StateDB(args.db or cfg["storage"]["sqlite_path"])
        fb_vec, fb_fr = db.feedback_vectors()
        if len(fb_vec) == 0:
            print("no feedback windows recorded — nothing to retrain")
            return
        fb_op = np.array([operating_point(v, f) for v, f in zip(fb_vec, fb_fr)])
        X = np.vstack([old["X_train"], fb_vec])
        OP = np.vstack([old["OP_train"], fb_op])
        print(f"retraining: {len(old['X_train'])} original + {len(fb_vec)} feedback windows")
        # Feedback windows arrive as feature vectors only (state.py never
        # stored raw crest for them), so there is nothing to recalibrate
        # against — carry the existing baseline's own crest_floor forward
        # rather than silently resetting it to the pre-T1.13 default.
        from features import DEFAULT_CREST_FLOOR
        retrain_crest_floor = (float(old["crest_floor"]) if "crest_floor" in old
                               else DEFAULT_CREST_FLOOR)
        learn_crest = None
    else:
        if args.simulate:
            # Two-regime learn schedule: the machine at 3000 RPM and 1800 RPM.
            # Regime handling must be trained in, not bolted on.
            def schedule(i):
                return {"kind": "normal", "severity": 0.0,
                        "fr": 50.0 if (i // 8) % 2 == 0 else 30.0}
            source = make_source(cfg, simulate=True, schedule=schedule)
        else:
            source = make_source(cfg, simulate=False)
        n = args.windows or cfg["window"]["learn_windows"]
        X, OP, learn_crest = collect_features(source, fs_a, fs_v, n)

    b = fit_baseline(X, OP, list(FEATURE_NAMES), learn_crest=learn_crest)
    if retrain_crest_floor is not None:
        b["crest_floor"] = retrain_crest_floor
    save_baseline(args.out, b)
    print(json.dumps({
        "k_regimes": b["k"],
        "windows_per_regime": b["counts"].tolist(),
        "thresholds": [round(t, 3) for t in b["thresholds"].tolist()],
        "thresholds_empirical": [round(t, 3) for t in b["thresholds_empirical"].tolist()],
        "thresholds_analytic": [round(t, 3) for t in b["thresholds_analytic"].tolist()],
        "empirical_over_analytic": [round(t, 3) for t in b["threshold_ratios"].tolist()],
        "learn_period_contaminated": b["threshold_contaminated"].tolist(),
        "crest_floor": round(float(b["crest_floor"]), 3),
        "saved": str(args.out),
    }, indent=2))
    if b["threshold_contaminated"].any():
        print("\n!! At least one regime's learn period looks contaminated. "
              "Relearn on a quiet machine before trusting these alerts.")


if __name__ == "__main__":
    main()
