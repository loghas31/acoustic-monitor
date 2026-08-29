"""
Tests for the per-regime alert threshold (backlog T1.6 / SELF-REVIEW F3).

The bug being fixed
-------------------
`fit_baseline` set each regime's alert threshold to `np.percentile(d, 99.5)`
of ~24 cross-validated distances. At that n the 99.5th percentile IS the
maximum observed value (measured ratio 0.99), so the threshold of every
deployed device was set by whichever single learn window happened to be
worst. One lorry reversing outside the factory during the learn period
permanently desensitised that unit, and nothing reported it.

What replaces it
----------------
A robustly-fitted scaled chi-square tail. For Gaussian data the squared
Mahalanobis distance is chi-square distributed, but NOT with p degrees of
freedom once Ledoit-Wolf shrinkage is applied — we measured mean(d^2) = 28.5
against p = 37, and a Kolmogorov-Smirnov test rejects chi2_p at p = 3e-90.
So the shape is fitted from two LOW quantiles of the data (median and 75th),
which no tail contamination can move, and the 99.5th percentile is
extrapolated from the fit rather than read off the sample.

Both estimates are computed, the lower is deployed (never less sensitive
than today's behaviour), and their ratio is stored: a ratio above
CONTAMINATION_RATIO means the empirical tail is far heavier than the fitted
body, which is what a contaminated learn period looks like.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.covariance import LedoitWolf

from baseline import (CONTAMINATION_RATIO, MIN_REGIME_WINDOWS, QUANTILE,
                      analytic_threshold, choose_k, choose_threshold,
                      fit_baseline, fit_scaled_chi2, load_baseline,
                      save_baseline)
from features import FEATURE_NAMES
from inference import MahalanobisScorer

D = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# helpers — a single-regime learn period in the real feature layout
# ---------------------------------------------------------------------------

RANK = 14   # see below


def make_learn(n=24, seed=0):
    """n healthy learn windows at one operating point.

    RANK-DEFICIENT ON PURPOSE. The first version of this fixture drew
    near-isotropic Gaussian features (effective rank 30 of 37) and every
    conditioning-sensitive assertion below passed for the wrong reason. The
    project's real feature vector is nothing like that: measured over 480
    simulated healthy windows its effective rank is **13.7 of 37**, which is
    the same degeneracy T1.10 found from the other direction (the simulator's
    band composition spans ~1 of 8 dimensions). A fixture that is better
    conditioned than production tests a detector we do not ship.
    """
    rng = np.random.default_rng(seed)
    root = rng.normal(0, 1, (RANK, D))              # RANK independent directions
    X = rng.normal(0, 1, (n, RANK)) @ root + 0.05 * rng.normal(0, 1, (n, D))
    OP = np.column_stack([np.full(n, 50.0) + rng.normal(0, 0.05, n),
                          rng.normal(-0.1, 0.01, n),
                          rng.normal(-0.1, 0.01, n)])
    return X, OP


def effective_rank(X):
    """exp(entropy of the correlation eigenvalue spectrum) — 'how many
    independent directions does this data really have'."""
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 1e-12, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def contaminate(X, sigmas=12.0, seed=1):
    """Push ONE learn window far out along a random direction — 'a lorry'."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    Xc = X.copy()
    Xc[0] = Xc[0] + sigmas * X.std(axis=0) * u
    return Xc


def oof_distances(X, n_folds=5, seed=0):
    """The out-of-fold Mahalanobis distances `fit_baseline` thresholds on."""
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(Z)), n_folds)
    out = []
    for f in range(n_folds):
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        lw = LedoitWolf().fit(Z[tr])
        delta = Z[folds[f]] - Z[tr].mean(axis=0)
        out.extend(np.sqrt(np.maximum(
            np.einsum("ij,jk,ik->i", delta, lw.precision_, delta), 0.0)))
    return np.array(out)


# ---------------------------------------------------------------------------
# 1. the bug itself, characterised so it can never be reintroduced silently
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [24, 48, 96])
def test_percentile_99_5_is_the_maximum_at_realistic_n(n):
    """This is WHY the estimator had to change, pinned as a fact.

    A learn period is 24-96 windows. At every one of those sizes the
    "99.5th percentile" and the maximum are the same number to within 5 %,
    so the empirical estimator has one effective degree of freedom: the
    worst window.
    """
    d = oof_distances(make_learn(n)[0])
    ratio = np.percentile(d, 99.5) / d.max()
    assert ratio > 0.95, (
        f"n={n}: p99.5/max = {ratio:.4f}; if this ever drops well below 1 the "
        "empirical estimator has become meaningful and this test's premise "
        "needs revisiting")


def test_empirical_threshold_scales_with_the_worst_window():
    """The failure mode stated precisely: the empirical threshold is not
    merely disturbed by a bad learn window, it is a monotone function of how
    bad that window is, with no ceiling. Measured on the pre-fix firmware at
    n=24: 1.47x at 12 sigma, 2.16x at 25 sigma, 3.30x at 50 sigma.

    An estimator with that property cannot be calibrated, because its value
    is set by an event nobody controls or observes.
    """
    X, _ = make_learn(24)
    clean = np.percentile(oof_distances(X), 99.5)
    got = [np.percentile(oof_distances(contaminate(X, s)), 99.5) / clean
           for s in (12.0, 25.0, 50.0)]
    assert got == sorted(got), f"expected monotone growth, got {got}"
    assert got[-1] > 2.0, (
        f"a 50-sigma window should more than double the empirical threshold; "
        f"got {got[-1]:.2f}x")


# ---------------------------------------------------------------------------
# 2. the refutation — F3's prescribed fix would have been worse
# ---------------------------------------------------------------------------

def test_naive_chi2_p_does_not_describe_our_distances():
    """SELF-REVIEW F3 proposed `chi2.ppf(0.995, p)` directly. It is wrong for
    THIS feature vector, and the reason is worth pinning down.

    The textbook result d^2 ~ chi2_p assumes p independent directions. Our
    features do not have p independent directions — effective rank 13.7 of
    37 on real simulated data. Ledoit-Wolf collapses the near-null directions
    rather than inverting them, so d^2 concentrates near the EFFECTIVE
    dimensionality, not the nominal one. Measured on 480 healthy simulator
    windows: mean(d^2) = 28.5 against p = 37, KS against chi2_37 rejected at
    p = 3e-90, and the resulting threshold gave an 11.0 % held-out
    false-alarm rate at n=24 versus 3.8 % for the estimator it was meant to
    replace.

    This is the whole reason the deployed fix FITS the degrees of freedom
    instead of asserting them.
    """
    # One seed, i.e. ONE generative process sampled 480 times. Stacking five
    # different seeds would union five different 14-dim subspaces and give an
    # effective rank of 27 — a fixture that no longer resembles a machine.
    X, _ = make_learn(480, seed=0)
    d2 = oof_distances(X) ** 2
    assert effective_rank(X) < 0.6 * D, "fixture is not rank-deficient"
    assert d2.mean() < 0.9 * D, (
        f"mean(d^2) = {d2.mean():.1f} vs p = {D}: expected d^2 to concentrate "
        "near the effective rank, well below the nominal dof")
    assert stats.kstest(d2, "chi2", args=(D,)).pvalue < 1e-6, \
        "chi2_p unexpectedly fits; the naive analytic threshold may be usable"


def test_squared_distance_tracks_effective_rank_not_feature_count():
    """The mechanism above, isolated: build data with a known number of
    independent directions and watch mean(d^2) follow it rather than p."""
    rng = np.random.default_rng(4)
    for rank in (8, 14, 24):
        root = rng.normal(0, 1, (rank, D))
        X = rng.normal(0, 1, (480, rank)) @ root + 0.05 * rng.normal(0, 1, (480, D))
        m = (oof_distances(X) ** 2).mean()
        assert abs(m - rank) < 0.35 * rank, (
            f"rank {rank}: mean(d^2) = {m:.1f}, expected near {rank}, not {D}")


# ---------------------------------------------------------------------------
# 3. the scaled chi-square fit
# ---------------------------------------------------------------------------

def test_fit_scaled_chi2_recovers_known_parameters():
    """Sanity: on data that really IS c * chi2_nu, recover c and nu."""
    rng = np.random.default_rng(0)
    c_true, nu_true = 0.7, 40.0
    d2 = c_true * rng.chisquare(nu_true, 20000)
    c, nu = fit_scaled_chi2(d2)
    assert c == pytest.approx(c_true, rel=0.10)
    assert nu == pytest.approx(nu_true, rel=0.15)
    # and the quantile it is actually used for
    got = np.sqrt(c * stats.chi2.ppf(0.995, nu))
    want = np.sqrt(c_true * stats.chi2.ppf(0.995, nu_true))
    assert got == pytest.approx(want, rel=0.05)


def test_fit_uses_only_the_body_so_the_tail_cannot_move_it():
    """The fit anchors on the median and 75th percentile. Replace the top
    10 % of the sample with arbitrarily large values and the fitted
    parameters must barely move — that is the entire robustness claim."""
    rng = np.random.default_rng(1)
    d2 = 0.7 * rng.chisquare(40.0, 2000)
    c0, nu0 = fit_scaled_chi2(d2)
    poisoned = d2.copy()
    poisoned[np.argsort(d2)[-200:]] *= 50.0
    c1, nu1 = fit_scaled_chi2(poisoned)
    assert c1 == pytest.approx(c0, rel=0.02)
    assert nu1 == pytest.approx(nu0, rel=0.05)


def test_fit_scaled_chi2_returns_none_on_degenerate_input():
    """Constant or zero distances have no estimable shape. The caller must
    get None and fall back, not a ZeroDivisionError on a device."""
    assert fit_scaled_chi2(np.zeros(50)) is None
    assert fit_scaled_chi2(np.full(50, 3.0)) is None
    assert fit_scaled_chi2(np.array([1.0, 2.0])) is None


def test_analytic_threshold_falls_back_to_empirical_when_unfittable():
    d = np.full(30, 4.0)
    assert analytic_threshold(d) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 4. the deployment rule
# ---------------------------------------------------------------------------

def test_choose_threshold_deploys_the_lower_of_the_two():
    """Never less sensitive than the estimator it replaces. `min` means this
    change cannot, on any input, cause a fault to be missed that the old
    code would have caught."""
    d = oof_distances(make_learn(24)[0])
    thr, info = choose_threshold(d)
    assert thr == pytest.approx(min(info["empirical"], info["analytic"]))
    assert thr <= info["empirical"] + 1e-12


def test_choose_threshold_flags_a_contaminated_learn_period():
    X, _ = make_learn(24)
    _, clean = choose_threshold(oof_distances(X))
    _, dirty = choose_threshold(oof_distances(contaminate(X)))
    assert not clean["contaminated"], \
        f"clean learn period false-flagged (ratio {clean['ratio']:.3f})"
    assert dirty["contaminated"], \
        f"contaminated learn period not flagged (ratio {dirty['ratio']:.3f})"
    assert dirty["ratio"] > CONTAMINATION_RATIO > clean["ratio"]


def test_quantile_constant_is_the_documented_one():
    assert QUANTILE == 0.995


# ---------------------------------------------------------------------------
# 5. end to end through fit_baseline — the property that actually matters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [24, 48])
def test_deployed_threshold_no_longer_tracks_the_outlier_magnitude(n):
    """THE regression test for T1.6, stated as the property that actually
    matters rather than as a single number.

    The defect was not "the threshold moves"; it was "the threshold is an
    unbounded increasing function of something nobody observes". Measured on
    the pre-fix code at n=24 the empirical threshold went 1.86x / 2.86x /
    4.37x / 7.00x as the outlier grew 12 -> 100 sigma. After the fix the
    deployed threshold is 1.39x / 1.24x / 1.01x / 0.80x — bounded, and no
    longer increasing.

    So the test asserts two things: bounded, and not monotone-increasing in
    the outlier size.
    """
    X, OP = make_learn(n)
    clean = fit_baseline(X, OP)["thresholds"][0]
    ratios = [fit_baseline(contaminate(X, s), OP)["thresholds"][0] / clean
              for s in (12.0, 25.0, 50.0, 100.0)]
    assert max(ratios) < 1.5, (
        f"n={n}: deployed threshold moved by up to {max(ratios):.2f}x")
    assert ratios[-1] <= ratios[0] * 1.05, (
        f"n={n}: threshold still grows with outlier size: {ratios}")


def test_estimator_itself_is_exactly_insensitive_to_outlier_size():
    """Isolate the estimator from the model it is fitted on.

    Freeze the Gaussian on clean data and contaminate only the distance
    sample. The fitted threshold is then IDENTICAL at 12, 25, 50 and 100
    sigma, because the fit reads only the median and 75th percentile and
    never looks at the tail. Any residual movement in the end-to-end test
    above is therefore the fitted covariance being corrupted, not the
    quantile estimator — a separate problem, recorded in DOC_STATUS.
    """
    X, _ = make_learn(24)
    d_clean = oof_distances(X)
    thresholds = set()
    for s in (12.0, 25.0, 50.0, 100.0):
        d = d_clean.copy()
        d[0] = d_clean.max() * s          # only the tail changes
        thresholds.add(round(choose_threshold(d)[0], 9))
    assert len(thresholds) == 1, (
        f"estimator moved with outlier magnitude: {sorted(thresholds)}")


def test_baseline_stores_both_estimates_and_the_disagreement():
    X, OP = make_learn(24)
    b = fit_baseline(X, OP)
    for key in ("thresholds_empirical", "thresholds_analytic",
                "threshold_ratios", "threshold_contaminated"):
        assert key in b, f"{key} missing — the diagnostic is not recorded"
        assert len(b[key]) == b["k"], f"{key} must have one entry per regime"
    assert np.all(b["thresholds"] <= b["thresholds_empirical"] + 1e-9)


def test_contamination_flag_survives_save_and_load(tmp_path: Path):
    """The flag is only useful if it reaches whoever reads the baseline
    afterwards, which means it has to survive the npz round trip."""
    X, OP = make_learn(24)
    b = fit_baseline(contaminate(X), OP)
    assert bool(b["threshold_contaminated"][0]) is True
    path = tmp_path / "b.npz"
    save_baseline(path, b)
    loaded = load_baseline(path)
    assert bool(loaded["threshold_contaminated"][0]) is True
    assert loaded["threshold_ratios"][0] == pytest.approx(b["threshold_ratios"][0])


def test_small_n_branch_is_also_robust():
    """Under 15 windows per regime `fit_baseline` uses in-sample distances
    inflated by 1.5. That branch needs the same protection — it is the one a
    rushed learn period actually hits."""
    X, OP = make_learn(12)
    clean = fit_baseline(X, OP)
    dirty = fit_baseline(contaminate(X), OP)
    assert dirty["thresholds"][0] / clean["thresholds"][0] < 1.35
    assert "thresholds_analytic" in dirty


def test_thresholds_remain_positive_and_finite():
    for n in (10, 16, 24, 60):
        b = fit_baseline(*make_learn(n))
        t = b["thresholds"]
        assert np.all(np.isfinite(t)) and np.all(t > 0), f"n={n}: {t}"


# ---------------------------------------------------------------------------
# 5b. the singleton-regime bug, found by running the contamination case
#     end to end through the real pipeline
# ---------------------------------------------------------------------------

def test_one_outlying_operating_point_does_not_become_its_own_regime():
    """A lorry outside during the learn period used to form a cluster of ONE.

    LedoitWolf was then fitted to a single sample, whose distance to its own
    mean is exactly zero, so that regime's threshold came out at 0.0 — and
    every window later assigned to it alarmed unconditionally. Measured
    before the fix: k=2, counts [47, 1], thresholds [7.4658, 0.0].

    False alarms are the project's #1 churn risk, so a regime that alarms on
    everything is the worst failure state the detector has.
    """
    X, OP = make_learn(48)
    OP[5] = [50.0, 3.0, 3.0]                 # one window at a wild level
    b = fit_baseline(X, OP)
    assert b["counts"].min() >= MIN_REGIME_WINDOWS, (
        f"regime sizes {b['counts'].tolist()} — an outlier became a regime")
    assert np.all(b["thresholds"] > 0), \
        f"a zero threshold alarms on everything: {b['thresholds'].tolist()}"


def test_min_regime_size_does_not_block_genuine_regimes():
    """The floor must not cost us the feature it protects. Two real,
    well-populated operating modes must still be recovered as two."""
    rng = np.random.default_rng(2)
    n = 24
    OP = np.vstack([
        np.column_stack([np.full(n, 50.0) + rng.normal(0, 0.05, n),
                         rng.normal(-0.1, 0.01, n), rng.normal(-0.1, 0.01, n)]),
        np.column_stack([np.full(n, 30.0) + rng.normal(0, 0.05, n),
                         rng.normal(-0.4, 0.01, n), rng.normal(-0.4, 0.01, n)])])
    m = OP.mean(axis=0)
    OPz = (OP - m) / np.array([0.05 * max(m[0], 1.0), 0.1, 0.1])
    k, labels = choose_k(OPz)
    assert k == 2, f"genuine two-regime learn period collapsed to k={k}"
    assert np.bincount(labels).min() >= MIN_REGIME_WINDOWS


# ---------------------------------------------------------------------------
# 6. no regression in the parts that were already right
# ---------------------------------------------------------------------------

def test_scorer_still_loads_and_scores(tmp_path: Path):
    """The new keys are additive; inference.py must be unaffected."""
    X, OP = make_learn(40)
    b = fit_baseline(X, OP, list(FEATURE_NAMES))
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    s = MahalanobisScorer(path)
    out = s.score(X[0], OP[0])
    assert np.isfinite(out["score"]) and out["threshold"] > 0
    assert out["anomalous"] is False or out["anomalous"] is True


def test_old_baseline_without_diagnostics_still_loads(tmp_path: Path):
    """Devices in the field hold baselines written before this change. They
    have no threshold_* diagnostic arrays and must keep working."""
    X, OP = make_learn(40)
    b = fit_baseline(X, OP, list(FEATURE_NAMES))
    for key in ("thresholds_empirical", "thresholds_analytic",
                "threshold_ratios", "threshold_contaminated"):
        b.pop(key)
    path = tmp_path / "old.npz"
    save_baseline(path, b)
    s = MahalanobisScorer(path)
    assert np.isfinite(s.score(X[1], OP[1])["score"])


def test_healthy_learn_windows_are_mostly_below_their_own_threshold():
    """A threshold that flags its own learn data is not a threshold."""
    X, OP = make_learn(48)
    b = fit_baseline(X, OP)
    Z = (X - b["global_mean"]) / b["global_std"]
    delta = Z - b["means"][0]
    d = np.sqrt(np.maximum(
        np.einsum("ij,jk,ik->i", delta, b["precisions"][0], delta), 0.0))
    assert (d > b["thresholds"][0]).mean() <= 0.02
