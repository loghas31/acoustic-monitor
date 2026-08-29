"""
Unit tests for regime clustering, Mahalanobis baseline, CV thresholds, and
alert gating. Synthetic Gaussian data — fast; the full-signal end-to-end
evidence lives in ml/evaluate.py.
"""

from pathlib import Path

import numpy as np
import pytest

from baseline import choose_k, fit_baseline, load_baseline, save_baseline
from features import FEATURE_NAMES
from inference import MahalanobisScorer
from inference import AlertGate

RNG = np.random.default_rng(0)
# Feature width comes from the contract, not a literal: it was 40 until T1.5
# removed three compositional redundancies, and `MahalanobisScorer` now refuses
# a baseline whose stored width disagrees with the firmware's.
D = len(FEATURE_NAMES)


def make_data(n_per_regime=30, two_regimes=True):
    """Feature matrices + operating points mimicking the real layout."""
    X1 = RNG.normal(0, 1, (n_per_regime, D))
    OP1 = np.column_stack([np.full(n_per_regime, 50.0) + RNG.normal(0, 0.05, n_per_regime),
                           RNG.normal(-0.1, 0.01, n_per_regime),
                           RNG.normal(-0.1, 0.01, n_per_regime)])
    if not two_regimes:
        return X1, OP1
    X2 = RNG.normal(0.8, 1.2, (n_per_regime, D))   # different feature island
    OP2 = np.column_stack([np.full(n_per_regime, 30.0) + RNG.normal(0, 0.05, n_per_regime),
                           RNG.normal(-0.3, 0.01, n_per_regime),
                           RNG.normal(-0.3, 0.01, n_per_regime)])
    return np.vstack([X1, X2]), np.vstack([OP1, OP2])


def _opz(OP):
    m = OP.mean(0)
    scale = np.array([0.05 * max(m[0], 1.0), 0.1, 0.1])
    return (OP - m) / scale


def test_choose_k_two_regimes():
    _, OP = make_data(two_regimes=True)
    k, labels = choose_k(_opz(OP))
    assert k == 2
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1


def test_choose_k_single_regime():
    _, OP = make_data(two_regimes=False)
    k, _ = choose_k(_opz(OP))
    assert k == 1, "must not invent regimes from unimodal noise"


def test_fit_score_roundtrip(tmp_path: Path):
    X, OP = make_data(two_regimes=True)
    b = fit_baseline(X, OP)
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    scorer = MahalanobisScorer(path)
    assert scorer.k == 2

    # healthy-like vector near regime 1's distribution, regime-1 OP
    x = RNG.normal(0, 1, D)
    s = scorer.score(x, np.array([50.0, -0.1, -0.1]))
    assert s["regime"] in (0, 1)
    assert not s["anomalous"], f"typical vector flagged: {s}"

    # wildly off vector must be anomalous in either regime
    s_bad = scorer.score(np.full(D, 6.0), np.array([50.0, -0.1, -0.1]))
    assert s_bad["anomalous"] and s_bad["score"] > 2 * s_bad["threshold"]


def test_regime_assignment_uses_operating_point(tmp_path: Path):
    X, OP = make_data(two_regimes=True)
    b = fit_baseline(X, OP)
    save_baseline(tmp_path / "b.npz", b)
    scorer = MahalanobisScorer(tmp_path / "b.npz")
    x = RNG.normal(0, 1, D)
    r_fast = scorer.score(x, np.array([50.0, -0.1, -0.1]))["regime"]
    r_slow = scorer.score(x, np.array([30.0, -0.3, -0.3]))["regime"]
    assert r_fast != r_slow, "operating point must drive regime selection"


def test_cv_threshold_wider_than_in_sample():
    """The threshold must come from out-of-fold distances; in-sample ones are
    biased low when n ~ d (we measured 79 % held-out FPR before this fix).
    The bias only appears when features are CORRELATED (the realistic case —
    RMS and band energies move together); with i.i.d. data Ledoit-Wolf shrinks
    to ~identity and the in-sample optimism mostly vanishes."""
    from sklearn.covariance import LedoitWolf
    n = 30
    L = RNG.normal(0, 1, (D, 8))                   # rank-8 latent structure
    X = RNG.normal(0, 1, (n, 8)) @ L.T + 0.3 * RNG.normal(0, 1, (n, D))
    OP = np.column_stack([50.0 + RNG.normal(0, 0.05, n),
                          -0.1 + RNG.normal(0, 0.01, n),
                          -0.1 + RNG.normal(0, 0.01, n)])
    b = fit_baseline(X, OP)
    Z = (X - b["global_mean"]) / b["global_std"]
    lw = LedoitWolf().fit(Z)
    mu = Z.mean(0)
    d_in = np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, lw.precision_, Z - mu))
    assert b["thresholds"][0] > np.percentile(d_in, 99.5) * 1.2


# ----------------------------------------------------------------------------
# Alert gating
# ----------------------------------------------------------------------------

def test_gate_ignores_transient():
    g = AlertGate(need=4)
    fired = [g.feed(a) for a in [False, True, False, True, True, False, False]]
    assert sum(fired) == 0


def test_gate_fires_once_per_episode():
    g = AlertGate(need=4, clear=4)
    seq = [False] * 3 + [True] * 10          # one long episode
    assert sum(g.feed(a) for a in seq) == 1


def test_gate_rearms_after_recovery():
    g = AlertGate(need=3, clear=2)
    episode = [True] * 3                     # fires
    recovery = [False] * 2                   # re-arms
    again = [True] * 3                       # fires again
    fired = [g.feed(a) for a in episode + recovery + again]
    assert sum(fired) == 2


def test_gate_does_not_rearm_during_flapping():
    g = AlertGate(need=3, clear=4)
    fired = [g.feed(a) for a in [True, True, True,      # alert
                                 False, True, False,    # flapping, < clear normals
                                 True, True, True]]
    assert sum(fired) == 1, "score wobbling around threshold must not re-alert"
