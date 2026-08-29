"""test_regimes_miconly.py — regime clustering when the accelerometer is absent.

Backlog T1.9, self-review F7. Mic-only is the *recommended primary build*, so
its clustering behaviour is not a corner case.

F7's suspicion was that mic-only clustering "runs on 2 of 3 dimensions" and
that `SILHOUETTE_MIN`, tuned in 3-D, had never been checked there. Executing it
gave a sharper — and worse — answer, in three parts, each pinned by a test
below:

1. F7's prescribed fix (drop the dead dimension before clustering) is a
   **no-op**. A constant column contributes exactly 0 to every pairwise
   distance after standardisation, so k-means and the silhouette are identical
   with it present or removed. `test_dropping_the_dead_dimension_is_a_no_op`.

2. The real problem is *one live dimension*, not *one dead* one. On a
   fixed-speed machine, mic-only leaves only the audio level varying, and the
   null distribution of the silhouette rises as dimensionality falls: pure
   single-cluster noise is split by the pre-fix rule 98 % of the time in 1-D
   and 0 % in 2-D or 3-D. `test_silhouette_null_rises_as_live_dimensions_fall`.

3. The cost was measured before fixing it: 48 healthy mic-only windows of one
   unchanging simulated machine were split into regimes of 30 and 18, and over
   100 bootstrap learn periods the held-out healthy false-alarm rate was
   0.1358 +/- 0.1445 against 0.0217 +/- 0.0290 with k forced to 1 — 6.3x the
   false alarms, on the failure mode the product cannot afford. The fix
   (`MIN_REGIME_SEPARATION`) reproduces the k=1 number exactly.
   `test_single_regime_mic_only_learn_period_is_one_regime` runs the shortened
   version of that end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

from baseline import (MIN_REGIME_SEPARATION, MIN_REGIME_WINDOWS,  # noqa: E402
                      SILHOUETTE_MIN, SILHOUETTE_MIN_1D, K_RANGE,
                      centroid_separation, choose_k, effective_dims,
                      fit_baseline, operating_point, silhouette_floor)
from features import extract_features  # noqa: E402

FS_A, FS_V, WINDOW_S = 16000, 6400, 30.0

# The measured within-regime spread of the operating point on the simulator,
# in OPz units (probe, 48 healthy windows, single speed): audio log-RMS sd
# 0.00098, accel log-RMS sd 0.00141, fr sd exactly 0. Tests that need "one
# unchanging machine" use this rather than an invented number.
OPZ_NOISE_SD = 1e-3


def opz(OP: np.ndarray) -> np.ndarray:
    """Reproduce fit_baseline's physical standardisation of operating points."""
    OP = np.asarray(OP, dtype=float)
    m = OP.mean(axis=0)
    return (OP - m) / np.array([0.05 * max(m[0], 1.0), 0.1, 0.1])


def legacy_choose_k(OPz: np.ndarray) -> tuple[int, np.ndarray]:
    """`choose_k` exactly as it stood before T1.9 — silhouette + size floor,
    with no physical-separation gate. Kept so the bug can be demonstrated
    rather than described."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_s, best_labels = 1, -1.0, np.zeros(len(OPz), dtype=int)
    for k in K_RANGE:
        if len(OPz) < k * 8:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(OPz)
        if len(set(km.labels_)) < 2:
            continue
        if np.bincount(km.labels_, minlength=k).min() < MIN_REGIME_WINDOWS:
            continue
        s = silhouette_score(OPz, km.labels_)
        if s > best_s:
            best_k, best_s, best_labels = k, s, km.labels_
    if best_s < SILHOUETTE_MIN:
        return 1, np.zeros(len(OPz), dtype=int)
    return best_k, best_labels


def noise_cloud(n: int, live_dims: int, seed: int, sd: float = 1.0) -> np.ndarray:
    """One cluster, no regimes, `live_dims` dimensions carrying variance."""
    rng = np.random.default_rng(seed)
    Z = np.zeros((n, 3))
    for d in range(live_dims):
        Z[:, d] = rng.normal(0.0, sd, n)
    return Z


def mic_only_accel(window_s: float = WINDOW_S, fs: int = FS_V) -> np.ndarray:
    """What `capture.HardwareSource._empty_accel` hands the feature extractor
    when the IIS3DWB is absent."""
    return np.zeros((int(window_s * fs), 3))


# ---------------------------------------------------------------------------
# 1. what mic-only actually does to the operating point
# ---------------------------------------------------------------------------

def test_mic_only_third_operating_dimension_is_the_dead_sentinel():
    """The accel log-RMS is not noisy-but-present; it is a constant. Anything
    reasoning about mic-only clustering must start from that."""
    from simulate import SimConfig, normal_signal

    ops = []
    for i in range(4):
        cfg = SimConfig(duration_s=4.0, fr=50.0, fs_audio=FS_A, fs_accel=FS_V)
        rng = np.random.default_rng(100 + i)
        audio = normal_signal(cfg, FS_A, rng)
        f = extract_features(audio, FS_A, mic_only_accel(4.0), FS_V)
        ops.append(operating_point(f["vector"], f["fr_hz"]))
    OP = np.array(ops)

    assert np.all(OP[:, 2] == -9.0), "dead-channel sentinel changed; retune this file"
    assert OP[:, 2].std() == 0.0
    # and the audio dimension is live, so mic-only is not degenerate everywhere
    assert OP[:, 1].std() > 0.0


def test_dropping_the_dead_dimension_is_a_no_op():
    """F7 proposed dropping dead dims before clustering. It changes nothing:
    a constant column is exactly 0 after standardisation and contributes 0 to
    every distance. Recorded because a refuted prescription is worth as much
    as a confirmed one."""
    from sklearn.metrics import silhouette_score

    for seed in range(5):
        Z = noise_cloud(48, live_dims=1, seed=seed)
        assert np.all(Z[:, 1:] == 0.0)
        k_full, lab_full = choose_k(Z)
        k_live, lab_live = choose_k(Z[:, :1])
        assert k_full == k_live
        assert np.array_equal(lab_full, lab_live)
        if k_full > 1:
            assert silhouette_score(Z, lab_full) == pytest.approx(
                silhouette_score(Z[:, :1], lab_live))


# ---------------------------------------------------------------------------
# 2. the actual mechanism: silhouette's null depends on dimensionality
# ---------------------------------------------------------------------------

def test_silhouette_null_rises_as_live_dimensions_fall():
    """The bug in one assertion. On data with NO regimes, the pre-T1.9 rule
    splits 1-D clouds almost always and 2-D/3-D clouds never.

    Measured at n=400 trials: P(k>1) = 0.980 / 0.000 / 0.000 and median best
    silhouette 0.586 / 0.381 / 0.283 for 1 / 2 / 3 live dims. This test uses
    40 trials to stay cheap; the margins are wide enough that it is not
    borderline."""
    rates = {}
    for live in (1, 2, 3):
        ks = [legacy_choose_k(noise_cloud(48, live, seed=1000 + t))[0]
              for t in range(40)]
        rates[live] = float(np.mean(np.array(ks) > 1))

    assert rates[1] > 0.8, (
        f"the 1-D over-splitting bug did not reproduce (P(k>1)={rates[1]}); "
        "if the estimator changed, re-measure before deleting this test")
    assert rates[2] == 0.0 and rates[3] == 0.0, (
        f"silhouette null moved in 2-D/3-D too: {rates}")


def test_fixed_choose_k_removes_the_over_splitting():
    """Same clouds, shipped `choose_k`: never a regime, in any dimensionality.

    Note which criterion does the work here. These clouds have unit variance,
    so a noise split separates its centroids by ~1.6 OPz units and PASSES the
    absolute `MIN_REGIME_SEPARATION` gate; it is the dimension-aware
    silhouette floor that rejects them. The deployed mic-only failure is the
    other way round. Both are needed."""
    for live in (1, 2, 3):
        ks = [choose_k(noise_cloud(48, live, seed=1000 + t))[0] for t in range(40)]
        assert set(ks) == {1}, f"invented regimes from noise at live_dims={live}"


def test_collinear_live_dimensions_count_as_one():
    """A full build whose audio and accelerometer levels move together is in
    the same geometry as a mic-only one, and must get the same protection.
    Measured null maximum for a collinear 2-column cloud: 0.702, i.e. above
    SILHOUETTE_MIN."""
    rng = np.random.default_rng(31)
    for t in range(30):
        Z = np.zeros((48, 3))
        Z[:, 0] = rng.standard_normal(48)
        Z[:, 1] = 0.98 * Z[:, 0] + 0.02 * rng.standard_normal(48)
        assert effective_dims(Z) == 1
        assert silhouette_floor(Z) == SILHOUETTE_MIN_1D
        assert choose_k(Z)[0] == 1


def test_effective_dims_counts_directions_not_columns():
    rng = np.random.default_rng(3)
    n = 40
    indep = np.column_stack([rng.standard_normal(n), rng.standard_normal(n),
                             rng.standard_normal(n)])
    assert effective_dims(indep) == 3
    dead = np.column_stack([rng.standard_normal(n), np.zeros(n), np.zeros(n)])
    assert effective_dims(dead) == 1
    assert effective_dims(np.zeros((n, 3))) == 0        # nothing varies at all
    assert silhouette_floor(indep) == SILHOUETTE_MIN


def test_silhouette_floor_only_relaxes_where_it_was_measured_safe():
    """The 3-D behaviour the rest of the suite was tuned against is untouched."""
    rng = np.random.default_rng(9)
    Z = rng.standard_normal((48, 3))
    assert silhouette_floor(Z) == SILHOUETTE_MIN == 0.5
    assert SILHOUETTE_MIN_1D > SILHOUETTE_MIN


def test_separation_gate_holds_at_realistic_sensor_noise():
    """The deployed case is far easier than the unit-variance null above: the
    measured within-regime spread is ~1e-3 OPz units, i.e. 0.0001 decades of
    level. Nothing may split that."""
    for t in range(20):
        Z = noise_cloud(48, live_dims=1, seed=2000 + t, sd=OPZ_NOISE_SD)
        assert choose_k(Z)[0] == 1


# ---------------------------------------------------------------------------
# 3. centroid_separation is in physical units, and says so
# ---------------------------------------------------------------------------

def test_centroid_separation_is_one_unit_per_five_percent_of_speed():
    """`op_scale`'s first entry is 5 % of mean speed, so a 50 Hz / 52.5 Hz
    machine has a centroid gap of exactly 1.0."""
    n = 24
    OP = np.vstack([np.column_stack([np.full(n, 50.0), np.full(n, -0.1), np.full(n, -0.1)]),
                    np.column_stack([np.full(n, 52.5), np.full(n, -0.1), np.full(n, -0.1)])])
    Z = opz(OP)
    labels = np.r_[np.zeros(n, int), np.ones(n, int)]
    # op_scale uses the MEAN speed (51.25), so the gap is 2.5 / (0.05*51.25)
    assert centroid_separation(Z, labels, 2) == pytest.approx(2.5 / (0.05 * 51.25))


def test_centroid_separation_is_one_unit_per_tenth_decade_of_level():
    n = 24
    OP = np.vstack([np.column_stack([np.full(n, 50.0), np.full(n, -0.4), np.full(n, -9.0)]),
                    np.column_stack([np.full(n, 50.0), np.full(n, -0.3), np.full(n, -9.0)])])
    labels = np.r_[np.zeros(n, int), np.ones(n, int)]
    assert centroid_separation(opz(OP), labels, 2) == pytest.approx(1.0)


def test_centroid_separation_of_one_regime_is_infinite():
    """k=1 has no pair to separate; the gate must never reject it."""
    Z = noise_cloud(24, live_dims=2, seed=3)
    assert centroid_separation(Z, np.zeros(24, int), 1) == float("inf")


def test_centroid_separation_reports_the_closest_pair():
    """With k=3 the gate must look at the tightest pair, not the average."""
    Z = np.array([[0.0, 0, 0]] * 8 + [[1.2, 0, 0]] * 8 + [[9.0, 0, 0]] * 8)
    labels = np.array([0] * 8 + [1] * 8 + [2] * 8)
    assert centroid_separation(Z, labels, 3) == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# 4. the gate must not cost us the feature it protects
# ---------------------------------------------------------------------------

def test_genuine_speed_regimes_survive_mic_only():
    """The commonest real regime pair — the same machine at two speeds — is
    recovered with no accelerometer at all. Measured centroid gap on the
    repo's own 50/30 Hz learn schedule: 10.0, i.e. 10x the gate."""
    n = 24
    OP = np.vstack([np.column_stack([np.full(n, 50.0), np.full(n, -0.14), np.full(n, -9.0)]),
                    np.column_stack([np.full(n, 30.0), np.full(n, -0.14), np.full(n, -9.0)])])
    Z = opz(OP)
    k, labels = choose_k(Z)
    assert k == 2
    assert np.bincount(labels).min() >= MIN_REGIME_WINDOWS
    assert centroid_separation(Z, labels, k) == pytest.approx(10.0)


@pytest.mark.parametrize("delta_decades,expect_k", [
    (0.02, 1),    # measured: merged
    (0.05, 1),    # measured: merged
    (0.20, 2),    # measured: recovered, gap 2.0
    (0.40, 2),    # measured: recovered, gap 4.0
])
def test_level_only_regimes_recovered_above_a_tenth_of_a_decade(delta_decades, expect_k):
    """The honest cost of the gate, stated as a number. Two operating points
    that differ only in level, at constant speed, are recovered from ~0.1
    decade apart (the gate's definition of a regime) and merged below it.
    Merging is the safe direction: one slightly-wide Gaussian, rather than two
    Gaussians fitted to half the data each."""
    rng = np.random.default_rng(4)
    n = 24
    lo, hi = -0.4, -0.4 + delta_decades
    OP = np.vstack([
        np.column_stack([np.full(n, 50.0), rng.normal(lo, OPZ_NOISE_SD * 0.1, n),
                         np.full(n, -9.0)]),
        np.column_stack([np.full(n, 50.0), rng.normal(hi, OPZ_NOISE_SD * 0.1, n),
                         np.full(n, -9.0)])])
    assert choose_k(opz(OP))[0] == expect_k


def test_gate_does_not_weaken_the_small_cluster_floor():
    """T1.6's MIN_REGIME_WINDOWS must still reject a well-separated outlier
    cluster: separation alone is not sufficient."""
    n = 40
    OP = np.vstack([np.column_stack([np.full(n, 50.0), np.full(n, -0.14), np.full(n, -9.0)]),
                    np.column_stack([np.full(3, 30.0), np.full(3, -0.14), np.full(3, -9.0)])])
    k, _ = choose_k(opz(OP))
    assert k == 1, "3 windows is an outlier, not an operating mode"


# ---------------------------------------------------------------------------
# 5. end to end on the real pipeline
# ---------------------------------------------------------------------------

def test_single_regime_mic_only_learn_period_is_one_regime():
    """The failure as it would have reached a customer: a mic-only node on an
    unchanging machine. Pre-T1.9 this returned k=2 (measured counts [30, 18]
    on 48 windows) and 6.3x the held-out false alarms."""
    from simulate import SimConfig, normal_signal

    X, OP = [], []
    for i in range(16):
        cfg = SimConfig(duration_s=8.0, fr=50.0, fs_audio=FS_A, fs_accel=FS_V)
        rng = np.random.default_rng(9000 + i)
        audio = normal_signal(cfg, FS_A, rng)
        f = extract_features(audio, FS_A, mic_only_accel(8.0), FS_V)
        X.append(f["vector"])
        OP.append(operating_point(f["vector"], f["fr_hz"]))
    X, OP = np.array(X), np.array(OP)

    assert OP[:, 0].std() == 0.0 and OP[:, 2].std() == 0.0, "expected 1 live dim"
    b = fit_baseline(X, OP)
    assert int(b["k"]) == 1, (
        f"invented {b['k']} regimes from one unchanging mic-only machine "
        f"(counts {b['counts'].tolist()})")
    assert b["counts"].tolist() == [16]
    # and the spurious-contamination side effect is gone with it
    assert not bool(b["threshold_contaminated"].any())
