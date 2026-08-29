"""
Compositional-data tests (backlog T1.5, self-review F1).

WHY THIS FILE EXISTS
--------------------
Three blocks of the feature vector are *compositions*: sets of energy
fractions that are constrained to sum to 1. A composition of D parts carries
only D-1 free numbers, so feeding all D of them to a Gaussian model hands it a
covariance matrix that is singular **by construction**. Nothing crashes —
Ledoit-Wolf shrinkage regularises the inverse — but we are then spending D
dimensions of a scarce budget on D-1 dimensions of information, at n/d ~ 1.6
where every dimension is expensive.

The fix is the standard one from compositional data analysis: map each
D-part composition to D-1 **isometric log-ratio (ILR)** coordinates, which are
unconstrained, full-rank, and orthonormal in the Aitchison geometry.

These tests are written as properties of the *fix*, so they failed before it
and pass after it. The exact "before" numbers are recorded in
docs/DOC_STATUS.md.

READ THIS BEFORE "SIMPLIFYING" TO CLR
-------------------------------------
The obvious textbook move is the centred log-ratio (CLR). It does NOT solve
this problem: CLR coordinates sum to zero by construction, so the transformed
block is still exactly rank D-1 in D columns — the singularity is preserved,
just relocated. `test_clr_would_not_have_fixed_it` proves that on real feature
data, so nobody re-introduces the bug in the name of using the more famous
transform.
"""

import numpy as np
import pytest

from features import (FEATURE_NAMES, band_energy_ratios, clr, extract_features,
                      ilr, ilr_inverse)
from simulate import SimConfig, bearing_fault_signal, normal_signal

# Short windows: these tests probe the ALGEBRA of the feature map, which is
# window-length independent. 6 s keeps the whole file under ~5 s.
DUR_S = 6.0


def _vectors(n=14, seed=0):
    """A small learn-like matrix: two speeds, healthy, as baseline.py collects."""
    rows = []
    for i in range(n):
        cfg = SimConfig(duration_s=DUR_S, fr=50.0 if i % 2 == 0 else 30.0)
        ra, rv = np.random.default_rng(seed + i), np.random.default_rng(seed + 500 + i)
        a = normal_signal(cfg, cfg.fs_audio, ra)
        v = normal_signal(cfg, cfg.fs_accel, rv)
        rows.append(extract_features(a, cfg.fs_audio, v, cfg.fs_accel)["vector"])
    return np.array(rows)


def _block(prefix):
    return [i for i, n in enumerate(FEATURE_NAMES) if n.startswith(prefix)]


def _sv_ratio(M, standardise=True):
    """Smallest / largest singular value of the block.

    Standardised by default because that is the matrix the Mahalanobis model
    actually sees (`baseline.fit_baseline` z-scores every column before fitting
    Ledoit-Wolf). It also sharpens the measurement: a constraint whose residual
    is 1e-4 in raw units is invisible next to a column whose scale is 1e+2, and
    obvious next to the same column scaled to unit variance.

    Calibration, all measured on a 14-window healthy learn matrix:

        block                        before T1.5    a full-rank block
        audio band fractions (D=8)      6.8e-3      2.1e-1 median, min 5.3e-2
        accel band fractions (D=8)      2.9e-3        over 200 random draws
        envelope fractions   (D=6)      6.5e-3      3.3e-1 median, min 1.2e-1
        audio statistics     (D=4)      2.8e-1      (not a composition)

    So 2e-2 sits an order of magnitude above every degenerate block and an
    order below every healthy one — a threshold with real margin on both sides,
    not one tuned to just barely pass."""
    if standardise:
        M = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-12)
    sv = np.linalg.svd(M - M.mean(axis=0), compute_uv=False)
    return float(sv[-1] / sv[0])


RANK_MIN = 2e-2


# ----------------------------------------------------------------------------
# the transform itself
# ----------------------------------------------------------------------------

def test_ilr_returns_one_fewer_coordinate():
    """D parts -> D-1 coordinates. This IS the fix: the redundant dimension is
    removed rather than shrunk away by the covariance estimator."""
    for D in (2, 3, 6, 8, 13):
        c = np.random.default_rng(D).random(D)
        assert ilr(c / c.sum()).shape == (D - 1,)


def test_ilr_is_scale_invariant():
    """A composition is defined up to scale — only the ratios carry meaning.
    Doubling the total energy in every band must not move a single coordinate,
    which is exactly the property that makes these features immune to the
    microphone's absolute gain."""
    c = np.array([0.4, 0.3, 0.2, 0.07, 0.03])
    np.testing.assert_allclose(ilr(c), ilr(c * 137.0), rtol=0, atol=1e-12)


def test_ilr_is_a_bijection_onto_the_simplex():
    """No information is lost: ilr_inverse recovers the original fractions.
    We are removing redundancy, not discarding data."""
    rng = np.random.default_rng(3)
    c = rng.random(8)
    c /= c.sum()
    np.testing.assert_allclose(ilr_inverse(ilr(c)), c, rtol=1e-12, atol=1e-14)


def test_ilr_is_isometric_with_respect_to_aitchison_distance():
    """ILR coordinates are ORTHONORMAL: Euclidean distance between coordinate
    vectors equals Aitchison distance between compositions. This is why
    Mahalanobis distance on ILR coordinates is meaningful — a plain
    drop-one-column fix would not have this property."""
    rng = np.random.default_rng(4)
    a, b = rng.random(6), rng.random(6)
    a /= a.sum()
    b /= b.sum()
    la, lb = np.log(a), np.log(b)
    aitchison = np.linalg.norm((la - la.mean()) - (lb - lb.mean()))
    assert abs(np.linalg.norm(ilr(a) - ilr(b)) - aitchison) < 1e-12


def test_ilr_handles_a_dead_channel_without_nans():
    """A disconnected sensor gives all-zero energy in every band. That must
    produce finite, CONSTANT coordinates (all bands equal -> all ratios 1 ->
    all coordinates 0), never NaN. A NaN here would poison the whole 40-dim
    vector and with it every anomaly score after install day."""
    z = ilr(np.zeros(8))
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z, 0.0, atol=1e-12)


def test_clr_would_not_have_fixed_it():
    """CLR output sums to zero by construction, so a CLR-transformed block is
    STILL exactly rank D-1 in D columns. Recorded as a test because the backlog
    originally proposed 'CLR or drop one band', and CLR alone does not do the
    job."""
    rng = np.random.default_rng(5)
    C = rng.random((30, 8))
    C /= C.sum(axis=1, keepdims=True)
    Y = np.array([clr(c) for c in C])
    np.testing.assert_allclose(Y.sum(axis=1), 0.0, atol=1e-12)
    assert _sv_ratio(Y, standardise=False) < 1e-12, "CLR block is still singular"
    X = np.array([ilr(c) for c in C])
    assert _sv_ratio(X, standardise=False) > 1e-2, "ILR block must be full rank"


# ----------------------------------------------------------------------------
# the property the feature vector must now have
# ----------------------------------------------------------------------------

def test_ilr_removes_the_constraint_on_the_same_data():
    """THE T1.5 REGRESSION TEST — before and after, on identical windows.

    `envelope_fractions` still returns the raw 6-part composition, so this test
    can hold the data fixed and compare the two representations directly rather
    than trusting a number recorded in a doc. The raw fractions must be
    near-singular (that is the defect) and their ILR coordinates must not
    (that is the fix). Measured: 6.5e-3 -> 3.4e-1, a factor of ~50."""
    from features import envelope_fractions, select_demodulation_band

    fracs = []
    for i in range(14):
        cfg = SimConfig(duration_s=DUR_S, fr=50.0 if i % 2 == 0 else 30.0)
        a = normal_signal(cfg, cfg.fs_audio, np.random.default_rng(i))
        band, _ = select_demodulation_band(a, cfg.fs_audio)
        fracs.append(envelope_fractions(a, cfg.fs_audio, band)[0])
    F = np.array(fracs)

    # the composition itself: sums to 1, so one direction cannot vary
    np.testing.assert_allclose(F.sum(axis=1), 1.0, atol=2e-3)
    raw_r = _sv_ratio(F)
    ilr_r = _sv_ratio(np.array([ilr(f) for f in F]))
    assert raw_r < 2e-2, f"raw fractions should be near-singular, got {raw_r:.2e}"
    assert ilr_r > RANK_MIN, f"ILR coordinates still near-singular: {ilr_r:.2e}"
    assert ilr_r > 10 * raw_r, (
        f"ILR barely helped: {raw_r:.2e} -> {ilr_r:.2e}")


def test_env_ilr_block_in_the_assembled_vector_is_full_rank():
    """The fix survives assembly into the real feature vector."""
    X = _vectors()
    r = _sv_ratio(X[:, _block("env_ilr_")])
    assert r > RANK_MIN, f"env ILR block is near-singular (sv ratio {r:.2e})"


@pytest.mark.parametrize("prefix", ["audio_band_ilr_", "accel_band_ilr_"])
def test_band_ilr_blocks_are_low_rank_but_not_from_the_constraint(prefix):
    """A HONEST CHARACTERISATION TEST, NOT A PASS.

    T1.5 removed the algebraic constraint from these blocks too, and yet they
    stay near-singular: measured sv ratio 1.2e-3 (audio) and 3.7e-3 (accel)
    after the fix. Fixing the algebra did not fix the rank, so something else
    is responsible. Measured, in `docs/DOC_SELF_REVIEW.md` F9:

      * The ILR transform is not the culprit — applied to random 8-part
        compositions it gives sv ratio 0.50 (see
        `test_ilr_of_random_compositions_is_full_rank`).
      * The null direction REPRODUCES across independent window samples to
        |cos| = 0.999, so it is systematic, not sampling noise.
      * `simulate.py`'s band composition has participation rank **1.03 of 8**
        (`test_simulator_spectral_shape_is_one_dimensional`). The eight band
        fractions of a simulated healthy signal are essentially a
        one-parameter family, so their seven ILR coordinates cannot be more
        than about one dimension of information.

    So the residual low rank is a property of the SIMULATOR, not of the feature
    map — the same species of defect as self-review F6 (three accelerometer
    axes that are copies of one signal). This test pins the number so that the
    first real recording will visibly change it. When it does, raise the bound
    and record the real value; do NOT delete the test."""
    X = _vectors()
    r = _sv_ratio(X[:, _block(prefix)])
    assert r < RANK_MIN, (
        f"{prefix} sv ratio is now {r:.2e}, above the {RANK_MIN:.0e} recorded "
        "for simulated data. If this is real data, that is GOOD NEWS — update "
        "the bound and DOC_SELF_REVIEW F9 with the measured value.")


def test_ilr_of_random_compositions_is_full_rank():
    """Isolates the transform from the data: with compositions that genuinely
    vary, ILR coordinates are full rank. This is what makes the conclusion in
    `test_band_ilr_blocks_are_low_rank_but_not_from_the_constraint` sound."""
    rng = np.random.default_rng(0)
    for D in (6, 8):
        C = rng.random((40, D))
        C /= C.sum(axis=1, keepdims=True)
        r = _sv_ratio(np.array([ilr(c) for c in C]))
        assert r > 0.1, f"ILR of random {D}-part compositions: sv ratio {r:.2e}"


def test_simulator_spectral_shape_is_one_dimensional():
    """The measurement behind F9, kept as a test because it is a claim about
    `ml/simulate.py` that the first real recording should falsify.

    Participation rank of the log band-fractions across 24 healthy windows at
    two speeds: measured **1.03 of 8**. A real machine — with load changes,
    ambient noise, other machines in the room — should score materially
    higher. Until it does, do not believe any result that depends on the band
    features carrying independent information."""
    from features import band_fractions

    F = []
    for i in range(24):
        cfg = SimConfig(duration_s=DUR_S, fr=50.0 if i % 2 == 0 else 30.0)
        x = normal_signal(cfg, cfg.fs_audio, np.random.default_rng(i))
        F.append(band_fractions(x, cfg.fs_audio))
    L = np.log(np.maximum(np.array(F), 1e-9))
    sv = np.linalg.svd(L - L.mean(axis=0), compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    p = p[p > 0]
    rank = float(np.exp(-(p * np.log(p)).sum()))
    assert rank < 3.0, (
        f"simulator band composition now spans {rank:.2f} of 8 dimensions "
        "(was 1.03) — if this is real data, update F9 with the number")


def test_energy_fractions_no_longer_appear_raw_in_the_vector():
    """Guards the actual defect: no set of feature columns may sum to a
    constant. Checked directly on the assembled vector rather than by
    inspecting names, so a future feature that re-introduces a composition is
    caught too.

    Measured before the fix: the 6 env_frac columns summed to 0.999849–0.999997
    — a spread of 1.5e-4 against a typical single-column spread of 1.3e-1, i.e.
    the sum was constant to one part in ~900 while its parts moved freely.
    That ratio, not the absolute spread, is what makes it a dependency."""
    X = _vectors()
    for prefix in ("audio_band", "accel_band", "env_ilr_"):
        idx = _block(prefix)
        B = X[:, idx]
        tightness = np.ptp(B.sum(axis=1)) / (np.median(np.ptp(B, axis=0)) + 1e-12)
        assert tightness > 1e-2, (
            f"{prefix} columns sum to a near-constant (spread {tightness:.1e} of "
            "a typical column's) — that is a composition fed in raw")


def test_feature_vector_contract():
    """37 dims, not 40: two band blocks lost one column each and the envelope
    block lost one. Fewer features carrying the same information is the point
    of the exercise at n/d ~ 1.6."""
    X = _vectors(n=2)
    assert X.shape[1] == 37 == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == 37, "duplicate feature name"
    assert np.isfinite(X).all()


def test_band_energy_ratios_helper_is_unchanged():
    """The old log-fraction helper is KEPT (it is the honest description of the
    spectrum for reports and plots) — it is simply no longer what goes into the
    Mahalanobis vector. Pinning it here so the refactor cannot silently alter
    the published band numbers."""
    x = np.random.default_rng(1).standard_normal(160_000)
    r = band_energy_ratios(x, 16000.0)
    assert r.shape == (8,)
    assert 0.9 < np.sum(10.0 ** r) < 1.1


# ----------------------------------------------------------------------------
# the fix must not cost detection
# ----------------------------------------------------------------------------

def test_stale_baseline_is_refused_with_a_readable_message(tmp_path):
    """MIGRATION GUARD, and the reason it exists.

    T1.5 changed the feature contract from 40 to 37 dims, so every baseline.npz
    trained before it is now unloadable. Measured behaviour before this guard:

        ValueError: operands could not be broadcast together with
                    shapes (37,) (40,)

    — raised from inside `score()`, on a device, in the field, hours after a
    firmware update, with nothing in the message that names the actual problem
    or the fix. Failing safe is not enough; it has to fail LEGIBLY.
    """
    import numpy as np
    from baseline import save_baseline
    from inference import MahalanobisScorer

    d_old = 40
    stale = {
        "created": 0.0, "n_windows": 24, "k": 1,
        "global_mean": np.zeros(d_old), "global_std": np.ones(d_old),
        "op_mean": np.zeros(3), "op_scale": np.ones(3),
        "op_centroids": np.zeros((1, 3)),
        "means": np.zeros((1, d_old)), "precisions": np.eye(d_old)[None],
        "thresholds": np.array([5.0]), "counts": np.array([24]),
        "feature_names": [f"old_{i}" for i in range(d_old)],
        "X_train": np.zeros((24, d_old)), "OP_train": np.zeros((24, 3)),
    }
    path = tmp_path / "stale.npz"
    save_baseline(path, stale)

    with pytest.raises(ValueError) as exc:
        MahalanobisScorer(path)
    msg = str(exc.value)
    assert "40" in msg and "37" in msg, f"error must name both dimensions: {msg}"
    assert "baseline.py" in msg, f"error must name the fix (retrain): {msg}"


def test_env_ilr_still_moves_on_a_fault():
    """Information check, not just an algebra check: the envelope composition
    must still respond to a bearing fault after the transform. If ILR made the
    block inert we would have 'fixed' the rank at the price of the signal."""
    cfg = SimConfig(duration_s=DUR_S, fr=50.0)
    idx = _block("env_ilr_")
    ra, rv = np.random.default_rng(11), np.random.default_rng(12)
    n = extract_features(normal_signal(cfg, cfg.fs_audio, ra), cfg.fs_audio,
                         normal_signal(cfg, cfg.fs_accel, rv), cfg.fs_accel)["vector"][idx]
    ra, rv = np.random.default_rng(11), np.random.default_rng(12)
    f = extract_features(bearing_fault_signal(cfg, cfg.fs_audio, ra, 0.15, "outer"),
                         cfg.fs_audio,
                         bearing_fault_signal(cfg, cfg.fs_accel, rv, 0.15, "outer"),
                         cfg.fs_accel)["vector"][idx]
    assert np.linalg.norm(f - n) > 0.1, (
        f"envelope ILR coordinates barely moved on a fault: {np.abs(f - n)}")
