"""T1.8 — the simulated accelerometer has three genuinely different axes.

WHY THIS FILE EXISTS
--------------------
Self-review finding F6. Until T1.8, `capture.SimulatedSource` built the
accelerometer as

    accel = [ax, 0.6*ax + n1, 0.35*ax + n2]

i.e. axes y and z were *scaled copies* of axis x with a little added noise.
Measured before the change (`tools/_scratch_f6_before.py`):

    r(x,y) = +0.9988   r(x,z) = +0.9964   r(y,z) = +0.9952
    the 12 per-axis accel statistics spanned effective rank 3.75 of 12

Three of those numbers being ~1.0 means the accelerometer path was never
really tested: `accel_y_kurt` cannot disagree with `accel_x_kurt` about
anything if y is 0.6*x. Any claim of the form "the 37-feature vector
separates fault from healthy" was really a claim about ~29 features.

`ml/simulate.py` is FROZEN, but it turns out the copying was never in
`simulate.py` at all — it is in `firmware/capture.py`, which is not frozen.
The frozen file's public building blocks are reused unchanged.

THE PHYSICS BEING MODELLED (see `capture.ACCEL_AXES` for the constants)
-----------------------------------------------------------------------
One shaft and one defect, so the *impact instants* are common to all three
axes — they arrive within ~0.1 ms of one another, far inside one sample at
6.4 kHz. What differs is the structural path from the impact to each sensing
element:

  x  radial, aligned with the load zone: the reference axis, unchanged.
  y  radial, 90 degrees round the housing: sees the rotating imbalance vector
     in QUADRATURE, and rings at a different housing mode (lower f0, lower Q).
  z  axial, through the bolted end shield: weakly coupled to a radial impact,
     softer and more damped again.

plus independent sensor self-noise per axis and a few per cent of transverse
(cross-axis) sensitivity, which is what a real MEMS triaxial part has.

WHAT THESE TESTS PIN
--------------------
1. the failing test that justified touching the code at all (correlation);
2. that axis x is BIT-IDENTICAL to the old single-axis signal, so every
   audio feature, the accel band-ILR block and `estimate_fr` — all of which
   read only channel 0 — are provably unaffected, and any change in
   detection is attributable solely to the eight y/z statistics;
3. that the three axes are different in the specific ways claimed, rather
   than merely decorrelated by noise.
"""

from pathlib import Path

import numpy as np
import pytest

# NOTE: no sys.path manipulation here. `conftest.py` inserts firmware/, ml/ and
# backend/ in an order that leaves BACKEND first, because `firmware/main.py`
# and `backend/main.py` are both importable as `main`. An earlier draft of this
# file inserted firmware/ at position 0 and broke collection of test_api.py
# with "cannot import name 'app' from 'main' (firmware/main.py)" — a genuinely
# confusing failure in a module it never touched.
ROOT = Path(__file__).resolve().parent.parent

from capture import ACCEL_AXES, SimulatedSource  # noqa: E402
from features import FEATURE_NAMES, extract_features  # noqa: E402

FS_A, FS_V = 16000, 6400
KINDS = ["normal", "bearing_outer", "bearing_inner", "imbalance"]


def sched(kind, sev=0.0, fr=50.0):
    return lambda i: {"kind": kind, "severity": sev, "fr": fr}


def one_window(kind, sev=0.0, fr=50.0, seed=99, window_s=4.0):
    src = SimulatedSource(window_s, FS_A, FS_V, sched(kind, sev, fr), seed=seed)
    return next(iter(src.windows()))


# ----------------------------------------------------------------------------
# 1. The failing test that justified the change
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind,sev", [
    ("normal", 0.0), ("bearing_outer", 0.15), ("bearing_outer", 0.5),
    ("bearing_inner", 0.3), ("imbalance", 0.5),
])
def test_inter_axis_correlation_is_below_0_9(kind, sev):
    """THE test. Before T1.8 every pair was 0.995-0.999 and this failed.

    0.9 is a deliberately generous bar: real triaxial measurements on a motor
    housing are correlated, sometimes strongly. What is not defensible is
    0.999, which is not a measurement of anything, it is `y = 0.6*x`.
    """
    _, accel = one_window(kind, sev)
    C = np.corrcoef(accel.T)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert abs(C[i, j]) < 0.9, (
            f"{kind} sev={sev}: axes {i},{j} correlate at {C[i, j]:+.4f} — "
            "the accelerometer is still copies of one signal")


def test_axes_are_not_merely_scaled_versions_of_each_other():
    """Correlation could in principle be broken by adding enough noise while
    leaving the axes information-free. Check the stronger property: the best
    least-squares scalar fit of y from x leaves most of y unexplained."""
    _, accel = one_window("bearing_outer", 0.3)
    x = accel[:, 0]
    for j in (1, 2):
        y = accel[:, j]
        k = float(x @ y / (x @ x))          # best scale factor
        resid = np.std(y - k * x) / np.std(y)
        assert resid > 0.5, (
            f"axis {j} is {resid:.3f} unexplained by a scaled copy of x; "
            "before T1.8 this was ~0.05")


# ----------------------------------------------------------------------------
# 2. Axis x, and therefore everything that reads only channel 0, is unchanged
# ----------------------------------------------------------------------------

def _legacy_single_axis(kind, sev, fr, seed, window_s=4.0):
    """Exactly what `SimulatedSource._generate` used to compute for `ax`:
    audio first, then one call to the same generator at the accel rate, from
    the same seeded Generator. Reproduced here from `ml/simulate.py`'s public
    API so the equality below is a real check and not a tautology."""
    from simulate import (SimConfig, bearing_fault_signal, imbalance_signal,
                          normal_signal)
    cfg = SimConfig(duration_s=window_s, fr=fr, fs_audio=FS_A, fs_accel=FS_V)
    gens = {
        "normal": lambda fs, r: normal_signal(cfg, fs, r),
        "bearing_outer": lambda fs, r: bearing_fault_signal(cfg, fs, r, sev, "outer"),
        "bearing_inner": lambda fs, r: bearing_fault_signal(cfg, fs, r, sev, "inner"),
        "imbalance": lambda fs, r: imbalance_signal(cfg, fs, r, sev),
    }
    rng = np.random.default_rng(seed)
    audio = gens[kind](FS_A, rng)
    ax = gens[kind](FS_V, rng)
    return audio, ax


@pytest.mark.parametrize("kind,sev", [
    ("normal", 0.0), ("bearing_outer", 0.15), ("bearing_inner", 0.3),
    ("imbalance", 0.5),
])
def test_audio_and_axis_x_are_bit_identical_to_the_old_simulator(kind, sev):
    """The whole experiment rests on this. Audio is untouched and axis 0 is
    the old single-axis signal to the last bit, so:
      * every audio feature is unchanged,
      * `band_energy_ilr(accel[:, 0])` is unchanged,
      * `estimate_fr`, which reads `accel[:, 0]`, is unchanged,
    and the only thing T1.8 can have altered is the eight y/z statistics."""
    audio, accel = one_window(kind, sev, seed=7)
    ref_audio, ref_ax = _legacy_single_axis(kind, sev, 50.0, seed=7)
    assert np.array_equal(audio, ref_audio)
    assert np.array_equal(accel[:, 0], ref_ax)


def test_features_that_read_only_channel_zero_are_unchanged():
    """The end-to-end version of the test above, at the feature level."""
    audio, accel = one_window("bearing_outer", 0.2, seed=11)
    ref_audio, ref_ax = _legacy_single_axis("bearing_outer", 0.2, 50.0, seed=11)
    new = extract_features(audio, FS_A, accel, FS_V)
    # feed the OLD 1-D accel: features.py pads it to 3 axes with zeros, so
    # only the x block and the channel-0-derived blocks are comparable
    old = extract_features(ref_audio, FS_A, ref_ax, FS_V)
    keep = [i for i, n in enumerate(FEATURE_NAMES)
            if n.startswith("audio_") or n.startswith("env_")
            or n.startswith("accel_band_") or n.startswith("accel_x_")]
    assert np.allclose(new["vector"][keep], old["vector"][keep], rtol=0, atol=0)
    assert new["fr_hz"] == old["fr_hz"]
    assert new["fr_reliable"] == old["fr_reliable"]
    assert new["band"] == old["band"]


# ----------------------------------------------------------------------------
# 3. The axes differ in the ways the physics claims, not just by noise
# ----------------------------------------------------------------------------

def _band_energy(x, fs, lo, hi):
    X = np.fft.rfft(x * np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    m = (f >= lo) & (f < hi)
    return float(np.sum(np.abs(X[m]) ** 2))


def test_each_axis_rings_at_its_own_structural_resonance():
    """A strong outer-race fault, so the resonance dominates. Each axis should
    put its burst energy in a different place; before T1.8 all three peaked at
    exactly the same frequency because they were the same filtered signal."""
    _, accel = one_window("bearing_outer", 0.8, window_s=8.0)
    peaks = []
    for j in range(3):
        x = accel[:, j]
        X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1 / FS_V)
        hf = f > 400.0                      # ignore the shaft hum
        peaks.append(float(f[hf][np.argmax(X[hf])]))
    # ordered x > y > z, and separated by more than either bandwidth
    assert peaks[0] > peaks[1] > peaks[2], peaks
    assert peaks[0] - peaks[1] > 200.0, peaks
    assert peaks[1] - peaks[2] > 200.0, peaks


def test_the_radial_pair_is_in_quadrature_at_shaft_frequency():
    """x and y are 90 degrees apart on the housing, so the rotating imbalance
    vector reaches them a quarter turn apart. Measured on the 1x line."""
    fr = 50.0
    _, accel = one_window("normal", 0.0, fr=fr, window_s=8.0)
    n = len(accel)
    f = np.fft.rfftfreq(n, 1 / FS_V)
    k = int(np.argmin(np.abs(f - fr)))
    X = np.fft.rfft(accel[:, 0])[k]
    Y = np.fft.rfft(accel[:, 1])[k]
    dphi = np.degrees(np.angle(Y / X))
    assert 80.0 < abs(dphi) < 100.0, f"1x phase difference {dphi:.1f} deg"


def test_the_axial_axis_is_weakly_coupled():
    """A radial impact reaches the axial sensing element through a bolted
    joint, so z carries much less of the fault energy than x. This is the
    reason z is worth measuring at all: it is a different view, not a copy."""
    _, healthy = one_window("normal", 0.0, window_s=8.0)
    _, faulty = one_window("bearing_outer", 0.8, window_s=8.0)
    gain = []
    for j in range(3):
        lo, hi = 400.0, 0.5 * FS_V
        gain.append(_band_energy(faulty[:, j], FS_V, lo, hi)
                    / _band_energy(healthy[:, j], FS_V, lo, hi))
    assert gain[0] > gain[1] > gain[2] > 1.0, gain


def test_all_three_axes_report_the_same_repetition_rate():
    """The axes must be decorrelated by the STRUCTURE, not by pretending the
    defect hits three different times. One shaft, one defect, one impulse
    train — so however different the three resonances are, demodulating each
    axis in its OWN band must recover the same BPFO.

    (An earlier draft of this test cross-correlated the raw Hilbert envelopes
    and demanded a peak at lag 0. Measured, that peaks at −2 and +1 samples
    with a correlation of only 0.04–0.12: the three modal filters have
    different group delays, and a broadband envelope is mostly noise. The
    offset is ±0.3 ms against a 6.6 ms BPFO period, so it is irrelevant to
    the quantity the detector actually reads — which is this one.)"""
    from features import envelope_spectrum
    from simulate import SimConfig
    cfg = SimConfig()
    bpfo = cfg.bearing.bpfo(50.0)
    _, accel = one_window("bearing_outer", 0.6, window_s=8.0)
    f0_x = min(cfg.resonance_hz, 0.4 * FS_V)
    for j, a in enumerate("xyz"):
        m = ACCEL_AXES[a]
        f0 = m["f0_ratio"] * f0_x
        bw = f0 / max(2.0, m["q_ratio"] * cfg.resonance_q)
        f, sp = envelope_spectrum(accel[:, j], FS_V, (f0 - 3 * bw, f0 + 3 * bw))
        sel = (f > 50.0) & (f < 400.0)
        peak = float(f[sel][np.argmax(sp[sel])])
        assert abs(peak - bpfo) / bpfo < 0.02, (
            f"axis {a} demodulates to {peak:.2f} Hz, BPFO is {bpfo:.2f} Hz")
        ratio = sp[sel].max() / np.median(sp[sel])
        assert ratio > 5.0, f"axis {a} BPFO peak only {ratio:.1f}x background"


def test_cross_axis_leakage_is_small_but_present():
    """Transverse sensitivity is a real property of a MEMS triaxial part, so
    the model includes it. It must be a few per cent — big enough to be
    honest, small enough not to reintroduce F6."""
    assert 0.0 < ACCEL_AXES["cross_axis"] < 0.10


# ----------------------------------------------------------------------------
# 4. The information the change was for
# ----------------------------------------------------------------------------

def _accel_stat_columns():
    return [i for i, n in enumerate(FEATURE_NAMES)
            if n.startswith("accel_") and "band" not in n]


def _effective_rank(X):
    Xc = X - X.mean(0)
    Xc = Xc / (Xc.std(0) + 1e-30)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def test_the_twelve_accel_statistics_now_span_more_than_four_dimensions():
    """The number F6 was actually about. Before T1.8: effective rank 3.75 of
    12, with the smallest four singular values at ~1e-3 of the largest — a
    four-dimensional near-null space, which is exactly what duplicating one
    axis twice produces. The bar is set at 6.0, comfortably above the old
    value and comfortably below what was measured after the fix, so this
    test fails if anyone reintroduces copies."""
    idx = _accel_stat_columns()
    rows = []
    for w in range(40):
        fr = 50.0 if w % 2 == 0 else 30.0
        audio, accel = one_window("normal", 0.0, fr=fr, seed=2000 + w)
        rows.append(extract_features(audio, FS_A, accel, FS_V)["vector"][idx])
    X = np.array(rows)
    assert _effective_rank(X) > 6.0, _effective_rank(X)
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-30)
    s = np.linalg.svd(Xc, compute_uv=False)
    # measured 1.83e-2 after the change, 1.3e-3 before; the bar sits between
    assert s[-1] / s[0] > 0.008, f"sv ratio {s[-1]/s[0]:.3e} (was 1.3e-3)"


def test_most_y_and_z_statistics_are_no_longer_determined_by_the_x_ones():
    """The sharpest version: regress each y/z statistic on the four x
    statistics across HEALTHY windows — the population the baseline covariance
    is actually fitted to. If y is a scaled copy of x, every R^2 is ~1.

    Measured after T1.8: 0.14 / 0.99 / 0.50 / 0.18 for y and
    0.17 / 0.04 / 0.10 / 0.54 for z, median 0.18.

    THE ONE THAT STAYS HIGH IS `accel_y_kurt` (0.99), and that is recorded
    rather than tuned away: y is the other radial axis, so it sees the same
    shaft hum at 0.72 of the amplitude, and on a healthy machine impulsiveness
    is set by the hum-to-noise ratio, which is then nearly common. The axial
    axis, whose hum gain is 0.28, breaks the tie (R^2 0.04). On a fault ramp
    three of the eight exceed 0.95 — also expected, and also not a bug: a
    growing defect drives level and impulsiveness up on every axis at once.
    """
    idx = _accel_stat_columns()
    rows = []
    for w in range(40):
        fr = 50.0 if w % 2 == 0 else 30.0
        audio, accel = one_window("normal", 0.0, fr=fr, seed=3000 + w)
        rows.append(extract_features(audio, FS_A, accel, FS_V)["vector"][idx])
    X = np.array(rows)
    A = np.column_stack([X[:, :4], np.ones(len(X))])   # x block + intercept
    r2 = []
    for j in range(4, 12):
        y = X[:, j]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss_res = np.sum((y - A @ beta) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2.append(1 - ss_res / (ss_tot + 1e-30))
    r2 = np.array(r2)
    assert np.median(r2) < 0.5, f"median R^2 {np.median(r2):.3f}: {r2.round(3)}"
    assert (r2 < 0.95).sum() >= 6, f"only {(r2 < 0.95).sum()} of 8 free: {r2.round(3)}"


# ----------------------------------------------------------------------------
# 5. Nothing else moved
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_shape_dtype_and_determinism(kind):
    a1, v1 = one_window(kind, 0.3, seed=5)
    a2, v2 = one_window(kind, 0.3, seed=5)
    assert v1.shape == (int(4.0 * FS_V), 3)
    assert np.isfinite(v1).all()
    assert np.array_equal(v1, v2) and np.array_equal(a1, a2)


def test_a_different_seed_gives_a_different_window():
    _, v1 = one_window("normal", 0.0, seed=5)
    _, v2 = one_window("normal", 0.0, seed=6)
    assert not np.array_equal(v1, v2)


def test_feature_vector_is_still_37_dimensional_and_finite():
    for kind in KINDS:
        audio, accel = one_window(kind, 0.3)
        v = extract_features(audio, FS_A, accel, FS_V)["vector"]
        assert v.shape == (37,)
        assert np.isfinite(v).all()


def test_mic_only_path_is_untouched():
    """Degraded mic-only mode feeds a zero accelerometer; T1.8 must not have
    made that produce NaNs or a different vector length."""
    audio, _ = one_window("bearing_outer", 0.3)
    zeros = np.zeros((int(4.0 * FS_V), 3))
    v = extract_features(audio, FS_A, zeros, FS_V)["vector"]
    assert v.shape == (37,)
    assert np.isfinite(v).all()


def test_soak_simulator_uses_the_same_axis_model():
    """`tools/simulate_soak.py` duplicated the old three-line axis model. If
    the two drift apart, soak numbers stop being comparable with everything
    else in the repo — so it must import the shared function."""
    src = (ROOT / "tools" / "simulate_soak.py").read_text()
    assert "simulated_accel_axes" in src
    assert "0.6 * ax" not in src
