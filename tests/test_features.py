"""
Stage-2 gate tests. Geometry (BPFO) appears ONLY here as ground truth for the
synthetic signals — the feature code itself is geometry-free by design.
"""

import time

import numpy as np
import pytest

from features import (FEATURE_NAMES, band_energy_ratios, channel_stats,
                      envelope_spectrum, estimate_fr, extract_features,
                      mel_filterbank, select_demodulation_band, DEFAULT_BAND)
from simulate import SimConfig, bearing_fault_signal, normal_signal

CFG = SimConfig(duration_s=30.0)


def _window(kind="normal", severity=0.0, fr=50.0, seed=0):
    cfg = SimConfig(duration_s=30.0, fr=fr)
    ra, rv = np.random.default_rng(seed), np.random.default_rng(seed + 1000)
    if kind == "normal":
        return (normal_signal(cfg, cfg.fs_audio, ra),
                normal_signal(cfg, cfg.fs_accel, rv), cfg)
    return (bearing_fault_signal(cfg, cfg.fs_audio, ra, severity, "outer"),
            bearing_fault_signal(cfg, cfg.fs_accel, rv, severity, "outer"), cfg)


# ----------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------

def test_mel_filterbank_shape_and_coverage():
    fb = mel_filterbank(64, 1024, 16000, 20.0, 8000.0)
    assert fb.shape == (64, 513)
    assert (fb.sum(axis=1) > 0).all()
    freqs = np.fft.rfftfreq(1024, 1 / 16000)
    covered = fb.sum(axis=0) > 0
    assert covered[(freqs > 100) & (freqs < 7800)].all()


def test_channel_stats_gaussian_reference():
    x = np.random.default_rng(0).standard_normal(100_000)
    s = channel_stats(x)                      # logrms, kurt, crest, skew
    assert abs(s[1]) < 0.1                    # Gaussian kurtosis ~ 0 (Fisher)
    assert 3.5 < s[2] < 6.0                   # Gaussian crest factor ~ 4-5
    assert abs(s[3]) < 0.1


def test_band_energy_ratios_are_fractions():
    x = np.random.default_rng(1).standard_normal(160_000)
    r = band_energy_ratios(x, 16000.0)
    assert r.shape == (8,)
    assert 0.9 < np.sum(10.0 ** r) < 1.1      # log-fractions sum back to ~1


# ----------------------------------------------------------------------------
# fr estimation: correct value, correct self-doubt
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("fr", [50.0, 30.0])
def test_fr_estimate_accurate_and_reliable(fr):
    audio, accel, cfg = _window("normal", fr=fr, seed=2)
    est, reliable = estimate_fr(audio, cfg.fs_audio, accel, cfg.fs_accel)
    assert reliable, "channels agree on synthetic data — must be flagged reliable"
    assert abs(est - fr) < 0.5, f"fr {est:.2f} vs true {fr}"


def test_fr_unreliable_when_channels_disagree():
    audio, _, cfg = _window("normal", fr=50.0, seed=3)
    _, accel, _ = _window("normal", fr=30.0, seed=4)   # mismatched machine
    _, reliable = estimate_fr(audio, cfg.fs_audio, accel, cfg.fs_accel)
    assert not reliable


# ----------------------------------------------------------------------------
# demodulation band + envelope: the stage-2 core gate
# ----------------------------------------------------------------------------

def test_band_selection_finds_resonance_on_fault():
    audio, _, cfg = _window("fault", severity=0.5, seed=5)
    (lo, hi), crest = select_demodulation_band(audio, cfg.fs_audio)
    assert lo <= cfg.resonance_hz <= hi, f"chose {lo:.0f}-{hi:.0f}, resonance at {cfg.resonance_hz}"
    assert crest > 50


def test_band_selection_defaults_when_healthy():
    audio, _, cfg = _window("normal", seed=6)
    band, crest = select_demodulation_band(audio, cfg.fs_audio)
    assert band == DEFAULT_BAND
    assert crest < 10


def test_envelope_detects_bpfo_at_severity_015():
    """GATE: severity-0.15 fault must show an envelope-spectrum peak at BPFO
    (ground truth from sim geometry; the detector itself never knows it)."""
    audio, _, cfg = _window("fault", severity=0.15, seed=7)
    bpfo = cfg.bearing.bpfo(cfg.fr)
    band, _ = select_demodulation_band(audio, cfg.fs_audio)
    freqs, mag = envelope_spectrum(audio, cfg.fs_audio, band)
    sel = (freqs >= 30) & (freqs <= 500)
    f_peak = freqs[sel][np.argmax(mag[sel])]
    assert abs(f_peak - bpfo) / bpfo < 0.05, f"peak {f_peak:.1f} Hz vs BPFO {bpfo:.1f} Hz"


def test_env_crest_feature_separates_early_fault():
    i_crest = FEATURE_NAMES.index("env_crest")
    a_n, v_n, cfg = _window("normal", seed=8)
    a_f, v_f, _ = _window("fault", severity=0.15, seed=8)
    c_n = extract_features(a_n, cfg.fs_audio, v_n, cfg.fs_accel)["vector"][i_crest]
    c_f = extract_features(a_f, cfg.fs_audio, v_f, cfg.fs_accel)["vector"][i_crest]
    assert c_f > c_n + 0.5, f"env_crest fault {c_f:.2f} vs normal {c_n:.2f} (log10 scale)"


# ----------------------------------------------------------------------------
# contract + performance gate
# ----------------------------------------------------------------------------

def test_extract_features_contract_and_speed():
    audio, accel, cfg = _window("normal", seed=9)
    t0 = time.perf_counter()
    out = extract_features(audio, cfg.fs_audio, accel, cfg.fs_accel)
    dt = time.perf_counter() - t0
    # 37 dims, not the original 40: T1.5 replaced three blocks of energy
    # fractions with their isometric log-ratio coordinates, each of which needs
    # one fewer number than the composition it encodes. See
    # tests/test_compositional.py.
    assert out["vector"].shape == (37,)
    assert len(FEATURE_NAMES) == 37
    assert out["mel"].shape[0] == 64
    assert np.isfinite(out["vector"]).all()
    # Perf gate proxy: < 2 s on a single A53 core ~= 8-10x this x86 sandbox.
    # We require < 0.5 s here, leaving 4x headroom for the Pi.
    assert dt < 0.5, f"extraction took {dt:.2f}s in sandbox — will miss the Pi budget"


def test_single_axis_accel_accepted():
    audio, accel, cfg = _window("normal", seed=10)
    out = extract_features(audio, cfg.fs_audio, accel, cfg.fs_accel)   # (n,) accel
    assert out["vector"].shape == (37,)


# ----------------------------------------------------------------------------
# Regression tests for the dead-channel / mic-only speed-estimation bug
# (found 2026-08-16 by adversarial self-review; see docs/DOC_SELF_REVIEW.md F2)
# ----------------------------------------------------------------------------

def test_hps_returns_zero_for_dead_channel():
    """A silent channel must report 'no signal' (0.0), NOT the search lower
    bound. argmax over a flat score surface returns the first candidate, which
    looks like a plausible 10 Hz reading from a channel that measured nothing."""
    from features import _hps_peak
    assert _hps_peak(np.zeros(16000), 16000.0, 10.0, 120.0) == 0.0
    assert _hps_peak(np.zeros(64), 6400.0, 10.0, 120.0) == 0.0


@pytest.mark.parametrize("fr", [50.0, 30.0])
def test_fr_mic_only_uses_audio(fr):
    """MIC-ONLY BUILD: with no accelerometer the audio estimate must be used.
    The original code preferred the accelerometer unconditionally and so
    returned a dead channel's 10 Hz boundary artefact instead of a correct
    50 Hz audio estimate — and speed feeds regime clustering."""
    audio, accel, cfg = _window("normal", fr=fr, seed=21)
    est, reliable = estimate_fr(audio, cfg.fs_audio,
                                np.zeros_like(accel), cfg.fs_accel)
    assert abs(est - fr) < 0.5, f"mic-only fr {est:.2f} should be ~{fr}"
    assert not reliable, "a single unconfirmed channel must not claim reliable"


def test_fr_both_channels_dead():
    """Neither channel live -> 0.0 and unreliable, never a fabricated number."""
    est, reliable = estimate_fr(np.zeros(16000), 16000.0,
                                np.zeros((6400, 3)), 6400.0)
    assert est == 0.0 and not reliable


# ----------------------------------------------------------------------------
# F10 regression: channel_stats must be blind to DC.
# Found 2026-08-19 from an EXTERNAL critique (a hardware fact our own review
# could not generate). See docs/DOC_SELF_REVIEW.md F10.
# ----------------------------------------------------------------------------

def test_channel_stats_is_invariant_to_dc_offset():
    """A DC offset is a constant and carries no machine information. Every
    statistic must be unchanged by it. Pre-fix, a 10 % offset put a healthy
    window at 2.62x threshold — permanent false alarm on an SPH0645, which is
    documented to carry an offset."""
    from features import channel_stats
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50_000)
    ref = channel_stats(x)
    for dc in (0.1, 1.0, 10.0, -5.0):
        got = channel_stats(x + dc)
        assert np.allclose(got, ref, atol=1e-9), (
            f"DC offset {dc} changed the statistics: {got} vs {ref}")


def test_accel_rms_tracks_vibration_not_gravity():
    """A real accelerometer sits in 1 g. Vibration is order 0.01-0.1 g RMS, so
    raw RMS reports gravity and is nearly blind to the machine. Pre-fix,
    quadrupling vibration moved logRMS by 0.008; it must move by ~log10(4)."""
    from features import channel_stats
    rng = np.random.default_rng(1)
    n = 50_000
    lo = channel_stats(0.05 * rng.standard_normal(n) + 1.0)[0]   # +1 g gravity
    rng = np.random.default_rng(1)
    hi = channel_stats(0.20 * rng.standard_normal(n) + 1.0)[0]
    assert abs((hi - lo) - np.log10(4.0)) < 0.02, (
        f"4x vibration moved logRMS by {hi - lo:.4f}, expected {np.log10(4):.4f}")


def test_crest_factor_survives_gravity():
    """Gravity raises max|x| and RMS together, compressing crest factor from
    ~4.7 to ~1.2 pre-fix. Crest is one of our two impact detectors."""
    from features import channel_stats
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50_000)
    assert abs(channel_stats(x + 1.0)[2] - channel_stats(x)[2]) < 1e-9


def test_dc_is_reported_as_a_diagnostic_not_a_feature():
    """DC must be visible to a field engineer (sensor fell off / I2S
    misaligned) but must NOT enter the feature vector — on a fixed install it
    is a constant, and F7 showed constant columns invite invented regimes."""
    audio, accel, cfg = _window("normal", seed=31)
    out = extract_features(audio + 0.4, cfg.fs_audio, accel + 1.0, cfg.fs_accel)
    assert out["vector"].shape == (37,), "DC must not change the vector length"
    assert abs(out["dc"]["audio"] - 0.4) < 0.01
    assert abs(out["dc"]["accel_x"] - 1.0) < 0.01


def test_dc_offset_does_not_move_the_anomaly_score():
    """End to end: the whole point. A DC-offset sensor on a healthy machine
    must not alarm."""
    from pathlib import Path
    from baseline import fit_baseline, operating_point, save_baseline
    from inference import MahalanobisScorer
    import tempfile
    clean, off = [], []
    for i in range(20):
        a, v, cfg = _window("normal", seed=100 + i)
        clean.append(extract_features(a, cfg.fs_audio, v, cfg.fs_accel))
        off.append(extract_features(a + 0.5 * float(np.std(a)), cfg.fs_audio,
                                    v + 1.0, cfg.fs_accel))
    X = np.array([f["vector"] for f in clean])
    OP = np.array([operating_point(f["vector"], f["fr_hz"]) for f in clean])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.npz"
        save_baseline(p, fit_baseline(X, OP))
        sc = MahalanobisScorer(p)
        for f in off:
            r = sc.score(f["vector"], operating_point(f["vector"], f["fr_hz"]))
            assert not r["anomalous"], (
                f"DC-offset healthy window alarmed at {r['score']/r['threshold']:.2f}x")
