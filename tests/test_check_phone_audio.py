"""
Tests for tools/check_phone_audio.py — the pre-flight check that decides
whether a phone recording is usable at all.

Why this file matters more than its size suggests: the tool's whole job is to
answer "did your phone's AGC destroy the level information", and a tool that
*claims* to detect AGC without ever being shown AGC is an opinion. These tests
construct signals where the answer is known by construction and check the tool
agrees.

The physics being pinned: sound pressure from a small source falls as 1/r, so
moving 10 cm -> 40 cm costs 20*log10(4) = 12.04 dB. AGC cancels that. The
detector is the difference between those two cases.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_phone_audio", ROOT / "tools" / "check_phone_audio.py")
cpa = importlib.util.module_from_spec(_spec)
sys.modules["check_phone_audio"] = cpa
_spec.loader.exec_module(cpa)

SR = 16000
DUR = 60.0


def _two_distance_signal(drop_db: float, seed: int = 0) -> np.ndarray:
    """Steady broadband source that drops by `drop_db` half way through —
    a synthetic near-then-far recording."""
    rng = np.random.default_rng(seed)
    n = int(SR * DUR)
    t = np.arange(n) / SR
    x = rng.normal(0, 1, n) * 0.05
    return x * np.where(t < DUR / 2, 1.0, 10 ** (-drop_db / 20.0))


def _apply_agc(x: np.ndarray, block_s: float = 0.5, target: float = 0.05):
    """Crude block-wise normaliser: what 'the phone turns the gain down' does."""
    blk = int(block_s * SR)
    y = x.copy()
    for i in range(0, len(y) - blk, blk):
        seg = y[i:i + blk]
        r = np.sqrt(np.mean(seg ** 2))
        if r > 1e-9:
            y[i:i + blk] = seg * (target / r)
    return y


def test_level_envelope_removes_dc_before_measuring():
    """F10's lesson applied here: a DC offset inflates raw RMS, which would
    mask the very level change this tool exists to measure."""
    x = np.random.default_rng(1).normal(0, 0.05, SR * 4)
    clean = cpa.level_envelope(x, SR)
    offset = cpa.level_envelope(x + 0.5, SR)
    assert np.allclose(clean, offset, atol=0.05), (
        "a DC offset moved the level envelope, so DC is not being removed")


def test_clean_recording_shows_the_inverse_r_drop():
    env = cpa.level_envelope(_two_distance_signal(cpa.EXPECTED_DROP_DB), SR)
    d = cpa.distance_test(env)
    assert d["drop_db"] == pytest.approx(cpa.EXPECTED_DROP_DB, abs=1.0)
    assert d["drop_db"] >= cpa.AGC_CLEAR_DB, "clean signal must not be flagged"


def test_agc_erases_the_drop_and_is_flagged():
    """The headline case. Same underlying signal, AGC applied: the 12 dB of
    real information is gone."""
    agc = _apply_agc(_two_distance_signal(cpa.EXPECTED_DROP_DB))
    d = cpa.distance_test(cpa.level_envelope(agc, SR))
    assert abs(d["drop_db"]) < cpa.AGC_SUSPECT_DB, (
        f"AGC left a {d['drop_db']:.1f} dB drop; detector would miss it")


def test_the_two_cases_are_actually_distinguishable():
    """Guards against a detector that flags everything or nothing — the
    failure mode that makes a check worthless rather than wrong."""
    clean = cpa.distance_test(cpa.level_envelope(
        _two_distance_signal(cpa.EXPECTED_DROP_DB), SR))["drop_db"]
    agc = cpa.distance_test(cpa.level_envelope(
        _apply_agc(_two_distance_signal(cpa.EXPECTED_DROP_DB)), SR))["drop_db"]
    assert clean - agc > 8.0, "clean and AGC cases are not separated"


def test_partial_agc_lands_in_the_ambiguous_band_not_a_confident_verdict():
    """A 6 dB drop is neither 1/r nor flat. The tool must say 'ambiguous'
    rather than pick a side — over-confidence on a marginal reading is how a
    bad recording gets trusted."""
    d = cpa.distance_test(cpa.level_envelope(_two_distance_signal(6.0), SR))
    assert cpa.AGC_SUSPECT_DB <= d["drop_db"] < cpa.AGC_CLEAR_DB


def test_short_recording_refuses_rather_than_guessing():
    with pytest.raises(SystemExit):
        cpa.distance_test(np.array([1.0, 2.0]))


def _brickwalled(x: np.ndarray, cutoff_hz: float, sr: int = SR) -> np.ndarray:  # noqa: F811
    """What a lossy codec does to the top of the spectrum: not attenuation,
    deletion. Measured on ffmpeg AAC at 32 kbps: flat to 10 kHz, then -78 dB.
    Reproduced here by zeroing the FFT above the cutoff, so the test needs no
    ffmpeg and runs in milliseconds."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[f >= cutoff_hz] = 0.0
    return np.fft.irfft(X, n=len(x))


def test_lossy_brick_wall_is_detected():
    """F17. A recording that has lost its top octave looks and sounds fine but
    cannot show a resonance above the cliff — a faulty machine reads as a
    quiet healthy one."""
    x = np.random.default_rng(5).normal(0, 0.05, SR * 8)
    assert cpa.lossy_cutoff_hz(_brickwalled(x, 5000.0)) == pytest.approx(5000, abs=1100)


def test_full_bandwidth_audio_is_not_flagged():
    """Guards the guard: a checker that flags everything is worthless."""
    x = np.random.default_rng(6).normal(0, 0.05, SR * 8)
    assert cpa.lossy_cutoff_hz(x) is None


def test_a_single_quiet_band_is_not_mistaken_for_a_codec_cliff():
    """A machine's own spectrum can have a notch. Only a cliff that persists
    all the way to Nyquist is a codec limit."""
    x = np.random.default_rng(7).normal(0, 0.05, SR * 8)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X[(f >= 4000) & (f < 5000)] = 0.0          # one dead band, then normal again
    notched = np.fft.irfft(X, n=len(x))
    assert cpa.lossy_cutoff_hz(notched) is None


def test_a_codec_cliff_destroys_a_fault_signature_above_it():
    """F17's danger claim, pinned end to end on the REAL feature code.

    A brick wall does not merely attenuate a high resonance — it removes the
    fault signature entirely, and `select_demodulation_band` then falls back
    to its default band and reports nothing unusual. That silent fallback is
    the failure mode: the machine reads as healthy.

    Measured with ffmpeg AAC at 32 kbps on a 12 kHz resonance: envelope-
    spectrum peak-to-background at BPFO fell 59.6x -> 2.9x and the selected
    band fell back from (7598, 12616) to (3000, 6000). Reproduced here with an
    FFT brick wall so the test needs no ffmpeg.
    """
    sys.path.insert(0, str(ROOT / "firmware"))
    from features import envelope_spectrum, select_demodulation_band

    fs, dur, bpfo, res = 44100, 20.0, 73.6, 12000.0
    n = int(fs * dur)
    t = np.arange(n) / fs
    rng = np.random.default_rng(11)

    # Impacts at BPFO ringing a 12 kHz resonance — the signature the detector
    # is built to find, deliberately placed above a 10 kHz codec cliff.
    sig = rng.normal(0, 0.01, n)
    strikes = (np.arange(0, dur, 1.0 / bpfo) * fs).astype(int)
    strikes = strikes[strikes < n - 1]
    imp = np.zeros(n)
    imp[strikes] = 1.0
    ring = np.exp(-t[:int(0.004 * fs)] * 900.0) * np.sin(2 * np.pi * res * t[:int(0.004 * fs)])
    sig += np.convolve(imp, ring)[:n] * 0.5

    def peak_ratio(x):
        band, _ = select_demodulation_band(x, fs)
        f, e = envelope_spectrum(x, fs, band)
        m = np.abs(f - bpfo) < 3 * (f[1] - f[0])
        bg = float(np.median(e[(f > 20) & (f < 400)]))
        return (float(np.max(e[m])) / max(bg, 1e-12) if m.any() else 0.0), band

    full_ratio, full_band = peak_ratio(sig)
    cut_ratio, cut_band = peak_ratio(_brickwalled(sig, 10000.0, sr=fs))

    assert full_ratio > 5.0, (
        f"the uncompressed signature is only {full_ratio:.1f}x — the fixture "
        f"is not producing a detectable fault, so the test proves nothing")
    assert cut_ratio < 0.5 * full_ratio, (
        f"brick wall left the signature at {cut_ratio:.1f}x of "
        f"{full_ratio:.1f}x — expected it to collapse")
    assert full_band[1] > 10000.0, "fixture's resonance was not above the cliff"

    # NOT asserted: that the selector falls back to the default band. With
    # real ffmpeg output it did — (7598, 12616) -> (3000, 6000) — but an FFT
    # brick wall leaves 7.6-10 kHz of that band intact, so the selector still
    # picks it here. The fallback is a property of that codec's output, not a
    # universal consequence, and this fixture does not demonstrate it. The
    # robust, fixture-independent claim is the collapse above.
    assert cut_band is not None


def test_self_test_passes():
    """The tool's own --self-test must pass, or its advice is unfounded."""
    assert cpa._self_test() == 0
