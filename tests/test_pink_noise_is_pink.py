"""
Pins the spectral character of `synth_phone_recording._pink_noise`.

Why this file exists. On 2026-08-23 that function was vectorised — profiling
showed it was **98 % of the module's runtime** (9.07 s of 9.23 s for 30 s of
audio, from a per-sample Python loop making 1.32 million `.sum()` calls). The
rewrite is 131× faster but **not bit-identical**: Voss-McCartney's random draws
now happen in a different order.

That is a legitimate change only if the thing the function is *for* survives.
It is not for producing one particular waveform; it is for producing a
realistic **pink** noise floor, because `docs/DOC_STATUS.md` records that the
protrugram band selector behaves differently on pink noise than on the white
noise `ml/simulate.py` generates. So the property to pin is the spectrum, not
the samples.

Without this file the speedup would be unverified — "it looked fine" — which
is exactly the standard this project does not accept.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
_spec = importlib.util.spec_from_file_location(
    "synth_phone_recording", ROOT / "ml" / "realdata" / "synth_phone_recording.py")
spr = importlib.util.module_from_spec(_spec)
sys.modules["synth_phone_recording"] = spr
_spec.loader.exec_module(spr)

FS = 44100


def _slope(x: np.ndarray, fs: int = FS, lo: float = 20.0, hi: float = 5000.0) -> float:
    """Slope of log10(PSD) against log10(f). Pink is -1, white is 0."""
    f, p = welch(x, fs=fs, nperseg=8192)
    m = (f > lo) & (f < hi)
    return float(np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0])


def test_output_is_pink_not_white():
    """The headline property. -1 is pink; 0 would mean the rewrite silently
    turned it into white noise, which would quietly invalidate every phone
    test that relies on a realistic floor."""
    x = spr._pink_noise(FS * 20, np.random.default_rng(0))
    s = _slope(x)
    assert -1.35 < s < -0.65, f"spectral slope {s:+.3f} is not pink (want ~-1)"


def test_white_noise_would_fail_this_test():
    """Guards the guard. A slope test that passes on white noise proves
    nothing — this asserts the measurement can actually tell them apart."""
    w = np.random.default_rng(1).standard_normal(FS * 20)
    assert _slope(w) > -0.4, "white noise is not measuring as flat; check _slope"


def test_normalised_to_zero_mean_unit_variance():
    x = spr._pink_noise(FS * 5, np.random.default_rng(2))
    assert abs(float(x.mean())) < 1e-6
    assert float(x.std()) == pytest.approx(1.0, abs=1e-6)


def test_length_is_exact_for_lengths_that_are_not_powers_of_two():
    """The vectorised version builds each octave with `np.repeat` and trims.
    An off-by-one here would silently shorten or lengthen every recording."""
    for n in (1, 2, 3, 1000, 44100, 44101):
        assert len(spr._pink_noise(n, np.random.default_rng(3))) == n


def test_is_deterministic_for_a_given_seed():
    a = spr._pink_noise(4096, np.random.default_rng(7))
    b = spr._pink_noise(4096, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_is_fast_enough_to_generate_a_real_learn_period():
    """The point of the rewrite. A 48-window learn period is 24 minutes of
    audio; at the old speed that was ~8 minutes of generation, which is why
    the phone path had never been run end to end. Deliberately loose — this
    guards against regressing to a per-sample loop, not against a slow CI box.
    """
    import time
    t0 = time.perf_counter()
    spr._pink_noise(FS * 30, np.random.default_rng(4))
    dt = time.perf_counter() - t0
    assert dt < 2.0, (
        f"{dt:.2f}s for 30 s of audio — the per-sample loop is back "
        f"(vectorised version measured 0.07 s)")
