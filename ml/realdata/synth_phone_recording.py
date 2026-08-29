"""
synth_phone_recording.py — a realistic mic-only healthy/faulty pair, for
testing the phone-recording path (docs/PHONE_RECORDING.md, T7.2) before a
real phone recording exists.

WHY THIS FILE EXISTS AND IS NOT JUST ml/simulate.py
----------------------------------------------------------------------------
`ml/simulate.py` (frozen) generates band-limited WHITE noise as the machine
noise floor. `docs/DOC_STATUS.md` already records a measured consequence of
that choice: on a *pink*-noise floor (the T1.1 CWRU surrogate),
`features.select_demodulation_band`'s protrugram never reached its
`crest_floor = 10.0` on any file and silently fell back to `DEFAULT_BAND`
(3-6 kHz) every time. That fallback happened to contain the resonance in the
T1.1 surrogate, so nothing broke there — but it means the band selector has
never been tested on realistic noise WHERE THE RESONANCE IS OUTSIDE THE
DEFAULT BAND, which is exactly the case a real housing can produce (OVERVIEW
says resonances run 1-20 kHz; the default band only covers a slice of that).

This file is an independent third signal model — not imported from
`ml/simulate.py`, not imported from `validate_public_dataset.py`'s `_pink`
(different generator: time-domain Voss-McCartney-style octave sum here,
frequency-domain 1/sqrt(f) shaping there) — so that agreement or disagreement
between them is informative rather than tautological.

WHAT IT MODELS
----------------------------------------------------------------------------
A small mains-driven motor (fan, pump, extractor) recorded on a phone
2-30 cm from the housing: a pink noise floor (real machinery/room noise is
pink, not white), 50 Hz mains hum with a second harmonic, broadband handling/
room noise, and a structural resonance that:
  * in the HEALTHY signal is excited only by random, non-periodic knocks
    (footsteps, door, unrelated machine nearby) - there is no periodicity at
    the fault rate;
  * in the FAULTY signal is ALSO excited periodically at BPFO by a
    bearing-defect impact train, on top of the same random knocks.
The resonance frequency defaults to 1600 Hz, deliberately OUTSIDE
`features.DEFAULT_BAND` (3-6 kHz), to stress the part of the pipeline that
the T1.1 finding could not: does the protrugram find a real resonance that
the default band would miss, once the noise floor is realistic pink rather
than simulate.py's white?

USAGE
----------------------------------------------------------------------------
    python ml/realdata/synth_phone_recording.py --out-dir /tmp/phone_test
    python ml/realdata/synth_phone_recording.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fault_frequencies import lookup, rpm_to_hz  # noqa: E402

PHONE_FS = 44100.0          # a typical phone voice-memo sample rate
DEFAULT_RPM = 1450.0        # a small mains-driven motor, ~synchronous 1500 rpm - slip
DEFAULT_BEARING = "6202"    # small deep-groove ball bearing (15x35x11 mm)
DEFAULT_RESONANCE_HZ = 1600.0   # deliberately outside features.DEFAULT_BAND (3-6 kHz)


def _pink_noise(n: int, rng: np.random.Generator, n_sources: int = 16) -> np.ndarray:
    """Time-domain pink noise: sum of `n_sources` octave-spaced random walks
    (Voss-McCartney). Independent of validate_public_dataset._pink, which
    shapes white noise in the frequency domain instead - a different
    algorithm reaching the same ~1/f spectral slope, so using both elsewhere
    in the repo is corroboration, not duplication.

    Vectorised 2026-08-23. The original was a per-sample Python loop with an
    inner loop over sources — O(n · n_sources) in interpreted code, which
    profiling showed was **98 % of this module's total runtime** (9.07 s of
    9.23 s for 30 s of audio; 1.32 million `.sum()` calls, one per sample).
    That made generating a realistic learn period — 48 windows × 30 s = 24
    minutes of audio — take roughly eight minutes, which is why the phone path
    had never actually been run end to end.

    Same algorithm, expressed as array operations. Voss-McCartney says source
    `j` re-draws every `2**j` samples and holds its value in between, so the
    whole of source `j` is one `np.repeat` of `ceil(n / 2**j)` draws. Summing
    the sources is then a single vectorised add.

    **Not bit-identical to the old version** — the random draws happen in a
    different order — but statistically the same process, and the property
    that matters is the spectral slope. Pinned by
    `tests/test_pink_noise_is_pink.py`, which measures the slope of
    log(PSD) vs log(f) and requires it near −1.

    ⚠ The FIRST attempt at this vectorisation used
    `out += np.repeat(draws, step)[:n]`, which is correct but allocates a
    full-length temporary on every one of the 16 octaves. At 1800 s × 44.1 kHz
    that is 635 MB per octave and the process was OOM-killed (exit 137) —
    trading a time problem for a memory one. The version below writes into a
    reshaped **view** of `out`, so the only allocation per octave is the
    `n / 2**j` draws themselves. Float32 halves it again; a noise floor does
    not need 15 significant figures.
    """
    out = np.zeros(n, dtype=np.float32)
    for j in range(n_sources):
        step = 1 << j                        # 2**j, the hold length for this source
        whole = n // step                    # complete blocks
        if whole:
            draws = rng.standard_normal(whole).astype(np.float32)
            # Reshape of a contiguous array is a VIEW, so this adds in place
            # rather than building an n-length temporary.
            out[:whole * step].reshape(whole, step)[:] += draws[:, None]
        tail = n - whole * step              # partial final block, if any
        if tail:
            out[whole * step:] += np.float32(rng.standard_normal())
    out -= out.mean()
    out /= (out.std() + 1e-12)
    return out.astype(float)


def _ring(n: int, fs: float, f0: float, q: float, rng: np.random.Generator,
         impact_samples: np.ndarray) -> np.ndarray:
    """Excite a bandpass resonance (centre f0, quality q) with short impulses
    at the given sample indices and let the filter's own decay do the
    ringing - the same physical idea as the system overview (not in this public copy) section 2 (an impact
    rings the housing resonance and decays over a few ms), built without
    reusing any of features.py's demodulation code."""
    x = np.zeros(n)
    valid = impact_samples[(impact_samples >= 0) & (impact_samples < n)]
    x[valid] += 1.0 + 0.1 * rng.standard_normal(len(valid))
    bw = f0 / q
    sos = butter(2, [max(f0 - bw / 2, 1.0), min(f0 + bw / 2, fs / 2 * 0.99)],
                btype="band", fs=fs, output="sos")
    return sosfilt(sos, x)


def _mains_hum(n: int, fs: float, f0: float = 50.0) -> np.ndarray:
    t = np.arange(n) / fs
    return (0.3 * np.sin(2 * np.pi * f0 * t)
            + 0.1 * np.sin(2 * np.pi * 2 * f0 * t))


def make_pair(seed: int = 1, duration_s: float = 40.0, fs: float = PHONE_FS,
             rpm: float = DEFAULT_RPM, bearing: str = DEFAULT_BEARING,
             resonance_hz: float = DEFAULT_RESONANCE_HZ, q: float = 8.0,
             severity: float = 0.35) -> dict:
    """Returns a dict with the healthy/faulty float arrays (range roughly
    [-1, 1]) plus every parameter needed to analyse them for real, so the
    caller never has to guess what this function assumed."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    geom = lookup(bearing)
    fr_hz = rpm_to_hz(rpm)
    bpfo_hz = geom.bpfo(fr_hz)

    floor = 0.05 * _pink_noise(n, rng)
    hum = 0.02 * _mains_hum(n, fs)
    room = 0.01 * rng.standard_normal(n)

    # Non-periodic knocks: present in BOTH signals, so any periodicity found
    # at BPFO in the faulty-minus-healthy comparison is genuinely the fault,
    # not an artefact of "faulty has knocks and healthy doesn't".
    n_knocks = max(1, int(duration_s * 0.5))
    knock_samples = rng.integers(0, n, size=n_knocks)
    shared_knock_ring = 0.15 * _ring(n, fs, resonance_hz, q, rng, knock_samples)

    healthy = floor + hum + room + shared_knock_ring

    period_samples = fs / bpfo_hz
    n_impacts = int(n / period_samples) + 1
    jitter = rng.normal(0, period_samples * 0.01, size=n_impacts)  # ~1% slip jitter
    impact_samples = (np.arange(n_impacts) * period_samples + jitter).astype(int)
    fault_ring = severity * _ring(n, fs, resonance_hz, q, rng, impact_samples)

    faulty = healthy + fault_ring

    return {
        "healthy": healthy, "faulty": faulty, "fs": fs,
        "fr_hz": fr_hz, "bpfo_hz": bpfo_hz, "rpm": rpm, "bearing": bearing,
        "resonance_hz": resonance_hz, "seed": seed, "duration_s": duration_s,
    }


def _to_phone_pcm(x: np.ndarray, dc_offset: float = 0.01,
                  clip_frac_target: float = 0.002) -> np.ndarray:
    """Turn a float signal into int16 PCM the way a phone actually would:
    normalise to use most of full scale, add a small DC offset (real ADCs
    are not perfectly centred), and let a handful of samples clip (automatic
    gain riding close to full scale). Mirrors tools/ingest.py's own test
    fixtures (see its --self-test) rather than inventing a new convention."""
    x = x / (np.max(np.abs(x)) + 1e-12)
    # push the target clip fraction over full scale
    target_peak = 1.0 / max(1e-6, (1.0 - clip_frac_target))
    x = x * target_peak + dc_offset
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    return pcm


def write_pair(out_dir: Path, pair: dict, stem_healthy: str = "phone_healthy",
              stem_faulty: str = "phone_faulty") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = out_dir / f"{stem_healthy}.wav"
    f_path = out_dir / f"{stem_faulty}.wav"
    wavfile.write(h_path, int(pair["fs"]), _to_phone_pcm(pair["healthy"]))
    wavfile.write(f_path, int(pair["fs"]), _to_phone_pcm(pair["faulty"]))
    return h_path, f_path


def _self_test() -> int:
    pair = make_pair(duration_s=10.0)
    h, f = pair["healthy"], pair["faulty"]
    assert h.shape == f.shape
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(f))
    # faulty must carry strictly more energy at the resonance than healthy,
    # since it is healthy + an additive fault ring
    assert np.sqrt(np.mean(f**2)) > np.sqrt(np.mean(h**2))
    print(f"self-test OK: fr={pair['fr_hz']:.4f} Hz, "
          f"BPFO={pair['bpfo_hz']:.4f} Hz, resonance={pair['resonance_hz']:.0f} Hz")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/phone_test"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--duration-s", type=float, default=40.0)
    p.add_argument("--rpm", type=float, default=DEFAULT_RPM)
    p.add_argument("--bearing", default=DEFAULT_BEARING)
    p.add_argument("--resonance-hz", type=float, default=DEFAULT_RESONANCE_HZ)
    p.add_argument("--severity", type=float, default=0.35)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--fs", type=float, default=PHONE_FS,
                   help=f"sample rate (default {PHONE_FS:.0f}, a phone's native "
                        f"voice-memo rate). Memory scales with fs x duration: "
                        f"this module needs ~2.5 MB per second of 44.1 kHz "
                        f"audio, so a full 48-window learn period (1440 s) "
                        f"wants ~3.6 GB and gets OOM-killed on a 4 GB box. "
                        f"Pass --fs 16000 — the rate the detector actually "
                        f"runs at, and what tools/ingest.py resamples to "
                        f"anyway — for 2.75x less memory.")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    # Nyquist guard: the resonance must be representable, or the file is
    # silently useless — the fault signature would alias instead of appearing.
    if args.resonance_hz >= 0.5 * args.fs:
        p.error(f"--resonance-hz {args.resonance_hz:.0f} needs fs > "
                f"{2*args.resonance_hz:.0f}; got --fs {args.fs:.0f}. The "
                f"resonance would alias and the recording would be worthless.")

    pair = make_pair(seed=args.seed, duration_s=args.duration_s, fs=args.fs,
                     rpm=args.rpm, bearing=args.bearing,
                     resonance_hz=args.resonance_hz, severity=args.severity)
    h_path, f_path = write_pair(args.out_dir, pair)
    print(f"wrote {h_path} and {f_path}")
    print(f"fs={pair['fs']:.0f} Hz  fr={pair['fr_hz']:.4f} Hz  "
          f"BPFO={pair['bpfo_hz']:.4f} Hz (bearing {pair['bearing']})  "
          f"resonance={pair['resonance_hz']:.0f} Hz  seed={pair['seed']}")
    print("*** SYNTHETIC — this is a realistic-noise proxy, not a real "
          "recording. See docs/PHONE_RECORDING.md. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
