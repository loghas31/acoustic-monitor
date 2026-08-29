"""
ingest.py — turn ANY recording into this project's canonical format.

WHY THIS EXISTS
--------------------------------------------------------------------------
Everything downstream of `ml/realdata/recording_io.py` assumes one exact
on-disk shape: a mono wav at the device audio rate, an optional
`<stem>_accel.csv` whose first column is a real monotonic time axis, and an
optional `<stem>.json` sidecar. `firmware/capture.FileSource` is even stricter
— it divides the PCM by a constant and infers the accelerometer rate from
`median(diff(t_s))`, with no validation whatsoever.

Real recordings are not that. A phone voice memo is 44.1 or 48 kHz, stereo,
sometimes 32-bit float, often with a DC offset and a few clipped samples. A
downloaded dataset is 12 kHz single-channel accelerometer data in a CSV. A USB
audio interface gives you 48 kHz int24-in-int32. Handing any of those to the
pipeline directly produces one of three failures, and **all three are silent**:

  1. **Wrong sample rate.** Nothing crashes. The frequency axis simply
     stretches by 48000/16000 = 3, so a 4 kHz resonance is reported at 12 kHz,
     the 3–6 kHz demodulation band lands on 1–2 kHz, and the envelope spectrum
     is searched at three times the true BPFO. You get a confident, wrong,
     beautifully plotted answer.
  2. **Wrong dtype scaling.** `FileSource` divides by 32767 unconditionally. An
     int32 wav then arrives with values around ±65000 instead of ±1, which does
     not change the dimensionless features (band ratios, kurtosis, crest) but
     shifts every log-RMS feature by ~11 nats — enough on its own to make a
     healthy machine look anomalous against a baseline learned in the other
     scaling.
  3. **Independent normalisation of the healthy and faulty captures.** If each
     file is separately scaled to full scale, the RMS *difference* between them
     — a real physical signal — is destroyed, and the two recordings are no
     longer comparable. This tool therefore applies **one common gain across
     every file in a single invocation** (see `plan_gain`), which is why you
     should always ingest the healthy/faulty pair together.

So: one place that converts, one place that checks, one place that records
what it did.

WHAT IT REFUSES TO DO QUIETLY
--------------------------------------------------------------------------
Conversion is easy; the value here is the audit. Every ingest prints, and
stores in the sidecar:

  * the resampling ratio as an exact integer fraction (`up/down`);
  * **whether the source has any bandwidth in the demodulation band at all.**
    This is the one that kills a week of work. `features.DEFAULT_BAND` is
    3000–6000 Hz. An 8 kHz phone recording has a 4 kHz Nyquist, so two thirds
    of that band is *empty*, and upsampling it to 16 kHz does not put the
    energy back. The detector goes quietly deaf and reports healthy for ever.
    We warn loudly and, with `--strict`, refuse.
  * the clipped-sample fraction. A clipped peak is a flat top, and a flat top
    is a broadband impulse — indistinguishable from the bearing impacts we are
    hunting. Clipping manufactures fake faults.
  * DC offset (removed from audio by default, KEPT on the accelerometer
    because there it is gravity and its magnitude is a free mounting check);
  * accelerometer timestamp jitter, which is how a dropped SPI FIFO read
    actually presents itself.

USAGE
--------------------------------------------------------------------------
    # the normal case: one healthy + one faulty capture, ingested TOGETHER so
    # they keep their relative level
    python tools/ingest.py raw/healthy.wav raw/faulty.wav \
        --out-dir data/real --rpm 2850 --bearing 6204

    # phone voice memo, mic-only, renamed on the way in
    python tools/ingest.py ~/memo.wav --out-dir data/real --stem healthy \
        --rpm 2850 --note "Bosch drill, no load, phone 30 cm from housing"

    # a CSV from a dataset or a logger: t_s + one column, or headerless + --fs-in
    python tools/ingest.py raw/de.csv --out-dir data/real --stem cwru_healthy

    # explicit accelerometer pairing
    python tools/ingest.py raw/faulty.wav --accel raw/imu.csv --out-dir data/real

    # prove the tool works, on signals it generates itself
    python tools/ingest.py --self-test

By default the tool **verifies its own output**: it re-loads what it wrote with
`recording_io.load_recording` and runs `firmware/features.extract_features` on
the first window. If the full feature vector does not come out (37 numbers as of
T1.5; the check reads the width from `FEATURE_NAMES` rather than hardcoding it),
the file is not canonical, whatever the tool claims. `--no-verify` skips it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "firmware"), str(_ROOT / "ml"), str(_ROOT / "ml" / "realdata")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recording_io import (RecordingError, accel_path_for,  # noqa: E402
                          load_recording, sidecar_path)

TOOL_VERSION = "1.0"

# Production rates. These match firmware/config.yaml (audio.sample_rate 16000,
# accelerometer.sample_rate 6400) and the constants in
# ml/realdata/validate_public_dataset.py. Changing them here without changing
# config.yaml would produce files the device cannot replay.
FS_AUDIO_DEVICE = 16000.0
FS_ACCEL_DEVICE = 6400.0

# firmware/features.DEFAULT_BAND, duplicated as a fallback so this tool still
# runs if firmware/ is not importable (e.g. a student ran it from a tarball
# without scipy). The real value is read from features.py when available.
_FALLBACK_BAND = (3000.0, 6000.0)


def demodulation_band() -> tuple[float, float]:
    """The band the detector actually demodulates in. Read from features.py so
    that re-tuning `DEFAULT_BAND` after the first real recording (see
    docs/DOC_STATUS.md — the protrugram never fired on surrogate data) also
    re-tunes this tool's bandwidth warning."""
    try:
        from features import DEFAULT_BAND
        return (float(DEFAULT_BAND[0]), float(DEFAULT_BAND[1]))
    except Exception:                                    # noqa: BLE001
        return _FALLBACK_BAND


class IngestError(Exception):
    """Anything that makes an input unusable. Printed as one line, never a
    traceback — this tool is used by tired students at a bench."""


def _err(msg: str) -> None:
    """Print to stderr, but flush stdout FIRST.

    stdout is block-buffered when piped and stderr is not, so a naive
    `print(file=sys.stderr)` surfaces the error *above* the report it belongs
    to as soon as anyone pipes the output into `tee` or `tail`. That is exactly
    when a student is trying to work out which file broke."""
    sys.stdout.flush()
    print(msg, file=sys.stderr, flush=True)


# ============================================================================
# 1. Reading — wav and csv, any rate, any dtype
# ============================================================================

def _sha256(path: Path, limit: int = 64 << 20) -> str:
    """Hash of the source file, for provenance. Capped at 64 MB because a
    2-hour soak wav is gigabytes and the first 64 MB already identifies it
    uniquely for our purposes (we record the cap in the sidecar so the number
    is never mistaken for a whole-file hash)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while read < limit:
            chunk = f.read(min(1 << 20, limit - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return f"sha256:{h.hexdigest()}" + ("" if read < limit else f" (first {limit} B)")


def read_wav_any(path: Path, channel: int | None = None):
    """Read a wav of any dtype/channel-count to float in [-1, 1].

    The dtype normalisation is the whole point and it is worth spelling out:
    wav stores int16 as −32768..32767, int32 as ±2^31, and **uint8 as 0..255
    with silence at 128** — the only unsigned case, and the one people forget.
    Dividing everything by 32767 (as FileSource does) is correct for exactly
    one of those three.

    Returns (x2d, fs, meta) with x2d shaped (n_samples, n_channels).
    """
    from scipy.io import wavfile
    try:
        fs, data = wavfile.read(path)
    except Exception as e:                               # noqa: BLE001
        raise IngestError(f"cannot read wav {path}: {e}") from e

    raw = np.asarray(data)
    if raw.ndim == 1:
        raw = raw[:, None]
    x = raw.astype(np.float64)

    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if raw.dtype == np.uint8:
            x = (x - 128.0) / 128.0
            scale_note = "uint8 (offset 128, /128)"
        else:
            x = x / float(-info.min)
            scale_note = f"{raw.dtype} (/{-info.min})"
    else:
        scale_note = f"{raw.dtype} (already ±1 by convention)"

    if len(x) == 0:
        raise IngestError(f"{path} contains no samples")

    meta = {"source_format": "wav", "source_dtype": str(raw.dtype),
            "source_channels": int(raw.shape[1]), "dtype_scaling": scale_note,
            # A wav has a KNOWN full scale, so "how many samples sit at ±1?" is
            # a meaningful clipping test. A CSV in g or volts does not — see
            # audit_audio.
            "full_scale_known": bool(np.issubdtype(raw.dtype, np.integer))}

    if channel is not None:
        if not 0 <= channel < x.shape[1]:
            raise IngestError(f"{path} has {x.shape[1]} channel(s); "
                              f"--channel {channel} is out of range")
        x = x[:, channel:channel + 1]
        meta["channel_selected"] = channel
    return x, float(fs), meta


def _looks_like_time_column(col: np.ndarray) -> bool:
    """Is this column a time axis, or is it signal?

    Test: strictly increasing, starting near zero, with a step that is stable
    for MOST samples. A vibration signal is not monotonic for more than a
    handful of samples, so false positives are not a real risk.

    "Most", not "all", and this matters: the first version of this function
    required *every* step to be within 50 % of the median, which rejected the
    exact file we most need to accept — an accelerometer log with a few dropped
    FIFO reads. Refusing to load it teaches the student nothing; loading it and
    reporting the jitter tells them their SPI drain is too slow. So we require
    95 % of steps to be regular and let `read_csv_any` report the rest.
    """
    if len(col) < 4 or not np.all(np.diff(col) > 0):
        return False
    if col[0] > 5.0:                                     # a time axis starts near 0
        return False
    dt = np.diff(col)
    med = float(np.median(dt))
    if med <= 0:
        return False
    return float(np.mean(np.abs(dt - med) < 0.5 * med)) >= 0.95


def read_csv_any(path: Path, fs_in: float | None = None,
                 channel: int | None = None):
    """Read a CSV signal, with or without a leading time column.

    Two shapes turn up in practice and we handle both:
      * `t_s, x[, y, z]` — our own canonical accel format, and what most
        loggers emit. The rate is inferred from median(diff(t)), exactly as
        FileSource does, so what we infer is what the device would infer.
      * bare columns of samples — dataset exports. Requires --fs-in; guessing
        a rate here would be the silent failure this whole tool exists to
        prevent.
    """
    # Header detection by INSPECTION, not by trial and error.
    #
    # The obvious implementation — try skiprows=1, fall back to skiprows=0 on
    # failure — is wrong, and wrong in the silent direction: on a headerless
    # numeric file `skiprows=1` succeeds and throws away the first sample. That
    # bug was in this function's first version and cost one row out of every
    # ingested CSV. So: read the first line and ask whether it is numbers.
    try:
        with open(path, "r") as f:
            first = f.readline()
    except Exception as e:                               # noqa: BLE001
        raise IngestError(f"cannot open CSV {path}: {e}") from e
    if not first.strip():
        raise IngestError(f"{path} is empty")
    try:
        [float(tok) for tok in first.replace("\t", ",").split(",") if tok.strip()]
        header_skipped = False
    except ValueError:
        header_skipped = True

    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1 if header_skipped else 0,
                          ndmin=2)
    except Exception as e:                               # noqa: BLE001
        raise IngestError(
            f"cannot parse CSV {path}: {e}. Expected comma-separated numbers, "
            f"optionally with one header row.") from e
    if data.size == 0:
        raise IngestError(f"{path} has no numeric rows")

    if data.shape[0] < 4:
        raise IngestError(f"{path} has only {data.shape[0]} row(s)")

    meta = {"source_format": "csv", "csv_header_row": header_skipped,
            "source_channels": int(data.shape[1])}

    t = None
    if data.shape[1] >= 2 and _looks_like_time_column(data[:, 0]):
        t = data[:, 0]
        sig = data[:, 1:]
        dt = np.diff(t)
        fs = 1.0 / float(np.median(dt))
        # Jitter is a hardware symptom, not a parse error: a dropped FIFO read
        # shows up here and nowhere else. Report it, do not refuse.
        jitter = float(np.max(np.abs(dt - np.median(dt))) / np.median(dt))
        meta.update({"time_column": True, "fs_from_time_column": fs,
                     "timestamp_jitter_frac": jitter,
                     "duration_from_time_column_s": float(t[-1] - t[0])})
        if fs_in and abs(fs_in - fs) / fs > 0.01:
            raise IngestError(
                f"{path}: --fs-in {fs_in:g} Hz disagrees with the file's own "
                f"time column ({fs:.3f} Hz). One of them is wrong; refusing to "
                f"guess. Drop --fs-in to trust the file.")
    else:
        sig = data
        meta["time_column"] = False
        if not fs_in:
            raise IngestError(
                f"{path} has no usable time column, so its sample rate is "
                f"unknown. Pass --fs-in HZ. (Guessing would silently stretch "
                f"every frequency in the analysis.)")
        fs = float(fs_in)

    if channel is not None:
        if not 0 <= channel < sig.shape[1]:
            raise IngestError(f"{path} has {sig.shape[1]} signal column(s); "
                              f"--channel {channel} is out of range")
        sig = sig[:, channel:channel + 1]
        meta["channel_selected"] = channel

    return np.asarray(sig, dtype=np.float64), fs, meta


def read_signal(path: Path, fs_in: float | None = None,
                channel: int | None = None):
    """Dispatch on extension. Returns (x2d, fs, meta)."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"no such file: {path}")
    suf = path.suffix.lower()
    if suf == ".wav":
        return read_wav_any(path, channel)
    if suf in (".csv", ".txt", ".tsv"):
        return read_csv_any(path, fs_in, channel)
    if suf == ".mat":
        raise IngestError(
            f"{path.name} is a Matlab file. Dataset .mat files (CWRU and "
            f"friends) are handled by ml/realdata/validate_public_dataset.py, "
            f"which knows their variable-naming conventions. Export to CSV or "
            f"wav first if you want them here.")
    raise IngestError(f"unsupported input type '{suf}' ({path.name}); "
                      f"this tool reads .wav and .csv")


def to_mono(x2d: np.ndarray) -> np.ndarray:
    """Collapse to one channel by averaging.

    Averaging, not picking channel 0: two mics on one machine both see the
    machine, and the average has ~sqrt(2) better SNR against uncorrelated
    self-noise. If the channels are genuinely different sensors, use --channel
    and say which one you mean."""
    return x2d.mean(axis=1) if x2d.ndim > 1 and x2d.shape[1] > 1 else x2d.reshape(-1)


# ============================================================================
# 2. Resampling
# ============================================================================

def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample with an anti-alias filter.

    Deliberately identical to `validate_public_dataset.resample_to` — the two
    are checked against each other in tests/test_ingest.py so they cannot
    drift apart, because a dataset resampled one way and a bench recording
    resampled another way are not comparable.

    `resample_poly`, not `scipy.signal.resample`: the latter is FFT-based and
    assumes periodicity within the window, so it smears the wrap-point
    discontinuity across the whole spectrum. When the entire measurement is
    "height of a narrow line above a broadband floor", raising the floor is
    precisely the wrong artefact.
    """
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    from scipy.signal import resample_poly
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64),
                         frac.numerator, frac.denominator).astype(np.float64)


def resample_ratio(fs_in: float, fs_out: float) -> tuple[int, int]:
    """The exact (up, down) integers used. Printed so the student can check
    them: 44100 → 16000 is 160/441, 48000 → 16000 is 1/3, 12000 → 6400 is
    8/15. If you ever see something like 397/1000 here, the rate is not what
    you think it is."""
    if abs(fs_in - fs_out) < 1e-9:
        return 1, 1
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return frac.numerator, frac.denominator


# ============================================================================
# 3. The audit — the part that earns this file its place in the repo
# ============================================================================

def band_energy_fraction(x: np.ndarray, fs: float,
                         band: tuple[float, float]) -> float:
    """Fraction of total signal power inside `band`. Computed on a decimated
    Welch-style average so a 30-minute file does not need one giant FFT."""
    n = int(min(len(x), 1 << 15))
    if n < 64:
        return float("nan")
    step = max(1, (len(x) - n) // 16 or 1)
    acc = None
    count = 0
    for start in range(0, max(len(x) - n, 0) + 1, step):
        seg = x[start:start + n]
        if len(seg) < n:
            break
        m = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(n))) ** 2
        acc = m if acc is None else acc + m
        count += 1
        if count >= 16:
            break
    if acc is None:
        return float("nan")
    f = np.fft.rfftfreq(n, 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    total = float(acc.sum())
    return float(acc[sel].sum() / total) if total > 0 else float("nan")


def clipped_fraction(x: np.ndarray, full_scale_known: bool) -> float:
    """Fraction of samples sitting on a flat top.

    TWO DIFFERENT TESTS, because "clipped" means two different things:

      * `full_scale_known` (an integer wav): full scale IS ±1 after dtype
        normalisation, so count samples at ±0.999. The 0.999 rather than 1.0
        allows for a resampler or a float32 export moving the flat top by a few
        ULP.
      * otherwise (a CSV in g, volts, or counts): ±1 means nothing. A file of
        accelerometer data peaking at 1.7 g is not clipped, and the first
        version of this function reported 20 % of the repo's own accel CSV as
        "clipped" for exactly that reason. So instead we look for the actual
        signature of clipping — MANY SAMPLES AT THE SAME EXTREME VALUE. A sine
        wave touches its peak on a vanishing fraction of samples; a clipped
        one sits there.
    """
    if len(x) == 0:
        return 0.0
    if full_scale_known:
        return float(np.mean(np.abs(x) >= 0.999))
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return 0.0
    return float(np.mean(np.abs(x) >= peak * (1.0 - 1e-9)))


def audit_audio(x: np.ndarray, fs_in: float, fs_out: float,
                band: tuple[float, float] | None = None,
                full_scale_known: bool = True) -> dict:
    """Everything that can be wrong with an audio channel, quantified.

    Each entry carries a severity: "info", "warn", or "blind". "blind" is
    reserved for the one failure that produces a working-looking system which
    can never detect anything — no bandwidth where the fault energy lives.
    """
    band = band or demodulation_band()
    nyq_in = fs_in / 2.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    dc = float(np.mean(x)) if len(x) else 0.0
    clipped = clipped_fraction(x, full_scale_known)

    findings: list[tuple[str, str]] = []

    if not full_scale_known and peak > 1.0:
        findings.append(("info", (
            f"source peaks at {peak:.3f}, i.e. it is not in ±1 units (g, volts "
            f"or counts). It will be scaled by {0.9 / peak:.4f} to fit int16, "
            f"so ABSOLUTE units are lost. That is expected: every feature is "
            f"either dimensionless or compared only against a baseline learned "
            f"in the same scaling.")))

    # --- bandwidth: the one that matters ---------------------------------
    band_lo, band_hi = band
    if nyq_in <= band_lo:
        findings.append(("blind", (
            f"source Nyquist is {nyq_in:g} Hz, entirely BELOW the demodulation "
            f"band {band_lo:g}-{band_hi:g} Hz. There is no bearing-resonance "
            f"energy in this file to find. Resampling up to {fs_out:g} Hz does "
            f"NOT create it. Re-record at >= {2 * band_hi:g} Hz.")))
    elif nyq_in < band_hi:
        covered = (nyq_in - band_lo) / (band_hi - band_lo)
        findings.append(("warn", (
            f"source Nyquist {nyq_in:g} Hz covers only {covered * 100:.0f} % of "
            f"the {band_lo:g}-{band_hi:g} Hz demodulation band; the rest is "
            f"empty and will read as a constant feature.")))

    if fs_out > fs_in:
        findings.append(("info", (
            f"upsampling {fs_in:g} -> {fs_out:g} Hz. This adds no information; "
            f"it only makes the file replayable by FileSource at the device "
            f"rate. Everything above {nyq_in:g} Hz stays empty.")))

    # --- clipping ---------------------------------------------------------
    if clipped > 0.001:
        findings.append(("warn", (
            f"{clipped * 100:.2f} % of samples are at full scale (clipped). A "
            f"clipped peak is a flat top, and a flat top is a broadband "
            f"impulse — i.e. it looks exactly like a bearing impact. Re-record "
            f"with lower gain; this file can manufacture a fault.")))
    elif clipped > 0:
        findings.append(("info", f"{clipped * 100:.3f} % of samples at full scale"))

    # --- level ------------------------------------------------------------
    if peak > 0 and peak < 0.02:
        findings.append(("warn", (
            f"peak level is {20 * np.log10(peak + 1e-12):.1f} dBFS — very quiet. "
            f"Quantisation noise is then a large fraction of the signal. Move "
            f"the mic closer or raise the input gain.")))
    if rms == 0:
        findings.append(("blind", "the channel is entirely silent (all zeros)."))

    # --- DC ---------------------------------------------------------------
    if abs(dc) > 0.01 * max(peak, 1e-9):
        findings.append(("info", (
            f"DC offset {dc:+.4f} ({abs(dc) / (peak + 1e-12) * 100:.1f} % of "
            f"peak) — removed. It eats headroom and inflates RMS features.")))

    if not np.all(np.isfinite(x)):
        findings.append(("blind", "channel contains NaN or Inf samples."))

    return {
        "fs_in": fs_in, "fs_out": fs_out, "n_samples_in": int(len(x)),
        "duration_s": len(x) / fs_in if fs_in else float("nan"),
        "peak": peak, "rms": rms, "dc_offset": dc, "clipped_frac": clipped,
        "peak_dbfs": 20 * float(np.log10(peak + 1e-12)),
        "demod_band": [band_lo, band_hi],
        "demod_band_energy_frac": band_energy_fraction(x, fs_in, band),
        "findings": findings,
    }


def audit_accel(a: np.ndarray, fs: float, jitter_frac: float | None) -> dict:
    """Accelerometer-specific checks.

    The useful one is gravity. A 3-axis sensor sitting on a machine reads a
    static vector of magnitude 1 g plus vibration. If the mean magnitude is not
    near 1, either the units are not g, or an axis is dead, or the part is not
    configured as you think — and firmware/bench/check_accel.py's whole
    gravity test is the same idea. We report it here so a dataset CSV gets the
    same sanity check a real sensor would.
    """
    findings: list[tuple[str, str]] = []
    means = a.mean(axis=0)
    grav = float(np.linalg.norm(means))
    if a.shape[1] == 3:
        if not (0.7 <= grav <= 1.4):
            findings.append(("warn", (
                f"mean acceleration vector magnitude is {grav:.3f}; a 3-axis "
                f"sensor at rest on a machine should read ~1.0 g. Either the "
                f"units are not g, or an axis is dead.")))
        else:
            findings.append(("info", f"gravity check: |mean| = {grav:.3f} g ✓"))
    if jitter_frac is not None and jitter_frac > 0.25:
        findings.append(("warn", (
            f"timestamp jitter is {jitter_frac * 100:.0f} % of the sample "
            f"period — the worst gap is more than a quarter of a sample out. "
            f"That is a dropped FIFO read, and FileSource infers the rate from "
            f"the MEDIAN step, so the frequency axis of the affected span is "
            f"wrong.")))
    for i in range(a.shape[1]):
        if float(np.std(a[:, i])) == 0.0:
            findings.append(("warn", f"axis {i} has zero variance — dead channel."))
    return {"n_axes": int(a.shape[1]), "fs_in": fs,
            "mean_per_axis": [float(v) for v in means],
            "gravity_magnitude": grav,
            "timestamp_jitter_frac": jitter_frac,
            "findings": findings}


def worst_severity(findings) -> str:
    order = {"info": 0, "warn": 1, "blind": 2}
    return max((s for s, _ in findings), key=lambda s: order[s], default="info")


# ============================================================================
# 4. Ingest
# ============================================================================

@dataclass
class Source:
    """One input recording, as specified on the command line."""
    audio_path: Path
    stem: str
    accel_path: Path | None = None

    # filled in by `prepare`
    audio: np.ndarray | None = None
    accel: np.ndarray | None = None
    audit: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _trim(x: np.ndarray, fs: float, start_s: float, duration_s: float | None):
    i0 = int(round(start_s * fs))
    i1 = len(x) if duration_s is None else i0 + int(round(duration_s * fs))
    out = x[i0:min(i1, len(x))]
    if len(out) == 0:
        raise IngestError(
            f"--start-s {start_s:g} / --duration-s {duration_s} selects nothing "
            f"from a {len(x) / fs:.2f} s signal")
    return out


def prepare(src: Source, args) -> Source:
    """Read, trim, DC-block and resample one source. No gain applied yet — the
    common gain can only be computed once every file in the batch is at its
    final rate, because resampling changes the peak (polyphase filters ring)."""
    x2d, fs_in, meta = read_signal(src.audio_path, args.fs_in, args.channel)
    x = to_mono(x2d)
    x = _trim(x, fs_in, args.start_s, args.duration_s)

    aud = audit_audio(x, fs_in, args.fs_audio,
                      full_scale_known=bool(meta.get("full_scale_known", False)))
    if not args.keep_dc:
        x = x - np.mean(x)

    up, down = resample_ratio(fs_in, args.fs_audio)
    src.audio = resample_to(x, fs_in, args.fs_audio)
    aud["resample_up"], aud["resample_down"] = up, down
    aud["n_samples_out"] = int(len(src.audio))
    src.audit["audio"] = aud
    src.meta.update(meta)

    # --- accelerometer, if there is one ----------------------------------
    ap = src.accel_path
    if ap is None and not args.mic_only:
        guess = accel_path_for(src.audio_path)
        if guess.exists():
            ap = guess
    if ap is not None:
        a2d, fs_a, ameta = read_csv_any(ap, args.fs_accel_in) if ap.suffix.lower() != ".wav" \
            else read_wav_any(ap)
        a2d = _trim(a2d, fs_a, args.start_s, args.duration_s)
        acc_audit = audit_accel(a2d, fs_a, ameta.get("timestamp_jitter_frac"))
        # Resample each axis independently. Gravity (the DC term) is preserved
        # by resample_poly, which is what we want: it is a real measurement.
        src.accel = np.column_stack([
            resample_to(a2d[:, i], fs_a, args.fs_accel) for i in range(a2d.shape[1])])
        up_a, down_a = resample_ratio(fs_a, args.fs_accel)
        acc_audit.update({"resample_up": up_a, "resample_down": down_a,
                          "fs_out": args.fs_accel,
                          "n_samples_out": int(len(src.accel)),
                          "source": str(ap)})
        src.audit["accel"] = acc_audit
        src.meta["accel_source_format"] = ameta.get("source_format")
    return src


def plan_gain(sources: list[Source], normalise: bool,
              independent: bool) -> dict[str, float]:
    """Decide the scale factor applied to each file's audio.

    THE DEFAULT IS ONE COMMON GAIN FOR THE WHOLE BATCH, and it is chosen to be
    1.0 unless something would clip. Reason: the RMS difference between a
    healthy and a faulty recording is data. `ml/simulate.py`'s `export_wav`
    normalises each file to 90 % full scale, which is right for making a demo
    wav audible and wrong for a healthy-vs-faulty comparison — scaling both
    files to the same peak removes exactly the level change you were trying to
    measure. So we preserve level and only intervene to prevent int16 wrapping.

    --normalise      : scale the batch so the LOUDEST file hits 90 % FS
                       (still one common factor: relative level survives).
    --independent-gain: normalise each file separately. Only correct when the
                       files are unrelated captures that will never be compared.
    """
    peaks = {s.stem: float(np.max(np.abs(s.audio))) for s in sources}
    if independent:
        return {k: (0.9 / (p + 1e-12) if (normalise or p > 0.999) else 1.0)
                for k, p in peaks.items()}
    top = max(peaks.values()) if peaks else 0.0
    if normalise or top > 0.999:
        g = 0.9 / (top + 1e-12)
    else:
        g = 1.0
    return {k: g for k in peaks}


def write_canonical(src: Source, out_dir: Path, gain: float, args) -> dict:
    """Write <stem>.wav, <stem>_accel.csv and <stem>.json.

    We write the wav here rather than calling recording_io.write_recording
    because that helper owns the *normalising* policy and we have already made
    a batch-level decision about gain. The byte layout is identical: mono
    int16, sample rate = args.fs_audio.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{src.stem}.wav"
    from scipy.io import wavfile

    x = src.audio * gain
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, int(round(args.fs_audio)), pcm)
    written = {"wav": wav_path}

    if src.accel is not None:
        csv_path = accel_path_for(wav_path)
        t = np.arange(len(src.accel)) / float(args.fs_accel)
        names = ["accel_x", "accel_y", "accel_z"][: src.accel.shape[1]]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s"] + names)
            for ti, row in zip(t, src.accel):
                # NINE decimal places, not six. This is not fussiness.
                # FileSource and recording_io both recover the sample rate as
                # 1/median(diff(t_s)), so the printed precision of this column
                # IS the sample rate. At 6400 Hz the step is 0.00015625 s;
                # written as "%.6f" it becomes 0.000156, which reads back as
                # 6410.26 Hz — a 0.16 % stretch of the accelerometer frequency
                # axis, applied silently, in a file this tool advertises as
                # canonical. (ml/simulate.py's export_accel_csv has the same
                # 6-dp habit, which is why the repo's own data/*_accel.csv
                # files all load at 6410.3 Hz; that file is frozen, and 0.16 %
                # does not change any published result, so it is recorded in
                # docs/DOC_STATUS.md rather than edited here.)
                w.writerow([f"{ti:.9f}"] + [f"{v:.6f}" for v in row])
        written["accel_csv"] = csv_path

    meta = {
        "stem": src.stem,
        "ingested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ingest_tool_version": TOOL_VERSION,
        "source_path": str(src.audio_path),
        "source_sha256": _sha256(src.audio_path),
        "fs_audio": args.fs_audio,
        "fs_accel_nominal": args.fs_accel,
        "gain_applied": gain,
        "dc_removed": not args.keep_dc,
        "mic_only": src.accel is None,
        "audit": _jsonable(src.audit),
        **{k: v for k, v in src.meta.items()},
    }
    for key in ("rpm", "bearing", "machine", "note", "label", "operator"):
        v = getattr(args, key, None)
        if v not in (None, ""):
            meta[key] = v
    if args.rpm:
        meta["fr_hz"] = float(args.rpm) / 60.0
    sc = sidecar_path(wav_path)
    sc.write_text(json.dumps(meta, indent=2, sort_keys=True))
    written["meta"] = sc
    return written


def _jsonable(obj):
    """Findings are tuples; numpy scalars are not JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def verify_written(wav_path: Path, mic_only: bool) -> dict:
    """Re-load what we just wrote and push it through the real feature
    extractor. This is the difference between "the tool says it wrote a
    canonical file" and "a canonical file exists": if extract_features cannot
    produce its full feature vector from this file, it is not canonical.

    The expected length is read from `FEATURE_NAMES` rather than hardcoded —
    it was a literal 40 until T1.5 removed three compositional redundancies and
    made it 37, and this check should track the contract, not a snapshot."""
    rec = load_recording(wav_path, require_accel=False)
    accel, fs_accel = rec.accel_for_features()
    from features import FEATURE_NAMES, extract_features
    win = min(len(rec.audio), int(5.0 * rec.fs_audio))
    nvi = max(int(win / rec.fs_audio * fs_accel), 8)
    out = extract_features(rec.audio[:win], rec.fs_audio, accel[:nvi], fs_accel)
    return {
        "ok": bool(len(out["vector"]) == len(FEATURE_NAMES)
                   and np.all(np.isfinite(out["vector"]))),
        "n_features": int(len(out["vector"])),
        # Reported back so callers can print "37/37" without importing
        # firmware/features themselves — this module is documented as still
        # runnable when firmware/ is not importable.
        "n_features_expected": len(FEATURE_NAMES),
        "band": [float(b) for b in out["band"]],
        "band_crest": float(out["band_crest"]),
        "fr_hz": float(out["fr_hz"]),
        "fr_reliable": bool(out["fr_reliable"]),
        "fs_audio": float(rec.fs_audio),
        "fs_accel": float(rec.fs_accel) if rec.has_accel else None,
        "mic_only": not rec.has_accel,
        "duration_s": float(rec.duration_s),
    }


# ============================================================================
# 5. Reporting
# ============================================================================

_SEV_TAG = {"info": "  .", "warn": " !!", "blind": "XXX"}


def print_source_report(src: Source, gain: float, written: dict,
                        ver: dict | None) -> None:
    p = print
    a = src.audit["audio"]
    p("-" * 74)
    p(f"{src.audio_path}")
    p(f"  -> {written['wav']}")
    p(f"  audio   : {a['n_samples_in']} samples @ {a['fs_in']:g} Hz "
      f"({a['duration_s']:.2f} s)  ->  {a['n_samples_out']} @ {a['fs_out']:g} Hz "
      f"[x{a['resample_up']}/{a['resample_down']}]")
    p(f"            peak {a['peak_dbfs']:.1f} dBFS, rms {a['rms']:.5f}, "
      f"clipped {a['clipped_frac'] * 100:.3f} %, gain applied x{gain:.4f}")
    bf = a["demod_band_energy_frac"]
    if np.isfinite(bf):
        p(f"            {a['demod_band'][0]:.0f}-{a['demod_band'][1]:.0f} Hz holds "
          f"{bf * 100:.1f} % of source power")
    if "accel" in src.audit:
        ac = src.audit["accel"]
        p(f"  accel   : {ac['n_axes']}-axis @ {ac['fs_in']:.1f} Hz -> "
          f"{ac['fs_out']:g} Hz [x{ac['resample_up']}/{ac['resample_down']}], "
          f"{ac['n_samples_out']} samples")
    else:
        p("  accel   : none (mic-only)")

    findings = list(a["findings"]) + list(src.audit.get("accel", {}).get("findings", []))
    if findings:
        p("")
        for sev, msg in findings:
            first = True
            for line in _wrap(msg, 66):
                p(f"  {_SEV_TAG[sev] if first else '   '} {line}")
                first = False
    if ver:
        p("")
        if ver["ok"]:
            p(f"  VERIFIED: reloaded and extract_features returned "
              f"{ver['n_features']}/{ver['n_features_expected']} finite features")
            p(f"            demod band {ver['band'][0]:.0f}-{ver['band'][1]:.0f} Hz "
              f"(crest {ver['band_crest']:.1f}), fr = {ver['fr_hz']:.2f} Hz "
              f"(reliable={ver['fr_reliable']})")
        else:
            p(f"  VERIFY FAILED: extract_features returned "
              f"{ver['n_features']} features / non-finite values")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ============================================================================
# 6. Self-test
# ============================================================================

def self_test(tmp_dir: Path) -> int:
    """Prove the conversion end to end on signals we generate here.

    The test signal is a 4 kHz carrier amplitude-modulated at 137 Hz — a
    deliberately crude stand-in for a bearing resonance rung by impacts at
    BPFO. It is written at 44.1 kHz (phone rate) with a DC offset, in stereo,
    as int16. If ingest works, then after conversion to 16 kHz:

      * the carrier is still at 4 kHz (not 4 * 16/44.1 = 1.45 kHz — the failure
        mode this whole tool exists to prevent);
      * the 137 Hz modulation still appears in the envelope spectrum;
      * extract_features returns 40 finite numbers from the written file.

    All three are asserted numerically, not eyeballed.
    """
    from scipy.io import wavfile

    tmp_dir.mkdir(parents=True, exist_ok=True)
    fs_src, dur = 44100.0, 6.0
    f_carrier, f_mod = 4000.0, 137.0
    t = np.arange(int(fs_src * dur)) / fs_src
    rng = np.random.default_rng(0)
    env = 1.0 + 0.8 * np.sign(np.sin(2 * np.pi * f_mod * t))
    x = 0.30 * env * np.sin(2 * np.pi * f_carrier * t) + 0.02 * rng.standard_normal(len(t))
    x = x + 0.05                                  # DC offset, as a cheap ADC gives
    stereo = np.column_stack([x, x * 0.98])       # stereo, as a phone gives
    src_wav = tmp_dir / "phone_44k1_stereo.wav"
    wavfile.write(src_wav, int(fs_src),
                  np.clip(stereo * 32767, -32768, 32767).astype(np.int16))

    print("=" * 74)
    print("INGEST SELF-TEST")
    print("=" * 74)
    print(f"synthetic source: {f_carrier:g} Hz carrier, AM at {f_mod:g} Hz, "
          f"{fs_src:g} Hz stereo int16, DC +0.05")
    print("(this exercises the tool; it is NOT evidence about real machines)\n")

    argv = [str(src_wav), "--out-dir", str(tmp_dir / "canonical"),
            "--stem", "selftest", "--mic-only", "--note", "ingest self-test"]
    rc = main(argv)
    if rc not in (0, 1):
        print("self-test FAILED: ingest returned", rc)
        return 2

    out_wav = tmp_dir / "canonical" / "selftest.wav"
    fs_out, pcm = wavfile.read(out_wav)
    y = pcm.astype(np.float64) / 32768.0

    checks: list[tuple[str, bool, str]] = []
    checks.append(("output rate is the device rate", fs_out == int(FS_AUDIO_DEVICE),
                   f"{fs_out} Hz"))

    mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1.0 / fs_out)
    peak_hz = float(freqs[int(np.argmax(mag))])
    err_pct = 100.0 * abs(peak_hz - f_carrier) / f_carrier
    checks.append(("carrier frequency survives resampling", err_pct < 0.5,
                   f"peak at {peak_hz:.1f} Hz vs {f_carrier:g} Hz "
                   f"({err_pct:.3f} % error)"))

    checks.append(("DC offset removed", abs(float(np.mean(y))) < 1e-3,
                   f"mean = {np.mean(y):+.2e} (source was +0.05)"))

    from features import envelope_spectrum
    ef, em = envelope_spectrum(y, float(fs_out), (3000.0, 6000.0))
    sel = (ef > 20) & (ef < 400)
    env_peak_hz = float(ef[sel][int(np.argmax(em[sel]))])
    env_err = 100.0 * abs(env_peak_hz - f_mod) / f_mod
    checks.append(("modulation survives in the envelope spectrum", env_err < 2.0,
                   f"envelope peak at {env_peak_hz:.2f} Hz vs {f_mod:g} Hz "
                   f"({env_err:.2f} % error)"))

    ver = verify_written(out_wav, mic_only=True)
    checks.append(("output is canonical (extract_features runs)", ver["ok"],
                   f"{ver['n_features']}/{ver['n_features_expected']} finite features"))

    print("\n--- self-test checks " + "-" * 52)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
    allok = all(c[1] for c in checks)
    print(f"\n  ==> SELF-TEST {'PASS' if allok else 'FAIL'}")
    print("=" * 74)
    return 0 if allok else 1


# ============================================================================
# 7. CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest.py",
        description="Convert arbitrary recordings (any sample rate, wav or "
                    "csv) into this project's canonical format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s raw/healthy.wav raw/faulty.wav --out-dir data/real "
            "--rpm 2850 --bearing 6204\n"
            "  %(prog)s ~/memo.m4a.wav --out-dir data/real --stem healthy "
            "--mic-only\n"
            "  %(prog)s raw/de.csv --fs-in 12000 --out-dir data/real "
            "--stem cwru_healthy\n"
            "  %(prog)s --self-test\n\n"
            "ingest the healthy and faulty captures in ONE invocation: they "
            "then share a single\ngain factor, so the RMS difference between "
            "them — which is data — survives.\n"),
    )
    p.add_argument("inputs", nargs="*", type=Path,
                   help="source recordings (.wav or .csv)")

    o = p.add_argument_group("output")
    o.add_argument("--out-dir", type=Path, default=Path("data/real"),
                   help="destination directory (default data/real)")
    o.add_argument("--stem", help="output name; only valid with ONE input "
                                  "(default: the input's own stem)")
    o.add_argument("--dry-run", action="store_true",
                   help="audit and report, write nothing")

    s = p.add_argument_group("source interpretation")
    s.add_argument("--fs-in", type=float,
                   help="sample rate of a headerless CSV input, Hz")
    s.add_argument("--fs-accel-in", type=float,
                   help="sample rate of a headerless accelerometer CSV, Hz")
    s.add_argument("--channel", type=int,
                   help="use this channel only (0-based); default averages all")
    s.add_argument("--accel", type=Path,
                   help="accelerometer source for a single input; default "
                        "looks for '<stem>_accel.csv' beside the audio")
    s.add_argument("--mic-only", action="store_true",
                   help="do not look for an accelerometer file "
                        "(supported Week-1 fallback)")
    s.add_argument("--start-s", type=float, default=0.0,
                   help="skip this many seconds (e.g. the run-up transient)")
    s.add_argument("--duration-s", type=float,
                   help="keep only this many seconds after --start-s")

    r = p.add_argument_group("rates")
    r.add_argument("--fs-audio", type=float, default=FS_AUDIO_DEVICE,
                   help=f"output audio rate (default {FS_AUDIO_DEVICE:g}, "
                        f"= firmware/config.yaml audio.sample_rate)")
    r.add_argument("--fs-accel", type=float, default=FS_ACCEL_DEVICE,
                   help=f"output accel rate (default {FS_ACCEL_DEVICE:g})")

    g = p.add_argument_group("level")
    g.add_argument("--normalise", action="store_true",
                   help="scale the batch so the loudest file peaks at 90 %% FS "
                        "(one common factor; relative level preserved)")
    g.add_argument("--independent-gain", action="store_true",
                   help="normalise each file separately. DESTROYS the level "
                        "relationship between healthy and faulty — only use on "
                        "unrelated captures")
    g.add_argument("--keep-dc", action="store_true",
                   help="do not remove the audio DC offset")

    m = p.add_argument_group("metadata for the sidecar (week 2 needs the rpm)")
    m.add_argument("--rpm", type=float, help="MEASURED shaft speed, rpm")
    m.add_argument("--bearing", help="designation, e.g. 6204")
    m.add_argument("--machine", help="what the machine is")
    m.add_argument("--label", choices=("healthy", "faulty", "unknown"),
                   help="ground truth, if known")
    m.add_argument("--operator", help="who recorded it")
    m.add_argument("--note", help="free text: mounting, load, room, anything "
                                  "you will not remember in three weeks")

    v = p.add_argument_group("checks")
    v.add_argument("--no-verify", action="store_true",
                   help="skip re-loading the output through extract_features")
    v.add_argument("--strict", action="store_true",
                   help="exit non-zero on any warning, not only on 'blind'")
    v.add_argument("--self-test", action="store_true",
                   help="generate a known signal, ingest it, and check the "
                        "result numerically; needs no input files")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        import tempfile
        return self_test(Path(tempfile.mkdtemp(prefix="ingest_selftest_")))

    if not args.inputs:
        _err("error: no inputs. Give one or more .wav/.csv files, or "
             "--self-test.")
        return 2
    if args.stem and len(args.inputs) > 1:
        _err("error: --stem names ONE output; you gave "
             f"{len(args.inputs)} inputs. Drop --stem and each file keeps its "
             "own name.")
        return 2
    if args.accel and len(args.inputs) > 1:
        _err("error: --accel applies to ONE input. For a batch, use the "
             "'<stem>_accel.csv' convention beside each audio file.")
        return 2
    if args.independent_gain and len(args.inputs) > 1:
        _err("note: --independent-gain scales each file separately, so the "
             "level relationship between these recordings is NOT preserved.")

    sources = [Source(audio_path=Path(p),
                      stem=args.stem or Path(p).stem,
                      accel_path=args.accel)
               for p in args.inputs]
    if len({s.stem for s in sources}) != len(sources):
        _err("error: two inputs would write to the same output name. Rename "
             "them or ingest separately.")
        return 2

    print("=" * 74)
    print(f"INGEST -> canonical format ({args.fs_audio:g} Hz audio / "
          f"{args.fs_accel:g} Hz accel)")
    print("=" * 74)

    try:
        for s in sources:
            prepare(s, args)
    except (IngestError, RecordingError) as e:
        _err(f"error: {e}")
        return 2
    except Exception as e:                               # noqa: BLE001
        _err(f"error: unexpected failure reading input: {e}")
        return 2

    gains = plan_gain(sources, args.normalise, args.independent_gain)
    if not args.independent_gain and len(sources) > 1:
        g = next(iter(gains.values()))
        print(f"one common gain x{g:.4f} across all {len(sources)} files "
              f"— relative level preserved.\n")

    severities: list[str] = []
    for s in sources:
        findings = list(s.audit["audio"]["findings"]) + \
            list(s.audit.get("accel", {}).get("findings", []))
        severities.append(worst_severity(findings))

        if args.dry_run:
            print("-" * 74)
            print(f"{s.audio_path}  [DRY RUN — nothing written]")
            a = s.audit["audio"]
            print(f"  would write {a['n_samples_out']} samples @ "
                  f"{a['fs_out']:g} Hz [x{a['resample_up']}/{a['resample_down']}]")
            for sev, msg in findings:
                for i, line in enumerate(_wrap(msg, 66)):
                    print(f"  {_SEV_TAG[sev] if i == 0 else '   '} {line}")
            continue

        try:
            written = write_canonical(s, args.out_dir, gains[s.stem], args)
        except Exception as e:                           # noqa: BLE001
            _err(f"error: could not write {s.stem}: {e}")
            return 2

        ver = None
        if not args.no_verify:
            try:
                ver = verify_written(written["wav"], s.accel is None)
            except Exception as e:                       # noqa: BLE001
                _err(f"warning: verification of {s.stem} failed: {e}")
                ver = {"ok": False, "n_features": 0, "n_features_expected": 0, "band": [0, 0],
                       "band_crest": float("nan"), "fr_hz": float("nan"),
                       "fr_reliable": False}
            if not ver["ok"]:
                severities.append("blind")
        print_source_report(s, gains[s.stem], written, ver)

    worst = max(severities, key=lambda x: {"info": 0, "warn": 1, "blind": 2}[x]) \
        if severities else "info"
    print("=" * 74)
    if worst == "blind":
        print("RESULT: at least one file CANNOT support detection (marked XXX "
              "above).")
        print("        Read the message; re-record rather than analysing this.")
        rc = 1
    elif worst == "warn":
        print("RESULT: written, with warnings (marked !! above). They are real "
              "— read them")
        print("        before you quote any number from these files.")
        rc = 1 if args.strict else 0
    else:
        print("RESULT: written cleanly.")
        rc = 0
    if not args.dry_run and not args.rpm:
        print("")
        print("NOTE: no --rpm recorded. ml/realdata/analyse_recording.py needs "
              "the shaft")
        print("      speed and will refuse to run without it. Measure it now, "
              "not later:")
        print("      the sidecar is where it belongs.")
    print("=" * 74)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
