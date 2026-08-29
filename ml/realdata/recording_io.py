"""
recording_io.py — read/write the project's canonical recording format.

WHAT THE CANONICAL FORMAT IS
----------------------------
It is defined by two pieces of code that already exist and must not change:

  * ml/simulate.py  `export_wav` / `export_accel_csv`  — the writer
  * firmware/capture.py `FileSource`                   — the reader

A "recording" is therefore:

    <stem>.wav        mono 16-bit PCM at the audio sample rate.
                      FileSource divides by 32767, so the working range is
                      [-1, 1] and any scale information is LOST. Do not expect
                      to recover pascals or g from the wav; this pipeline is
                      about spectral SHAPE and relative change, not absolute
                      calibrated level.

    <stem>_accel.csv  header row, then rows of
                          t_s, accel_x[, accel_y, accel_z]
                      FileSource infers the accelerometer sample rate from
                      median(diff(t_s)) rather than trusting a header — so the
                      time column must be real and monotonic.

    <stem>.json       OPTIONAL sidecar this module adds (FileSource ignores
                      unknown files). Carries the things a wav cannot: shaft
                      rpm, bearing designation, what the machine was doing,
                      and the provenance of any resampling. Week 2 lives or
                      dies on knowing the rpm, and "I think it was about
                      2800" is not a measurement.

MIC-ONLY IS A FIRST-CLASS CASE
------------------------------
The execution plan's Week-1 fallback is explicit: if the SPI accelerometer
fights you, run mic-only and move on. So every loader here accepts a missing
CSV and returns accel=None. Callers must handle that; nothing in this package
requires an accelerometer to produce the Week-2 evidence, because the
microphone alone carries the housing-resonance band where the fault energy is.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile


class RecordingError(Exception):
    """Anything wrong with a recording on disk. Callers catch this and print a
    one-line message instead of dumping a traceback at a tired student."""


@dataclass
class Recording:
    """One loaded recording. `accel` is None for mic-only captures."""

    audio: np.ndarray                 # (n,) float, nominally [-1, 1]
    fs_audio: float
    accel: np.ndarray | None = None   # (m, k) float, k in 1..3
    fs_accel: float | None = None
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.fs_audio

    @property
    def has_accel(self) -> bool:
        return self.accel is not None and len(self.accel) > 0

    def accel_for_features(self) -> tuple[np.ndarray, float]:
        """What firmware/features.extract_features wants for its accel input.

        Mic-only case: extract_features REQUIRES an accel array (it computes 12
        accel statistics and 8 accel band ratios from it). We hand it a
        zero array at the configured rate. That is not a fudge that invents
        evidence — channel_stats has an explicit dead-channel guard that maps a
        flat input to the fixed sentinel [-9, 0, 0, 0], and band_energy_ratios
        on zeros gives a constant vector. So those 20 dimensions become
        CONSTANTS, contributing nothing to the Mahalanobis distance in either
        direction (Ledoit-Wolf shrinkage keeps the covariance invertible
        despite the zero-variance columns).

        The honest summary: mic-only runs a 20-dimensional detector wearing a
        40-dimensional coat. It is the documented Week-1 fallback, not a
        silent degradation — callers should say so in their output.
        """
        if self.has_accel:
            return self.accel, float(self.fs_accel)
        fs = float(self.meta.get("fs_accel_nominal", 6400.0))
        n = max(int(round(self.duration_s * fs)), 8)
        return np.zeros((n, 3)), fs

    def describe(self) -> str:
        bits = [f"{self.duration_s:.2f} s audio @ {self.fs_audio:g} Hz"]
        if self.has_accel:
            bits.append(f"{self.accel.shape[1]}-axis accel @ {self.fs_accel:g} Hz")
        else:
            bits.append("NO accelerometer (mic-only)")
        if self.meta.get("rpm"):
            bits.append(f"{self.meta['rpm']:g} rpm")
        return " | ".join(bits)


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------

def _read_wav(path: Path) -> tuple[np.ndarray, float]:
    """Read a wav to float in roughly [-1, 1], whatever the source dtype.

    scipy hands back int16 / int32 / uint8 / float32 depending on the file.
    Phone voice-memo exports and `arecord` defaults differ, so normalise by the
    dtype's full scale rather than assuming int16 — dividing an int32 file by
    32767 gives numbers around 65000 and silently destroys every feature."""
    try:
        fs, data = wavfile.read(path)
    except Exception as e:
        raise RecordingError(f"cannot read wav {path}: {e}") from e

    x = np.asarray(data)
    if x.ndim > 1:
        # Multi-channel: average to mono. Two mics on one machine are still
        # one machine; the detector expects a single audio stream.
        x = x.mean(axis=1)
    x = x.astype(np.float64)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if data.dtype == np.uint8:          # wav 8-bit is UNSIGNED, offset 128
            x = (x - 128.0) / 128.0
        else:
            x = x / float(-info.min)        # int16 -> /32768, int32 -> /2**31
    # float wavs are already in [-1, 1] by convention; leave them alone.

    if len(x) == 0:
        raise RecordingError(f"{path} contains no samples")
    return x, float(fs)


def _read_accel_csv(path: Path) -> tuple[np.ndarray, float]:
    """Read a (t_s, x[, y, z]) CSV the way FileSource does, but with errors a
    human can act on and a sample-rate sanity check FileSource does not do."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception as e:
        raise RecordingError(
            f"cannot parse accel CSV {path}: {e}. Expected a header row then "
            f"comma-separated 't_s,accel_x[,accel_y,accel_z]'."
        ) from e

    if data.ndim == 1:                      # single row, or single column
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RecordingError(
            f"{path} has {data.shape[1]} column(s); need at least 2 "
            f"(t_s and one acceleration axis)."
        )
    if len(data) < 4:
        raise RecordingError(f"{path} has only {len(data)} sample rows.")

    t = data[:, 0]
    cols = data[:, 1:4] if data.shape[1] >= 4 else data[:, 1:2]

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise RecordingError(
            f"{path}: the t_s column is not strictly increasing. FileSource "
            f"infers the sample rate from median(diff(t_s)), so a scrambled or "
            f"repeated timestamp column silently corrupts the rate."
        )
    fs = 1.0 / float(np.median(dt))
    # A jittery clock is a real hardware symptom (dropped FIFO reads), not a
    # parse error — warn through the return value rather than refusing to load.
    return np.asarray(cols, dtype=np.float64), fs


def sidecar_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(".json")


def accel_path_for(wav_path: Path) -> Path:
    """The CSV that ml/simulate.py's naming convention pairs with a wav."""
    return wav_path.with_name(wav_path.stem + "_accel.csv")


def load_recording(wav_path: str | Path, accel_csv: str | Path | None = None,
                   require_accel: bool = False) -> Recording:
    """Load one recording.

    `accel_csv` defaults to the '<stem>_accel.csv' convention; pass an explicit
    path to override, or set it to False-y and require_accel=False for
    mic-only. Raises RecordingError (never a bare traceback) on any problem.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise RecordingError(f"no such recording: {wav_path}")

    audio, fs_audio = _read_wav(wav_path)

    meta: dict = {}
    sc = sidecar_path(wav_path)
    if sc.exists():
        try:
            meta = json.loads(sc.read_text())
        except Exception as e:
            raise RecordingError(f"sidecar {sc} is not valid JSON: {e}") from e

    accel, fs_accel = None, None
    candidate = Path(accel_csv) if accel_csv else accel_path_for(wav_path)
    if candidate.exists():
        accel, fs_accel = _read_accel_csv(candidate)
    elif accel_csv:
        raise RecordingError(f"accel CSV not found: {candidate}")
    elif require_accel:
        raise RecordingError(
            f"no accelerometer CSV alongside {wav_path.name} (looked for "
            f"{candidate.name}). Pass --mic-only if this is a microphone-only "
            f"capture — that is a supported Week-1 fallback."
        )

    return Recording(audio=audio, fs_audio=fs_audio, accel=accel,
                     fs_accel=fs_accel, meta=meta, path=wav_path)


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------

def write_recording(out_stem: str | Path, audio: np.ndarray, fs_audio: float,
                    accel: np.ndarray | None = None,
                    fs_accel: float | None = None,
                    meta: dict | None = None,
                    normalise: bool = True) -> dict:
    """Write a recording in the canonical format. Returns the paths written.

    `normalise` scales the audio to 90 % of int16 full scale, exactly as
    ml/simulate.py's export_wav does. That destroys absolute level — which is
    already lost downstream, because FileSource divides by a constant — but it
    prevents clipping. Set normalise=False to preserve relative level BETWEEN
    two recordings, which matters if you want the healthy and faulty captures
    to be comparable in RMS. tools/ingest.py exposes this as --no-normalise and
    defaults to preserving level for exactly that reason.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_stem.with_suffix(".wav")

    x = np.asarray(audio, dtype=np.float64)
    if normalise:
        peak = float(np.max(np.abs(x))) + 1e-12
        x = 0.9 * x / peak
    else:
        # Still guard against clipping: int16 wraps around on overflow, which
        # turns a loud transient into a full-scale square wave and a fake
        # broadband impulse — i.e. a fake bearing fault.
        peak = float(np.max(np.abs(x)))
        if peak > 1.0:
            x = x / peak * 0.999
    wavfile.write(wav_path, int(round(fs_audio)),
                  np.clip(x * 32767.0, -32768, 32767).astype(np.int16))

    written = {"wav": wav_path}

    if accel is not None and fs_accel:
        a = np.asarray(accel, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        csv_path = accel_path_for(wav_path)
        t = np.arange(len(a)) / float(fs_accel)
        names = ["accel_x", "accel_y", "accel_z"][: a.shape[1]]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s"] + names)
            for ti, row in zip(t, a):
                w.writerow([f"{ti:.6f}"] + [f"{v:.6f}" for v in row])
        written["accel_csv"] = csv_path

    if meta:
        sc = sidecar_path(wav_path)
        sc.write_text(json.dumps(meta, indent=2, sort_keys=True))
        written["meta"] = sc

    return written


# ----------------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------------

def iter_windows(rec: Recording, window_s: float, hop_s: float | None = None):
    """Yield (audio_window, accel_window, fs_accel) tuples.

    Mirrors FileSource.windows() but adds an optional hop for OVERLAPPING
    windows. Overlap is a genuine trade-off and the caller must own it:

      * more windows from a short record (CWRU files are only ~10-20 s, far
        too short for the 30 s production window), which the baseline fitter
        needs — it wants tens of windows to estimate a 40x40 covariance;
      * but overlapping windows are NOT statistically independent. Any ROC AUC
        or false-positive rate computed across them is optimistic, because the
        same samples appear in several "observations".

    Default hop = window (no overlap). Scripts that turn on overlap should say
    so in their output.
    """
    hop_s = window_s if hop_s is None else hop_s
    na, ha = int(window_s * rec.fs_audio), int(hop_s * rec.fs_audio)
    if na <= 0 or ha <= 0:
        raise RecordingError("window_s and hop_s must be positive")
    if len(rec.audio) < na:
        raise RecordingError(
            f"recording is {rec.duration_s:.2f} s but the window is "
            f"{window_s:g} s — nothing to analyse. Use a shorter --window-s."
        )

    if rec.has_accel:
        accel, fs_v = rec.accel, float(rec.fs_accel)
    else:
        accel, fs_v = rec.accel_for_features()
    nv, hv = int(window_s * fs_v), int(hop_s * fs_v)

    n = 1 + (len(rec.audio) - na) // ha
    if nv > 0:
        n = min(n, 1 + max(0, (len(accel) - nv)) // max(hv, 1))
    for i in range(max(n, 0)):
        a = rec.audio[i * ha: i * ha + na]
        v = accel[i * hv: i * hv + nv] if nv > 0 else accel
        if len(a) < na:
            break
        yield a, v, fs_v
