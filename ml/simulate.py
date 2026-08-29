"""
simulate.py — Synthetic machine signal generator.

Physics rationale (read this before touching the code):

A rolling-element bearing defect does NOT produce a sinusoid at the fault
frequency (BPFO/BPFI). Each time a rolling element strikes the defect, it
produces a short broadband IMPULSE. That impulse excites the structural
resonances of the bearing housing — typically somewhere in the 1–20 kHz
region. What repeats at the fault frequency is the *envelope* of those
high-frequency resonance bursts, not a low-frequency tone.

Consequence for our product: a sensor chain band-limited to a few hundred Hz
(ADXL345-class accelerometer) or 4 kHz (8 kHz audio sampling) will miss early
faults entirely. Detection requires (a) enough bandwidth to capture the
resonance band, and (b) envelope (demodulation) analysis to recover the fault
periodicity. verify_signals.py demonstrates this concretely.

Fault frequencies for a bearing with N rolling elements, ball diameter d,
pitch diameter D, contact angle phi, shaft rotation frequency fr:

    BPFO = (N/2) * fr * (1 - (d/D) * cos(phi))   # outer race
    BPFI = (N/2) * fr * (1 + (d/D) * cos(phi))   # inner race

Rolling elements slip slightly (~1–2%), so real fault impulse trains are
quasi-periodic. We model that jitter; a perfectly periodic train would make
the detection problem artificially easy.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


# ----------------------------------------------------------------------------
# Bearing geometry
# ----------------------------------------------------------------------------

@dataclass
class BearingGeometry:
    """Defaults: SKF 6204 deep-groove ball bearing (very common small-motor
    bearing). In production we will NOT know these values for the customer's
    machine — which is exactly why v1 ships envelope-based anomaly detection,
    not named-fault classification. Geometry here is only for generating
    realistic test signals."""
    n_elements: int = 8
    ball_diameter_mm: float = 7.94
    pitch_diameter_mm: float = 33.5
    contact_angle_deg: float = 0.0

    def bpfo(self, fr: float) -> float:
        ratio = self.ball_diameter_mm / self.pitch_diameter_mm
        return (self.n_elements / 2.0) * fr * (1.0 - ratio * math.cos(math.radians(self.contact_angle_deg)))

    def bpfi(self, fr: float) -> float:
        ratio = self.ball_diameter_mm / self.pitch_diameter_mm
        return (self.n_elements / 2.0) * fr * (1.0 + ratio * math.cos(math.radians(self.contact_angle_deg)))


# ----------------------------------------------------------------------------
# Simulation config
# ----------------------------------------------------------------------------

@dataclass
class SimConfig:
    fs_audio: int = 16000          # Hz. 16k minimum so the 4–6 kHz resonance band is observable.
    fs_accel: int = 6400           # Hz. IIS3DWB-class bandwidth, NOT ADXL345-class (~800 Hz usable).
    duration_s: float = 10.0
    fr: float = 50.0               # shaft running frequency, Hz (50 Hz = 3000 RPM)
    resonance_hz: float = 4500.0   # bearing housing structural resonance excited by impacts
    resonance_q: float = 30.0      # resonance sharpness; sets the burst decay time
    snr_db: float = 20.0           # broadband noise floor relative to machine hum
    slip_jitter: float = 0.015     # 1.5% quasi-periodic slip on fault impulse spacing
    bearing: BearingGeometry = field(default_factory=BearingGeometry)
    seed: int | None = 42


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

def _time(cfg: SimConfig, fs: int) -> np.ndarray:
    return np.arange(int(cfg.duration_s * fs)) / fs


def _machine_hum(t: np.ndarray, fr: float, rng: np.random.Generator) -> np.ndarray:
    """Healthy rotating machine: dominant tone at fr plus weak harmonics.
    2x and 3x harmonics at -20 dB (amplitude factor 0.1) — every real machine
    has some residual imbalance and misalignment; zero harmonics would be
    unrealistically clean."""
    phase = rng.uniform(0, 2 * np.pi, size=3)
    sig = np.sin(2 * np.pi * fr * t + phase[0])
    sig += 0.1 * np.sin(2 * np.pi * 2 * fr * t + phase[1])
    sig += 0.1 * np.sin(2 * np.pi * 3 * fr * t + phase[2])
    return sig


def _noise_floor(n: int, fs: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Band-limited Gaussian noise: ambient acoustic + sensor self-noise.
    Low-passed at 0.45*fs to avoid energy piling up at Nyquist."""
    noise = rng.standard_normal(n)
    sos = butter(4, 0.45 * fs, btype="low", fs=fs, output="sos")
    noise = sosfilt(sos, noise)
    amplitude = 10 ** (-snr_db / 20.0)
    return amplitude * noise / np.std(noise)


def _impulse_train(t: np.ndarray, fs: int, fault_freq: float, jitter: float,
                   rng: np.random.Generator, modulate_at: float | None = None) -> np.ndarray:
    """Quasi-periodic unit impulses at fault_freq with multiplicative timing
    jitter (rolling-element slip). For inner-race faults the defect rotates
    through the load zone once per shaft rev, amplitude-modulating the impacts
    at fr — pass modulate_at=fr for that case. Outer-race defects are
    stationary relative to the load zone, so no modulation."""
    impulses = np.zeros_like(t)
    period = 1.0 / fault_freq
    t_next = rng.uniform(0, period)  # random phase start
    while t_next < t[-1]:
        idx = int(round(t_next * fs))
        if idx < len(impulses):
            amp = 1.0
            if modulate_at is not None:
                # load-zone modulation: impacts strongest once per shaft rev
                amp = 0.5 * (1.0 + np.cos(2 * np.pi * modulate_at * t_next))
            impulses[idx] = amp * rng.uniform(0.8, 1.2)  # impact-to-impact variation
        t_next += period * (1.0 + rng.normal(0, jitter))
    return impulses


def _resonance_filter(fs: int, f0: float, q: float):
    """2nd-order bandpass modelling the bearing-housing structural resonance.
    Convolving the impulse train through this turns each impact into a short
    decaying burst at f0 — the physically correct fault signature."""
    bw = f0 / q
    return butter(2, [f0 - bw / 2, f0 + bw / 2], btype="band", fs=fs, output="sos")


# ----------------------------------------------------------------------------
# Public signal generators
# ----------------------------------------------------------------------------

def normal_signal(cfg: SimConfig, fs: int, rng: np.random.Generator) -> np.ndarray:
    t = _time(cfg, fs)
    return _machine_hum(t, cfg.fr, rng) + _noise_floor(len(t), fs, cfg.snr_db, rng)


def bearing_fault_signal(cfg: SimConfig, fs: int, rng: np.random.Generator,
                         severity: float = 0.5, race: str = "outer") -> np.ndarray:
    """Normal signal + resonance bursts repeating at BPFO/BPFI.
    severity in [0, 1] scales burst energy relative to the machine hum.
    At severity ~0.1 the fault is inaudible and invisible in the raw
    low-frequency spectrum — exactly the early-stage case we must catch."""
    t = _time(cfg, fs)
    base = normal_signal(cfg, fs, rng)
    if race == "outer":
        f_fault, mod = cfg.bearing.bpfo(cfg.fr), None
    elif race == "inner":
        f_fault, mod = cfg.bearing.bpfi(cfg.fr), cfg.fr
    else:
        raise ValueError("race must be 'outer' or 'inner'")
    train = _impulse_train(t, fs, f_fault, cfg.slip_jitter, rng, modulate_at=mod)
    sos = _resonance_filter(fs, min(cfg.resonance_hz, 0.4 * fs), cfg.resonance_q)
    bursts = sosfilt(sos, train)
    bursts = bursts / (np.std(bursts) + 1e-12)  # unit-energy, then scale by severity
    return base + severity * bursts


def imbalance_signal(cfg: SimConfig, fs: int, rng: np.random.Generator,
                     severity_end: float = 1.0) -> np.ndarray:
    """Imbalance grows the 1x fr component linearly over the record —
    models a fan shedding a blade deposit or a coupling working loose."""
    t = _time(cfg, fs)
    base = normal_signal(cfg, fs, rng)
    growth = np.linspace(0.0, severity_end, len(t))
    return base + growth * np.sin(2 * np.pi * cfg.fr * t)


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

def export_wav(path: str, signal: np.ndarray, fs: int) -> None:
    peak = np.max(np.abs(signal)) + 1e-12
    wavfile.write(path, fs, (0.9 * signal / peak * 32767).astype(np.int16))


def export_accel_csv(path: str, signal: np.ndarray, fs: int) -> None:
    """Single-axis CSV (timestamp, accel_g). Firmware ingests this in
    --simulate mode in place of the SPI driver."""
    t = np.arange(len(signal)) / fs
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "accel_g"])
        for ti, xi in zip(t, signal):
            w.writerow([f"{ti:.6f}", f"{xi:.6f}"])


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic machine signals")
    p.add_argument("--outdir", default="output")
    p.add_argument("--severity", type=float, default=0.15,
                   help="bearing fault severity (default 0.15 = early-stage)")
    p.add_argument("--duration", type=float, default=10.0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = SimConfig(duration_s=args.duration)
    rng = np.random.default_rng(cfg.seed)

    print(f"fr = {cfg.fr} Hz | BPFO = {cfg.bearing.bpfo(cfg.fr):.1f} Hz | "
          f"BPFI = {cfg.bearing.bpfi(cfg.fr):.1f} Hz | resonance = {cfg.resonance_hz} Hz")

    cases = {
        "normal": normal_signal(cfg, cfg.fs_audio, rng),
        "bearing_outer": bearing_fault_signal(cfg, cfg.fs_audio, rng, severity=args.severity, race="outer"),
        "bearing_inner": bearing_fault_signal(cfg, cfg.fs_audio, rng, severity=args.severity, race="inner"),
        "imbalance": imbalance_signal(cfg, cfg.fs_audio, rng),
    }
    for name, sig in cases.items():
        export_wav(os.path.join(args.outdir, f"{name}.wav"), sig, cfg.fs_audio)
        print(f"wrote {name}.wav  ({len(sig)/cfg.fs_audio:.1f}s @ {cfg.fs_audio} Hz)")

    rng2 = np.random.default_rng(cfg.seed)
    accel = bearing_fault_signal(cfg, cfg.fs_accel, rng2, severity=args.severity, race="outer")
    export_accel_csv(os.path.join(args.outdir, "bearing_outer_accel.csv"), accel, cfg.fs_accel)
    print("wrote bearing_outer_accel.csv")


if __name__ == "__main__":
    main()
