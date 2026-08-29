"""
verify_signals.py — Layer 1 exit criterion.

Produces two figures that settle the key design argument empirically:

  fig1_spectrograms.png   — normal vs early-stage bearing fault, full band.
                            The fault energy lives in the 4–5 kHz resonance
                            band. A pipeline that low-passes at 800 Hz
                            (ADXL345) or 4 kHz (8 kHz audio sampling) is
                            blind to it.

  fig2_envelope.png       — raw spectrum vs envelope spectrum, zoomed to
                            0–400 Hz. The raw spectrum shows NO peak at BPFO
                            for an early fault. The envelope spectrum
                            (bandpass around the resonance -> Hilbert
                            magnitude -> FFT) shows BPFO and its harmonics
                            unambiguously. This is why features.py must
                            implement envelope analysis.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfilt, stft

from simulate import SimConfig, bearing_fault_signal, normal_signal

OUT = "output"
SEVERITY = 0.15  # early-stage: inaudible over the hum, the case that matters


def envelope_spectrum(x: np.ndarray, fs: int, band=(3500.0, 5500.0)):
    """Classic envelope analysis: isolate the resonance band, demodulate via
    the analytic-signal magnitude, remove DC, FFT. Fault periodicity appears
    as a peak at BPFO/BPFI in the result."""
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfilt(sos, x)))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), 1.0 / fs)
    return freqs, spec / (spec.max() + 1e-12)


def raw_spectrum(x: np.ndarray, fs: int):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, spec / (spec.max() + 1e-12)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = SimConfig(duration_s=10.0)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)

    healthy = normal_signal(cfg, cfg.fs_audio, rng_a)
    faulty = bearing_fault_signal(cfg, cfg.fs_audio, rng_b, severity=SEVERITY, race="outer")
    bpfo = cfg.bearing.bpfo(cfg.fr)

    # ---- Figure 1: spectrograms --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, sig, title in [(axes[0], healthy, "Healthy"),
                           (axes[1], faulty, f"Bearing fault (severity {SEVERITY})")]:
        f, t, Z = stft(sig, fs=cfg.fs_audio, window="hann", nperseg=1024, noverlap=512)
        ax.pcolormesh(t, f, 20 * np.log10(np.abs(Z) + 1e-9), shading="gouraud",
                      vmin=-100, vmax=-20, cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[1].axhline(800, color="cyan", ls="--", lw=1)
    axes[1].axhline(4000, color="lime", ls="--", lw=1)
    axes[1].text(0.1, 900, "ADXL345 usable limit (~800 Hz)", color="cyan", fontsize=8)
    axes[1].text(0.1, 4150, "Nyquist @ 8 kHz sampling (4 kHz)", color="lime", fontsize=8)
    fig.suptitle("Fault energy sits in the 4–5 kHz resonance band — above both proposed sensor limits")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_spectrograms.png"), dpi=130)
    print("wrote fig1_spectrograms.png")

    # ---- Figure 2: raw vs envelope spectrum --------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fr_raw, sp_raw = raw_spectrum(faulty, cfg.fs_audio)
    fr_env, sp_env = envelope_spectrum(faulty, cfg.fs_audio)

    for ax, (fx, sx), title in [
        (axes[0], (fr_raw, sp_raw), "Raw spectrum of faulty signal (what the original pipeline inspects)"),
        (axes[1], (fr_env, sp_env), "Envelope spectrum of faulty signal (bandpass 3.5–5.5 kHz → Hilbert → FFT)"),
    ]:
        m = fx <= 400
        ax.plot(fx[m], sx[m], lw=0.9)
        ax.set_title(title)
        ax.set_ylabel("Normalised magnitude")
        for k in range(1, 3):
            ax.axvline(k * bpfo, color="red", ls="--", lw=1, alpha=0.7)
        ax.axvline(cfg.fr, color="gray", ls=":", lw=1)
        ax.text(cfg.fr + 2, 0.9, "fr", color="gray", fontsize=8)
        ax.text(bpfo + 2, 0.9, f"BPFO = {bpfo:.0f} Hz", color="red", fontsize=8)
    axes[1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Early fault: absent at BPFO in the raw spectrum, unmistakable in the envelope spectrum")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_envelope.png"), dpi=130)
    print("wrote fig2_envelope.png")

    # ---- Quantitative check -------------------------------------------------
    # NOTE: bw must be tighter than the gap between BPFO (152.6 Hz) and the
    # machine's 3rd harmonic (150 Hz) — 2.6 Hz apart. With a 10 s record the
    # spectral resolution is 0.1 Hz, so a ±1 Hz window resolves them. This
    # near-collision is itself a lesson: naive peak-picking in the raw
    # spectrum confuses shaft harmonics with fault frequencies.
    def peak_ratio(freqs, spec, target, bw=1.0):
        in_band = spec[(freqs > target - bw) & (freqs < target + bw)].max()
        background = np.median(spec[(freqs > 20) & (freqs < 400)])
        return in_band / (background + 1e-12)

    r_raw = peak_ratio(fr_raw, sp_raw, bpfo)
    r_env = peak_ratio(fr_env, sp_env, bpfo)
    print(f"BPFO peak-to-background ratio — raw: {r_raw:.1f}x | envelope: {r_env:.1f}x")
    if r_env > 5 * r_raw and r_env > 10:
        print("PASS: envelope analysis recovers the fault signature; raw spectrum does not.")
    else:
        print("CHECK: separation weaker than expected — inspect figures.")


if __name__ == "__main__":
    main()
