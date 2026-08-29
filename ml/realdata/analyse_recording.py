"""
analyse_recording.py — the Week-2 workhorse.

WHAT THIS ANSWERS
-----------------
the execution plan (not in this public copy), Gate 2, the project's go/no-go:

    "the envelope-spectrum peak at the computed BPFO is clearly above the
     healthy background, and the Mahalanobis score on faulty windows
     separates from healthy ones"

ml/verify_signals.py already answers the first half ON SYNTHETIC DATA, and got
a decisive answer: at the computed BPFO the peak-to-background ratio was 2.2x
in the raw spectrum and 56.7x in the envelope spectrum. That 25-fold advantage
is the single most load-bearing measurement in the repo — it is why the sensor
is a 16 kHz microphone and not an 800 Hz accelerometer, and why features.py
demodulates at all.

This script runs THE SAME MEASUREMENT on a real healthy/faulty recording pair.
If the ratio collapses on real data, the product hypothesis is in trouble and
we want to know in Week 2, not Week 12.

WHAT IS DIFFERENT ABOUT REAL DATA (and what this script does about it)
---------------------------------------------------------------------
1. SLIP. Rolling elements slide by ~1-2 %, so the observed line sits slightly
   below the computed BPFO and wanders. verify_signals.py could use a +/-1 Hz
   bin because it knew the synthetic frequency exactly. Here we search a
   +/-tol_pct window and REPORT WHERE THE PEAK ACTUALLY LANDED, so you can see
   the slip rather than assume it. A peak at 2 % below prediction is a physical
   result; a peak 20 % away means the geometry, the rpm, or the wiring is wrong.

2. THE RECORDING IS LONG. Two hours is not one FFT. We cut the record into
   windows, compute an envelope spectrum per window, and AVERAGE the
   magnitudes. This is Welch's method applied to the envelope: random noise
   averages down as 1/sqrt(n_windows) while a genuinely periodic fault line
   stays put and stands up. A single 30 s FFT of a noisy shop-floor recording
   can easily hide a line that 200 averaged windows make obvious.

3. THE BAND MUST BE THE SAME FOR BOTH RECORDINGS. features.py picks the
   demodulation band per window by protrugram. For a FAIR healthy-vs-faulty
   comparison we pick the band ONCE, on the faulty recording (that is where a
   resonance is being excited), and apply the identical band to the healthy
   one. Letting healthy pick its own band would compare two different
   measurements and quietly flatter the result.

4. NO ACCELEROMETER IS FINE. --mic-only is supported throughout; the
   microphone carries the housing-resonance band, which is where the evidence
   is. You lose the audio/accel cross-check on shaft speed, nothing else.

THE THREE-PART VERDICT
----------------------
A single "the ratio was big" number is not evidence — big compared to what?
So the gate has three independent conditions, and all must hold:

  (A) ABSOLUTE   env ratio at BPFO on the FAULTY recording >= --min-ratio.
                 There is a line at all.
  (B) CONTRAST   faulty ratio / healthy ratio >= --min-contrast.
                 The line is a property of the FAULT, not of the machine, the
                 room, or the mounting. This is the control condition and it
                 is the one most student projects skip.
  (C) METHOD     faulty env ratio > faulty raw ratio.
                 Demodulation is doing the work — i.e. we have reproduced the
                 project's central claim on real data.

Usage
-----
    python ml/realdata/analyse_recording.py \
        --healthy data/real/healthy.wav --faulty data/real/faulty.wav \
        --bearing 6204 --rpm 2850 --out output/week2_real.png

    # mic-only phone recording, geometry you had to guess:
    python ml/realdata/analyse_recording.py \
        --healthy h.wav --faulty f.wav --mic-only \
        --bore-mm 20 --od-mm 47 --n 8 --rpm 2850

    # self-test on synthetic data (no recordings needed):
    python ml/realdata/analyse_recording.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "firmware"), str(_ROOT / "ml"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fault_frequencies import (BearingGeometry, from_boundary_dimensions,  # noqa: E402
                               lookup, rpm_to_hz)
from recording_io import Recording, RecordingError, iter_windows, load_recording  # noqa: E402

# The synthetic reference this script exists to reproduce. From
# ml/verify_signals.py on a severity-0.15 outer-race fault.
SYNTHETIC_REFERENCE = {"raw": 2.2, "env": 56.7}


# ----------------------------------------------------------------------------
# Spectra
# ----------------------------------------------------------------------------

def _window_rfft(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """|FFT| of one Hann-windowed block. Hann because we are measuring the
    height of narrow lines against a broadband floor: a rectangular window's
    -13 dB sidelobes would smear the shaft harmonics across the very region
    we use as 'background'."""
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft((x - x.mean()) * w))
    return np.fft.rfftfreq(len(x), 1.0 / fs), mag


def averaged_raw_spectrum(rec: Recording, window_s: float,
                          max_windows: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Welch-style averaged magnitude spectrum of the raw audio."""
    freqs, acc, n = None, None, 0
    for audio, _, _ in iter_windows(rec, window_s):
        f, m = _window_rfft(audio, rec.fs_audio)
        acc = m if acc is None else acc + m
        freqs = f
        n += 1
        if n >= max_windows:
            break
    if n == 0:
        raise RecordingError("no complete windows in recording")
    return freqs, acc / n


def averaged_envelope_spectrum(rec: Recording, band: tuple[float, float],
                               window_s: float, max_windows: int = 200,
                               method: str = "rectify") -> tuple[np.ndarray, np.ndarray]:
    """Averaged ENVELOPE spectrum over windows, in the given demodulation band.

    method:
      "rectify" — bandpass, rectify, low-pass, decimate. This is EXACTLY
                  firmware/features.envelope_spectrum, i.e. the code that
                  actually ships to the Pi. Validating this path is the point:
                  a result obtained with a different demodulator would not tell
                  us the product works.
      "hilbert" — analytic-signal magnitude, as ml/verify_signals.py uses. For
                  a narrowband carrier the two are equivalent; features.py
                  prefers rectification because Hilbert needs two full-length
                  FFTs (measured ~6x slower, and the Pi has a 500 ms budget).
                  Offered here so you can confirm the choice costs nothing.
    """
    from features import envelope_spectrum as feat_env_spectrum

    freqs, acc, n = None, None, 0
    for audio, _, _ in iter_windows(rec, window_s):
        if method == "rectify":
            f, m = feat_env_spectrum(audio, rec.fs_audio, band)
        elif method == "hilbert":
            from scipy.signal import butter, hilbert, sosfilt
            lo = max(band[0], 1.0)
            hi = min(band[1], rec.fs_audio / 2 * 0.98)
            sos = butter(4, [lo, hi], btype="band", fs=rec.fs_audio, output="sos")
            env = np.abs(hilbert(sosfilt(sos, audio)))
            f, m = _window_rfft(env, rec.fs_audio)
        else:
            raise ValueError("method must be 'rectify' or 'hilbert'")
        acc = m if acc is None else acc + m
        freqs = f
        n += 1
        if n >= max_windows:
            break
    if n == 0:
        raise RecordingError("no complete windows in recording")
    return freqs, acc / n


# ----------------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------------

def peak_to_background(freqs: np.ndarray, mag: np.ndarray, target_hz: float,
                       tol_pct: float = 2.0, bg_lo: float = 20.0,
                       bg_hi: float = 500.0, fr_hz: float | None = None,
                       n_shaft_harmonics: int = 12) -> dict:
    """Height of the line near `target_hz` relative to the local noise floor.

    The number verify_signals.py reports, generalised for real data.

      * in-band peak: the MAXIMUM within +/-tol_pct of the target, because slip
        moves the true line off the computed frequency;
      * background: the MEDIAN magnitude over bg_lo..bg_hi. Median, not mean,
        so that the shaft harmonics and the fault line itself (which are
        exactly the tall things we are measuring) do not inflate the floor they
        are being compared against.

    WHY THERE IS SHAFT-HARMONIC EXCLUSION HERE (read this; it is subtle and it
    bit us the first time this script was run)
    ---------------------------------------------------------------------------
    verify_signals.py could use a +/-1 Hz search bin because it knew the
    synthetic BPFO exactly. Its code carries this warning:

        "bw must be tighter than the gap between BPFO (152.6 Hz) and the
         machine's 3rd harmonic (150 Hz) - 2.6 Hz apart."

    Real data forces the opposite pressure: slip means you need a WIDE window
    (+/-2 % = +/-3.05 Hz at 152.6 Hz). But a window that wide SWALLOWS the
    150 Hz shaft harmonic. The first run of this script duly reported a raw
    peak-to-background of 175x with the peak "found" at exactly 150.00 Hz —
    a shaft harmonic being confidently mistaken for a bearing fault, in both
    the healthy AND faulty recording. Slip tolerance and harmonic rejection
    are in direct tension, and you cannot have one without handling the other.

    So when `fr_hz` is supplied we mask out bins within `guard` of any shaft
    harmonic k*fr before taking the maximum. `guard` is a few spectral bins
    wide, enough to cover the Hann window's main-lobe leakage from a strong
    tone, and far narrower than the slip window.

    If the target itself falls inside a guard band, the two lines are simply
    NOT SEPARABLE by this measurement. We say so (`confounded`) and return NaN
    rather than a number that would be indistinguishable from a shaft
    harmonic. An unresolvable measurement is a fact about the experiment, not
    a result to be rounded off.

    Returns the ratio, where the peak actually was, and the implied slip — the
    last one being the diagnostic that tells you whether you are looking at
    your bearing or at something else entirely.
    """
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")

    nan = float("nan")
    blank = {"ratio": nan, "peak_hz": nan, "slip_pct": nan,
             "background": nan, "peak_mag": nan, "confounded": False,
             "excluded_hz": [], "note": ""}

    half = target_hz * tol_pct / 100.0
    sel = (freqs >= target_hz - half) & (freqs <= target_hz + half)
    if not np.any(sel):
        return {**blank, "note": "target outside the analysed frequency range"}

    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 0.0
    guard = max(3.0 * df, 0.35)      # main-lobe leakage of a Hann-windowed tone

    excluded: list[float] = []
    confounded = False
    keep = sel.copy()
    if fr_hz and fr_hz > 0:
        for k in range(1, n_shaft_harmonics + 1):
            hk = k * fr_hz
            if hk < target_hz - half - guard or hk > target_hz + half + guard:
                continue
            excluded.append(hk)
            if abs(hk - target_hz) <= guard:
                # The fault line and the shaft harmonic are the same line at
                # this resolution. Nothing honest can be reported.
                confounded = True
            keep &= ~((freqs >= hk - guard) & (freqs <= hk + guard))

    if confounded:
        return {**blank, "confounded": True, "excluded_hz": excluded,
                "note": (f"target {target_hz:.2f} Hz is within {guard:.2f} Hz of a "
                         f"shaft harmonic — not separable at this resolution")}
    if not np.any(keep):
        return {**blank, "excluded_hz": excluded,
                "note": "slip window is entirely covered by shaft-harmonic "
                        "guard bands — reduce --tol-pct"}

    band_mag, band_f = mag[keep], freqs[keep]
    i = int(np.argmax(band_mag))
    peak_mag, peak_hz = float(band_mag[i]), float(band_f[i])

    bg_sel = (freqs >= bg_lo) & (freqs <= bg_hi)
    background = float(np.median(mag[bg_sel])) if np.any(bg_sel) else nan

    return {
        "ratio": peak_mag / (background + 1e-12),
        "peak_hz": peak_hz,
        "slip_pct": 100.0 * (peak_hz - target_hz) / target_hz,
        "background": background,
        "peak_mag": peak_mag,
        "confounded": False,
        "excluded_hz": excluded,
        "note": "",
    }


def choose_band(rec: Recording, window_s: float, n_probe: int = 8) -> tuple[tuple[float, float], float]:
    """Pick the demodulation band with features.select_demodulation_band, the
    shipping protrugram, averaged over a few windows for stability.

    We take the MODAL choice across probe windows rather than the first one: a
    single window on a noisy machine can pick a band on a fluke, and a band
    that only wins once is not where the resonance is."""
    from features import DEFAULT_BAND, select_demodulation_band

    picks: list[tuple[tuple[float, float], float]] = []
    for i, (audio, _, _) in enumerate(iter_windows(rec, window_s)):
        picks.append(select_demodulation_band(audio, rec.fs_audio))
        if i + 1 >= n_probe:
            break
    if not picks:
        return DEFAULT_BAND, 0.0

    tally: dict[tuple[float, float], list[float]] = {}
    for band, crest in picks:
        tally.setdefault(band, []).append(crest)
    best = max(tally.items(), key=lambda kv: (len(kv[1]), np.mean(kv[1])))
    return best[0], float(np.mean(best[1]))


def analyse_pair(healthy: Recording, faulty: Recording, geom: BearingGeometry,
                 fr_hz: float, window_s: float = 10.0, tol_pct: float = 2.0,
                 max_windows: int = 200, method: str = "rectify",
                 band: tuple[float, float] | None = None,
                 race: str = "outer") -> dict:
    """Run the full Week-2 comparison. Returns everything needed to print a
    verdict and draw the figure."""
    target_name = "BPFO" if race == "outer" else "BPFI"
    target_hz = geom.bpfo(fr_hz) if race == "outer" else geom.bpfi(fr_hz)

    # One band, chosen on the faulty record, applied to both. See module docstring.
    if band is None:
        band, band_crest = choose_band(faulty, window_s)
    else:
        band_crest = float("nan")

    f_raw_h, m_raw_h = averaged_raw_spectrum(healthy, window_s, max_windows)
    f_raw_f, m_raw_f = averaged_raw_spectrum(faulty, window_s, max_windows)
    f_env_h, m_env_h = averaged_envelope_spectrum(healthy, band, window_s, max_windows, method)
    f_env_f, m_env_f = averaged_envelope_spectrum(faulty, band, window_s, max_windows, method)

    # fr_hz is passed so shaft harmonics are masked out of the search window —
    # see peak_to_background's docstring for why this is not optional.
    kw = dict(target_hz=target_hz, tol_pct=tol_pct, fr_hz=fr_hz)
    res = {
        "target_name": target_name,
        "target_hz": target_hz,
        "fr_hz": fr_hz,
        "band": band,
        "band_crest": band_crest,
        "method": method,
        "window_s": window_s,
        "tol_pct": tol_pct,
        "raw_healthy": peak_to_background(f_raw_h, m_raw_h, **kw),
        "raw_faulty": peak_to_background(f_raw_f, m_raw_f, **kw),
        "env_healthy": peak_to_background(f_env_h, m_env_h, **kw),
        "env_faulty": peak_to_background(f_env_f, m_env_f, **kw),
        "spectra": {
            "raw_healthy": (f_raw_h, m_raw_h), "raw_faulty": (f_raw_f, m_raw_f),
            "env_healthy": (f_env_h, m_env_h), "env_faulty": (f_env_f, m_env_f),
        },
        "geom": geom,
    }

    # Harmonics are the strongest confirmation available: noise does not
    # produce a peak at exactly 2x and 3x your predicted frequency. A real
    # outer-race defect almost always shows a comb.
    res["harmonics"] = [
        peak_to_background(f_env_f, m_env_f, target_hz=h * target_hz,
                           tol_pct=tol_pct, fr_hz=fr_hz)
        for h in (1, 2, 3)
    ]
    return res


def verdict(res: dict, min_ratio: float = 4.0, min_contrast: float = 2.0) -> dict:
    """Apply the three-part Gate-2 test described in the module docstring.

    If the fault frequency could not be separated from a shaft harmonic, the
    result is INCONCLUSIVE rather than PASS or FAIL. That is a real third
    outcome and collapsing it into "FAIL" would send the students chasing a
    mounting problem they do not have. The fix for inconclusive is to change
    the EXPERIMENT (run at a different speed so BPFO moves off the harmonic),
    not the analysis."""
    env_f = res["env_faulty"]["ratio"]
    env_h = res["env_healthy"]["ratio"]
    raw_f = res["raw_faulty"]["ratio"]

    if res["env_faulty"]["confounded"]:
        return {
            "passed": False, "inconclusive": True, "checks": [],
            "contrast": float("nan"), "envelope_gain": float("nan"),
            "reason": res["env_faulty"]["note"],
        }

    contrast = env_f / (env_h + 1e-12)
    # The raw measurement may be unusable (confounded) while the envelope one
    # is fine — that asymmetry is itself the project's argument, so treat a
    # confounded raw ratio as "raw told us nothing", which check C passes.
    raw_usable = np.isfinite(raw_f)
    gain = env_f / (raw_f + 1e-12) if raw_usable else float("inf")

    checks = [
        ("A absolute", bool(env_f >= min_ratio),
         f"envelope ratio at {res['target_name']} on FAULTY = {env_f:.1f}x "
         f"(need >= {min_ratio:g}x)"),
        ("B contrast", bool(contrast >= min_contrast),
         f"faulty/healthy envelope ratio = {contrast:.1f}x "
         f"(need >= {min_contrast:g}x)   [{env_f:.1f}x vs {env_h:.1f}x]"),
        ("C method", bool((not raw_usable) or env_f > raw_f),
         (f"envelope {env_f:.1f}x vs raw {raw_f:.1f}x on the same faulty "
          f"recording = {gain:.1f}x advantage for demodulation") if raw_usable
         else (f"envelope {env_f:.1f}x vs raw UNMEASURABLE (fault frequency "
               f"confounded with a shaft harmonic in the raw spectrum)")),
    ]
    return {"passed": all(c[1] for c in checks), "inconclusive": False,
            "checks": checks, "contrast": contrast, "envelope_gain": gain}


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def make_figure(res: dict, out_path: Path, title_extra: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target, name = res["target_hz"], res["target_name"]
    fr = res["fr_hz"]
    fmax = min(500.0, 4.2 * target)
    band = res["band"]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11))

    def _mark(ax):
        for h in range(1, 4):
            if h * target <= fmax:
                ax.axvline(h * target, color="red", ls="--", lw=1, alpha=0.75)
                ax.text(h * target, ax.get_ylim()[1] * 0.86, f" {name}x{h}",
                        color="red", fontsize=8, rotation=90, va="top")
        for h in range(1, 6):
            if h * fr <= fmax:
                ax.axvline(h * fr, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlim(0, fmax)
        ax.grid(alpha=0.25, lw=0.5)

    # --- 1. raw spectra ------------------------------------------------------
    ax = axes[0]
    for key, lab, c in (("raw_healthy", "healthy", "tab:blue"),
                        ("raw_faulty", "faulty", "tab:red")):
        f, m = res["spectra"][key]
        sel = f <= fmax
        ax.semilogy(f[sel], m[sel] / (np.median(m[(f >= 20) & (f <= 500)]) + 1e-12),
                    lw=0.85, color=c, label=lab, alpha=0.85)
    ax.set_title(f"RAW spectrum (averaged, {res['window_s']:g} s windows) — "
                 f"{name} = {target:.1f} Hz should be INVISIBLE here for an early fault")
    ax.set_ylabel("magnitude / median")
    ax.legend(fontsize=8)
    _mark(ax)

    # --- 2. envelope spectra -------------------------------------------------
    ax = axes[1]
    for key, lab, c in (("env_healthy", "healthy", "tab:blue"),
                        ("env_faulty", "faulty", "tab:red")):
        f, m = res["spectra"][key]
        sel = f <= fmax
        ax.semilogy(f[sel], m[sel] / (np.median(m[(f >= 20) & (f <= 500)]) + 1e-12),
                    lw=0.95, color=c, label=lab, alpha=0.85)
    ax.set_title(f"ENVELOPE spectrum, demod band {band[0]:.0f}-{band[1]:.0f} Hz "
                 f"({res['method']}) — the fault line lives here")
    ax.set_ylabel("magnitude / median")
    ax.legend(fontsize=8)
    _mark(ax)

    # --- 3. the bar chart that IS the result ---------------------------------
    ax = axes[2]
    labels = ["raw\nhealthy", "raw\nfaulty", "envelope\nhealthy", "envelope\nfaulty"]
    vals = [res["raw_healthy"]["ratio"], res["raw_faulty"]["ratio"],
            res["env_healthy"]["ratio"], res["env_faulty"]["ratio"]]
    cols = ["lightsteelblue", "tab:blue", "lightcoral", "tab:red"]
    bars = ax.bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}x",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(SYNTHETIC_REFERENCE["env"], color="green", ls="--", lw=1)
    ax.text(0.02, SYNTHETIC_REFERENCE["env"],
            f" synthetic envelope reference {SYNTHETIC_REFERENCE['env']:.1f}x",
            color="green", fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax.axhline(SYNTHETIC_REFERENCE["raw"], color="darkgreen", ls=":", lw=1)
    ax.text(0.02, SYNTHETIC_REFERENCE["raw"],
            f" synthetic raw reference {SYNTHETIC_REFERENCE['raw']:.1f}x",
            color="darkgreen", fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylabel(f"peak-to-background at {name}")
    ax.set_title(f"The Week-2 measurement: peak-to-background at {name} "
                 f"(+/-{res['tol_pct']:g} % slip window)")
    ax.grid(alpha=0.25, axis="y", lw=0.5)

    axes[1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Week-2 real-data check — {res['geom'].designation}, "
                 f"fr = {fr:.2f} Hz ({fr * 60:.0f} rpm){title_extra}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def print_report(res: dict, v: dict, healthy: Recording, faulty: Recording,
                 fig_path: Path | None) -> None:
    g = res["geom"]
    name = res["target_name"]
    p = print

    p("=" * 74)
    p("WEEK-2 REAL-DATA ENVELOPE CHECK")
    p("=" * 74)
    p(f"healthy : {healthy.path.name if healthy.path else '<generated>'}  "
      f"[{healthy.describe()}]")
    p(f"faulty  : {faulty.path.name if faulty.path else '<generated>'}  "
      f"[{faulty.describe()}]")
    p("")
    p(f"bearing : {g.designation}   (geometry confidence: {g.confidence})")
    p(f"          N={g.n_elements}  d={g.ball_diameter_mm:.3f} mm  "
      f"D={g.pitch_diameter_mm:.3f} mm  gamma={g.gamma:.5f}")
    p(f"shaft   : fr = {res['fr_hz']:.3f} Hz ({res['fr_hz'] * 60:.0f} rpm)")
    p(f"predict : {name} = {res['target_hz']:.2f} Hz   "
      f"(BPFO {g.bpfo(res['fr_hz']):.2f} / BPFI {g.bpfi(res['fr_hz']):.2f} / "
      f"FTF {g.ftf(res['fr_hz']):.2f} Hz)")
    bc = res["band_crest"]
    p(f"demod   : {res['band'][0]:.0f}-{res['band'][1]:.0f} Hz "
      f"(protrugram on faulty" + (f", crest {bc:.1f}" if np.isfinite(bc) else "") + ")")
    p(f"analysis: {res['window_s']:g} s windows, {res['method']} demodulation, "
      f"+/-{res['tol_pct']:g} % slip window")
    if g.confidence != "published":
        p("")
        p("  NOTE: the bearing geometry is ESTIMATED, so the predicted "
          f"{name} is good to")
        p("  a few percent only. Treat a near-miss as inconclusive, not as a "
          "failure.")
    p("")

    p(f"--- peak-to-background at {name} " + "-" * 40)
    p(f"  {'spectrum':<22}{'healthy':>12}{'faulty':>12}   peak found at")
    p("  " + "-" * 62)
    for label, hk, fk in (("raw", "raw_healthy", "raw_faulty"),
                          ("envelope", "env_healthy", "env_faulty")):
        h, f_ = res[hk], res[fk]
        if f_["confounded"] or not np.isfinite(f_["ratio"]):
            p(f"  {label:<22}{'--':>12}{'--':>12}   NOT MEASURABLE: {f_['note']}")
        else:
            p(f"  {label:<22}{h['ratio']:>11.1f}x{f_['ratio']:>11.1f}x   "
              f"{f_['peak_hz']:.2f} Hz ({f_['slip_pct']:+.2f} % vs predicted)")
    ex = res["env_faulty"].get("excluded_hz") or res["raw_faulty"].get("excluded_hz")
    if ex:
        p("")
        p(f"  shaft harmonics masked out of the search window: "
          + ", ".join(f"{v:.1f} Hz" for v in ex))
        p(f"  (without this mask a shaft harmonic inside the +/-"
          f"{res['tol_pct']:g} % slip window is")
        p("   reported as a bearing fault — in the healthy recording too.)")
    p("")
    p(f"  synthetic reference (ml/verify_signals.py, severity 0.15):")
    p(f"    raw {SYNTHETIC_REFERENCE['raw']:.1f}x   "
      f"envelope {SYNTHETIC_REFERENCE['env']:.1f}x")
    p("")

    p(f"--- {name} harmonic comb in the faulty envelope spectrum " + "-" * 12)
    p("  (a real defect rings at 1x, 2x, 3x — noise does not)")
    for i, h in enumerate(res["harmonics"], start=1):
        if np.isfinite(h["ratio"]):
            p(f"    {name}x{i}  predicted {i * res['target_hz']:7.2f} Hz   "
              f"found {h['peak_hz']:7.2f} Hz   {h['ratio']:6.1f}x  "
              f"({h['slip_pct']:+.2f} %)")
        else:
            p(f"    {name}x{i}  outside analysed range")
    p("")

    p("--- GATE 2 verdict " + "-" * 54)
    if v.get("inconclusive"):
        p(f"  ==> WEEK-2 GATE: INCONCLUSIVE")
        p(f"      {v['reason']}.")
        p("      This is an EXPERIMENT problem, not an analysis problem: at")
        p("      this shaft speed the fault line lands on top of a shaft")
        p("      harmonic. Re-run the rig at a different speed (even 10 %")
        p("      away is plenty) so the two separate, then repeat.")
        if fig_path:
            p("")
            p(f"  figure: {fig_path}")
        p("=" * 74)
        return
    for tag, ok, msg in v["checks"]:
        p(f"  [{'PASS' if ok else 'FAIL'}] {tag:<12} {msg}")
    p("")
    p(f"  ==> WEEK-2 GATE: {'PASS' if v['passed'] else 'FAIL'}")
    if v["passed"]:
        p("      The envelope signature appears on real data at the frequency")
        p("      physics predicts. The product hypothesis survives.")
    else:
        p("      Do not panic and do not rewrite the detector. Work through,")
        p("      in order: (1) is the rpm right? measure it, do not assume;")
        p("      (2) is the mount rigid and on the BEARING HOUSING, not the")
        p("      casing? (3) is the demodulation band on an actual resonance")
        p("      — try --band to force one; (4) is the defect big enough to")
        p("      excite anything? See the execution plan (not in this public copy) Gate 2 and the")
        p("      kill-criteria section.")
    if fig_path:
        p("")
        p(f"  figure: {fig_path}")
    p("=" * 74)


# ----------------------------------------------------------------------------
# Demo mode (self-test without recordings)
# ----------------------------------------------------------------------------

def _demo_recordings(severity: float = 0.15, duration_s: float = 60.0):
    """Build a healthy/faulty pair with ml/simulate.py so the script can be
    exercised end-to-end before any real recording exists. This is a SELF-TEST
    of this script, not evidence about real machines — the report says so."""
    from simulate import SimConfig, bearing_fault_signal, normal_signal

    cfg = SimConfig(duration_s=duration_s)
    h = normal_signal(cfg, cfg.fs_audio, np.random.default_rng(1))
    f = bearing_fault_signal(cfg, cfg.fs_audio, np.random.default_rng(2),
                             severity=severity, race="outer")
    mk = lambda x: Recording(audio=x, fs_audio=cfg.fs_audio,  # noqa: E731
                             meta={"synthetic": True})
    return mk(h), mk(f), cfg


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyse_recording.py",
        description="Week-2 gate: does the envelope signature appear at the "
                    "computed BPFO on a real healthy/faulty recording pair?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --demo\n"
            "  %(prog)s --healthy h.wav --faulty f.wav --bearing 6204 --rpm 2850\n"
            "  %(prog)s --healthy h.wav --faulty f.wav --mic-only \\\n"
            "      --bore-mm 20 --od-mm 47 --n 8 --rpm 2850 --race outer\n"
        ),
    )
    io = p.add_argument_group("recordings")
    io.add_argument("--healthy", type=Path, help="healthy .wav (canonical format)")
    io.add_argument("--faulty", type=Path, help="faulty .wav")
    io.add_argument("--healthy-accel", type=Path, help="override accel CSV path")
    io.add_argument("--faulty-accel", type=Path, help="override accel CSV path")
    io.add_argument("--mic-only", action="store_true",
                    help="no accelerometer expected (documented Week-1 fallback)")
    io.add_argument("--demo", action="store_true",
                    help="self-test on synthetic signals; no recordings needed")

    b = p.add_argument_group("bearing + speed")
    b.add_argument("--bearing", help="designation, e.g. 6204 (see "
                                     "fault_frequencies.py --list)")
    b.add_argument("--n", "--n-elements", dest="n_elements", type=int)
    b.add_argument("--ball-mm", type=float)
    b.add_argument("--pitch-mm", type=float)
    b.add_argument("--bore-mm", type=float)
    b.add_argument("--od-mm", type=float)
    b.add_argument("--contact-angle", type=float, default=0.0)
    b.add_argument("--rpm", type=float, help="shaft speed, rpm (MEASURE it)")
    b.add_argument("--fr", type=float, help="shaft speed, Hz")
    b.add_argument("--race", choices=("outer", "inner"), default="outer",
                   help="which race is defective (default outer — the usual "
                        "seeded fault and the easier signature)")

    a = p.add_argument_group("analysis")
    a.add_argument("--window-s", type=float, default=10.0,
                   help="analysis window, s (default 10; NOT the 30 s "
                        "production inference window — this one only sets "
                        "spectral resolution and averaging)")
    a.add_argument("--max-windows", type=int, default=200,
                   help="cap on windows averaged per recording (default 200)")
    a.add_argument("--tol-pct", type=float, default=2.0,
                   help="slip search window, %% of the target (default 2.0)")
    a.add_argument("--method", choices=("rectify", "hilbert"), default="rectify",
                   help="demodulator: rectify = the shipping features.py path "
                        "(default); hilbert = the verify_signals.py path")
    a.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"),
                   help="force the demodulation band in Hz instead of using "
                        "the protrugram")

    g = p.add_argument_group("gate thresholds")
    g.add_argument("--min-ratio", type=float, default=4.0)
    g.add_argument("--min-contrast", type=float, default=2.0)

    o = p.add_argument_group("output")
    o.add_argument("--out", type=Path, default=Path("output/week2_real_data.png"),
                   help="figure path (default output/week2_real_data.png)")
    o.add_argument("--no-figure", action="store_true")
    return p


def resolve_geometry(args) -> BearingGeometry:
    """Shared with validate_public_dataset.py."""
    if args.bearing:
        return lookup(args.bearing)
    if args.n_elements and args.ball_mm and args.pitch_mm:
        return BearingGeometry(
            designation="custom (explicit geometry)",
            n_elements=args.n_elements, ball_diameter_mm=args.ball_mm,
            pitch_diameter_mm=args.pitch_mm,
            contact_angle_deg=args.contact_angle,
            source="supplied on the command line", confidence="published")
    if args.bore_mm and args.od_mm and args.n_elements:
        return from_boundary_dimensions(args.bore_mm, args.od_mm, args.n_elements,
                                        ball_diameter_mm=args.ball_mm,
                                        contact_angle_deg=args.contact_angle)
    raise ValueError(
        "no bearing geometry. Use --bearing 6204, or --n/--ball-mm/--pitch-mm, "
        "or --bore-mm/--od-mm/--n. Run "
        "'python ml/realdata/fault_frequencies.py --list' to see the table.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.demo:
            healthy, faulty, cfg = _demo_recordings()
            geom = lookup("6204")
            fr_hz = cfg.fr
            print("*** DEMO MODE: synthetic signals from ml/simulate.py. ***")
            print("*** This exercises the script; it is NOT evidence about "
                  "real machines. ***\n")
        else:
            if not args.healthy or not args.faulty:
                print("error: need --healthy and --faulty (or --demo).",
                      file=sys.stderr)
                return 2
            geom = resolve_geometry(args)
            if args.rpm is None and args.fr is None:
                print("error: need --rpm (or --fr). Measure the shaft speed; "
                      "an assumed rpm invalidates the whole check.",
                      file=sys.stderr)
                return 2
            fr_hz = args.fr if args.fr is not None else rpm_to_hz(args.rpm)
            healthy = load_recording(args.healthy, args.healthy_accel,
                                     require_accel=not args.mic_only)
            faulty = load_recording(args.faulty, args.faulty_accel,
                                    require_accel=not args.mic_only)
            if abs(healthy.fs_audio - faulty.fs_audio) > 1e-6:
                print(f"error: sample-rate mismatch — healthy "
                      f"{healthy.fs_audio:g} Hz vs faulty {faulty.fs_audio:g} Hz. "
                      f"Run both through tools/ingest.py first.", file=sys.stderr)
                return 2

        band = tuple(args.band) if args.band else None
        res = analyse_pair(healthy, faulty, geom, fr_hz,
                           window_s=args.window_s, tol_pct=args.tol_pct,
                           max_windows=args.max_windows, method=args.method,
                           band=band, race=args.race)
    except (RecordingError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    v = verdict(res, args.min_ratio, args.min_contrast)

    fig_path = None
    if not args.no_figure:
        try:
            fig_path = make_figure(res, args.out,
                                   " [DEMO/SYNTHETIC]" if args.demo else "")
        except Exception as e:                       # plotting must never
            print(f"warning: could not write figure: {e}", file=sys.stderr)

    print_report(res, v, healthy, faulty, fig_path)
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
