#!/usr/bin/env python3
"""cold_start_screen.py — look for a fault WITHOUT a healthy baseline.

THE PROBLEM THIS EXISTS FOR
---------------------------
The main detector is self-baselining: it learns what a machine normally sounds
like, then flags departures. That is the right design and F18 measured why (a
different HEALTHY unit of the same model scores 4.27x threshold against unit
A's baseline — unit-to-unit variation is larger than the fault signal, so a
shared reference library cannot work).

But it has a cold-start hole, and it is the case most customers are actually
in: **if the machine is already faulty when it learns, the fault becomes part
of "normal" and the detector goes silent forever.** Nobody buys this for a
brand-new machine. They buy it because they are worried about an old one.

WHAT THIS DOES INSTEAD
----------------------
It asks a question that needs no history at all:

    "Is there a periodic impact train in this recording?"

Healthy rotating machinery is broadly aperiodic — bearings roll, fluid moves,
nothing strikes anything rhythmically. A spalled bearing, a chipped gear tooth
or a broken rotor bar produces an *impact once per revolution of something*,
which shows up as a comb of harmonics in the envelope spectrum: energy at f0,
2*f0, 3*f0, ... That comb is the fault's signature and it is present whether or
not anyone recorded the machine last year.

So: demodulate, take the envelope spectrum, and search every plausible
fundamental for a harmonic comb. Report the strongest.

WHAT IT CANNOT DO, STATED UP FRONT
----------------------------------
1. **It only finds IMPULSIVE faults.** Bearings, gears, rotor bars, cavitation.
   It will not find imbalance, misalignment, worn seals, gas-charge loss, or a
   tired compressor — faults that change broadband level or resonance without
   impacting. The self-baselined detector is better at those. These two are
   complements, not competitors.
2. **It does not name the fault** unless you give it the bearing geometry and
   shaft speed, which for a sealed fridge compressor you do not have. Without
   them it reports "periodic impacting at 87 Hz", which is a reason to
   investigate, not a diagnosis.
3. **It is a screen, not a verdict.** A clean result is weak evidence of health;
   a dirty result is a reason to look closer.

THE CONFOUNDER THAT ALREADY BIT THIS PROJECT ONCE
-------------------------------------------------
`ml/realdata/analyse_recording.py` records that a 150 Hz **shaft harmonic** was
once confidently reported as a bearing fault — in the healthy AND the faulty
recording. Rotating machines are full of legitimate periodicity: shaft rate and
its harmonics, blade/vane passing, and on any mains-powered machine a strong
50 Hz (UK) line plus harmonics from magnetostriction and torque ripple.

A naive comb search finds those first, every time, on a perfectly good machine.
So candidates coinciding with mains (`--mains`) or a known shaft rate (`--fr`)
are **flagged and still ranked** — never removed.

That distinction was itself learned the hard way. An earlier version discarded
them, and on `data/bearing_inner.wav` it threw away the true answer: an
inner-race fault is amplitude-modulated at the SHAFT rate (the defect passes
through the load zone once per revolution), and on a direct-drive machine
running at 50 rev/s that rate IS the mains frequency. The screen binned a comb
scoring 39.6 as "hum" and confidently reported 7.6 of noise instead. Silent
rejection is how you end up trusting a number you should not — including when
the thing doing the rejecting is this file.

THREE WAYS A RECORDING CAN MAKE THIS TOOL LIE
---------------------------------------------
All three were found by adversarial testing, and all three are now caught and
reported rather than silently producing a confident number:

1. **Non-finite samples** — NaN propagates to the score, and `nan > threshold`
   is False, so a corrupted file reports "nothing found". Now refused outright.
2. **Clipping** — flat-topping a waveform makes it a square wave, i.e. a
   harmonic series, i.e. precisely what this searches for. A hard-clipped sine
   scores 2.9e7 against a real fault's ~35 — seven orders of magnitude. Now
   detected (see `clipped_fraction`) and warned about.
3. **An empty demodulation band** — if there is nothing above 1 kHz, the score
   becomes noise divided by noise. A clean 137 Hz tone scored 13431 this way.
   Now caught by a band-energy ratio check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "firmware", ROOT / "ml" / "realdata"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features import envelope_spectrum, select_demodulation_band  # noqa: E402

F0_LO, F0_HI = 8.0, 300.0     # plausible impact rates for the machines in scope
N_HARMONICS = 5               # comb depth scored
TOL_HZ = 1.5                  # half-width of the peak search around k*f0
MAINS_REJECT_HZ = 1.5         # how close to a mains harmonic counts as mains
CLIP_FRAC_WARN = 0.001        # >0.1% of samples at full scale = suspect clipping
BAND_ENERGY_FLOOR = 1e-3      # band-envelope/signal RMS below this = empty band
MIN_FS = 12000.0              # below this the 1-20 kHz search band collapses

# WHY CLIPPING IS SPECIFICALLY DANGEROUS FOR *THIS* TOOL, more than for the
# rest of the project. Clipping flattens waveform peaks, and a flattened
# sinusoid is a square wave — whose defining property is a rich harmonic
# series. This tool's entire method is looking for harmonic series. Measured
# in the adversarial pass: a clean sine scores ~5, and the same sine clipped
# scores **2.9e7**. Clipping does not merely add noise, it
# manufactures exactly the evidence being searched for, at a level no genuine
# fault in this repo has ever reached.
#
# Phone recordings clip easily — a fridge compressor kicking in with the phone
# resting on the casing is a realistic way to get there. So this is warned
# about loudly rather than silently corrected.


def clipped_fraction(x: np.ndarray) -> float:
    """Fraction of samples sitting on a FLAT TOP at full scale.

    Naive "how many samples are near the peak" does not work, and the failure
    is not hypothetical — measured in the adversarial pass, a clean 137 Hz sine
    reads 2.8 % by that rule, because a sinusoid genuinely dwells near its own
    peak. Warning on that would train the user to ignore the warning, which is
    worse than not having one.

    What actually distinguishes clipping is that the clipped samples are
    *bit-identical* to each other — the waveform has been sliced flat, so
    consecutive samples differ by zero. A sine near its peak still curves: at
    137 Hz and 16 kHz consecutive samples there differ by ~7e-4 of full scale,
    small but never zero.

    So: near full scale AND flat (successive difference below a quantisation-
    sized epsilon), for at least two samples in a row.

    ⚠ KNOWN LIMITATION — A LOSSY CODEC DEFEATS THIS, AND LOSSY IS THE NORMAL
    INPUT. AAC decoder overshoot perturbs every sample, so bit-identical
    neighbours vanish. Measured on the same clipped signal:

        before codec              flat-top 0.4936  -> warned
        after AAC 128k round-trip flat-top 0.0009  -> NOT warned (floor 0.001)

    while the score stayed ~250x above any real fault. Since `fridge_scan.py`
    converts .m4a via ffmpeg, a clipped phone recording can reach the analysis
    unwarned.

    TWO REPLACEMENTS WERE TRIED AND BOTH REJECTED. Recorded so they are not
    retried:

      1. Level-domain (fraction of samples within a hair of the peak). Failed
         BOTH ways: still 0.0009 post-AAC, while false-positiving at 0.0029 on
         a CLEAN sine, because a sinusoid dwells near its own peak. A warning
         that fires on good audio is worse than no warning.
      2. Crest factor (peak/RMS), on the theory that a flat-topped wave is
         flat whatever the codec did to individual samples. Measured:
         clipped post-AAC 1.82 — against `normal.wav` 2.10, `bearing_outer`
         2.11, `bearing_inner` 2.30. A 1.82-vs-2.10 gap is far too narrow to
         threshold; it would misclassify real machine audio.

    So the flat-top rule is kept for what it genuinely catches (WAV input, and
    any file that has not been through a lossy codec), and the codec case is
    an OPEN limitation — filed in the task backlog (not in this public copy). The practical remedy is
    upstream anyway: clipping cannot be undone after the fact, so the answer is
    not to clip while recording. `TESTS.md` says so, and `tools/ingest.py` runs
    its own clipping audit on the path `fridge_scan.py` actually uses.
    """
    if len(x) < 3:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return 0.0
    near = np.abs(x) >= 0.999 * peak
    flat = np.abs(np.diff(x)) <= 1e-6 * peak
    return float(np.sum(near[:-1] & near[1:] & flat)) / len(x)


def comb_score(freqs: np.ndarray, mag: np.ndarray, f0: float,
               n_harm: int = N_HARMONICS, tol: float = TOL_HZ) -> float:
    """Harmonic-sum score for a candidate fundamental, in units of local noise.

    For each harmonic k*f0, take the strongest bin within +/-tol and divide by
    the MEDIAN magnitude of the whole band (a robust noise floor — the mean
    would be dragged up by the very peaks being measured). Average over
    harmonics.

    Averaging rather than summing matters: a single enormous line (mains, a
    resonance) should NOT beat five moderate ones. The comb is the evidence,
    not the height of any one peak.

    TWO ALTERNATIVES WERE TRIED AND ARE WORSE. Recorded so nobody spends the
    afternoon rediscovering it. At severity 0.20 this picks 36.8 Hz and 18.2 Hz
    — exact sub-harmonics of the true BPFO at 73.65 Hz. A comb at f0/2 contains
    every tooth of f0 plus noise between them, so the obvious fix is to punish
    combs with weak teeth:

        statistic        sev 0.35    sev 0.20    sev 0.10
        mean  (shipped)     6/6         2/6         4/6
        min                 0/6         2/6         4/6
        geometric mean      6/6         2/6         4/6

    (Re-measured after `averaged_envelope_spectrum` landed. The pre-averaging
    figures were 6/6, 0/6, 2/6. The 0.20 and 0.10 columns are coin-flips in
    both versions — the medians there are indistinguishable, so the win counts
    carry no signal; see `self_test`, which now says so explicitly.)

    `min` destroys the result it was meant to protect (6/6 -> 0/6 on the loud
    fault) because one harmonic landing in a spectral null zeroes the whole
    score. The geometric mean is indistinguishable from the arithmetic one.
    **Neither recovers severity 0.20**, which means the sub-harmonic pick is a
    symptom, not the disease: at that severity the true comb is not reliably
    above the noise to begin with. That is consistent with F18, where severity
    0.20 faults scored 0.55-0.70x threshold against their OWN correct baseline
    — i.e. the self-baselined detector misses them on this generator too.

    Conclusion: the ceiling here is the signal, not the statistic. Do not
    retune this scoring function hoping to reach early faults.
    """
    floor = np.median(mag) + 1e-12
    hits = []
    for k in range(1, n_harm + 1):
        target = k * f0
        if target > freqs[-1]:
            break
        sel = np.abs(freqs - target) <= tol
        if not sel.any():
            continue
        hits.append(np.max(mag[sel]) / floor)
    if len(hits) < 3:            # need a comb, not a line or two
        return 0.0
    return float(np.mean(hits))


def is_mains_related(f0: float, mains: float, tol: float = MAINS_REJECT_HZ) -> bool:
    """True if f0 sits on a mains harmonic OR mains sits on a harmonic of f0.

    Both directions matter. 50 Hz obviously must go. But so must 25 Hz, whose
    2nd harmonic IS the mains line — a comb built mostly out of mains energy
    wearing a lower fundamental as a disguise.
    """
    if f0 <= 0 or mains <= 0:
        return False
    for k in range(1, 9):
        if abs(f0 - k * mains) <= tol:
            return True
    for k in range(2, 9):
        if abs(k * f0 - mains) <= tol:
            return True
    return False


def averaged_envelope_spectrum(x: np.ndarray, fs: float, band, win_s: float
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Mean envelope spectrum over consecutive windows.

    WHY THIS IS WORTH THE COMPLICATION. A real fault's comb sits at the SAME
    frequency in every window, because it is set by the geometry and the shaft
    speed. Noise peaks land wherever they like. So averaging N windows adds the
    comb coherently while the noise floor falls roughly as sqrt(N).

    Measured on 3-minute recordings (18 windows) versus analysing the whole
    thing in one shot: at severity 0.35 the reported fundamental became 74 Hz
    on 6 of 6 machines (true BPFO 73.65) where the single-shot version was
    already correct but noisier, and the healthy-to-faulty contrast widened
    from 1.8x to 2.6x. This is the main reason to record for 40 minutes rather
    than 40 seconds even for the baseline-free screen.
    """
    n = int(win_s * fs)
    n_win = max(1, len(x) // n)
    acc = None
    freqs = None
    for i in range(n_win):
        f, m = envelope_spectrum(x[i * n:(i + 1) * n], fs, band)
        freqs = f
        acc = m if acc is None else acc + m
    return freqs, acc / n_win


def screen(x: np.ndarray, fs: float, mains: float = 50.0,
           fr: float | None = None, top: int = 5,
           win_s: float = 10.0) -> dict:
    """Search for the strongest non-trivial harmonic comb in one recording.

    Windows longer than `win_s` are split and their envelope spectra averaged
    (see `averaged_envelope_spectrum`). Short recordings fall back to a single
    transform, which is what makes this usable on a 20-second clip as well as
    a 40-minute one.
    """
    # ---- input validation, before any maths -------------------------------
    # Both of these were found by an adversarial pass, and both are silent
    # failures rather than crashes, which makes them worse than crashes.
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise ValueError(
            f"recording contains {n_bad} non-finite sample(s) (NaN or inf). "
            f"Refusing rather than analysing it: NaN propagates straight "
            f"through to the score, and because `nan > threshold` is False, a "
            f"corrupted recording would silently report 'nothing found' — the "
            f"most dangerous possible answer. Re-export the file.")

    # Sample rate. Bearing resonances live at 1-20 kHz and the demodulation
    # search starts at 1 kHz, so a low-rate recording has nowhere to look.
    # Measured on `bearing_outer.wav` (true answer 33.0 at 152.25 Hz):
    #     16 kHz -> 33.0 @ 152.25    (correct)
    #      8 kHz -> 10.6 @  12.50    (wrong frequency, near-healthy score)
    #      4 kHz ->  9.6 @  10.00    (wrong frequency, near-healthy score)
    # A real fault silently becomes a clean bill of health. 8 kHz is what
    # WhatsApp voice notes and telephony exports produce, so this is a
    # realistic way to be handed one.
    if fs < MIN_FS:
        raise ValueError(
            f"sample rate is {fs:g} Hz; this needs at least {MIN_FS:g} Hz. "
            f"Bearing signatures live at 1-20 kHz and the demodulation search "
            f"starts at 1 kHz, so below this the band collapses and a REAL "
            f"fault reads as healthy at the wrong frequency (measured: a fault "
            f"scoring 33.0 at 152 Hz reads 10.6 at 12.5 Hz when resampled to "
            f"8 kHz). Re-export at 16 kHz or higher; do not resample up, the "
            f"information is already gone.")

    clipped_frac = clipped_fraction(x)

    band, band_crest = select_demodulation_band(
        x[:int(win_s * fs)] if len(x) > win_s * fs else x, fs, crest_floor=0.0)

    # ---- is there anything in the band to demodulate? ---------------------
    # Found in the adversarial pass: a clean 137 Hz sine scored 13431, higher
    # than any real fault by two orders of magnitude. Cause: the tone sits
    # BELOW the 1 kHz demodulation floor, so the chosen band contains only
    # numerical leakage, and comb_score divides a noise peak by a noise median
    # to produce a huge meaningless number.
    #
    # Measured band-envelope-to-signal RMS ratios: degenerate tone 2.7e-4;
    # `normal.wav` 3.0e-2; `bearing_outer.wav` 1.9e-1; white noise 2.0e-1.
    # Everything genuine sits at least 30x above the threshold below, and the
    # degenerate case sits 4x under it.
    from features import envelope as _envelope
    env, _ = _envelope(x, fs, band)
    sig_rms = float(np.sqrt(np.mean(x ** 2))) + 1e-30
    band_ratio = float(np.sqrt(np.mean(env ** 2))) / sig_rms
    degenerate = band_ratio < BAND_ENERGY_FLOOR

    if len(x) >= 2 * win_s * fs:
        freqs, mag = averaged_envelope_spectrum(x, fs, band, win_s)
    else:
        freqs, mag = envelope_spectrum(x, fs, band)

    sel = (freqs >= 1.0) & (freqs <= 600.0)
    freqs, mag = freqs[sel], mag[sel]

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    step = max(df, 0.25)
    candidates = np.arange(F0_LO, F0_HI + step, step)

    # ANNOTATE, DO NOT DISCARD. An earlier version dropped every candidate
    # coinciding with mains or a shaft harmonic. That silently destroyed the
    # strongest evidence on `data/bearing_inner.wav`: `ml/simulate.py` line 166
    # amplitude-modulates an inner-race fault at the SHAFT RATE by design,
    # because that is the real physics — the defect passes through the load
    # zone once per revolution. With a direct-drive machine at 50 rev/s the
    # fault's modulation frequency IS the mains frequency. The screen threw
    # away a score of 39.6 as "mains hum" and reported 7.6 of noise instead.
    #
    # Any rejection rule strong enough to remove mains is strong enough to
    # remove that. So nothing is removed: coincidences are flagged and ranked
    # alongside everything else, and the human decides. This tool's own
    # docstring says silent rejection is how you end up trusting a number you
    # should not; this is that lesson applied to itself.
    scored = []
    for f0 in candidates:
        s = comb_score(freqs, mag, float(f0))
        if s <= 0:
            continue
        flag = ""
        if is_mains_related(float(f0), mains):
            flag = f"coincides with mains {mains:g} Hz"
        elif fr is not None and is_mains_related(float(f0), fr, tol=TOL_HZ):
            flag = f"coincides with shaft {fr:g} Hz"
        scored.append((float(f0), s, flag))

    scored.sort(key=lambda t: -t[1])

    # Collapse near-duplicate fundamentals (adjacent bins of one true peak).
    peaks: list[tuple[float, float, str]] = []
    for f0, s, flag in scored:
        if all(abs(f0 - p) > 2.0 and abs(f0 / p - round(f0 / p)) > 0.02
               for p, _, _ in peaks):
            peaks.append((f0, s, flag))
        if len(peaks) >= top:
            break

    clean = [p for p in peaks if not p[2]]
    return {"band": band, "band_crest": float(band_crest),
            "peaks": peaks,
            "flagged": [p for p in peaks if p[2]],
            "best_f0": peaks[0][0] if peaks else None,
            "best_score": peaks[0][1] if peaks else 0.0,
            "best_unflagged_f0": clean[0][0] if clean else None,
            "best_unflagged_score": clean[0][1] if clean else 0.0,
            "clipped_fraction": clipped_frac,
            "clipping_suspected": clipped_frac > CLIP_FRAC_WARN,
            "band_energy_ratio": band_ratio,
            "degenerate_band": degenerate}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, nargs="?")
    ap.add_argument("--mains", type=float, default=50.0,
                    help="mains frequency to reject (50 UK/EU, 60 US)")
    ap.add_argument("--fr", type=float, default=None,
                    help="shaft speed in Hz, if you know it — rejects its "
                         "harmonics too. You usually will not know this.")
    ap.add_argument("--self-test", action="store_true",
                    help="run against known healthy/faulty synthetic signals "
                         "and report whether the screen actually separates them")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.recording is None:
        ap.error("give me a recording, or --self-test")

    from scipy.io import wavfile
    fs, data = wavfile.read(args.recording)
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= (np.max(np.abs(x)) + 1e-12)

    r = screen(x, float(fs), mains=args.mains, fr=args.fr)

    print(f"\ndemodulation band : {r['band'][0]:.0f}-{r['band'][1]:.0f} Hz "
          f"(crest {r['band_crest']:.1f})")

    if r["degenerate_band"]:
        print(f"\n  !!  NO USABLE SIGNAL in the demodulation band "
              f"(ratio {r['band_energy_ratio']:.1e}).\n"
              f"      Every score below is noise divided by noise and means "
              f"nothing. This happens\n"
              f"      when the recording has almost no content above 1 kHz — a "
              f"near-silent room, a\n"
              f"      pure tone, or a codec that removed the high end. Check "
              f"the recording first.")

    if r["clipping_suspected"]:
        print(f"\n  !!  CLIPPING SUSPECTED — {100 * r['clipped_fraction']:.2f}% "
              f"of samples at full scale.\n"
              f"      Do not trust any score below. Clipping turns waveform "
              f"peaks into flat tops,\n"
              f"      which is a square wave, which is a harmonic series — the "
              f"exact thing this\n"
              f"      tool searches for. A clipped sine scores ~2.9e7 here; a "
              f"real fault\n"
              f"      scores about 35. Re-record further from the machine or "
              f"with less gain.")

    print("\ncandidate impact rates:")
    if not r["peaks"]:
        print("  none found")
    for f0, s, flag in r["peaks"]:
        note = f"   <- {flag}" if flag else ""
        print(f"  {f0:6.1f} Hz  comb score {s:5.1f}{note}")

    if r["flagged"]:
        print("\nAbout the flagged rows: they are NOT dismissed. A machine's own\n"
              "shaft rate often sits at or near the mains frequency, and an\n"
              "INNER-RACE bearing fault is amplitude-modulated at exactly the\n"
              "shaft rate — so a real fault can land precisely there. Treat a\n"
              "flagged row with a high score as worth investigating, not as hum.")

    print("\n" + "=" * 66)
    print("This is a SCREEN, not a diagnosis. It looks for periodic impacting")
    print("only — it cannot see imbalance, wear, or a tired compressor.")
    print("A clean result is weak evidence of health. Read the module docstring.")
    print("=" * 66)
    return 0


def self_test() -> int:
    """Does this screen actually separate healthy from faulty? Measure it.

    A tool that reports a number is worthless until someone has checked the
    number moves in the right direction. This runs the screen over known
    healthy and known faulty signals at three severities and prints the
    separation, so the honest answer is visible rather than assumed.
    """
    from synth_phone_recording import make_pair

    print("\ncold-start screen, measured against known ground truth")
    print("(6 machines, 16 kHz, mains rejection on)\n")
    print(f"{'severity':>9} {'healthy score':>14} {'faulty score':>13} "
          f"{'separated':>10}")
    print("-" * 50)

    rows = []
    for sev in (0.35, 0.20, 0.10):
        h_scores, f_scores = [], []
        for seed in range(1, 7):
            pair = make_pair(seed=seed, duration_s=20.0, fs=16000.0,
                             severity=sev)
            h_scores.append(screen(pair["healthy"], pair["fs"])["best_score"])
            f_scores.append(screen(pair["faulty"], pair["fs"])["best_score"])
        h, f = float(np.median(h_scores)), float(np.median(f_scores))
        sep = sum(1 for a, b in zip(h_scores, f_scores) if b > a)
        # A win count alone lies when the two distributions are the same: 6
        # coin flips give 3-4 "wins" and look like partial detection. Compare
        # the medians too and say so, rather than letting the count imply a
        # signal that is not there.
        flat = abs(f - h) / max(h, 1e-9) < 0.10
        note = "  (medians indistinguishable — this is noise)" if flat else ""
        effective = 0 if flat else sep
        rows.append((sev, h, f, effective))
        # Print the SAME number that is reasoned about below. An earlier
        # version printed the raw win count here and the flat-adjusted one in
        # the summary, so the two lines contradicted each other on screen.
        print(f"{sev:>9.2f} {h:>14.1f} {f:>13.1f} {effective:>8}/6{note}")

    print()
    loud = rows[0]
    if loud[2] > loud[1] and loud[3] >= 5:
        print("PASS: the screen separates a clear fault from healthy without "
              "any baseline.")
    else:
        print("FAIL: the screen does NOT separate them. Do not ship this as a "
              "cold-start check —\n      the harmonic comb is not carrying the "
              "fault on these signals.")
        return 1
    quiet = rows[-1]
    if quiet[3] < 4:
        print(f"NOTE: at severity {quiet[0]:.2f} it separates only "
              f"{quiet[3]}/6. Early faults are NOT reliably\n      caught "
              f"baseline-free — that remains the self-baselined detector's job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
