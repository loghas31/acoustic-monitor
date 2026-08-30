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
   detected by TWO independent tests and warned about: `clipped_fraction`
   (exact on un-transcoded WAV, blind after a lossy codec) and
   `true_peak_dbtp` (survives the codec, marginal on pure tones). Both are
   needed; a phone recording is always lossy, and the flat-top test alone
   missed a clipped one by a factor of 2 on the warning floor.
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
TRUE_PEAK_WARN_DBTP = 0.0     # reaching digital full scale = suspect clipping
TRUE_PEAK_OVERSAMPLE = 4      # ITU-R BS.1770-4 uses >=4x for true-peak

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

    ⚠ A LOSSY CODEC DEFEATS THIS TEST, AND LOSSY IS THE NORMAL INPUT. AAC
    decoder overshoot perturbs every sample, so bit-identical neighbours
    vanish. Measured on the same clipped signal:

        before codec              flat-top 0.4936  -> warned
        after AAC 128k round-trip flat-top 0.0009  -> NOT warned (floor 0.001)

    while the score stayed ~250x above any real fault. Since `fridge_scan.py`
    converts .m4a via ffmpeg, a clipped phone recording can reach the analysis
    unwarned.

    **That gap is now covered by `true_peak_dbtp()`, not by this function** —
    see its docstring for the measurements. This function is kept unchanged for
    what it genuinely catches (WAV input, and any file that has not been
    through a lossy codec), because on that input it is exact and needs no
    threshold. The two are complementary, and `screen()` warns on either.

    THREE REPLACEMENTS FOR *THIS* FUNCTION WERE TRIED AND ALL REJECTED.
    Recorded so they are not retried:

      1. Level-domain (fraction of samples within a hair of the peak). Failed
         BOTH ways: still 0.0009 post-AAC, while false-positiving at 0.0029 on
         a CLEAN sine, because a sinusoid dwells near its own peak. A warning
         that fires on good audio is worse than no warning.
      2. Crest factor (peak/RMS), on the theory that a flat-topped wave is
         flat whatever the codec did to individual samples. Measured:
         clipped post-AAC 1.82 — against `normal.wav` 2.10, `bearing_outer`
         2.11, `bearing_inner` 2.30. A 1.82-vs-2.10 gap is far too narrow to
         threshold; it would misclassify real machine audio.
      3. Amplitude-histogram shape (2026-08-30). The idea was that clipping
         piles samples at the clip level, leaving an interior mode once codec
         overshoot pushes the maximum above it. It cannot work, and the reason
         is worth keeping: a sinusoid's amplitude density ALSO diverges at its
         own peak, so tonal and clipped audio are not separable in the
         amplitude domain at all. Measured p99.9/p90: clipped-post-AAC 1.126,
         clean sine post-AAC 1.208, `normal.wav` 1.261 — the "clipped" case
         sits between two clean ones. Same root cause as rejection 1.

    The practical remedy is still upstream: clipping cannot be undone after
    the fact, so the answer is not to clip while recording. `TESTS.md` says
    so, and `tools/ingest.py` runs its own clipping audit on the path
    `fridge_scan.py` actually uses.
    """
    if len(x) < 3:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return 0.0
    near = np.abs(x) >= 0.999 * peak
    flat = np.abs(np.diff(x)) <= 1e-6 * peak
    return float(np.sum(near[:-1] & near[1:] & flat)) / len(x)


def full_scale_float(data: np.ndarray) -> np.ndarray:
    """A wav's samples as read, rescaled so +-1.0 IS digital full scale.

    wav stores int16 as -32768..32767, int32 as +-2^31, uint8 as 0..255 with
    128 as silence, and float as already +-1 by convention — the same four
    cases `tools/ingest.py` documents at length. This exists as its own
    function only so that `true_peak_dbtp`'s callers, including its tests,
    cannot quietly diverge from what `main()` does: getting the scaling wrong
    does not raise, it silently reports every 16-bit recording at -90 dBTP.

    Note what this deliberately does NOT do: normalise. Peak-normalising
    before measuring makes the true peak 0 dBTP for every input in existence.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        info = np.iinfo(np.asarray(data).dtype)
        if np.asarray(data).dtype == np.uint8:
            return (x - 128.0) / 128.0
        return x / -float(info.min)
    return x


def true_peak_dbtp(x: np.ndarray,
                   oversample: int = TRUE_PEAK_OVERSAMPLE,
                   block: int = 1 << 20) -> float:
    """Inter-sample true peak in dB relative to digital full scale (dBTP).

    THE PHYSICS, because the whole point is that this survives a codec where
    `clipped_fraction` does not. A lossy codec is a frequency-domain
    quantiser: it band-limits and re-synthesises the waveform. Band-limiting a
    signal with flat tops is exactly the conditions for Gibbs ringing, so the
    reconstruction OVERSHOOTS the level the original was sliced at. The
    sample-level evidence (bit-identical neighbours) is destroyed; the
    overshoot is *created* by the same operation. So a clipped recording comes
    out of a codec reaching or exceeding 0 dBFS, and an unclipped one, which
    had headroom before encoding, still has headroom after.

    Reconstructing between samples matters, and that is what `oversample`
    does — it is the standard true-peak measurement of ITU-R BS.1770-4 /
    EBU R128, which exists for this exact reason. It is needed here because
    `fridge_scan.py`, `fan_experiment.py` and `check_phone_audio.py` all let
    ffmpeg decode to its default 16-bit PCM, which HARD-LIMITS the decoded
    overshoot back to +-1.0 and would otherwise hide it (measured below).

    MEASURED, 2026-08-30, all through a real ffmpeg AAC 128 kbps round trip.
    "flat-top" is `clipped_fraction`; the 0.001 warning floor is in brackets:

        signal                              flat-top          true peak
        ------------------------------------------------------------------
        real fan audio, ADC-clipped         0.00165 (marginal)   +3.27 dBTP
          ... same file, decoded to f32     0.00000 (MISSED)     +3.27 dBTP
        broadband noise, ADC-clipped        0.00046 (MISSED)     +6.31 dBTP
        137 Hz sine, ADC-clipped            0.00034 (MISSED)     +0.05 dBTP

    and the negative controls, none of which may be flagged:

        real fan audio, 6 dB headroom       0.00000              -5.72 dBTP
        clean 137 Hz sine at full scale     0.00000              -0.00 dBTP
        `data/normal.wav`                   0.00000              -0.85 dBTP
        `data/bearing_inner.wav`            0.00000              -0.91 dBTP
        the six real phone recordings
          behind RESULTS.md Experiment 0    0.00000     -11.0 to -27.7 dBTP

    ⚠ THE REMAINING BLIND SPOT, stated so nobody reads this as solved: a
    clipped PURE TONE clears the threshold by 0.05 dB, against 3-6 dB for
    anything broadband. Tones are the one case where clipping adds almost no
    inter-sample overshoot, because the waveform between the plateaux is
    already smooth and narrowband. Real machine audio is broadband, which is
    the case this is for; a clipped test tone is not reliably caught and
    `clipped_fraction` is the test that covers it (exactly, on WAV input).

    A FOURTH FIX WAS TRIED AND MEASURED COUNTERPRODUCTIVE: making the
    converters decode to `pcm_f32le` so the overshoot is not hard-limited.
    It makes `clipped_fraction` STRICTLY WORSE — on the realistic case above
    it goes 0.00165 (warned) -> 0.00000 (missed), because 16-bit re-clipping
    is precisely what accidentally restores some bit-identical neighbours.
    True peak reads +3.27 dBTP under either decode, so it needs no change to
    any converter. The converters were therefore left alone.

    Returns -inf for an empty or all-zero signal. `x` must be scaled so that
    +-1.0 is digital full scale — measure it on the samples as read, BEFORE
    any peak normalisation, or the answer is 0.0 dBTP for every input.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(x)))
    # Oversample in blocks: a 25-minute 16 kHz recording is 24M samples, and
    # 4x upsampling all of it at once would allocate ~800 MB on a Pi.
    if oversample > 1 and x.size >= 64:
        from scipy.signal import resample_poly
        pad = 64  # discard the filter's edge transient at each block seam
        for s in range(0, x.size, block):
            seg = x[max(0, s - pad):min(x.size, s + block + pad)]
            if seg.size < 64:
                continue
            up = resample_poly(seg, oversample, 1)
            trim = pad * oversample
            up = up[trim:-trim] if up.size > 2 * trim else up
            if up.size:
                peak = max(peak, float(np.max(np.abs(up))))
    if peak <= 0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


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

    ON THE SUB-HARMONIC TOLERANCE, because a review asked for a change that is
    already here. T1.16 item 6 said "the tolerance should scale as 1.5/k, which
    is the physically correct slop". It already does, and not by accident:
    `abs(k*f0 - mains) <= tol` is `abs(f0 - mains/k) <= tol/k` rearranged. The
    slop is applied where the two combs actually meet — at the k-th harmonic —
    which is the same place `comb_score` applies its own +/-TOL_HZ, so the flag
    and the score agree about what "near" means by construction.

    Measured 2026-08-30 across the whole real candidate grid (8-300 Hz, 0.05 Hz
    steps, n=5841): this function's output is IDENTICAL to the explicit
    `abs(f0 - mains/k) <= tol/k` form on every candidate, and differs from a
    fixed `abs(f0 - mains/k) <= tol` form on 153 of them, which that form would
    over-flag. So item 6's prescription was a no-op; its *symptom* (every row
    of the top-5 table flagged) was a consequence of item 5 and went away when
    item 5 was fixed. See `_is_ratio_alias` below.
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


def dominant_comb_bin(freqs: np.ndarray, mag: np.ndarray, f0: float,
                      n_harm: int = N_HARMONICS,
                      tol: float = TOL_HZ) -> int | None:
    """Index of the single spectral line that contributes most to f0's score.

    `comb_score` averages `mag/floor` over the harmonics of f0. This returns
    the bin behind the largest of those terms — the one line the candidate's
    score mostly rests on. Returns None if no harmonic of f0 lands in range.

    Used to collapse aliases (see `screen`). Two candidates that win on the
    SAME line are not two pieces of evidence; they are one line described two
    ways.
    """
    best: int | None = None
    best_mag = -np.inf
    for k in range(1, n_harm + 1):
        target = k * f0
        if target > freqs[-1]:
            break
        sel = np.abs(freqs - target) <= tol
        if not sel.any():
            continue
        j = int(np.argmax(np.where(sel, mag, -np.inf)))
        if mag[j] > best_mag:
            best_mag = float(mag[j])
            best = j
    return best


def _is_ratio_alias(f0: float, p: float, rel_tol: float = 0.02) -> bool:
    """True if candidate f0 is an adjacent bin, a multiple, OR a sub-multiple of p.

    THE BUG THIS FIXES (T1.16 item 5). The original test was one-directional:
    it asked whether `f0/p` was near an integer, which catches f0 = 2p, 3p...
    but never f0 = p/2, p/3... A comb at p/k passes through every tooth of p,
    so sub-multiples are exactly the aliases you get, and they are what the
    real output was full of: `data/bearing_inner.wav` reported 49.25, 16.25,
    10.25, 12.50, 24.75 Hz — four sub-multiples of the first row, presented as
    five independent candidates.

    THE ROW'S OWN SUGGESTED FIX ("test p/f0 as well as f0/p") IS NOT ENOUGH,
    and this is the part worth remembering. Measured on that file: adding the
    reciprocal test changes exactly ONE of the four alias rows (24.75 ->
    25.25) and leaves the symptom — all five rows still mains-flagged,
    `best_unflagged_f0` still None. The reason is that a RELATIVE tolerance is
    the wrong currency. The grid is 0.25 Hz and the true line is at 50.00, so
    the k=3 alias lands at 16.25 whose ratio error against 49.25 is 0.031 —
    over this 0.02 — even though 3 x 16.25 = 48.75 is well inside the +/-1.5 Hz
    the comb detector itself uses. Tightening or loosening `rel_tol` to paper
    over that just trades misses for false collapses.

    So this test is kept (it is cheap and it is right as far as it goes) and
    `screen` additionally collapses candidates that share a dominant line —
    see `dominant_comb_bin`. That second test needs no tolerance at all,
    because two candidates either did or did not win on the same bin.

    Deliberately an exact SUPERSET of the original test: the `f0/p` arm and the
    2 Hz adjacency arm are unchanged, so nothing that used to be collapsed
    stops being collapsed.
    """
    if p <= 0 or f0 <= 0:
        return False
    if abs(f0 - p) <= 2.0:
        return True
    for a, b in ((f0, p), (p, f0)):          # the p/f0 arm is the new one
        r = a / b
        if abs(r - round(r)) <= rel_tol:
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
           win_s: float = 10.0, true_peak: float | None = None) -> dict:
    """Search for the strongest non-trivial harmonic comb in one recording.

    Windows longer than `win_s` are split and their envelope spectra averaged
    (see `averaged_envelope_spectrum`). Short recordings fall back to a single
    transform, which is what makes this usable on a 20-second clip as well as
    a 40-minute one.

    `true_peak` is the recording's true peak in dBTP, measured by
    `true_peak_dbtp()` on the samples AS READ FROM THE FILE — before the peak
    normalisation `main()` applies. It cannot be computed in here, and that is
    not an oversight: by the time a signal reaches `screen()` it has usually
    been normalised to peak 1.0, which makes its true peak 0 dBTP whatever the
    recording actually was. Callers that have not measured it pass None, and
    the result then reports `true_peak_dbtp: None` and
    `true_peak_clipping_suspected: False` — "not measured", not "clean".
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
    # Two independent clipping tests, because neither covers the other's case:
    # the flat-top rule is exact on un-transcoded WAV and blind after a lossy
    # codec; true peak survives the codec (that is what creates the overshoot
    # it measures) but is marginal on pure tones. Warn on either.
    tp_suspected = (true_peak is not None
                    and np.isfinite(true_peak)
                    and true_peak >= TRUE_PEAK_WARN_DBTP)

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

    # Collapse aliases of an already-accepted peak (T1.16 items 5 and 6).
    #
    # WHAT WAS WRONG. The old test asked only whether the candidate was an
    # integer MULTIPLE of an accepted peak, so every sub-multiple survived and
    # the top-5 table filled up with aliases of row 1. On
    # `data/bearing_inner.wav` the shipped output was 49.25, 16.25, 10.25,
    # 12.50, 24.75 Hz — one peak and four of its own sub-harmonics — and since
    # a comb resting on the 50 Hz line is correctly mains-flagged whichever
    # sub-multiple it wears, all five rows were flagged and `best_unflagged_f0`
    # came back None. That is item 6's symptom, and it is item 5's bug.
    #
    # WHY TWO TESTS AND NOT ONE. `_is_ratio_alias` is a frequency-ratio test
    # and it is not sufficient on its own — measured, see its docstring. The
    # second test is exact: if two candidates' scores rest on the SAME
    # spectral line, the second one is not new evidence. Here the whole table
    # was five ways of scoring a single line at 50.00 Hz that stands 149.6x
    # above the noise floor while every other harmonic each candidate touched
    # was 5-8x, i.e. noise.
    #
    # Row 1 is never collapsed (it has nothing to be collapsed against), so
    # `best_f0`/`best_score` — the numbers RESULTS.md quotes — cannot move.
    # Only rows 2-5 and the `best_unflagged_*` pair can change.
    peaks: list[tuple[float, float, str]] = []
    claimed_lines: list[int] = []
    for f0, s, flag in scored:
        if any(_is_ratio_alias(f0, p) for p, _, _ in peaks):
            continue
        line = dominant_comb_bin(freqs, mag, f0)
        if line is not None and line in claimed_lines:
            continue
        peaks.append((f0, s, flag))
        claimed_lines.append(line)
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
            "clipping_suspected": (clipped_frac > CLIP_FRAC_WARN
                                   or tp_suspected),
            "flat_top_clipping_suspected": clipped_frac > CLIP_FRAC_WARN,
            "true_peak_dbtp": true_peak,
            "true_peak_clipping_suspected": tp_suspected,
            "band_energy_ratio": band_ratio,
            "degenerate_band": degenerate,
            # Diagnostic only, and prefixed to say so. The alias-collapse rule
            # above cannot be tested from the outside without the ranked list
            # it consumes and the spectrum it consults — reconstructing either
            # in a test would mean duplicating band selection and window
            # averaging, and a test that reimplements the thing it checks
            # tests nothing. Nothing in `tools/` reads these.
            "_scored": scored,
            "_spectrum": (freqs, mag)}


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

    # Measure the true peak on the samples as read, scaled so +-1.0 is digital
    # full scale — BEFORE the normalisation below.
    tp = true_peak_dbtp(full_scale_float(data))

    # Only now normalise. The true peak of a peak-normalised signal is 0 dBTP
    # by construction, so measuring it after this line would flag everything.
    x /= (np.max(np.abs(x)) + 1e-12)

    r = screen(x, float(fs), mains=args.mains, fr=args.fr, true_peak=tp)

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

    if r["true_peak_dbtp"] is not None:
        print(f"true peak         : {r['true_peak_dbtp']:+.2f} dBTP "
              f"(warn at {TRUE_PEAK_WARN_DBTP:+.1f})")

    if r["clipping_suspected"]:
        how = ("flat tops in the waveform"
               if r["flat_top_clipping_suspected"] else
               f"true peak {r['true_peak_dbtp']:+.2f} dBTP — the recording "
               f"reaches digital full scale")
        print(f"\n  !!  CLIPPING SUSPECTED — {how}; "
              f"{100 * r['clipped_fraction']:.2f}% "
              f"of samples on a flat top.\n"
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
