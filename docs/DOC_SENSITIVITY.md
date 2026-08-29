# Sensitivity study — how bad can reality be?

Backlog **T1.4**. Every AUC this project has reported before this file
(`ml/evaluate.py`: 1.000; the CWRU surrogate: 0.9889) was measured at ONE
point in signal-quality space — `ml/simulate.py`'s defaults (SNR 20 dB,
resonance 4500 Hz, no mounting loss, no neighbouring machinery). That point
was chosen because it looks like a reasonably clean small motor, not because
anyone measured where the detector actually breaks. This sweeps four axes
around it, honestly, before the bench does.

**Tool credit:** `ml/sensitivity/sweep.py` and `tests/test_sensitivity.py`
(21 tests, all passing) already existed on disk, built and tested by a
concurrent agent run, and are not redone here — this run verified them,
executed the full 20-point sweep the tool was built for, and wrote this
report. The tool's own module docstring is worth reading alongside this
file; it explains the method and the deliberate scope limits (single
regime, single accelerometer axis — T1.8/T1.9 already cover regime and axis
questions separately) in more depth than repeated here.

Reproduce: `cd ml/sensitivity && for axis point; do python3 sweep.py run
--axis $axis --value $point --out /tmp/results/$axis_$point.json; done`
then `python3 sweep.py combine --in-dir /tmp/results` (writes
`ml/artifacts/sensitivity.json` and `.png`, gitignored like the rest of
`ml/artifacts/`). Each point fits a fresh baseline on 32 learn windows and
scores 12 held-out healthy + 15 held-out fault windows (12 for the
attenuation axis) through the real `firmware/baseline.py`/`inference.py` —
~13-17 s per point, ~5 minutes for all 20.

## Method notes that affect how to read the numbers below

- **Sample sizes are small.** 12 healthy and 5-15 fault windows per point.
  A `TPR = 0.0` built from 5 trials, or an `FPR` built from 12, has wide
  binomial uncertainty — these are a sensitivity MAP, not a calibrated
  false-alarm-rate specification. Do not quote a single point's FPR as "the"
  false-alarm rate; H4's real 7-day soak is what settles that (see
  DOC_STATUS.md).
- **The resonance axis is confounded by `ml/simulate.py`'s own clamp**,
  which this sweep deliberately reuses rather than works around
  (`min(cfg.resonance_hz, 0.4 * fs)`, in the frozen simulator, unmodified
  here). At `fs_audio = 16000`, the audio channel clamps at **6400 Hz**; at
  `fs_accel = 6400`, the accelerometer channel clamps at **2560 Hz**. So the
  swept values do not mean what their label says once resonance is pushed
  high: at `resonance_hz = 4500` the accelerometer channel is already
  silently capped at 2560 Hz; at `6400` and `8000` BOTH channels are capped
  (audio at 6400, accel at 2560). The "resonance frequency (Hz)" panel in
  the figure below is therefore only a clean test of the detector below
  about **2.5 kHz** — above that it is increasingly a test of "does the
  detector still work when the resonance filter is stuck at the clamp",
  which is a real and useful question, just not the one the x-axis label
  implies.
- **One regime, one accelerometer axis, `fr = 50 Hz` fixed throughout** —
  by design, so the four axes below are isolated from questions T1.8/T1.9
  already answered. A real machine varies on more axes than this file
  tests.

## Results

![Sensitivity sweep: AUC / TPR / FPR vs four axes](../ml/artifacts/sensitivity.png)

*(mounting attenuation, interference amplitude, resonance frequency, SNR — left to right; a 0.95 AUC line is drawn as a rough gate, though nothing tested crossed it going down.)*

### 1. SNR — the one axis where detection actually breaks in range

| SNR (dB) | AUC | TPR (overall) | TPR @ severity 0.1 | FPR | fault ratio (median ×thr) |
|---|---|---|---|---|---|
| 20 | 1.000 | 1.00 | 1.00 | 0.00 | 71.3× |
| 10 | 1.000 | 1.00 | 1.00 | 0.00 | 21.6× |
| 5  | 1.000 | 1.00 | 1.00 | 0.00 | 12.4× |
| 0  | 1.000 | 1.00 | 1.00 | 0.083 | 5.0× |
| -5 | **0.878** | **0.67** | **0.00** | 0.083 | 2.0× |

At -5 dB SNR, AUC drops from a clean 1.000 to 0.878, overall TPR falls to
2/3, and every one of the 5 mildest-severity (0.1) fault windows was missed
(0/5). This is the one axis in this sweep where the detector's headline
numbers visibly move, and it moves in exactly the direction physical
intuition predicts: a quieter fault signal buried in more noise is harder to
tell from nothing. The margin (fault-window score as a multiple of
threshold) shrinks steadily and predictably from 71.3× at 20 dB to 5.0× at
0 dB even while TPR/AUC stay nominally perfect down to 0 dB — the numbers
that don't move (AUC, TPR) are hiding a real, monotone loss of margin that
the ratio column shows plainly.

### 2. Mounting attenuation — TPR holds to 24 dB tested, but margin collapses ~54×

Fixed true fault severity 0.3; the axis asks how much mechanical coupling
loss (magnetic mount, paint, off-axis placement) between the housing and the
sensor survives before a moderate fault stops being caught.

| Attenuation (dB) | AUC | TPR | FPR | fault ratio (median ×thr) |
|---|---|---|---|---|
| 0  | 1.000 | 1.00 | 0.00 | 140.6× |
| 6  | 1.000 | 1.00 | 0.00 | 64.1× |
| 12 | 1.000 | 1.00 | 0.167 | 32.1× |
| 18 | 1.000 | 1.00 | 0.00 | 11.7× |
| 24 | 1.000 | 1.00 | 0.00 | **2.6×** |

TPR and AUC stayed perfect across the whole 0-24 dB range tested — but that
is a binary detection result at one fixed severity (0.3, moderate), and it
is not the whole story: the median fault-window margin fell from 140.6× the
threshold at 0 dB to just **2.6×** at 24 dB, a **~54× collapse**. A moderate
fault at 24 dB of mechanical loss is still detected in this test, but only
just — and the trend says it would stop being detected well before 30 dB.
**This is the axis most likely to matter first on a real machine**: a
magnetic mount, paint, or an off-axis placement is entirely plausible to
lose more than 24 dB, and this sweep did not test far enough to find where
it actually breaks. Worth a follow-up sweep extending to 30-40 dB before the
bench does it for us.

### 3. Interfering machinery — no measured degradation in range

Second machine's hum at 73 Hz (shares no low harmonic with the 50 Hz shaft),
amplitude 0-4× the primary machine's own hum level.

| Interferer gain (×primary hum) | AUC | TPR | FPR | regimes found |
|---|---|---|---|---|
| 0.0 | 1.000 | 1.00 | 0.00 | 1 |
| 0.5 | 1.000 | 1.00 | 0.00 | 1 |
| 1.0 | 1.000 | 1.00 | 0.167 | **2** |
| 2.0 | 1.000 | 1.00 | 0.00 | 1 |
| 4.0 | 1.000 | 1.00 | 0.00 | 1 |

No measured breakdown even at 4× the primary machine's own hum level — the
73 Hz interferer sits far enough from the 50 Hz shaft harmonics and the
resonance band that the envelope/band features stay clean. The one blip is
at gain 1.0, where clustering happened to split the learn period into 2
regimes instead of 1, nudging FPR to 0.167 at that single point — consistent
with T1.9's finding that regime-count selection is the more fragile part of
this pipeline, not raw detection.

### 4. Resonance frequency — flat, but see the clamp caveat above

| Resonance (Hz) | AUC | TPR | FPR | contaminated? |
|---|---|---|---|---|
| 1000 | 1.000 | 1.00 | 0.00 | **true** |
| 2500 | 1.000 | 1.00 | 0.00 | false |
| 4500 | 1.000 | 1.00 | 0.00 | false |
| 6400 | 1.000 | 1.00 | 0.083 | false |
| 8000 | 1.000 | 1.00 | 0.00 | false |

Flat across the full tested range — but per the method note above, only the
1000 Hz and (nearly) 2500 Hz points are testing what their label says for
the accelerometer channel; 4500 Hz and above are increasingly re-testing the
same clamped 2560 Hz (accel) / 6400 Hz (audio) filter. The 1000 Hz point's
learn period was flagged **contaminated** by T1.6's guard (a single outlier
window inflated the empirical 99.5th percentile relative to the robust
fit) — the safer threshold was deployed automatically, as designed, and
detection was unaffected (AUC still 1.000).

## Honest summary

Of the four axes tested, **SNR is the only one where this sweep directly
measured a breakdown in range** (AUC 0.878, TPR 0.67, complete miss of the
mildest severity at -5 dB). **Mounting attenuation is the axis most likely
to matter on a real machine and the one this sweep least conclusively
cleared** — TPR/AUC held to 24 dB but the safety margin collapsed ~54× over
that range, meaning the true breaking point is plausibly just beyond what
was tested. Interference showed no measured degradation up to 4× the
primary machine's hum. Resonance frequency showed no measured degradation,
but roughly the top half of the tested range was confounded by the
simulator's own clamp and does not cleanly answer the question its label
asks.

None of this replaces H2/H3/H4 (see the handover notes (not in this public copy)): every number here
comes from `ml/simulate.py`'s and `ml/sensitivity/sweep.py`'s signal model,
not a real bearing, a real mount, or a real neighbouring machine. What this
sweep buys is a ranked list of what to worry about first when the bench
disagrees with the simulator — mounting attenuation and low SNR, in that
order — rather than finding out the ranking the hard way.

---

## T1.12 — the two severity scales are not the same knob (F18)

**Why this section exists.** F18 measured `ml/realdata/synth_phone_recording.py`
faults at severity 0.05/0.10/0.20 scoring **0.55–0.70x threshold against
their own correct baseline** (not detected), while every number above this
line, and every "detection down to severity 0.02" style claim elsewhere in
this repo, was measured on `ml/simulate.py`'s severity knob, which detects
comfortably below 0.02. Both modules call the parameter `severity`; nothing
before this measured whether it means the same thing. Two hypotheses were on
the table: (1) it is simply a ~10x rescaling, or (2) a realistic pink noise
floor masks faults a white floor does not. **Neither is quite right.**

**Method** (`ml/sensitivity/calibrate_severity_scales.py`, `tests/
test_severity_calibration.py`, 10 tests): stopped using `severity` as the
x-axis. Both generators excite a *known* structural resonance (an impulse
train through a resonance band-pass filter), so the same band-pass filter
was applied to each generator's healthy AND faulty signal and RMS compared
in dB — **band RMS re the healthy floor**, a physical energy measurement
common to both, not a knob value.

### Calibration curves: severity → band RMS dB re healthy floor

| `ml/simulate.py` severity | dB | | `synth_phone_recording.py` severity | dB |
|---|---|---|---|---|
| 0.02 | 3.9 | | 0.05 | 0.03 |
| 0.05 | 10.0 | | 0.10 | 0.13 |
| 0.10 | 15.7 | | 0.20 | 0.53 |
| 0.15 | 19.1 | | 0.35 | 1.48 |
| 0.20 | 21.6 | | 0.50 | 2.64 |
| 0.30 | 25.1 | | 1.00 | 6.42 |
| 0.40 | 27.6 | | 2.00 | 11.64 |
| 0.50 | 29.5 | | 5.00 | 19.35 |
| | | | 10.00 | 25.33 |

**Hypothesis 1 (constant ~10x factor) is FALSE.** A fixed multiplicative
severity ratio would show up as a fixed dB *offset* between the curves
(dB is logarithmic). Measured instead: comparing simulate severity 0.02 against
phone severity 0.20 (the "10x" pairing) gives an offset of 3.9 − 0.53 =
**3.4 dB**; comparing simulate 0.20 against phone 2.0 (the same "10x"
pairing) gives 21.6 − 11.64 = **9.7 dB**. The offset drifts by ~6 dB across
the tested range — not a constant, so there is no single rescaling factor
that reconciles the two knobs. `tests/test_severity_calibration.py::
test_the_two_severity_scales_are_not_related_by_a_constant_factor` pins this.

**Hypothesis 2, as literally stated ("pink masks faults"), is also not what
is happening.** Matching the two generators at nearly the same absolute dB
(mic-only detection, `firmware/baseline.py`/`inference.py`, real fit +
score, 16 learn windows, 6 held-out healthy + 6 held-out fault):

| Generator | severity | band RMS dB | FPR | TPR | AUC | fault ratio (median ×thr) |
|---|---|---|---|---|---|---|
| simulate (white) | 0.02 | 3.9 | 0.33 | 1.00 | 0.67 | **3.0×** |
| phone (pink) | 1.00 | 6.7 | 0.17 | 1.00 | 1.00 | **19.7×** |
| simulate (white) | 0.05 | 10.0 | 0.33 | 1.00 | 1.00 | **15.5×** |
| phone (pink) | 2.00 | 11.9 | 0.17 | 1.00 | 1.00 | **45.0×** |

At matched physical fault energy, the pink generator is detected **at least
as well as, and in this measurement better than**, the white one — the
opposite of what "pink noise masks faults" predicts. (FPR sits around
0.17–0.33 rather than the ~0.04 documented elsewhere because this table uses
only 16 learn / 6 test windows for speed — small-n noise, not a new finding;
see T1.6.)

**What is actually happening, a third answer neither hypothesis named:**
`synth_phone_recording.make_pair` deliberately adds a `shared_knock_ring` —
non-periodic knocks exciting the SAME resonance band — to **both** the
healthy and faulty signal, "so any periodicity found in the faulty-minus-
healthy comparison is genuinely the fault, not an artefact" (that module's
own docstring). `ml/simulate.py`'s healthy signal has no resonance-band
energy at all. That knock ring has a fixed amplitude (0.15, in the same
units as `severity`), so it acts as an **in-band noise floor specific to the
fault frequency**, not a generic broadband SNR effect:

| phone severity | band RMS dB | fault ratio (×thr) | TPR |
|---|---|---|---|
| 0.05 | 0.03 | 0.61 | 0.17 |
| 0.10 | 0.13 | 0.61 | 0.00 |
| 0.20 | 0.53 | 6.4 | 0.83 |
| 0.50 | 2.80 | 10.5 | 1.00 |
| 1.00 | 6.66 | 19.7 | 1.00 |

Detection is near-chance while `severity` (fault ring amplitude) is well
below the knock ring's own 0.15 amplitude, and recovers sharply once fault
energy starts to dominate the knock floor, between severity 0.1 and 0.5 —
matching the F18/T7.2 "severity-gated band-selector fallback" finding
exactly (crest 5–8.5 below this crossover, 13–20 above it): same mechanism,
seen from the Mahalanobis-score side rather than the crest side.
`tests/test_severity_calibration.py::
test_phone_detection_is_near_chance_below_its_own_knock_floor` and
`::test_phone_detection_recovers_above_its_own_knock_floor` pin both ends.

**What this means for every other claim in this repository:** "detection
down to severity 0.02" (this file, above) is a true, measured statement
about `ml/simulate.py`'s own severity scale, on a healthy signal with zero
resonance-band energy. It does not transfer to `synth_phone_recording.py`'s
severity scale — not because pink noise is a worse noise floor, but because
that generator's own healthy reference already contains resonance-band
energy the white generator's healthy signal structurally cannot. Any new
generator that adds fault-band energy to its OWN healthy signal (as any
honest recording of a real machine in a real room will) needs this same
calibration before its severity numbers can be compared to `ml/simulate.py`'s.

Reproduce: `python ml/sensitivity/calibrate_severity_scales.py curves` (the
dB tables) and `... detect-both` (the matched-dB detection table). Both are
fast (seconds per point); `pytest tests/test_severity_calibration.py` pins
the properties above as regression tests.
