# Results — the lab notebook

This is where the hardware weeks (the execution plan (not in this public copy)) get written down
as they happen, not reconstructed from memory afterwards. Every other number
in this repository — `AUC 1.000`, `56.7×` envelope contrast, `0` false
alarms — comes from `ml/simulate.py`. This file is where the first numbers
that come from a real machine go, and it exists specifically so a real,
disappointing result doesn't get quietly smoothed over between the bench and
the pitch deck. Fill in a section the day you get its data, in whatever
state it's in — a `FAIL` with a real number is worth more here than a `PASS`
you're not sure you actually earned.

**Rules for filling this in**, same ones the rest of the repo follows:

- Every number gets the command that produced it. "It worked" is not a
  result; `python ml/realdata/analyse_recording.py --healthy ... → PASSED:
  True, contrast 6.2x` is.
- Label synthetic and real clearly. If you re-run something from
  `RUN_IT.md` to sanity-check the pipeline before trusting a real result,
  say so — don't let a synthetic number sit next to a real one with nothing
  to tell them apart.
- A `FAIL` or an `INCONCLUSIVE` gets a "why, and what would settle it" note,
  not just the verdict. `docs/DOC_STATUS.md`'s whole existence is proof that
  this project treats a disproved claim as more valuable than a silent one.
- Real recordings (`.wav`, `.csv`, `.mat`) never get committed to git —
  `CONTRIBUTING.md` covers why. Put them on Drive/OneDrive and paste the
  link in the relevant section below.

---

## Experiment 0 — Desk fan, induced impulsive fault (2026-08-29)

**The first real measurement in this project's history.** Everything else in
this repository, up to this date, came from signals the project generated
itself. No hardware, no purchases; a desk fan, a card and a phone.

### Protocol

Three recordings of the same 3-blade desk fan, same position, same room,
same session, in this order:

| # | Condition |
|---|---|
| 1 | healthy, nothing fitted |
| 2 | stiff card taped so each blade strikes it |
| 3 | card removed — reversibility control |

Recorded on iPhone Voice Memos, converted with `afconvert` to 16 kHz mono WAV.
Lengths 315.6 / 316.1 / 318.6 s, **trimmed to 315.6 s** by the tool, because
the comb score is not comparable across durations (T1.16 #8: the same fault
scores 33.0 at 20 s and 12.7 at 1 s).

```
python tools/fan_experiment.py "data/Healthy fan take 1.wav" \
    "data/Card in fan.wav" "data/Healthy fan take 2 (after card).wav" \
    --predict-hz 26.1
```

### Result — baseline-free screen (`tools/cold_start_screen.py`)

| condition | comb score | peak Hz |
|---|---|---|
| before (healthy) | **1.9** | 9.2 |
| **during (card)** | **20.4** | **25.8** |
| after (healthy) | **5.3** | 15.0 |

**10.7× separation** between the faulted and the first healthy recording, and
the score falls back when the card is removed.

### Shaft speed, measured independently from the raw spectrum

`--estimate-rpm` on each recording, reading the aerodynamic blade-pass tone
out of the ordinary magnitude spectrum — a different transform from the
envelope spectrum the detector uses:

| recording | dominant tone | ÷ 3 blades = shaft | rpm |
|---|---|---|---|
| healthy 1 | 84.38 Hz (4982× median) | 28.13 Hz | 1688 |
| healthy 2 | 87.62 Hz (4729× median) | 29.21 Hz | 1752 |
| **card** | **78.38 Hz** (566× median) | **26.13 Hz** | **1568** |

**The card slowed the fan by 8.9%** relative to the healthy mean (86.0 Hz) —
comfortably outside the 3.8% run-to-run variation between the two healthy
takes. Consistent with drag from a card rubbing the blades.

### The claim, stated precisely

**The strong, non-circular result:**

$$\frac{78.38\ \text{Hz (raw spectrum)}}{25.8\ \text{Hz (envelope spectrum)}} = 3.04$$

Two spectral features computed by different transforms, whose ratio equals the
blade count **counted visually off the fan**. The detector — given no geometry,
no rpm, no baseline and no prior recording of this machine — reported the fan's
shaft rotation rate from the fault signature alone.

That the impacts modulate at *shaft* rate rather than *blade-pass* rate is the
expected behaviour, not a discrepancy: three blades cannot strike a taped card
identically, so the impact pattern repeats once per revolution. Unequal impacts
per revolution produce a comb at shaft rate with a strong third harmonic, which
is exactly what a harmonic-sum score locks onto.

**The weaker framing, and why it is weaker.** The tool reports "peak matches
predicted 26.1 Hz — found 25.8 Hz". That 26.1 was obtained by dividing the same
recording's raw-spectrum tone by 3. It is therefore a **within-recording
consistency check between two transforms, not an a-priori prediction.** A truly
independent prediction would have used the healthy recording's rpm — which
would have *failed*, because the fan slowed under load. Say the former in a
write-up, not the latter.

---

## Experiment 0b — the same fault at a second fan speed (2026-08-29)

**The control that decides whether any of the above means anything.** A real
mechanical fault frequency scales with shaft speed. Room noise, mains hum and
recording artefacts do not. So the whole experiment was repeated at the fan's
high setting.

**The prediction was written down before the recordings were analysed:** both
the raw tone and the envelope peak must rise by the same proportion, and the
ratio between them must stay at ≈3.0, because the blade count cannot change.
Stated falsification: if the envelope peak stayed near 25.8 Hz while the fan
was audibly faster, the detector was tracking the room, not the machine.

### Result

| condition | comb score | peak Hz |
|---|---|---|
| before (healthy) | 3.1 | 16.5 |
| **during (card)** | **62.1** | **31.5** |
| after (healthy) | 2.1 | 33.2 |

### The two numbers that matter

| | low setting | high setting | ratio |
|---|---|---|---|
| raw blade-pass tone, card fitted | 78.38 Hz | 95.38 Hz | **1.2169** |
| envelope peak (detector output) | 25.8 Hz | 31.5 Hz | **1.2209** |
| **raw ÷ envelope** | **3.038** | **3.028** | — |

**1. The ratio is 3.03 at both speeds**, against a blade count of 3 confirmed
visually. It did not drift when the operating point changed.

**2. The two channels scaled by 1.2169 and 1.2209 — agreeing to 0.33%.** The
raw magnitude spectrum and the envelope spectrum are different transforms
measuring different physics (aerodynamic tone vs mechanical impacts). Speeding
the fan up moved both by the same factor to within a third of a percent.

Implied shaft speed with the card fitted: **1568 rpm** (low), **1908 rpm**
(high). Both physically sensible for a desk fan. The 190.62 Hz line at 1894×
median is 2 × 95.31 Hz — the second harmonic, confirming 95.38 Hz is the
fundamental rather than itself a harmonic of something lower.

### What Experiment 0b closes

The strongest objection to Experiment 0 was that the detector had locked onto
something in the room rather than the fan — and the fact that 25.8 Hz was
flagged as a mains sub-harmonic (2 × 25.8 ≈ 50) made that a live concern rather
than a pedantic one.

**Mains hum does not change frequency when a fan is switched to high.** The
detector's output moved 22%, in lockstep with an independent measurement, to a
ratio fixed by a physically counted integer. The room-noise hypothesis is dead.

Secondary observations:

- The effect is **3× stronger at high speed** (62.1 vs 20.4) — consistent with
  faster blades striking the card harder.
- **The control was tighter this time**: 3.1 → 2.1, against 1.9 → 5.3 at low
  speed. That retrospectively makes the low-speed 5.3 look like a single noisy
  recording rather than a systematic drift, though with n = 2 pairs that is an
  observation and not a conclusion.
- At high speed the peak is **not flagged** as mains-coincident, because
  31.5 Hz is not near 50/k for any integer k.

---

### What both experiments together establish

- The detector responds to a genuine mechanical fault on a real machine, in a
  real room, with real background noise.
- The response is **reversible** — removing the fault removes the signal.
- The frequency it reports **tracks the machine's shaft speed across two
  operating points**, at a ratio fixed by the blade count, verified against an
  independent measurement from a different transform.
- It does this with **no geometry, no rpm, no baseline and no prior recording
  of the machine supplied**.

### What this does NOT establish

*(Updated after Experiment 0b. The "one fan speed" limitation below is now
retired; everything else stands.)*

- **n = 1 machine.** One fan, one room, one afternoon. Two speeds, but one
  device — nothing here says anything about a second unit, let alone a
  different machine type.
- **The fault is a card, not a bearing defect.** Impulsive, and the right class
  of signal, but not the same mechanism as a spalled race.
- **Only the baseline-free screen was exercised.** The self-baselined detector
  (`fridge_scan.py`) — the actual product — was not, because that needs ≥25
  minutes of continuous audio with a healthy learn period.
- **Nothing about sensitivity.** This fault was loud. The severity floor
  remains unmeasured on real audio.
- **Nothing about false alarms over time.** That is Gate 3 and needs a week.
- **The healthy control drifted**: 1.9 → 5.3, a 2.8× change between two
  recordings that should be identical, alongside a 3.8% shift in fan speed.
  So run-to-run variability on this machine is roughly 3×, and the 10.7×
  separation should be read against that, not against zero.

### Finding: the mains-coincidence flag false-alarmed on real data

The tool flagged 25.8 Hz as "coincides with mains 50 Hz", because 2 × 25.8 ≈ 50.
It is not mains; it is the shaft rate. This is the **same failure mode fixed
earlier the same day** on `data/bearing_inner.wav`, where an inner-race fault
modulated at the shaft rate sat exactly on the mains frequency and an earlier
version of the screen *discarded* it. That fix — flag but never remove — is
what allowed this result to be seen at all. Logged as **F25** in
`docs/DOC_SELF_REVIEW.md`.

### Artefacts

Recordings in `data/` (gitignored, not committed). Photographs of the fan and
the card in place: `IMG_0251/0252/0253.HEIC`. **Keep these** — they are the
methods-section figures and cannot be reconstructed later.

---

## Experiment 0c — the same six recordings, scored in 30-second windows (2026-08-29)

**No new recordings. A re-analysis, because Experiments 0 and 0b could not
answer the question anyone competent asks second.**

0 and 0b produced one score per recording: six numbers. That supports
"10.7x separation" and nothing else. A **detection rate** and a **false-alarm
count** are properties of a population of decisions against a threshold, and
six numbers with no threshold defined is neither. `tools/cold_start_screen.py`
does not define one, so both quantities were, strictly, undefined.

### Method

Each recording split into non-overlapping **30-second windows** — the window
length `firmware/` itself learns and scores on, so this is a result about the
deployable thing rather than a five-minute laboratory convenience. Only the
first 300 s of each recording is used, so every condition contributes exactly
ten windows and none is weighted more heavily for having run longer. Every
window is the same length, because the comb score is not comparable across
durations (T1.16 #8). Normalisation is per recording, as in
`fan_experiment.load`, so no window is silently re-gained.

```
python tools/fan_windows.py data/ --out docs/fan_window_scores.csv
```

**60 windows: 40 healthy (four recordings, two fan speeds), 20 faulted (two
recordings, two fan speeds).** Per-window scores: `docs/fan_window_scores.csv`.

### Result — separation

| | n | min | median | max |
|---|---|---|---|---|
| healthy | 40 | 2.64 | 5.89 | **13.52** |
| faulted | 20 | **67.76** | 110.50 | 197.24 |

**The distributions do not overlap.** Not one of the 20 faulted windows scores
below any of the 40 healthy ones. The worst-case margin — lowest faulted window
against highest healthy window — is **5.01x**; on medians it is 18.8x.

Any threshold in the open interval (13.52, 67.76) therefore gives:

- **detection: 20/20 faulted windows, 100%**
- **false alarms: 0/40 healthy windows, 0%**

That interval is a factor of five wide, so the result is not sensitive to where
in it the threshold is placed. **It was chosen after seeing these data**, which
makes this a separation result, not a validated operating point — the honest
statement is that a threshold exists, not that this threshold is calibrated.

### Result — the operating-state test, which is the interesting one

The hard problem in condition monitoring is not spotting a fault; it is not
crying wolf when a machine legitimately changes operating state. A speed change
looks like a fault to a naive detector, and false alarms are what get a
monitoring system switched off.

The two fan speeds make that testable at no extra cost. **Set the threshold
from the low-speed healthy windows alone** (max = 13.52), then apply it to
recordings the threshold never saw:

| population | n | windows above the low-speed healthy threshold |
|---|---|---|
| healthy, **high** speed — a 22% operating-point change | 20 | **0** |
| faulted, low speed | 10 | **10** |
| faulted, high speed | 10 | **10** |

**Zero false alarms from the speed change; every faulted window still caught.**
Increasing the fan speed by 22% moved the healthy score *down* (low-speed
healthy 2.64–13.52, high-speed healthy 2.75–6.51), not up toward the fault
population.

Note what this does and does not test. `cold_start_screen.py` is baseline-free:
it has no notion of a regime because it has no baseline to depart from, so it
cannot fail this test the way a poorly-regimed anomaly detector would. What the
result establishes is narrower and still worth stating — a legitimate change of
operating point did not generate a fault-level score. **The per-regime
Mahalanobis path in `firmware/` — the design that exists specifically to handle
this — is still untested on real audio.** That needs a continuous recording with
a healthy learn period, and remains the single highest-value experiment
outstanding.

### Result — frequency stability as a second, independent discriminator

The score is not the only thing that separates the two populations. The
frequency the screen locks onto does too:

| population | peak Hz across its windows | spread | sd |
|---|---|---|---|
| faulted, low | 24.75 – 26.25 | 1.50 Hz | 0.50 |
| faulted, high | 31.50 – 31.75 | **0.25 Hz** | **0.10** |
| healthy, low | 9.25 – 15.25 | 6.00 Hz | 1.64 |
| healthy, high | 8.25 – 50.75 | **42.50 Hz** | 10.38 |

A real periodic impact train pins the peak to a quarter of a hertz across ten
independent windows. Healthy windows wander over the whole search range,
because there is no comb there and the screen is ranking noise. **A detector
could use agreement-across-windows as a confirmation rule and would have made
the same call here** — that is a genuinely different piece of evidence from the
score, not a restatement of it.

Mean faulted peak: 25.65 Hz low, 31.55 Hz high, **ratio 1.2300**. The raw
blade-pass tone ratio from Experiment 0b is 1.2169. The two channels agree to
**1.08%** on this estimator. (0b's whole-recording figure was 0.33%; the
per-window mean is a different estimator and the larger number is the one to
quote alongside it, not instead of it.)

No window was flagged for clipping, and none had a degenerate demodulation band.

### What Experiment 0c does NOT establish

- **Forty healthy windows is not forty independent samples.** Ten windows cut
  from one continuous recording share the fan, the room, the microphone and its
  position — every nuisance variable there is. This is the evidential weight of
  **four healthy recordings**, not forty. The correct sentence is "zero false
  alarms across 40 windows drawn from four healthy recordings at two fan
  speeds", and a reader should discount it accordingly. A false-alarm *rate* in
  any population sense is still Gate 3: many machines, many rooms, over weeks.
- **The threshold is post hoc**, chosen from these data. See above.
- **The fault is audible.** A card striking three blades changes both mass
  balance and airflow, and a person in the room can hear it. So "the detector
  noticed" is a weak claim and should not be the headline. The claims that
  survive the audibility objection are the ones a listener cannot make:
  recovering 25.65 Hz and 31.55 Hz as *numbers*, having them agree with an
  independent transform to ~1%, and having their ratio equal a visually counted
  blade count. No ear does that.
- Everything in Experiment 0b's limitations section still stands: n = 1
  machine, a card rather than a bearing defect, nothing about sensitivity to
  early faults, nothing about weeks of operation.

---

## Week 1 — Bring-up (Gate 1)

**Gate 1**, from the execution plan (not in this public copy): real audio and vibration land in a
file `features.py` can read, with a correct tone peak and a sane gravity
reading.

Date: `<fill in>`

| Check | Command | Result |
|---|---|---|
| Microphone alive, correct rate | `python firmware/bench/check_audio.py --tone 1000` (play a 1 kHz tone at the mic while it runs) | `<PASS/FAIL, measured rate, peak Hz>` |
| Accelerometer WHO_AM_I | `python firmware/bench/check_accel.py` | `<0x__ (expect 0x7B)>` |
| Gravity magnitude, board flat | same | `<__ g (expect 1.0 ± 0.1)>` |
| Achieved sample rates (measured, not configured) | same tools, counting samples over 60 s | audio `<__ Hz>` / accel `<__ Hz>` |
| Full bring-up self-test | `python firmware/bench/selftest.py` | `<PASS / first failing stage>` |

**Housing resonance (feeds Week 2's demodulation band):**

```
python firmware/bench/check_mount.py --taps 5
```

Resonance found: `<__ Hz>`, Q: `<__>`. Compare against the 3–6 kHz default
`features.py` falls back to — if the real resonance sits outside that range,
say so here; `docs/PHONE_RECORDING.md` already found this exact failure mode
on synthetic pink noise and it is the single most likely reason Week 2 comes
back inconclusive.

**Replace the invented accelerometer axis model.** `firmware/capture.ACCEL_AXES`
is a physically plausible guess (`docs/DOC_STATUS.md` says so explicitly).
Once `check_mount.py` gives you real f0/Q per axis, put the measured numbers
here directly — `tools/accel_axis_report.py` is simulate-only (no CLI for a
real recording; checked while writing this template, not assumed), so there
is no ready-made command for this one. A quick substitute that needs no new
tool: load the three-axis accel CSV `tools/ingest.py` wrote and compute
`np.corrcoef` between the three columns yourself, and compare against the
values `python tools/accel_axis_report.py` prints for the simulator (r =
+0.04 / −0.68 / +0.51 as of T1.8). If a real housing needs this often,
that's the justification for turning `accel_axis_report.py`'s existing
analysis into a proper tool with a `--recording` flag — note it in
the task backlog (not in this public copy) Tier 4 rather than reinventing it here each time.

Measured f0/Q per axis: `<fill in>`. Measured inter-axis correlation:
`<fill in>`. Compare against the simulated values in `docs/DOC_STATUS.md`'s
"Assumed, not proven" table and update that row once you have.

**What broke this week:** `<fill in, however small — wrong pin, wrong
overlay, a driver that needed a reboot. Future-you troubleshooting the same
thing wants this more than the parts that worked first try.>`

---

## Week 2 — The go/no-go experiment (Gate 2)

The question this answers: **does the envelope-spectrum signature appear on
a real bearing with a real seeded defect?** This is the execution plan (not in this public copy)'s
kill-gate. If this fails and you can't explain why within a further week, the
sensing chain is wrong and the product needs rethinking, not more code.

Date: `<fill in>`
Machine: `<make/model, or "second-hand induction motor, source">`
Bearing: `<designation, e.g. 6204 — or dimensions if not a standard part>`
Shaft speed: `<__ rpm, MEASURED — how: tachometer / strobe / stopwatch-and-mark>`
Defect: `<Dremel groove, ~__ mm wide (method: the test-rig notes (not in this public copy) §3.4) / run-to-failure>`
Race: `<outer / inner>`

Recording links (Drive/OneDrive, not committed): healthy `<link>` · faulty `<link>`

**Command:**

```
python tools/ingest.py healthy_raw.wav --out-dir data/real --stem healthy --rpm <measured>
python tools/ingest.py faulty_raw.wav  --out-dir data/real --stem faulty  --rpm <measured>
python ml/realdata/analyse_recording.py \
    --healthy data/real/healthy.wav --faulty data/real/faulty.wav \
    --bearing <designation> --rpm <measured>
```

(`--stem` is only valid ingesting one file at a time — verified this
template by ingesting a pair without it: with two inputs and no `--stem`,
each output keeps its OWN source filename as its stem, e.g.
`data/real/healthy_raw.wav`, not `healthy.wav`. Either name your source
recordings `healthy_raw`/`faulty_raw` and adjust the `--healthy`/`--faulty`
paths above to match, or run `--stem` twice as shown.)

(add `--mic-only` if the accelerometer isn't wired yet — see Week 1's
fallback note in the execution plan (not in this public copy))

| Metric | Value |
|---|---|
| Demodulation band used | `<__–__ Hz>` (auto-selected / forced with `--band`) |
| Envelope ratio, faulty (ABSOLUTE condition) | `<__x>` |
| Envelope ratio, healthy (control) | `<__x>` |
| Contrast (faulty / healthy) | `<__x>` |
| Raw-spectrum ratio, faulty | `<__x>` (compare against the envelope ratio — this is the "why demodulate" evidence, on real data) |
| Peak location vs computed BPFO | `<__ Hz measured vs __ Hz predicted, __% slip>` |
| **Verdict** | `<PASSED / FAILED / INCONCLUSIVE>` |

**Figures** (from the same command, `--out <dir>` — or attach manually):
raw-spectrum comparison, envelope-spectrum comparison. `<paste / link>`

**If FAILED or INCONCLUSIVE**, work through in this order before concluding
"no signature" — `docs/PHONE_RECORDING.md` measured all three of these
mattering on synthetic data first:

1. Was the demodulation band right? Re-run with `--band` forced to Week 1's
   measured resonance rather than the auto-selected one.
2. Was the geometry/rpm right? A miscounted ball count moves BPFO by
   ~12 %, four times the slip search window (`docs/DOC_STATUS.md`,
   T1.3's finding) — recount, don't just re-measure rpm.
3. Is this bearing's BPFO too close to a shaft harmonic to resolve at all?
   Check with `python ml/realdata/fault_frequencies.py --bearing <X> --rpm <Y>`
   before assuming the experiment failed rather than the bearing choice.

**Decision:** `<Gate 2 passed → proceed to Week 3. Gate 2 failed, explained
by __ → fix and re-run before Week 3. Gate 2 failed, unexplained → this is
the the execution plan (not in this public copy) kill-criterion; say so plainly rather than quietly
moving on.>`

**What broke this week:** `<fill in>`

---

## Week 3 — The false-alarm number (Gate 3)

The number that actually decides whether anyone keeps this plugged in.
**Gate 3: ≤ 1 false alarm per node-week.** Detection is the easy half; this
is the half nobody else in the student-project space measures.

Date started: `<fill in>` · Date ended: `<fill in>` (7 days untouched)
Machine: `<a fridge compressor / HVAC unit / fume hood — something boring and continuous>`

```
python tools/soak_report.py --db <device state db> --outdir results/soak_week3
```

This writes `soak_report.md` + figures directly — link or paste the summary
here rather than retyping the numbers by hand:

| Metric | Value |
|---|---|
| False alarms observed | `<__>` over `<__>` node-days |
| Rate (per node-week) | `<__>` |
| Poisson upper bound (95% confidence) | `<__>` — zero alarms in one week does not prove the rate is safely below one; this is the number that actually makes that claim |
| Regime switches during the soak | `<__>`, `<__>` false alarms attributed to them |
| Contamination flags fired | `<__>` (see `docs/DOC_STATUS.md`'s "learn-period contamination" residual — a real one here is data, not noise) |
| **Gate 3** | `<PASS (≤1) / FAIL (__, tune persist_minutes and re-run)>` |

If above target, the two knobs the execution plan (not in this public copy) names, in order: raise
`persist_minutes`, then check whether the learn period missed an operating
regime this machine actually has (`docs/DOC_STATUS.md`'s regime-clustering
findings are the place to start reading before assuming the model is wrong).

**Within-regime wander, while you have the data** — this settles a residual
risk `docs/DOC_STATUS.md` has flagged as unmeasured since T1.9: plot the
operating point (`baseline.operating_point`) per window and compare its
spread, within one regime, against `MIN_REGIME_SEPARATION`'s 5%-speed /
0.1-decade scale. `<fill in: comparable / much smaller — the constants hold /
much larger — MIN_REGIME_SEPARATION needs re-deriving from this measurement>`

**What broke this week:** `<fill in — this is also where "the Pi rebooted
on day 4 and I don't trust the DB after that" goes, honestly, not hidden>`

---

## Week 4 — Demo and wrap-up

- [ ] `systemd` unit installed (`firmware/acoustic-monitor.service`),
  survives a reboot: `<verified how>`
- [ ] Backend deployed, one node reporting to it live: `<URL / how to view>`
- [ ] Demo video recorded (healthy → swap bearing → alert fires):
  `<link, duration>`
- [ ] This file complete for Weeks 1–3

**One-paragraph honest summary**, for the pitch deck and for future-you —
what this sprint actually proved, in the same register as the system overview (not in this public copy)
§7 ("proven by execution" vs "not proven"), not the optimistic version:

`<fill in>`

**Kill-criteria check** (the execution plan (not in this public copy), decided in advance while
unattached to the outcome — resist re-litigating them now):

- Week 2 signature appeared and was explainable: `<yes/no>`
- Week 3 false alarms ≤ ~3/node-week after tuning: `<yes/no>`
- (Month 3, separate from this sprint) at least one site said yes to a free
  pilot: `<fill in later>`

---

## Data & artefacts index

Every real recording and raw state DB referenced above, in one place so
nothing gets orphaned on a laptop:

| What | Where | Referenced in |
|---|---|---|
| `<healthy_raw.wav>` | `<Drive/OneDrive link>` | Week 2 |
| `<faulty_raw.wav>` | `<Drive/OneDrive link>` | Week 2 |
| `<soak week3 state.db>` | `<link>` | Week 3 |
| `<demo video>` | `<link>` | Week 4 |
