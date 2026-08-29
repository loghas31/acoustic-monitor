# COLD_START.md — detecting a fault on a machine you can't assume is healthy

## The problem

The detector is self-baselining: it learns what a machine normally sounds like,
then flags departures. That design is not negotiable — F18 measured why. Score
a healthy unit against a *different* healthy unit's baseline and it reads
**4.27× threshold**, i.e. condemned, while unit A's own real faults read
0.55–0.70×. Unit-to-unit variation is larger than the fault signal, so a shared
reference library is impossible, not merely hard.

But self-baselining has a hole, and Logan named it exactly:

> nobody installs this on a brand-new machine. They install it *because* they
> are worried about an old one.

**If the machine is already faulty while it learns, the fault becomes part of
"normal" and the detector goes silent permanently.** The detector measures
change; a steady-state existing fault produces no change.

This document is the honest survey of ways round it, with what has been
measured and what has not.

---

## 1. Physics: look for the fault directly, with no history — ✅ BUILT AND MEASURED

**`tools/cold_start_screen.py`.**

A healthy rotating machine is broadly aperiodic. Nothing strikes anything
rhythmically. A spalled bearing, chipped gear tooth or broken rotor bar impacts
*once per revolution of something*, producing a comb of harmonics in the
envelope spectrum at f0, 2f0, 3f0…

That comb needs no baseline. It is a property of the recording, not of the
machine's history. So: demodulate, take the envelope spectrum, search every
plausible fundamental for a comb, reject the ones that are legitimate machine
periodicity, report the rest.

### What it actually does, measured

6 machines, 20 s, 16 kHz, no baseline and no bearing geometry supplied:

| fault severity | healthy score | faulty score | separated |
|---|---|---|---|
| 0.35 (advanced) | 6.6 | **11.9** | **6/6** |
| 0.20 (moderate) | 6.6 | 5.8 | 0/6 |
| 0.10 (early) | 6.6 | 6.6 | 2/6 |

Longer recordings are averaged window-by-window before the comb search, because
a real fault sits at the same frequency in every window while noise does not —
so the comb adds coherently and the floor falls as roughly √N. On 3-minute
recordings this locked the reported fundamental to 74 Hz on 6/6 machines and
widened the healthy-to-faulty contrast from 1.8× to 2.6×. It did **not** lower
the severity floor (see below). It is another reason to record 40 minutes.

**The 6/6 row is the real result, and it is stronger than the number suggests:
the fundamental it reports is 73–74 Hz, and the true BPFO for that bearing is
73.65 Hz.** It is not separating healthy from faulty by some incidental
artefact — it finds the actual bearing fault frequency, from one recording,
having never been told the bearing type or the shaft speed. That is pinned in
`tests/test_cold_start_screen.py`.

### What it cannot do

**It does not find early faults.** At severity 0.20 it picks 36.8 Hz and
18.2 Hz — exact sub-harmonics of the true 73.65 Hz. Two principled fixes were
tried and measured:

| scoring statistic | sev 0.35 | sev 0.20 | sev 0.10 |
|---|---|---|---|
| arithmetic mean (shipped) | 6/6 | 0/6 | 2/6 |
| minimum across harmonics | 0/6 | 0/6 | 2/6 |
| geometric mean | 6/6 | 0/6 | 2/6 |

`min` destroys the result it was meant to protect. Neither recovers 0.20. So
the sub-harmonic pick is a *symptom*, not the disease — at that severity the
comb simply is not reliably above the noise. This agrees with F18, where
severity-0.20 faults scored 0.55–0.70× against their own correct baseline: the
self-baselined detector misses them too, on this generator. **The ceiling is the
signal, not the statistic.** Do not retune it hoping to reach early faults.

Also blind to: imbalance, misalignment, worn seals, gas-charge loss, a tired
compressor — anything that changes level or resonance without impacting. The
two detectors are complements. Neither replaces the other.

### Honest summary

This answers **"is this machine already obviously broken?"** — which is exactly
the question an owner of an old machine is asking. It does not answer "is a
fault developing?", which remains the self-baselined detector's job and needs a
clean learn period.

---

## 2. The commercial reframe: the constraint is weakest where it matters most

**This is the most important point in the document and it costs nothing to
exploit.**

Logan's fridge is a sealed unit. He does not know the compressor's bearing
designation, and he cannot know its shaft speed. That is the *hardest possible*
case for physics-based detection.

An industrial customer is the opposite. A maintenance engineer can tell you the
motor is a 6205 bearing running at 1450 rpm, because it is on the nameplate and
in the maintenance record. Give the tool that, and it stops blindly searching
300 candidate frequencies and instead tests the ~4 frequencies where a fault
*must* appear (BPFO, BPFI, BSF, FTF — all already computed by
`ml/realdata/fault_frequencies.py`). Testing 4 known frequencies is enormously
more sensitive than searching 300 unknown ones, because the multiple-comparisons
penalty vanishes.

**So the cold-start problem is severe for consumer appliances and mild for the
actual target market.** That is a fortunate asymmetry and the product should be
positioned to use it: *ask the customer for the nameplate*. It is one field in
an install form and it costs nothing.

### ⚠ RETRACTION — "expect it to be large" was wrong, and measured wrong

The first version of this section ended "Untested, and the obvious next step:
quantify how much sensitivity is gained by supplying geometry versus searching
blind. **Expect it to be large.**" It was then measured, same day, and the
expectation did not survive. Targeted comb scoring at the exact known BPFO
versus blind search over 8–300 Hz, 6 machines, 3-minute recordings:

| severity | targeted at known BPFO | blind search |
|---|---|---|
| 0.35 | 6/6 | 6/6 |
| 0.20 | 3/6 (chance) | 2/6 |
| 0.10 | 3/6 (chance) | 1/6 |

**Knowing the exact fault frequency does not rescue a fault the blind search
misses.** At severity 0.20 the targeted score is 1.6 for healthy and 1.6 for
faulty — identical. This is not a search problem. The energy is not there.

### Why it is not there, and why that is NOT a verdict on the method

`synth_phone_recording.make_pair` contains this, by deliberate design:

```python
shared_knock_ring = 0.15 * _ring(n, fs, resonance_hz, q, rng, knock_samples)
healthy = floor + hum + room + shared_knock_ring
```

**The generator's HEALTHY signal already contains impact energy at amplitude
0.15, at the same resonance as the fault.** That is a deliberately conservative
choice — its comment says it is there "so any periodicity found is not an
artefact of 'faulty has knocks and healthy doesn't'" — and it is the right call
for avoiding self-flattery. But it means a severity-0.20 fault is competing
against 0.15 of impact energy already in the reference. The detection cliff
between 0.20 and 0.35 tracks that 0.15, and T1.12 independently found the same
thing: detection is "near-chance below that 0.15 floor and recovers sharply
above it".

**So the honest position is that this generator cannot resolve the question
below severity ~0.35 at all, for any method.** The geometry advantage is
therefore *unmeasured*, not disproved — and so is the true sensitivity floor of
the screen itself. Both need real audio: DCASE (labelled, free, immediate) or
the fridge test.

The commercial argument for asking for the nameplate still stands on its own
merits — it converts "periodic impacting at 87 Hz" into "outer-race defect",
which is the difference between an alert and a work order. Just do not claim it
buys sensitivity until someone has measured it on real data.

---

## 3. Sell what it actually does: deterioration from install

Even on a machine of unknown health, learning today and watching from today
catches everything that *gets worse* from today. Faults progress — that is the
premise the entire persistence gate rests on.

This is what most condition-monitoring vendors genuinely sell, and it is honest.
The claim is "we will tell you when this machine changes", not "we will tell you
whether this machine is healthy". A customer who understands that distinction is
not disappointed by it.

Combine with §1 at install time and the pitch is clean:

> On day one we screen for advanced faults directly. From day one onward we
> watch for anything getting worse. What we cannot do is certify a machine
> healthy today.

---

## 4. Load modulation as a probe — UNTESTED, cheap, worth trying

Make the machine change state and compare it against *itself* in the other
state. A fridge with the door held open runs its compressor much harder.

The reasoning: a healthy machine's features should scale roughly predictably
with load. A fault's impact energy typically grows *disproportionately* — more
load means harder impacts. So the ratio between the loaded and unloaded state
carries fault information **without needing any healthy history**, because the
contrast is internal to a single session.

This is genuinely testable tonight with no hardware: record 20 minutes normal,
20 minutes with the door ajar, compare. It is also the safe half of
`FRIDGE_TEST.md` Part B, so it costs nothing extra.

**Status: hypothesis. Not measured. Do not quote it as a capability.**

---

## 5. Self-contrast within one recording — UNTESTED

Many real faults are intermittent or load-dependent, so a faulty machine spends
*some* windows near its own healthy behaviour. Using a low percentile of the
recording as a pseudo-healthy reference and scoring the rest against it gives
internal contrast with no history.

Fails completely on a uniformly, constantly faulty machine. Worth measuring
because it is nearly free to implement, but it is not a general answer.

---

## 6. Robust baseline learning — PARTIALLY BUILT ALREADY

`firmware/baseline.py` already computes a `threshold_contaminated` flag per
regime, and `main()` prints a warning when any regime's learn period looks
contaminated. That is a real, existing partial defence that nobody had connected
to this problem: **the system can already tell you when its own learn period
looks wrong.**

The obvious extension is to fit the baseline robustly — to the majority of learn
windows, treating outliers as suspect rather than as normal. This helps a
partly-faulty machine (intermittent fault during learning). It does nothing for
a uniformly faulty one, because then the fault *is* the majority.

Worth doing, bounded value. The existing contamination flag should at minimum
be surfaced in `fridge_scan.py`'s output, where a user will actually see it.

---

## What to do about it, in order

1. **Run the cold-start screen at install, always.** Built, measured, free.
   Catches advanced faults on a machine of unknown history.
2. **Ask for the nameplate.** Bearing designation and rpm turn a blind search
   into a targeted test. One form field; largest single sensitivity win
   available.
3. **Be honest in the pitch.** "Detects deterioration from install, plus a
   day-one screen for advanced faults." Do not claim health certification.
4. **Surface the existing contamination flag** in user-facing output.
5. **Measure the load-modulation probe** (§4) — cheapest untested idea with the
   highest upside.

## What NOT to do

- **Do not build a reference library.** F18 measured it dead: 4.27× on a healthy
  unit. It has now come up five times; the measurement settles it.
- **Do not retune the comb scoring to chase severity 0.20.** Three statistics
  measured, the ceiling is the signal.
- **Do not claim this certifies a machine healthy.** It cannot, and that claim
  is the one that would lose a customer permanently.

---

Related: `docs/DOC_SELF_REVIEW.md` F18 (the library measurement),
`ml/realdata/fault_frequencies.py` (BPFO/BPFI/BSF from geometry),
`tools/cold_start_screen.py`, `TESTS.md`.
