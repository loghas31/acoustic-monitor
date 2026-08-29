# Acoustic Machine Health Monitor

Detecting mechanical faults in rotating machinery from sound, with no prior
data about the machine being monitored.

Python · NumPy/SciPy · scikit-learn · pytest · FastAPI · React · 680 tests

---

## The result

A desk fan, a stiff card taped so each blade strikes it, and a phone. The card
is a deliberately induced impulsive fault — the same class of signal a spalled
bearing produces. Recorded healthy, then faulted, then healthy again, at two
different fan speeds.

| | low speed | high speed |
|---|---|---|
| healthy | 1.9 | 3.1 |
| **card fitted** | **20.4** | **62.1** |
| healthy again | 5.3 | 2.1 |

The detector reported a fault frequency of **25.8 Hz** at low speed and
**31.5 Hz** at high speed, from the envelope spectrum, with no geometry, rpm or
baseline supplied.

**Independently**, the raw magnitude spectrum — a different transform, measuring
the aerodynamic blade-pass tone rather than mechanical impacts — gives 78.38 Hz
and 95.38 Hz for the same two recordings.

| | low | high | ratio |
|---|---|---|---|
| raw blade-pass tone | 78.38 Hz | 95.38 Hz | **1.2169** |
| envelope peak (detector) | 25.8 Hz | 31.5 Hz | **1.2209** |
| **raw ÷ envelope** | **3.038** | **3.028** | |

Two things follow, and the second is the one that matters:

1. **The ratio is 3.03 at both speeds**, against a blade count of **3**
   confirmed by inspection. The detector recovered the fan's shaft rotation
   rate from the fault signature alone.
2. **The two independent measurements scaled with fan speed to within 0.33%.**

That second row is what rules out the alternative explanations. Mains hum does
not change frequency when a fan is switched to high; room acoustics do not
scale by 1.22; a recording artefact does not land on an integer counted by
hand. Full protocol, raw numbers and stated limitations: **[RESULTS.md](RESULTS.md)**.

### Scored in 30-second windows: 20/20 detected, 0/40 false alarms

The numbers above are one score per recording. Re-scored in the 30-second
windows the detector actually runs on — 60 windows, 40 healthy across four
recordings, 20 faulted across two — the two populations do not overlap at all:

| | n | min | median | max |
|---|---|---|---|---|
| healthy | 40 | 2.64 | 5.89 | **13.52** |
| faulted | 20 | **67.76** | 110.50 | 197.24 |

Any threshold between 13.52 and 67.76 — a factor of five wide — gives **20/20
detection and 0/40 false alarms**.

### The result worth reading twice: an operating-state change is not a fault

Spotting a fault is the easy half. The hard half is not crying wolf when a
machine legitimately changes what it is doing, because false alarms are what
get a monitoring system switched off. A speed change looks a great deal like a
fault to a naive detector.

Set the threshold using **only the low-speed healthy windows**, then apply it
to data that threshold never saw:

| | n | flagged |
|---|---|---|
| healthy at **22% higher fan speed** | 20 | **0** |
| faulted, low speed | 10 | 10 |
| faulted, high speed | 10 | 10 |

The speed change moved the healthy score *down*, not up. Every faulted window
was still caught.

A second, independent discriminator falls out of the same run: across ten
windows the faulted peak frequency is pinned to **0.25 Hz** at high speed,
while healthy windows wander across the whole search range (8.25–50.75 Hz)
because there is no comb there to find.

Method, full per-window table and limitations: **[RESULTS.md](RESULTS.md)**,
Experiment 0c. Reproduce with `python tools/fan_windows.py data/`.

### Can you just hear it?

Yes — probably. A card striking three blades changes mass balance and airflow,
and a person in the room can hear the difference. So "the detector noticed" is
a weak claim and is deliberately not the headline here.

What survives that objection is everything a listener cannot do: recovering
25.65 Hz and 31.55 Hz as *numbers* with no geometry or rpm supplied, having
them agree with a completely different transform to about 1%, having their
ratio equal a blade count counted by eye, and holding a threshold across a
change of operating point. No ear does any of that.

### What this does not show

- **One machine, one room, one afternoon.** Two speeds, but a single device.
- The fault is **a card, not a bearing defect** — impulsive and the right class
  of signal, but not the same mechanism.
- Only the **baseline-free screen** was exercised, not the self-baselined
  detector described below.
- Nothing about sensitivity to early faults, or false-alarm rates over weeks.
- The healthy controls differ by up to 2.8×, which is the honest noise floor
  any claim here should be read against.
- **Forty healthy windows is not forty independent samples.** They come from
  four recordings; windows within one share the fan, the room and the
  microphone. Read it as the weight of four recordings.
- **The threshold above was chosen after seeing the data.** It shows a
  separation exists, not that this operating point is calibrated.

---

## How it works

**The physics.** A defect in a rotating machine strikes something once per
revolution of the defective part, exciting a structural resonance in the
1–20 kHz range. The impacts are periodic; the resonance is not the signal, the
*rhythm* of the impacts is. Extracting that rhythm means band-pass filtering
around the resonance and taking the envelope — demodulation — then looking for
periodicity in the envelope spectrum rather than the raw one. For a bearing of
known geometry the expected rate is computable:

$$\mathrm{BPFO} = \frac{N}{2} f_r \left(1 - \frac{d}{D}\cos\phi\right)$$

`ml/realdata/fault_frequencies.py` implements this and the related inner-race,
ball-spin and cage frequencies.

**Two detectors, answering different questions.**

*Self-baselined anomaly detection* (`firmware/`) learns what one specific
machine normally sounds like — 30-second windows, a 37-dimensional feature
vector, per-regime Gaussian models fitted with Ledoit–Wolf shrinkage, scored by
Mahalanobis distance — then flags departures. It needs no geometry and no model
number, but it requires a healthy learn period, and it goes silent on a machine
that was already faulty when it learned.

*Baseline-free screening* (`tools/cold_start_screen.py`) asks a question needing
no history at all: is there a periodic impact train here? It searches the
envelope spectrum for a harmonic comb. That covers the cold-start case, but only
sees impulsive faults — it is blind to imbalance and wear, which the
self-baselined path handles. They are complements.

**Why not a library of reference recordings per machine type?** Because it
cannot work, and that was measured rather than argued: score a *healthy* unit
against a different healthy unit of the same model and it reads 4.27× threshold
— condemned — while that unit's own genuine early faults read 0.55–0.70×.
Unit-to-unit variation is larger than the fault signal. There is no threshold
that both avoids false alarms and catches early faults. See F18 in
[DOC_SELF_REVIEW.md](docs/DOC_SELF_REVIEW.md).

---

## Run it yourself

No hardware, no recordings needed — about ten minutes:

```bash
pip install -r ml/requirements.txt
python ml/simulate.py --outdir data          # generate test signals
python tools/cold_start_screen.py --self-test
pytest tests/                                 # 680 tests
```

[RUN_IT.md](docs/RUN_IT.md) walks through the whole pipeline on synthetic
signals. [TESTS.md](TESTS.md) is the run sheet for testing it on a real machine
of your own. [FAN_EXPERIMENT.md](docs/FAN_EXPERIMENT.md) is the protocol for
reproducing the result above.

---

## The engineering record

[**DOC_SELF_REVIEW.md**](docs/DOC_SELF_REVIEW.md) is a register of 26 findings
against this project's own work — including the ones that were wrong, and the
ones that were retracted after measurement contradicted them. A sample:

- **F18** — the reference-library product idea, killed with a measurement
  after argument failed to settle it four times.
- **F20** — a proven false-alarm rate regressed from 0.000 to 0.107 and every
  check stayed green, because the number was written to a file and never
  asserted. The fix was to make it a tested claim.
- **F26** — the very first real recording landed in exactly the failure mode a
  synthetic finding had predicted 48 hours earlier: a fault modulating at shaft
  rate, sitting on the mains frequency, which an earlier version of the screen
  would have discarded as hum.

It is kept because a project that only records its successes cannot be
corrected, and because most of the useful decisions here came from a
measurement disagreeing with an assumption.

---

## Repository map

| | |
|---|---|
| `firmware/` | feature extraction, baseline learning, inference — runs on a Raspberry Pi in pure NumPy |
| `ml/` | signal simulation, evaluation, sensitivity analysis, bearing fault frequencies |
| `tools/` | the analysis command line: cold-start screen, one-command scan, experiment harnesses |
| `backend/` | FastAPI + SQLAlchemy service, MQTT bridge, alert dispatch |
| `frontend/` | React dashboard |
| `tests/` | 680 tests, run on every push |
| `docs/` | physics, pipeline, detector, alerting, sensitivity — and the findings register |

---

Logan Hastie · physics undergraduate, University of Edinburgh · 2026

AI tooling was used throughout, as an implementation aid and as a reviewer to
argue with. The architecture, the product decisions, the experimental design
and the calls about which results to trust are mine — several of the entries in
the findings register are cases where I pushed back on a confident answer and
the measurement proved it wrong.
