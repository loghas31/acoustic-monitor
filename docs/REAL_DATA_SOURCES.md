# Real data you can get without owning a machine

Written 2026-08-21, in answer to "could you access any recordings online?" —
yes, and one of them is a better first test than anything you can record.

Everything here is free, and all of it is real machines. None of it needs a Pi.

---

## 1. MIMII — real industrial machines, microphone-recorded, labelled faulty 🎯

**[zenodo.org/records/3384388](https://zenodo.org/records/3384388)** · DOI
[10.5281/zenodo.3384388](https://doi.org/10.5281/zenodo.3384388)

Verified from the Zenodo record, 2026-08-21:

| | |
|---|---|
| Machines | valves, **pumps**, **fans**, slide rails — 4 product models each |
| Normal audio | **5,000–10,000 seconds per model** (83–166 min) |
| Anomalous audio | ~1,000 s per model |
| Fault types | contamination, leakage, **rotating unbalance**, rail damage |
| Sample rate | **16 kHz, 16-bit** — *exactly this project's design rate* |
| Mic distance | 50 cm from pump/fan (8-channel array) |
| Background | **real factory noise**, mixed at three SNRs: **+6, 0, −6 dB** |
| Licence | **CC BY-SA 4.0** — usable and citable, with attribution |

**Why this beats a phone recording of your fridge as the first test.** It is
the actual product question — mic-only, real machine, real factory noise, learn
from normal and score the rest — with a *labelled answer* attached. Your fridge
has no known fault, so the best it can tell you is "the detector didn't cry
wolf". MIMII can tell you whether the detector *finds* something real.

**Why it beats CWRU too**, for this specific project: CWRU is accelerometer
data from a lab test rig, and it publishes **no licence** (only "© Case Western
Reserve University"), which is why `.gitignore` blocks committing it. MIMII is
**microphone** data — the channel this product actually ships — recorded at
**your exact sample rate**, in a **factory**, under **CC BY-SA 4.0**. For a
dissertation, a properly-licensed citable dataset is worth a great deal.

**And it tests the one weakness your own sweep found.** `DOC_SENSITIVITY.md`
records SNR as the axis that visibly breaks: AUC 1.000 → 0.878, TPR → 0.67, and
0 of 5 mildest faults caught at −5 dB. MIMII ships **real factory noise at −6
dB**. That is a direct, real-world test of the exact failure mode the synthetic
sweep predicted — not a proxy for it.

### Practical notes before you download

⚠️ **The full dataset is 100 GB. Do not download it.** Files are per
machine-type per SNR, 7–11 GB each. **Take one: `6_dB_pump.zip` (7.7 GB).**
Highest SNR is the easiest case, and you should establish that the pipeline
works at all before making it hard. Pumps and fans are rotating machinery with
bearings; valves and sliders are not, and are a poor fit for this detector.

⚠️ **The audio comes in 10-second clips, and this pipeline needs 30-second
windows with at least 48 of them** (24 min) to learn a baseline —
`DOC_STATUS.md` measured that fewer gives a 55–59 % held-out false-alarm rate,
i.e. noise. So the clips **must be concatenated** before `tools/ingest.py` sees
them, and only channel 0 of the 8-channel array is needed.

**That glue now exists and has been executed:** `tools/dcase_eval.py`, verified
2026-08-23.

Two things worth knowing before you download anything:

- **It has a `--self-test` that needs no download.** It builds a tree with
  DCASE's exact layout from `ml/simulate.py` and runs the real learn→score
  path. Run `python tools/dcase_eval.py --self-test` first — 90 seconds, and
  it tells you the machinery works before you spend a gigabyte. Measured on
  synthetic data: AUC 1.0000, 0 % false alarms, 100 % caught. **That is the
  script working, not the detector being good** — the signals came from our
  own simulator.
- **The clip-splice worry was checked, not assumed.** Concatenating 10 s clips
  into 30 s windows risks manufacturing a broadband impulse at every join,
  which is exactly what the detector hunts. Re-running with `--fade-ms 0`
  changed almost nothing (AUC 1.0000 both ways; normal 0.48x → 0.64x,
  anomalous 38.1x → 37.0x). **The joins are not doing the work.** Repeat that
  check on the real data — it is one flag and it is the difference between a
  result and an artefact.

⚠ **Note this also uses `frontend/src` only, never `node_modules`.** Copying
`node_modules` into `/tmp` on every run is what exhausted the sandbox disk and
blocked all scheduled work from 20–22 August.

---

## 2. CWRU bearing dataset — the accelerometer comparison

Already tooled: `ml/realdata/validate_public_dataset.py --help-download`. Built
and unit-tested; per `DOC_STATUS.md` it has **never been run against the real
`.mat` files**. Still worth doing — it is accelerometer data with documented
seeded defects and published fault frequencies, so it tests the envelope
analysis against a known ground truth.

**Never commit the `.mat` files.** No licence. `.gitignore` covers `data/`.

---

## 3. Your own phone

`docs/PHONE_RECORDING.md`, and `tools/phone_monitor.py` runs the real pipeline
on the result. Its job is **not** validation — you have no known-faulty
machine, so it cannot tell you the detector works. Its jobs are:

1. **Characterise your recording chain** — above all, does the phone apply
   automatic gain control (the distance test in `PHONE_RECORDING.md`). You need
   this answer before any phone recording means anything.
2. **Practise the demo.** Recording a customer's machine in front of them and
   showing the envelope spectrum is a far better opener than a slide.
3. **Catch real-world nasties the simulator has no model of** — room
   reverberation, other machines, handling noise, mains hum.

---

## Suggested order

1. **MIMII `6_dB_pump.zip`** — the real test, with an answer key.
2. **Phone distance test** (2 min) — is AGC on?
3. **Phone recording of a fridge** — real-world noise, and demo practice.
4. **CWRU** — the accelerometer cross-check, once the mic path is understood.

Record every outcome in `RESULTS.md`, including the failures. A dataset that
defeats the detector is a finding worth more than a fridge that doesn't.
