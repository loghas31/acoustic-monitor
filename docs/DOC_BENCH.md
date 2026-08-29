# Bench tools — what to run the day the parts arrive

Companion to the system overview (not in this public copy) §6. Directory: `firmware/bench/`.

---

## The one command to remember

```bash
python firmware/bench/selftest.py
```

It runs every check in dependency order and stops at the first failure,
because a failure early makes every later measurement meaningless. Exit codes:
`0` all passed, `1` a check failed, `2` no hardware attached, `3` aborted.

Crucially it distinguishes **"no sensor attached"** from **"sensor is
broken"** — conflating those sends you debugging software that is fine.

## The checks

### 1. `check_audio.py` — can we hear, at the right rate?

Enumerates ALSA devices, records, then reports the **measured** sample rate
(samples counted over wall-clock time, *not* the configured value), DC offset,
RMS, clipping count and noise floor.

`--tone 1000` records while you play a known tone from your phone and asserts
the dominant FFT peak lands within 1 % of it.

**Why:** a sample rate that is wrong by 10 % stretches every frequency axis by
10 %. A 4.5 kHz resonance would appear at 4.05 kHz, the demodulation band would
be chosen wrong, and nothing downstream would announce the problem. Thirty
seconds here saves a week of confusion.

### 2. `check_accel.py` — is the chip alive and correctly scaled?

- **WHO_AM_I == 0x7B.** The cheapest wiring test there is. If this byte is
  wrong, MISO/MOSI are swapped or the SPI mode is wrong. Nothing else runs.
- **Gravity magnitude = 1.000 g ± 0.1** over a still capture. This is the only
  **free absolute calibration** available anywhere on Earth. If it reads 2 g,
  the sensitivity constant is wrong by 2× — and so is every RMS, crest factor
  and threshold in the system. A miscalibrated monitor still *detects change*,
  but its numbers mean nothing and cannot be compared between machines.
- **Achieved ODR, measured.** If Python cannot drain the FIFO fast enough the
  effective rate silently drops.
- Noise density per axis, and a saturation check.

### 3. `check_mount.py` — where does *this* machine resonate?

**Scientifically the most important script here.** Tap the housing; an impulse
excites every resonance at once and the structure answers by ringing at its
own natural frequencies. That is classical impulse modal testing, done in
thirty seconds.

It reports the dominant resonance, its Q, and a **recommended demodulation
band** — and it checks tap-to-tap repeatability as a mounting-quality verdict.
A resonance that moves between taps means the magnet is rocking, and a rocking
magnet is a false-alarm generator.

**Why it matters:** the current default band (3–6 kHz) came from *simulation*,
where we modelled a 4.5 kHz resonance. Your motor is not our simulation. If
its real resonance sits elsewhere, the default would demodulate a band with no
fault energy in it and the detector would go quietly deaf.

The bearing impacts you are hunting ring the *same* structure, so the peaks
found here are exactly where fault energy will appear.

### 4. `record_session.py` — produce the week-2 dataset

Writes timestamped `.wav` + `.csv` + **JSON metadata sidecar**, chunked into
segments (a crash loses one segment, not two hours).

```bash
python firmware/bench/record_session.py --machine "grinder" --label healthy \
    --minutes 120 --rpm 2850 --bearing 6204
```

`--label` is ground truth and is the most important field in the whole
toolkit: it is what turns a pile of recordings into a dataset you can compute
an ROC curve from. In six weeks you will not remember which motor, which
bearing, or whether the fault was in yet. **A recording without provenance is
not data.**

Output format is exactly what `capture.FileSource` reads and
`ml/realdata/analyse_recording.py` analyses.

## Verified behaviour without hardware

Every script was run in a sandbox with no sensors. Each prints a friendly
block naming what is missing and what to install, and exits `2` — no
tracebacks. `selftest.py` correctly reports `NO HARDWARE (nothing attached)`
and stops.

## The bring-up sequence

1. `check_audio.py` — with `--tone 1000` from your phone
2. `check_accel.py` — board flat and still
3. `check_mount.py` — sensor on the machine, machine **off**, tap it
4. Put the recommended band in `config.yaml`
5. `record_session.py --label healthy --minutes 120`
6. Fit a baseline, then go to [DOC_TOOLS.md](DOC_TOOLS.md)

Record the gravity magnitude and achieved sample rate in your lab notebook.
They are the two numbers that make sessions comparable to each other.
