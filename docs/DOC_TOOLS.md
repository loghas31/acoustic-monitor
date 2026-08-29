# Analysis tools — turning recordings into answers

Companion to the system overview (not in this public copy) §5–6.
Directories: `ml/`, `ml/realdata/`, `tools/`.

---

## Simulation and evidence (`ml/`)

| Script | What it answers |
|---|---|
| `simulate.py` | generates healthy / outer-race / inner-race / imbalance signals |
| `verify_signals.py` | **the physics evidence**: raw vs envelope spectrum |
| `evaluate.py` | detector performance: ROC, regime switches, gating |
| `model.py`, `train_offline.py`, `export_tflite.py` | optional v1.5 cloud autoencoder |

`simulate.py` earned its keep before any hardware existed: it is what proved
the original 8 kHz / 800 Hz sensing chain was blind, which set the bill of
materials. It stays forever as the regression-test fixture.

```bash
python ml/verify_signals.py
# BPFO peak-to-background — raw: 2.2x | envelope: 56.7x
# PASS: envelope analysis recovers the fault signature; raw spectrum does not.
```

## Real-data tools (`ml/realdata/`)

### `fault_frequencies.py` — where to look on *your* motor

```bash
python ml/realdata/fault_frequencies.py --bearing 6204 --rpm 2850
```

Prints BPFO, BPFI, BSF, FTF, each as a multiple of shaft rate, plus the
self-check **BPFO + BPFI = N·f_r** (the γ terms cancel exactly — a free proof
the implementation is right).

It also states its **confidence**: geometries for common bearings are
estimated from boundary dimensions, so predicted frequencies carry a few
percent of uncertainty. Widen your search window accordingly — do not hunt for
a peak at exactly 144.967 Hz.

This is a *development* tool. The shipping detector never uses it, because
customers cannot supply bearing geometry (see [DOC_DETECTOR.md](DOC_DETECTOR.md)).

### `analyse_recording.py` — the week-2 verdict

```bash
python ml/realdata/analyse_recording.py \
    --healthy sessions/grinder_healthy/seg000.wav \
    --faulty  sessions/grinder_faulty/seg000.wav \
    --bearing 6204 --rpm 2850
```

Produces envelope spectra for both, marks the expected fault frequencies,
computes the peak-to-background ratio **raw vs envelope**, and gives a PASS/FAIL
on the week-2 gate — reproducing on real data the evidence we generated
synthetically. Works with mic-only recordings.

### `recording_io.py`

Loads and validates recordings and their metadata sidecars; shared by the
tools above.

## Operations tools (`tools/`)

### `soak_report.py` — the number the business rests on

```bash
python tools/soak_report.py --db /var/lib/acoustic-monitor/state.db --outdir report/
```

Outputs markdown, JSON and four figures:

- **false alarms per node-week** with a 95 % Poisson upper bound
- score distributions vs threshold, per regime
- a timeline with alert episodes marked
- regime occupancy over time (did the machine use the regimes we learned?)
- **headroom**: how close healthy operation came to the threshold — the
  leading indicator of future false alarms
- a **recommended `persist_minutes`** computed from the observed transient
  run-length statistics

That last item matters: the tool tells you how to tune the gate *from your
data* rather than leaving you to guess.

Verified on a synthetic 7-day, 20 160-window soak with three injected
transients — all three correctly suppressed:

```
FALSE ALARMS PER NODE-WEEK : 0.00
  95% upper bound          : 3.00
  Gate 3 (<= 1)            : PASS (point only)
  headroom @ p99           : 20.7 %
```

**"PASS (point only)"** is deliberate wording. Zero alerts in one week does
not prove a rate below one per week; the upper bound says the truth could be
as bad as 3. Two or three weeks of soak tighten it.

### `ingest.py` — the front door for any recording that was not made by us

Converts anything (wav or csv, any sample rate, any dtype, mono or multi-channel)
into the canonical format: mono wav at 16 kHz, `<stem>_accel.csv` at 6400 Hz,
and a `<stem>.json` sidecar carrying the rpm, the bearing, and the provenance
of every conversion.

```bash
# ALWAYS ingest the healthy/faulty pair in ONE command
python tools/ingest.py raw/healthy.wav raw/faulty.wav \
    --out-dir data/real --rpm 2850 --bearing 6204

python tools/ingest.py --self-test     # prove the tool, no input files needed
```

Conversion is the easy part; the **audit** is why this file exists. Every run
reports, and stores in the sidecar:

| Check | Why |
|---|---|
| resample ratio as exact integers (44100→16000 = `x160/441`) | a wrong ratio stretches the whole frequency axis, so BPFO is searched for in the wrong place and "healthy" is reported for ever |
| **bandwidth in the demodulation band** | the one failure that yields a system which looks fine and can never detect anything. Band-limiting a known-faulty recording to 8 kHz takes its BPFO envelope contrast from 35.8x to **1.2x** |
| clipped-sample fraction | a flat top is a broadband impulse, i.e. clipping *manufactures* a bearing fault |
| DC offset (removed from audio, kept on the accelerometer) | on the accelerometer the DC term is gravity, and its magnitude is a free mounting check |
| accelerometer timestamp jitter | how a dropped SPI FIFO read actually presents itself |

Two policies worth knowing before you use it:

* **One common gain across the whole invocation.** The RMS difference between a
  healthy and a faulty capture is data; normalising each file separately
  destroys it. The gain is 1.0 unless something would clip. `--normalise`
  scales the batch so the loudest file peaks at 90 % FS (still one factor);
  `--independent-gain` opts out, and says so on stderr.
* **It verifies its own output.** After writing, it re-loads the file with
  `recording_io.load_recording` and runs `firmware/features.extract_features`
  on it. If the full feature vector does not come out, the file is not
  canonical whatever the tool claims. `--no-verify` skips it.

Exit codes: `0` clean, `0` with warnings (`1` under `--strict`), `1` when a
file cannot support detection at all, `2` on a usage or read error.

### `simulate_soak.py` — build a fixture before real data exists

Generates a realistic multi-day soak (idle/run regimes, defrost cycles,
occasional transients) and writes it to a state DB so the analysis tooling can
be developed and tested before any hardware runs. Supports `--jobs`,
`--resume` and `--max-windows`; generating a full day of 16 kHz audio takes
several minutes of CPU.

### `regime_miconly_cost.py` — why `choose_k` has two gates instead of one

Evidence for a frozen-file change, kept runnable rather than quoted. Stage
`null` re-derives how often the clustering invents regimes in data that has
none, as a function of how many directions the operating point varies in (the
pre-T1.9 rule: **98.8 %** in one effective dimension, 0 % in two or three).
Stage `cost` runs 96 healthy + 24 faulty mic-only windows through the real
pipeline and bootstraps 100 learn periods under three rules — current, pre-T1.9
and a forced k=1 oracle — reporting held-out false-alarm rate and AUC
(**0.0217** vs **0.1358** vs 0.0217). Takes about a minute; features cache to
`/tmp`.

Run it if you change anything about regime selection, `op_scale`, or the
operating-point definition.

## Recommended order of use

| Stage | Tool |
|---|---|
| Before hardware | `verify_signals.py`, `evaluate.py`, `simulate_soak.py` + `soak_report.py` |
| Parts arrive | `firmware/bench/selftest.py` |
| Bench rig | `record_session.py` → **`tools/ingest.py`** → `analyse_recording.py` |
| Anything recorded on a phone or downloaded | **`tools/ingest.py` first, always** |
| Week-long soak | `soak_report.py` |
