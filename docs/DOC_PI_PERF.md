# Pi performance harness — T4.4

"A profiling script reporting per-stage timing and peak memory for the
feature pipeline, so the A53 budget can be checked the moment hardware
exists rather than guessed at." — the task backlog (not in this public copy) T4.4. Tool:
`tools/pi_perf_harness.py`, calling `firmware/features.py`'s own stage
functions directly, in the order `extract_features` calls them — not a
reimplementation. Fast regression tests for the tool's own mechanics in
`tests/test_pi_perf_harness.py`.

## Method

30 repeats per stage per case (healthy, and a bearing fault at severity
0.15 — the same two cases `features.py`'s own `__main__` demo uses), timed
with `time.perf_counter()`. Two memory measurements, reported side by side
because each misses something the other catches: `tracemalloc` (precise for
Python-level allocation, a documented **lower bound** since numpy's own C
allocator isn't fully covered on every build) and peak RSS over the repeated
calls (catches everything, noisier). Full caveats in the tool's own
docstring.

**Every timing number below is x86, this sandbox, not the Pi's A53.** This
report inherits the standing 8–10× single-core slowdown assumption already
used everywhere else in this repository (the handover notes (not in this public copy)'s memory table,
`README.md`'s latency budget) — that factor is NOT measured this run, and
is applied as a range for that reason. Settle it for real on H2's bring-up.

## Results: per-stage timing (healthy window)

| stage | mean (ms) | p95 (ms) | % of `extract_features` |
|---|---|---|---|
| `select_demodulation_band` | 88.9 | 89.2 | **57.9%** |
| `estimate_fr` | 16.3 | 16.3 | 10.6% |
| `envelope_features` | 15.0 | 15.0 | 9.7% |
| `stft_mag` | 5.9 | 5.9 | 3.9% |
| `band_energy_ilr(audio)` | 5.7 | 5.8 | 3.7% |
| `channel_stats(audio)` | 4.6 | 4.6 | 3.0% |
| `band_energy_ilr(accel)` | 2.5 | 2.5 | 1.6% |
| `channel_stats(accel_x)` | 2.3 | 2.3 | 1.5% |
| **`extract_features` (whole)** | **153.4** | **153.8** | — |
| `MahalanobisScorer.score` | 0.01 | 0.01 | 0.0% |

(Bearing-fault case: statistically the same shape — `select_demodulation_
band` 88.9 ms/58.6%, whole 151.8 ms. The demodulation band search cost does
not depend meaningfully on whether a fault is present.)

**One real finding worth acting on if the Pi budget ever gets tight:**
`select_demodulation_band` alone is **58% of the entire feature-extraction
time** — nearly 6× the next most expensive stage. It searches 6 candidate
frequency bands, computing a full envelope spectrum for each
(`_env_crest`, called once per band), to find the most "peaky" one. Every
other stage this audit measured is a small, roughly-even slice of the
remaining time. This was not previously broken out anywhere in this
repository's timing claims — the ~150 ms/window figure quoted everywhere
else (`README.md`, the handover notes (not in this public copy)) was always correct as a total, but
gave no indication that over half of it lives in one function. If Pi timing
ever needs to be trimmed, this is where the time actually is.

`MahalanobisScorer.score` is confirmed negligible — 0.01 ms, matching
`README.md`'s existing "inference alone is microseconds" claim exactly.

## A53 estimate and the stage-2 gate

x86 measured: **153.4 ms** mean. At this repo's standing 8–10× single-core
assumption: **1,227–1,534 ms** on the Pi Zero 2W's A53. Against
`docs/DOC_FIRMWARE.md`'s **2,000 ms** stage-2 gate: **passes, with ~470 ms
(24%) of margin at the high end of the assumed slowdown range** — real
margin, but this audit does not call it comfortable; a Pi that turns out
closer to the 10× end of the assumption, or a future feature added to the
pipeline, could close most of that gap. This is the same conclusion
`README.md`'s existing latency-budget table already draws ("inside the 2 s
stage-2 gate"), now with the per-stage breakdown behind it rather than only
the total.

## Memory

Peak RSS over 30 repeated `extract_features` calls on a healthy window:
delta of **+4 kB** from before the first call — consistent with T4.1's
memory-soak finding of no leak in the extraction path, at a much smaller
scale here (30 calls of one function, not 3,200 full loop iterations).
`tracemalloc`'s Python-level peak: **~25.6 MB** — a lower bound (see
method), and well inside the ~15 MB "feature extraction peak" line already
in `README.md`'s memory-budget table is now known to be an underestimate
relative to this number; both are x86 measurements of different things
(that line appears to be a hand estimate, this is a measured
lower bound) and neither has been reconciled against the Pi's real
allocator behaviour. Recorded as a gap, not resolved here — see
`docs/DOC_STATUS.md`.

## Reproduce

```bash
python firmware/baseline.py --simulate --windows 48 --out /tmp/baseline.npz --db /tmp/learn.db
python tools/pi_perf_harness.py --reps 30 --baseline /tmp/baseline.npz
```
