# Memory-leak soak — T4.1

"Run the firmware loop for thousands of simulated windows tracking RSS;
assert no growth trend. A monitor that OOMs after three weeks is worse than
no monitor." — the task backlog (not in this public copy) T4.1. No hardware needed: this drives the
exact functions `firmware/main.py`'s real loop calls
(`extract_features`, `operating_point`, `MahalanobisScorer.score`,
`AlertGate.feed`, `StateDB.record_window`, `ScoreReporter.report`,
`SeverityReference.relative`) via `tools/memory_soak.py`, not a
reimplementation of them, so a leak anywhere in that real path would show up
here. Reproduce with the commands in the tool's own docstring; a fast unit
test for the tool's own mechanics (chunk continuity, the slope arithmetic)
lives in `tests/test_memory_soak.py`.

## What was actually run

**3,200 simulated windows** (96,000 seconds of simulated machine time — 26.7
hours), through the two-speed 50/30 Hz duty cycle `firmware/baseline.py
--simulate` itself learns from, plus a single-window transient every 200
windows that is never allowed to persist into an alert. RSS (current
resident set size, read from `/proc/self/status`, not
`resource.getrusage(...).ru_maxrss` — see the tool's own docstring for why
the peak-only stat is the wrong one for this) was sampled every 5th window.

**Run in 6 chunks, not one continuous process — and that matters for how to
read the result.** This agent's shell caps a single call well under the wall
time 3,200 windows takes at ~230-270 ms/window (measured; matches every
other extraction-speed note in this repo). Each chunk is therefore a
*separate OS process*, restarting Python/numpy/scipy/sklearn's own import
and allocator state from scratch. That is the correct, and only, workaround
available in this sandbox (see the task backlog (not in this public copy) rule 7) — but it means the
naive "RSS at window 0 vs RSS at window 3195" comparison mixes real
within-process memory growth together with ordinary inter-process allocator
noise, and the two are not the same thing.

## The result

Measured, per continuous-process chunk (the only comparison that isolates
real growth from inter-process noise):

| chunk (windows) | samples | RSS start → end | Δ | slope | extrapolated |
|---|---|---|---|---|---|
| 0–599     | 120 | 148,324 → 158,504 kB | +10,180 kB | −0.44 kB/window | −8.6 MB/week |
| 600–1,199 | 120 | 138,200 → 139,896 kB | +1,696 kB  | −5.45 kB/window | −107.2 MB/week |
| 1,200–1,699 | 100 | 138,232 → 138,420 kB | +188 kB    | +14.72 kB/window | +289.8 MB/week |
| 1,700–2,199 | 100 | 148,348 → 154,532 kB | +6,184 kB  | −7.12 kB/window | −140.3 MB/week |
| 2,200–2,699 | 100 | 138,192 → 138,364 kB | +172 kB    | +0.46 kB/window | +9.1 MB/week |
| 2,700–3,199 | 100 | 148,320 → 152,256 kB | +3,936 kB  | +6.78 kB/window | +133.4 MB/week |

**No consistent direction.** Three chunks trend up, three trend down; the
sign flips chunk to chunk with no relationship to window count. That is the
signature of noise, not a leak — a real leak would show the same sign in
every chunk, growing larger (not smaller) in later chunks as fragmentation
or an unbounded structure compounds. It is not what these six runs show.

**A second, independent piece of evidence for "no leak, just noise":** RSS
at the very START of a fresh process (before any windows are scored, right
after imports) itself varies by about **10 MB** between otherwise-identical
runs (148,324 / 138,200 / 138,232 / 148,348 / 138,192 / 148,320 kB — an
almost exact alternation between two values, 6 processes in a row). Nothing
about window content differs at process start; this is allocator/ASLR/
import-order noise external to anything this project's code does, and it is
roughly the same MAGNITUDE as several of the per-chunk deltas above — direct
evidence that a 500-600-window single-process run in this environment
cannot resolve a trend much smaller than about ±150-300 MB/week-equivalent.
**This soak can rule out a gross leak (the kind that would OOM a 350 MB
`MemoryMax` cap within days-to-weeks) and cannot, on its own, rule out a
slow one.**

**Practically, the number that matters for the systemd `MemoryMax=350M`
cap** (`firmware/acoustic-monitor.service`): RSS never exceeded **165.2 MB**
across all 3,200 windows and 6 process restarts — well inside the cap, with
the caveat that this is Python/numpy/scipy import overhead plus the working
set on x86, not yet measured on the Pi's ARM build of the same libraries
(same caveat this repo already applies to every other x86-measured timing
number — see the handover notes (not in this public copy)'s memory table).

## What this settles, and what it does not

**Settled:** nothing in the firmware loop's real code path — feature
extraction, scoring, gate, state persistence, reporting, severity — grows
memory unboundedly across 3,200 windows (26.7 simulated hours) run through
it. No chunk showed the compounding, same-signed growth a real leak
produces.

**Not settled, and the honest limit of what a chunked, sandbox-constrained
soak can show:** whether a SLOW leak (smaller than the ~150-300 MB/week
noise floor measured above) exists. The only way to lower that noise floor
is a genuinely continuous single process running for much longer than one
sandboxed shell call allows — which is exactly what H4's real 7-day soak
will be, on a real Pi, as one continuous `systemd` service, with none of
this measurement's process-restart noise. If Logan wants a cleaner software
answer before then: run `tools/memory_soak.py` as ONE long-running local
process (not chunked) on a laptop overnight — `for i in $(seq 0 599 20000);
do python tools/memory_soak.py --db /tmp/soak.db --out samples.jsonl
--windows 600 --start-index $i; done` chains chunks the same way this run
did, but running the SAME command directly in a real terminal (not this
agent's per-call-capped shell) removes the artificial per-chunk process
restart entirely if it is instead one `python` invocation with
`--windows 20000` in a single call.

## Reproduce

```bash
python firmware/baseline.py --simulate --windows 48 --out /tmp/baseline.npz --db /tmp/learn.db
python tools/memory_soak.py --baseline /tmp/baseline.npz --db /tmp/soak.db \
    --out /tmp/samples.jsonl --windows 3000 --start-index 0 --sample-every 5
python tools/memory_soak.py --out /tmp/samples.jsonl --summary --chunk-size 600 \
    --db x --baseline x
```

(On a machine without this agent's shell timeout, drop the chunking
entirely — `--windows 3000` in one call is one continuous process, which is
the more decisive measurement described above.)
