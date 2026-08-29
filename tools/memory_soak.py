#!/usr/bin/env python3
"""
tools/memory_soak.py — backlog T4.1, memory-leak soak.

"Run the firmware loop for thousands of simulated windows tracking RSS;
assert no growth trend. A monitor that OOMs after three weeks is worse than
no monitor." No hardware needed — this drives the exact same functions
`firmware/main.py`'s real loop calls (extract_features, operating_point,
MahalanobisScorer.score, AlertGate.feed, StateDB.record_window,
ScoreReporter.report, SeverityReference.relative) rather than reimplementing
or approximating them, so a leak anywhere in that real path shows up here.

WHY THIS IS A SEPARATE SCRIPT FROM firmware/main.py, NOT A NEW FLAG ON IT
--------------------------------------------------------------------------
`main.py` is frozen. This imports its collaborators directly and drives them
in the same order `main.run()` does (see that function, and the comment
there about the gate being fed before telemetry is published) rather than
adding an RSS-tracking flag to frozen code with no failing test to justify
it.

WHY THIS RUNS IN CHUNKS, NOT ONE CALL
--------------------------------------------------------------------------
Feature extraction is the dominant cost (~150-230 ms/window in this
sandbox, per the handover notes (not in this public copy) and every prior soak/timing note in this
repo) and this agent's shell has a hard per-call wall-clock cap far short of
what "thousands of windows" needs in wall time. Same pattern as every other
long job in the task backlog (not in this public copy) rule 7: each invocation processes one chunk of
windows and appends its RSS samples to `--out` (a JSON Lines file, one
sample per line so a crashed/killed chunk loses at most its own tail, not
prior chunks); `--start-index` keeps the window/seed sequence continuous
across chunks so the schedule (below) does not repeat itself, and `--summary`
reads the WHOLE accumulated file back and fits the trend once every chunk is
done. Gate/reporter state (the persistence-gate streak counter) does NOT
carry across chunk boundaries — irrelevant here, since nothing below reads
whether an alert fired, only whether RSS grows.

WHAT SCHEDULE THIS RUNS
--------------------------------------------------------------------------
The two-speed 50/30 Hz duty cycle `firmware/baseline.py --simulate` itself
learns from (so windows fall into both regimes the baseline actually has,
not an unseen third one), plus a single-window transient every 200 windows
(a forklift, a door — the same shape T1.6's contamination guard and T3.4's
feedback tests use) that is never allowed to persist into a real alert. This
exercises the regime-switch and gate-reset code paths a genuinely idle
soak would not, without ever triggering `main.py`'s MQTT/webhook publish
branch (irrelevant to memory and would need a broker to observe safely).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"), str(ROOT / "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline import operating_point                      # noqa: E402
from capture import SimulatedSource                       # noqa: E402
from config_schema import load_config                     # noqa: E402
from features import extract_features                     # noqa: E402
from inference import AlertGate, MahalanobisScorer         # noqa: E402
from reporting import ScoreReporter, SeverityReference, physical_severity  # noqa: E402
from state import StateDB                                 # noqa: E402


def _rss_kb() -> float:
    """Current (not peak) resident set size in KB, read from /proc — a leak
    audit needs the CURRENT trend, not `resource.getrusage(...).ru_maxrss`,
    which only ever goes up and would report "no growth" even while RSS
    climbed and never came back down within one process's lifetime (which,
    for a genuine leak, it wouldn't — but a monotonic-max stat can't tell a
    leak apart from one large, harmless, early allocation, and this can)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1])  # kB
    raise RuntimeError("VmRSS not found in /proc/self/status")


def soak_schedule(i: int) -> dict:
    fr = 50.0 if (i // 24) % 2 == 0 else 30.0     # same 24-window regimes as baseline.py --simulate
    if i > 0 and i % 200 == 0:
        return {"kind": "bearing_outer", "severity": 0.5, "fr": fr}  # one-off transient
    return {"kind": "normal", "severity": 0.0, "fr": fr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "firmware" / "config.yaml")
    p.add_argument("--baseline", type=Path, default=ROOT / "firmware" / "baseline.npz")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="JSONL file, appended")
    p.add_argument("--windows", type=int, default=500, help="windows THIS chunk processes")
    p.add_argument("--start-index", type=int, default=0, help="global window index this chunk starts at")
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--summary", action="store_true",
                   help="instead of running a chunk, read --out and report the trend")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="with --summary: ALSO report the trend within each "
                        "individual chunk of this many windows, separately. "
                        "Each chunk of this soak runs as its own OS process "
                        "(see the module docstring) — the whole-run slope "
                        "mixes real within-process growth together with "
                        "inter-process allocator noise (measured: RSS at "
                        "process start alone varies ~10 MB run to run with "
                        "nothing else different), so it is not, by itself, "
                        "a leak signal. The per-chunk slope is.")
    args = p.parse_args()

    if args.summary:
        _summarise(args.out, args.chunk_size)
        return

    cfg = load_config(args.config)
    scorer = MahalanobisScorer(args.baseline)
    reporter = ScoreReporter(args.baseline)
    sev_ref = SeverityReference(args.baseline)
    db = StateDB(args.db, retention_days=cfg["storage"]["retention_days"])
    gate = AlertGate(need=round(2 * 60 / cfg["window"]["seconds"]))  # short gate; alert firing is irrelevant here

    # SimulatedSource seeds window j of ITS OWN generator as self.seed + j
    # and calls schedule(j) with that same 0-based j. To make chunk N of
    # this soak generate bit-identical windows to what a single unbroken
    # run would have produced at GLOBAL index start_index+j — without
    # actually re-generating and discarding every earlier chunk's windows,
    # which would make chunking pointless — shift both the seed base and
    # the schedule's own index by start_index up front, so the source's
    # internal 0-based j lines up with the global index for this chunk.
    source = SimulatedSource(cfg["window"]["seconds"], cfg["audio"]["sample_rate"],
                             cfg["accelerometer"]["sample_rate"],
                             schedule=lambda j: soak_schedule(args.start_index + j),
                             realtime=False, seed=4242 + args.start_index)

    fs_a, fs_v = cfg["audio"]["sample_rate"], cfg["accelerometer"]["sample_rate"]
    t_chunk_start = time.monotonic()
    out_f = open(args.out, "a")
    try:
        gen = source.windows()
        for j in range(args.windows):
            i = args.start_index + j
            audio, accel = next(gen)
            feats = extract_features(audio, fs_a, accel, fs_v)
            op = operating_point(feats["vector"], feats["fr_hz"])
            score = scorer.score(feats["vector"], op)
            db.record_window(score, feats)
            gate.feed(score["anomalous"])
            report = reporter.report(score["score"], score["regime"],
                                     score["anomalous"], alerting=gate.in_episode)
            sev_ref.relative(physical_severity(audio, fs_a, feats["band"]))  # T1.7 display cost, included on purpose
            _ = report
            if i % args.sample_every == 0:
                out_f.write(json.dumps({"i": i, "rss_kb": _rss_kb(),
                                        "t": time.time()}) + "\n")
                out_f.flush()
    finally:
        out_f.close()
        db.close()
    elapsed = time.monotonic() - t_chunk_start
    print(f"chunk done: windows {args.start_index}..{args.start_index + args.windows - 1} "
          f"in {elapsed:.1f}s ({elapsed / args.windows * 1000:.0f} ms/window), "
          f"rss now {_rss_kb():.0f} kB")


def fit_slope_mb_per_week(idx, rss, window_s: float = 30.0):
    """kB-per-window slope, and what it extrapolates to in MB/week at the
    shipped 30 s window — factored out so tests can pin the arithmetic
    against a known-slope synthetic series, independent of any real run."""
    import numpy as np
    slope, intercept = np.polyfit(idx, rss, 1)
    windows_per_week = 7 * 24 * 3600 / window_s
    return slope, slope * windows_per_week / 1024


def _summarise(out_path: Path, chunk_size: int | None) -> None:
    import numpy as np
    rows = [json.loads(line) for line in open(out_path) if line.strip()]
    if len(rows) < 2:
        print("not enough samples yet")
        return
    idx = np.array([r["i"] for r in rows], dtype=float)
    rss = np.array([r["rss_kb"] for r in rows], dtype=float)
    slope, growth_per_week_mb = fit_slope_mb_per_week(idx, rss)

    print(f"samples: {len(rows)}  windows: {int(idx.min())}..{int(idx.max())}")
    print(f"RSS: min {rss.min():.0f} kB  max {rss.max():.0f} kB  "
         f"range {rss.max() - rss.min():.0f} kB")
    print(f"whole-run slope (mixes inter-process noise — see --chunk-size): "
         f"{slope:+.4f} kB/window -> {growth_per_week_mb:+.1f} MB/week extrapolated")
    print(f"first sample: {rss[0]:.0f} kB   last sample: {rss[-1]:.0f} kB   "
         f"delta: {rss[-1] - rss[0]:+.0f} kB over {int(idx[-1] - idx[0])} windows")

    if not chunk_size:
        return
    print(f"\nPer-chunk trend (each chunk = one continuous process, "
         f"chunk_size={chunk_size}):")
    lo = int(idx.min())
    while lo <= idx.max():
        hi = lo + chunk_size - 1
        mask = (idx >= lo) & (idx <= hi)
        if mask.sum() >= 2:
            seg_idx, seg_rss = idx[mask], rss[mask]
            seg_slope, seg_growth = fit_slope_mb_per_week(seg_idx, seg_rss)
            print(f"  chunk {lo:5d}-{hi:5d}: n={int(mask.sum()):3d}  "
                 f"rss {seg_rss[0]:7.0f}->{seg_rss[-1]:7.0f} kB "
                 f"(delta {seg_rss[-1]-seg_rss[0]:+7.0f})  "
                 f"slope {seg_slope:+8.4f} kB/window -> {seg_growth:+8.1f} MB/week")
        lo += chunk_size


if __name__ == "__main__":
    main()
