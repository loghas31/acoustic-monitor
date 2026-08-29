#!/usr/bin/env python3
"""
tools/pi_perf_harness.py — backlog T4.4, Pi performance harness.

"A profiling script reporting per-stage timing and peak memory for the
feature pipeline, so the A53 budget can be checked the moment hardware
exists rather than guessed at."

Calls the REAL stage functions `firmware/features.py`'s own
`extract_features` calls, in the same order, rather than reimplementing or
guessing at internal timing — `select_demodulation_band`, `stft_mag`,
`channel_stats` (×4: audio + 3 accel axes), `band_energy_ilr` (×2: audio +
accel), `envelope_features`, `estimate_fr` — plus `MahalanobisScorer.score`
(the actual downstream consumer) and the full `extract_features` call
itself as an end-to-end cross-check that per-stage times sum to roughly the
whole.

WHAT "PEAK MEMORY" MEANS HERE, AND ITS LIMIT
--------------------------------------------------------------------------
Two measurements, reported side by side because each misses something the
other catches:
  - `tracemalloc`: tracks PYTHON-level allocations precisely, but numpy's
    large array buffers are allocated through numpy's own C allocator,
    which recent numpy/tracemalloc integration covers on some builds and
    not others — treat this as a LOWER BOUND on real memory use, not a
    complete count.
  - RSS delta (same `/proc/self/status` VmRSS technique as T4.1's memory
    soak): catches everything, including C-level allocation, but is noisy
    at sub-millisecond granularity and can UNDER-report short-lived peaks
    that the allocator reuses before the next sample, and can OVER-report
    if the OS hasn't reclaimed freed pages yet. Reported as the max RSS
    observed across many repeated calls, which smooths sampling noise
    without hiding a real large one-off allocation.

Neither number is measured on the Pi's own numpy/scipy build. Every timing
number here inherits this repo's standing x86-vs-A53 caveat (see
the handover notes (not in this public copy)'s memory table and `README.md`'s latency budget): an
8-10x single-core slowdown factor is ASSUMED from prior measurement on this
project's own extraction time, not measured on this run, and is reported
as a range for that reason.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"), str(ROOT / "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline import operating_point                      # noqa: E402
from features import (band_energy_ilr, channel_stats,     # noqa: E402
                      envelope_features, estimate_fr,
                      extract_features, select_demodulation_band,
                      stft_mag)
from inference import MahalanobisScorer                    # noqa: E402
from simulate import SimConfig, bearing_fault_signal, normal_signal  # noqa: E402

A53_SLOWDOWN_LOW, A53_SLOWDOWN_HIGH = 8.0, 10.0    # this repo's standing assumption, not measured here


def _rss_kb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1])
    raise RuntimeError("VmRSS not found")


def _time_stage(fn, args, reps: int) -> list[float]:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return times


def profile_window(audio: np.ndarray, fs_a: float, accel: np.ndarray, fs_v: float,
                   scorer: MahalanobisScorer | None, reps: int) -> dict:
    # Same padding extract_features itself does internally (features.py:568-573)
    # — the simulator hands back 1-axis accel, same as its own __main__ demo.
    if accel.ndim == 1:
        accel = accel[:, None]
    while accel.shape[1] < 3:
        accel = np.column_stack([accel, accel[:, -1] * 0.0])

    stages = {}

    band, _ = select_demodulation_band(audio, fs_a)   # need a real band for envelope_features below
    stages["select_demodulation_band"] = _time_stage(select_demodulation_band, (audio, fs_a), reps)
    stages["stft_mag"] = _time_stage(stft_mag, (audio, fs_a), reps)
    stages["channel_stats(audio)"] = _time_stage(channel_stats, (audio,), reps)
    stages["channel_stats(accel_x)"] = _time_stage(channel_stats, (accel[:, 0],), reps)
    stages["band_energy_ilr(audio)"] = _time_stage(band_energy_ilr, (audio, fs_a), reps)
    stages["band_energy_ilr(accel)"] = _time_stage(band_energy_ilr, (accel[:, 0], fs_v), reps)
    stages["envelope_features"] = _time_stage(envelope_features, (audio, fs_a, band), reps)
    stages["estimate_fr"] = _time_stage(estimate_fr, (audio, fs_a, accel, fs_v), reps)
    stages["extract_features (whole)"] = _time_stage(extract_features, (audio, fs_a, accel, fs_v), reps)

    if scorer is not None:
        feats = extract_features(audio, fs_a, accel, fs_v)
        op = operating_point(feats["vector"], feats["fr_hz"])
        stages["scorer.score"] = _time_stage(scorer.score, (feats["vector"], op), reps)

    return stages


def summarise(stages: dict) -> None:
    whole = stages.get("extract_features (whole)", [0.0])
    whole_mean = statistics.mean(whole)
    print(f"{'stage':30s} {'mean ms':>9s} {'p95 ms':>9s} {'% of extract_features':>22s}")
    for name, times in stages.items():
        mean_ms = statistics.mean(times)
        p95_ms = sorted(times)[int(0.95 * (len(times) - 1))]
        pct = f"{100 * mean_ms / whole_mean:.1f}%" if name != "extract_features (whole)" else "—"
        print(f"{name:30s} {mean_ms:9.2f} {p95_ms:9.2f} {pct:>22s}")

    print(f"\nx86 measured, this sandbox: extract_features mean "
         f"{whole_mean:.1f} ms, p95 {sorted(whole)[int(0.95 * (len(whole) - 1))]:.1f} ms")
    print(f"A53 estimate (this repo's standing {A53_SLOWDOWN_LOW:.0f}-{A53_SLOWDOWN_HIGH:.0f}x "
         f"single-core assumption, NOT measured this run): "
         f"{whole_mean * A53_SLOWDOWN_LOW:.0f}-{whole_mean * A53_SLOWDOWN_HIGH:.0f} ms")
    print("2000 ms stage-2 gate (DOC_FIRMWARE.md): "
         f"{'PASSES with margin' if whole_mean * A53_SLOWDOWN_HIGH < 2000 else 'AT RISK'} "
         f"under the A53 estimate above")


def memory_profile(audio, fs_a, accel, fs_v, reps: int) -> None:
    tracemalloc.start()
    rss_before = _rss_kb()
    peak_rss = rss_before
    for _ in range(reps):
        extract_features(audio, fs_a, accel, fs_v)
        peak_rss = max(peak_rss, _rss_kb())
    current, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\ntracemalloc peak (Python-level allocations only, LOWER BOUND — "
         f"see module docstring): {tm_peak / 1024:.1f} kB")
    print(f"RSS: before {rss_before:.0f} kB, peak observed over {reps} calls "
         f"{peak_rss:.0f} kB, delta {peak_rss - rss_before:+.0f} kB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=30, help="repeats per stage, for a stable mean/p95")
    p.add_argument("--baseline", type=Path, default=None,
                   help="if given, also time MahalanobisScorer.score against it")
    args = p.parse_args()

    cfg = SimConfig(duration_s=30.0)
    rng = np.random.default_rng(0)
    cases = {
        "healthy": (normal_signal(cfg, cfg.fs_audio, rng), normal_signal(cfg, cfg.fs_accel, rng)),
        "bearing fault sev 0.15": (
            bearing_fault_signal(cfg, cfg.fs_audio, rng, 0.15, "outer"),
            bearing_fault_signal(cfg, cfg.fs_accel, rng, 0.15, "outer")),
    }
    scorer = MahalanobisScorer(args.baseline) if args.baseline else None

    for name, (audio, accel) in cases.items():
        print(f"\n===== {name} =====")
        stages = profile_window(audio, cfg.fs_audio, accel, cfg.fs_accel, scorer, args.reps)
        summarise(stages)

    print("\n===== memory (healthy window, extract_features only) =====")
    audio, accel = cases["healthy"]
    memory_profile(audio, cfg.fs_audio, accel, cfg.fs_accel, reps=args.reps)


if __name__ == "__main__":
    main()
