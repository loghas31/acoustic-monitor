#!/usr/bin/env python3
"""
tools/db_growth_audit.py — backlog T4.2, database growth + SD-wear audit.

"Measure bytes/day at the real telemetry rate, confirm retention pruning
holds, and estimate SD write endurance. Recommend settings."

WHY THIS DOES NOT RUN REAL FEATURE EXTRACTION
--------------------------------------------------------------------------
Unlike T4.1's memory soak, database GROWTH depends only on the STRUCTURE and
SIZE of what `state.py`'s `record_window` writes per call, not on whether
the feature values came from a real signal — a 37-dim vector of random
floats, rounded to the same precision `record_window` itself uses
(`round(float(v), 5)`), takes exactly the same number of JSON bytes as a
real one. Skipping `extract_features`/`MahalanobisScorer` turns a
would-be multi-hour run (thousands of windows at the ~230 ms/window T4.1
measured) into one that finishes in well under a minute, real feature
extraction contributing nothing this measurement needs.

WHY THE "DAYS" ARE SIMULATED WITHOUT SLEEPING, AND WHY THAT NEEDS CARE
--------------------------------------------------------------------------
`StateDB._trusted_prune_ts` (T4.3, see its own long comment in state.py) is
DELIBERATELY suspicious of a `ts` that advances faster than real elapsed
time — that is precisely the NTP-forward-jump protection, and it is
supposed to refuse to prune on a suspicious jump. Feeding it 20 simulated
days of `ts` inside one real fast loop IS exactly the shape of jump it is
built to distrust, and it would (correctly!) refuse to prune almost
anything, making this audit measure "database growth if pruning never
worked" rather than "database growth with pruning working as designed."
So this script's fake clock advances `state.time.monotonic()` in lockstep
with the simulated `ts` — as far as `_trusted_prune_ts` can tell, real time
really did pass. This is a deliberate, documented monkeypatch of the
process-global `time.monotonic`, restored before `run_audit` returns; none
of the real firmware ever runs with a fake clock, and nothing else in this
script depends on real monotonic time either.

WHAT IS AND ISN'T MODELLED
--------------------------------------------------------------------------
Every window's `anomalous` flag is set true with `--anomaly-rate` (default
0.01) probability, independent of the underlying "signal" (there isn't
one) — a crude stand-in for a healthy machine's real false-alarm rate, so
the `anomalies` table's contribution to growth isn't simply ignored. This is
NOT this project's own measured false-alarm rate (see docs/DOC_STATUS.md —
that number needs H4's real soak); it is a round, slightly pessimistic
number chosen so the estimate is not flattered by assuming zero anomalies.
`feedback` table growth is NOT modelled — it only grows when a customer
clicks "this was normal", is bounded by however many episodes actually get
marked (not a background rate), and T3.4/T3.6 already exercise it
separately.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import state as state_mod                                  # noqa: E402
from state import StateDB                                  # noqa: E402


def _fake_vector(rng: random.Random, dim: int = 37) -> list[float]:
    # Same shape and rounding record_window itself applies
    # (json.dumps([round(float(v), 5) for v in feats["vector"]])) — the
    # actual VALUES are irrelevant to row size, only the digit count is.
    return [round(rng.uniform(-3.0, 3.0), 5) for _ in range(dim)]


def _fake_mel_mean(rng: random.Random, n: int = 4) -> list[float]:
    return [round(rng.uniform(-40.0, 10.0), 3) for _ in range(n)]


def run_audit(db_path: Path, days: int, retention_days: int, window_s: float,
             anomaly_rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    windows_per_day = round(86400 / window_s)

    real_mono_t0 = time.monotonic()

    def fake_monotonic() -> float:
        # Advances in lockstep with simulated ts (t0 + elapsed_simulated_s),
        # so state.py's own clock-jump guard sees real-looking elapsed time
        # for whatever ts this script is currently feeding it — see the
        # module docstring for why this is necessary, not a shortcut.
        return real_mono_t0 + fake_monotonic.elapsed_s
    fake_monotonic.elapsed_s = 0.0
    real_monotonic = state_mod.time.monotonic
    state_mod.time.monotonic = fake_monotonic

    try:
        db = StateDB(db_path, retention_days=retention_days)
        t0 = 1_700_000_000.0
        checkpoints = []
        for day in range(days):
            for w in range(windows_per_day):
                i = day * windows_per_day + w
                ts = t0 + i * window_s
                fake_monotonic.elapsed_s = i * window_s
                anomalous = rng.random() < anomaly_rate
                score = {"score": rng.uniform(1.0, 12.0) if not anomalous else rng.uniform(9.0, 40.0),
                         "regime": rng.randint(0, 1), "threshold": 8.2, "anomalous": anomalous}
                feats = {"fr_hz": 50.0, "fr_reliable": True,
                         "vector": _fake_vector(rng),
                         "mel": _MelStub(_fake_mel_mean(rng)),
                         "band": (3000.0, 6000.0)}
                db.record_window(score, feats, ts=ts)
            db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # make file size on disk honest, not WAL-inflated
            size_bytes = os.path.getsize(db_path)
            row_count = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
            anomaly_count = db.conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
            checkpoints.append({"day": day + 1, "size_bytes": size_bytes,
                                "readings_rows": row_count, "anomaly_rows": anomaly_count})
        db.close()
    finally:
        state_mod.time.monotonic = real_monotonic
    return checkpoints


class _MelStub:
    """`feats["mel"].mean(axis=1)` is the only thing record_window calls on
    this — a real mel array is (n_bands, n_frames); faking just the method
    actually used avoids pulling in a full numpy array for no reason."""
    def __init__(self, precomputed_mean: list[float]):
        self._m = precomputed_mean

    def mean(self, axis=1):
        return self._m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--days", type=int, default=18)
    p.add_argument("--retention-days", type=int, default=7)
    p.add_argument("--window-seconds", type=float, default=30.0)
    p.add_argument("--anomaly-rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if args.db.exists():
        args.db.unlink()

    checkpoints = run_audit(args.db, args.days, args.retention_days,
                            args.window_seconds, args.anomaly_rate, args.seed)

    print(f"{'day':>4}  {'size_kB':>10}  {'readings':>9}  {'anomalies':>9}")
    for c in checkpoints:
        print(f"{c['day']:4d}  {c['size_bytes']/1024:10.1f}  "
             f"{c['readings_rows']:9d}  {c['anomaly_rows']:9d}")

    # Growth rate BEFORE retention kicks in (days 1..retention_days, where
    # nothing has been pruned yet — the true bytes/day at the real
    # telemetry rate) vs AFTER (should plateau once one full retention
    # window's worth of rows is being pruned for every window written).
    pre = [c for c in checkpoints if c["day"] <= args.retention_days]
    post = [c for c in checkpoints if c["day"] > args.retention_days]
    if len(pre) >= 2:
        pre_rate = (pre[-1]["size_bytes"] - pre[0]["size_bytes"]) / (pre[-1]["day"] - pre[0]["day"])
        print(f"\npre-retention growth: {pre_rate/1024:.1f} kB/day "
             f"(days {pre[0]['day']}-{pre[-1]['day']}, before any pruning)")
    if len(post) >= 2:
        post_rate = (post[-1]["size_bytes"] - post[0]["size_bytes"]) / (post[-1]["day"] - post[0]["day"])
        print(f"post-retention growth: {post_rate/1024:+.2f} kB/day "
             f"(days {post[0]['day']}-{post[-1]['day']}, steady state)")

    if checkpoints:
        bytes_per_row = checkpoints[0]["size_bytes"] / max(checkpoints[0]["readings_rows"], 1)
        print(f"\nmeasured bytes/reading-row (day 1, before any DELETE): "
             f"{bytes_per_row:.1f} B")

    if args.out_json:
        args.out_json.write_text(json.dumps({
            "checkpoints": checkpoints,
            "params": vars(args) | {"db": str(args.db), "out_json": str(args.out_json)},
        }, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
