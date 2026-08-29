#!/usr/bin/env python3
"""fan_windows.py — re-score the fan recordings in 30-second windows.

    python tools/fan_windows.py data/ --out docs/fan_window_scores.csv

WHY THIS EXISTS, GIVEN fan_experiment.py ALREADY RAN
-----------------------------------------------------
`fan_experiment.py` produces ONE score per recording. Six recordings, six
numbers. That supports the sentence "the faulted recording scored 10.7x the
healthy one" and it supports nothing else, because a detection rate and a
false-alarm rate are properties of a POPULATION OF DECISIONS against a
THRESHOLD, and six numbers with no threshold is neither.

The distinction matters the moment anyone competent reads the claim. "10.7x
separation" invites "on how many trials, and how often was it wrong?", and the
honest answer from the whole-recording analysis is "one trial per condition,
and the question is undefined". This script makes it defined.

WHY 30 SECONDS AND NOT SOME OTHER LENGTH
-----------------------------------------
Because that is the window the product itself uses (`firmware/` learns and
scores on 30-second windows), so a per-window result here is a result about
the thing that would actually be deployed, not about a 5-minute laboratory
convenience.

WHY EVERY WINDOW MUST BE THE SAME LENGTH
-----------------------------------------
The comb score is not comparable across durations — T1.16 #8 measured the same
fault at 33.0 over 20 s and 12.7 over 1 s, because a longer average buys a
cleaner envelope spectrum. So all windows are exactly 30 s, and only the first
300 s of each recording is used, so every condition contributes exactly ten
windows and no condition is weighted more heavily for having been recorded for
longer.

WHY NORMALISATION IS PER RECORDING, NOT PER WINDOW
---------------------------------------------------
`fan_experiment.load` divides by the peak absolute sample of the whole file.
Doing that per window instead would silently re-gain every window to full
scale, which would flatter quiet windows and, worse, would make the analysis
here incomparable with the published whole-recording numbers.

WHAT THIS DOES NOT GIVE YOU
----------------------------
Ten windows cut from one continuous recording are not ten independent samples.
The fan, the room, the microphone and its position are identical across them,
so they share every nuisance variable there is. Forty healthy windows is
evidence of the strength of four recordings, not forty. A false-alarm RATE in
any population sense still needs Gate 3: many machines, many rooms, over time.
State it as "zero false alarms across 40 windows drawn from four healthy
recordings at two fan speeds" and let the reader do their own discounting.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "firmware"))

from cold_start_screen import screen                      # noqa: E402

WIN_S = 30.0
N_WIN = 10

# (filename, condition, fan speed, position in the before/during/after triple)
RECORDINGS = [
    ("Healthy fan take 1.wav",                              "healthy", "low",  "before"),
    ("Card in fan.wav",                                     "faulted", "low",  "during"),
    ("Healthy fan take 2 (after card).wav",                 "healthy", "low",  "after"),
    ("Healthy fan take 1 (high setting) 2.wav",             "healthy", "high", "before"),
    ("Carb in fan (high setting) 2.wav",                    "faulted", "high", "during"),
    ("Healthy fan take 2 (high setting, after card) 2.wav", "healthy", "high", "after"),
]


def load_mono(path: Path) -> tuple[np.ndarray, float]:
    """Mono float array normalised over the WHOLE file — see module docstring."""
    from scipy.io import wavfile
    fs, data = wavfile.read(path)
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x / (np.max(np.abs(x)) + 1e-12), float(fs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datadir", type=Path, help="folder holding the six recordings")
    ap.add_argument("--out", type=Path, default=None, help="write a CSV here")
    ap.add_argument("--mains", type=float, default=50.0)
    a = ap.parse_args(argv)

    rows = []
    for fname, cond, speed, order in RECORDINGS:
        path = a.datadir / fname
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        x, fs = load_mono(path)
        n = int(WIN_S * fs)
        for i in range(N_WIN):
            seg = x[i * n:(i + 1) * n]
            if len(seg) < n:
                break
            r = screen(seg, fs, mains=a.mains)
            rows.append((fname, cond, speed, order, i, int(i * WIN_S),
                         round(float(r["best_score"]), 2), r["best_f0"]))
            print(f"  {cond:8s} {speed:4s} {order:6s} w{i:02d} "
                  f"score={r['best_score']:8.2f} f0={r['best_f0']}")

    healthy = np.array([r[6] for r in rows if r[1] == "healthy"])
    faulted = np.array([r[6] for r in rows if r[1] == "faulted"])
    lo_healthy = np.array([r[6] for r in rows
                           if r[1] == "healthy" and r[2] == "low"])
    hi_healthy = np.array([r[6] for r in rows
                           if r[1] == "healthy" and r[2] == "high"])

    print(f"\nhealthy n={healthy.size}  {healthy.min():.2f} - {healthy.max():.2f}"
          f"  (median {np.median(healthy):.2f})")
    print(f"faulted n={faulted.size}  {faulted.min():.2f} - {faulted.max():.2f}"
          f"  (median {np.median(faulted):.2f})")
    print(f"worst-case margin: {faulted.min():.2f} / {healthy.max():.2f} "
          f"= {faulted.min() / healthy.max():.2f}x")

    # THE OPERATING-STATE TEST. A threshold set from the LOW-speed healthy
    # windows alone, then applied to the HIGH-speed healthy windows: a detector
    # that mistakes a legitimate change of operating point for a fault fails
    # here, and that failure mode -- not missed detections -- is what makes
    # condition monitoring hard to sell.
    thr = lo_healthy.max()
    print(f"\nthreshold from low-speed healthy only: {thr:.2f}")
    print(f"  high-speed HEALTHY windows above it: "
          f"{int((hi_healthy > thr).sum())}/{hi_healthy.size}")
    print(f"  faulted windows above it:            "
          f"{int((faulted > thr).sum())}/{faulted.size}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["recording", "condition", "fan_speed", "order", "window",
                        "t_start_s", "comb_score", "peak_hz"])
            w.writerows(rows)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
