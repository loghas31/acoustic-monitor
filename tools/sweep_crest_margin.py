#!/usr/bin/env python3
"""Sweep `baseline.CREST_FLOOR_MARGIN` and measure BOTH sides of the trade.

WHY THIS FILE EXISTS
--------------------
F20's defect was a measured number that lived nowhere. Its Part 2 investigation
then repeated the mistake in a smaller way: the Option-A/Option-B sweep tables
in `docs/DOC_SELF_REVIEW.md` were produced by "a throwaway `/tmp` script, not in
this repository", so nobody can re-run them and nobody can check them. When
T1.14 Part 2 shipped, the margin sweep it was justified by had the same problem.

This script is that sweep, committed. It measures, for each candidate margin:

  * **F19 recovery** — on `synth_phone_recording`'s pink-noise machines, whose
    resonance sits at 1600 Hz, deliberately OUTSIDE `features.DEFAULT_BAND`
    (3-6 kHz). A machine counts as "recovered" when the selector picks a band
    containing the resonance on the FAULTY signal instead of falling back to
    DEFAULT_BAND. This is the exact quantity F19 measured as 1/14 at the old
    fixed floor of 10.0.
  * **deployed FPR** — `ml/evaluate.py`'s own `compute_metrics()` against a
    baseline refit at that margin, so the number is the same one the STAGE 3
    gate and `tests/test_evaluate_pinned.py` talk about.

The two move in opposite directions. The point of the script is to show the
frontier, not to find a margin that wins on both -- F20 Part 1 already proved
no such margin exists for the hysteresis variant, and the same shape holds here.

USAGE
    python tools/sweep_crest_margin.py                  # the shipped grid
    python tools/sweep_crest_margin.py --margins 0.3 0.7
    python tools/sweep_crest_margin.py --machines 6 --json out.json

Each margin refits a baseline from scratch, so this is slow (minutes). It is a
measurement tool, not part of the test suite; `tests/test_evaluate_pinned.py`
pins the shipped point cheaply.

NOTE ON REPRODUCIBILITY. Seeds are fixed and stated in `--help`. Re-running
this must reproduce the table in F20's "Part 2 CLOSED" section exactly. If it
does not, that is a finding -- write it up rather than quietly editing the doc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "firmware", ROOT / "ml", ROOT / "ml" / "realdata"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import baseline as baseline_mod          # noqa: E402
import features as features_mod          # noqa: E402
from features import DEFAULT_BAND, select_demodulation_band  # noqa: E402

SEED_MACHINES = 1        # synth_phone_recording.make_pair seeds: 1..n_machines
SEED_LEARN = 20250823    # learn-period seed for the refit baseline
RESONANCE_HZ = 1600.0    # synth_phone_recording.DEFAULT_RESONANCE_HZ


def f19_recovery(margin: float, n_machines: int = 6,
                 duration_s: float = 20.0,
                 fs: float = 16000.0,
                 severity: float = 0.20) -> tuple[int, int]:
    """How many machines' FAULTY signals get a band containing the resonance.

    Calibrates the floor per machine from that machine's own HEALTHY signal --
    mirroring what `fit_baseline` does in deployment -- then asks the selector
    to pick a band on the faulty signal. Returns (recovered, total).

    `fs` DEFAULTS TO THE REPO'S 16 kHz AUDIO RATE, NOT `synth_phone_recording`'s
    44.1 kHz PHONE RATE, AND THIS IS LOAD-BEARING. Measured while building this
    script: at 44.1 kHz the healthy band crests come out at 3.5-4.1 and the
    faulty ones at ~4.4, so `calibrate_crest_floor` clamps to `MIN_CREST_FLOOR`
    (6.5) for EVERY margin, nothing ever clears the floor, and recovery is 0/6
    at margin 0.3 and 0.7 alike -- the sweep looks flat because the margin has
    been rendered inoperative, not because it does not matter. The same
    quantities at 16 kHz are 5.4-7.7, which is what `calibrate_crest_floor`'s
    own comment block records and what F19/T1.13 measured against.

    Crest is not scale-free across sample rates: more bandwidth means more
    candidate bands and more noise-only bands diluting the envelope peak. Any
    future measurement of this quantity must state its rate.

    `severity` DEFAULTS TO 0.20, NOT `make_pair`'s 0.35, AND THIS IS ALSO
    LOAD-BEARING. Measured: mean faulty band crest is 23.1 at severity 0.35,
    8.3 at 0.20, and 6.0 at 0.10, against calibrated floors in the 6.9-7.7
    range. So 0.35 is saturated (recovery 6/6 at every margin including 3.0 --
    the sweep is vacuous) and 0.10 is dead (0/6 at every margin). Only around
    0.20 is the quantity marginal enough for the floor to decide anything,
    which is why F19 measured there. A sweep run at the default severity looks
    like a clean 6/6 result and is worth nothing.
    """
    from synth_phone_recording import make_pair

    recovered = 0
    for seed in range(SEED_MACHINES, SEED_MACHINES + n_machines):
        pair = make_pair(seed=seed, duration_s=duration_s, fs=fs,
                         severity=severity)
        rate = pair["fs"]

        # Per-machine calibration from the healthy signal, chunked into
        # learn-like windows so `calibrate_crest_floor` sees a distribution.
        win = int(rate * 2.0)
        healthy = pair["healthy"]
        crests = []
        for i in range(0, len(healthy) - win + 1, win):
            _, crest = select_demodulation_band(
                healthy[i:i + win], rate, crest_floor=0.0)
            crests.append(crest)
        if len(crests) < baseline_mod.MIN_CREST_SAMPLES:
            raise RuntimeError(
                f"only {len(crests)} learn windows from {duration_s}s at "
                f"{rate} Hz; need >= {baseline_mod.MIN_CREST_SAMPLES}. Raise "
                f"--duration, do not lower MIN_CREST_SAMPLES to make this pass.")

        old = baseline_mod.CREST_FLOOR_MARGIN
        try:
            baseline_mod.CREST_FLOOR_MARGIN = margin
            floor = baseline_mod.calibrate_crest_floor(np.asarray(crests))
        finally:
            baseline_mod.CREST_FLOOR_MARGIN = old

        band, _ = select_demodulation_band(
            pair["faulty"], rate, crest_floor=floor)
        if band != DEFAULT_BAND and band[0] <= RESONANCE_HZ <= band[1]:
            recovered += 1
    return recovered, n_machines


def deployed_fpr(margin: float, windows: int = 40,
                 learn_windows: int = 48) -> dict:
    """Refit a baseline at this margin and run `evaluate.compute_metrics`.

    Mirrors `firmware/baseline.py`'s own `main()` learn path exactly -- same
    config, same `--simulate` two-regime schedule (8 windows at 50 Hz, 8 at
    30 Hz, alternating), same `collect_features` two-pass calibration, same
    `fit_baseline`. It is done in-process rather than by subprocess only so
    the margin can be patched; every other input is the shipped one, because
    a sweep measured on a different pipeline than the deployed one would be
    worse than no sweep at all.
    """
    import tempfile

    import evaluate
    from capture import make_source
    from config_schema import load_config
    from features import FEATURE_NAMES
    from inference import MahalanobisScorer

    cfg = load_config(ROOT / "firmware" / "config.yaml")
    fs_a, fs_v = cfg["audio"]["sample_rate"], cfg["accelerometer"]["sample_rate"]

    def schedule(i):    # identical to baseline.main()'s --simulate schedule
        return {"kind": "normal", "severity": 0.0,
                "fr": 50.0 if (i // 8) % 2 == 0 else 30.0}

    old = baseline_mod.CREST_FLOOR_MARGIN
    try:
        baseline_mod.CREST_FLOOR_MARGIN = margin
        source = make_source(cfg, simulate=True, schedule=schedule)
        X, OP, learn_crest = baseline_mod.collect_features(
            source, fs_a, fs_v, learn_windows, progress=lambda _m: None)
        b = baseline_mod.fit_baseline(
            X, OP, list(FEATURE_NAMES), learn_crest=learn_crest)
    finally:
        baseline_mod.CREST_FLOOR_MARGIN = old

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sweep_baseline.npz"
        baseline_mod.save_baseline(path, b)
        scorer = MahalanobisScorer(path)
        m = evaluate.compute_metrics(scorer, windows=windows)

    return {"crest_floor": float(scorer.crest_floor),
            "deployed_threshold_fpr": float(m["deployed_threshold_fpr"]),
            "deployed_threshold_tpr": float(m["deployed_threshold_tpr"]),
            "auc": float(m["auc"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"seeds: machines {SEED_MACHINES}.., learn {SEED_LEARN}, "
               f"detection 7 (evaluate.py default)")
    ap.add_argument("--margins", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--machines", type=int, default=6)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--severity", type=float, nargs="+", default=[0.20],
                    help="fault severity for the F19 recovery side. 0.20 is "
                         "the only marginal one -- 0.35 saturates at 6/6 and "
                         "0.10 dies at 0/6, at EVERY margin. See f19_recovery.")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    rows = []
    for margin in args.margins:
        met = deployed_fpr(margin, windows=args.windows)
        rec = {s: f19_recovery(margin, n_machines=args.machines, severity=s)
               for s in args.severity}
        rows.append({"margin": margin,
                     "recovery": {str(s): f"{r}/{t}" for s, (r, t) in rec.items()},
                     **met})
        recs = "  ".join(f"sev{s:.2f} {r}/{t}" for s, (r, t) in rec.items())
        print(f"margin {margin:>4}  floor {met['crest_floor']:.3f}  {recs}  "
              f"FPR {met['deployed_threshold_fpr']:.4f}  "
              f"TPR {met['deployed_threshold_tpr']:.3f}", flush=True)

    heads = " | ".join(f"F19 recovery @ sev {s:.2f}" for s in args.severity)
    print(f"\n| `CREST_FLOOR_MARGIN` | deployed floor | {heads} | deployed FPR |")
    print("|---|---|" + "---|" * (len(args.severity) + 1))
    for r in rows:
        cells = " | ".join(r["recovery"][str(s)] for s in args.severity)
        print(f"| {r['margin']} | {r['crest_floor']:.3f} | {cells} | "
              f"{r['deployed_threshold_fpr']:.4f} |")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
