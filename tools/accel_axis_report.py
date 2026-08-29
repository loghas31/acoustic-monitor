#!/usr/bin/env python3
"""accel_axis_report.py — is the simulated accelerometer measuring three things
or one thing three times?

Self-review finding F6 said the second, and it was right. This script is the
measurement, kept in the repo so the claim can be re-checked in one command
after any change to `capture.simulated_accel_axes` — and, more importantly, so
that the SAME numbers can be computed on the first real recording and compared.

    python tools/accel_axis_report.py

Four measurements, in increasing order of how much they tell you:

  1. inter-axis correlation of the raw signals — the crudest check, and the
     one that fails loudest when the axes are copies.
  2. effective dimensionality of the 12 per-axis accel statistics in the
     feature vector — the number that actually matters, because those 12
     columns are what the Mahalanobis distance sees.
  3. R^2 of each y/z statistic regressed on the four x statistics — which
     individual features are along for the ride.
  4. the envelope-spectrum repetition rate recovered from each axis in its own
     demodulation band — the property that must SURVIVE decorrelation, because
     one shaft and one defect means one fault frequency.

MEASURED BEFORE T1.8 (axes were `[ax, 0.6*ax + n, 0.35*ax + n]`):
    r(x,y) 0.9988   r(x,z) 0.9964   r(y,z) 0.9952
    12 accel statistics: effective rank 3.75 of 12, smallest/largest sv 1.3e-3

WHAT TO DO WITH THE REAL RECORDING (H2/H3)
------------------------------------------
Run this on a real triaxial capture. If the real inter-axis correlations are
far from what `ACCEL_AXES` produces, replace those constants with the measured
ones — `firmware/bench/check_mount.py`'s tap test gives f0 and Q per axis
directly — rather than leaving a plausible invention in the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

from capture import ACCEL_AXES, SimulatedSource            # noqa: E402
from features import (FEATURE_NAMES, envelope_spectrum,    # noqa: E402
                      extract_features)
from simulate import SimConfig                             # noqa: E402

FS_A, FS_V = 16000, 6400


def _win(kind, sev, fr=50.0, seed=99, window_s=4.0):
    src = SimulatedSource(window_s, FS_A, FS_V,
                          lambda i: {"kind": kind, "severity": sev, "fr": fr},
                          seed=seed)
    return next(iter(src.windows()))


def _accel_stat_columns():
    return [i for i, n in enumerate(FEATURE_NAMES)
            if n.startswith("accel_") and "band" not in n]


def _effective_rank(X):
    """exp(entropy of the normalised singular-value spectrum).

    Reported alongside the raw spectrum and a hard singular-value ratio,
    because T1.5 established the hard way that effective rank measures how
    variance is SPREAD, not how much information is present — it fell 17.4/40
    to 9.0/37 across a change that provably lost none. Read all three."""
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-30)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), s


def _feature_matrix(kind, sev, n=40, seed0=2000, ramp=False):
    idx = _accel_stat_columns()
    rows = []
    for w in range(n):
        fr = 50.0 if w % 2 == 0 else 30.0
        s = sev * (w / n) if ramp else sev
        audio, accel = _win(kind, s, fr, seed=seed0 + w)
        rows.append(extract_features(audio, FS_A, accel, FS_V)["vector"][idx])
    return np.array(rows), [FEATURE_NAMES[i] for i in idx]


def main() -> int:
    print("=" * 72)
    print("1. INTER-AXIS CORRELATION OF THE RAW SIGNALS")
    print("=" * 72)
    for kind, sev in [("normal", 0.0), ("bearing_outer", 0.15),
                      ("bearing_outer", 0.5), ("bearing_inner", 0.3),
                      ("imbalance", 0.5)]:
        _, a = _win(kind, sev)
        C = np.corrcoef(a.T)
        print(f"  {kind:>14s} sev={sev:<5} r(x,y)={C[0,1]:+.4f} "
              f"r(x,z)={C[0,2]:+.4f} r(y,z)={C[1,2]:+.4f}")
    print("  (before T1.8: +0.9988 / +0.9964 / +0.9952 for every case)")

    print()
    print("=" * 72)
    print("2. EFFECTIVE DIMENSIONALITY OF THE 12 PER-AXIS ACCEL STATISTICS")
    print("=" * 72)
    for label, kind, sev, ramp in [("healthy", "normal", 0.0, False),
                                   ("fault ramp", "bearing_outer", 0.2, True)]:
        X, _ = _feature_matrix(kind, sev, ramp=ramp)
        er, sv = _effective_rank(X)
        print(f"  {label:>10s}: effective rank {er:5.2f} of 12   "
              f"smallest/largest sv {sv[-1]/sv[0]:.3e}")
        print(f"              {np.array2string(sv, precision=2, suppress_small=True)}")
    print("  (before T1.8, healthy: effective rank 3.75, sv ratio 1.3e-3)")

    print()
    print("=" * 72)
    print("3. R^2 OF EACH y/z STATISTIC ON THE FOUR x STATISTICS")
    print("=" * 72)
    for label, kind, sev, ramp in [("healthy", "normal", 0.0, False),
                                   ("fault ramp", "bearing_outer", 0.3, True)]:
        X, names = _feature_matrix(kind, sev, seed0=3000, ramp=ramp)
        A = np.column_stack([X[:, :4], np.ones(len(X))])
        r2 = []
        for j in range(4, 12):
            y = X[:, j]
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2.append(1 - np.sum((y - A @ beta) ** 2)
                      / (np.sum((y - y.mean()) ** 2) + 1e-30))
        print(f"  -- {label} --")
        for j, v in zip(range(4, 12), r2):
            flag = "  <-- still determined" if v > 0.95 else ""
            print(f"     {names[j]:>16s}  R^2 = {v:.4f}{flag}")
        print(f"     median {np.median(r2):.3f}, "
              f"{int(sum(v > 0.95 for v in r2))} of 8 above 0.95")

    print()
    print("=" * 72)
    print("4. THE REPETITION RATE MUST SURVIVE: BPFO FROM EACH AXIS")
    print("=" * 72)
    cfg = SimConfig()
    bpfo = cfg.bearing.bpfo(50.0)
    _, accel = _win("bearing_outer", 0.6, window_s=8.0)
    f0_x = min(cfg.resonance_hz, 0.4 * FS_V)
    print(f"  true BPFO at fr = 50 Hz: {bpfo:.2f} Hz")
    for j, a in enumerate("xyz"):
        m = ACCEL_AXES[a]
        f0 = m["f0_ratio"] * f0_x
        bw = f0 / max(2.0, m["q_ratio"] * cfg.resonance_q)
        band = (f0 - 3 * bw, f0 + 3 * bw)
        f, sp = envelope_spectrum(accel[:, j], FS_V, band)
        sel = (f > 50.0) & (f < 400.0)
        peak = float(f[sel][np.argmax(sp[sel])])
        ratio = float(sp[sel].max() / np.median(sp[sel]))
        print(f"  axis {a}: band {band[0]:6.0f}-{band[1]:6.0f} Hz -> "
              f"peak {peak:7.2f} Hz ({100*(peak-bpfo)/bpfo:+.2f} %), "
              f"{ratio:5.1f}x background")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
