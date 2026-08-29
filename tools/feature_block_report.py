#!/usr/bin/env python3
"""feature_block_report.py — backlog T1.10 / self-review F9.

F9 measured that two of the five feature blocks (the audio and accel band-ILR
blocks) are close to one-dimensional on simulated data. It did not measure the
other three blocks with the same yardstick, and it only ever looked at
BEARING faults. This script closes both gaps in one place:

  1. EFFECTIVE DIMENSIONALITY of all five feature blocks, under three
     conditions: the healthy two-speed learn period `baseline.py` actually
     collects, a bearing-fault severity ramp, and an IMBALANCE severity ramp
     (new — F9 never tried a fault that is not envelope-shaped).

  2. BLOCK-WISE DETECTION POWER: train a per-block Mahalanobis distance on
     healthy-only windows, score held-out healthy + faulty windows, and report
     ROC AUC per block per fault kind. This is the answer to "which features
     earn their place" — a block can be full rank (part 1) and still carry no
     information about a given fault, or be low rank and still be exactly
     where all the signal lives (this is expected for the envelope block).

Run:  python tools/feature_block_report.py
Writes nothing; this is a measurement tool, like accel_axis_report.py. Numbers
quoted in docs/DOC_PIPELINE.md and docs/DOC_STATUS.md come from this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

from capture import SimulatedSource              # noqa: E402
from features import FEATURE_NAMES, extract_features  # noqa: E402

FS_A, FS_V = 16000, 6400
WINDOW_S = 8.0          # short window: this probes spectral SHAPE, not timing;
                        # matches the convention in test_compositional.py (6 s)
                        # and accel_axis_report.py (4-8 s).

BLOCKS = {
    "audio_stat":     [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("audio_stat_")],
    "accel_stat":     [i for i, n in enumerate(FEATURE_NAMES)
                       if n.startswith("accel_") and "band" not in n],
    "audio_band_ilr": [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("audio_band_ilr_")],
    "accel_band_ilr": [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("accel_band_ilr_")],
    "envelope":       [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("env_")],
}
assert sorted(sum(BLOCKS.values(), [])) == list(range(37)), "blocks must partition the vector"


def effective_rank(X: np.ndarray):
    """exp(entropy of the normalised singular-value spectrum) of the
    STANDARDISED block, plus the raw spectrum.

    Same convention as `tools/accel_axis_report.py` and `test_threshold.py`
    (independently re-derived twice already in this repo — kept local rather
    than imported so this stays a self-contained measurement script, matching
    the existing pattern). Read alongside the sv ratio: T1.5 found effective
    rank can move in the "wrong" direction under a change that provably lost
    no information, because it measures how variance is spread, not how much
    is present."""
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-30)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    er = float(np.exp(-(p * np.log(p)).sum()))
    return er, s


def _matrix(cases, seed0=5000):
    """Extract the full 37-dim vector for a list of (kind, severity, fr)."""
    rows = []
    for i, (kind, sev, fr) in enumerate(cases):
        src = SimulatedSource(WINDOW_S, FS_A, FS_V,
                              lambda j, k=kind, s=sev, f=fr: {"kind": k, "severity": s, "fr": f},
                              seed=seed0 + i)
        audio, accel = next(iter(src.windows()))
        rows.append(extract_features(audio, FS_A, accel, FS_V)["vector"])
    return np.array(rows)


def healthy_two_speed(n=48):
    return [("normal", 0.0, 50.0 if i % 2 == 0 else 30.0) for i in range(n)]


def fault_ramp(kind, n=40, sev_max=0.5, fr=50.0):
    return [(kind, sev_max * (i + 1) / n, fr) for i in range(n)]


def part1_dimensionality():
    print("=" * 78)
    print("1. EFFECTIVE DIMENSIONALITY PER BLOCK, THREE CONDITIONS")
    print("=" * 78)
    conditions = {
        "healthy (2-speed learn period)": (healthy_two_speed(48), 20000),
        "bearing_outer ramp (0->0.5)":    (fault_ramp("bearing_outer", 40, 0.5), 21000),
        "imbalance ramp (0->1.0)":        (fault_ramp("imbalance", 40, 1.0), 22000),
    }
    results = {}
    for cond_name, (cases, seed0) in conditions.items():
        X = _matrix(cases, seed0=seed0)
        print(f"\n-- {cond_name} --")
        results[cond_name] = {}
        for block_name, idx in BLOCKS.items():
            er, sv = effective_rank(X[:, idx])
            ratio = sv[-1] / sv[0]
            results[cond_name][block_name] = (er, ratio)
            print(f"  {block_name:>16s} (D={len(idx):2d}): "
                  f"effective rank {er:5.2f}   sv ratio {ratio:9.2e}")
    return results


def _ledoit_wolf_d2(train, test):
    from sklearn.covariance import LedoitWolf
    mu, sd = train.mean(0), train.std(0) + 1e-12
    tr, te = (train - mu) / sd, (test - mu) / sd
    lw = LedoitWolf().fit(tr)
    prec = lw.precision_
    diff = te - lw.location_
    return np.einsum("ij,jk,ik->i", diff, prec, diff)


def part2_block_auc():
    from sklearn.metrics import roc_auc_score

    print()
    print("=" * 78)
    print("2. BLOCK-WISE DETECTION POWER (held-out AUC, healthy vs faulty)")
    print("=" * 78)
    print("Train: 48 healthy windows (24 per speed, matches baseline.py's default")
    print("learn period). Test: 48 fresh healthy + a severity ramp per fault kind.")
    print("Each block is scored on ITS OWN Mahalanobis distance, fit independently")
    print("-- this isolates what that block alone can tell healthy from faulty,")
    print("with no help from the other 30 features.\n")

    train = _matrix(healthy_two_speed(48), seed0=10000)
    test_healthy = _matrix(healthy_two_speed(48), seed0=11000)

    fault_kinds = {
        "bearing_outer (envelope fault)": fault_ramp("bearing_outer", 48, 0.5),
        "imbalance (1x tonal growth)":    fault_ramp("imbalance", 48, 1.0),
    }
    results = {}
    for fault_name, cases in fault_kinds.items():
        test_fault = _matrix(cases, seed0=12000)
        y = np.concatenate([np.zeros(len(test_healthy)), np.ones(len(test_fault))])
        print(f"-- {fault_name} --")
        results[fault_name] = {}
        block_names = list(BLOCKS) + ["FULL (37-dim, reference)"]
        for block_name in block_names:
            idx = BLOCKS[block_name] if block_name in BLOCKS else list(range(37))
            d2 = np.concatenate([
                _ledoit_wolf_d2(train[:, idx], test_healthy[:, idx]),
                _ledoit_wolf_d2(train[:, idx], test_fault[:, idx]),
            ])
            auc = roc_auc_score(y, d2)
            results[fault_name][block_name] = auc
            flag = "  <-- carries the fault signal" if auc > 0.9 else (
                   "  (no better than chance)" if auc < 0.6 else "")
            print(f"  {block_name:>28s}: AUC {auc:.3f}{flag}")
        print()
    return results


def main() -> int:
    r1 = part1_dimensionality()
    r2 = part2_block_auc()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("A block 'earns its place' if it is EITHER high-rank (genuinely varies)")
    print("OR high-AUC (carries fault signal despite being low-rank) for at least")
    print("one fault kind tested. Low-rank AND low-AUC on every fault kind tested")
    print("means: on this simulator, that block is along for the ride.\n")
    for block_name in BLOCKS:
        healthy_er, _ = r1["healthy (2-speed learn period)"][block_name]
        aucs = [r2[f][block_name] for f in r2]
        verdict = "EARNS ITS PLACE" if max(aucs) > 0.9 else "ALONG FOR THE RIDE (so far)"
        print(f"  {block_name:>16s}: healthy eff. rank {healthy_er:5.2f}, "
              f"max AUC across tested faults {max(aucs):.3f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
