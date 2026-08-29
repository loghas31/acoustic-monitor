#!/usr/bin/env python3
"""accel_axis_compare.py — did giving the accelerometer three real axes change
whether the detector works?

T1.8 replaced a simulated accelerometer whose axes were scaled copies of one
another (self-review F6) with three axes that share an impulse train but see
it through three different structural paths. That makes the accelerometer
features *capable of disagreeing*, which is the point. It does not follow that
detection improves, and the honest thing — the same thing T1.5 did for the
compositional fix — is to measure it rather than assume it.

METHOD
------
Two feature matrices are built over the SAME schedule, the SAME seeds and the
SAME audio channel, differing only in how the three accelerometer axes are
synthesised:

    legacy : accel = [ax, 0.6*ax + n1, 0.35*ax + n2]      (pre-T1.8)
    triaxial: capture.simulated_accel_axes(...)            (post-T1.8)

Then, for each representation independently, `n_boot` bootstrap splits of the
healthy pool into a learn period and a held-out remainder:

    fit_baseline(learn) -> MahalanobisScorer -> score(holdout healthy)
                                             -> score(faulty)

reporting held-out healthy false-positive rate (per 30 s window, before the
persistence gate) and ROC AUC. The comparison is PAIRED — split b uses the
same window indices for both representations — so the difference is not
contaminated by which windows happened to land in the learn period.

    python tools/accel_axis_compare.py                 # full run, ~2-3 min
    python tools/accel_axis_compare.py --stage features   # cache only
    python tools/accel_axis_compare.py --stage boot        # bootstrap only

The two stages exist because a bash call in the development sandbox times out
at ~3 minutes and background jobs do not survive between calls; features are
cached to /tmp between stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

CACHE = Path("/tmp/accel_axis_compare_features.npz")


# ----------------------------------------------------------------------------
# The two accelerometer models
# ----------------------------------------------------------------------------

from accel_axis_legacy import legacy_axes  # noqa: E402  (pre-T1.8 model)


def build_features(n_healthy=96, n_fault=24, severity=0.02, window_s=30.0,
                   fs_a=16000, fs_v=6400):
    """Both representations of the same windows. Severity 0.02 is deliberate:
    it is the weakest fault the repo has ever claimed to detect, so it is the
    operating point where a change in the feature vector could actually show
    up. At severity 0.5 everything separates and nothing is learned."""
    from capture import simulated_accel_axes
    from features import extract_features
    from simulate import (SimConfig, bearing_fault_signal, normal_signal)
    from baseline import operating_point

    rows = {"legacy": [], "triaxial": []}
    ops = {"legacy": [], "triaxial": []}
    labels = []

    plan = ([("normal", 0.0, 50.0 if i % 2 == 0 else 30.0, 9000 + i)
             for i in range(n_healthy)]
            + [("bearing_outer", severity, 50.0 if i % 2 == 0 else 30.0, 9500 + i)
               for i in range(n_fault)])

    for k, (kind, sev, fr, seed) in enumerate(plan):
        cfg = SimConfig(duration_s=window_s, fr=fr, fs_audio=fs_a, fs_accel=fs_v)
        gen = (normal_signal if kind == "normal"
               else lambda c, fs, r: bearing_fault_signal(c, fs, r, sev, "outer"))
        for name, maker in (("legacy", legacy_axes),
                            ("triaxial", simulated_accel_axes)):
            # identical seed -> identical audio and identical shared causes
            rng = np.random.default_rng(seed)
            audio = gen(cfg, fs_a, rng)
            accel = maker(kind, sev, fr, fs_v, window_s, rng, fs_a)
            f = extract_features(audio, fs_a, accel, fs_v)
            rows[name].append(f["vector"])
            ops[name].append(operating_point(f["vector"], f["fr_hz"]))
        labels.append(0 if kind == "normal" else 1)
        if (k + 1) % 12 == 0:
            print(f"  {k + 1}/{len(plan)} windows", flush=True)

    np.savez_compressed(
        CACHE,
        X_legacy=np.array(rows["legacy"]), OP_legacy=np.array(ops["legacy"]),
        X_tri=np.array(rows["triaxial"]), OP_tri=np.array(ops["triaxial"]),
        y=np.array(labels), severity=severity)
    print(f"cached -> {CACHE}")


# ----------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------

def _fpr_auc(X, OP, y, learn_idx, hold_idx, fault_idx):
    from baseline import fit_baseline
    from sklearn.metrics import roc_auc_score
    b = fit_baseline(X[learn_idx], OP[learn_idx])
    d_h = _score(b, X[hold_idx], OP[hold_idx])
    d_f = _score(b, X[fault_idx], OP[fault_idx])
    thr = np.array(b["thresholds"])
    fpr = float(np.mean([s > thr[r] for s, r in d_h]))
    scores = [s for s, _ in d_h] + [s for s, _ in d_f]
    lab = [0] * len(d_h) + [1] * len(d_f)
    return fpr, float(roc_auc_score(lab, scores))


def _score(b, X, OP):
    """Nearest-regime Mahalanobis distance, the same arithmetic
    `inference.MahalanobisScorer` performs, done in-process against a baseline
    dict so the bootstrap does not have to write 600 .npz files."""
    out = []
    centres = np.array(b["op_centroids"])
    op_mean, op_scale = np.array(b["op_mean"]), np.array(b["op_scale"])
    g_mean, g_std = np.array(b["global_mean"]), np.array(b["global_std"])
    for x, op in zip(X, OP):
        opz = (op - op_mean) / op_scale
        r = int(np.argmin(((centres - opz) ** 2).sum(axis=1)))
        z = (x - g_mean) / g_std
        d = z - np.array(b["means"])[r]
        d2 = float(d @ np.array(b["precisions"])[r] @ d)
        out.append((np.sqrt(max(d2, 0.0)), r))
    return out


def bootstrap(n_boot=200, n_learn=48, seed=0):
    """`n_learn` is TOTAL windows, and the healthy pool alternates two speed
    regimes, so 48 is 24 per regime — the operating point `firmware/baseline.py
    --simulate --windows 48` actually ships. At 24 total (12 per regime) the
    held-out FPR is ~0.55 for BOTH representations: 12 samples in 37 dimensions
    is not enough to fit even a Ledoit-Wolf covariance, and the comparison then
    measures small-sample failure rather than the accelerometer."""
    z = np.load(CACHE)
    y = z["y"]
    healthy = np.flatnonzero(y == 0)
    fault = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    res = {"legacy": {"fpr": [], "auc": []}, "triaxial": {"fpr": [], "auc": []}}
    for b in range(n_boot):
        perm = rng.permutation(healthy)
        learn, hold = perm[:n_learn], perm[n_learn:]
        for name, xk, ok in (("legacy", "X_legacy", "OP_legacy"),
                             ("triaxial", "X_tri", "OP_tri")):
            try:
                fpr, auc = _fpr_auc(z[xk], z[ok], y, learn, hold, fault)
            except Exception as e:                       # noqa: BLE001
                print(f"  split {b} {name}: {type(e).__name__}: {e}")
                continue
            res[name]["fpr"].append(fpr)
            res[name]["auc"].append(auc)
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{n_boot} splits", flush=True)

    summary = {}
    for name in res:
        f = np.array(res[name]["fpr"])
        a = np.array(res[name]["auc"])
        summary[name] = dict(fpr_mean=float(f.mean()), fpr_sd=float(f.std()),
                             auc_mean=float(a.mean()), auc_sd=float(a.std()),
                             n=len(f))
    d = np.array(res["triaxial"]["fpr"]) - np.array(res["legacy"]["fpr"])
    da = np.array(res["triaxial"]["auc"]) - np.array(res["legacy"]["auc"])
    summary["paired_fpr_diff"] = dict(
        mean=float(d.mean()),
        ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    summary["paired_auc_diff"] = dict(
        mean=float(da.mean()),
        ci95=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))])
    summary["severity"] = float(z["severity"])
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["features", "boot", "all"], default="all")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-learn", type=int, default=48)
    p.add_argument("--n-healthy", type=int, default=96)
    p.add_argument("--severity", type=float, default=0.02)
    a = p.parse_args()
    if a.stage in ("features", "all"):
        build_features(n_healthy=a.n_healthy, severity=a.severity)
    if a.stage in ("boot", "all"):
        bootstrap(n_boot=a.n_boot, n_learn=a.n_learn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
