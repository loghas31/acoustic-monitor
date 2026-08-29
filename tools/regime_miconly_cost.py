#!/usr/bin/env python3
"""regime_miconly_cost.py — what did the mic-only regime-splitting bug cost?

Backlog T1.9, self-review F7. This is the evidence behind a frozen-file edit to
`firmware/baseline.py`, so it lives in the repo rather than in a scratch
directory: the claim "6.3x the false alarms" has to be re-runnable.

Two experiments.

**A — the null table.** How often does `choose_k` invent regimes in data that
has none, as a function of how many directions the operating-point cloud
varies in? The pre-T1.9 rule (silhouette >= 0.5 plus a cluster-size floor) is
reproduced here as `legacy_choose_k` so the two can be compared on identical
clouds. This is a geometry experiment: no feature extraction, seconds to run.

**B — the cost, end to end.** 96 healthy + 24 faulty (severity 0.02) MIC-ONLY
windows from the real simulator through the real feature extractor, then
`n_boot` bootstrap learn/held-out splits of the healthy pool, scored the way
`inference.MahalanobisScorer` scores. Three arms on the SAME splits:

    as-is      : whatever choose_k currently does
    legacy     : the pre-T1.9 rule, i.e. the bug
    forced k=1 : the oracle, since these windows contain exactly one regime

Reported: held-out healthy false-positive rate per window (before the
persistence gate), ROC AUC, and the distribution of k. Features are cached to
/tmp between runs because extraction is ~0.17 s per window and the development
sandbox times a bash call out at ~3 minutes.

    python tools/regime_miconly_cost.py              # both experiments
    python tools/regime_miconly_cost.py --stage null # A only (fast)
    python tools/regime_miconly_cost.py --stage cost # B only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

import baseline as B  # noqa: E402

CACHE = Path("/tmp/regime_miconly_features.npz")
FS_A, FS_V, WINDOW_S = 16000, 6400, 30.0


# ----------------------------------------------------------------------------
# A. the null table
# ----------------------------------------------------------------------------

def legacy_choose_k(OPz: np.ndarray) -> tuple[int, np.ndarray]:
    """`choose_k` exactly as it stood before T1.9: silhouette + size floor."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_s, best_labels = 1, -1.0, np.zeros(len(OPz), dtype=int)
    for k in B.K_RANGE:
        if len(OPz) < k * 8:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(OPz)
        if len(set(km.labels_)) < 2:
            continue
        if np.bincount(km.labels_, minlength=k).min() < B.MIN_REGIME_WINDOWS:
            continue
        s = silhouette_score(OPz, km.labels_)
        if s > best_s:
            best_k, best_s, best_labels = k, s, km.labels_
    if best_s < B.SILHOUETTE_MIN:
        return 1, np.zeros(len(OPz), dtype=int)
    return best_k, best_labels


def null_table(n: int = 48, trials: int = 400) -> None:
    """No regimes are present in any of these clouds. Every k > 1 is a mistake."""
    print(f"A. invented regimes on single-cluster noise  (n={n}, {trials} trials)\n")
    print(f"{'cloud':>12} {'eff dims':>9} {'P(k>1) legacy':>15} {'P(k>1) now':>12}")
    cases = (("1 live dim", 1, False), ("2 live dims", 2, False),
             ("3 live dims", 3, False), ("2 collinear", 2, True))
    for name, live, collinear in cases:
        rng = np.random.default_rng(21)
        old, new, eds = [], [], []
        for _ in range(trials):
            Z = np.zeros((n, 3))
            for d in range(live):
                Z[:, d] = rng.standard_normal(n)
            if collinear:
                Z[:, 1] = 0.98 * Z[:, 0] + 0.02 * Z[:, 1]
            old.append(legacy_choose_k(Z)[0] > 1)
            new.append(B.choose_k(Z)[0] > 1)
            eds.append(B.effective_dims(Z))
        print(f"{name:>12} {int(np.median(eds)):>9} {np.mean(old):>15.3f} "
              f"{np.mean(new):>12.3f}")
    print("\n  A mic-only node on a fixed-speed machine is the '1 live dim' row:\n"
          "  fr constant, accel log-RMS pinned at the -9.0 dead-channel sentinel.")


# ----------------------------------------------------------------------------
# B. the cost, on the real pipeline
# ----------------------------------------------------------------------------

def build_features(n_healthy: int = 96, n_fault: int = 24,
                   severity: float = 0.02) -> None:
    """Severity 0.02 is the weakest fault the repo claims to detect, so it is
    the operating point where a clustering change could plausibly hurt
    detection as well as false alarms."""
    from features import extract_features
    from simulate import SimConfig, bearing_fault_signal, normal_signal

    t0 = time.time()
    X, OP, y = [], [], []
    plan = ([("normal", 0.0, 6000 + i) for i in range(n_healthy)]
            + [("fault", severity, 6500 + i) for i in range(n_fault)])
    dead_accel = np.zeros((int(WINDOW_S * FS_V), 3))     # mic-only build
    for kind, sev, seed in plan:
        cfg = SimConfig(duration_s=WINDOW_S, fr=50.0, fs_audio=FS_A, fs_accel=FS_V)
        rng = np.random.default_rng(seed)
        audio = (normal_signal(cfg, FS_A, rng) if kind == "normal"
                 else bearing_fault_signal(cfg, FS_A, rng, sev, "outer"))
        f = extract_features(audio, FS_A, dead_accel, FS_V)
        X.append(f["vector"])
        OP.append(B.operating_point(f["vector"], f["fr_hz"]))
        y.append(0 if kind == "normal" else 1)
    np.savez(CACHE, X=np.array(X), OP=np.array(OP), y=np.array(y),
             severity=severity)
    print(f"  extracted {len(y)} mic-only windows in {time.time() - t0:.1f} s "
          f"-> {CACHE}")


def _score(b: dict, Xs: np.ndarray, OPs: np.ndarray):
    """The arithmetic of `MahalanobisScorer.score`, without needing a file."""
    opz = (OPs - b["op_mean"]) / b["op_scale"]
    r = np.argmin(((b["op_centroids"][None, :, :] - opz[:, None, :]) ** 2).sum(-1),
                  axis=1)
    Z = (Xs - b["global_mean"]) / b["global_std"]
    d = np.array([float(np.sqrt(max(
        (Z[i] - b["means"][r[i]]) @ b["precisions"][r[i]] @ (Z[i] - b["means"][r[i]]),
        0.0))) for i in range(len(Z))])
    return d, b["thresholds"][r]


def cost(n_boot: int = 100, n_learn: int = 48) -> None:
    from sklearn.metrics import roc_auc_score

    if not CACHE.exists():
        build_features()
    z = np.load(CACHE)
    X, OP, y = z["X"], z["OP"], z["y"]
    healthy, faulty = np.flatnonzero(y == 0), np.flatnonzero(y == 1)

    print(f"\nB. cost of the split  ({n_boot} bootstrap learn periods of "
          f"{n_learn} mic-only windows, one unchanging machine)\n")
    print(f"{'rule':>12} {'k=1':>5} {'k=2':>5} {'k=3':>5} {'k=4':>5} "
          f"{'held-out healthy FPR':>23} {'AUC':>7}")

    arms = {
        "as-is": None,
        "legacy": legacy_choose_k,
        "forced k=1": lambda OPz: (1, np.zeros(len(OPz), dtype=int)),
    }
    original = B.choose_k
    for label, replacement in arms.items():
        B.choose_k = replacement or original
        try:
            rng = np.random.default_rng(11)
            fprs, aucs, ks = [], [], []
            for _ in range(n_boot):
                perm = rng.permutation(healthy)
                learn, hold = perm[:n_learn], perm[n_learn:]
                b = B.fit_baseline(X[learn], OP[learn])
                dh, th = _score(b, X[hold], OP[hold])
                df, tf = _score(b, X[faulty], OP[faulty])
                fprs.append(float(np.mean(dh > th)))
                aucs.append(roc_auc_score(
                    np.r_[np.zeros(len(dh)), np.ones(len(df))],
                    np.r_[dh / th, df / tf]))
                ks.append(int(b["k"]))
            counts = np.bincount(ks, minlength=5)[1:5]
            print(f"{label:>12} {counts[0]:>5} {counts[1]:>5} {counts[2]:>5} "
                  f"{counts[3]:>5} {np.mean(fprs):>14.4f} +/- {np.std(fprs):<6.4f} "
                  f"{np.mean(aucs):>7.4f}")
        finally:
            B.choose_k = original
    print("\n  AUC is 1.000 in every arm: at severity 0.02 the fault is still\n"
          "  obvious. The damage is entirely to the false-alarm rate, which is\n"
          "  churn risk #1 — each spurious regime fits a 37-dim Gaussian to\n"
          f"  ~{n_learn // 2} windows instead of {n_learn}.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("null", "cost", "all"), default="all")
    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--boot", type=int, default=100)
    args = p.parse_args()

    if args.stage in ("null", "all"):
        null_table(trials=args.trials)
    if args.stage in ("cost", "all"):
        cost(n_boot=args.boot)


if __name__ == "__main__":
    main()
