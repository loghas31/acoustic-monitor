"""
evaluate.py — Stage-3 exit evidence for the v1 detector
(Mahalanobis + regimes + persistence gating).

Three claims, each tested against the trained baseline (firmware/baseline.npz):

  1. DETECTION  — 70 % healthy / 30 % faulty windows across two regimes:
                  ROC AUC >= 0.95.
  2. REGIMES    — alternating regime switches raise ZERO anomalous windows.
  3. GATING     — a single-window transient raises no alert; a persistent
                  fault raises exactly one.

Run:  python ml/evaluate.py [--windows 40]
Writes ml/artifacts/roc.png + metrics.json. The v1.5 autoencoder evaluation
lives with the cloud-side code (model.py / train_offline.py) and is not part
of the v1 gate.

BACKLOG T1.14 / SELF-REVIEW F20: `compute_metrics()` below is the part of this
module `tests/test_evaluate_pinned.py` imports directly, so `deployed_threshold_fpr`
(and TPR, and the regime false-alarm count) are pinned against the REAL deployed
`firmware/baseline.npz`, not just printed to a JSON file nobody diffs. F20 measured
that T1.13's calibrated `crest_floor` moved this number 0.000 -> 0.107 while every
existing check (AUC, gating counts) stayed green — this refactor exists so that
class of silent regression fails a test instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "firmware"))
ARTIFACTS = ROOT / "artifacts"

from baseline import operating_point                     # noqa: E402
from capture import SimulatedSource                       # noqa: E402
from features import extract_features                     # noqa: E402
from inference import AlertGate, MahalanobisScorer          # noqa: E402


def score_windows(scorer, cases: list[dict], seed: int = 9000):
    """cases: list of schedule dicts. Returns (scores, anomalous_flags, regimes)."""
    # clamp: the generator computes window i before our loop's break fires
    src = SimulatedSource(30.0, 16000, 6400,
                          schedule=lambda i: cases[min(i, len(cases) - 1)], seed=seed)
    scores, anoms, regimes = [], [], []
    for i, (audio, accel) in enumerate(src.windows()):
        if i >= len(cases):
            break
        feats = extract_features(audio, 16000, accel, 6400, crest_floor=scorer.crest_floor)
        s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
        scores.append(s["score"] / s["threshold"])   # normalise so regimes share a scale
        anoms.append(s["anomalous"])
        regimes.append(s["regime"])
    return np.array(scores), np.array(anoms), np.array(regimes)


def _build_detection_cases(n: int, rng: np.random.Generator):
    """Same 70/30 healthy/fault schedule `main()` always used, factored out so
    `compute_metrics` and any test can build the identical cases."""
    n_fault = round(0.3 * n)
    cases, labels = [], []
    sevs = [0.1, 0.15, 0.3, 0.5]
    for i in range(n - n_fault):
        cases.append({"kind": "normal", "severity": 0.0,
                      "fr": 50.0 if i % 2 == 0 else 30.0})
        labels.append(0)
    for i in range(n_fault):
        cases.append({"kind": "bearing_outer", "severity": sevs[i % len(sevs)],
                      "fr": 50.0 if i % 2 == 0 else 30.0})
        labels.append(1)
    order = rng.permutation(len(cases))
    cases = [cases[j] for j in order]
    labels = np.array(labels)[order]
    return cases, labels


def compute_metrics(scorer, windows: int = 40, seed: int = 7) -> dict:
    """Everything `main()` prints and gates on, minus the plot — a pure
    function so a test can assert pinned numbers against the real deployed
    baseline without touching matplotlib or the filesystem.

    Every default (windows=40, detection rng seed=7, switch-case seed=12000,
    AlertGate need=4, the transient/persistent schedules) is unchanged from
    the pre-T1.14 `main()` so this refactor cannot itself move a number.
    """
    from sklearn.metrics import auc, roc_curve

    rng = np.random.default_rng(seed)

    # ---- 1. detection: 70/30 across two regimes ----------------------------
    cases, labels = _build_detection_cases(windows, rng)
    scores, anoms, regimes = score_windows(scorer, cases)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    op_fpr = float(np.mean(anoms[labels == 0]))
    op_tpr = float(np.mean(anoms[labels == 1]))

    # ---- 2. regime switches must be free ------------------------------------
    switch_cases = [{"kind": "normal", "severity": 0.0,
                     "fr": 50.0 if i % 2 == 0 else 30.0} for i in range(12)]
    _, switch_anoms, switch_regimes = score_windows(scorer, switch_cases, seed=12000)
    regime_switch_false_alarms = int(switch_anoms.sum())

    # ---- 3. persistence gating ----------------------------------------------
    gate = AlertGate(need=4)
    transient = [False, False, True, False] + [False] * 8       # 1-window blip
    persistent = [False] * 4 + [True] * 8                       # real episode
    alerts_transient = sum(gate.feed(a) for a in transient)
    gate2 = AlertGate(need=4)
    alerts_persistent = sum(gate2.feed(a) for a in persistent)

    metrics = {
        "auc": round(roc_auc, 4),
        "deployed_threshold_fpr": op_fpr, "deployed_threshold_tpr": op_tpr,
        "n_normal": int((labels == 0).sum()), "n_fault": int((labels == 1).sum()),
        "regime_switch_false_alarms": regime_switch_false_alarms,
        "regimes_visited_during_switching": sorted(set(int(r) for r in switch_regimes)),
        "gating_alerts_transient": int(alerts_transient),
        "gating_alerts_persistent": int(alerts_persistent),
    }
    # carried for main()'s plot only; harmless extra keys for any caller that
    # just wants the JSON-shaped metrics (tests can pop them or ignore them)
    metrics["_plot_data"] = {"fpr": fpr, "tpr": tpr, "scores": scores, "labels": labels}
    return metrics


def stage3_gate_passes(metrics: dict) -> bool:
    return (metrics["auc"] >= 0.95 and metrics["regime_switch_false_alarms"] == 0
            and metrics["gating_alerts_transient"] == 0
            and metrics["gating_alerts_persistent"] == 1)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=40)
    p.add_argument("--baseline", type=Path,
                   default=ROOT.parent / "firmware" / "baseline.npz")
    args = p.parse_args()

    scorer = MahalanobisScorer(args.baseline)
    metrics = compute_metrics(scorer, windows=args.windows)
    plot_data = metrics.pop("_plot_data")
    fpr, tpr = plot_data["fpr"], plot_data["tpr"]
    scores, labels = plot_data["scores"], plot_data["labels"]
    op_fpr, op_tpr = metrics["deployed_threshold_fpr"], metrics["deployed_threshold_tpr"]
    roc_auc = metrics["auc"]

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "k--", lw=0.5)
    axes[0].scatter([op_fpr], [op_tpr], c="r", zorder=5,
                    label=f"deployed thresholds\nFPR={op_fpr:.2f} TPR={op_tpr:.2f}")
    axes[0].set(title=f"ROC (AUC = {roc_auc:.3f})", xlabel="FPR", ylabel="TPR")
    axes[0].legend(fontsize=8)
    axes[1].hist(scores[labels == 0], bins=24, alpha=0.6, label="healthy", density=True)
    axes[1].hist(scores[labels == 1], bins=24, alpha=0.6, label="fault", density=True)
    axes[1].axvline(1.0, c="r", ls="--", label="threshold (normalised)")
    axes[1].set(title="score / threshold", xlabel="normalised score")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "roc.png", dpi=120)

    print(json.dumps(metrics, indent=2))
    print("STAGE 3 GATE:", "PASS" if stage3_gate_passes(metrics) else "FAIL")


if __name__ == "__main__":
    main()
