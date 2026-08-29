"""
On-device "learn normal" — what actually runs in the customer's 24-72 h
baseline period. Triggered by button press, `--now`, or MQTT cmd start_learning.

What it does (and what it deliberately does NOT do):

    DOES: collect feature windows from THIS machine -> fit the Isolation
    Forest on its accel statistics -> push this machine's normal audio through
    the deployed (offline-trained) autoencoder -> set the AE threshold from the
    distribution of ITS OWN reconstruction errors.

    DOES NOT: train the autoencoder. Backprop through TF needs ~10x the RAM a
    Pi Zero 2W has. Per-machine threshold calibration on a shared AE gives most
    of the personalisation at ~none of the cost; collected features are also
    dumped to disk so the AE can be retrained offline per-site later if needed.

Output: baseline.json + isoforest.joblib next to the model file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from capture import make_source          # noqa: E402
from features import extract_all          # noqa: E402
from inference import AnomalyScorer       # noqa: E402

log = logging.getLogger("train")


def learn_normal(cfg: dict, source, scorer: AnomalyScorer, n_windows: int,
                 out_dir: Path, progress=print) -> dict:
    """Collect n_windows of (assumed-normal) operation and calibrate."""
    from sklearn.ensemble import IsolationForest
    import joblib

    if_vectors, ae_errors = [], []
    fs_a, fs_v = cfg["audio"]["sample_rate"], cfg["accelerometer"]["sample_rate"]

    for i, (audio, accel) in enumerate(source.windows()):
        if i >= n_windows:
            break
        feats = extract_all(audio, fs_a, accel, fs_v, cfg)
        if_vectors.append(feats["if_vector"])
        errs = scorer.ae_errors(feats["patches"])
        if errs is not None:
            ae_errors.extend(errs.tolist())
        progress(f"learning {i + 1}/{n_windows}")

    X = np.array(if_vectors)
    # contamination='auto' would assume outliers exist in training; during a
    # supervised learn period we assert the machine is healthy, so keep the
    # boundary tight but score against a calibrated threshold instead.
    forest = IsolationForest(n_estimators=100, random_state=0).fit(X)
    train_df = forest.decision_function(X)
    if_threshold = float(np.percentile(train_df, 0.5))    # 0.5 % of normal below this

    baseline = {"created": time.time(), "n_windows": len(if_vectors),
                "if_threshold": if_threshold}
    if ae_errors:
        e = np.array(ae_errors)
        baseline.update({
            "ae_threshold_percentile": float(np.percentile(e, 99.5)),
            "ae_threshold_sigma": float(e.mean() + cfg["anomaly"]["sigma_k"] * e.std()),
            "ae_error_mean": float(e.mean()), "ae_error_std": float(e.std()),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, out_dir / "isoforest.joblib")
    (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    np.save(out_dir / "learn_features.npy", X)            # for later offline retraining
    return baseline


def main() -> None:
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--windows", type=int, default=None, help="override learn_windows")
    p.add_argument("--artifacts", type=Path, default=ROOT.parent / "ml" / "artifacts")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    n = args.windows or cfg["window"]["learn_windows"]

    scorer = AnomalyScorer(args.artifacts / "model.tflite", args.artifacts / "scaler.json",
                           baseline_path=Path("/nonexistent"))   # calibrating: no baseline yet
    source = make_source(cfg, simulate=args.simulate)
    baseline = learn_normal(cfg, source, scorer, n, args.artifacts, log.info)
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
