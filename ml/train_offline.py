"""
train_offline.py — v1.5 (OPTIONAL, CLOUD-SIDE) autoencoder training.

NOT part of the v1 device. v1 detection is the Mahalanobis baseline in
firmware/, which needs no TensorFlow anywhere. This script exists for the
v1.5 upgrade path: train a convolutional autoencoder server-side on a
customer's uploaded learn-period audio, quantise to .tflite
(export_tflite.py), push to the device for INFERENCE ONLY via tflite-runtime.
TFLite cannot train and full TF does not fit in the Pi's 512 MB — the
split is architectural.

Requires: pip install tensorflow (server only).
Run:      python ml/train_offline.py [--minutes 10] [--epochs 15]
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

from simulate import SimConfig, normal_signal                  # noqa: E402
from features import log_mel, stft_mag                         # noqa: E402

ARTIFACTS = ROOT / "artifacts"
PATCH_FRAMES, PATCH_HOP = 32, 16


def mel_patches(audio: np.ndarray, fs: float) -> np.ndarray:
    """(n_patches, 64, 32) overlapping log-Mel patches — the AE's input."""
    _, mag = stft_mag(audio, fs)
    mel = log_mel(mag, fs)
    n = mel.shape[1]
    if n < PATCH_FRAMES:
        raise ValueError(f"need >= {PATCH_FRAMES} frames, got {n}")
    starts = range(0, n - PATCH_FRAMES + 1, PATCH_HOP)
    return np.stack([mel[:, s:s + PATCH_FRAMES] for s in starts])


def make_normal_patches(minutes: float, seed: int = 0) -> np.ndarray:
    """Synthetic normal operation with per-window speed wander, so the model
    learns 'this machine', not one frozen waveform."""
    out = []
    rng_master = np.random.default_rng(seed)
    for i in range(int(minutes * 2)):                      # 30 s windows
        fr = 50.0 * (1 + 0.01 * rng_master.standard_normal())
        cfg = SimConfig(duration_s=30.0, fr=fr)
        audio = normal_signal(cfg, cfg.fs_audio, np.random.default_rng(seed + i))
        out.append(mel_patches(audio, cfg.fs_audio))
    return np.concatenate(out)


def fit_scaler(patches: np.ndarray) -> dict:
    """Robust [0,1] scaling (1st/99th percentile) — one loud transient in the
    training audio must not compress the whole dynamic range."""
    return {"lo": float(np.percentile(patches, 1)),
            "hi": float(np.percentile(patches, 99))}


def apply_scaler(patches: np.ndarray, scaler: dict) -> np.ndarray:
    return np.clip((patches - scaler["lo"]) / (scaler["hi"] - scaler["lo"] + 1e-9), 0, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from model import build_autoencoder, reconstruction_errors   # imports TF

    patches = make_normal_patches(args.minutes, args.seed)
    np.random.default_rng(args.seed).shuffle(patches)
    scaler = fit_scaler(patches)
    X = apply_scaler(patches, scaler)[..., None].astype(np.float32)
    n_val = max(int(0.15 * len(X)), 1)
    X_train, X_val = X[:-n_val], X[-n_val:]
    print(f"patches: train={len(X_train)} val={len(X_val)}")

    model = build_autoencoder()
    hist = model.fit(X_train, X_train, validation_data=(X_val, X_val),
                     epochs=args.epochs, batch_size=args.batch, verbose=2)

    # Percentile threshold on held-out errors (recon errors are right-skewed;
    # a Gaussian mean+3sigma rule underestimates the tail).
    val_err = reconstruction_errors(model, X_val)
    stats = {"threshold_percentile": float(np.percentile(val_err, 99.5)),
             "mean": float(val_err.mean()), "std": float(val_err.std())}

    ARTIFACTS.mkdir(exist_ok=True)
    model.save(ARTIFACTS / "model.keras")
    (ARTIFACTS / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (ARTIFACTS / "ae_threshold.json").write_text(json.dumps(stats, indent=2))
    (ARTIFACTS / "training_log.json").write_text(json.dumps(
        {"loss": [float(v) for v in hist.history["loss"]],
         "val_loss": [float(v) for v in hist.history["val_loss"]]}, indent=2))
    print(f"saved -> {ARTIFACTS}/model.keras  threshold={stats['threshold_percentile']:.6f}")


if __name__ == "__main__":
    main()
