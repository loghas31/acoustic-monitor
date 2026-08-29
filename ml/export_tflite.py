"""
Convert the trained Keras AE to a quantised .tflite for the Pi.

Dynamic-range quantisation (weights -> int8, activations stay float):
~4x smaller, ~2-3x faster on Cortex-A53, and — unlike full-int8 — needs no
representative dataset and introduces negligible error in the *reconstruction
error*, which is the quantity we actually threshold. Full int8 is available
behind --int8 if we ever need more speed; it requires re-calibrating the
threshold afterwards (the README explains why).

Run:  python ml/export_tflite.py            # writes ml/artifacts/model.tflite
      python ml/export_tflite.py --check    # + parity check vs the Keras model
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


def export(int8: bool = False) -> Path:
    import tensorflow as tf
    model = tf.keras.models.load_model(ARTIFACTS / "model.keras")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from train_offline import make_normal_patches, apply_scaler
        scaler = json.loads((ARTIFACTS / "scaler.json").read_text())
        rep = apply_scaler(make_normal_patches(0.5), scaler)[..., None].astype(np.float32)

        def rep_gen():
            for i in range(0, min(200, len(rep))):
                yield [rep[i:i + 1]]
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    blob = conv.convert()
    out = ARTIFACTS / "model.tflite"
    out.write_bytes(blob)
    print(f"wrote {out}  ({len(blob)/1024:.0f} KiB)")
    return out


def parity_check(n: int = 64) -> None:
    """Same inputs through Keras and tflite; the anomaly scores must agree.
    If quantisation shifted scores systematically, the stored threshold would
    be wrong — this check is what catches that before it ships."""
    import tensorflow as tf
    from train_offline import make_normal_patches, apply_scaler
    from model import reconstruction_errors

    scaler = json.loads((ARTIFACTS / "scaler.json").read_text())
    X = apply_scaler(make_normal_patches(0.5, seed=99), scaler)[..., None].astype(np.float32)[:n]

    keras_model = tf.keras.models.load_model(ARTIFACTS / "model.keras")
    keras_err = reconstruction_errors(keras_model, X)

    interp = tf.lite.Interpreter(model_path=str(ARTIFACTS / "model.tflite"))
    interp.allocate_tensors()
    i_in = interp.get_input_details()[0]["index"]
    i_out = interp.get_output_details()[0]["index"]
    tfl_err = np.empty(len(X))
    for i, x in enumerate(X):
        interp.set_tensor(i_in, x[None])
        interp.invoke()
        recon = interp.get_tensor(i_out)[0]
        tfl_err[i] = np.mean((recon - x) ** 2)

    rel = np.abs(tfl_err - keras_err) / (keras_err + 1e-12)
    print(f"parity: median rel. score diff = {np.median(rel)*100:.2f} %  "
          f"max = {rel.max()*100:.2f} %")
    assert np.median(rel) < 0.05, "quantisation moved anomaly scores > 5 % — recalibrate threshold"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--int8", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    export(args.int8)
    if args.check:
        parity_check()
