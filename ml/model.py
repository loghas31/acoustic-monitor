"""
Convolutional autoencoder for 40x32 log-Mel patches.

Why an autoencoder: we have zero labelled fault data (and so does every
customer on day one). Train it to reconstruct *normal* sound only. A fault
produces spectral structure the network has never seen, reconstruction fails,
and the error IS the anomaly score. No fault library required.

Why convolutional: fault signatures are local in time-frequency (a bearing
line is a horizontal stripe in the envelope-modulated bands; an imbalance
change is energy at one Mel bin). Conv layers share weights across the patch,
so the model stays small enough to quantise for a Pi Zero 2W.

Architecture (per brief): Conv 32-64-128 + maxpool -> Dense(64) bottleneck ->
mirrored ConvTranspose decoder. ~0.4 M params -> ~0.4 MB as a dynamic-range-
quantised .tflite.

DEVIATION FROM BRIEF (deliberate): this model is NOT trained on the device.
Full TensorFlow cannot run usefully in the Zero 2W's 512 MB; tflite-runtime is
inference-only by design. The device instead (a) collects normal-period
features, (b) fits the Isolation Forest + calibrates the AE threshold locally
(firmware/train.py). The AE itself is trained offline (this folder) on synthetic
+ pooled normal data. Same user experience; actually runs on the hardware.
"""

from __future__ import annotations

PATCH_SHAPE = (64, 32, 1)      # 64 Mel bins (v2, 0-8 kHz) x 32 frames
BOTTLENECK = 64


def build_autoencoder(lr: float = 1e-3):
    """Returns a compiled Keras model. Import of TF kept inside the function so
    firmware code can import this package without TF installed."""
    import tensorflow as tf
    from tensorflow.keras import layers as L

    inp = L.Input(shape=PATCH_SHAPE)
    # Encoder: 64x32 -> 32x16 -> 16x8 -> 8x4
    x = L.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = L.MaxPool2D(2)(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = L.MaxPool2D(2)(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = L.MaxPool2D(2)(x)

    x = L.Flatten()(x)                       # 8*4*128 = 4096
    z = L.Dense(BOTTLENECK, activation="relu", name="bottleneck")(x)
    # The bottleneck is the point of the exercise: 4096 -> 64 forces the network
    # to keep only the regularities of NORMAL operation. Anything it can't
    # squeeze through 64 numbers comes back wrong -> high reconstruction error.

    x = L.Dense(4096, activation="relu")(z)
    x = L.Reshape((8, 4, 128))(x)
    x = L.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = L.Conv2DTranspose(1, 3, strides=2, padding="same", activation="sigmoid")(x)
    # sigmoid + MSE: inputs are scaled to [0,1] by the robust scaler in
    # train_offline.py (stored in scaler.json and reused verbatim on-device).

    model = tf.keras.Model(inp, x, name="mel_patch_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def reconstruction_errors(model, patches):
    """Mean-squared reconstruction error per patch. Works for a Keras model."""
    import numpy as np
    recon = model.predict(patches, verbose=0)
    return np.mean((recon - patches) ** 2, axis=(1, 2, 3))
