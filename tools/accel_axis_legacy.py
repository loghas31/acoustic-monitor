"""accel_axis_legacy.py — the pre-T1.8 simulated accelerometer, kept on purpose.

This is the model self-review finding F6 was about, reproduced verbatim from
`firmware/capture.SimulatedSource` as it stood before 2026-08-18:

    ax    = gen(fs_accel, rng)
    noise = 0.03 * std(ax)
    accel = [ax, 0.6*ax + noise*N(0,1), 0.35*ax + noise*N(0,1)]

It is NOT dead code and it is NOT used by the firmware. It exists so that
`tools/accel_axis_compare.py` can measure the new axis model against what the
repo actually did, rather than against a paraphrase of it written from memory —
which is how "the fix improved things" gets asserted instead of measured.

Measured properties of this model, for the record:
    inter-axis correlation      r(x,y) 0.9988, r(x,z) 0.9964, r(y,z) 0.9952
    12 accel statistics span    effective rank 3.75 of 12
    smallest/largest sv         1.3e-3  (a four-dimensional near-null space)

Delete this file only when the simulator is retired in favour of real
recordings, i.e. after H3.
"""

from __future__ import annotations

import numpy as np


def legacy_axes(kind: str, severity: float, fr: float, fs: int,
                window_s: float, rng, fs_audio: int = 16000,
                snr_db: float | None = None) -> np.ndarray:
    """Same signature as `capture.simulated_accel_axes`, so the two are
    interchangeable in a comparison harness."""
    from simulate import (SimConfig, bearing_fault_signal, imbalance_signal,
                          normal_signal)

    cfg = SimConfig(duration_s=window_s, fr=fr, fs_audio=fs_audio, fs_accel=fs)
    if snr_db is not None:
        cfg.snr_db = snr_db
    gens = {
        "normal": lambda r: normal_signal(cfg, fs, r),
        "bearing_outer": lambda r: bearing_fault_signal(cfg, fs, r, severity, "outer"),
        "bearing_inner": lambda r: bearing_fault_signal(cfg, fs, r, severity, "inner"),
        "imbalance": lambda r: imbalance_signal(cfg, fs, r, severity),
    }
    ax = gens[kind](rng)
    noise = 0.03 * np.std(ax)
    return np.column_stack([
        ax,
        0.6 * ax + noise * rng.standard_normal(len(ax)),
        0.35 * ax + noise * rng.standard_normal(len(ax)),
    ])
