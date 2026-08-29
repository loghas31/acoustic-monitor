"""
T1.13 / F19 — two properties of the per-machine `crest_floor` that
`tests/test_crest_calibration.py` does not cover.

That file (written alongside the implementation) covers the calibration
function thoroughly: bounds, monotonicity, sparse-data fallback, backward
compatibility, and that a default-floor run still misses the F19 fault while a
calibrated one finds it. **Do not duplicate it here.** An earlier version of
this file repeated four of its tests; they were removed rather than left to
rot into two slightly different versions of the same assertion.

What remains are the two questions an implementation can pass all of those and
still get wrong — both about whether *lowering* the bar breaks something else:

1. does a calibrated floor make HEALTHY machines start picking bands? That is
   the failure mode a naive fix introduces, and F19 measured it: a global
   floor of 6.0 makes 12 of 14 healthy machines select a spurious band, which
   would feed the detector a different band every window and manufacture the
   false alarms the project's risk assessment (not in this public copy) calls churn risk #1.
2. do the calibrated floors actually DIFFER between machines? If every machine
   calibrated to the same number this would be a constant with extra steps,
   and the per-machine architecture would be unjustified complexity.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

from baseline import calibrate_crest_floor  # noqa: E402
from features import DEFAULT_BAND, select_demodulation_band  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "synth_phone_recording", ROOT / "ml" / "realdata" / "synth_phone_recording.py")
spr = importlib.util.module_from_spec(_spec)
sys.modules["synth_phone_recording"] = spr
_spec.loader.exec_module(spr)

FS = 16000.0
RESONANCE = 1600.0          # deliberately OUTSIDE DEFAULT_BAND (3-6 kHz)
WIN = 30.0
N = int(WIN * FS)


def _healthy_and_floor(seed: int, windows: int = 8):
    """Learn-period simulation: healthy audio, its per-window crests, and the
    floor this machine would calibrate to from them."""
    p = spr.make_pair(seed=seed, duration_s=WIN * windows, fs=FS,
                      resonance_hz=RESONANCE, severity=0.20)
    healthy = p["healthy"]
    crests = np.array([select_demodulation_band(healthy[i * N:(i + 1) * N], FS)[1]
                       for i in range(len(healthy) // N)])
    return healthy, crests, calibrate_crest_floor(crests)


def test_a_calibrated_floor_does_not_make_healthy_machines_pick_a_band():
    """Lowering the bar must not turn healthy noise into a band selection.

    Scored on the SAME windows the floor was calibrated from, which is the
    hardest case — if the floor were fitted too tightly to this machine's own
    learn data, it would trip here first.
    """
    for seed in (0, 1, 2, 3):
        healthy, _, floor = _healthy_and_floor(seed)
        picked = 0
        for i in range(len(healthy) // N):
            band, _ = select_demodulation_band(healthy[i * N:(i + 1) * N], FS,
                                               crest_floor=floor)
            picked += tuple(band) != tuple(DEFAULT_BAND)
        assert picked == 0, (
            f"seed {seed}: {picked} healthy windows selected a band under this "
            f"machine's own calibrated floor {floor:.2f} — each one is a "
            f"potential false alarm")


def test_calibrated_floors_actually_differ_between_machines():
    """If they were all equal, this would be a constant with extra steps.

    F19 measured healthy crest spanning 5.56-7.33 across machines, so the
    floors should spread too — a quiet machine earns a lower bar than a noisy
    one, which is the entire justification for the per-machine design.
    """
    # >= baseline.MIN_CREST_SAMPLES (8) windows, or calibration correctly
    # refuses to fit and every machine returns the default — which is the
    # sparse-data fallback working, not a spread. An earlier version of this
    # test used 6 and "failed" for exactly that reason.
    floors = [_healthy_and_floor(seed, windows=10)[2] for seed in range(5)]
    spread = max(floors) - min(floors)
    assert spread > 0.3, (
        f"calibrated floors span only {spread:.2f} "
        f"({[round(f, 2) for f in floors]}) — per-machine calibration would be "
        f"pointless if every machine landed on the same number")
