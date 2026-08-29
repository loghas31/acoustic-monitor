"""Tests for ml/realdata/validate_public_dataset.py.

Scope note: these guard the parts of the public-dataset validator that would
fail SILENTLY and produce a beautiful, meaningless number —

  * the healthy/faulty classification (mislabel one class and the AUC is
    fiction);
  * the resampling that bridges 12 kHz dataset data to our 16 kHz / 6.4 kHz
    production rates (get the ratio wrong and every frequency axis stretches);
  * the train/test time split (leak one window and the AUC is optimistic).

The fault-frequency identity and the ingest path get their own coverage in
tests/test_realdata.py (backlog T1.3); we deliberately do not duplicate them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "realdata"))

import validate_public_dataset as V  # noqa: E402


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,label,race,size", [
    ("97.mat", 0, None, None),
    ("100.mat", 0, None, None),
    ("Normal_2.mat", 0, None, None),
    ("baseline_healthy.npz", 0, None, None),
    ("IR007_0.mat", 1, "inner", 0.007),
    ("OR021@6_1.mat", 1, "outer", 0.021),
    ("B014_3.mat", 1, "ball", 0.014),
])
def test_classify_name(name, label, race, size):
    info = V.classify_name(Path(name))
    assert info["label"] == label
    assert info["race"] == race
    if size is None:
        assert info["fault_size_in"] is None
    else:
        assert info["fault_size_in"] == pytest.approx(size)


def test_unknown_name_fails_pessimistic():
    """An unrecognised filename must be treated as FAULTY, never as healthy.

    Direction matters: a healthy file wrongly counted as faulty drags the
    reported AUC down (we look worse than we are). A faulty file wrongly
    counted as healthy would poison the learned baseline AND inflate the
    metric. Only one of those two errors is survivable."""
    assert V.classify_name(Path("mystery_recording_42.mat"))["label"] == 1


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

def test_resample_lengths_and_identity():
    x = np.random.default_rng(0).standard_normal(12000)
    assert V.resample_to(x, 12000, 12000) is not None
    assert len(V.resample_to(x, 12000, 12000)) == 12000       # no-op path
    assert len(V.resample_to(x, 12000, 16000)) == 16000       # up 4/3
    assert len(V.resample_to(x, 12000, 6400)) == 6400         # down 8/15


def test_resample_preserves_tone_frequency():
    """A 1 kHz tone must still be at 1 kHz after 12 kHz -> 16 kHz.

    This is the test that catches an inverted up/down ratio, which is the
    classic resampling bug and is invisible downstream: every feature still
    computes, the band ratios just quietly describe a different machine."""
    fs_in, fs_out, f0 = 12000.0, 16000.0, 1000.0
    t = np.arange(int(2 * fs_in)) / fs_in
    y = V.resample_to(np.sin(2 * np.pi * f0 * t), fs_in, fs_out)
    freqs = np.fft.rfftfreq(len(y), 1.0 / fs_out)
    peak = freqs[int(np.argmax(np.abs(np.fft.rfft(y * np.hanning(len(y))))))]
    assert peak == pytest.approx(f0, rel=0.01)


def test_to_recording_maps_channels_to_device_rates():
    rf = V.RawFile(path=Path("x.mat"), fs=12000.0,
                   de=np.random.default_rng(1).standard_normal(24000),
                   fe=np.random.default_rng(2).standard_normal(24000))
    rec = V.to_recording(rf, "device")
    assert rec.fs_audio == V.FS_AUDIO_DEVICE
    assert rec.fs_accel == V.FS_ACCEL_DEVICE
    # durations must match to within a window of slack, or windows desynchronise
    assert abs(rec.duration_s - len(rec.accel) / rec.fs_accel) < 0.05

    rec_native = V.to_recording(rf, "native")
    assert rec_native.fs_audio == 12000.0 and rec_native.fs_accel == 12000.0


# ---------------------------------------------------------------------------
# the train/test split must not leak
# ---------------------------------------------------------------------------

def test_train_and_heldout_windows_are_disjoint_in_time():
    """Every train window must end before every held-out window begins."""
    fs = 1000.0
    rec = V.Recording(audio=np.arange(10 * int(fs), dtype=float), fs_audio=fs,
                      accel=np.arange(10 * int(fs), dtype=float)[:, None],
                      fs_accel=fs)
    train = list(V._windows(rec, 1.0, 0.5, 0.0, 0.6))
    held = list(V._windows(rec, 1.0, 1.0, 0.6, 1.0))
    assert train and held
    last_train_sample = max(float(w[0][-1]) for w in train)
    first_held_sample = min(float(w[0][0]) for w in held)
    assert last_train_sample < first_held_sample


def test_overlap_is_only_used_in_the_train_slice():
    """Held-out windows are generated with hop == window, so consecutive
    windows share no samples. If this ever regresses, every reported AUC and
    FPR silently becomes optimistic."""
    fs = 1000.0
    rec = V.Recording(audio=np.arange(10 * int(fs), dtype=float), fs_audio=fs,
                      accel=np.arange(10 * int(fs), dtype=float)[:, None],
                      fs_accel=fs)
    held = [w[0] for w in V._windows(rec, 1.0, 1.0, 0.6, 1.0)]
    for a, b in zip(held, held[1:]):
        assert not np.intersect1d(a, b).size


# ---------------------------------------------------------------------------
# end-to-end smoke test on a tiny surrogate
# ---------------------------------------------------------------------------

def test_end_to_end_on_tiny_surrogate(tmp_path):
    """Runs discover -> resample -> features -> fit_baseline -> scorer on a
    3-second surrogate. Deliberately does NOT assert a good AUC: this proves
    the plumbing, and asserting a detection number on synthetic data we wrote
    ourselves would be exactly the circularity the script exists to escape."""
    data_dir = tmp_path / "surr"
    V.make_surrogate(data_dir, dur_s=3.0, seed=11)

    cfg = V.Config(data_dir=data_dir, out_dir=tmp_path / "out",
                   window_s=0.5, train_hop_s=0.25, surrogate=True,
                   max_faulty_files=2)
    res = V.run(cfg)

    assert res["n_healthy_files"] == 4
    assert res["n_faulty_files"] == 2
    assert res["n_train_windows"] > res["n_heldout_windows"] > 0
    assert res["n_faulty_windows"] > 0
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert res["surrogate"] is True
    assert (tmp_path / "out" / "public_dataset_result.json").exists()
    assert (tmp_path / "out" / "public_baseline.npz").exists()
    # the surrogate warning must survive into the machine-readable output
    assert any("SURROGATE" in n for n in res["notes"])
