#!/usr/bin/env python3
"""
dcase_eval.py — run THIS project's detector against the DCASE 2020 Task 2
development data (real industrial pumps/fans, labelled normal vs anomalous)
and report whether it separates them.

    STATUS: executed 2026-08-23 against SYNTHETIC DCASE-shaped data
    (`--self-test`). Never yet run against the real download.

    It was originally committed having never been run at all — the shell
    sandbox was wedged for three days. That banner is now retired, but the
    distinction it protected still matters:

      * The machinery is verified. `--self-test` builds a tree with DCASE's
        exact layout from `ml/simulate.py`, runs the real learn→score path,
        and reports AUC 1.0000 with 0 % false alarms. That proves the code
        runs; it says NOTHING about real pumps.
      * Executing it found a real bug in it: the summary hardcoded
        "Everything above is REAL machine audio", which was false on
        self-test data. Fixed. Writing a script and running a script are
        different activities, and this is what the difference looks like.
      * The clip-splice mitigation was checked rather than assumed: with
        `--fade-ms 0`, AUC stayed 1.0000 and the score ratios barely moved
        (normal 0.48x → 0.64x, anomalous 38.1x → 37.0x). **The joins are not
        doing the work.** Re-run that check on real data too.

WHY THIS DATASET
---------------
DCASE 2020 Task 2's development set is MIMII and ToyADMOS, already
preprocessed into the form this project wants:

  * single channel  (MIMII ships 8; DCASE already took channel 0)
  * 16 kHz          (exactly firmware/config.yaml's audio.sample_rate)
  * labelled        (train/ is all normal; test/ has normal_* and anomaly_*)
  * ~1 GB per machine type, versus 7.7 GB for the raw MIMII equivalent

    https://zenodo.org/records/3678171   (dev_data_pump.zip, 1.0 GB)

⚠ LICENCE: DCASE 2020 Task 2 dev data is **CC BY-NC-SA 4.0 — NonCommercial**.
Fine for validating, for a dissertation, and for deciding whether this works.
NOT fine as training data inside a product you sell. If a result from here
ends up in a commercial pitch, re-run it on raw MIMII (CC BY-SA 4.0, no NC
clause) and quote that number instead. See docs/REAL_DATA_SOURCES.md.

THE SPLICING PROBLEM, AND WHY IT IS HANDLED
-------------------------------------------
DCASE clips are 10 s. This pipeline's window is 30 s and its learn period
needs 48 of them. So clips must be concatenated — and a naive concatenation
puts a step discontinuity every 10 s, which is broadband, impulsive, and
looks *exactly* like the bearing impact the detector is built to find. That
would be a self-inflicted fault signature in every window.

Two mitigations, both applied:
  1. A short raised-cosine fade (default 5 ms) at every clip edge, so joins
     are continuous rather than stepped.
  2. Clips are grouped in exact multiples of the window, so every window
     contains the same number of joins. Any residual artefact is then
     COMMON-MODE — present in normal and anomalous windows alike — so it
     cannot by itself create a difference between the two classes.

This is a mitigation, not a proof. If the results look too good, suspect the
joins first: rerun with --fade-ms 0 and see whether the numbers move. If they
move a lot, the splices are doing the work, not the physics.

USAGE
-----
    unzip dev_data_pump.zip            # gives ./pump/train and ./pump/test
    python tools/dcase_eval.py ./pump --machine-id 00

    python tools/dcase_eval.py ./pump --machine-id 00 --fade-ms 0   # the check
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("firmware", "ml"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

from capture import FileSource                                       # noqa: E402
from features import FEATURE_NAMES, extract_features                 # noqa: E402
from baseline import (collect_features, fit_baseline,                 # noqa: E402
                      operating_point, save_baseline)
from inference import MahalanobisScorer                              # noqa: E402

SR = 16000


class _ListSource:
    """Same adapter phone_monitor.py uses: feeds a pre-sliced window list to
    `collect_features`, which expects a `.windows()` iterator."""

    def __init__(self, windows):
        self._windows = windows

    def windows(self):
        return iter(self._windows)


def _read_clip(path: Path) -> np.ndarray:
    """One DCASE clip as float32 in [-1, 1]. 16 kHz, 16-bit, mono by spec —
    anything else is a surprise worth crashing on rather than resampling
    silently, because a wrong sample rate would move every frequency in the
    analysis without any visible error."""
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if sr != SR:
        raise ValueError(
            f"{path.name}: {sr} Hz, expected {SR}. This is not DCASE 2020 "
            f"Task 2 data, or it has been resampled.")
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    return np.asarray(x, dtype=np.float32)


def _fade(x: np.ndarray, n: int) -> np.ndarray:
    """Raised-cosine fade in and out over `n` samples, so concatenated clips
    join continuously instead of stepping. A step is a broadband impulse and
    this detector hunts broadband impulses."""
    if n <= 0 or len(x) < 2 * n:
        return x
    w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    y = x.copy()
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y


def _concat(clips: list[Path], fade_ms: float, window_s: float) -> np.ndarray:
    """Concatenate clips, trimmed to a whole number of windows so every window
    holds the same number of joins (see the splicing note in the docstring)."""
    n_fade = int(round(fade_ms * 1e-3 * SR))
    parts = [_fade(_read_clip(c), n_fade) for c in clips]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    x = np.concatenate(parts)
    per_window = int(round(window_s * SR))
    usable = (len(x) // per_window) * per_window
    return x[:usable]


def _windows_from(clips: list[Path], fade_ms: float, window_s: float,
                  fs_accel_default: int):
    """Concatenate → temp WAV → FileSource → windows.

    Deliberately routed through FileSource, the same reader the firmware and
    phone_monitor.py use, rather than hand-building (audio, accel) tuples.
    That keeps the mic-only / dead-accelerometer handling (finding F2) on the
    tested path instead of reimplementing it here untested.
    """
    from scipy.io import wavfile
    x = _concat(clips, fade_ms, window_s)
    if len(x) == 0:
        return []
    tmp = Path(tempfile.mkdtemp()) / "concat.wav"
    wavfile.write(tmp, SR, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
    src = FileSource(tmp, None, window_s=window_s,
                     fs_accel_default=fs_accel_default)
    return list(src.windows()), src


def _sorted(d: Path, pattern: str) -> list[Path]:
    return sorted(d.glob(pattern))


def _build_self_test_tree(dest: Path, window_s: float) -> Path:
    """Write a synthetic tree with DCASE's exact layout and naming, so this
    script can be EXECUTED without the 1 GB download.

    This exists because the first version of this file was committed having
    never been run — the sandbox was wedged — and "untested code in the repo"
    is the one thing this project does not tolerate. A self-test that needs no
    download means the next person can verify the machinery in 90 seconds.

    It tests the SCRIPT, not the detector. The signals come from
    ml/simulate.py, so a good score here says the plumbing works and says
    nothing whatsoever about real pumps.
    """
    from scipy.io import wavfile
    sys.path.insert(0, str(ROOT / "ml"))
    from simulate import SimConfig, bearing_fault_signal, normal_signal

    train, test = dest / "train", dest / "test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)

    def _w(path: Path, x: np.ndarray) -> None:
        x = x / (np.max(np.abs(x)) + 1e-12) * 0.7
        wavfile.write(path, SR, (x * 32767).astype(np.int16))

    # Enough 10 s clips for a 48-window learn period plus 20 windows a side.
    n_train = int(np.ceil(48 * window_s / 10.0)) + 12
    n_test = int(np.ceil(20 * window_s / 10.0))
    for i in range(n_train):
        _w(train / f"normal_id_00_{i:08d}.wav",
           normal_signal(SimConfig(duration_s=10.0, seed=1000 + i), SR,
                         np.random.default_rng(1000 + i)))
    for i in range(n_test):
        _w(test / f"normal_id_00_{i:08d}.wav",
           normal_signal(SimConfig(duration_s=10.0, seed=5000 + i), SR,
                         np.random.default_rng(5000 + i)))
        _w(test / f"anomaly_id_00_{i:08d}.wav",
           bearing_fault_signal(SimConfig(duration_s=10.0, seed=9000 + i), SR,
                                np.random.default_rng(9000 + i), severity=0.35))
    print(f"self-test tree: {n_train} train, {n_test} test-normal, "
          f"{n_test} test-anomaly clips in {dest}")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine_dir", type=Path, nargs="?",
                    help="unzipped machine folder, e.g. ./pump (contains train/ and test/)")
    ap.add_argument("--self-test", action="store_true",
                    help="generate a synthetic DCASE-shaped tree and run against "
                         "it. Proves the SCRIPT works with no download. Says "
                         "nothing about real machines.")
    ap.add_argument("--machine-id", default="00",
                    help="DCASE Machine ID, e.g. 00, 02, 04, 06")
    ap.add_argument("--learn-windows", type=int, default=48,
                    help="48 x 30 s = 24 min. DOC_STATUS measured that fewer "
                         "than 48 gives a 55-59%% held-out false-alarm rate.")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--fade-ms", type=float, default=5.0,
                    help="clip-join fade. Rerun with 0 to test whether the "
                         "joins are doing the work.")
    ap.add_argument("--fs-accel-default", type=int, default=6400)
    a = ap.parse_args(argv)

    synthetic = bool(a.self_test)
    if synthetic:
        a.machine_dir = _build_self_test_tree(
            Path(tempfile.mkdtemp(prefix="dcase_selftest_")) / "pump", a.window_s)
        a.machine_id = "00"
    elif a.machine_dir is None:
        ap.error("give a machine_dir, or use --self-test")

    train_d, test_d = a.machine_dir / "train", a.machine_dir / "test"
    for d in (train_d, test_d):
        if not d.is_dir():
            print(f"FAIL: no {d}. Unzip dev_data_<machine>.zip and pass the "
                  f"machine folder (the one containing train/ and test/).",
                  file=sys.stderr)
            return 1

    mid = a.machine_id
    train = _sorted(train_d, f"normal_id_{mid}_*.wav")
    t_norm = _sorted(test_d, f"normal_id_{mid}_*.wav")
    t_anom = _sorted(test_d, f"anomaly_id_{mid}_*.wav")
    if not train or not t_norm or not t_anom:
        print(f"FAIL: machine id {mid!r} gave train={len(train)} "
              f"test-normal={len(t_norm)} test-anomaly={len(t_anom)}. "
              f"Try another --machine-id (00, 02, 04, 06 are typical).",
              file=sys.stderr)
        return 1

    print(f"machine dir : {a.machine_dir}   id {mid}")
    print(f"clips       : train {len(train)}, test normal {len(t_norm)}, "
          f"test anomaly {len(t_anom)}")

    # -- learn ---------------------------------------------------------------
    learn_windows, learn_src = _windows_from(train, a.fade_ms, a.window_s,
                                             a.fs_accel_default)
    if len(learn_windows) < a.learn_windows:
        print(f"FAIL: only {len(learn_windows)} learn windows available, need "
              f"{a.learn_windows}. Use more clips or a shorter --window-s.",
              file=sys.stderr)
        return 1
    learn_windows = learn_windows[:a.learn_windows]
    X, OP, learn_crest = collect_features(_ListSource(learn_windows), learn_src.fs_audio,
                                          learn_src.fs_accel, len(learn_windows))
    baseline = fit_baseline(X, OP, feature_names=FEATURE_NAMES, learn_crest=learn_crest)
    print(f"learned     : {len(learn_windows)} windows, "
          f"{len(np.atleast_1d(baseline['thresholds']))} regime(s), "
          f"thresholds {np.round(np.atleast_1d(baseline['thresholds']), 3).tolist()}")

    # -- score ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        bpath = Path(td) / "baseline.npz"
        save_baseline(bpath, baseline)
        scorer = MahalanobisScorer(bpath)

        def score_all(clips):
            wins, src = _windows_from(clips, a.fade_ms, a.window_s,
                                      a.fs_accel_default)
            out = []
            for audio, accel in wins:
                f = extract_features(audio, src.fs_audio, accel, src.fs_accel,
                                     crest_floor=scorer.crest_floor)
                op = operating_point(f["vector"], f["fr_hz"])
                s = scorer.score(f["vector"], op)
                out.append((s["score"], s["threshold"], bool(s["anomalous"])))
            return out

        norm = score_all(t_norm)
        anom = score_all(t_anom)

    if not norm or not anom:
        print("FAIL: no scored windows. Too few clips for one 30 s window?",
              file=sys.stderr)
        return 1

    ns = np.array([r[0] for r in norm])
    as_ = np.array([r[0] for r in anom])
    fpr = float(np.mean([r[2] for r in norm]))
    tpr = float(np.mean([r[2] for r in anom]))

    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(ns)), np.ones(len(as_))]
    auc = float(roc_auc_score(y, np.r_[ns, as_]))

    print()
    print(f"  windows scored     : {len(ns)} normal, {len(as_)} anomalous")
    print(f"  median score       : normal {np.median(ns):8.2f}   "
          f"anomalous {np.median(as_):8.2f}")
    print(f"  ratio to threshold : normal {np.median(ns)/norm[0][1]:6.2f}x  "
          f"anomalous {np.median(as_)/anom[0][1]:6.2f}x")
    print(f"  flagged            : normal {fpr:6.1%} (false alarms)   "
          f"anomalous {tpr:6.1%} (caught)")
    print(f"  ROC AUC            : {auc:.4f}")
    print()
    if auc >= 0.9:
        print("  Separates the two classes well on this machine.")
    elif auc >= 0.7:
        print("  Partial separation. Real, but not the AUC 1.000 the "
              "synthetic data gives.")
    else:
        print("  Does NOT separate them. This is the most useful result "
              "available: record it in RESULTS.md, then find out why.")
    print()
    if synthetic:
        print("  ⚠ SYNTHETIC. This was --self-test data from ml/simulate.py, not "
              "a real machine.")
        print("  It proves this SCRIPT works. It predicts nothing about DCASE, "
              "and nothing about a fridge.")
    else:
        print("  This is whatever you pointed it at. If that was DCASE 2020 "
              "Task 2 data, it is real machine audio AND NonCommercial-"
              "licensed — see the licence note at the top of this file.")
    print("  Sanity check before believing it: rerun with --fade-ms 0. If the "
          "numbers move much, the clip joins are doing the work, not the "
          "physics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
