#!/usr/bin/env python3
"""
phone_monitor.py — T7.2: run the ACTUAL product pipeline (learn -> score) on
ONE real recording, with no accelerometer and no known bearing geometry.

WHY THIS IS A DIFFERENT TOOL FROM ml/realdata/analyse_recording.py
--------------------------------------------------------------------------
analyse_recording.py answers the Week-2 GATE question: does the envelope
signature appear at the COMPUTED BPFO, comparing a healthy and a faulty
recording of the SAME machine whose bearing geometry you know? That needs a
seeded fault and a caliper. A phone recording of your fridge has neither.

This tool answers the question a single recording of an ordinary appliance
can actually answer, and the one the product itself is built around: after
LEARNING from part of the recording, does the anomaly detector agree that the
rest is more of the same? No bearing geometry, no seeded fault, no "before
and after" — exactly the zero-knowledge install the product promises (see
the system overview (not in this public copy) §3).

It is deliberately simple: split the recording into 30 s windows, fit a
baseline (firmware.baseline.fit_baseline — the SAME function the real
firmware calls) on the first --learn-windows of them, then score the rest
with firmware.inference.MahalanobisScorer — again, the same class the
firmware runs. There is no separate "phone" code path to trust; this is the
real pipeline run against a real file.

USAGE
-----
Record ~30-60+ minutes of a machine running on your phone (voice memo app,
default quality is fine), then:

    python tools/ingest.py my_fridge.m4a --mic-only --out-dir data/real
    python tools/phone_monitor.py data/real/my_fridge.wav --learn-windows 48

--learn-windows 48 (24 minutes) is the documented minimum — DOC_STATUS.md
measured that 24 windows for a 37-dim feature vector gives a 55-59 % held-out
false-alarm rate, i.e. useless; 48 gives ~3 %. Use fewer only to prove the
tool runs, never to judge whether the machine is healthy.

Self-test with no recording at all, to prove the tool itself works and to
show what a HEALTHY run should look like on a signal with no periodic
mechanical fault in it:

    python tools/phone_monitor.py --self-test

Every number that command produces is SYNTHETIC — see the generator's own
docstring below. It exists to test the CODE, not to predict what your fridge
will do.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("firmware",):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

from capture import FileSource                                    # noqa: E402
from features import DEFAULT_BAND, FEATURE_NAMES, extract_features  # noqa: E402
from baseline import collect_features, fit_baseline, operating_point, save_baseline  # noqa: E402
from inference import MahalanobisScorer                            # noqa: E402


class _ListSource:
    """Adapts a pre-sliced list of (audio, accel) windows to the `.windows()`
    protocol `collect_features` expects, so the learn slice and the score
    slice of ONE recording can be fed through the same real functions without
    re-reading the file twice."""

    def __init__(self, windows):
        self._windows = windows

    def windows(self):
        return iter(self._windows)


def analyse(wav_path: Path, csv_path: Path | None = None, window_s: float = 30.0,
           learn_windows: int = 48, fs_accel_default: int = 6400,
           max_windows: int | None = None, progress=None) -> tuple[list[dict], dict]:
    """Runs the real product pipeline end to end on one recording.

    Returns (rows, summary). `rows` has one dict per SCORED window (i.e. every
    window after the learn period) with the band selector's decision, the
    speed estimate, the regime, the score and whether it crossed threshold.
    `summary` answers the three questions T7.2 asks for by name.
    """
    src = FileSource(wav_path, csv_path, window_s=window_s,
                     fs_accel_default=fs_accel_default)
    windows = list(src.windows())
    if max_windows:
        windows = windows[:max_windows]
    n = len(windows)
    if n <= learn_windows:
        raise ValueError(
            f"only {n} windows ({n * window_s / 60:.1f} min) in this "
            f"recording after slicing into {window_s:g} s windows; need MORE "
            f"than --learn-windows {learn_windows} ({learn_windows * window_s / 60:.1f} "
            f"min) to have anything left to score. Record for longer, or "
            f"lower --learn-windows to prove the tool runs (but see the "
            f"module docstring: fewer than 48 windows does not tell you "
            f"whether the machine is healthy, only whether the code works).")

    quiet = progress or (lambda msg: None)
    learn_src = _ListSource(windows[:learn_windows])
    X, OP, learn_crest = collect_features(learn_src, src.fs_audio, src.fs_accel,
                                          learn_windows, progress=quiet)
    # T1.13 / F19: this tool is the exact case the finding was measured on —
    # mic-only, pink-noise-realistic recordings, where the fixed crest_floor
    # of 10.0 misses most real faults. Calibrate it from this recording's own
    # learn slice rather than deploying the old constant unexamined.
    baseline = fit_baseline(X, OP, feature_names=FEATURE_NAMES, learn_crest=learn_crest)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        bpath = Path(td) / "baseline.npz"
        save_baseline(bpath, baseline)
        scorer = MahalanobisScorer(bpath)

        for i, (audio, accel) in enumerate(windows[learn_windows:], start=learn_windows):
            out = extract_features(audio, src.fs_audio, accel, src.fs_accel,
                                   crest_floor=scorer.crest_floor)
            op = operating_point(out["vector"], out["fr_hz"])
            sc = scorer.score(out["vector"], op)
            band_fired = tuple(out["band"]) != DEFAULT_BAND
            rows.append({
                "window": i,
                "band_lo_hz": round(float(out["band"][0]), 1),
                "band_hi_hz": round(float(out["band"][1]), 1),
                "band_crest": round(float(out["band_crest"]), 2),
                "band_fired": bool(band_fired),
                "fr_hz": round(float(out["fr_hz"]), 2),
                "fr_reliable": bool(out["fr_reliable"]),
                "regime": sc["regime"],
                "score": round(sc["score"], 3),
                "threshold": round(sc["threshold"], 3),
                "score_over_threshold": round(sc["score"] / sc["threshold"], 3)
                                        if sc["threshold"] > 0 else float("inf"),
                "anomalous": bool(sc["anomalous"]),
            })
            quiet(f"scored window {i + 1}/{n}")

    n_scored = len(rows)
    n_fired = sum(r["band_fired"] for r in rows)
    n_reliable = sum(r["fr_reliable"] for r in rows)
    n_anom = sum(r["anomalous"] for r in rows)
    summary = {
        "recording": str(wav_path),
        "mic_only": csv_path is None,
        "total_windows": n,
        "learn_windows": learn_windows,
        "scored_windows": n_scored,
        # Q1: did the demodulation band selector fire, or fall back to the
        # 3-6 kHz default? "Fired" means SOME band's envelope spectrum was
        # peaky enough (crest >= 10) to look periodic. A healthy machine with
        # no mechanical fault should mostly NOT fire — see the self-test
        # note below, and DOC_STATUS.md's surrogate finding that a realistic
        # noise floor never reached the crest floor at all.
        "band_selector_fired_pct": round(100.0 * n_fired / n_scored, 1) if n_scored else None,
        # Q2: was the shaft-speed estimate reliable? In mic-only mode this is
        # ALWAYS False by design (features.estimate_fr: a single live channel
        # is "a working assumption, not a measurement") — reported here so
        # that fact is measured on your file, not just asserted from the code.
        "speed_estimate_reliable_pct": round(100.0 * n_reliable / n_scored, 1) if n_scored else None,
        # Q3: did a (presumed) healthy machine stay below its own threshold?
        "windows_above_threshold_pct": round(100.0 * n_anom / n_scored, 1) if n_scored else None,
        "baseline_k_regimes": int(baseline["k"]),
        "baseline_thresholds": [round(float(t), 3) for t in baseline["thresholds"]],
        "baseline_contaminated": [bool(c) for c in baseline["threshold_contaminated"]],
    }
    return rows, summary


def _print_report(rows: list[dict], summary: dict) -> None:
    print(f"\nrecording       : {summary['recording']}")
    print(f"mic-only        : {summary['mic_only']}")
    print(f"windows         : {summary['total_windows']} total "
         f"({summary['learn_windows']} learn + {summary['scored_windows']} scored)")
    print(f"regimes learned : k={summary['baseline_k_regimes']}, "
         f"thresholds={summary['baseline_thresholds']}")
    if any(summary["baseline_contaminated"]):
        print("  !! learn-period contamination flagged in at least one regime "
             "— see firmware/baseline.py's warning; consider re-recording the "
             "learn period on a quiet machine.")
    print()
    header = f"{'w':>4} {'band Hz':>13} {'crest':>6} {'fired':>5} " \
            f"{'fr Hz':>7} {'rel':>4} {'reg':>3} {'score':>8} {'thr':>7} {'x thr':>6} {'flag':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = "!!" if r["anomalous"] else ""
        print(f"{r['window']:>4} {r['band_lo_hz']:>5.0f}-{r['band_hi_hz']:<6.0f} "
             f"{r['band_crest']:>6.1f} {'yes' if r['band_fired'] else 'no':>5} "
             f"{r['fr_hz']:>7.1f} {'y' if r['fr_reliable'] else 'n':>4} "
             f"{r['regime']:>3} {r['score']:>8.2f} {r['threshold']:>7.2f} "
             f"{r['score_over_threshold']:>6.2f} {flag:>5}")
    print()
    print("--- honest answers, T7.2's three questions ---")
    print(f"Q1  demod band selector fired (found a periodic band) on "
         f"{summary['band_selector_fired_pct']}% of scored windows "
         f"({'fell back to the 3-6 kHz default' if summary['band_selector_fired_pct'] == 0 else 'see per-window table'} "
         f"the rest of the time)")
    print(f"Q2  shaft-speed estimate was reliable on "
         f"{summary['speed_estimate_reliable_pct']}% of scored windows "
         f"({'expected: 0% in mic-only mode by design' if summary['mic_only'] else 'audio+accel agreement'})")
    print(f"Q3  {summary['windows_above_threshold_pct']}% of scored windows were "
         f"above their regime's threshold "
         f"({'GOOD — behaves like a healthy machine, if this recording was one' if summary['windows_above_threshold_pct'] == 0 else 'INVESTIGATE — see the flagged rows above'})")


# ----------------------------------------------------------------------------
# Self-test: prove the tool works without a real recording
# ----------------------------------------------------------------------------

def _self_test_signal(fs: int, dur_s: float, rng: np.random.Generator) -> np.ndarray:
    """A healthy-appliance-ish audio floor, written from scratch for this
    self-test — it does NOT import ml/simulate.py, so a clean pass here is
    evidence the TOOL's plumbing works on an input the detector has never
    been tuned against, not a circular check against the project's own
    simulator.

    Model: 50 Hz UK mains hum + its 2nd/3rd harmonics (every mains-fed motor
    and most compressors have this), and a pink noise floor (machinery noise
    is pink, not white; using white would make this an easier test than a
    real room). This is SYNTHETIC. It is not a prediction about your fridge,
    only a fixture for testing this file's code.

    Deliberately STATIONARY — no slow amplitude drift. A real appliance
    cycling on and off is exactly the kind of level change the product's
    regime clustering exists to handle, but that is a claim about
    firmware/baseline.py's regime logic (already tested in
    tests/test_regimes_miconly.py), not about this file's plumbing. Mixing
    the two into one fixture would make a failure here ambiguous about which
    layer broke it.
    """
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    hum = (0.05 * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 6.28))
          + 0.02 * np.sin(2 * np.pi * 100.0 * t + rng.uniform(0, 6.28))
          + 0.01 * np.sin(2 * np.pi * 150.0 * t + rng.uniform(0, 6.28)))
    # Pink floor via 1/sqrt(f) spectral shaping (matches the recipe already
    # used for the same reason in ml/realdata/validate_public_dataset.py).
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec /= np.sqrt(np.maximum(f, 1.0))
    pink = np.fft.irfft(spec, n)
    pink /= (np.std(pink) + 1e-12)
    return hum + 0.08 * pink


def _run_self_test(learn_windows: int, score_windows: int, window_s: float,
                   fs: int, verbose: bool) -> int:
    from scipy.io import wavfile
    n_windows = learn_windows + score_windows
    rng = np.random.default_rng(20260820)
    audio = _self_test_signal(fs, n_windows * window_s, rng)
    peak = np.max(np.abs(audio)) + 1e-9
    pcm = np.clip(audio / peak * 0.8, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "self_test.wav"
        wavfile.write(wav_path, fs, pcm16)
        progress = (lambda m: print(f"  {m}")) if verbose else None
        rows, summary = analyse(wav_path, None, window_s=window_s,
                                learn_windows=learn_windows, progress=progress)

    _print_report(rows, summary)

    print("\n--- self-test verdict ---")
    ok = True
    if summary["windows_above_threshold_pct"] > 20.0:
        print(f"FAIL: {summary['windows_above_threshold_pct']}% of windows on a "
             f"synthetic HEALTHY signal scored above threshold — the pipeline "
             f"or this self-test signal is broken.")
        ok = False
    if summary["speed_estimate_reliable_pct"] != 0.0:
        print("FAIL: mic-only mode reported a reliable speed estimate — "
             "estimate_fr's mic-only branch should always return "
             "reliable=False (no cross-check channel exists).")
        ok = False
    if ok:
        print("PASS: tool runs end to end; a healthy synthetic signal with no "
             "periodic fault stays below threshold; mic-only correctly "
             "reports the speed estimate as unreliable. This proves the "
             "CODE works — it says nothing about a real recording.")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the real learn->score pipeline on one recording "
                    "(mic-only or with accel), no bearing geometry needed.",
        epilog="See docs/PHONE_RECORDING.md for the full recording-to-report walkthrough.")
    p.add_argument("wav", nargs="?", type=Path,
                   help="canonical .wav from tools/ingest.py")
    p.add_argument("--accel", type=Path, default=None,
                   help="accelerometer CSV, if the recording has one "
                       "(default: mic-only, i.e. none)")
    p.add_argument("--learn-windows", type=int, default=48,
                   help="windows (of --window-s each) to learn from before "
                       "scoring the rest (default 48 = 24 min; the "
                       "documented minimum, see the module docstring)")
    p.add_argument("--window-s", type=float, default=30.0)
    p.add_argument("--max-windows", type=int, default=None,
                   help="cap total windows read (debugging / quick checks)")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the full per-window report as JSON")
    p.add_argument("--self-test", action="store_true",
                   help="run against a synthetic healthy signal instead of a "
                       "file, to prove the tool works with no recording")
    p.add_argument("--self-test-learn-windows", type=int, default=48,
                   help="default 48 matches the documented minimum learn "
                       "period (docs/DOC_STATUS.md: below this, held-out "
                       "false-alarm rate is measured at 55-59%%, so a "
                       "smaller value here would not be a meaningful check)")
    p.add_argument("--self-test-score-windows", type=int, default=16)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return _run_self_test(args.self_test_learn_windows,
                             args.self_test_score_windows, args.window_s,
                             16000, args.verbose)
    if args.wav is None:
        build_parser().error("wav is required unless --self-test is given")
    progress = (lambda m: print(f"  {m}")) if args.verbose else None
    rows, summary = analyse(args.wav, args.accel, window_s=args.window_s,
                            learn_windows=args.learn_windows,
                            max_windows=args.max_windows, progress=progress)
    _print_report(rows, summary)
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
