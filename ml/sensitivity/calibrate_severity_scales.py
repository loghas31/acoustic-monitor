"""
calibrate_severity_scales.py — backlog T1.12 (self-review F18).

F18 measured `ml/realdata/synth_phone_recording.py` faults at severity
0.05/0.10/0.20 scoring 0.55/0.55/0.70x threshold against their own correct
baseline (i.e. NOT detected), while `docs/DOC_SENSITIVITY.md` reports
`ml/simulate.py` detecting down to severity 0.02 with huge margin (median
71.3x threshold at SNR 20dB). Both modules call their knob "severity" but nothing
before this measured whether they are the same physical quantity.

THE TWO HYPOTHESES (from the task text), and which this file settles:
  1. The two `severity` parameters are simply different scales (~10x apart),
     and nothing is wrong once you convert between them.
  2. A realistic PINK noise floor masks faults a WHITE floor does not, so
     even at the "same" physical fault level, pink-floor detection is worse.

METHOD: stop using `severity` as the measurement axis. Both generators excite
a known structural resonance (simulate.py: f0=resonance_hz, Q=resonance_q;
synth_phone_recording.py: f0=resonance_hz, Q=8.0, hardcoded — see NOTE below)
with an impulse train. Band-pass EACH signal (healthy and faulty) at that
known resonance and compare RMS in dB — "band RMS re the healthy floor" is a
physical energy measurement, not a knob value, and is directly comparable
between the two generators because both are built the same way (impulse
train -> resonance filter -> add to a noise-floor + hum background).

This does NOT reimplement each generator's signal model - it imports the
real `ml.simulate` (frozen) and `ml.realdata.synth_phone_recording` (not
frozen but independently written, see that module's own docstring) functions
directly, exactly as `ml/sensitivity/sweep.py` already does for the white
generator.

NOTE on synth_phone_recording's resonance Q: `make_pair` takes `q` as a
parameter (default 8.0) but does not return it in the pair dict. Rather than
edit that module (not necessary - it is not frozen but touching it risks
disturbing T7.2's already-verified F18 measurement), this file hardcodes
`PHONE_Q = 8.0` to match `synth_phone_recording.make_pair`'s own default and
documents the coupling here.

USAGE
    python ml/sensitivity/calibrate_severity_scales.py curves
    python ml/sensitivity/calibrate_severity_scales.py detect --n-learn 16 --n-test 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "firmware"), str(_ROOT / "ml"), str(_ROOT / "ml" / "realdata"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulate import SimConfig, normal_signal, bearing_fault_signal          # noqa: E402
from synth_phone_recording import make_pair as phone_make_pair               # noqa: E402

from baseline import fit_baseline, operating_point, save_baseline            # noqa: E402
from features import FEATURE_NAMES, extract_features                        # noqa: E402
from inference import MahalanobisScorer                                      # noqa: E402

FS_AUDIO = 16000          # production audio rate; both generators can run at it
FS_ACCEL = 6400           # production accel rate (mic-only build -> dead channel)
WINDOW_S = 30.0           # matches production window length
PHONE_Q = 8.0             # must match synth_phone_recording.make_pair's own default


def band_rms_db(healthy: np.ndarray, faulty: np.ndarray, fs: float,
                f0: float, q: float) -> float:
    """RMS of each signal inside the resonance band (f0 +/- f0/(2q)), faulty
    re healthy, in dB. This is the common PHYSICAL measure T1.12 asks for:
    both signals pass through the identical filter, so the number reflects
    energy actually added at the fault's own resonance, not a knob value."""
    bw = f0 / q
    lo, hi = max(f0 - bw / 2, 1.0), min(f0 + bw / 2, fs / 2 * 0.98)
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    h_rms = float(np.sqrt(np.mean(sosfilt(sos, healthy) ** 2)))
    f_rms = float(np.sqrt(np.mean(sosfilt(sos, faulty) ** 2)))
    return 20.0 * np.log10((f_rms + 1e-15) / (h_rms + 1e-15))


# ----------------------------------------------------------------------------
# Calibration curves: severity -> band RMS dB re healthy floor
# ----------------------------------------------------------------------------

def simulate_curve(severities, n_seeds: int = 3, attenuation_db: float = 0.0):
    """ml/simulate.py's own severity knob, averaged over n_seeds draws."""
    out = []
    for sev in severities:
        dbs = []
        for seed in range(n_seeds):
            cfg = SimConfig(fs_audio=FS_AUDIO, duration_s=WINDOW_S, seed=seed)
            rng_h = np.random.default_rng(1000 + seed)
            rng_f = np.random.default_rng(1000 + seed)
            healthy = normal_signal(cfg, FS_AUDIO, rng_h)
            eff_sev = sev * (10.0 ** (-attenuation_db / 20.0))
            faulty = bearing_fault_signal(cfg, FS_AUDIO, rng_f, severity=eff_sev, race="outer")
            dbs.append(band_rms_db(healthy, faulty, FS_AUDIO, cfg.resonance_hz, cfg.resonance_q))
        out.append({"severity": sev, "attenuation_db": attenuation_db,
                    "band_rms_db": float(np.mean(dbs)), "band_rms_db_std": float(np.std(dbs))})
    return out


def phone_curve(severities, n_seeds: int = 3, resonance_hz: float = 1600.0):
    """synth_phone_recording.py's own severity knob, averaged over n_seeds draws."""
    out = []
    for sev in severities:
        dbs = []
        for seed in range(n_seeds):
            pair = phone_make_pair(seed=2000 + seed, duration_s=WINDOW_S, fs=FS_AUDIO,
                                   resonance_hz=resonance_hz, severity=sev)
            dbs.append(band_rms_db(pair["healthy"], pair["faulty"], FS_AUDIO,
                                   resonance_hz, PHONE_Q))
        out.append({"severity": sev, "band_rms_db": float(np.mean(dbs)),
                    "band_rms_db_std": float(np.std(dbs))})
    return out


def invert_curve(curve, target_db: float) -> float | None:
    """Linear-interpolate severity for a target dB level from a
    severity-sorted, monotone-in-dB curve. Returns None if target_db is
    outside the measured range (extrapolation is not meaningful here)."""
    sevs = [r["severity"] for r in curve]
    dbs = [r["band_rms_db"] for r in curve]
    if not (min(dbs) <= target_db <= max(dbs)):
        return None
    # dbs is increasing with severity for both generators (more fault energy
    # -> more band energy); np.interp needs an increasing xp
    order = np.argsort(dbs)
    return float(np.interp(target_db, np.array(dbs)[order], np.array(sevs)[order]))


# ----------------------------------------------------------------------------
# Detection at a given severity, phone (pink) generator, mic-only
# ----------------------------------------------------------------------------

def _phone_window_features(severity: float, seed: int, resonance_hz: float,
                           which: str, crest_floor: float | None = None):
    pair = phone_make_pair(seed=seed, duration_s=WINDOW_S, fs=FS_AUDIO,
                           resonance_hz=resonance_hz, severity=severity)
    audio = pair[which]
    accel = np.zeros_like(audio)  # mic-only: dead channel, matches phone_monitor.py
    kwargs = {} if crest_floor is None else {"crest_floor": crest_floor}
    feats = extract_features(audio, FS_AUDIO, accel, FS_ACCEL, **kwargs)
    return feats


def detect_phone(severity: float, n_learn: int = 16, n_test_healthy: int = 6,
                 n_test_fault: int = 6, resonance_hz: float = 1600.0,
                 seed_base: int = 0) -> dict:
    """Fit a mic-only baseline on healthy phone-generator windows, score
    held-out healthy + faulty windows at `severity`. Same discipline as
    ml/sensitivity/sweep.py's run_config: real fit_baseline/MahalanobisScorer,
    not a hand-rolled score."""
    from sklearn.metrics import roc_auc_score

    seed = seed_base
    X_learn, OP_learn, crest_learn = [], [], []
    for _ in range(n_learn):
        feats = _phone_window_features(0.0, seed, resonance_hz, "healthy")
        seed += 1
        X_learn.append(feats["vector"])
        OP_learn.append(operating_point(feats["vector"], feats["fr_hz"]))
        crest_learn.append(feats["band_crest"])
    X_learn, OP_learn = np.array(X_learn), np.array(OP_learn)
    baseline = fit_baseline(X_learn, OP_learn, list(FEATURE_NAMES),
                            learn_crest=np.array(crest_learn))

    tmp_path = Path(f"/tmp/phone_calib_baseline_{severity}_{seed_base}.npz")
    save_baseline(tmp_path, baseline)
    try:
        scorer = MahalanobisScorer(tmp_path)

        healthy_ratio = []
        for _ in range(n_test_healthy):
            feats = _phone_window_features(0.0, seed, resonance_hz, "healthy",
                                           crest_floor=scorer.crest_floor)
            seed += 1
            s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
            healthy_ratio.append(s["score"] / s["threshold"])
        fpr = float(np.mean(np.array(healthy_ratio) > 1.0))

        fault_ratio = []
        for _ in range(n_test_fault):
            feats = _phone_window_features(severity, seed, resonance_hz, "faulty",
                                           crest_floor=scorer.crest_floor)
            seed += 1
            s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
            fault_ratio.append(s["score"] / s["threshold"])
        tpr = float(np.mean(np.array(fault_ratio) > 1.0))

        labels = [0] * len(healthy_ratio) + [1] * len(fault_ratio)
        scores = healthy_ratio + fault_ratio
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    finally:
        tmp_path.unlink(missing_ok=True)

    db = phone_curve([severity], n_seeds=1, resonance_hz=resonance_hz)[0]["band_rms_db"]
    return {
        "severity": severity, "band_rms_db": db,
        "fpr": fpr, "tpr": tpr, "auc": auc,
        "healthy_ratio_median": float(np.median(healthy_ratio)),
        "fault_ratio_median": float(np.median(fault_ratio)),
        "k_regimes": int(baseline["k"]),
        "contaminated": bool(np.any(baseline["threshold_contaminated"])),
    }


def _simulate_window_features(severity: float, seed: int, resonance_hz: float,
                              resonance_q: float, which: str,
                              crest_floor: float | None = None):
    cfg = SimConfig(fs_audio=FS_AUDIO, duration_s=WINDOW_S,
                    resonance_hz=resonance_hz, resonance_q=resonance_q)
    rng = np.random.default_rng(seed)
    if which == "healthy":
        audio = normal_signal(cfg, FS_AUDIO, rng)
    else:
        audio = bearing_fault_signal(cfg, FS_AUDIO, rng, severity=severity, race="outer")
    accel = np.zeros_like(audio)
    kwargs = {} if crest_floor is None else {"crest_floor": crest_floor}
    return extract_features(audio, FS_AUDIO, accel, FS_ACCEL, **kwargs)


def detect_simulate(severity: float, n_learn: int = 16, n_test_healthy: int = 6,
                    n_test_fault: int = 6, resonance_hz: float = 4500.0,
                    resonance_q: float = 30.0, seed_base: int = 0) -> dict:
    """Mirror of `detect_phone`, but for ml/simulate.py's own (white-floor)
    generator, single accelerometer axis dead-channel-padded exactly like the
    phone build - so the ONLY thing that differs between this function and
    `detect_phone` is which module generated the signal. Lets the two be
    compared at matched band_rms_db rather than matched severity index."""
    from sklearn.metrics import roc_auc_score

    seed = seed_base
    X_learn, OP_learn, crest_learn = [], [], []
    for _ in range(n_learn):
        feats = _simulate_window_features(0.0, seed, resonance_hz, resonance_q, "healthy")
        seed += 1
        X_learn.append(feats["vector"])
        OP_learn.append(operating_point(feats["vector"], feats["fr_hz"]))
        crest_learn.append(feats["band_crest"])
    X_learn, OP_learn = np.array(X_learn), np.array(OP_learn)
    baseline = fit_baseline(X_learn, OP_learn, list(FEATURE_NAMES),
                            learn_crest=np.array(crest_learn))

    tmp_path = Path(f"/tmp/sim_calib_baseline_{severity}_{seed_base}.npz")
    save_baseline(tmp_path, baseline)
    try:
        scorer = MahalanobisScorer(tmp_path)

        healthy_ratio = []
        for _ in range(n_test_healthy):
            feats = _simulate_window_features(0.0, seed, resonance_hz, resonance_q,
                                              "healthy", crest_floor=scorer.crest_floor)
            seed += 1
            s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
            healthy_ratio.append(s["score"] / s["threshold"])
        fpr = float(np.mean(np.array(healthy_ratio) > 1.0))

        fault_ratio = []
        for _ in range(n_test_fault):
            feats = _simulate_window_features(severity, seed, resonance_hz, resonance_q,
                                              "faulty", crest_floor=scorer.crest_floor)
            seed += 1
            s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
            fault_ratio.append(s["score"] / s["threshold"])
        tpr = float(np.mean(np.array(fault_ratio) > 1.0))

        labels = [0] * len(healthy_ratio) + [1] * len(fault_ratio)
        scores = healthy_ratio + fault_ratio
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    finally:
        tmp_path.unlink(missing_ok=True)

    db = simulate_curve([severity], n_seeds=1)[0]["band_rms_db"]
    return {
        "severity": severity, "band_rms_db": db,
        "fpr": fpr, "tpr": tpr, "auc": auc,
        "healthy_ratio_median": float(np.median(healthy_ratio)),
        "fault_ratio_median": float(np.median(fault_ratio)),
        "k_regimes": int(baseline["k"]),
        "contaminated": bool(np.any(baseline["threshold_contaminated"])),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("curves", help="print severity->dB calibration curves for both generators")
    c.add_argument("--out", type=Path, default=None)

    d = sub.add_parser("detect", help="run mic-only detection across phone-generator severities")
    d.add_argument("--severities", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00])
    d.add_argument("--n-learn", type=int, default=16)
    d.add_argument("--n-test", type=int, default=6)
    d.add_argument("--out", type=Path, default=None)

    b = sub.add_parser("detect-both", help="run detection for BOTH generators at matched "
                                           "severities, to compare at matched band_rms_db")
    b.add_argument("--sim-severities", type=float, nargs="+",
                   default=[0.02, 0.05, 0.10, 0.15, 0.20])
    b.add_argument("--phone-severities", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 1.0, 2.0, 5.0, 7.0])
    b.add_argument("--n-learn", type=int, default=16)
    b.add_argument("--n-test", type=int, default=6)
    b.add_argument("--out", type=Path, default=None)

    args = p.parse_args()

    if args.cmd == "curves":
        sim_severities = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        phone_severities = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
        result = {
            "simulate": simulate_curve(sim_severities),
            "phone": phone_curve(phone_severities),
        }
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.write_text(json.dumps(result, indent=2))

    elif args.cmd == "detect":
        rows = [detect_phone(sev, n_learn=args.n_learn, n_test_healthy=args.n_test,
                             n_test_fault=args.n_test)
               for sev in args.severities]
        print(json.dumps(rows, indent=2))
        if args.out:
            args.out.write_text(json.dumps(rows, indent=2))

    elif args.cmd == "detect-both":
        result = {
            "simulate": [detect_simulate(sev, n_learn=args.n_learn, n_test_healthy=args.n_test,
                                         n_test_fault=args.n_test)
                        for sev in args.sim_severities],
            "phone": [detect_phone(sev, n_learn=args.n_learn, n_test_healthy=args.n_test,
                                   n_test_fault=args.n_test)
                     for sev in args.phone_severities],
        }
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
