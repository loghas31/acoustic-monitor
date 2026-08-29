"""
sweep.py — backlog T1.4: "how bad can reality be?"

Every AUC this project has ever reported (ml/evaluate.py: 1.000; the CWRU
surrogate: 0.9889) was measured at ONE point in signal-quality space: the
`ml/simulate.py` defaults (SNR 20 dB, resonance 4500 Hz, no mounting loss, no
neighbouring machinery). That point was chosen because it looks like a
reasonably clean small motor, not because anyone measured where the detector
actually breaks. This script measures it, on four axes we can vary honestly
before the bench exists:

  1. SNR             — ml/simulate.py's own `snr_db` knob.
  2. Resonance freq   — where the bearing-housing resonance sits, 1-8 kHz.
                        ml/simulate.py itself clamps this to 0.4*fs before
                        filtering (`min(cfg.resonance_hz, 0.4*fs)`), so part of
                        what this axis measures is that clamp, not the
                        detector — see the DOC_SENSITIVITY.md note on it.
  3. Mounting attenuation — a magnetic mount, paint, or an off-axis placement
                        loses some of the impact energy between the housing
                        and the sensor before it ever becomes an SNR problem.
                        Modelled as a dB loss applied ONLY to the fault-burst
                        term, at a fixed "true" severity — i.e. this axis asks
                        "how much of a real, moderate fault's signature can we
                        lose in the mechanical path and still catch it",
                        which is mathematically an severity rescaling (see the
                        docstring on `bearing_fault_signal_ext`) but a
                        physically distinct and more actionable question.
  4. Interfering machinery — a second, unrelated machine's hum, at a
                        frequency that shares no harmonic with the target
                        (73 Hz against a 50 Hz shaft), added at increasing
                        amplitude. Distinct from (1): SNR is unstructured
                        Gaussian noise, this is structured tonal interference
                        — the gym/workshop-floor scenario, not a noisier
                        sensor.

WHAT THIS SCRIPT DOES NOT TEST (by design, and said here so nobody assumes it
was overlooked): regime clustering (T1.9), accelerometer axis decorrelation
(T1.8), and feature-block informativeness (T1.10) already have dedicated,
executed studies. This one holds all of that fixed — single regime (fr = 50
Hz constant), single accelerometer axis padded to 3 by `extract_features`'
existing dead-channel path (exactly the pattern `firmware/features.py`'s own
`__main__` self-test uses) — so that signal-quality degradation is the only
thing moving.

METHOD, per (axis, value):
  1. Fit a baseline on `n_learn` HEALTHY windows generated at that point in
     parameter space (`firmware/baseline.py:fit_baseline`, unmodified).
  2. Save + reload it through the REAL `inference.MahalanobisScorer` (not a
     hand-rolled scoring formula) so this exercises the exact code path that
     ships, the same discipline `ml/evaluate.py` uses.
  3. Score `n_healthy_test` held-out healthy windows (independent seeds) for
     the false-positive rate, and a set of held-out FAULT windows (severities
     0.1 / 0.2 / 0.4 for the SNR/resonance/interference axes; a single fixed
     severity 0.3 for the attenuation axis, since attenuation IS the severity
     axis there) for the true-positive rate and ROC AUC.

WHY THE CUSTOM SIGNAL FUNCTIONS BELOW, INSTEAD OF IMPORTING `ml/simulate.py`
DIRECTLY: `simulate.normal_signal` / `bearing_fault_signal` do not expose a
mounting-attenuation or interfering-machine knob, and `ml/simulate.py` is
FROZEN (no failing test justifies editing it — the frozen functions are
correct and unrelated to this study). `normal_signal_ext` /
`bearing_fault_signal_ext` below are built from the SAME private building
blocks simulate.py uses internally (`_time`, `_machine_hum`, `_noise_floor`,
`_impulse_train`, `_resonance_filter`) — the same pattern `firmware/capture.py`
already uses for the triaxial accelerometer (T1.8) — and are PINNED
bit-identical to the frozen functions when the two new knobs are at their
neutral values (`tests/test_sensitivity.py::test_extensions_match_frozen_simulator_at_neutral_settings`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "firmware"), str(_ROOT / "ml"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.signal import sosfilt                                    # noqa: E402
from simulate import (SimConfig, _impulse_train, _machine_hum,      # noqa: E402
                      _noise_floor, _resonance_filter, _time)

from baseline import fit_baseline, operating_point, save_baseline   # noqa: E402
from features import FEATURE_NAMES, extract_features                # noqa: E402
from inference import MahalanobisScorer                              # noqa: E402

WINDOW_S = 30.0                 # matches production; do not shorten — window
                                 # length is its own confound (fewer impacts
                                 # per window), and this study is about signal
                                 # quality only.
FS_AUDIO, FS_ACCEL = 16000, 6400
DEFAULT_FR = 50.0
INTERFERER_FR = 73.0            # arbitrary, shares no low harmonic with 50 Hz
                                 # (73/50 is not close to any small ratio)


# ----------------------------------------------------------------------------
# Signal generation — thin extensions of the frozen simulator
# ----------------------------------------------------------------------------

def normal_signal_ext(cfg: SimConfig, fs: int, rng: np.random.Generator,
                      interferer_gain: float = 0.0,
                      interferer_fr: float = INTERFERER_FR) -> np.ndarray:
    """Identical to `simulate.normal_signal` when interferer_gain == 0.

    Draw order (hum, then noise floor) matches the frozen function exactly,
    and the interferer draw only happens — and only ever happens — after
    both, so at gain 0 not even an extra rng draw occurs. That is what makes
    the bit-identical test possible: it is not "close", it is the same
    function call sequence with one optional extra term appended.
    """
    t = _time(cfg, fs)
    sig = _machine_hum(t, cfg.fr, rng) + _noise_floor(len(t), fs, cfg.snr_db, rng)
    if interferer_gain > 0:
        sig = sig + interferer_gain * _machine_hum(t, interferer_fr, rng)
    return sig


def bearing_fault_signal_ext(cfg: SimConfig, fs: int, rng: np.random.Generator,
                             severity: float, race: str,
                             interferer_gain: float = 0.0,
                             interferer_fr: float = INTERFERER_FR,
                             attenuation_db: float = 0.0) -> np.ndarray:
    """Identical to `simulate.bearing_fault_signal` when interferer_gain == 0
    and attenuation_db == 0. Attenuation scales ONLY the burst term, applied
    AFTER the frozen function's own unit-variance normalisation — i.e. it is
    algebraically `severity * 10**(-attenuation_db/20)`, so sweeping this axis
    at a fixed severity is the same measurement as sweeping severity itself
    at a fixed attenuation. That equivalence is deliberate and stated in the
    module docstring: what changes is the QUESTION being asked of the same
    number ("how much mechanical loss survives a moderate fault"), not the
    maths.
    """
    base = normal_signal_ext(cfg, fs, rng, interferer_gain, interferer_fr)
    if race == "outer":
        f_fault, mod = cfg.bearing.bpfo(cfg.fr), None
    elif race == "inner":
        f_fault, mod = cfg.bearing.bpfi(cfg.fr), cfg.fr
    else:
        raise ValueError("race must be 'outer' or 'inner'")
    t = _time(cfg, fs)
    train = _impulse_train(t, fs, f_fault, cfg.slip_jitter, rng, modulate_at=mod)
    sos = _resonance_filter(fs, min(cfg.resonance_hz, 0.4 * fs), cfg.resonance_q)
    bursts = sosfilt(sos, train)
    bursts = bursts / (np.std(bursts) + 1e-12)
    eff_severity = severity * (10.0 ** (-attenuation_db / 20.0))
    return base + eff_severity * bursts


def make_window(kind: str, severity: float, params: dict,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One (audio, accel) pair. `params` may hold any of: fr, resonance_hz,
    snr_db, interferer_gain, attenuation_db — unset ones use SimConfig's
    defaults (the repo's own baseline operating point).

    Single accelerometer axis, not the T1.8 triaxial model: `extract_features`
    pads a (n,1) accel array with zero y/z columns (its existing dead-channel
    path), so this is the same degraded-but-supported shape as a mic-only or
    accel-partial build. Using it here means this study measures signal
    quality, not axis geometry — that question already has T1.8/T1.9."""
    fr = params.get("fr", DEFAULT_FR)
    cfg_kwargs = dict(duration_s=WINDOW_S, fr=fr)
    if "resonance_hz" in params:
        cfg_kwargs["resonance_hz"] = params["resonance_hz"]
    if "snr_db" in params:
        cfg_kwargs["snr_db"] = params["snr_db"]
    interferer_gain = params.get("interferer_gain", 0.0)
    attenuation_db = params.get("attenuation_db", 0.0)

    cfg_a = SimConfig(fs_audio=FS_AUDIO, fs_accel=FS_ACCEL, **cfg_kwargs)
    cfg_v = SimConfig(fs_audio=FS_AUDIO, fs_accel=FS_ACCEL, **cfg_kwargs)

    if kind == "normal":
        audio = normal_signal_ext(cfg_a, FS_AUDIO, rng, interferer_gain)
        accel = normal_signal_ext(cfg_v, FS_ACCEL, rng, interferer_gain)
    elif kind in ("bearing_outer", "bearing_inner"):
        race = "outer" if kind == "bearing_outer" else "inner"
        audio = bearing_fault_signal_ext(cfg_a, FS_AUDIO, rng, severity, race,
                                         interferer_gain, attenuation_db=attenuation_db)
        accel = bearing_fault_signal_ext(cfg_v, FS_ACCEL, rng, severity, race,
                                         interferer_gain, attenuation_db=attenuation_db)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return audio, accel


# ----------------------------------------------------------------------------
# One (axis, value) measurement
# ----------------------------------------------------------------------------

AXES = {
    "snr": dict(param="snr_db", values=[20, 10, 5, 0, -5],
               label="SNR (dB; higher = quieter noise floor)",
               baseline_value=20),
    "resonance": dict(param="resonance_hz", values=[1000, 2500, 4500, 6400, 8000],
                      label="Resonance frequency (Hz)", baseline_value=4500),
    "attenuation": dict(param="attenuation_db", values=[0, 6, 12, 18, 24],
                        label="Mounting attenuation (dB loss on fault energy, "
                              "at fixed true severity 0.3)", baseline_value=0),
    "interference": dict(param="interferer_gain", values=[0.0, 0.5, 1.0, 2.0, 4.0],
                         label="Interfering-machine amplitude (x primary hum, "
                               "73 Hz)", baseline_value=0.0),
}
ATTENUATION_FIXED_SEVERITY = 0.3


def run_config(axis: str, value: float, n_learn: int = 32,
               n_healthy_test: int = 12, seed_base: int = 0) -> dict:
    from sklearn.metrics import roc_auc_score

    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}, choose from {list(AXES)}")
    params = {AXES[axis]["param"]: value}

    seed = seed_base
    X_learn, OP_learn = [], []
    for _ in range(n_learn):
        rng = np.random.default_rng(seed); seed += 1
        audio, accel = make_window("normal", 0.0, params, rng)
        feats = extract_features(audio, FS_AUDIO, accel, FS_ACCEL)
        X_learn.append(feats["vector"])
        OP_learn.append(operating_point(feats["vector"], feats["fr_hz"]))
    X_learn, OP_learn = np.array(X_learn), np.array(OP_learn)
    baseline = fit_baseline(X_learn, OP_learn, list(FEATURE_NAMES))

    tmp_path = Path(f"/tmp/sensitivity_baseline_{axis}_{value}_{os.getpid()}_{seed_base}.npz")
    save_baseline(tmp_path, baseline)
    try:
        scorer = MahalanobisScorer(tmp_path)

        healthy_ratio = []
        for _ in range(n_healthy_test):
            rng = np.random.default_rng(seed); seed += 1
            audio, accel = make_window("normal", 0.0, params, rng)
            feats = extract_features(audio, FS_AUDIO, accel, FS_ACCEL)
            s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
            healthy_ratio.append(s["score"] / s["threshold"])
        fpr = float(np.mean(np.array(healthy_ratio) > 1.0))

        if axis == "attenuation":
            severities, n_per_sev = [ATTENUATION_FIXED_SEVERITY], 12
        else:
            severities, n_per_sev = [0.1, 0.2, 0.4], 5

        fault_ratio, tpr_by_sev = [], {}
        for sev in severities:
            sev_ratios = []
            for _ in range(n_per_sev):
                rng = np.random.default_rng(seed); seed += 1
                audio, accel = make_window("bearing_outer", sev, params, rng)
                feats = extract_features(audio, FS_AUDIO, accel, FS_ACCEL)
                s = scorer.score(feats["vector"], operating_point(feats["vector"], feats["fr_hz"]))
                sev_ratios.append(s["score"] / s["threshold"])
            fault_ratio.extend(sev_ratios)
            tpr_by_sev[sev] = float(np.mean(np.array(sev_ratios) > 1.0))

        tpr_overall = float(np.mean(np.array(fault_ratio) > 1.0))
        labels = [0] * len(healthy_ratio) + [1] * len(fault_ratio)
        scores = healthy_ratio + fault_ratio
        auc = float(roc_auc_score(labels, scores))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "axis": axis, "value": value,
        "n_learn": n_learn, "k_regimes": int(baseline["k"]),
        "fpr": fpr, "tpr": tpr_overall, "tpr_by_severity": tpr_by_sev,
        "auc": auc,
        "healthy_ratio_median": float(np.median(healthy_ratio)),
        "fault_ratio_median": float(np.median(fault_ratio)),
        "thresholds": [round(float(t), 3) for t in baseline["thresholds"]],
        "contaminated": bool(np.any(baseline["threshold_contaminated"])),
    }


# ----------------------------------------------------------------------------
# CLI: run one config, or combine partial results into report + figure
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure one (axis, value) point")
    r.add_argument("--axis", required=True, choices=list(AXES))
    r.add_argument("--value", type=float, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--seed-base", type=int, default=0)
    r.add_argument("--n-learn", type=int, default=32)
    r.add_argument("--n-healthy-test", type=int, default=12)

    c = sub.add_parser("combine", help="merge run_config JSON files -> figure + summary")
    c.add_argument("--in-dir", type=Path, required=True)
    c.add_argument("--out-json", type=Path, default=_ROOT / "ml" / "artifacts" / "sensitivity.json")
    c.add_argument("--out-png", type=Path, default=_ROOT / "ml" / "artifacts" / "sensitivity.png")

    args = p.parse_args()

    if args.cmd == "run":
        t0 = time.perf_counter()
        result = run_config(args.axis, args.value, args.n_learn,
                            args.n_healthy_test, args.seed_base)
        result["wall_s"] = round(time.perf_counter() - t0, 1)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))

    elif args.cmd == "combine":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        files = sorted(args.in_dir.glob("*.json"))
        if not files:
            raise SystemExit(f"no JSON result files in {args.in_dir}")
        rows = [json.loads(f.read_text()) for f in files]
        by_axis: dict[str, list[dict]] = {}
        for row in rows:
            by_axis.setdefault(row["axis"], []).append(row)
        for axis in by_axis:
            by_axis[axis].sort(key=lambda r: r["value"])

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(by_axis, indent=2))

        n = len(by_axis)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
        if n == 1:
            axes = [axes]
        for ax, axis in zip(axes, sorted(by_axis)):
            pts = by_axis[axis]
            xs = [r["value"] for r in pts]
            aucs = [r["auc"] for r in pts]
            fprs = [r["fpr"] for r in pts]
            tprs = [r["tpr"] for r in pts]
            ax.plot(xs, aucs, "o-", label="AUC", color="tab:blue")
            ax.plot(xs, tprs, "s--", label="TPR", color="tab:green", alpha=0.7)
            ax.plot(xs, fprs, "^--", label="FPR", color="tab:red", alpha=0.7)
            ax.axhline(0.95, color="k", ls=":", lw=0.8, label="AUC=0.95 gate")
            ax.set_title(AXES[axis]["label"], fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6)
        fig.tight_layout()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=120)

        print(json.dumps({axis: [{"value": r["value"], "auc": r["auc"],
                                  "fpr": r["fpr"], "tpr": r["tpr"]}
                                 for r in pts]
                          for axis, pts in by_axis.items()}, indent=2))


if __name__ == "__main__":
    main()
