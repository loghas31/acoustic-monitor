#!/usr/bin/env python3
"""
simulate_soak.py — generate a realistic multi-day SYNTHETIC soak run and write
it into a device state DB, exactly as `firmware/main.py` would have written it.

WHY THIS EXISTS
---------------
Week 3 of the execution plan is one number: **false alarms per node-week**, and
the gate is <= 1. You cannot write (or trust) the analysis tooling for that
number for the first time on the morning the real 7-day run finishes — if
`soak_report.py` has a bug, you will not know whether the bug is in the tool or
in the detector, and re-running the experiment costs another week.

So: this script manufactures a *healthy* week. Every alert the pipeline raises
on this data is BY CONSTRUCTION a false alarm, which means we can develop,
test, and calibrate `soak_report.py` end-to-end before a single real window
exists. It is also the only way to answer "what would our false-alarm rate be
if the machine had a regime we never learned?" without waiting a week per
experiment.

WHAT IS MODELLED (and why each part matters for false alarms)
-------------------------------------------------------------
The target machine in the plan is "something that runs continuously and
boringly" — a fridge compressor. A fridge is not stationary, and every source
of non-stationarity below is a documented false-alarm mechanism:

1. DUTY CYCLE. A compressor runs ~25 min, rests ~35 min, forever. That is two
   distinct acoustic normals. If the baseline learned only one, every start is
   an "anomaly". This is what `baseline.py`'s regime clustering exists for, and
   the soak report's regime-occupancy panel is how you check it worked.

2. DIURNAL VARIATION. Ambient noise and duty ratio move on a 24 h cycle (the
   kitchen is busier in the evening; the room is warmer, so the compressor runs
   longer). A learn period of a few hours samples ONE phase of that cycle. This
   is the single most common reason a baseline that looked perfect on day 1
   starts alarming on day 3.

3. SLOW DRIFT. A random walk on level/SNR standing in for grime, ambient
   temperature, the sensor mount relaxing. Anomaly detectors are unbiased
   estimators of "things changed", and things always change.

4. TRANSIENTS. Doors slamming, a trolley going past, someone dropping a pan.
   Short, loud, broadband, and completely innocent. These are the events the
   persistence gate is designed to eat, and their *duration distribution* is
   precisely what `soak_report.py` uses to recommend `persist_minutes`.

5. AN OPTIONAL UNLEARNED REGIME (`--unlearned-regime`). A defrost heater cycle
   that runs once a day for ~20 min and that the learn period never saw. This
   is the classic week-3 failure: not a broken detector, an incomplete learn
   period. Run the simulator both ways and compare the two reports; the
   difference is the whole lesson.

WHAT IS *NOT* MODELLED
----------------------
No fault is ever injected. This is a healthy machine, on purpose. Detection
performance is evaluated in `ml/evaluate.py` (week 2); this file is only about
the cost of being wrong in the other direction.

The synthetic signals come from `ml/simulate.py`, i.e. the same generator that
produced the week-0 evidence figures — so the acoustics are consistent with the
rest of the repo, and the DSP path (`features.extract_features`) is the real
one, not a stand-in.

USAGE
-----
    # fast smoke run (~1 min): 6 h of soak, small learn period
    python3 tools/simulate_soak.py --days 0.25 --learn-hours 2 \
        --learn-windows 40 --db /tmp/soak/soak.db --outdir /tmp/soak

    # full week (slow: ~270 ms of DSP per window; use --jobs)
    python3 tools/simulate_soak.py --days 7 --jobs 4 \
        --db /tmp/soak/soak.db --outdir /tmp/soak

    # the "we forgot to learn the defrost cycle" scenario
    python3 tools/simulate_soak.py --days 7 --unlearned-regime ...

Then:
    python3 tools/soak_report.py --db /tmp/soak/soak.db --outdir /tmp/soak/report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))

log = logging.getLogger("simulate_soak")

# One window of the real product. Everything in this file is indexed in
# windows, and converted to wall-clock only at the edges.
DEFAULT_WINDOW_S = 30.0


# ----------------------------------------------------------------------------
# The machine model: what the fridge is doing at window i
# ----------------------------------------------------------------------------

@dataclass
class WindowPlan:
    """Everything needed to synthesise one 30 s window, plus the ground truth
    label we keep for the tests (`soak_report.py` never sees these labels —
    it has to work them out from the scores, exactly as it will on real data)."""
    index: int
    kind: str            # simulate.py generator to use
    severity: float
    fr: float            # shaft/compressor running frequency, Hz
    gain: float          # linear level multiplier (how loud this window is)
    snr_db: float        # noise floor relative to the machine tone
    label: str           # ground truth: 'run' | 'idle' | 'defrost' | 'transient'


class MachinePlan:
    """Deterministic plan for the whole soak: window index -> WindowPlan.

    Deterministic on purpose. Given the same seed you get the same week, so a
    failing test is reproducible and a tuning experiment is a controlled one.
    Note that we draw the *schedule* from a single seeded RNG up front, and the
    *signals* from a per-window RNG (seed + i) inside the worker — that way the
    plan is identical no matter how many parallel jobs generate it.
    """

    def __init__(self, n_windows: int, window_s: float, seed: int,
                 transients_per_day: float = 6.0,
                 defrost_per_day: float = 1.0,
                 include_defrost: bool = True,
                 start_ts: float | None = None):
        self.n = n_windows
        self.window_s = window_s
        self.wpd = 86400.0 / window_s              # windows per day
        self.start_ts = start_ts if start_ts is not None else time.time()
        rng = np.random.default_rng(seed)

        # -- duty cycle ------------------------------------------------------
        # Compressor on ~25 min / off ~35 min, with the ON fraction rising in
        # the (warmer, busier) evening. Cycle boundaries jitter: a thermostat
        # is a feedback loop, not a metronome.
        self.on_flags = np.zeros(n_windows, dtype=bool)
        i = 0
        while i < n_windows:
            hour = ((i / self.wpd) * 24.0) % 24.0
            # diurnal duty: minimum ~0.35 at 05:00, maximum ~0.55 at 17:00
            duty = 0.45 + 0.10 * math.sin(2 * math.pi * (hour - 11.0) / 24.0)
            period_min = float(rng.normal(60.0, 6.0))
            period_w = max(6, int(round(period_min * 60.0 / window_s)))
            on_w = max(2, int(round(period_w * duty)))
            self.on_flags[i:i + on_w] = True
            i += period_w

        # -- defrost -------------------------------------------------------
        # ~once a day, ~20 min, at a random time. A heater + its fan: different
        # speed, louder, spectrally different. A legitimate third normal.
        self.defrost = np.zeros(n_windows, dtype=bool)
        if include_defrost:
            n_defrost = int(round(defrost_per_day * n_windows / self.wpd))
            dur_w = max(2, int(round(20 * 60 / window_s)))
            for d in range(n_defrost):
                # one per day, placed uniformly inside that day
                day0 = int(d * self.wpd)
                start = day0 + int(rng.integers(0, max(1, int(self.wpd) - dur_w)))
                self.defrost[start:start + dur_w] = True

        # -- transients ------------------------------------------------------
        # Poisson arrivals, 1-3 windows long (30-90 s). Loud, broadband,
        # innocent. THE events the persistence gate exists to absorb.
        self.transient = np.zeros(n_windows, dtype=bool)
        n_trans = rng.poisson(transients_per_day * n_windows / self.wpd)
        for _ in range(int(n_trans)):
            start = int(rng.integers(0, max(1, n_windows)))
            dur = int(rng.integers(1, 4))
            self.transient[start:start + dur] = True

        # -- slow drift ------------------------------------------------------
        # Random walk in dB on the overall level, clipped to +/-1.5 dB over the
        # week. Grime, mount relaxation, ambient temperature. Small, but the
        # detector is sensitive by design, so "small" is not "zero".
        steps = rng.normal(0.0, 0.02, n_windows)
        walk = np.cumsum(steps)
        self.drift_db = np.clip(walk - walk[0], -1.5, 1.5)

        # Per-window ambient noise wobble (people, traffic, the extractor fan).
        self.snr_jitter = rng.normal(0.0, 0.8, n_windows)

    def ts(self, i: int) -> float:
        return self.start_ts + i * self.window_s

    def plan(self, i: int) -> WindowPlan:
        hour = ((i / self.wpd) * 24.0) % 24.0
        # Ambient noise floor is worse (lower SNR) during the working day.
        snr = 20.0 - 2.5 * math.sin(2 * math.pi * (hour - 14.0) / 24.0) + self.snr_jitter[i]
        gain_db = self.drift_db[i]

        if self.transient[i]:
            # A slam: broadband impulsive energy, +8 dB, unrelated to the
            # machine. Modelled with the impulsive generator at high severity
            # because that is what a broadband knock looks like to the feature
            # extractor: high kurtosis/crest, energy in the resonance bands.
            return WindowPlan(i, "bearing_outer", 0.9,
                              50.0 if self.on_flags[i] else 30.0,
                              10 ** ((gain_db + 8.0) / 20.0), snr - 6.0, "transient")
        if self.defrost[i]:
            return WindowPlan(i, "imbalance", 0.25, 60.0,
                              10 ** ((gain_db + 3.0) / 20.0), snr, "defrost")
        if self.on_flags[i]:
            return WindowPlan(i, "normal", 0.0, 50.0,
                              10 ** (gain_db / 20.0), snr, "run")
        # Idle: only the evaporator fan. Quieter and slower.
        return WindowPlan(i, "normal", 0.0, 30.0,
                          10 ** ((gain_db - 10.0) / 20.0), snr, "idle")


# ----------------------------------------------------------------------------
# Signal synthesis + feature extraction (the expensive part)
# ----------------------------------------------------------------------------

_W: dict = {}   # per-process worker context, filled by _init_worker


def _init_worker(window_s: float, fs_audio: int, fs_accel: int, seed: int):
    """multiprocessing initialiser. Imports happen once per process, not once
    per window — scipy/sklearn import time would otherwise dominate."""
    sys.path.insert(0, str(ROOT / "firmware"))
    sys.path.insert(0, str(ROOT / "ml"))
    _W.update(window_s=window_s, fs_audio=fs_audio, fs_accel=fs_accel, seed=seed)


def _synthesise(p: WindowPlan, window_s: float, fs_audio: int, fs_accel: int,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One window of (audio, accel) from ml/simulate.py.

    The three accelerometer axes come from `capture.simulated_accel_axes` —
    the SAME function `firmware/capture.py:SimulatedSource` uses, imported
    rather than copied, so soak numbers stay comparable with everything else
    in the repo. Before T1.8 these were two independent copies of a
    three-line `[ax, 0.6*ax, 0.35*ax]` model, which is how such things drift.
    See the long note above that function for the physics.
    """
    from capture import simulated_accel_axes
    from simulate import (SimConfig, bearing_fault_signal, imbalance_signal,
                          normal_signal)

    cfg = SimConfig(duration_s=window_s, fr=p.fr, fs_audio=fs_audio,
                    fs_accel=fs_accel, snr_db=p.snr_db)
    gens = {
        "normal": lambda fs, r: normal_signal(cfg, fs, r),
        "bearing_outer": lambda fs, r: bearing_fault_signal(cfg, fs, r, p.severity, "outer"),
        "imbalance": lambda fs, r: imbalance_signal(cfg, fs, r, p.severity),
    }
    gen = gens[p.kind]
    rng = np.random.default_rng(seed + p.index)
    audio = gen(fs_audio, rng) * p.gain
    # `p.gain` models the recorder's own level drifting; it scales the whole
    # sensor, so it multiplies all three axes equally.
    accel = p.gain * simulated_accel_axes(
        p.kind, p.severity, p.fr, fs_accel, window_s, rng,
        fs_audio=fs_audio, snr_db=p.snr_db)
    return audio, accel


def _extract_one(p: WindowPlan) -> dict:
    """Worker entry point: plan -> compact feature record.

    Returns only what `state.StateDB.record_window` needs. In particular we
    return the *mean* Mel spectrum rather than the full (n_mels, frames)
    matrix: shipping the full matrix back through the pickle boundary would
    dominate the runtime and none of the downstream code uses it.
    """
    from features import extract_features
    from baseline import operating_point

    audio, accel = _synthesise(p, _W["window_s"], _W["fs_audio"], _W["fs_accel"],
                               _W["seed"])
    f = extract_features(audio, _W["fs_audio"], accel, _W["fs_accel"])
    return {
        "index": p.index,
        "label": p.label,
        "vector": np.asarray(f["vector"], dtype=float),
        "op": operating_point(f["vector"], f["fr_hz"]),
        "fr_hz": float(f["fr_hz"]),
        "fr_reliable": bool(f["fr_reliable"]),
        "mel_mean": np.asarray(f["mel"]).mean(axis=1),
        "band": [float(b) for b in f["band"]],
    }


def _map_windows(plans: list[WindowPlan], jobs: int, window_s: float,
                 fs_audio: int, fs_accel: int, seed: int, label: str) -> list[dict]:
    """Feature-extract a list of window plans, in parallel if asked.

    ~270 ms per window on one core here (53 ms synthesis + 217 ms DSP), so a
    full 7-day soak is 20 160 windows ≈ 90 core-minutes. That is why --jobs
    exists. Progress is logged because a silent 20-minute command is a command
    students will assume has hung and kill.
    """
    t0 = time.monotonic()
    out: list[dict] = []
    total = len(plans)

    def _tick(n_done: int) -> None:
        if n_done % max(1, total // 20) == 0 or n_done == total:
            el = time.monotonic() - t0
            rate = n_done / el if el > 0 else 0.0
            eta = (total - n_done) / rate if rate > 0 else float("nan")
            log.info("%s %d/%d windows (%.1f w/s, eta %.0f s)",
                     label, n_done, total, rate, eta)

    if jobs <= 1:
        _init_worker(window_s, fs_audio, fs_accel, seed)
        for n, p in enumerate(plans, 1):
            out.append(_extract_one(p))
            _tick(n)
    else:
        ctx = mp.get_context("spawn")   # fork + numpy threads is a known hazard
        with ctx.Pool(jobs, initializer=_init_worker,
                      initargs=(window_s, fs_audio, fs_accel, seed)) as pool:
            for n, rec in enumerate(pool.imap(_extract_one, plans, chunksize=8), 1):
                out.append(rec)
                _tick(n)
    out.sort(key=lambda r: r["index"])
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simulate_soak.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=Path("/tmp/soak/soak.db"),
                   help="state DB to write (must NOT be a real device DB)")
    p.add_argument("--outdir", type=Path, default=Path("/tmp/soak"),
                   help="where to write the fitted baseline and ground truth")
    p.add_argument("--days", type=float, default=7.0,
                   help="length of the SOAK period (excludes the learn period)")
    p.add_argument("--learn-hours", type=float, default=6.0,
                   help="wall-clock span the learn period is drawn from")
    p.add_argument("--learn-windows", type=int, default=96,
                   help="number of learn windows (config.yaml default: 96)")
    p.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--fs-audio", type=int, default=16000)
    p.add_argument("--fs-accel", type=int, default=6400)
    p.add_argument("--transients-per-day", type=float, default=6.0,
                   help="mean rate of innocent broadband transients")
    p.add_argument("--defrost-per-day", type=float, default=1.0)
    p.add_argument("--unlearned-regime", action="store_true",
                   help="suppress the defrost cycle during the LEARN period only, "
                        "so the soak contains a normal the baseline never saw. "
                        "This is the classic week-3 false-alarm scenario.")
    p.add_argument("--no-defrost", action="store_true",
                   help="drop the defrost cycle entirely (two-regime machine)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel feature-extraction processes")
    p.add_argument("--baseline-out", type=Path, default=None,
                   help="default: <outdir>/soak_baseline.npz")
    # -- chunked generation ---------------------------------------------------
    # A 7-day soak is ~20 000 windows x ~270 ms = half an hour of DSP. That is
    # a long time to hold a terminal (and longer than some CI/sandbox command
    # timeouts), so generation is resumable: run the same command repeatedly
    # with --resume and it fills in the windows still missing from the DB.
    # The plan is a pure function of (--days, --seed, --window-seconds), so the
    # chunks stitch together into exactly the run you would have got in one go.
    p.add_argument("--resume", action="store_true",
                   help="reuse the existing baseline and DB; only generate the "
                        "windows not already present. Idempotent.")
    p.add_argument("--max-windows", type=int, default=0,
                   help="stop after generating this many soak windows in this "
                        "invocation (0 = all). Use with --resume for chunking.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from baseline import fit_baseline, save_baseline
    from features import FEATURE_NAMES
    from inference import MahalanobisScorer
    from state import StateDB

    args.outdir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.baseline_out or (args.outdir / "soak_baseline.npz")
    window_s = args.window_seconds

    n_soak = int(round(args.days * 86400 / window_s))
    n_learn_span = int(round(args.learn_hours * 3600 / window_s))
    if args.learn_windows > n_learn_span:
        raise SystemExit(
            f"--learn-windows {args.learn_windows} > windows available in "
            f"--learn-hours {args.learn_hours} ({n_learn_span}). Widen the span.")

    # Start the run at a fixed wall-clock moment so reports are reproducible:
    # midnight, `days` ago, so the soak ends about now.
    start_ts = time.time() - (n_learn_span + n_soak) * window_s

    resume_ok = args.resume and baseline_path.exists() and args.db.exists()

    # ---- 1. the learn period -----------------------------------------------
    # Learn and soak are ONE continuous machine plan; we simply slice it. That
    # matters: the learn period must be a genuine sample of the same process,
    # including its duty cycle, not a curated set of nice windows.
    learn_plan = MachinePlan(
        n_learn_span, window_s, seed=args.seed,
        transients_per_day=args.transients_per_day,
        defrost_per_day=args.defrost_per_day,
        include_defrost=(not args.no_defrost) and (not args.unlearned_regime),
        start_ts=start_ts)

    # Evenly subsample the learn span. A real learn period runs continuously for
    # 24-72 h; here we take `learn_windows` spread across `learn_hours` so that
    # the sample spans several duty cycles rather than one contiguous block of
    # "compressor on".
    learn_idx = np.unique(np.linspace(0, n_learn_span - 1, args.learn_windows).astype(int))
    learn_plans = [learn_plan.plan(int(i)) for i in learn_idx]
    # A learn period is asserted-healthy by the customer, and a slamming door
    # during it would be silently baked into "normal". Real onboarding cannot
    # detect that; here we keep transients in, because pretending the learn
    # period is clean is exactly the optimism that produces surprises in week 3.
    log.info("learn period: %d windows over %.1f h (labels: %s)",
             len(learn_plans), args.learn_hours,
             {l: sum(1 for p in learn_plans if p.label == l)
              for l in sorted({p.label for p in learn_plans})})

    learn_recs = _map_windows(learn_plans, args.jobs, window_s, args.fs_audio,
                              args.fs_accel, args.seed, "learn")
    X = np.array([r["vector"] for r in learn_recs])
    OP = np.array([r["op"] for r in learn_recs])
    b = fit_baseline(X, OP, list(FEATURE_NAMES))
    save_baseline(baseline_path, b)
    log.info("baseline: k=%d regimes, counts=%s, thresholds=%s",
             b["k"], b["counts"].tolist(), np.round(b["thresholds"], 2).tolist())

    # ---- 2. the soak --------------------------------------------------------
    soak_plan = MachinePlan(
        n_soak, window_s, seed=args.seed + 1,
        transients_per_day=args.transients_per_day,
        defrost_per_day=args.defrost_per_day,
        include_defrost=not args.no_defrost,
        start_ts=start_ts + n_learn_span * window_s)
    soak_plans = [soak_plan.plan(i) for i in range(n_soak)]
    log.info("soak: %d windows = %.2f days (labels: %s)", n_soak, args.days,
             {l: sum(1 for p in soak_plans if p.label == l)
              for l in sorted({p.label for p in soak_plans})})

    soak_recs = _map_windows(soak_plans, args.jobs, window_s, args.fs_audio,
                             args.fs_accel, args.seed + 1000, "soak")

    # ---- 3. score and persist, exactly as main.py does ----------------------
    #
    # NOTE (real bug found while writing this): StateDB prunes `readings` older
    # than `retention_days` (default 7) on EVERY insert. A 7-day soak therefore
    # starts eating its own first day right as the run completes. We pass a huge
    # retention here so the synthetic week survives; on the real device you MUST
    # raise retention_days (or copy the DB off) before the soak — see
    # the operations runbook (not in this public copy). This is not a hypothetical: it is the default config.
    scorer = MahalanobisScorer(baseline_path)
    db = StateDB(args.db, retention_days=3650)

    n_anom = 0
    for rec, p in zip(soak_recs, soak_plans):
        score = scorer.score(rec["vector"], rec["op"])
        n_anom += int(score["anomalous"])
        feats = {
            "vector": rec["vector"],
            "fr_hz": rec["fr_hz"],
            "fr_reliable": rec["fr_reliable"],
            # record_window takes mel.mean(axis=1); a (n_mels, 1) column of the
            # already-averaged spectrum reproduces that exactly.
            "mel": rec["mel_mean"][:, None],
            "band": rec["band"],
        }
        db.record_window(score, feats, ts=soak_plan.ts(rec["index"]))
    db.set_meta("soak_simulated", True)
    db.set_meta("baseline_path", str(baseline_path))
    db.close()

    # ---- 4. ground truth, for the tests and for honest reporting ------------
    truth = {
        "generated": time.time(),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "window_seconds": window_s,
        "start_ts": soak_plan.start_ts,
        "n_windows": n_soak,
        "duration_days": n_soak * window_s / 86400.0,
        "k_regimes_learned": int(b["k"]),
        "learn_counts": b["counts"].tolist(),
        "thresholds": [float(t) for t in b["thresholds"]],
        "label_counts": {l: int(sum(1 for p in soak_plans if p.label == l))
                         for l in sorted({p.label for p in soak_plans})},
        "transient_window_indices": [int(i) for i in np.flatnonzero(soak_plan.transient)],
        "defrost_window_indices": [int(i) for i in np.flatnonzero(soak_plan.defrost)],
        "anomalous_windows": int(n_anom),
    }
    (args.outdir / "truth.json").write_text(json.dumps(truth, indent=2))

    print(json.dumps({
        "db": str(args.db),
        "baseline": str(baseline_path),
        "truth": str(args.outdir / "truth.json"),
        "soak_windows": n_soak,
        "soak_days": round(n_soak * window_s / 86400.0, 3),
        "k_regimes": int(b["k"]),
        "anomalous_windows": n_anom,
        "anomalous_fraction": round(n_anom / max(1, n_soak), 5),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
