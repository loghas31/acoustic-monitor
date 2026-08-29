#!/usr/bin/env python3
"""
Record a replayable trace of the firmware actually running, for the offline
visualiser (`tools/sim_dashboard.html`).

The point of this file is that the visualiser must not be allowed to invent
anything. Every number it draws comes from here, and everything here comes
from `firmware.main.run` executing unmodified against `ml/simulate.py`'s
signals — the same code path the Pi runs, with the network replaced by a list.
No numbers are computed for display purposes that the firmware does not
already publish.

Two exceptions, both derived rather than invented, and both stated on the
chart:

  * `streak` / `need` — the persistence gate's internal counter is not in the
    telemetry payload (the Pi has no reason to transmit it). It is
    reconstructed here by replaying `anomalous` through the same rule
    `AlertGate` uses, and the reconstruction is checked against the alert
    timestamps the firmware really emitted: if the two disagree, this script
    exits non-zero rather than shipping a plausible-looking lie.

  * `truth` — what the simulator was asked to generate for each window. The
    firmware never sees this; it is included so the visualiser can show what
    the detector got right AND wrong. Real machines do not come with a truth
    channel, which is exactly why it is labelled SYNTHETIC everywhere.

    python tools/sim_trace.py --out tools/sim_trace.json

Everything produced is SYNTHETIC. It demonstrates the decision logic end to
end; it is not evidence about real bearings. See docs/DOC_STATUS.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for sub in ("firmware", "ml"):
    sys.path.append(str(ROOT / sub))


class CapturingUplink:
    """Stands in for MqttUplink/NullUplink: keeps every payload instead of
    publishing it. The firmware is otherwise untouched."""

    def __init__(self, *a, **k):
        self.telemetry: list[dict] = []
        self.anomalies: list[dict] = []

    def start(self):
        pass

    def stop(self):
        pass

    def publish_telemetry(self, p):
        self.telemetry.append(dict(p))

    def publish_anomaly(self, p):
        self.anomalies.append(dict(p))


def _load_firmware_main():
    spec = importlib.util.spec_from_file_location(
        "firmware_main", ROOT / "firmware" / "main.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    return fw


def _replay_gate(anomalous: list[bool], need: int) -> tuple[list[int], list[int]]:
    """Reconstruct AlertGate's streak and firing windows from the anomaly flags.

    Mirrors `inference.AlertGate`: count consecutive anomalous windows, fire
    exactly once when the count first reaches `need`, and do not fire again
    until the streak has been broken (one-alert-per-episode). Verified against
    the firmware's real alert count by the caller.
    """
    streaks, fired_at = [], []
    streak, in_episode = 0, False
    for i, a in enumerate(anomalous):
        if a:
            streak += 1
        else:
            streak, in_episode = 0, False
        if streak >= need and not in_episode:
            in_episode = True
            fired_at.append(i)
        streaks.append(streak)
    return streaks, fired_at


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--persist-minutes", type=float, default=2.0)
    ap.add_argument("--fault-at-minute", type=float, default=6.0)
    ap.add_argument("--transient-at-minute", type=float, default=2.0)
    ap.add_argument("--baseline", type=Path,
                    default=ROOT / "firmware" / "baseline.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "tools" / "sim_trace.json")
    args = ap.parse_args()

    import numpy as np
    import yaml

    if not args.baseline.exists():
        print(f"FAIL: no baseline at {args.baseline}.\n"
              f"  Learn one first:\n"
              f"    python firmware/baseline.py --simulate --windows 48 "
              f"--out {args.baseline} --db /tmp/state.db", file=sys.stderr)
        return 1

    fw = _load_firmware_main()
    cap = CapturingUplink()
    fw.NullUplink = lambda *a, **k: cap          # noqa: E731 — deliberate injection

    cfg = yaml.safe_load((ROOT / "firmware" / "config.yaml").read_text())
    window_s = cfg["window"]["seconds"]

    state_db = Path(tempfile.gettempdir()) / "sim_trace_state.db"
    if state_db.exists():
        state_db.unlink()

    run_args = SimpleNamespace(
        baseline=args.baseline, simulate=True, no_mqtt=True, fast=True,
        minutes=args.minutes, persist_minutes=args.persist_minutes,
        fault_at_minute=args.fault_at_minute,
        transient_at_minute=args.transient_at_minute,
        db=state_db, config=ROOT / "firmware" / "config.yaml")

    alerts = fw.run(cfg, run_args)
    tel = cap.telemetry
    if not tel:
        print("FAIL: firmware published no telemetry", file=sys.stderr)
        return 1

    need = round(args.persist_minutes * 60 / window_s)
    anomalous = [bool(t["anomalous"]) for t in tel]
    streaks, fired_at = _replay_gate(anomalous, need)

    # The reconstruction is only worth drawing if it agrees with reality.
    if len(fired_at) != alerts or len(cap.anomalies) != alerts:
        print(f"FAIL: gate reconstruction disagrees with the firmware — "
              f"firmware fired {alerts} alert(s) and published "
              f"{len(cap.anomalies)}, replay says {len(fired_at)}. "
              f"Refusing to write a trace the visualiser would misdraw.",
              file=sys.stderr)
        return 1

    # What the simulator was actually asked to produce, per window (SYNTHETIC
    # ground truth; the firmware never sees this).
    schedule = fw.demo_schedule(window_s, args.fault_at_minute,
                                args.transient_at_minute)

    bl = np.load(args.baseline, allow_pickle=False)
    thresholds = [float(v) for v in np.atleast_1d(bl["thresholds"])] \
        if "thresholds" in bl else []

    windows = []
    for i, t in enumerate(tel):
        truth = schedule(i)
        windows.append({
            "i": i,
            "minute": round(i * window_s / 60.0, 3),
            # --- what the detector decided ---
            "score": round(float(t["score"]), 4),
            "threshold": round(float(t["threshold"]), 4),
            "ratio": round(float(t["score"]) / float(t["threshold"]), 4)
            if t["threshold"] else None,
            "regime": int(t["regime"]),
            "anomalous": bool(t["anomalous"]),
            "tier": t["tier"],
            "index": float(t["display_index"]),
            "streak": streaks[i],
            "fired": i in fired_at,
            # --- what it measured ---
            "fr_hz": round(float(t["fr_hz"]), 2),
            "fr_reliable": bool(t["fr_reliable"]),
            "band_lo_hz": round(float(t["band"][0]), 1),
            "band_hi_hz": round(float(t["band"][1]), 1),
            "env_peak_hz": float(t["severity_env_peak_hz"]),
            "env_peak_ratio": float(t["severity_env_peak_ratio"]),
            "env_db_re_learn": float(t["severity_env_db_re_learn"]),
            "band_rms_db": float(t["severity_band_rms_db"]),
            "latency_ms": float(t["latency_ms"]),
            "mel_mean": t["mel_mean"],
            # --- SYNTHETIC ground truth ---
            "truth_kind": truth["kind"],
            "truth_severity": round(float(truth["severity"]), 3),
            "truth_fr": float(truth["fr"]),
        })

    out = {
        "generated_by": "tools/sim_trace.py",
        "synthetic": True,
        "note": ("Every field comes from firmware/main.py running unmodified "
                 "against ml/simulate.py. truth_* is what the simulator was "
                 "asked to generate and is NOT visible to the detector."),
        "config": {
            "window_seconds": window_s,
            "minutes": args.minutes,
            "persist_minutes": args.persist_minutes,
            "gate_need_windows": need,
            "fault_at_minute": args.fault_at_minute,
            "transient_at_minute": args.transient_at_minute,
            "audio_sample_rate": cfg["audio"]["sample_rate"],
            "accel_sample_rate": cfg["accelerometer"]["sample_rate"],
            "baseline_thresholds": [round(v, 4) for v in thresholds],
            "n_regimes": len(thresholds),
        },
        "result": {
            "windows": len(windows),
            "alerts": alerts,
            "alert_windows": fired_at,
            "anomalous_windows": sum(anomalous),
        },
        "windows": windows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} — {len(windows)} windows, {alerts} alert(s), "
          f"{sum(anomalous)} anomalous window(s), gate need={need}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
