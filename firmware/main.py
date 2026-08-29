"""
main.py — firmware entry point (v2):
capture -> features -> Mahalanobis score vs nearest regime -> persist gate ->
store -> publish -> alert (cloud MQTT + LAN webhook).

Demo (no hardware, no broker):
    python firmware/main.py --simulate --no-mqtt --fast --minutes 10 \
        --persist-minutes 2 --fault-at-minute 6 --transient-at-minute 2

Alert policy (the false-alarm defence stack, in order):
  1. regime assignment        — mode changes are not anomalies (baseline.py)
  2. persistence gate         — score must stay above threshold for
                                persist_minutes CONTINUOUSLY. A forklift, a
                                door slam, a wash-down: minutes. A bearing
                                fault: it does not heal. Time is the cheapest
                                feature that separates them.
  3. one alert per episode    — re-alerts only after scores return to normal.
  4. customer feedback        — "this was normal" folds the episode's windows
                                back into the baseline (baseline.py --retrain).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ml"))

from baseline import operating_point                     # noqa: E402
from capture import make_source                          # noqa: E402
from config_schema import ConfigError, load_config        # noqa: E402
from features import extract_features                    # noqa: E402
from inference import (AlertGate, BaselineMismatchError,          # noqa: E402
                       MahalanobisScorer, STARTUP_CHECK_WINDOWS)
from mqtt_client import MqttUplink, NullUplink           # noqa: E402
from reporting import (ScoreReporter, SeverityReference,  # noqa: E402
                       physical_severity)
from state import StateDB                                # noqa: E402

log = logging.getLogger("main")



def local_webhook(url: str, payload: dict) -> None:
    """Fire-and-forget LAN POST. Must never crash the loop."""
    if not url:
        return
    try:
        req = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:                                # noqa: BLE001
        log.warning("local webhook failed: %s", e)


def demo_schedule(window_s: float, fault_at_min: float, transient_at_min: float):
    """Scripted 10-minute demo:
      - regimes alternate every 4 windows (2 min) — must NOT alert
      - one single-window transient fault       — must NOT alert
      - persistent growing fault from fault_at  — must alert EXACTLY ONCE."""
    wpm = 60.0 / window_s

    def schedule(i: int) -> dict:
        minute = i / wpm
        fr = 50.0 if (i // 4) % 2 == 0 else 30.0
        if transient_at_min <= minute < transient_at_min + 1.0 / wpm:
            return {"kind": "bearing_outer", "severity": 0.5, "fr": fr}
        if minute >= fault_at_min:
            sev = min(0.15 + 0.05 * (minute - fault_at_min), 0.8)
            return {"kind": "bearing_outer", "severity": sev, "fr": fr}
        return {"kind": "normal", "severity": 0.0, "fr": fr}
    return schedule


def run(cfg: dict, args) -> int:
    baseline_path = args.baseline
    if not baseline_path.exists():
        raise SystemExit(f"{baseline_path} missing — run firmware/baseline.py first")

    scorer = MahalanobisScorer(baseline_path)
    # Display only — see reporting.py. Neither of these can change whether an
    # alert fires; `scorer.score()` and `gate.feed()` are untouched by them.
    reporter = ScoreReporter(baseline_path)
    sev_ref = SeverityReference(baseline_path)
    db = StateDB(args.db or cfg["storage"]["sqlite_path"],
                 cfg["storage"]["retention_days"])

    def on_command(cmd: dict) -> None:
        # Cloud downlink. mark_normal: the customer pressed "this was normal" —
        # bank those windows for the next baseline retrain.
        if cmd.get("cmd") == "mark_normal":
            n = db.mark_normal(float(cmd["ts_from"]), float(cmd["ts_to"]))
            log.info("feedback: %d windows banked for retrain", n)

    uplink = NullUplink() if args.no_mqtt else MqttUplink(cfg, on_command)
    uplink.start()

    window_s = cfg["window"]["seconds"]
    persist_min = args.persist_minutes or cfg["anomaly"]["persist_minutes"]
    gate = AlertGate(need=round(persist_min * 60 / window_s))
    fs_a, fs_v = cfg["audio"]["sample_rate"], cfg["accelerometer"]["sample_rate"]

    schedule = (demo_schedule(window_s, args.fault_at_minute, args.transient_at_minute)
                if args.simulate else None)
    source = make_source(cfg, simulate=args.simulate, schedule=schedule,
                         realtime=args.simulate and not args.fast, seed=4242)

    n_windows = round(args.minutes * 60 / window_s) if args.minutes else 0
    alerts = 0
    # T3.7: the first STARTUP_CHECK_WINDOWS real windows this PROCESS scores
    # are checked once against the baseline's own learn-period fingerprint —
    # see MahalanobisScorer.check_startup_fingerprint's docstring for why
    # this lives here (an explicit, one-time call at real startup) rather
    # than inside scorer.score() itself.
    startup_ratios: list[float] = []
    try:
        for i, (audio, accel) in enumerate(source.windows()):
            if n_windows and i >= n_windows:
                break
            t0 = time.monotonic()
            # T1.13: use this unit's own calibrated crest_floor (falls back to
            # the pre-T1.13 constant for a baseline that predates the field —
            # see MahalanobisScorer.crest_floor).
            feats = extract_features(audio, fs_a, accel, fs_v, crest_floor=scorer.crest_floor)
            op = operating_point(feats["vector"], feats["fr_hz"])
            score = scorer.score(feats["vector"], op)

            if startup_ratios is not None and len(startup_ratios) < STARTUP_CHECK_WINDOWS:
                startup_ratios.append(
                    score["score"] / score["threshold"] if score["threshold"] > 0 else float("inf"))
                if len(startup_ratios) == STARTUP_CHECK_WINDOWS:
                    try:
                        scorer.check_startup_fingerprint(startup_ratios)
                    except BaselineMismatchError as e:
                        # Deliberately NOT a normal alert — a persistence-gated
                        # fault alert says "your machine broke"; this says
                        # "your firmware and your baseline disagree", which
                        # needs a retrain, not a mechanic.
                        print(f"\nbaseline mismatch: {e}", file=sys.stderr)
                        sys.exit(1)
            latency_ms = (time.monotonic() - t0) * 1000
            db.record_window(score, feats)

            # The gate is fed BEFORE telemetry is published so that the window
            # which fires an alert reports tier="red" in its own message rather
            # than one window later. `feed` is deterministic and its return
            # value is used only for alerting, so moving the call changes the
            # order of two statements and nothing else — verified by the
            # 90-minute simulation still raising exactly one alert.
            fired = gate.feed(score["anomalous"])

            # Display layer (T1.7). Adds ~75 ms to a ~490 ms window in this
            # sandbox — 15 %, inside the 30 s budget by three orders of
            # magnitude, but measured rather than assumed because the Pi's A53
            # is slower than this container.
            report = reporter.report(score["score"], score["regime"],
                                     score["anomalous"], alerting=gate.in_episode)
            sev = sev_ref.relative(
                physical_severity(audio, fs_a, feats["band"]))

            uplink.publish_telemetry({
                "ts": time.time(), "window": i,
                "score": score["score"], "threshold": score["threshold"],
                "regime": score["regime"], "anomalous": score["anomalous"],
                # Display index, tier and PHYSICAL severity. The raw score above
                # stays in the payload: it is what the alert decision used, and
                # an engineer debugging a unit needs the number the decision saw.
                "display_index": round(report["index"], 2),
                "tier": report["tier"],
                "score_percentile": round(report["percentile"], 4),
                "severity_band_rms_db": round(sev["band_rms_db"], 2),
                "severity_env_peak_hz": round(sev["env_peak_hz"], 2),
                "severity_env_peak_ratio": round(sev["env_peak_ratio"], 2),
                "severity_env_db_re_learn": round(sev["env_energy_db_re_learn"], 2),
                "fr_hz": feats["fr_hz"], "fr_reliable": feats["fr_reliable"],
                "band": list(feats["band"]),
                "mel_mean": [round(float(v), 3) for v in feats["mel"].mean(axis=1)],
                "latency_ms": round(latency_ms, 1),
            })

            log.info("w%02d regime=%d score=%6.2f thr=%5.2f idx=%5.1f %-5s "
                     "sev=%+5.1f dB @%6.1f Hz streak=%d/%d (%.0f ms)",
                     i, score["regime"], score["score"], score["threshold"],
                     report["index"], report["tier"],
                     sev["env_energy_db_re_learn"], sev["env_peak_hz"],
                     gate.streak, gate.need, latency_ms)

            if fired:
                alerts += 1
                episode_start = time.time() - gate.need * window_s
                event = {"ts": time.time(), "device_id": cfg["device"]["id"],
                         "machine": cfg["device"]["name"],
                         "score": score["score"], "threshold": score["threshold"],
                         "regime": score["regime"],
                         "ts_from": episode_start, "ts_to": time.time(),
                         "persisted_minutes": persist_min}
                log.warning("ALERT #%d  score=%.2f (thr %.2f) persisted %.0f min",
                            alerts, score["score"], score["threshold"], persist_min)
                uplink.publish_anomaly(event)
                local_webhook(cfg["local_alert"]["webhook_url"], event)
    finally:
        uplink.stop()
        db.close()
    return alerts


def main() -> None:
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--no-mqtt", action="store_true")
    p.add_argument("--fast", action="store_true", help="don't pace to wall clock")
    p.add_argument("--minutes", type=float, default=0, help="stop after N simulated minutes")
    p.add_argument("--persist-minutes", type=float, default=None)
    p.add_argument("--fault-at-minute", type=float, default=6.0)
    p.add_argument("--transient-at-minute", type=float, default=2.0)
    p.add_argument("--baseline", type=Path, default=ROOT / "baseline.npz")
    p.add_argument("--db", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.simulate and args.db is None:
        args.db = ROOT.parent / "data" / "sim_state.db"

    alerts = run(cfg, args)
    print(f"\ndone: {alerts} alert(s) raised")


if __name__ == "__main__":
    main()
