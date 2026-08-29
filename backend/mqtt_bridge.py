"""
MQTT -> database bridge (v2). Runs as its own process (see docker-compose):
the API restarts freely; the bridge holds a persistent session so QoS-1
anomaly events queue at the broker while it's down.

Topic contract (mirrors firmware/mqtt_client.py):
    devices/{id}/telemetry  -> readings row + health/last_seen update
    devices/{id}/anomaly    -> anomaly_events row + alert fan-out
    devices/{id}/status     -> online flag

Downlink (published by the API, consumed by the device):
    devices/{id}/cmd        e.g. {"cmd": "mark_normal", "ts_from": ..., "ts_to": ...}
"""

from __future__ import annotations

import json
import logging
import os
import time

import paho.mqtt.client as mqtt

from alerts import dispatch
from models import (Alert, AlertConfig, AnomalyEvent, Device, Reading,
                    SessionLocal, init_db)

log = logging.getLogger("bridge")

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))


def make_client(client_id: str, clean_session: bool = False) -> mqtt.Client:
    """paho 1.x/2.x compat shim (2.x requires an explicit callback API version)."""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                           client_id=client_id, clean_session=clean_session)
    except AttributeError:
        return mqtt.Client(client_id=client_id, clean_session=clean_session)


def health_from_score(score, threshold, anomalous: bool, tier: str | None = None) -> str:
    """green/amber/red for the fleet view.

    CORRECTED, backlog T1.7 / self-review F5. The old rule was
    "amber = score above 70 % of threshold", a band in score MAGNITUDE. F5
    predicted that band would never fire. Measured over 200 fresh healthy
    windows against the repo baseline it fired on **16.5 %** of them — and on
    only **12.5 %** of the windows of a fault ramped from severity 0.002 to
    0.05. It was a "watch this one" badge more likely on a healthy machine
    than on a failing one, which is worse than useless: it teaches the
    customer that colour on this dashboard means nothing.

    The cause is that the healthy score distribution's own upper tail lives
    inside the band (median 0.580x threshold, p95 0.762x, max 1.034x), while a
    developing fault crosses it in roughly one severity doubling.

    The device now sends a `tier` computed from STATE rather than magnitude
    (reporting.tier_from): red = the persistence gate has fired, amber = this
    window is above threshold but not yet persistent, green = below. Measured
    0.5 % of healthy windows vs 40 % of ramp windows — it separates. We prefer
    the device's tier when present and fall back to `anomalous` for units on
    older firmware, which is the same rule minus the dead magnitude band.
    """
    if tier in ("green", "amber", "red"):
        return tier
    if anomalous:
        return "red"
    return "green"


def _num(v):
    """None-preserving float coercion.

    The severity fields are legitimately absent (pre-T1.7 firmware) and can
    legitimately be -inf (a silent window: `_db20(0)`), and those two must not
    be confused. Absent stays NULL; non-finite becomes NULL too, because a
    JSON payload cannot carry Infinity and a chart cannot plot it — but the
    reading row itself is still written, so the window is not lost.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


TELEMETRY_FLOAT_FIELDS = ("display_index", "score_percentile",
                          "severity_band_rms_db", "severity_env_peak_hz",
                          "severity_env_peak_ratio", "severity_env_db_re_learn")


def handle_telemetry(db, device_id: str, p: dict) -> None:
    # T1.11: the reportable fields are persisted as well as consumed. Before
    # this, `tier` moved the fleet colour and was then thrown away, and the
    # four severity fields were dropped entirely — so the device computed a
    # trend every 30 s that nothing could ever draw.
    tier = p.get("tier")
    db.add(Reading(device_id=device_id, ts=p.get("ts", time.time()),
                   score=p.get("score"), threshold=p.get("threshold"),
                   regime=p.get("regime"), anomalous=bool(p.get("anomalous")),
                   fr_hz=p.get("fr_hz"), fr_reliable=bool(p.get("fr_reliable")),
                   band=p.get("band"), mel_mean=p.get("mel_mean"),
                   tier=tier if tier in ("green", "amber", "red") else None,
                   **{f: _num(p.get(f)) for f in TELEMETRY_FLOAT_FIELDS}))
    dev = db.get(Device, device_id)
    if dev:
        dev.last_seen_ts = time.time()
        dev.online = True
        dev.health = health_from_score(p.get("score"), p.get("threshold"),
                                       bool(p.get("anomalous")), p.get("tier"))
    db.commit()


def handle_anomaly(db, device_id: str, p: dict) -> None:
    event = AnomalyEvent(device_id=device_id, ts=p.get("ts", time.time()),
                         score=p.get("score"), threshold=p.get("threshold"),
                         regime=p.get("regime"),
                         ts_from=p.get("ts_from"), ts_to=p.get("ts_to"),
                         persisted_minutes=p.get("persisted_minutes"))
    db.add(event)
    dev = db.get(Device, device_id)
    if dev:
        dev.health = "red"
    db.commit()

    cfg = db.get(AlertConfig, device_id)
    for rec in dispatch(p, cfg):
        db.add(Alert(event_id=event.id, **rec))
    db.commit()


def handle_status(db, device_id: str, p: dict) -> None:
    dev = db.get(Device, device_id)
    if dev:
        dev.online = bool(p.get("online"))
        db.commit()


HANDLERS = {"telemetry": handle_telemetry, "anomaly": handle_anomaly,
            "status": handle_status}


def on_message(client, userdata, msg):
    parts = msg.topic.split("/")          # devices/{id}/{kind}
    if len(parts) != 3:
        return
    _, device_id, kind = parts
    handler = HANDLERS.get(kind)
    if handler is None:
        return
    try:
        payload = json.loads(msg.payload)
    except json.JSONDecodeError:
        log.warning("bad payload on %s", msg.topic)
        return
    db = SessionLocal()
    try:
        if db.get(Device, device_id) is None:
            log.warning("telemetry from unregistered device %s dropped", device_id)
            return
        handler(db, device_id, payload)
    except Exception:                                        # noqa: BLE001
        log.exception("handler failed for %s", msg.topic)
        db.rollback()
    finally:
        db.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    client = make_client("cloud-bridge")
    client.on_message = on_message
    client.on_connect = lambda c, u, f, rc: (log.info("bridge connected rc=%s", rc),
                                             c.subscribe("devices/+/+", qos=1))
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
