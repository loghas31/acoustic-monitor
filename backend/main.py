"""
FastAPI backend (v2). Local dev: uvicorn main:app --reload
Production: docker-compose up (Postgres + Mosquitto + this + the bridge).

Auth domains, deliberately separate:
    devices -> X-API-Key header (issued once at registration)
    humans  -> Bearer JWT (email/password login)

HTTP ingest duplicates the MQTT path on purpose (some customer networks block
non-HTTP egress; curl beats mosquitto during development). Both paths share
the handler functions in mqtt_bridge.py — one source of truth.

The feedback endpoint is the false-alarm kill switch: one tap on "this was
normal" (a) records the verdict, (b) tells the device — via MQTT cmd — to fold
that episode's windows back into its baseline at next retrain. Alert fatigue
is the #1 churn risk; this is the customer's direct lever on it.
"""

from __future__ import annotations

import json
import logging
import os
import time

from pathlib import Path

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Header,
                     HTTPException, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import Integer, cast, func

import auth
import models
import recordings
from models import (Alert, AlertConfig, AnomalyEvent, Device, Reading,
                    SessionLocal, User, init_db)
from mqtt_bridge import handle_anomaly, handle_telemetry, make_client

log = logging.getLogger("api")

app = FastAPI(title="Acoustic Monitor API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])   # MVP: tighten to dashboard origin before launch

MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))


@app.on_event("startup")
def _startup() -> None:
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def publish_cmd(device_id: str, cmd: dict) -> bool:
    """Best-effort one-shot downlink. The DB stays the source of truth; if the
    broker is unreachable the verdict is still recorded and the device picks
    it up on a later retrain cycle."""
    try:
        c = make_client(f"api-cmd-{time.time():.0f}", clean_session=True)
        c.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
        c.loop_start()
        info = c.publish(f"devices/{device_id}/cmd", json.dumps(cmd), qos=1)
        info.wait_for_publish(timeout=3)
        c.loop_stop(); c.disconnect()
        return True
    except Exception as e:                                   # noqa: BLE001
        log.warning("cmd publish to %s failed: %s", device_id, e)
        return False


# -- auth dependencies ---------------------------------------------------------

def current_user(db=Depends(get_db), authorization: str = Header(default="")) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    user_id = auth.verify_token(authorization.removeprefix("Bearer "))
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(401, "invalid token")
    return user


def current_device(db=Depends(get_db), x_api_key: str = Header(default="")) -> Device:
    if not x_api_key:
        raise HTTPException(401, "missing X-API-Key")
    dev = db.query(Device).filter_by(api_key_hash=auth.hash_api_key(x_api_key)).first()
    if dev is None:
        raise HTTPException(401, "invalid API key")
    return dev


def owned_device(device_id: str, user: User, db) -> Device:
    dev = db.get(Device, device_id)
    if dev is None or dev.user_id != user.id:
        raise HTTPException(404, "device not found")   # 404 not 403: don't leak existence
    return dev


# -- schemas ---------------------------------------------------------------------

class RegisterUserIn(BaseModel):
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class RegisterDeviceIn(BaseModel):
    name: str

class ReadingIn(BaseModel):
    ts: float | None = None
    score: float | None = None
    threshold: float | None = None
    regime: int | None = None
    anomalous: bool = False
    # Device-computed fleet colour (firmware/reporting.tier_from). Optional so
    # units on pre-T1.7 firmware keep working; `health_from_score` falls back
    # to `anomalous` when it is absent.
    tier: str | None = None
    # T1.11 reportable layer. All optional for the same reason as `tier`; the
    # bridge coerces and stores them (see mqtt_bridge.TELEMETRY_FLOAT_FIELDS).
    display_index: float | None = None
    score_percentile: float | None = None
    severity_band_rms_db: float | None = None
    severity_env_peak_hz: float | None = None
    severity_env_peak_ratio: float | None = None
    severity_env_db_re_learn: float | None = None
    fr_hz: float | None = None
    fr_reliable: bool = False
    band: list[float] | None = None
    mel_mean: list[float] | None = None

class AlertConfigIn(BaseModel):
    email: EmailStr | None = None
    webhook_url: str | None = None
    enabled: bool = True

class FeedbackIn(BaseModel):
    verdict: str        # "normal" -> retrain baseline | "fault" -> confirmed catch


# -- users ------------------------------------------------------------------------

@app.post("/auth/register", status_code=201)
def register_user(body: RegisterUserIn, db=Depends(get_db)):
    if db.query(User).filter_by(email=body.email).first():
        raise HTTPException(409, "email already registered")
    user = User(email=body.email, password_hash=auth.hash_password(body.password))
    db.add(user); db.commit()
    return {"token": auth.make_token(user.id)}


@app.post("/auth/login")
def login(body: LoginIn, db=Depends(get_db)):
    user = db.query(User).filter_by(email=body.email).first()
    if user is None or not auth.verify_password(body.password, user.password_hash):
        raise HTTPException(401, "bad credentials")
    return {"token": auth.make_token(user.id)}


# -- devices ------------------------------------------------------------------------

@app.post("/devices/register", status_code=201)
def register_device(body: RegisterDeviceIn, user: User = Depends(current_user),
                    db=Depends(get_db)):
    key, key_hash = auth.new_api_key()
    dev = Device(user_id=user.id, name=body.name, api_key_hash=key_hash)
    db.add(dev); db.commit()
    return {"device_id": dev.id, "api_key": key}   # plaintext key: shown exactly once


@app.get("/devices/{device_id}/status")
def device_status(device_id: str, user: User = Depends(current_user), db=Depends(get_db)):
    dev = owned_device(device_id, user, db)
    last = (db.query(Reading).filter_by(device_id=dev.id)
            .order_by(Reading.ts.desc()).first())
    return {"device_id": dev.id, "name": dev.name, "online": dev.online,
            "health": dev.health, "last_seen_ts": dev.last_seen_ts,
            "latest": None if last is None else {
                "ts": last.ts, "score": last.score, "threshold": last.threshold,
                "regime": last.regime, "fr_hz": last.fr_hz,
                "fr_reliable": last.fr_reliable, "band": last.band,
                "tier": last.tier, "display_index": last.display_index,
                "severity_band_rms_db": last.severity_band_rms_db,
                "severity_env_peak_hz": last.severity_env_peak_hz,
                "severity_env_peak_ratio": last.severity_env_peak_ratio,
                "severity_env_db_re_learn": last.severity_env_db_re_learn}}


@app.get("/devices/{device_id}/anomalies")
def device_anomalies(device_id: str, since: float = 0.0,
                     user: User = Depends(current_user), db=Depends(get_db)):
    dev = owned_device(device_id, user, db)
    rows = (db.query(AnomalyEvent).filter(AnomalyEvent.device_id == dev.id,
                                          AnomalyEvent.ts >= since)
            .order_by(AnomalyEvent.ts.desc()).limit(500).all())
    return [{"id": r.id, "ts": r.ts, "score": r.score, "threshold": r.threshold,
             "regime": r.regime, "ts_from": r.ts_from, "ts_to": r.ts_to,
             "persisted_minutes": r.persisted_minutes,
             "feedback": r.feedback, "acknowledged": r.acknowledged} for r in rows]


@app.get("/devices/{device_id}/readings")
def device_readings(device_id: str, since: float = 0.0, limit: int = 2880,
                    user: User = Depends(current_user), db=Depends(get_db)):
    dev = owned_device(device_id, user, db)
    rows = (db.query(Reading).filter(Reading.device_id == dev.id, Reading.ts >= since)
            .order_by(Reading.ts.desc()).limit(limit).all())
    return [{"ts": r.ts, "score": r.score, "threshold": r.threshold,
             "regime": r.regime, "anomalous": r.anomalous, "fr_hz": r.fr_hz,
             "mel_mean": r.mel_mean,
             # T1.11: the chartable layer. `display_index` is what the device
             # page plots (comparable across regimes; 70 == threshold);
             # `score` stays for engineers debugging the alert decision.
             "tier": r.tier, "display_index": r.display_index,
             "severity_band_rms_db": r.severity_band_rms_db,
             "severity_env_peak_hz": r.severity_env_peak_hz,
             "severity_env_peak_ratio": r.severity_env_peak_ratio,
             "severity_env_db_re_learn": r.severity_env_db_re_learn}
            for r in reversed(rows)]


# -- ingest (HTTP twin of the MQTT path) ----------------------------------------------

@app.post("/readings", status_code=201)
def ingest_reading(body: ReadingIn, dev: Device = Depends(current_device),
                   db=Depends(get_db)):
    handle_telemetry(db, dev.id, body.model_dump())
    return {"ok": True}


@app.post("/anomalies", status_code=201)
def ingest_anomaly(body: dict, dev: Device = Depends(current_device), db=Depends(get_db)):
    handle_anomaly(db, dev.id, body)
    return {"ok": True}


# -- phone recordings: audio in, verdict out ------------------------------------------
#
# NOT the production path. The Pi extracts features on-device and posts ~400
# bytes to /readings; it never uploads audio. This exists so a phone can drive
# the same pipeline without that pipeline being ported to iOS — see
# backend/recordings.py and the phone deployment guide (not in this public copy).

MAX_UPLOAD_BYTES = 200 * 1024 * 1024      # ~3 h of 16 kHz mono WAV


@app.post("/recordings", status_code=202)
async def upload_recording(background: BackgroundTasks,
                           file: UploadFile = File(...),
                           learn_windows: int = 48,
                           dev: Device = Depends(current_device),
                           db=Depends(get_db)):
    """Accept an audio file and queue it. Returns immediately with an id.

    202, not 201: the work has not been done yet. A 28-minute recording takes
    about a minute to analyse and a phone on mobile data will not hold the
    request open, so the client polls GET /recordings/{id}.

    `learn_windows` defaults to 48 (24 minutes) because `DOC_STATUS.md`
    measured that fewer gives a **55-59 % held-out false-alarm rate** for this
    37-dimensional feature vector — a verdict from a short learn period is
    noise, not a measurement. It is exposed anyway so the route can be
    exercised without 24 minutes of audio, and any value below 48 is marked
    `learn_period_too_short` in the verdict so nobody mistakes a plumbing
    check for a health check.
    """
    if learn_windows < 1:
        raise HTTPException(400, "learn_windows must be >= 1")
    recordings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rec = models.Recording(device_id=dev.id, uploaded_ts=time.time(),
                           filename=file.filename or "upload", status="queued")
    db.add(rec)
    db.flush()                                    # need rec.id for the path

    suffix = Path(file.filename or "").suffix.lower()[:8] or ".wav"
    dest = recordings.UPLOAD_DIR / f"{rec.id}{suffix}"
    written = 0
    with open(dest, "wb") as fh:
        # Streamed in chunks: a 200 MB upload read into memory at once would
        # take the whole server down, and "the server died" is a much worse
        # failure than "your file is too big".
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                db.rollback()
                raise HTTPException(
                    413, f"recording exceeds {MAX_UPLOAD_BYTES // (1 << 20)} MB")
            fh.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        db.rollback()
        raise HTTPException(400, "empty upload")

    rec.path, rec.bytes = str(dest), written
    db.commit()
    background.add_task(recordings.process, rec.id, models.SessionLocal,
                        learn_windows)
    return {"recording_id": rec.id, "status": "queued", "bytes": written,
            "learn_windows": learn_windows}


@app.get("/recordings/{recording_id}")
def get_recording(recording_id: str, dev: Device = Depends(current_device),
                  db=Depends(get_db)):
    rec = db.query(models.Recording).filter_by(id=recording_id).first()
    if rec is None or rec.device_id != dev.id:
        # Same response for "not yours" as "not found": a device must not be
        # able to enumerate another device's recordings by id.
        raise HTTPException(404, "no such recording")
    return {"recording_id": rec.id, "status": rec.status,
            "filename": rec.filename, "bytes": rec.bytes,
            "uploaded_ts": rec.uploaded_ts,
            "verdict": rec.verdict, "error": rec.error}


@app.get("/recordings")
def list_recordings(dev: Device = Depends(current_device), db=Depends(get_db),
                    limit: int = 20):
    rows = (db.query(models.Recording).filter_by(device_id=dev.id)
            .order_by(models.Recording.uploaded_ts.desc()).limit(limit).all())
    return [{"recording_id": r.id, "status": r.status, "filename": r.filename,
             "uploaded_ts": r.uploaded_ts,
             "flagged_pct": (r.verdict or {}).get("windows_above_threshold_pct")}
            for r in rows]


# -- feedback: the false-alarm kill switch ---------------------------------------------

@app.post("/anomalies/{event_id}/feedback")
def anomaly_feedback(event_id: str, body: FeedbackIn,
                     user: User = Depends(current_user), db=Depends(get_db)):
    if body.verdict not in ("normal", "fault"):
        raise HTTPException(422, "verdict must be 'normal' or 'fault'")
    event = db.get(AnomalyEvent, event_id)
    if event is None:
        raise HTTPException(404, "event not found")
    owned_device(event.device_id, user, db)        # authorisation check

    event.feedback = body.verdict
    event.acknowledged = True
    pushed = False
    if body.verdict == "normal" and event.ts_from and event.ts_to:
        # Tell the device to bank those windows for its next baseline retrain.
        pushed = publish_cmd(event.device_id, {
            "cmd": "mark_normal", "ts_from": event.ts_from, "ts_to": event.ts_to})
        dev = db.get(Device, event.device_id)
        if dev and dev.health == "red":
            dev.health = "amber"                   # de-escalate after human verdict
    db.commit()
    return {"ok": True, "verdict": body.verdict, "device_notified": pushed}


# -- alert configuration -----------------------------------------------------------------

@app.post("/alerts/configure")
def configure_alerts(device_id: str, body: AlertConfigIn,
                     user: User = Depends(current_user), db=Depends(get_db)):
    dev = owned_device(device_id, user, db)
    cfg = db.get(AlertConfig, dev.id) or AlertConfig(device_id=dev.id)
    cfg.email, cfg.webhook_url, cfg.enabled = body.email, body.webhook_url, body.enabled
    db.merge(cfg); db.commit()
    return {"ok": True}


@app.get("/alerts/log/{device_id}")
def alert_log(device_id: str, user: User = Depends(current_user), db=Depends(get_db)):
    dev = owned_device(device_id, user, db)
    rows = (db.query(Alert).join(AnomalyEvent, Alert.event_id == AnomalyEvent.id)
            .filter(AnomalyEvent.device_id == dev.id)
            .order_by(Alert.ts.desc()).limit(200).all())
    return [{"ts": r.ts, "channel": r.channel, "target": r.target,
             "status": r.status, "detail": r.detail} for r in rows]


# -- dashboard summary ----------------------------------------------------------------------

@app.get("/dashboard/summary")
def dashboard_summary(user: User = Depends(current_user), db=Depends(get_db)):
    devices = db.query(Device).filter_by(user_id=user.id).all()
    out = []
    week_ago = time.time() - 7 * 86400
    for dev in devices:
        bucket = cast(Reading.ts / 3600, Integer)   # hourly sparkline buckets
        window = (Reading.device_id == dev.id, Reading.ts >= week_ago)

        # T1.11. The sparkline used to average the RAW Mahalanobis score over
        # each hour. That is an hour-long mean of a quantity whose scale is set
        # by whichever regime each window landed in — a machine that idles for
        # 30 min and then runs loaded produces a bar that is the mean of two
        # different units. `display_index` is pinned to 70.0 at threshold in
        # EVERY regime, so its hourly mean is meaningful and "above 70" reads
        # the same on every device in the fleet.
        #
        # Chosen per device, not per bucket: a unit on pre-T1.7 firmware sends
        # no index and falls back to the raw score, but a series is never a
        # mixture of the two, which would be unreadable and dishonest.
        has_index = db.query(func.count(Reading.display_index)).filter(*window).scalar()
        field = Reading.display_index if has_index else Reading.score
        spark = (db.query(bucket.label("h"), func.avg(field))
                 .filter(*window).filter(field.isnot(None))
                 .group_by("h").order_by("h").all())
        n_anom = (db.query(func.count(AnomalyEvent.id))
                  .filter(AnomalyEvent.device_id == dev.id,
                          AnomalyEvent.ts >= week_ago).scalar())
        out.append({"device_id": dev.id, "name": dev.name, "online": dev.online,
                    "health": dev.health, "last_seen_ts": dev.last_seen_ts,
                    "anomalies_7d": int(n_anom or 0),
                    # Named so the UI never has to guess what it is drawing.
                    "sparkline_field": "display_index" if has_index else "score",
                    "sparkline": [float(s[1]) for s in spark if s[1] is not None]})
    return {"devices": out}
