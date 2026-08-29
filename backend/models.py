"""
ORM models (v2). PostgreSQL in production, SQLite in tests — DATABASE_URL
decides; nothing here is Postgres-specific on purpose.

Schema changes from v1 (mirrors the geometry-free Mahalanobis firmware):
* readings carry (score, threshold, regime) — not autoencoder fields.
* anomaly_events carry the episode window [ts_from, ts_to] so the feedback
  loop can tell the device exactly WHICH windows to fold back into its
  baseline, and a `feedback` verdict ("", "normal", "fault").
* api keys stored as SHA-256 hashes: a leaked DB must not leak credentials.
"""

from __future__ import annotations

import os
import time
import uuid

from sqlalchemy import (JSON, Boolean, Column, Float, ForeignKey, Integer,
                        String, create_engine, inspect, text, Index)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./acoustic_dev.db")

engine = create_engine(DATABASE_URL,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_ts = Column(Float, default=time.time)


class Device(Base):
    __tablename__ = "devices"
    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    api_key_hash = Column(String, nullable=False, index=True)
    created_ts = Column(Float, default=time.time)
    last_seen_ts = Column(Float, default=0.0)
    online = Column(Boolean, default=False)
    # green/amber/red precomputed at ingest — the fleet page must be one scan.
    health = Column(String, default="unknown")


class Reading(Base):
    __tablename__ = "readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String, ForeignKey("devices.id"), nullable=False)
    ts = Column(Float, nullable=False)
    score = Column(Float)              # Mahalanobis distance to nearest regime
    threshold = Column(Float)          # that regime's CV-calibrated threshold
    regime = Column(Integer)
    anomalous = Column(Boolean, default=False)
    fr_hz = Column(Float)
    fr_reliable = Column(Boolean, default=False)
    band = Column(JSON)                # demodulation band used [lo, hi]
    mel_mean = Column(JSON)            # 64-bin spectrogram sketch (heatmap)

    # -- T1.11: the reportable layer the firmware has published since T1.7 ----
    # All nullable, because a unit on pre-T1.7 firmware sends none of them and
    # must keep working. Anything that reads these must tolerate None.
    #
    # WHY these and not the raw score. `score` is a Mahalanobis distance whose
    # scale is set by each regime's own covariance, so 9.1 in regime 0 and 9.1
    # in regime 1 do not mean the same thing, and averaging them (which the
    # sparkline does, hourly) mixes incomparable units. `display_index` is
    # log-linear in score/threshold and pinned to exactly 70.0 at threshold in
    # every regime, so it is the only one of the two that can honestly be
    # averaged across regimes or compared between machines.
    tier = Column(String)              # green|amber|red, computed on-device from STATE
    display_index = Column(Float)      # 0-100, 70.0 == that regime's threshold
    score_percentile = Column(Float)   # chi2-fit percentile; SATURATES near 100 (T1.7)
    # Physical severity — the trend a customer is actually shown. Unlike the
    # score these are in physical units and are comparable over time even if
    # the baseline is retrained, which is exactly what makes them trendable.
    severity_band_rms_db = Column(Float)      # RMS in the demodulation band, dB
    severity_env_peak_hz = Column(Float)      # detected repetition rate, Hz
    # LINEAR ratio (peak / median background), NOT dB — measured 3.6 to 582
    # over one simulated fault ramp, so anything plotting it needs a log axis.
    # It is exactly gain-invariant (23.19 at gains 1/2/4/8, measured
    # 2026-08-18), which is what makes it complementary to band_rms_db: level
    # and contrast move independently and neither alone identifies a fault.
    severity_env_peak_ratio = Column(Float)
    severity_env_db_re_learn = Column(Float)  # envelope energy re the learn period, dB
    __table_args__ = (Index("ix_readings_device_ts", "device_id", "ts"),)


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    id = Column(String, primary_key=True, default=_uuid)
    device_id = Column(String, ForeignKey("devices.id"), nullable=False, index=True)
    ts = Column(Float, nullable=False)
    score = Column(Float)
    threshold = Column(Float)
    regime = Column(Integer)
    ts_from = Column(Float)            # episode window — what feedback retrains on
    ts_to = Column(Float)
    persisted_minutes = Column(Float)
    feedback = Column(String, default="")    # "" | "normal" | "fault"
    acknowledged = Column(Boolean, default=False)


class Recording(Base):
    """An audio file uploaded from a phone, and the verdict derived from it.

    WHY THIS EXISTS, AND WHY IT IS NOT THE PRODUCTION SHAPE. The Pi extracts
    features on-device and POSTs ~400 bytes to `/readings`; it never uploads
    audio. That is deliberate — bandwidth, and not shipping recordings of a
    customer's factory floor off-site.

    A phone cannot do that without the whole NumPy/SciPy pipeline ported to
    it, which is weeks of work for a device that is not the product. So the
    phone route trades the production design for a prototype one: upload the
    audio, let the server run the pipeline that already exists, unchanged.

    Consequence to keep in view: **this path uploads raw audio.** Fine for
    your own fridge. Before pointing it at a customer's machine, that is a
    consent and data-retention question, not just a technical one.
    """
    __tablename__ = "recordings"
    id = Column(String, primary_key=True, default=_uuid)
    device_id = Column(String, ForeignKey("devices.id"), nullable=False, index=True)
    uploaded_ts = Column(Float, nullable=False)
    filename = Column(String)           # as sent, for the human's benefit
    path = Column(String)               # where the bytes landed on disk
    bytes = Column(Integer)
    # queued -> running -> done | failed. Polled by the phone after upload,
    # because a 28-minute recording takes ~60 s to analyse and a mobile
    # connection will not hold a request open that long.
    status = Column(String, default="queued", index=True)
    verdict = Column(JSON)              # the phone_monitor summary, or None
    error = Column(String, default="")  # populated only when status == "failed"


class AlertConfig(Base):
    __tablename__ = "alert_configs"
    device_id = Column(String, ForeignKey("devices.id"), primary_key=True)
    email = Column(String)
    webhook_url = Column(String)
    enabled = Column(Boolean, default=True)


class Alert(Base):
    """Dispatch log — every notification attempted and whether it landed.
    'You never warned me' is an auditable claim; this table is the audit."""
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, default=_uuid)
    event_id = Column(String, ForeignKey("anomaly_events.id"), nullable=False)
    channel = Column(String, nullable=False)          # email | webhook
    target = Column(String)
    status = Column(String, default="pending")        # pending | sent | failed
    ts = Column(Float, default=time.time)
    detail = Column(String)


def add_missing_columns(tables: tuple[str, ...] = ("readings",)) -> list[str]:
    """Poor-man's migration: `ALTER TABLE ... ADD COLUMN` for any column that
    exists in the ORM model but not in the live database.

    `Base.metadata.create_all` creates missing *tables* and silently ignores
    missing *columns*, so without this the T1.11 severity columns would exist
    on a fresh test database and be absent on Logan's dev database and on any
    node already running — the reading would insert fine and the value would
    vanish. That is precisely the class of bug T1.11 exists to fix, so it is
    fixed properly rather than by "delete the dev database".

    Deliberately narrow: it only ever ADDs nullable columns. It never drops,
    renames or retypes anything, so it cannot lose data, and it is idempotent.
    Anything beyond that needs Alembic and a human. Returns the columns added,
    so callers (and tests) can see what happened.

    `ADD COLUMN <name> <type> NULL` with FLOAT/VARCHAR/BOOLEAN/INTEGER is
    accepted by both SQLite and PostgreSQL, which is why the schema above
    stays deliberately vanilla.
    """
    added: list[str] = []
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for tname in tables:
            if tname not in existing_tables:
                continue                       # create_all will make it whole
            table = Base.metadata.tables[tname]
            have = {c["name"] for c in insp.get_columns(tname)}
            for col in table.columns:
                if col.name in have:
                    continue
                if not col.nullable:
                    # Adding a NOT NULL column to a populated table needs a
                    # default and a backfill decision — a human's call.
                    raise RuntimeError(
                        f"refusing to auto-add NOT NULL column {tname}.{col.name}")
                ddl = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE {tname} ADD COLUMN {col.name} {ddl}'))
                added.append(f"{tname}.{col.name}")
    return added


def init_db() -> None:
    Base.metadata.create_all(engine)
    for col in add_missing_columns():
        print(f"[models] migrated: added {col}")
