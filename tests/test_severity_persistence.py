"""
T1.11 — the reportable layer survives ingest and can be charted.

Context. Since T1.7 the firmware has published, every 30 s, a `display_index`
(0-100, pinned to 70.0 at the regime's own threshold), a state-based `tier`,
and four physical severity fields. `models.Reading` had columns for none of
them, so the trend the product is commercially *for* — "your machine is
getting worse" — could not be drawn. These tests pin the whole path:

    firmware payload -> ReadingIn -> handle_telemetry -> Reading row
                     -> /devices/{id}/readings and /dashboard/summary

and, separately, the schema migration that lets an ALREADY POPULATED database
gain the columns without losing its rows.

Uses its own SQLite file so the migration tests can build a deliberately
old-shaped table without disturbing the module-scoped database in
`test_api.py`.
"""

import importlib
import os
import time

import pytest
from sqlalchemy import create_engine, inspect, text

# The path is qualified by the current uid. SQLite databases have to live under
# /tmp (the repo mount lacks the POSIX locks SQLite needs), but /tmp is shared
# and NOT cleared between agent containers: a run on 2026-08-18 found this file
# left behind owned by `nobody`, mode 644, which made every test in this module
# error with "attempt to write a readonly database" (5 errors, suite 333 -> 328)
# even though nothing in the repo had changed. The uid suffix makes a foreign
# leftover harmless; the path is still fixed within a run, which the migration
# tests below rely on.
DB_PATH = f"/tmp/test_severity_persistence_{os.getuid()}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

import models  # noqa: E402
import mqtt_bridge  # noqa: E402

# test_api.py may already have imported models against ITS url. Both modules
# share one process, so rebind this module's engine explicitly rather than
# relying on import order — flaky ordering is how this kind of test rots.
_engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


@pytest.fixture()
def db(monkeypatch):
    """A session bound to this file's own database, with the real schema."""
    from sqlalchemy.orm import sessionmaker
    monkeypatch.setattr(models, "engine", _engine)
    Session = sessionmaker(bind=_engine, autoflush=False)
    monkeypatch.setattr(models, "SessionLocal", Session)
    monkeypatch.setattr(mqtt_bridge, "SessionLocal", Session, raising=False)
    models.Base.metadata.drop_all(_engine)
    models.Base.metadata.create_all(_engine)
    s = Session()
    user = models.User(email=f"t{time.time()}@x.io", password_hash="x")
    s.add(user)
    s.flush()
    dev = models.Device(user_id=user.id, name="Compressor A", api_key_hash="h")
    s.add(dev)
    s.commit()
    s.device_id = dev.id
    yield s
    s.close()


# The exact payload firmware/main.py publishes (T1.7 telemetry block).
FIRMWARE_PAYLOAD = {
    "ts": 1_700_000_000.0, "window": 7,
    "score": 12.5, "threshold": 8.35, "regime": 1, "anomalous": True,
    "display_index": 74.31, "tier": "amber", "score_percentile": 100.0,
    "severity_band_rms_db": -14.62, "severity_env_peak_hz": 152.5,
    "severity_env_peak_ratio": 31.08, "severity_env_db_re_learn": 9.44,
    "fr_hz": 49.8, "fr_reliable": True, "band": [3866.0, 5420.0],
    "mel_mean": [0.5] * 64, "latency_ms": 480.0,
}


# -- the columns exist at all --------------------------------------------------

def test_reading_has_every_field_the_firmware_publishes():
    """A regression guard on the actual bug T1.11 fixes: the payload had six
    keys with nowhere to go, and SQLAlchemy did not complain — it just lost
    them. If firmware adds a reported field, this fails until the column does."""
    cols = set(models.Reading.__table__.columns.keys())
    for key in ("tier", "display_index", "score_percentile",
                "severity_band_rms_db", "severity_env_peak_hz",
                "severity_env_peak_ratio", "severity_env_db_re_learn"):
        assert key in cols, f"models.Reading has no column for published field {key}"


def test_new_columns_are_all_nullable():
    """Pre-T1.7 firmware sends none of these. A NOT NULL column would reject
    the whole reading and take the machine off the dashboard."""
    for name in ("tier", "display_index", "severity_band_rms_db"):
        assert models.Reading.__table__.columns[name].nullable


# -- ingest --------------------------------------------------------------------

def test_handle_telemetry_persists_the_reportable_layer(db):
    mqtt_bridge.handle_telemetry(db, db.device_id, dict(FIRMWARE_PAYLOAD))
    r = db.query(models.Reading).one()
    assert r.display_index == pytest.approx(74.31)
    assert r.tier == "amber"
    assert r.score_percentile == pytest.approx(100.0)
    assert r.severity_band_rms_db == pytest.approx(-14.62)
    assert r.severity_env_peak_hz == pytest.approx(152.5)
    assert r.severity_env_peak_ratio == pytest.approx(31.08)
    assert r.severity_env_db_re_learn == pytest.approx(9.44)
    # and the pre-existing fields are untouched
    assert r.score == pytest.approx(12.5) and r.regime == 1 and r.anomalous


def test_old_firmware_payload_still_ingests(db):
    """A unit that never heard of T1.7 must still produce a row."""
    old = {k: v for k, v in FIRMWARE_PAYLOAD.items()
           if not k.startswith("severity_") and k not in
           ("display_index", "tier", "score_percentile")}
    mqtt_bridge.handle_telemetry(db, db.device_id, old)
    r = db.query(models.Reading).one()
    assert r.score == pytest.approx(12.5)
    assert r.display_index is None and r.tier is None
    assert r.severity_band_rms_db is None


def test_non_finite_severity_is_stored_as_null_not_dropped(db):
    """A silent window gives band_rms = 0 -> _db20 = -inf. JSON cannot carry
    it and a chart cannot plot it, but the WINDOW must not be lost: the row is
    written, that one field is NULL."""
    p = dict(FIRMWARE_PAYLOAD, severity_band_rms_db=float("-inf"),
             severity_env_peak_ratio=float("nan"))
    mqtt_bridge.handle_telemetry(db, db.device_id, p)
    r = db.query(models.Reading).one()
    assert r.severity_band_rms_db is None
    assert r.severity_env_peak_ratio is None
    assert r.display_index == pytest.approx(74.31)   # the rest survives


def test_garbage_tier_from_the_network_is_not_stored(db):
    """`tier` reaches the column straight off the wire. Only the three known
    values are stored, so the frontend's colour lookup cannot be driven by an
    arbitrary string from an unauthenticated-ish broker topic."""
    mqtt_bridge.handle_telemetry(db, db.device_id, dict(FIRMWARE_PAYLOAD, tier="<script>"))
    assert db.query(models.Reading).one().tier is None


def test_string_numbers_are_coerced(db):
    """JSON from a future firmware (or a hand-written test rig) may quote its
    numbers; a chart that receives "74.31" plots nothing and says nothing."""
    mqtt_bridge.handle_telemetry(db, db.device_id, dict(FIRMWARE_PAYLOAD, display_index="74.31"))
    assert db.query(models.Reading).one().display_index == pytest.approx(74.31)


# -- the migration -------------------------------------------------------------

def _old_shaped_db(path: str) -> None:
    """A `readings` table exactly as it was before T1.11, with a row in it."""
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as c:
        c.execute(text("""
            CREATE TABLE readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id VARCHAR NOT NULL, ts FLOAT NOT NULL,
                score FLOAT, threshold FLOAT, regime INTEGER,
                anomalous BOOLEAN, fr_hz FLOAT, fr_reliable BOOLEAN,
                band JSON, mel_mean JSON)"""))
        c.execute(text("INSERT INTO readings (device_id, ts, score) "
                       "VALUES ('d1', 100.0, 6.5)"))
    eng.dispose()


def test_create_all_alone_does_not_add_the_columns(tmp_path):
    """The reason `add_missing_columns` has to exist. Pinned so nobody deletes
    it believing create_all handles migrations — it does not, and the failure
    is silent."""
    path = str(tmp_path / "old.db")
    _old_shaped_db(path)
    eng = create_engine(f"sqlite:///{path}")
    models.Base.metadata.create_all(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("readings")}
    assert "display_index" not in cols
    eng.dispose()


def test_add_missing_columns_migrates_a_populated_db(tmp_path, monkeypatch):
    path = str(tmp_path / "old.db")
    _old_shaped_db(path)
    eng = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(models, "engine", eng)

    added = models.add_missing_columns()
    assert sorted(added) == sorted([
        "readings.tier", "readings.display_index", "readings.score_percentile",
        "readings.severity_band_rms_db", "readings.severity_env_peak_hz",
        "readings.severity_env_peak_ratio", "readings.severity_env_db_re_learn"])

    # the old row is still there and readable, with NULLs in the new columns
    with eng.begin() as c:
        row = c.execute(text("SELECT score, display_index, severity_env_peak_hz "
                             "FROM readings")).one()
    assert row[0] == pytest.approx(6.5) and row[1] is None and row[2] is None
    eng.dispose()


def test_add_missing_columns_is_idempotent(tmp_path, monkeypatch):
    path = str(tmp_path / "old.db")
    _old_shaped_db(path)
    eng = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(models, "engine", eng)
    assert models.add_missing_columns()          # first run does work
    assert models.add_missing_columns() == []    # second run is a no-op
    eng.dispose()


def test_migrated_db_accepts_a_full_reading(tmp_path, monkeypatch):
    """End of the migration story: after migrating, the new payload actually
    round-trips. A schema that ALTERs cleanly but rejects an INSERT is no use."""
    from sqlalchemy.orm import sessionmaker
    path = str(tmp_path / "old.db")
    _old_shaped_db(path)
    eng = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(models, "engine", eng)
    models.Base.metadata.create_all(eng)
    models.add_missing_columns()
    s = sessionmaker(bind=eng)()
    mqtt_bridge.handle_telemetry(s, "d1", dict(FIRMWARE_PAYLOAD))
    got = s.query(models.Reading).filter_by(device_id="d1").order_by(
        models.Reading.ts.desc()).first()
    assert got.severity_env_db_re_learn == pytest.approx(9.44)
    s.close()
    eng.dispose()


def test_migration_refuses_to_invent_a_not_null_column(tmp_path, monkeypatch):
    """Guard rail. Backfilling NOT NULL is a human decision; the helper must
    stop rather than guess a default."""
    from sqlalchemy import Column, Float
    path = str(tmp_path / "old.db")
    _old_shaped_db(path)
    eng = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(models, "engine", eng)
    table = models.Base.metadata.tables["readings"]
    col = Column("must_exist", Float, nullable=False)
    table.append_column(col)
    try:
        with pytest.raises(RuntimeError, match="NOT NULL"):
            models.add_missing_columns()
    finally:
        table._columns.remove(col)     # leave the ORM as we found it
        eng.dispose()
