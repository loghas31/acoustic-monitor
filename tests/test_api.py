"""
Backend API tests (v2) via FastAPI TestClient + SQLite. The HTTP ingest path
shares handler code with the MQTT bridge, so this covers both paths' logic.
The feedback endpoint test is the stage-5 'mutates baseline state' evidence:
verdict recorded, event acknowledged, device health de-escalated. (The MQTT
downlink to the device is best-effort by design and returns device_notified
=False here — no broker runs in this test environment.)
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# /tmp, not the repo dir: SQLite needs POSIX locks that network-mounted
# filesystems (CI sandboxes, NFS) often don't provide ("disk I/O error").
# The uid suffix stops a leftover file owned by a different user from making
# this whole module error with "attempt to write a readonly database" — see the
# long note in conftest.py, which this deliberately mirrors rather than
# overrides.
DB_PATH = f"/tmp/test_acoustic_{os.getuid()}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

import models  # noqa: E402

# NOT `from main import app`. `firmware/main.py` and `backend/main.py` are both
# called `main`, and whichever imports first owns `sys.modules["main"]` for the
# whole pytest session. This module got away with the bare import only because
# "test_api" sorts before every test file that loads firmware's `main` — a
# property of the alphabet, not of the code. `pytest tests/
# --ignore=tests/test_api.py` therefore failed COLLECTION of
# test_frontend_backend_integration.py (measured 2026-08-29, T3.1/F27), and a
# collection error means those tests silently do not run rather than reporting
# as failures.
#
# Loading backend/main.py from its path under a unique module name removes the
# ordering dependency outright. Same pattern as
# `tests/test_recordings_upload.py` and `tools/e2e_severity_trend.py`.
_bspec = importlib.util.spec_from_file_location(
    "backend_main_api", ROOT / "backend" / "main.py")
_backend_main = importlib.util.module_from_spec(_bspec)
# Registered BEFORE exec: pydantic resolves forward references via
# `sys.modules[cls.__module__]`, and a module missing from there makes every
# request body fail with "is not fully defined".
sys.modules["backend_main_api"] = _backend_main
_bspec.loader.exec_module(_backend_main)
app = _backend_main.app


@pytest.fixture(scope="module")
def client():
    models.Base.metadata.drop_all(models.engine)
    models.Base.metadata.create_all(models.engine)
    with TestClient(app) as c:
        yield c
    models.Base.metadata.drop_all(models.engine)
    # `dispose()` BEFORE the unlink, and it is not optional. `backend/models.py`
    # builds ONE module-level engine at import time, and Python caches modules,
    # so every backend test file in the suite shares this exact engine object no
    # matter what DATABASE_URL they set afterwards (a later module's
    # `os.environ[...] = ...` is a no-op once `models` is in sys.modules —
    # verified directly, not assumed). Removing the file while the pool still
    # holds an open handle leaves the NEXT module's fixture writing to an
    # unlinked inode, which SQLite reports as the thoroughly misleading
    # "attempt to write a readonly database". That was a real cross-file
    # failure: this module passed, and `test_frontend_backend_integration.py`
    # errored at setup, but only when the full suite ran — it passed alone.
    # Disposing returns the pool to a clean state so the next `create_all`
    # opens a fresh connection and recreates the file.
    models.engine.dispose()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


@pytest.fixture(scope="module")
def session(client):
    r = client.post("/auth/register", json={"email": "logan@example.com", "password": "hunter22"})
    assert r.status_code == 201
    hu = {"Authorization": f"Bearer {r.json()['token']}"}
    r = client.post("/devices/register", json={"name": "Compressor A"}, headers=hu)
    assert r.status_code == 201
    body = r.json()
    return hu, {"X-API-Key": body["api_key"]}, body["device_id"]


def test_login_roundtrip(client, session):
    assert client.post("/auth/login", json={"email": "logan@example.com",
                                            "password": "hunter22"}).status_code == 200
    assert client.post("/auth/login", json={"email": "logan@example.com",
                                            "password": "wrong"}).status_code == 401


def test_ingest_requires_api_key(client, session):
    assert client.post("/readings", json={"score": 1.0}).status_code == 401
    assert client.post("/readings", json={"score": 1.0},
                       headers={"X-API-Key": "forged"}).status_code == 401


def test_ingest_and_query_readings(client, session):
    hu, hd, dev_id = session
    now = time.time()
    for i in range(5):
        r = client.post("/readings", headers=hd, json={
            "ts": now + 30 * i, "score": 4.0 + 0.2 * i, "threshold": 8.7,
            "regime": i % 2, "anomalous": False, "fr_hz": 49.6, "fr_reliable": True,
            "band": [3000.0, 6000.0], "mel_mean": [0.5] * 64})
        assert r.status_code == 201
    r = client.get(f"/devices/{dev_id}/readings", headers=hu)
    assert r.status_code == 200 and len(r.json()) == 5
    body = client.get(f"/devices/{dev_id}/status", headers=hu).json()
    assert body["health"] == "green"
    assert body["latest"]["regime"] in (0, 1)
    assert body["latest"]["fr_reliable"] is True


def test_amber_comes_from_the_device_tier_not_the_score_magnitude(client, session):
    """CHANGED by backlog T1.7 / self-review F5, with the measurement that
    forced it.

    This test used to assert that score 7.0 against threshold 8.7 (0.80x) was
    amber, encoding the old rule "amber = 70-100 % of threshold". That rule
    was measured against the repo baseline over 200 fresh healthy windows and
    fires on **16.5 %** of them, versus only **12.5 %** of the windows of a
    fault ramped from severity 0.002 to 0.05 — a badge more likely on a
    healthy machine than on a failing one, because the healthy score
    distribution's own upper tail (median 0.580x, p95 0.762x, max 1.034x)
    lives inside the band.

    The assertion is not weakened, it is redirected: amber must now come from
    the device's state-based tier, and a bare 0.80x reading must NOT produce
    it. Both directions are asserted.
    """
    hu, hd, dev_id = session

    # old rule's case: 0.80x threshold, no tier field -> must be GREEN now
    r = client.post("/readings", headers=hd, json={
        "ts": time.time(), "score": 7.0, "threshold": 8.7, "regime": 0,
        "anomalous": False})
    assert r.status_code == 201
    assert client.get(f"/devices/{dev_id}/status", headers=hu).json()["health"] == "green"

    # device says amber (above threshold, persistence gate not yet satisfied)
    r = client.post("/readings", headers=hd, json={
        "ts": time.time(), "score": 18.6, "threshold": 8.7, "regime": 0,
        "anomalous": True, "tier": "amber"})
    assert r.status_code == 201
    assert client.get(f"/devices/{dev_id}/status", headers=hu).json()["health"] == "amber"

    # back below threshold
    r = client.post("/readings", headers=hd, json={
        "ts": time.time(), "score": 4.6, "threshold": 8.7, "regime": 0,
        "anomalous": False, "tier": "green"})
    assert r.status_code == 201
    assert client.get(f"/devices/{dev_id}/status", headers=hu).json()["health"] == "green"


def test_old_firmware_without_a_tier_still_colours_the_fleet(client, session):
    """Forward compatibility is not optional here: a deployed node may run
    pre-T1.7 firmware for months. Anomalous with no tier must still be red."""
    hu, hd, dev_id = session
    r = client.post("/readings", headers=hd, json={
        "ts": time.time(), "score": 60.0, "threshold": 8.7, "regime": 0,
        "anomalous": True})
    assert r.status_code == 201
    assert client.get(f"/devices/{dev_id}/status", headers=hu).json()["health"] == "red"


def test_an_unrecognised_tier_is_ignored_rather_than_stored(client, session):
    """The tier comes off the network. A device sending 'purple' (or an
    attacker sending anything) must not write an unrenderable colour into the
    fleet view."""
    hu, hd, dev_id = session
    r = client.post("/readings", headers=hd, json={
        "ts": time.time(), "score": 60.0, "threshold": 8.7, "regime": 0,
        "anomalous": True, "tier": "purple"})
    assert r.status_code == 201
    assert client.get(f"/devices/{dev_id}/status", headers=hu).json()["health"] == "red"


def test_anomaly_event_summary_and_feedback(client, session):
    hu, hd, dev_id = session
    now = time.time()
    r = client.post("/anomalies", headers=hd, json={
        "ts": now, "score": 60.0, "threshold": 8.7, "regime": 0,
        "ts_from": now - 240, "ts_to": now, "persisted_minutes": 2.0})
    assert r.status_code == 201

    events = client.get(f"/devices/{dev_id}/anomalies", headers=hu,
                        params={"since": 0}).json()
    assert len(events) == 1 and events[0]["feedback"] == ""
    assert client.get("/dashboard/summary", headers=hu).json()["devices"][0]["health"] == "red"

    # --- the feedback loop: stage-5 gate -------------------------------------
    ev_id = events[0]["id"]
    r = client.post(f"/anomalies/{ev_id}/feedback", headers=hu,
                    json={"verdict": "normal"})
    assert r.status_code == 200
    assert r.json()["verdict"] == "normal"

    events = client.get(f"/devices/{dev_id}/anomalies", headers=hu).json()
    assert events[0]["feedback"] == "normal" and events[0]["acknowledged"] is True
    # health de-escalated after the human verdict
    assert client.get("/dashboard/summary", headers=hu).json()["devices"][0]["health"] == "amber"

    # invalid verdict rejected
    assert client.post(f"/anomalies/{ev_id}/feedback", headers=hu,
                       json={"verdict": "maybe"}).status_code == 422


# -- T1.11: the reportable layer reaches the dashboard ---------------------------

def _fresh_device(client, hu, name):
    """A device of its own, so a sparkline assertion is not polluted by the
    readings other tests posted to the shared fixture device."""
    r = client.post("/devices/register", json={"name": name}, headers=hu)
    assert r.status_code == 201
    b = r.json()
    return {"X-API-Key": b["api_key"]}, b["device_id"]


def test_readings_endpoint_returns_the_severity_trend(client, session):
    """T1.11. The device page cannot plot what the API does not return."""
    hu, _, _ = session
    hd, dev_id = _fresh_device(client, hu, "Trend rig")
    now = time.time()
    for i in range(4):
        assert client.post("/readings", headers=hd, json={
            "ts": now + 30 * i, "score": 5.0 + 3.0 * i, "threshold": 8.7,
            "regime": 0, "anomalous": i >= 3, "tier": "red" if i >= 3 else "green",
            "display_index": 55.0 + 6.0 * i,
            "severity_band_rms_db": -23.8 + 4.0 * i,
            "severity_env_peak_hz": 152.5,
            "severity_env_peak_ratio": 20.0 + 10.0 * i,
            "severity_env_db_re_learn": 0.0 + 5.0 * i}).status_code == 201

    rows = client.get(f"/devices/{dev_id}/readings", headers=hu).json()
    assert len(rows) == 4
    assert [r["display_index"] for r in rows] == [55.0, 61.0, 67.0, 73.0]
    # the physical trend is monotone and, unlike the score, in real units
    assert [r["severity_band_rms_db"] for r in rows] == [-23.8, -19.8, -15.8, -11.8]
    assert all(r["severity_env_peak_hz"] == 152.5 for r in rows)
    assert rows[-1]["tier"] == "red" and rows[0]["tier"] == "green"

    latest = client.get(f"/devices/{dev_id}/status", headers=hu).json()["latest"]
    assert latest["display_index"] == 73.0
    assert latest["severity_env_db_re_learn"] == 15.0


def test_sparkline_switches_to_the_display_index(client, session):
    """The fleet sparkline used to average the RAW score hourly, mixing
    regimes whose thresholds differ. It now averages `display_index`, which is
    70.0 at threshold in every regime, and says which field it drew."""
    hu, _, _ = session
    hd, dev_id = _fresh_device(client, hu, "Sparkline rig")
    now = time.time()
    for i in range(3):
        client.post("/readings", headers=hd, json={
            "ts": now + i, "score": 4.0 + i, "threshold": 8.7, "regime": i,
            "anomalous": False, "display_index": 60.0 + 3.0 * i})

    dev = next(d for d in client.get("/dashboard/summary", headers=hu).json()["devices"]
               if d["device_id"] == dev_id)
    assert dev["sparkline_field"] == "display_index"
    # one hourly bucket, mean of 60/63/66 — NOT the mean of the raw 4/5/6
    assert dev["sparkline"] == pytest.approx([63.0])


def test_sparkline_falls_back_to_score_for_old_firmware(client, session):
    """A node on pre-T1.7 firmware must still draw a line, and the response
    must say the line is a raw score so nobody reads 5.0 as "well below 70"."""
    hu, _, _ = session
    hd, dev_id = _fresh_device(client, hu, "Legacy rig")
    now = time.time()
    for i in range(3):
        client.post("/readings", headers=hd, json={
            "ts": now + i, "score": 4.0 + i, "threshold": 8.7, "regime": 0,
            "anomalous": False})

    dev = next(d for d in client.get("/dashboard/summary", headers=hu).json()["devices"]
               if d["device_id"] == dev_id)
    assert dev["sparkline_field"] == "score"
    assert dev["sparkline"] == pytest.approx([5.0])


def test_a_sparkline_is_never_a_mixture_of_both_scales(client, session):
    """The failure mode the per-device choice exists to prevent: a unit
    upgraded mid-week would otherwise plot 5.2 next to 63.0 on one axis."""
    hu, _, _ = session
    hd, dev_id = _fresh_device(client, hu, "Upgraded mid-week rig")
    now = time.time()
    client.post("/readings", headers=hd, json={           # before the upgrade
        "ts": now - 7200, "score": 5.2, "threshold": 8.7, "regime": 0})
    client.post("/readings", headers=hd, json={           # after
        "ts": now, "score": 5.4, "threshold": 8.7, "regime": 0,
        "display_index": 63.0})

    dev = next(d for d in client.get("/dashboard/summary", headers=hu).json()["devices"]
               if d["device_id"] == dev_id)
    assert dev["sparkline_field"] == "display_index"
    assert dev["sparkline"] == pytest.approx([63.0])      # the raw 5.2 is not drawn


def test_alert_configure(client, session):
    hu, hd, dev_id = session
    assert client.post("/alerts/configure", params={"device_id": dev_id}, headers=hu,
                       json={"email": "ops@example.com",
                             "webhook_url": "http://10.0.0.5/hook"}).status_code == 200


def test_cross_user_isolation(client, session):
    hu, hd, dev_id = session
    r = client.post("/auth/register", json={"email": "mallory@example.com",
                                            "password": "pw123456"})
    other = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.get(f"/devices/{dev_id}/status", headers=other).status_code == 404
    events = client.get(f"/devices/{dev_id}/anomalies", headers=hu).json()
    assert client.post(f"/anomalies/{events[0]['id']}/feedback", headers=other,
                       json={"verdict": "normal"}).status_code == 404
