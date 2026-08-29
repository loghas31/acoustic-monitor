"""
tests/test_frontend_backend_integration.py — backlog T3.6, frontend <-> backend
live integration check.

WHAT THIS DOES AND WHY
----------------------------------------------------------------------------
`tests/test_api.py` exercises the backend against a real FastAPI TestClient
and asserts the JSON shape it returns. `frontend/src/api/mock.js` exists so
`npm run dev` demos with zero backend, and its own header comment makes an
explicit promise: "Shapes mirror the v2 FastAPI responses exactly —
swap-in/swap-out guarantee." Nothing before this file checked that promise.
The two files could drift silently in either direction — the backend adds
a field a component now reads and the mock never gets it (breaks the
zero-backend demo), or the mock has a field the real backend no longer
sends (breaks the moment a real device is registered) — and neither
test_api.py (backend-only) nor a browser (none available in this sandbox;
see docs/DOC_STATUS.md's "Frontend never exercised against a live backend
in a browser") would catch it.

SCOPE, stated plainly: this is a JSON-contract check, not a rendered-pixel
check. `firmware/main.py --simulate` has no analogue on the frontend side —
there is no headless browser here — so "exercises the real backend the way
the frontend does" is interpreted as: seed the real backend with realistic
data via the real API (as a real device/dashboard would), capture its JSON,
and diff it field-by-field against (a) `mock.js`'s own shapes and (b) the
exact field names this repo's own React pages (`Overview.jsx`,
`DeviceDetail.jsx`, `AlertConfig.jsx`, `Onboarding.jsx`, `lib/trend.js`)
read off the response — grepped directly out of those files, not guessed.
The actual rendered-pixels gap remains open and undisputed in DOC_STATUS.

This file (Python side) starts the real backend via TestClient, seeds three
devices covering the three shapes the mock also covers (healthy/green,
faulty/red-with-an-acknowledged-alert, and legacy pre-T1.7 firmware with no
tier/severity fields), captures every endpoint the frontend calls, and hands
the JSON to `frontend/src/api/contract.test.mjs` (node, no browser needed —
same pattern as `frontend/src/lib/trend.test.mjs` from T1.11) to compare
against `mockApi`'s own output and the components' actual field reads.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NODE_SCRIPT = ROOT / "frontend" / "src" / "api" / "contract.test.mjs"

# This does NOT isolate this module from test_api.py, and an earlier version of
# this comment claimed it did. `backend/models.py` builds its engine at import
# time from DATABASE_URL, and Python caches modules — so if any other backend
# test file imported `models` first (test_api.py does, alphabetically), the
# engine is already bound to ITS path and the assignment below is a no-op.
# Verified by probe: after two modules set two different DATABASE_URLs,
# `models.engine.url` is whichever one imported first, and both modules hold
# the same `models` object.
#
# The line is kept because it IS correct when this module runs alone (which is
# how it is usually debugged), and harmless otherwise. What made it dangerous
# was believing it, and then deleting a database file out from under a shared
# connection pool — see the dispose() note in test_api.py's fixture.
DB_PATH = f"/tmp/test_frontend_integration_{os.getuid()}.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{DB_PATH}")

from fastapi.testclient import TestClient  # noqa: E402

import models  # noqa: E402

# NOT `from main import app` — see the long note in tests/test_api.py. Both
# `firmware/main.py` and `backend/main.py` are importable as `main`, and this
# module is the one that actually broke: it failed collection with
# `ImportError: cannot import name 'app' from 'main' (.../firmware/main.py)`
# whenever any firmware test loaded `main` first (T3.1/F27, 2026-08-29).
_bspec = importlib.util.spec_from_file_location(
    "backend_main_frontend_integration", ROOT / "backend" / "main.py")
_backend_main = importlib.util.module_from_spec(_bspec)
sys.modules["backend_main_frontend_integration"] = _backend_main
_bspec.loader.exec_module(_backend_main)
app = _backend_main.app

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture(scope="module")
def client():
    models.Base.metadata.drop_all(models.engine)
    models.Base.metadata.create_all(models.engine)
    with TestClient(app) as c:
        yield c
    models.Base.metadata.drop_all(models.engine)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


def _register_device(client, hu, name):
    r = client.post("/devices/register", json={"name": name}, headers=hu)
    assert r.status_code == 201, r.text
    body = r.json()
    return body, {"X-API-Key": body["api_key"]}, body["device_id"]


def test_the_real_backend_matches_the_mocks_own_contract(client, tmp_path):
    hu = None
    r = client.post("/auth/register", json={"email": "integration@example.com",
                                            "password": "hunter22222"})
    assert r.status_code == 201, r.text
    hu = {"Authorization": f"Bearer {r.json()['token']}"}
    now = time.time()

    # -- device 1: healthy/green, current firmware, two regimes -------------
    reg_healthy, hd_healthy, id_healthy = _register_device(client, hu, "Compressor A")
    for i in range(6):
        assert client.post("/readings", headers=hd_healthy, json={
            "ts": now + 30 * i, "score": 4.5 + 0.3 * i, "threshold": 8.7,
            "regime": i % 2, "anomalous": False, "tier": "green",
            "display_index": 55.0 + 1.5 * i,
            "severity_band_rms_db": -23.8 + 0.4 * i,
            "severity_env_peak_hz": 152.5,
            "severity_env_peak_ratio": 19.8 + 0.5 * i,
            "severity_env_db_re_learn": 0.2 * i,
            "fr_hz": 49.6, "fr_reliable": True,
            "band": [3866.0, 5420.0], "mel_mean": [0.4] * 64,
        }).status_code == 201

    # -- device 2: faulty/red, an acknowledged anomaly + a dispatched alert -
    reg_faulty, hd_faulty, id_faulty = _register_device(client, hu, "Injection moulder 3")
    for i in range(4):
        assert client.post("/readings", headers=hd_faulty, json={
            "ts": now + 30 * i, "score": 20.0 + 5.0 * i, "threshold": 8.7,
            "regime": 0, "anomalous": i >= 2, "tier": "red" if i >= 2 else "amber",
            "display_index": 70.0 + 4.0 * i,
            "severity_band_rms_db": -10.0 + 2.0 * i, "severity_env_peak_hz": 155.0,
            "severity_env_peak_ratio": 40.0 + 5.0 * i, "severity_env_db_re_learn": 10.0 + i,
            "fr_hz": 50.1, "fr_reliable": True,
            "band": [3866.0, 5420.0], "mel_mean": [0.9] * 64,
        }).status_code == 201
    # configure alert channels BEFORE the anomaly so dispatch() fires and a
    # real Alert row exists for /alerts/log/{id} — a fast-failing webhook URL
    # (nothing listens on 127.0.0.1:1) so this does not hang on a network call.
    assert client.post("/alerts/configure", params={"device_id": id_faulty}, headers=hu,
                       json={"email": "ops@example.com",
                             "webhook_url": "http://127.0.0.1:1/hook"}).status_code == 200
    r = client.post("/anomalies", headers=hd_faulty, json={
        "ts": now, "score": 40.0, "threshold": 8.7, "regime": 0,
        "ts_from": now - 240, "ts_to": now, "persisted_minutes": 2.0})
    assert r.status_code == 201
    events = client.get(f"/devices/{id_faulty}/anomalies", headers=hu).json()
    assert len(events) == 1
    event_id = events[0]["id"]
    feedback_resp = client.post(f"/anomalies/{event_id}/feedback", headers=hu,
                                json={"verdict": "normal"})
    assert feedback_resp.status_code == 200

    # -- device 3: legacy pre-T1.7 firmware, no tier/severity/display_index -
    reg_legacy, hd_legacy, id_legacy = _register_device(client, hu, "Conveyor B (packing)")
    for i in range(3):
        assert client.post("/readings", headers=hd_legacy, json={
            "ts": now + 30 * i, "score": 4.0 + i, "threshold": 8.7,
            "regime": 0, "anomalous": False,
            "fr_hz": 49.6, "fr_reliable": True,
            "band": [3866.0, 5420.0], "mel_mean": [0.3] * 64,
        }).status_code == 201

    # -- capture every endpoint the frontend calls ---------------------------
    real = {
        "summary": client.get("/dashboard/summary", headers=hu).json(),
        "status": {
            "healthy": client.get(f"/devices/{id_healthy}/status", headers=hu).json(),
            "faulty": client.get(f"/devices/{id_faulty}/status", headers=hu).json(),
            "legacy": client.get(f"/devices/{id_legacy}/status", headers=hu).json(),
        },
        "readings": {
            "healthy": client.get(f"/devices/{id_healthy}/readings", headers=hu).json(),
            "faulty": client.get(f"/devices/{id_faulty}/readings", headers=hu).json(),
            "legacy": client.get(f"/devices/{id_legacy}/readings", headers=hu).json(),
        },
        "anomalies": {
            "faulty": client.get(f"/devices/{id_faulty}/anomalies", headers=hu).json(),
            "healthy": client.get(f"/devices/{id_healthy}/anomalies", headers=hu).json(),
        },
        "alertLog": client.get(f"/alerts/log/{id_faulty}", headers=hu).json(),
        "feedback": feedback_resp.json(),
        "registerDevice": reg_healthy,
        "configureAlerts": {"ok": True},   # the endpoint's own literal response
    }

    # sanity on the fixture itself, before handing off to node: a dispatched
    # alert should really be there, or the alertLog contract check is vacuous
    assert len(real["alertLog"]) >= 1, "alerts.dispatch() did not record anything to check"
    assert real["status"]["legacy"]["latest"]["tier"] is None, \
        "legacy device fixture leaked a tier — the contract check needs a real None case"

    dump_path = tmp_path / "real_backend.json"
    dump_path.write_text(json.dumps(real))

    assert NODE_SCRIPT.exists(), f"missing {NODE_SCRIPT}"
    r = subprocess.run(["node", str(NODE_SCRIPT), str(dump_path)],
                       capture_output=True, text=True, timeout=60)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
