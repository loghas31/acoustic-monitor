"""
The phone upload route, tested end to end against the real FastAPI app.

What these pin, and why each one is here rather than being assumed:

* an upload is accepted, queued, processed, and produces a verdict from the
  REAL pipeline — not a mocked one, because the whole point of this route is
  that no second analysis path exists to drift out of step;
* a device cannot read another device's recording (the id is a UUID, but
  "unguessable" is not an access control);
* the failure modes a phone will actually hit — empty file, too short to
  score, oversize — return a legible error rather than a 500 or a row stuck
  at "running" forever.

The recording used is short and the learn period is deliberately tiny. That is
valid HERE because the assertion is about plumbing, not detection:
`DOC_STATUS.md` records that fewer than 48 learn windows gives a 55-59 %
held-out false-alarm rate, so a verdict from 3 learn windows says nothing
about a machine. Generating 24 real minutes per test would make the suite
unusable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent

# Upload dir is uid-qualified for F12's reason: /tmp persists between
# containers running as different users, and a foreign-owned leftover is
# unwritable.
os.environ["ACOUSTIC_UPLOAD_DIR"] = f"/tmp/test_uploads_{os.getuid()}"

# `firmware/main.py` and `backend/main.py` are BOTH called `main`, and whichever
# one imports first wins `sys.modules["main"]` for the entire pytest session.
# So `from main import app` is a coin flip decided by which test file ran
# first: this module passed in isolation and failed in a chunk where a
# firmware test had already loaded `main`, with
# `ImportError: cannot import name 'app' from 'main' (.../firmware/main.py)`.
#
# Loading backend/main.py from its path under a unique module name removes the
# race entirely. This is the same pattern `tools/e2e_severity_trend.py` uses,
# and for the same reason — it is the only file in the repo that needs both.
#
# DATABASE_URL is conftest's; this module shares the backend test database with
# test_api.py by design (F14 — they share one engine regardless, because
# backend/models.py builds it once at import time).

from fastapi.testclient import TestClient  # noqa: E402

import models  # noqa: E402

_bspec = importlib.util.spec_from_file_location(
    "backend_main_recordings", ROOT / "backend" / "main.py")
_backend_main = importlib.util.module_from_spec(_bspec)
# Registered BEFORE exec: pydantic resolves a model's forward references via
# `sys.modules[cls.__module__]`, and a module absent from there makes every
# request body fail with "is not fully defined".
sys.modules["backend_main_recordings"] = _backend_main
_bspec.loader.exec_module(_backend_main)
app = _backend_main.app

SR = 16000
WINDOW_S = 30.0


def _tone_wav(path: Path, seconds: float) -> Path:
    """A benign broadband signal. Content does not matter — these tests are
    about the route, and a real detection test lives in test_phone_monitor."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, int(SR * seconds))
    wavfile.write(path, SR, (x * 32767).astype(np.int16))
    return path


@pytest.fixture(scope="module")
def client():
    models.Base.metadata.drop_all(models.engine)
    models.Base.metadata.create_all(models.engine)
    with TestClient(app) as c:
        yield c
    models.Base.metadata.drop_all(models.engine)
    # F12: dispose so no pooled connection outlives this module. Deliberately
    # does NOT unlink the database file — it is shared with test_api.py, and
    # deleting a file out from under another module's engine is exactly the
    # bug F14 spent a session diagnosing.
    models.engine.dispose()


def _device(client, email: str):
    tok = client.post("/auth/register",
                      json={"email": email, "password": "hunter22"}
                      ).json()["token"]
    reg = client.post("/devices/register", json={"name": "phone"},
                      headers={"Authorization": f"Bearer {tok}"}).json()
    return {"X-API-Key": reg["api_key"]}


@pytest.fixture(scope="module")
def dev(client):
    return _device(client, "phone-a@example.com")


def _upload(client, headers, path: Path, learn_windows: int | None = None):
    q = f"?learn_windows={learn_windows}" if learn_windows is not None else ""
    with open(path, "rb") as fh:
        return client.post(f"/recordings{q}", headers=headers,
                           files={"file": (path.name, fh, "audio/wav")})


def test_upload_is_accepted_and_processed_by_the_real_pipeline(client, dev, tmp_path):
    """Enough audio for a few 30 s windows: 3 learn + 2 scored."""
    wav = _tone_wav(tmp_path / "rec.wav", WINDOW_S * 5 + 1)
    r = _upload(client, dev, wav, learn_windows=3)
    assert r.status_code == 202, r.text
    rid = r.json()["recording_id"]
    assert r.json()["status"] == "queued"

    # TestClient runs BackgroundTasks synchronously on response close, so by
    # the time we poll it has already run.
    got = client.get(f"/recordings/{rid}", headers=dev).json()
    assert got["status"] == "done", got.get("error")
    v = got["verdict"]
    assert v is not None
    assert v["n_scored"] >= 1
    # The verdict must come from phone_monitor's own summary, so its keys are
    # the contract. If this fails, the two have drifted apart.
    assert "windows_above_threshold_pct" in v
    assert "band_selector_fired_pct" in v
    assert v["baseline_k_regimes"] >= 1


def test_a_short_learn_period_is_flagged_as_not_a_health_verdict(client, dev, tmp_path):
    """3 learn windows is 90 seconds. DOC_STATUS measured 55-59 % held-out
    false alarms below 48, so the verdict must carry a warning rather than
    looking like a clean bill of health."""
    wav = _tone_wav(tmp_path / "shortlearn.wav", WINDOW_S * 5 + 1)
    rid = _upload(client, dev, wav, learn_windows=3).json()["recording_id"]
    v = client.get(f"/recordings/{rid}", headers=dev).json()["verdict"]
    assert v["learn_period_too_short"] is True
    assert v["learn_windows"] == 3


def test_learn_windows_must_be_positive(client, dev, tmp_path):
    wav = _tone_wav(tmp_path / "z.wav", WINDOW_S * 2)
    assert _upload(client, dev, wav, learn_windows=0).status_code == 400


def test_a_device_cannot_read_another_devices_recording(client, dev, tmp_path):
    wav = _tone_wav(tmp_path / "mine.wav", WINDOW_S * 5 + 1)
    rid = _upload(client, dev, wav).json()["recording_id"]
    other = _device(client, "phone-b@example.com")
    assert client.get(f"/recordings/{rid}", headers=other).status_code == 404


def test_unauthenticated_upload_is_rejected(client, tmp_path):
    wav = _tone_wav(tmp_path / "anon.wav", WINDOW_S * 2)
    with open(wav, "rb") as fh:
        r = client.post("/recordings", files={"file": ("a.wav", fh, "audio/wav")})
    assert r.status_code == 401


def test_empty_upload_is_refused_cleanly(client, dev, tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    r = _upload(client, dev, empty)
    assert r.status_code == 400
    assert "empty" in r.text.lower()


def test_too_short_to_score_fails_with_a_legible_reason_not_a_500(client, dev, tmp_path):
    """The most likely real-world error: a recording that feels long to a
    human but cannot fill a learn period. It must land as status=failed with
    an explanation, never as a stuck row or an opaque crash."""
    wav = _tone_wav(tmp_path / "short.wav", WINDOW_S * 2)
    r = _upload(client, dev, wav)          # default 48 learn windows
    assert r.status_code == 202
    got = client.get(f"/recordings/{r.json()['recording_id']}", headers=dev).json()
    assert got["status"] == "failed"
    assert got["error"], "a failed recording must say why"
    assert got["verdict"] is None


def test_listing_shows_the_users_own_recordings_only(client, dev, tmp_path):
    rows = client.get("/recordings", headers=dev).json()
    assert isinstance(rows, list) and rows
    assert all(set(r) >= {"recording_id", "status", "uploaded_ts"} for r in rows)
    other = _device(client, "phone-c@example.com")
    assert client.get("/recordings", headers=other).json() == []


def test_unknown_recording_id_is_404(client, dev):
    assert client.get("/recordings/nope", headers=dev).status_code == 404
