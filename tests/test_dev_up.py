"""
tests/test_dev_up.py — backlog T2.3, "verify it actually serves in the
sandbox".

`scripts/dev_up.sh` brings up the backend without Docker: `python3 -m
uvicorn main:app` against a SQLite file, no Postgres, no Mosquitto. Every
other backend test in this suite (`test_api.py`, `test_frontend_backend_
integration.py`) talks to the app through FastAPI's in-process `TestClient`,
which never actually binds a socket or runs a real ASGI server — so none of
them would have caught a real uvicorn-specific problem. This file runs the
actual script as a subprocess, over real HTTP, and was how the first version
of the script was caught: it defaulted to a SQLite file under `backend/`,
which is on the mount this whole project already knows fails with
`sqlite3.OperationalError: disk I/O error` (see `test_api.py`'s own
`DATABASE_URL` comment) — found by running this file, not by reading the
script.
"""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "dev_up.sh"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("uvicorn") is None,
    reason="uvicorn not installed (it is a real backend/requirements.txt "
           "dependency; CI installs it, this environment has not)")


def _get(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _post(url: str, body: dict, headers: dict | None = None,
         timeout: float = 5.0) -> tuple[int, dict]:
    import json
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", **(headers or {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait_for_server(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            status, _ = _get(f"{base_url}/openapi.json")
            if status == 200:
                return
        except Exception as e:                                # noqa: BLE001
            last_err = e
        time.sleep(0.3)
    raise TimeoutError(f"{base_url} never came up ({last_err})")


def _start(db_path: Path, port: int) -> subprocess.Popen:
    env = {**os.environ,
          "DATABASE_URL": f"sqlite:///{db_path}",
          "DEV_UP_HOST": "127.0.0.1",
          "DEV_UP_PORT": str(port)}
    return subprocess.Popen(
        ["bash", str(SCRIPT)], env=env, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _stop(proc: subprocess.Popen) -> str:
    """SIGTERM, same as Ctrl+C, and collect the log for assertions."""
    proc.send_signal(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate(timeout=5)
    return out


def test_the_script_brings_up_a_real_server_end_to_end(tmp_path):
    """Not a smoke test against /openapi.json alone: register a user,
    register a device, POST a reading over HTTP (the ingest path a real
    device or firmware/main.py --simulate --no-mqtt would use), and read it
    back through the dashboard endpoint — the whole non-Docker path."""
    db_path = tmp_path / "dev_up_test.db"
    port = 8931
    base = f"http://127.0.0.1:{port}"
    proc = _start(db_path, port)
    try:
        _wait_for_server(base)

        status, body = _post(f"{base}/auth/register",
                             {"email": "devup@example.com", "password": "hunter22"})
        assert status == 201, body
        token = body["token"]
        hu = {"Authorization": f"Bearer {token}"}

        status, body = _post(f"{base}/devices/register", {"name": "Dev-up rig"}, hu)
        assert status == 201, body
        device_id, api_key = body["device_id"], body["api_key"]

        status, body = _post(f"{base}/readings", {
            "ts": time.time(), "score": 4.2, "threshold": 8.7,
            "regime": 0, "anomalous": False}, {"X-API-Key": api_key})
        assert status == 201, body

        import json
        req = urllib.request.Request(f"{base}/dashboard/summary", headers=hu)
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
            summary = json.loads(r.read())
        assert any(d["device_id"] == device_id for d in summary["devices"])
    finally:
        log = _stop(proc)
        assert "disk I/O error" not in log, log
        assert "Application shutdown complete" in log, log
    assert db_path.exists(), "the script never actually wrote to the SQLite file"


def test_fresh_flag_removes_an_existing_database_file(tmp_path):
    """--fresh acts on the script's OWN default path ($TMPDIR/acoustic-
    monitor-dev-<uid>.db), not an overridden DATABASE_URL — that is the one
    case a caller cannot just point DATABASE_URL somewhere clean instead,
    so it is the one worth exercising for real. Point $TMPDIR at a scratch
    directory so this test does not touch a real dev database.

    Filename is uid-qualified (see dev_up.sh's own comment, and F12 in
    the commit log (not in this public copy) for the shared-/tmp collision bug this avoids) —
    match that here rather than hardcoding the old bare filename."""
    port = 8932
    fresh_tmpdir = tmp_path / "fresh_tmpdir"
    fresh_tmpdir.mkdir()
    target = fresh_tmpdir / f"acoustic-monitor-dev-{os.getuid()}.db"
    target.write_bytes(b"stale, must be removed")
    env = {**os.environ, "TMPDIR": str(fresh_tmpdir),
          "DEV_UP_HOST": "127.0.0.1", "DEV_UP_PORT": str(port)}
    env.pop("DATABASE_URL", None)
    proc = subprocess.Popen(["bash", str(SCRIPT), "--fresh"], env=env,
                            cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for_server(f"http://127.0.0.1:{port}")
        assert target.exists(), "a fresh, real sqlite db should exist after startup"
        assert target.read_bytes() != b"stale, must be removed"
    finally:
        log = _stop(proc)
        assert "removing" in log.lower()


def test_missing_python3_fails_fast_and_says_so(tmp_path):
    """PATH with no python3 on it — the script must name the problem, not
    fail three layers down inside uvicorn with a confusing error.

    `bash` is invoked by its absolute path (not looked up on PATH) because
    the whole point of this test is to break PATH for the SCRIPT's own
    `command -v python3` check, and Popen's own executable lookup would
    break the same way if it depended on PATH too."""
    import shutil
    bash = shutil.which("bash") or "/usr/bin/bash"
    env = {**os.environ, "PATH": "/nonexistent-bin-only"}
    r = subprocess.run([bash, str(SCRIPT)], env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=15)
    assert r.returncode != 0
    assert "python3 not found" in (r.stdout + r.stderr)
