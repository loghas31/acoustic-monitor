#!/usr/bin/env python3
"""
End-to-end evidence for backlog T1.11: simulated microphone -> firmware ->
telemetry payload -> HTTP ingest -> database -> the exact JSON the dashboard
charts draw from.

This exists because unit tests can only prove that a field the *test* invents
survives ingest. What T1.11 actually claims is that the numbers the FIRMWARE
computes reach the chart, so this runs `firmware.main.run` unmodified with a
capturing uplink, then replays every captured window through the real FastAPI
app, then reads the API back and checks the severity trend is (a) present and
(b) rising as the simulated bearing fault grows.

    python tools/e2e_severity_trend.py

Prints a table and exits non-zero if the chain is broken anywhere.
Everything it reports is SYNTHETIC — the source is ml/simulate.py, not a
machine. It proves plumbing and monotonicity, not detection performance.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
# `firmware/main.py` and `backend/main.py` are both called `main`, and this
# script is the only place in the repo that needs both in one process, so
# NEITHER is imported by name — both are loaded from their file path below
# under unambiguous module names. Their sibling imports (models, features, …)
# still resolve through these path entries.
for sub in ("firmware", "ml", "backend"):
    sys.path.append(str(ROOT / sub))

# SQLite needs POSIX locks the repo mount does not provide, and the backend
# reads DATABASE_URL at import time — so this must happen before any import.
DB = Path(tempfile.gettempdir()) / "e2e_severity_trend.db"
if DB.exists():
    DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"


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
        self.telemetry.append(p)

    def publish_anomaly(self, p):
        self.anomalies.append(p)


def run_firmware(minutes: float, persist_minutes: float) -> CapturingUplink:
    import importlib.util

    import yaml

    spec = importlib.util.spec_from_file_location(
        "firmware_main", ROOT / "firmware" / "main.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)

    cap = CapturingUplink()
    fw.NullUplink = lambda *a, **k: cap    # noqa: E731 — deliberate injection
    cfg = yaml.safe_load((ROOT / "firmware" / "config.yaml").read_text())
    state_db = Path(tempfile.gettempdir()) / "e2e_state.db"
    if state_db.exists():
        state_db.unlink()
    args = SimpleNamespace(
        baseline=ROOT / "firmware" / "baseline.npz", simulate=True, no_mqtt=True,
        fast=True, minutes=minutes, persist_minutes=persist_minutes,
        fault_at_minute=6.0, transient_at_minute=2.0, db=state_db,
        config=ROOT / "firmware" / "config.yaml")
    alerts = fw.run(cfg, args)
    print(f"firmware: {len(cap.telemetry)} telemetry windows, "
          f"{len(cap.anomalies)} anomaly event(s), {alerts} alert(s)")
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--persist-minutes", type=float, default=2.0)
    args = ap.parse_args()

    cap = run_firmware(args.minutes, args.persist_minutes)
    if not cap.telemetry:
        print("FAIL: firmware published nothing")
        return 1

    published = set(cap.telemetry[0])
    wanted = {"display_index", "tier", "severity_band_rms_db",
              "severity_env_peak_hz", "severity_env_peak_ratio",
              "severity_env_db_re_learn"}
    missing = wanted - published
    if missing:
        print(f"FAIL: firmware does not publish {sorted(missing)}")
        return 1

    # -- the cloud half, through the real app --------------------------------
    import importlib.util

    from fastapi.testclient import TestClient

    import models
    spec = importlib.util.spec_from_file_location(
        "backend_main", ROOT / "backend" / "main.py")
    backend_main = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: pydantic resolves a model's forward references
    # through `sys.modules[cls.__module__]`, and a module that is not in there
    # yet makes every request body fail with "is not fully defined".
    sys.modules["backend_main"] = backend_main
    spec.loader.exec_module(backend_main)
    app = backend_main.app

    models.Base.metadata.drop_all(models.engine)
    models.Base.metadata.create_all(models.engine)
    with TestClient(app) as c:
        tok = c.post("/auth/register",
                     json={"email": "e2e@example.com", "password": "hunter22"}
                     ).json()["token"]
        hu = {"Authorization": f"Bearer {tok}"}
        reg = c.post("/devices/register", json={"name": "E2E rig"}, headers=hu).json()
        hd = {"X-API-Key": reg["api_key"]}
        dev_id = reg["device_id"]

        for p in cap.telemetry:
            r = c.post("/readings", headers=hd, json=p)
            if r.status_code != 201:
                print(f"FAIL: ingest rejected a real firmware payload: "
                      f"{r.status_code} {r.text[:300]}")
                return 1

        rows = c.get(f"/devices/{dev_id}/readings", headers=hu).json()
        summary = next(d for d in c.get("/dashboard/summary", headers=hu).json()["devices"]
                       if d["device_id"] == dev_id)

    stored = sum(1 for r in rows if r.get("display_index") is not None)
    print(f"api: {len(rows)} readings returned, {stored} carry a display_index")
    if stored != len(cap.telemetry):
        print("FAIL: the reportable layer did not survive ingest")
        return 1

    print(f"\n{'w':>3} {'regime':>6} {'score':>8} {'index':>6} {'tier':>5} "
          f"{'bandRMS dB':>10} {'envPk x':>9} {'env re learn':>12} {'peak Hz':>8}")
    for i, r in enumerate(rows):
        if i % 4 and i != len(rows) - 1:
            continue                                  # every 4th window, plus the last
        print(f"{i:>3} {r['regime']:>6} {r['score']:>8.2f} {r['display_index']:>6.1f} "
              f"{str(r['tier']):>5} {r['severity_band_rms_db']:>10.2f} "
              f"{r['severity_env_peak_ratio']:>9.2f} "
              f"{r['severity_env_db_re_learn']:>12.2f} "
              f"{r['severity_env_peak_hz']:>8.1f}")

    # -- does the trend actually trend? --------------------------------------
    # The demo schedule is healthy until minute 6 then ramps the fault, so the
    # late mean must exceed the early mean. Means, not endpoints: one window is
    # a sample of a noisy process and proving a trend from two of them is the
    # mistake self-review F5 made.
    n = len(rows)
    early = [r for r in rows[: n // 4]]
    late = [r for r in rows[-n // 4:]]

    ok = True
    # NB `severity_env_peak_ratio` is a LINEAR ratio, not dB — see
    # firmware/reporting.physical_severity. Mislabelling it as dB would make a
    # 580x contrast read as a modest number.
    for field, unit in (("severity_band_rms_db", "dB"),
                        ("severity_env_peak_ratio", "x"),
                        ("severity_env_db_re_learn", "dB"),
                        ("display_index", "")):
        e = sum(r[field] for r in early) / len(early)
        l = sum(r[field] for r in late) / len(late)
        verdict = "rises" if l > e else "DOES NOT RISE"
        if l <= e:
            ok = False
        print(f"{field:>26}: healthy mean {e:8.2f} -> fault mean {l:8.2f} {unit}  {verdict}")

    print(f"\nsparkline field: {summary['sparkline_field']}  "
          f"({len(summary['sparkline'])} hourly point(s), "
          f"first {summary['sparkline'][0]:.1f})")
    if summary["sparkline_field"] != "display_index":
        print("FAIL: fleet sparkline did not switch to the display index")
        ok = False

    print("\nPASS — the severity trend reaches the dashboard (SYNTHETIC source)"
          if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
