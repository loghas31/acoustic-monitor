"""
state.py — local SQLite persistence (v2). Survives reboots and outages.

Tables:
    readings   one row per 30 s window, including the 40-dim FEATURE VECTOR.
               Storing the vector is what makes the feedback loop real: when a
               customer marks an alert "this was normal", baseline.py can
               refit on those exact windows. ~400 B/row, pruned at 7 days.
    feedback   windows the customer flagged as normal (via dashboard ->
               cloud -> MQTT cmd). Consumed by `baseline.py --retrain`.
    meta       key/value odds and ends.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts          REAL PRIMARY KEY,
    score       REAL, regime INTEGER, threshold REAL,
    anomalous   INTEGER NOT NULL DEFAULT 0,
    fr_hz       REAL, fr_reliable INTEGER,
    vector_json TEXT, mel_mean_json TEXT
);
CREATE TABLE IF NOT EXISTS anomalies (
    ts          REAL PRIMARY KEY,
    score       REAL, regime INTEGER, threshold REAL,
    detail_json TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    ts          REAL PRIMARY KEY,        -- window timestamp marked normal
    vector_json TEXT NOT NULL,
    fr_hz       REAL,
    created     REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class StateDB:
    # T4.3 fault-injection audit. A Pi with no RTC boots with whatever clock
    # the OS image had baked in (often years stale) until NTP steps it
    # forward — a STEP, not the small continuous slew NTP normally does once
    # synced. `ts - retention_s` trusted that jump blindly: a two-year step
    # produces a prune cutoff two years past every row this process has ever
    # written, and `DELETE FROM readings WHERE ts < cutoff` empties the whole
    # table in one call — not "at most one window lost" (the documented
    # design in DOC_FIRMWARE.md) but the entire learn history and every
    # queued severity trend, seconds after boot. `_trusted_prune_ts` bounds
    # how far the prune cutoff may advance using `time.monotonic()`, which a
    # wall-clock step does not touch, while still storing the corrected wall
    # time on the row itself — the dashboard should show the right date, the
    # deletion should not trust it blindly. See
    # tests/test_fault_injection.py::test_ntp_forward_jump_does_not_wipe_retention.
    MAX_CLOCK_SLACK_S = 3600.0     # generous room for a slow window, DST, etc.

    def __init__(self, path: Path | str, retention_days: int = 7):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SCHEMA)
        self.retention_s = retention_days * 86400
        self._mono0 = time.monotonic()
        self._wall0: float | None = None

    def _trusted_prune_ts(self, ts: float) -> float:
        """Clamp `ts` for retention pruning against a wall-clock step.

        `_wall0` anchors to the first timestamp this instance ever saw (i.e.
        roughly process start). From then on the trusted clock only advances
        by real elapsed time (`time.monotonic()`), so a later step in `ts` —
        forward OR backward — cannot move the prune cutoff by more than
        `MAX_CLOCK_SLACK_S` beyond what has genuinely elapsed. A real week of
        uptime still prunes a week of rows normally; a one-off correction
        cannot prune rows written moments before it.
        """
        if self._wall0 is None:
            self._wall0 = ts
        trusted = self._wall0 + (time.monotonic() - self._mono0) + self.MAX_CLOCK_SLACK_S
        return min(ts, trusted)

    def record_window(self, score: dict, feats: dict, ts: float | None = None) -> None:
        ts = ts or time.time()
        # T4.3 fault-injection audit. This is meant to be all-or-nothing per
        # window (no `commit()` is reached unless all three statements
        # succeed), but a failure that raises WITHOUT an explicit rollback()
        # leaves its statements sitting in SQLite's implicit transaction —
        # pending, not gone. A later, unrelated call that succeeds then
        # commits THAT transaction too, and the failed window reappears on
        # disk glued onto whichever window happened to come next: a fault
        # window (the anomaly half failed — disk full, say) resurfacing as a
        # bare reading with no anomaly record, silently reclassified as
        # unremarkable. `rollback()` on the exception path makes "one call,
        # one atomic unit" true regardless of what the caller does with the
        # exception (crash-and-restart today; some future retry loop
        # tomorrow). Measured before this fix:
        # tests/test_fault_injection.py::test_disk_full_on_anomaly_insert_does_not_leak_into_a_later_commit.
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, score["score"], score["regime"], score["threshold"],
                 int(score["anomalous"]), feats["fr_hz"], int(feats["fr_reliable"]),
                 json.dumps([round(float(v), 5) for v in feats["vector"]]),
                 json.dumps([round(float(v), 3) for v in feats["mel"].mean(axis=1)])))
            if score["anomalous"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO anomalies VALUES (?,?,?,?,?)",
                    (ts, score["score"], score["regime"], score["threshold"],
                     json.dumps({"band": feats.get("band"), "fr_hz": feats["fr_hz"]})))
            prune_ts = self._trusted_prune_ts(ts)
            self.conn.execute("DELETE FROM readings WHERE ts < ?", (prune_ts - self.retention_s,))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- feedback loop ---------------------------------------------------------

    def mark_normal(self, ts_from: float, ts_to: float) -> int:
        """Customer said 'this was normal' for an alert episode: copy the
        affected windows' vectors into the feedback table. Returns count."""
        rows = self.conn.execute(
            "SELECT ts, vector_json, fr_hz FROM readings WHERE ts BETWEEN ? AND ?",
            (ts_from, ts_to)).fetchall()
        now = time.time()
        for ts, vec, fr in rows:
            self.conn.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)",
                              (ts, vec, fr, now))
        self.conn.commit()
        return len(rows)

    def feedback_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.conn.execute(
            "SELECT vector_json, fr_hz FROM feedback ORDER BY ts").fetchall()
        if not rows:
            return np.empty((0,)), np.empty((0,))
        return (np.array([json.loads(r[0]) for r in rows]),
                np.array([r[1] for r in rows]))

    # -- misc -------------------------------------------------------------------

    def recent_anomalies(self, since_ts: float = 0.0) -> list:
        rows = self.conn.execute(
            "SELECT ts, score, regime, threshold, detail_json FROM anomalies "
            "WHERE ts >= ? ORDER BY ts", (since_ts,)).fetchall()
        return [{"ts": r[0], "score": r[1], "regime": r[2], "threshold": r[3],
                 "detail": json.loads(r[4])} for r in rows]

    def set_meta(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def close(self) -> None:
        self.conn.close()
