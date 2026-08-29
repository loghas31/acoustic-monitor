"""
tests/test_fault_injection.py — backlog T4.3, the fault-injection audit.

PRIORITY OVERRIDE, 2026-08-19: with hardware indefinitely blocked and the
subscription ending in ~2 weeks, this was promoted above everything else in
Tier 1-6. The framing (the task backlog (not in this public copy)): "it cannot predict what breaks, but
it can guarantee that when a sensor misbehaves the software SAYS IT DOESN'T
KNOW instead of returning a plausible wrong number." That is the F2 lesson
(docs/DOC_SELF_REVIEW.md) made systematic: F2 was a dead accelerometer
returning a perfectly believable 10 Hz shaft speed, found by luck. This file
goes looking for the same *shape* of bug — code that fails quietly into a
plausible number — in the five scenarios the backlog names:

    1. corrupt baseline file
    2. clock jump (NTP correction mid-window)
    3. disk full
    4. broker unreachable for days
    5. sensor unplugged mid-run

Two genuine bugs of exactly this shape were found by writing these tests and
are fixed under the frozen-file exception (a failing test demonstrating a
real bug):

  * `MahalanobisScorer` loaded a corrupt/truncated `baseline.npz` and either
    crashed with a raw, unhelpful exception (BadZipFile / EOFError / KeyError)
    or — worse, for a baseline whose numbers were merely wrong rather than
    unreadable — loaded successfully and scored silently. A NaN threshold
    makes `score > threshold` False by IEEE754 definition, so a corrupted
    threshold or precision matrix makes the unit report "not anomalous"
    forever. Fixed: `inference.py` now validates the schema and the
    finiteness of every numeric field at load time and refuses with the
    retrain command, the same policy the existing dimension-mismatch check
    uses.
  * `StateDB` pruned retention using the raw wall-clock timestamp of the
    CURRENT window. A Pi with no RTC that steps its clock forward by months
    or years the moment NTP syncs (a step, not the small continuous slew NTP
    does once already synced) produces a prune cutoff far in the future,
    and `DELETE FROM readings WHERE ts < cutoff` empties the ENTIRE table in
    one call — not "at most one window lost" (the design DOC_FIRMWARE.md
    documents) but the whole learn history seconds after boot. Fixed:
    `state.py` now bounds the prune cutoff by real elapsed time
    (`time.monotonic()`, which a wall-clock step does not touch) while still
    storing the corrected wall time on the row.

The other three scenarios were tested and NOT found to need a code change —
recorded as verification, not assumed:

  * Disk full mid-window: `record_window`'s three statements are one SQLite
    transaction; a failure on any of them raises before `commit()`, so the
    failing window is not partially written and earlier committed windows
    are untouched. The exception propagates (matches the documented
    "systemd restarts on crash, RestartSec=10" design in DOC_FIRMWARE.md,
    the same contract `local_webhook`'s "must never crash the loop" comment
    explicitly does NOT apply to database writes) — the loop dies loudly
    rather than continuing on a half-written state.
  * Broker unreachable for days: `mqtt_client.py` (not a frozen file) already
    queues anomalies in a bounded deque and drops telemetry, per its own
    module docstring. Confirmed by test rather than re-read; the one gap
    found — dropping the oldest queued anomaly on overflow was silent — was
    hardened with a log line, not a behaviour change.
  * Accelerometer dying mid-run: `features.channel_stats`' dead-channel
    sentinel (-9.0 logrms) sits far outside a live-signal learn
    distribution, so the Mahalanobis distance genuinely spikes and the
    window is scored anomalous — the system does not know WHY (it never
    diagnoses, by design, the system overview (not in this public copy) §3), but it does not carry on
    reporting "normal" on a channel it can no longer see either.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"), str(ROOT / "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline import fit_baseline, operating_point, save_baseline  # noqa: E402
from capture import SimulatedSource                                 # noqa: E402
from features import FEATURE_NAMES, extract_features                # noqa: E402
from inference import MahalanobisScorer                              # noqa: E402
from mqtt_client import MqttUplink                                   # noqa: E402
from state import StateDB                                            # noqa: E402

RNG = np.random.default_rng(0)
D = len(FEATURE_NAMES)


def _gaussian_baseline_dict(n_per_regime=30):
    """A minimal, well-formed baseline dict — mirrors test_baseline.py's
    make_data, kept local so this file can corrupt copies of it freely."""
    X = RNG.normal(0, 1, (n_per_regime, D))
    OP = np.column_stack([np.full(n_per_regime, 50.0) + RNG.normal(0, 0.05, n_per_regime),
                          RNG.normal(-0.1, 0.01, n_per_regime),
                          RNG.normal(-0.1, 0.01, n_per_regime)])
    return fit_baseline(X, OP)


def _dummy_score_and_feats(anomalous=False):
    """Minimal (score, feats) pair shaped like what main.py's loop passes to
    StateDB.record_window — enough fields for the schema, nothing signal-
    specific, because these tests are about persistence, not detection."""
    score = {"score": 1.23, "regime": 0, "threshold": 4.56, "anomalous": anomalous}
    feats = {"vector": np.zeros(D), "fr_hz": 50.0, "fr_reliable": True,
             "mel": np.zeros((4, 4)), "band": (3000.0, 6000.0)}
    return score, feats


# ============================================================================
# 1. Corrupt baseline file
# ============================================================================

def test_garbage_baseline_file_fails_legibly(tmp_path):
    """Not a zip file at all — the crudest corruption (wrong file, disk
    scribbled over it). Must raise something a student can act on, not a
    bare zipfile/OSError traceback."""
    path = tmp_path / "baseline.npz"
    path.write_bytes(b"this is not an npz file, just 64 bytes of noise!!")
    with pytest.raises(ValueError) as exc:
        MahalanobisScorer(path)
    msg = str(exc.value)
    assert "firmware/baseline.py" in msg, f"must name the retrain command: {msg}"


def test_truncated_baseline_file_fails_legibly(tmp_path):
    """A real baseline, cut off mid-write — the realistic case (power loss
    during `np.savez_compressed`, which is not atomic)."""
    b = _gaussian_baseline_dict()
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    full = path.read_bytes()
    path.write_bytes(full[: len(full) // 2])       # simulate a cut-short write
    with pytest.raises(ValueError) as exc:
        MahalanobisScorer(path)
    assert "firmware/baseline.py" in str(exc.value)


def test_baseline_missing_fields_fails_legibly(tmp_path):
    """A syntactically valid npz that is not a baseline (wrong producer,
    old/future schema) must name what is missing, not raise a raw KeyError
    from inside score() the first time a field is touched."""
    path = tmp_path / "baseline.npz"
    np.savez_compressed(path, some_other_field=np.array([1, 2, 3]))
    with pytest.raises(ValueError) as exc:
        MahalanobisScorer(path)
    msg = str(exc.value)
    assert "global_mean" in msg or "missing" in msg.lower()


@pytest.mark.parametrize("field", ["thresholds", "means", "precisions",
                                   "global_mean", "op_centroids"])
def test_nan_poisoned_baseline_is_refused_not_silently_scored(tmp_path, field):
    """THE DANGEROUS CASE. A baseline that loads without error but contains
    a NaN must be refused at load time, not scored. Left unchecked: NaN
    propagates into `score()`, and `score > threshold` is False whenever
    either operand is NaN (IEEE754), so a corrupted threshold or precision
    matrix makes the device silently report "not anomalous" — forever, on
    exactly the unit whose baseline just got corrupted. This is the F2
    failure shape (docs/DOC_SELF_REVIEW.md): a plausible-looking result
    (False, i.e. "healthy") produced by data that measured nothing."""
    b = _gaussian_baseline_dict()
    b[field] = np.asarray(b[field], dtype=float).copy()
    b[field].flat[0] = np.nan
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    with pytest.raises(ValueError) as exc:
        MahalanobisScorer(path)
    msg = str(exc.value)
    assert field in msg, f"error must name the corrupted field: {msg}"


def test_zero_std_baseline_is_refused_not_amplified_into_a_fake_extreme(tmp_path):
    """A zero (or negative, from bit-flip corruption) global_std divides a
    window's z-score by ~0, manufacturing a huge distance that LOOKS like a
    genuine extreme reading rather than a broken file. Must be refused."""
    b = _gaussian_baseline_dict()
    b["global_std"] = np.asarray(b["global_std"], dtype=float).copy()
    b["global_std"][3] = 0.0
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    with pytest.raises(ValueError) as exc:
        MahalanobisScorer(path)
    assert "global_std" in str(exc.value)


def test_well_formed_baseline_still_loads(tmp_path):
    """Sanity check on the fixture and the new validation: a genuinely clean
    baseline must NOT be rejected by the new checks — the whole point is to
    catch corruption, not become a stricter false-positive machine."""
    b = _gaussian_baseline_dict()
    path = tmp_path / "baseline.npz"
    save_baseline(path, b)
    scorer = MahalanobisScorer(path)
    s = scorer.score(RNG.normal(0, 1, D), np.array([50.0, -0.1, -0.1]))
    assert np.isfinite(s["score"])


# ============================================================================
# 2. Clock jump (NTP correction mid-window)
# ============================================================================

def test_ntp_forward_jump_does_not_wipe_retention(tmp_path):
    """THE DANGEROUS CASE. A Pi with no RTC boots with a stale/default
    clock and steps it forward by months or years the moment NTP syncs.
    Before the fix, `record_window`'s retention prune used that jumped
    timestamp directly: `DELETE FROM readings WHERE ts < (jumped_ts -
    retention_s)` deletes every row ever written by this process, not "at
    most one window" (DOC_FIRMWARE.md's documented contract)."""
    db = StateDB(tmp_path / "s.db", retention_days=7)
    t0 = 1_700_000_000.0                              # arbitrary reference epoch
    for i in range(5):
        score, feats = _dummy_score_and_feats()
        db.record_window(score, feats, ts=t0 + i * 30.0)
    before = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    assert before == 5

    # NTP steps the clock forward two years in a single subsequent call.
    jumped_ts = t0 + 2 * 365 * 86400
    score, feats = _dummy_score_and_feats()
    db.record_window(score, feats, ts=jumped_ts)

    after = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    assert after >= 5, (
        f"a forward clock step must not prune history that was just "
        f"written: had {before}, have {after} after the step")
    # The new row itself keeps the corrected (accurate) wall time — the
    # dashboard should show the right date even though pruning must not
    # trust it blindly.
    stored_ts = db.conn.execute(
        "SELECT ts FROM readings WHERE ts > ?", (t0 + 1000,)).fetchone()[0]
    assert stored_ts == jumped_ts


def test_ntp_backward_jump_does_not_crash_or_lose_data(tmp_path):
    """The other direction: a clock correcting BACKWARD (e.g. a fast image
    clock being corrected to real time) must not raise and must not silently
    merge distinct windows onto the same primary key."""
    db = StateDB(tmp_path / "s.db", retention_days=7)
    t0 = 1_700_100_000.0
    for i in range(3):
        score, feats = _dummy_score_and_feats()
        db.record_window(score, feats, ts=t0 + i * 30.0)
    # step backward by a year, then keep recording
    back_ts = t0 - 365 * 86400
    for i in range(3):
        score, feats = _dummy_score_and_feats()
        db.record_window(score, feats, ts=back_ts + i * 30.0)
    n = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    assert n == 6, "distinct windows before and after a backward step must both survive"


def test_normal_operation_still_prunes_old_readings(tmp_path):
    """The fix must not disable retention altogether — genuinely old rows,
    reached via ordinary elapsed time (not a clock step), still get pruned.
    Uses a 0-day retention window so 'old' is trivially anything before
    'now'."""
    db = StateDB(tmp_path / "s.db", retention_days=0)
    t0 = 1_700_000_000.0
    score, feats = _dummy_score_and_feats()
    db.record_window(score, feats, ts=t0)
    # a second window a day later, with elapsed wall-clock time also having
    # passed (simulated by advancing ts only — monotonic time in a fast test
    # process is ~0, so the slack constant is what allows this prune; a real
    # day of uptime would allow it on its own merits)
    db.record_window(score, feats, ts=t0 + 90000.0)
    n = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    assert n == 1, "retention must still work when the clock is NOT jumping"


# ============================================================================
# 3. Disk full mid-window
# ============================================================================

class _FailingConn:
    """Wraps a real sqlite3.Connection and raises OperationalError from
    `execute()` when `fail(sql)` is True, otherwise delegates. A direct
    `monkeypatch.setattr(conn, "execute", ...)` cannot work here: sqlite3's
    C-level Connection type has a read-only `execute` slot, so the fake has
    to sit at the `StateDB.conn` attribute level instead."""

    def __init__(self, real, fail):
        self._real, self._fail = real, fail

    def execute(self, sql, *a, **kw):
        if self._fail(sql):
            raise sqlite3.OperationalError("database or disk is full")
        return self._real.execute(sql, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_disk_full_loses_at_most_one_window(tmp_path, monkeypatch):
    """VERIFICATION, not a fix: record_window's three statements are one
    SQLite transaction (no commit() until all three succeed), so a failure
    on any of them must not touch rows already committed by earlier calls,
    and must not leave a half-written row for the failing window either."""
    db = StateDB(tmp_path / "s.db")
    for i in range(3):
        score, feats = _dummy_score_and_feats()
        db.record_window(score, feats, ts=1000.0 + i)
    assert db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 3

    real_conn = db.conn
    monkeypatch.setattr(db, "conn", _FailingConn(
        real_conn, lambda sql: sql.startswith("INSERT OR REPLACE INTO readings")))
    score, feats = _dummy_score_and_feats()
    with pytest.raises(sqlite3.OperationalError):
        db.record_window(score, feats, ts=1003.0)

    monkeypatch.setattr(db, "conn", real_conn)          # disk space freed
    n = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    assert n == 3, "the failed window must not have partially committed"

    # the device recovers on its own once space is available again — no
    # special reset needed, the connection is not left wedged
    score, feats = _dummy_score_and_feats()
    db.record_window(score, feats, ts=1004.0)
    assert db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 4


def test_disk_full_on_anomaly_insert_does_not_leak_into_a_later_commit(tmp_path, monkeypatch):
    """THE DANGEROUS CASE. The READING insert succeeds but the ANOMALY
    insert (same call, an actual fault window) fails. Both statements sit in
    one SQLite implicit transaction that is never explicitly rolled back on
    the exception path — so the reading half does not vanish, it just stays
    PENDING on the connection. If a later, ordinary window then succeeds and
    calls `commit()`, that commit flushes BOTH: the earlier fault window
    reappears on disk as a bare reading with no matching anomaly record,
    silently reclassified as unremarkable and glued onto whatever window
    happened to come after it. Verified against a FRESH connection to the
    same file, which shows only what actually reached disk — the live
    connection can still see its own uncommitted work, which is why that
    check is not enough on its own."""
    db = StateDB(tmp_path / "s.db")
    real_conn = db.conn
    monkeypatch.setattr(db, "conn", _FailingConn(
        real_conn, lambda sql: sql.startswith("INSERT OR REPLACE INTO anomalies")))
    score, feats = _dummy_score_and_feats(anomalous=True)
    with pytest.raises(sqlite3.OperationalError):
        db.record_window(score, feats, ts=2000.0)

    monkeypatch.setattr(db, "conn", real_conn)          # disk space freed
    score2, feats2 = _dummy_score_and_feats(anomalous=False)
    db.record_window(score2, feats2, ts=2030.0)         # an ordinary later window

    fresh = sqlite3.connect(str(tmp_path / "s.db"))
    rows = [r[0] for r in fresh.execute("SELECT ts FROM readings ORDER BY ts").fetchall()]
    fresh.close()
    assert rows == [2030.0], (
        f"the failed window's reading must not leak into a later, unrelated "
        f"commit: found {rows}")


# ============================================================================
# 4. Broker unreachable for days
# ============================================================================

def _uplink(tmp_path=None, maxlen=None):
    cfg = {"device": {"id": "dev-test"},
          "mqtt": {"host": "localhost", "port": 8883, "tls": False,
                   "api_key": "", "base_topic": "devices"}}
    up = MqttUplink(cfg)
    if maxlen is not None:
        import collections
        up._queue = collections.deque(maxlen=maxlen)
    return up


def test_telemetry_is_dropped_while_offline_not_queued():
    """Per mqtt_client.py's own module docstring: telemetry is a time series
    and stale points are worthless, so it must be dropped, not queued,
    while disconnected — queueing it would eventually re-introduce the
    unbounded-memory problem the anomaly queue is deliberately bounded
    against."""
    up = _uplink()
    assert up.connected is False
    calls = []
    up.client.publish = lambda *a, **kw: calls.append(a)
    for i in range(10):
        up.publish_telemetry({"window": i})
    assert calls == [], "telemetry must not be queued or sent while offline"


def test_anomalies_queue_while_offline_and_replay_in_order_on_reconnect():
    up = _uplink()
    published = []
    up.client.publish = lambda topic, data, qos=0, **kw: published.append((topic, data, qos))
    up.client.subscribe = lambda *a, **kw: None

    for i in range(5):
        up.publish_anomaly({"episode": i})
    assert published == [], "must not attempt to publish while offline"
    assert len(up._queue) == 5

    up._on_connect(up.client, None, {}, 0)             # rc=0 -> connected
    assert up.connected is True
    assert len(up._queue) == 0, "queue must drain fully on reconnect"
    episodes = [data for _, data, _ in published if "episode" in data]
    assert len(episodes) == 5
    # order preserved: FIFO replay, oldest anomaly first
    import json
    assert [json.loads(e)["episode"] for e in episodes] == [0, 1, 2, 3, 4]


def test_offline_queue_overflow_does_not_crash_a_multi_day_outage():
    """A multi-day outage with the bounded deque full must keep accepting
    new anomalies (never raise, never block) rather than take the device's
    local alerting down with it — MqttUplink is explicitly a display/cloud
    concern, never the alert decision itself (main.py's docstring)."""
    up = _uplink(maxlen=5)
    for i in range(20):
        up.publish_anomaly({"episode": i})           # must not raise
    assert len(up._queue) == 5
    import json
    kept = [json.loads(d)["episode"] for _, d in up._queue]
    assert kept == [15, 16, 17, 18, 19], "oldest queued anomalies are evicted first"


def test_offline_queue_overflow_is_logged_not_silent(caplog):
    """T4.3 hardening: dropping a queued anomaly used to be silent (a plain
    `deque(maxlen=...)` eviction). A multi-day outage that drops anomaly
    events should leave a trace in the log, even though continuing to accept
    new ones is still the right behaviour."""
    import logging
    caplog.set_level(logging.WARNING, logger="mqtt_client")
    up = _uplink(maxlen=2)
    for i in range(4):
        up.publish_anomaly({"episode": i})
    assert any("drop" in r.message.lower() for r in caplog.records), (
        "an offline-queue eviction must be logged")


# ============================================================================
# 5. Sensor unplugged mid-run
# ============================================================================

def test_accelerometer_dying_mid_run_is_not_silently_reported_healthy(tmp_path):
    """An accelerometer that dies mid-run (a connector working loose is the
    realistic mechanism, not a clean unplug) must not let the unit keep
    reporting "normal" on a channel it can no longer see. Verified against
    the REAL simulated signal path and the REAL scorer, not a synthetic
    Gaussian fixture: a baseline is learned with a live 3-axis accelerometer
    (capture.SimulatedSource, the supported dual-channel build), then the
    accelerometer channel is zeroed — the exact dead-channel signature
    `capture.HardwareSource._open_accel`'s docstring specifies for a missing
    IIS3DWB — partway through a run."""
    def schedule(i):
        return {"kind": "normal", "severity": 0.0, "fr": 50.0}   # single regime
    src = SimulatedSource(window_s=6.0, schedule=schedule, seed=99)
    gen = src.windows()

    X, OP = [], []
    for _ in range(64):
        audio, accel = next(gen)
        feats = extract_features(audio, src.fs_audio, accel, src.fs_accel)
        X.append(feats["vector"])
        OP.append(operating_point(feats["vector"], feats["fr_hz"]))
    b = fit_baseline(np.array(X), np.array(OP))
    path = tmp_path / "b.npz"
    save_baseline(path, b)
    scorer = MahalanobisScorer(path)

    N_PRE = 10
    results = []
    for i in range(N_PRE + 8):
        audio, accel = next(gen)
        if i >= N_PRE:
            accel = np.zeros_like(accel)                 # the accelerometer dies here
        feats = extract_features(audio, src.fs_audio, accel, src.fs_accel)
        op = operating_point(feats["vector"], feats["fr_hz"])
        score = scorer.score(feats["vector"], op)
        results.append((score["score"], score["anomalous"], feats["fr_reliable"]))

    assert all(np.isfinite(s) for s, _, _ in results), "score must never be NaN/Inf"
    pre, post = results[:N_PRE], results[N_PRE:]
    # threshold is a 99.5th-percentile-ish estimate at n~64, not a guarantee —
    # docs/DOC_STATUS.md measures held-out per-window FPR up to ~4-6% at this
    # learn size, so tolerate at most one stray false positive in 10 live
    # windows rather than demanding statistical impossibility of one.
    n_false_positive = sum(1 for _, a, _ in pre if a)
    assert n_false_positive <= 1, f"too many false positives on live windows: {pre}"
    assert all(a for _, a, _ in post), (
        f"a dead accelerometer must not silently continue scoring healthy: {post}")
    # honest about what changed: fr no longer claims a cross-checked reading
    assert all(not rel for _, _, rel in post), (
        "fr_reliable must drop once the accelerometer channel is dead")


def test_audio_dying_mid_run_is_not_silently_reported_healthy(tmp_path):
    """Mirror case: the MICROPHONE dies (the more load-bearing channel —
    the system overview (not in this public copy) notes the mic alone covers the resonance band). Same
    contract: must not carry on reporting normal on a channel it can no
    longer see."""
    def schedule(i):
        return {"kind": "normal", "severity": 0.0, "fr": 50.0}
    src = SimulatedSource(window_s=6.0, schedule=schedule, seed=7)
    gen = src.windows()

    X, OP = [], []
    for _ in range(64):
        audio, accel = next(gen)
        feats = extract_features(audio, src.fs_audio, accel, src.fs_accel)
        X.append(feats["vector"])
        OP.append(operating_point(feats["vector"], feats["fr_hz"]))
    b = fit_baseline(np.array(X), np.array(OP))
    path = tmp_path / "b.npz"
    save_baseline(path, b)
    scorer = MahalanobisScorer(path)

    N_PRE = 10
    results = []
    for i in range(N_PRE + 8):
        audio, accel = next(gen)
        if i >= N_PRE:
            audio = np.zeros_like(audio)                  # the microphone dies here
        feats = extract_features(audio, src.fs_audio, accel, src.fs_accel)
        op = operating_point(feats["vector"], feats["fr_hz"])
        score = scorer.score(feats["vector"], op)
        results.append((score["score"], score["anomalous"]))

    assert all(np.isfinite(s) for s, _ in results)
    pre, post = results[:N_PRE], results[N_PRE:]
    n_false_positive = sum(1 for _, a in pre if a)
    assert n_false_positive <= 1, f"too many false positives on live windows: {pre}"
    assert all(a for _, a in post), (
        f"a dead microphone must not silently continue scoring healthy: {post}")
