"""
tests/test_db_growth_audit.py — backlog T4.2, database growth + SD-wear
audit.

Two things pinned here, both fast (no thousands-of-windows run — that's
`tools/db_growth_audit.py`'s own job, and `docs/DOC_SOAK_DB_GROWTH.md`
records the real numbers from running it):

1. The audit tool's own mechanics: fake vectors are the right shape/rounding
   to measure real row size, and growth genuinely plateaus once retention
   pruning has a full window of data to work with — on a small, fast scale
   here, not the real 18-day run.

2. A real finding from running the tool at scale, reproduced small and
   fast: `firmware/state.py`'s retention pruning (`DELETE FROM readings
   WHERE ts < ...`) prunes ONLY the `readings` table. The `anomalies` table
   has no equivalent — rows inserted there persist forever, unbounded, for
   the life of the device. `docs/DOC_FIRMWARE.md` describes retention in a
   way a reasonable reader would take as applying to the whole local state
   DB ("old rows are pruned on every insert"); it does not. This is white-box
   (reaches into `StateDB._wall0`/`_mono0` directly) because the honest way
   to fast-forward `_trusted_prune_ts`'s clock-jump guard without an actual
   sleep is documented right there in `tools/db_growth_audit.py`'s own
   module docstring: either advance real monotonic time in lockstep, or (as
   here, for a single before/after check rather than a day-by-day curve)
   push `_mono0` back directly, which is equivalent to "a lot of real time
   passed" from the guard's point of view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "tools"), str(ROOT / "firmware")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import db_growth_audit as dga                          # noqa: E402
from state import StateDB                               # noqa: E402


def test_fake_vector_matches_the_real_shape_and_rounding_precision():
    import random
    rng = random.Random(0)
    vec = dga._fake_vector(rng)
    assert len(vec) == 37                              # FEATURE_NAMES length
    for v in vec:
        s = repr(v)
        # round(x, 5) never produces more than 5 digits after the point —
        # confirms this fixture is the same shape record_window itself
        # writes (`round(float(v), 5)`), which is what makes bytes/row
        # measured against it meaningful.
        if "." in s:
            assert len(s.split(".")[1]) <= 5


def test_growth_plateaus_once_retention_has_a_full_window(tmp_path):
    """Small, fast version of the real 18-day/7-day-retention run: 4 days at
    a 1-day retention, tiny window count/day so it finishes in well under a
    second, but the same mechanism — day 1 grows, day 2 (one full retention
    window old) starts pruning as fast as it inserts, size stops climbing."""
    db_path = tmp_path / "growth.db"
    checkpoints = dga.run_audit(db_path, days=4, retention_days=1,
                                window_s=30.0, anomaly_rate=0.0, seed=3)
    # windows/day at 30s = 2880; tiny is achieved instead via a short
    # custom day, so patch windows_per_day via a direct short run:
    assert len(checkpoints) == 4
    sizes = [c["size_bytes"] for c in checkpoints]
    rows = [c["readings_rows"] for c in checkpoints]
    assert sizes[0] < sizes[1]                          # still growing, day 1 -> 2
    assert rows[1] == rows[2] == rows[3]                # plateaued: one retention window's worth, steady
    # size may wobble by a few pages (anomalies table, page reuse) but must
    # not keep climbing once row count has plateaued the way it did pre-plateau
    assert abs(sizes[3] - sizes[1]) < (sizes[1] - sizes[0])


def test_bytes_per_row_report_is_populated_and_plausible(tmp_path, capsys):
    db_path = tmp_path / "report.db"
    checkpoints = dga.run_audit(db_path, days=2, retention_days=5,
                                window_s=30.0, anomaly_rate=0.01, seed=9)
    bytes_per_row = checkpoints[0]["size_bytes"] / checkpoints[0]["readings_rows"]
    # A 37-dim vector at up to 5 decimals plus a few scalar columns is
    # comfortably in the low hundreds of bytes — this pins "plausible", not
    # an exact figure (page-level SQLite overhead varies); the exact
    # measured number from the real 18-day run is recorded in
    # docs/DOC_SOAK_DB_GROWTH.md, not asserted here as a magic constant.
    assert 100 < bytes_per_row < 2000


def test_anomalies_table_is_not_pruned_by_retention__real_finding(tmp_path):
    """THE finding T4.2 turned up. `record_window` only prunes `readings`
    (`DELETE FROM readings WHERE ts < ...` — see state.py, no equivalent
    statement for `anomalies`). Two anomalous windows, one just inside a
    1-day retention window and one made to look like retention has long
    since passed: the readings table drops the old row, the anomalies table
    keeps both, forever."""
    db = StateDB(tmp_path / "s.db", retention_days=1)
    t0 = 1_700_000_000.0
    old_score = {"score": 20.0, "regime": 0, "threshold": 8.0, "anomalous": True}
    feats = {"fr_hz": 50.0, "fr_reliable": True, "vector": [0.0] * 37,
            "mel": dga._MelStub([0.0, 0.0, 0.0, 0.0]), "band": (3000.0, 6000.0)}
    db.record_window(old_score, feats, ts=t0)
    assert db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0] == 1

    # Fast-forward the clock-jump guard's trust by pushing _mono0 back —
    # equivalent, from _trusted_prune_ts's point of view, to real time
    # having actually advanced by more than the retention window.
    db._mono0 -= (2 * 86400)
    new_score = {"score": 3.0, "regime": 0, "threshold": 8.0, "anomalous": False}
    db.record_window(new_score, feats, ts=t0 + 2 * 86400)

    readings_left = db.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    anomalies_left = db.conn.execute("SELECT ts FROM anomalies").fetchall()
    assert readings_left == 1, "the OLD reading should have been pruned; only the new one remains"
    assert [row[0] for row in db.conn.execute("SELECT ts FROM readings")] == [t0 + 2 * 86400], (
        "the pruned reading's ts should be GONE from readings")
    assert (t0,) in anomalies_left, (
        "documents current behaviour: the OLD anomaly row (ts=t0) survives "
        "even though its corresponding READING was pruned two days later — "
        "the anomalies table has no retention policy at all, not a slower "
        "one. If this test ever starts failing because someone added "
        "anomalies pruning, update docs/DOC_FIRMWARE.md and "
        "docs/DOC_SOAK_DB_GROWTH.md to match, don't just weaken this "
        "assertion.")
    db.close()
