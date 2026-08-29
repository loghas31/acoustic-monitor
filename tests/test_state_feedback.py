"""
tests/test_state_feedback.py — backlog T3.4, the feedback round-trip.

the system overview (not in this public copy) names the "this was normal" button as defence #4 against
false alarms, and the *only* one that adapts to a specific site: a customer
marks an alert episode normal, and `firmware/baseline.py --retrain` is
supposed to fold those exact windows back into the model so the same thing
does not alarm again. Before this file, nothing tested that this actually
happens — `state.py`'s `mark_normal` / `feedback_vectors` and `baseline.py`'s
`--retrain` path each existed, and nothing connected them under test. This is
a real, previously-unverified gap, distinct from `tests/test_fault_injection.py`
(which covers `state.py`'s OTHER halves: retention pruning and record_window's
transaction safety) and from `tests/test_reporting.py`/`test_threshold.py`
(which retrain baselines from scratch, never via customer feedback).

`tests/test_phone_monitor.py` already covers T3.4's `capture.py` ask
(FileSource round-trip, mic-only degradation) as a side effect of T7.2 — not
repeated here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"), str(ROOT / "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline import (fit_baseline, load_baseline, operating_point,   # noqa: E402
                      save_baseline)
from capture import SimulatedSource                                    # noqa: E402
from features import FEATURE_NAMES, extract_features                   # noqa: E402
from inference import MahalanobisScorer                                # noqa: E402
from state import StateDB                                              # noqa: E402

D = len(FEATURE_NAMES)


def _score_feats_pair(scorer: MahalanobisScorer, vector, fr_hz=50.0, fr_reliable=True):
    """Real (score, feats) pair, shaped exactly like what firmware/main.py's
    loop passes to StateDB.record_window — built from the REAL scorer, not
    the fixed dummy dict test_fault_injection.py uses, because this file's
    tests need the score/threshold relationship to be genuine."""
    op = operating_point(vector, fr_hz)
    score = scorer.score(vector, op)
    feats = {"vector": vector, "fr_hz": fr_hz, "fr_reliable": fr_reliable,
             "mel": np.zeros((4, 4)), "band": (3000.0, 6000.0)}
    return score, feats


# ----------------------------------------------------------------------------
# 1. mark_normal / feedback_vectors, direct
# ----------------------------------------------------------------------------

def _seed_readings(db: StateDB, n: int, base_ts: float, vector_fn=None):
    """Writes n readings 30 s apart, returns their timestamps."""
    tss = []
    for i in range(n):
        ts = base_ts + i * 30.0
        vec = vector_fn(i) if vector_fn else np.zeros(D)
        score = {"score": 1.0, "regime": 0, "threshold": 5.0, "anomalous": False}
        feats = {"vector": vec, "fr_hz": 50.0, "fr_reliable": True,
                 "mel": np.zeros((4, 4)), "band": (3000.0, 6000.0)}
        db.record_window(score, feats, ts=ts)
        tss.append(ts)
    return tss


def test_mark_normal_copies_only_the_windows_in_range(tmp_path):
    db = StateDB(tmp_path / "state.db")
    tss = _seed_readings(db, 5, base_ts=1000.0,
                         vector_fn=lambda i: np.full(D, float(i)))
    # mark the middle three windows (index 1..3) as the feedback episode
    n = db.mark_normal(tss[1], tss[3])
    assert n == 3

    vecs, frs = db.feedback_vectors()
    assert vecs.shape == (3, D)
    # exactly windows 1, 2, 3's vectors (each column is a constant == its index)
    assert sorted(vecs[:, 0].tolist()) == [1.0, 2.0, 3.0]
    assert np.all(frs == 50.0)


def test_mark_normal_on_an_empty_range_marks_nothing(tmp_path):
    db = StateDB(tmp_path / "state.db")
    _seed_readings(db, 3, base_ts=1000.0)
    n = db.mark_normal(5000.0, 6000.0)          # no readings in this range
    assert n == 0
    vecs, frs = db.feedback_vectors()
    assert vecs.shape == (0,) and frs.shape == (0,)


def test_mark_normal_twice_on_the_same_range_does_not_duplicate(tmp_path):
    """INSERT OR REPLACE on the feedback table's ts primary key — re-marking
    an episode (e.g. the customer clicks the button twice, or a retry after a
    dropped MQTT command) must not double-count the same window when
    --retrain later folds it in."""
    db = StateDB(tmp_path / "state.db")
    tss = _seed_readings(db, 4, base_ts=2000.0)
    n1 = db.mark_normal(tss[0], tss[-1])
    n2 = db.mark_normal(tss[0], tss[-1])
    assert n1 == n2 == 4
    vecs, _ = db.feedback_vectors()
    assert vecs.shape[0] == 4, "re-marking the same range must not duplicate feedback rows"


def test_feedback_vectors_survive_a_reconnect(tmp_path):
    """The whole point of persisting feedback in SQLite rather than RAM: a
    reboot between the customer's click and the next retrain must not lose
    it. Simulated by closing and reopening a StateDB on the same path."""
    path = tmp_path / "state.db"
    db1 = StateDB(path)
    tss = _seed_readings(db1, 2, base_ts=3000.0)
    db1.mark_normal(tss[0], tss[-1])
    db1.close()

    db2 = StateDB(path)
    vecs, frs = db2.feedback_vectors()
    assert vecs.shape == (2, D)


# ----------------------------------------------------------------------------
# 2. The real --retrain CLI, end to end: does folding feedback in actually
#    change what the model calls anomalous?
# ----------------------------------------------------------------------------

def test_retrain_folds_feedback_into_a_new_baseline_and_desensitises_it(tmp_path):
    """The real regression test for the feedback loop's whole reason to
    exist. Build an initial baseline the same way `baseline.py --simulate`
    does; construct ONE genuinely anomalous-looking window (a loud transient,
    not a fixture dict) that the initial baseline correctly flags; record it
    and mark it 'normal'; run the REAL `firmware/baseline.py --retrain` CLI
    as a subprocess (not a reimplementation of its logic) against the saved
    baseline and the state DB; then check the retrained baseline scores that
    SAME window less severely than the original did — proving the fold-in
    changed the model, not just that the CLI exits 0."""
    out = tmp_path / "baseline.npz"
    init_db = tmp_path / "init_state.db"

    # 1. Initial baseline, exactly as `baseline.py --simulate --windows 48`
    #    would produce (two-regime 50/30 Hz schedule, matching main()'s own
    #    --simulate branch) so `X_train`/`OP_train` are realistic, not a toy.
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"), "--simulate",
         "--windows", "48", "--out", str(out), "--db", str(init_db)],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    initial = load_baseline(out)
    n_before = int(initial["X_train"].shape[0])

    # 2. A REALISTIC feedback episode: `mark_normal(ts_from, ts_to)` folds in
    #    every window an alert episode covered, and in production an episode
    #    is only ever raised after `persist_minutes` (default 30 min = 60
    #    windows) of continuous anomalous scoring — never one window. A
    #    single loud outlier is exactly what T1.6's contamination guard
    #    exists to distrust (measured below); ten windows of the SAME
    #    recurring transient (a nearby forklift on a schedule, a door that
    #    bangs every cycle — the system overview (not in this public copy)'s own example of defence #4) is the
    #    realistic shape of "this was normal" feedback and is what this test
    #    uses.
    def _transient_feats(seed):
        src = SimulatedSource(fs_audio=16000, fs_accel=6400, seed=seed)
        audio, accel = next(iter(src.windows()))
        audio = audio.copy()
        audio[: len(audio) // 4] += 3.0 * np.max(np.abs(audio))
        return extract_features(audio, 16000, accel, 6400)

    scorer = MahalanobisScorer(out)
    episode_feats = [_transient_feats(900 + i) for i in range(10)]
    episode_scores = []
    for f in episode_feats:
        op = operating_point(f["vector"], f["fr_hz"])
        episode_scores.append(scorer.score(f["vector"], op))
    assert all(s["anomalous"] for s in episode_scores), (
        "fixture is supposed to look anomalous against the initial baseline; "
        "if this stops being true the transient needs to be louder")
    ratio_before = np.mean([s["score"] / s["threshold"] for s in episode_scores])

    # A single window from the same pattern, scored the SAME way, is the
    # control for the "one outlier vs a real recurring pattern" comparison
    # below — built exactly like episode_feats[0] but never fed back.
    control_feats = _transient_feats(seed=99)
    control_op = operating_point(control_feats["vector"], control_feats["fr_hz"])
    control_before = scorer.score(control_feats["vector"], control_op)

    # 3. Record the whole episode and mark it normal, in a FRESH state DB.
    db_path = tmp_path / "state.db"
    db = StateDB(db_path)
    base_ts = time.time()
    for i, (f, s) in enumerate(zip(episode_feats, episode_scores)):
        db.record_window(s, f, ts=base_ts + i * 30.0)
    n_fed_back = db.mark_normal(base_ts, base_ts + 9 * 30.0)
    assert n_fed_back == 10
    db.close()

    # 4. The real CLI.
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"), "--retrain",
         "--out", str(out), "--db", str(db_path)],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    # main() prints a one-line "retraining: N original + M feedback windows"
    # progress message BEFORE the JSON summary, and (when contaminated) a
    # "!! At least one regime..." warning AFTER it, all on the same stream —
    # raw_decode from the first '{' and ignore anything past the object,
    # rather than assuming stdout is pure JSON top to bottom.
    summary, _ = json.JSONDecoder().raw_decode(r.stdout[r.stdout.index("{"):])
    assert sum(summary["windows_per_regime"]) == n_before + n_fed_back, (
        "retrain must fold in exactly the fed-back windows, on top of the "
        "original learn set — not replace it")

    # 5. Score the fed-back episode AND the never-fed-back control against
    #    the RETRAINED baseline.
    retrained_scorer = MahalanobisScorer(out)
    ratios_after = []
    for f in episode_feats:
        op = operating_point(f["vector"], f["fr_hz"])
        after = retrained_scorer.score(f["vector"], op)
        ratios_after.append(after["score"] / after["threshold"])
    ratio_after = float(np.mean(ratios_after))
    assert ratio_after < ratio_before, (
        f"retraining on a recurring pattern's feedback should make it look "
        f"LESS anomalous on average, not more: ratio {ratio_before:.2f} -> "
        f"{ratio_after:.2f}")
    assert all(r_ < 1.0 for r_ in ratios_after), (
        f"a genuinely RECURRING pattern, fed back as its own realistic-sized "
        f"episode (10 windows, not one), should fold in cleanly and stop "
        f"alarming — measured ratios {[round(x, 2) for x in ratios_after]}; "
        f"if this fails, the fold-in is too weak to be useful even for "
        f"real-episode feedback volumes, which is a product-breaking bug, "
        f"not a safety feature")

    # 6. The control: a ONE-OFF loud window that was NEVER fed back must
    #    still be flagged after retrain — folding in a recurring pattern
    #    must not quietly widen the whole regime's tolerance for loud
    #    transients in general, only for the specific pattern reported.
    control_after = retrained_scorer.score(control_feats["vector"], control_op)
    assert control_after["anomalous"], (
        "an unrelated one-off transient that was never marked normal must "
        "still be flagged after retrain — otherwise the fold-in desensitised "
        "the regime broadly instead of learning the specific reported pattern"
    )
    print(f"\n[T3.4] episode ratio {ratio_before:.2f}x -> {ratio_after:.2f}x "
         f"after retrain; never-fed-back control stayed at "
         f"{control_after['score'] / control_after['threshold']:.2f}x "
         f"(anomalous={control_after['anomalous']}); contaminated="
         f"{summary['learn_period_contaminated']}")


def test_retrain_with_no_feedback_says_so_and_does_not_touch_the_file(tmp_path):
    """A customer who never clicks the button, or a `--retrain` run before
    any feedback exists, must not silently rewrite (or corrupt) the deployed
    baseline — DOC_STATUS.md's whole T3.7 justification is that a baseline
    change nobody meant to happen is a fleet-wide false-alarm risk."""
    out = tmp_path / "baseline.npz"
    db_path = tmp_path / "state.db"
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"), "--simulate",
         "--windows", "24", "--out", str(out), "--db", str(tmp_path / "init.db")],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    before_bytes = out.read_bytes()
    StateDB(db_path).close()          # exists, but nothing ever marked normal

    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"), "--retrain",
         "--out", str(out), "--db", str(db_path)],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "no feedback windows recorded" in r.stdout.lower()
    assert out.read_bytes() == before_bytes, (
        "a --retrain with nothing to fold in must leave the deployed "
        "baseline file byte-identical, not silently rewrite it")
