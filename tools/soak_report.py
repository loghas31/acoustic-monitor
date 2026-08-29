#!/usr/bin/env python3
"""
soak_report.py — turn a device's SQLite state DB into the Week-3 number.

THE NUMBER
----------
    false alarms per node-week

The execution plan's Gate 3 is <= 1. Nothing else in this project is worth as
much commercially: detection is a physics problem that the literature has
largely solved, whereas *not crying wolf on a healthy machine for a week* is
the thing that decides whether a customer keeps the sensor plugged in. It is
also, as the plan says, the number nobody in the student-project space ever
measures — so measure it properly, and report it honestly.

WHAT THIS TOOL DOES
-------------------
1. Reads `readings` (one row per 30 s window: ts, score, regime, threshold,
   anomalous) and `anomalies` from a device state DB (`firmware/state.py`).
2. REPLAYS the alert gate (`firmware/inference.py:AlertGate`) over the stored
   per-window anomaly flags to reconstruct the alert episodes.

   Why replay rather than read them? Because **the firmware never persists
   alerts.** `main.py` raises them to MQTT and a webhook and logs them, but
   nothing writes an alert row to SQLite, and the `AlertGate` lives in RAM, so
   a restart resets the streak. Replaying from `readings` is therefore the only
   auditable source of truth for "how many alerts would this week have
   produced", and it has the large bonus that we can re-run it at other
   settings of `persist_minutes` — which is what makes the tuning
   recommendation below possible. (See "Risks" in the generated report.)
3. Computes the headline rate, with an honest Poisson confidence bound and an
   explicit EXTRAPOLATED label when the run is shorter than a week.
4. Describes the score distribution against the threshold, per regime, and the
   *headroom* — how close healthy operation got to the line. Headroom is the
   leading indicator: a week with zero alerts but 2 % headroom is a week away
   from a bad month.
5. Reports regime occupancy over time: did the machine actually use the regimes
   the learn period found, and did it visit a regime we do not have?
6. Recommends a `persist_minutes` computed from the observed transient
   statistics, rather than guessed.
7. Writes markdown + PNGs.

USAGE
    python3 tools/soak_report.py --db /var/lib/acoustic-monitor/state.db \
        --outdir output/soak_2026-08 --persist-minutes 30

    python3 tools/soak_report.py --help

Every non-trivial calculation in this file is a module-level function with no
I/O, so `tests/test_soak.py` can check the arithmetic directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("soak_report")

SECONDS_PER_WEEK = 7 * 86400.0
DEFAULT_TARGET = 1.0          # Gate 3: <= 1 false alarm per node-week
# The gate's re-arm hysteresis. Hard-coded to 4 in inference.AlertGate's
# default; exposed here so the replay can be kept in sync if that changes.
DEFAULT_CLEAR_WINDOWS = 4


# ============================================================================
# 1. Loading
# ============================================================================

@dataclass
class Soak:
    """A loaded soak run. Everything downstream works on these arrays."""
    ts: np.ndarray            # (n,) unix seconds, ascending
    score: np.ndarray         # (n,) Mahalanobis distance
    regime: np.ndarray        # (n,) int regime id assigned at scoring time
    threshold: np.ndarray     # (n,) the threshold in force for that window
    anomalous: np.ndarray     # (n,) bool, as the device decided at the time
    window_s: float
    db_path: str = ""
    anomaly_rows: int = 0     # rows in the `anomalies` table (cross-check)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.ts)

    @property
    def span_s(self) -> float:
        """Wall-clock span first->last window, inclusive of the last window."""
        return float(self.ts[-1] - self.ts[0] + self.window_s) if self.n else 0.0

    @property
    def exposure_s(self) -> float:
        """Actual monitored time = number of windows x window length.

        This, not the wall-clock span, is the denominator of the false-alarm
        rate. If the node was off, or the SD card filled, or systemd restarted
        it 40 times, those minutes were not monitored and must not be counted
        as evidence of not-alarming. `coverage` below reports the difference.
        """
        return self.n * self.window_s

    @property
    def coverage(self) -> float:
        return self.exposure_s / self.span_s if self.span_s > 0 else 0.0


def infer_window_seconds(ts: np.ndarray, fallback: float = 30.0) -> float:
    """Median inter-window gap. Median, not mean: one 6-hour outage would drag
    a mean far from the truth, and the median is immune to a minority of gaps."""
    if len(ts) < 2:
        return fallback
    d = np.diff(np.sort(ts))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else fallback


def load_soak(db_path: Path, since: float | None = None,
              until: float | None = None,
              window_s: float | None = None) -> Soak:
    """Read a device state DB. Read-only: we open with mode=ro so that pointing
    this at a LIVE device DB can never corrupt an in-flight soak."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        q = ("SELECT ts, score, regime, threshold, anomalous FROM readings "
             "WHERE ts BETWEEN ? AND ? ORDER BY ts")
        rows = conn.execute(q, (since if since is not None else 0.0,
                                until if until is not None else 4e18)).fetchall()
        n_anom_rows = conn.execute(
            "SELECT COUNT(*) FROM anomalies WHERE ts BETWEEN ? AND ?",
            (since if since is not None else 0.0,
             until if until is not None else 4e18)).fetchone()[0]
    finally:
        conn.close()

    if not rows:
        raise SystemExit(f"no readings in {db_path} for the requested window")

    ts = np.array([r[0] for r in rows], dtype=float)
    soak = Soak(
        ts=ts,
        score=np.array([r[1] if r[1] is not None else np.nan for r in rows], float),
        regime=np.array([r[2] if r[2] is not None else -1 for r in rows], int),
        threshold=np.array([r[3] if r[3] is not None else np.nan for r in rows], float),
        anomalous=np.array([bool(r[4]) for r in rows], bool),
        window_s=window_s or infer_window_seconds(ts),
        db_path=str(db_path),
        anomaly_rows=int(n_anom_rows),
    )

    # -- integrity cross-checks. A soak report whose input is quietly corrupt
    #    is worse than no soak report, because it will be believed.
    n_flagged = int(soak.anomalous.sum())
    if n_anom_rows != n_flagged:
        soak.notes.append(
            f"`anomalies` table has {n_anom_rows} rows but `readings` flags "
            f"{n_flagged} anomalous windows. The most likely cause is the "
            f"7-day retention prune in state.py, which deletes old `readings` "
            f"but never prunes `anomalies` — so on a >=7-day run the readings "
            f"table silently loses its oldest day. Check `--since`/`--until` "
            f"and the retention_days setting.")
    if np.isnan(soak.score).any():
        soak.notes.append("some readings have a NULL score; they were kept as NaN.")
    return soak


# ============================================================================
# 2. Gaps / coverage
# ============================================================================

def find_gaps(ts: np.ndarray, window_s: float, tol: float = 1.5) -> list[dict]:
    """Intervals where the node stopped reporting for > tol windows.

    Every gap is a question for the runbook: did the Pi reboot, did the process
    OOM under the 350 MB MemoryMax, did the SD card wedge? A 7-day soak with 40
    gaps is not a passing soak even if it raised zero alerts, because a node
    that is not running cannot alert.
    """
    gaps = []
    if len(ts) < 2:
        return gaps
    d = np.diff(ts)
    for i in np.flatnonzero(d > tol * window_s):
        gaps.append({"from_ts": float(ts[i]), "to_ts": float(ts[i + 1]),
                     "seconds": float(d[i]),
                     "missed_windows": int(round(d[i] / window_s)) - 1})
    return gaps


# ============================================================================
# 3. Alert-gate replay  (the core of the false-alarm count)
# ============================================================================

def persist_minutes_to_windows(persist_minutes: float, window_s: float) -> int:
    """Same arithmetic main.py uses: round(persist_minutes*60/window_s)."""
    return max(1, int(round(persist_minutes * 60.0 / window_s)))


def replay_gate(anomalous, need: int, clear: int = DEFAULT_CLEAR_WINDOWS) -> list[dict]:
    """Re-implementation of `inference.AlertGate` over a whole run.

    Kept byte-for-byte faithful to the firmware semantics:
      * alert fires exactly once, on the window where the anomalous streak
        first reaches `need`;
      * the episode latches, and only re-arms after `clear` consecutive
        normal windows (hysteresis).
    `tests/test_soak.py` asserts this against the real AlertGate so the two can
    never silently diverge.

    Returns one dict per alert with the indices of the episode start, the
    firing window, and the window at which the episode cleared.
    """
    flags = [bool(a) for a in anomalous]
    need = max(1, int(need))
    alerts: list[dict] = []
    streak = normal_streak = 0
    in_episode = False
    for i, a in enumerate(flags):
        if a:
            streak += 1
            normal_streak = 0
            if not in_episode and streak >= need:
                in_episode = True
                alerts.append({"start_index": i - need + 1, "alert_index": i,
                               "clear_index": None})
        else:
            streak = 0
            normal_streak += 1
            if in_episode and normal_streak >= clear:
                in_episode = False
                if alerts:
                    alerts[-1]["clear_index"] = i
    return alerts


def excursion_runs(anomalous) -> list[int]:
    """Lengths (in windows) of every maximal run of consecutive anomalous
    windows. On a healthy machine these ARE the transient statistics: a door
    slam is a run of 1-3, a wash-down cycle a run of 20, a developing bearing
    fault a run that never ends. The persistence gate is a threshold on this
    quantity, so this distribution is exactly what you need to set it."""
    runs: list[int] = []
    cur = 0
    for a in anomalous:
        if a:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


# ============================================================================
# 4. Rate arithmetic + uncertainty
# ============================================================================

def false_alarms_per_node_week(n_alerts: int, exposure_s: float) -> float:
    """Linear rate. Simple, but the *labelling* around it is the whole point:
    3 alerts in 2 days is 10.5/node-week, and quoting "3" without the
    denominator is how projects fool themselves."""
    if exposure_s <= 0:
        return float("nan")
    return n_alerts * SECONDS_PER_WEEK / exposure_s


def poisson_upper_bound(k: int, exposure_weeks: float, conf: float = 0.95) -> float:
    """Exact (Garwood) upper confidence limit on a Poisson rate.

    Why this matters more than the point estimate: **zero alerts in three days
    is not evidence of <= 1 per week.** Under a true rate of 2/week you would
    still see zero in a 3-day window about 42 % of the time. The upper limit
    answers "what is the worst rate still consistent with what we saw", which
    is the number to put in front of an investor who can do arithmetic.

    Upper limit = chi2.ppf(conf, 2(k+1)) / 2 / exposure. Falls back to a closed
    form for k=0 (-ln(1-conf)/T) if scipy is unavailable.
    """
    if exposure_weeks <= 0:
        return float("nan")
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(conf, 2 * (k + 1)) / 2.0 / exposure_weeks)
    except Exception:                                        # noqa: BLE001
        if k == 0:
            return float(-math.log(1.0 - conf) / exposure_weeks)
        return float((k + 1.96 * math.sqrt(k + 1) + 1) / exposure_weeks)


def rate_summary(n_alerts: int, exposure_s: float, target: float = DEFAULT_TARGET,
                 conf: float = 0.95) -> dict:
    """The headline block. `extrapolated` is True whenever the run is shorter
    than a node-week, and the report must say so in the same sentence as the
    number."""
    weeks = exposure_s / SECONDS_PER_WEEK
    rate = false_alarms_per_node_week(n_alerts, exposure_s)
    upper = poisson_upper_bound(n_alerts, weeks, conf)
    return {
        "alerts": int(n_alerts),
        "exposure_days": exposure_s / 86400.0,
        "exposure_weeks": weeks,
        "rate_per_node_week": rate,
        "extrapolated": weeks < 1.0,
        "upper_bound_conf": conf,
        "rate_upper_bound": upper,
        "target": target,
        # A run passes on the point estimate; it passes CONVINCINGLY only when
        # even the upper confidence limit is under target. Report both.
        "passes_point": bool(rate <= target),
        "passes_upper": bool(upper <= target),
    }


# ============================================================================
# 5. Score distribution and headroom
# ============================================================================

def headroom_stats(score: np.ndarray, threshold: np.ndarray) -> dict:
    """How close did healthy operation get to the line?

    We work in the ratio r = score / threshold, because thresholds differ per
    regime and comparing raw distances across regimes is meaningless.

      r < 1   : below threshold (normal)
      r = 1   : exactly at the line
      r > 1   : an anomalous window

    `headroom_pct` = 100*(1 - p99(r)) — the percentage margin at the 99th
    percentile of ordinary operation. This is the leading indicator for future
    false alarms: the count of alerts is a lagging, discrete, high-variance
    statistic (0 or 1 events per week), whereas headroom is continuous and
    estimated from thousands of windows. A node at 40 % headroom is safe; a
    node at 2 % headroom will alarm the first time the weather changes, and it
    will look fine in this week's alert count right up until it doesn't.
    """
    ok = np.isfinite(score) & np.isfinite(threshold) & (threshold > 0)
    if not ok.any():
        return {"n": 0}
    r = score[ok] / threshold[ok]
    p = lambda q: float(np.percentile(r, q))                          # noqa: E731
    return {
        "n": int(ok.sum()),
        "ratio_p50": p(50), "ratio_p95": p(95), "ratio_p99": p(99),
        "ratio_p999": p(99.9), "ratio_max": float(r.max()),
        "score_p50": float(np.percentile(score[ok], 50)),
        "score_p99": float(np.percentile(score[ok], 99)),
        "score_max": float(score[ok].max()),
        "threshold_median": float(np.median(threshold[ok])),
        "exceed_fraction": float((r > 1.0).mean()),
        "headroom_pct": 100.0 * (1.0 - p(99)),
        "worst_headroom_pct": 100.0 * (1.0 - float(r.max())),
    }


def per_regime_stats(soak: Soak) -> dict[int, dict]:
    """Score distribution vs threshold, split by the regime the device assigned.

    Split matters. A machine with a rarely-used regime can look fine in
    aggregate while that one regime sits permanently at r = 0.98 — and the
    aggregate will not tell you, because the regime is 3 % of the windows.
    """
    out: dict[int, dict] = {}
    for r in sorted(set(int(x) for x in soak.regime)):
        m = soak.regime == r
        st = headroom_stats(soak.score[m], soak.threshold[m])
        st.update({
            "regime": int(r),
            "windows": int(m.sum()),
            "occupancy": float(m.mean()),
            "hours": float(m.sum() * soak.window_s / 3600.0),
            "anomalous_windows": int(soak.anomalous[m].sum()),
        })
        out[int(r)] = st
    return out


def regime_occupancy_series(soak: Soak, bin_hours: float = 1.0) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Fraction of each time bin spent in each regime.

    "Did the machine actually use the regimes we learned?" is answered here.
    Two failure modes to look for in the plot:
      * a learned regime with ~0 occupancy -> the learn period contained a
        transient state (a warm-up, a door left open) that got promoted to a
        regime. Harmless but wasteful, and it inflates k.
      * occupancy that changes shape partway through the week -> the machine
        did something new. If that coincides with the alerts, you have your
        explanation, and the fix is a retrain, not a bigger persist_minutes.
    """
    if soak.n == 0:
        return np.array([]), {}
    t0 = soak.ts[0]
    bin_s = bin_hours * 3600.0
    idx = ((soak.ts - t0) / bin_s).astype(int)
    n_bins = int(idx.max()) + 1
    centres = t0 + (np.arange(n_bins) + 0.5) * bin_s
    regs = sorted(set(int(x) for x in soak.regime))
    series = {r: np.zeros(n_bins) for r in regs}
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    for r in regs:
        series[r] = np.bincount(idx[soak.regime == r], minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        for r in regs:
            series[r] = np.where(counts > 0, series[r] / np.maximum(counts, 1), np.nan)
    return centres, series


# ============================================================================
# 6. persist_minutes recommendation  (the "tell me how to tune it" bit)
# ============================================================================

def sweep_persist(anomalous, window_s: float, exposure_s: float,
                  candidates_minutes, clear: int = DEFAULT_CLEAR_WINDOWS,
                  target: float = DEFAULT_TARGET, conf: float = 0.95) -> list[dict]:
    """Replay the gate at each candidate persist_minutes.

    This is a genuine what-if, not a model: the same anomaly flags, the same
    gate code, a different constant. The cost of a larger `persist_minutes` is
    detection latency (you wait that long before the first alert on a real
    fault), so the tool reports the whole curve rather than only the answer —
    students should see the shape of the trade they are making.
    """
    rows = []
    for pm in candidates_minutes:
        need = persist_minutes_to_windows(pm, window_s)
        alerts = replay_gate(anomalous, need, clear)
        s = rate_summary(len(alerts), exposure_s, target, conf)
        rows.append({"persist_minutes": float(pm), "need_windows": need,
                     "alerts": len(alerts),
                     "rate_per_node_week": s["rate_per_node_week"],
                     "rate_upper_bound": s["rate_upper_bound"],
                     "passes_point": s["passes_point"],
                     "passes_upper": s["passes_upper"]})
    return rows


def geometric_tail_persist(runs: list[int], exposure_s: float, window_s: float,
                           target: float = DEFAULT_TARGET) -> dict:
    """Extrapolate BEYOND the longest excursion actually observed.

    The empirical sweep can only tell you about excursion lengths present in
    the data. If the longest healthy excursion in the week was 7 windows, the
    sweep says "4 minutes is enough" — but you only sampled ~a dozen
    transients, and the 50th transient will be longer than any of the first 12.

    Model: treat an excursion, once started, as ending with constant
    probability per window (a geometric run length). Then
        P(run >= L) = q^(L-1),  q = 1 - 1/mean_run_length,
    and the expected number of gate-firing excursions per week is
        rate(L) = runs_per_week * q^(L-1).
    Solve rate(L) <= target for L.

    The geometric assumption is optimistic-to-fair: real acoustic excursions
    are somewhat heavier-tailed than geometric (a cleaning cycle is not a coin
    flip), so treat the answer as a floor, not a ceiling. It is stated here
    rather than hidden because the students will be asked in a viva why they
    chose the number they chose, and "the tool computed it from a geometric
    run-length model fitted to 41 observed excursions" is an answer.
    """
    weeks = exposure_s / SECONDS_PER_WEEK
    if not runs or weeks <= 0:
        return {"applicable": False, "reason": "no anomalous excursions observed",
                "n_runs": 0, "runs_per_week": 0.0,
                "mean_run_windows": 0.0, "max_run_windows": 0,
                "need_windows": 1, "persist_minutes": window_s / 60.0}
    mean_run = float(statistics.fmean(runs))
    runs_per_week = len(runs) / weeks
    # Degenerate case: every excursion is exactly 1 window -> q = 0, so any
    # need >= 2 gives a predicted rate of 0. Clamp q away from 0 and 1 so the
    # logarithm below is finite and the answer stays conservative.
    q = min(max(1.0 - 1.0 / mean_run, 1e-6), 1.0 - 1e-9)
    if runs_per_week <= target:
        need = 1
    else:
        # runs_per_week * q^(L-1) <= target
        need = int(math.ceil(1.0 + math.log(target / runs_per_week) / math.log(q)))
    need = max(1, need)
    return {
        "applicable": True,
        "n_runs": len(runs),
        "runs_per_week": runs_per_week,
        "mean_run_windows": mean_run,
        "median_run_windows": float(statistics.median(runs)),
        "p95_run_windows": float(np.percentile(runs, 95)),
        "max_run_windows": int(max(runs)),
        "q_continue": q,
        "need_windows": need,
        "persist_minutes": need * window_s / 60.0,
        "predicted_rate_at_need": runs_per_week * q ** (need - 1),
    }


def recommend_persist_minutes(soak: Soak, current_minutes: float,
                              clear: int = DEFAULT_CLEAR_WINDOWS,
                              target: float = DEFAULT_TARGET,
                              max_minutes: float = 120.0,
                              candidates=None) -> dict:
    """Combine the empirical sweep and the tail model into one recommendation.

    Decision rule, in words:
      * Take the smallest candidate whose *upper confidence bound* meets the
        target (not merely the point estimate — see poisson_upper_bound).
      * Take the tail-model answer, which can see past the observed maximum.
      * Recommend the LARGER of the two, rounded up to a sane number of
        minutes, then add one window of margin.
      * If that exceeds `max_minutes`, refuse to recommend it and say so: at
        two hours of persistence you are no longer running a condition monitor,
        you are running a very slow smoke alarm. The real fix at that point is
        a better baseline (missing regime -> retrain), and the regime-occupancy
        section of this report is where you look for it.
    """
    if candidates is None:
        # Geometric-ish ladder in minutes; includes the config default (30) and
        # the demo value (2) so the report always covers what students run.
        candidates = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120]
        if current_minutes not in candidates:
            candidates = sorted(set(candidates + [float(current_minutes)]))

    sweep = sweep_persist(soak.anomalous, soak.window_s, soak.exposure_s,
                          candidates, clear, target)
    runs = excursion_runs(soak.anomalous)
    tail = geometric_tail_persist(runs, soak.exposure_s, soak.window_s, target)

    empirical = next((r for r in sweep if r["passes_upper"]), None)
    empirical_point = next((r for r in sweep if r["passes_point"]), None)

    emp_min = empirical["persist_minutes"] if empirical else None
    tail_min = tail["persist_minutes"] if tail.get("applicable") else 0.0

    if emp_min is None and not tail.get("applicable"):
        rec, reason = current_minutes, "no anomalous windows at all — nothing to tune."
    elif emp_min is None:
        rec = tail_min
        reason = ("no candidate up to %g min met the target on the empirical "
                  "replay; the tail model's answer is shown but is not "
                  "supported by the data." % max(candidates))
    else:
        rec = max(emp_min, tail_min)
        reason = ("larger of the empirical sweep (%g min, smallest setting whose "
                  "95%% upper bound meets the target) and the geometric "
                  "run-length model (%g min)." % (emp_min, tail_min))

    # One window of margin: the recommendation sits on a step boundary, and a
    # single extra anomalous window in a future transient should not flip it.
    margin_min = soak.window_s / 60.0
    rec_padded = float(rec) + margin_min

    feasible = rec_padded <= max_minutes
    return {
        "current_minutes": float(current_minutes),
        "recommended_minutes": round(rec_padded, 2),
        "recommended_windows": persist_minutes_to_windows(rec_padded, soak.window_s),
        "feasible": feasible,
        "max_minutes": max_minutes,
        "reason": reason,
        "empirical_minutes": emp_min,
        "empirical_point_minutes": (empirical_point["persist_minutes"]
                                    if empirical_point else None),
        "tail_model": tail,
        "sweep": sweep,
        "detection_latency_minutes": round(rec_padded, 2),
        "runs": runs,
    }


# ============================================================================
# 7. Plots
# ============================================================================

def _fmt_time_axis(ax, ts: np.ndarray) -> None:
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    span_h = (ts[-1] - ts[0]) / 3600.0 if len(ts) > 1 else 1.0
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_h / 8) or 1)))


def _to_dates(ts: np.ndarray):
    return [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]


def plot_timeline(soak: Soak, alerts: list[dict], out: Path) -> Path:
    """Score vs time, with the threshold, the anomalous windows, the alert
    episodes, and a regime strip. This is the plot that goes in RESULTS.md and
    in the deck: one glance says "flat, well under the line, with these
    excursions"."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = _to_dates(soak.ts)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})

    ax.plot(dates, soak.score, lw=0.7, color="#2b6cb0", label="Mahalanobis score")
    ax.plot(dates, soak.threshold, lw=1.2, color="#c53030", ls="--",
            label="threshold (per regime)")
    m = soak.anomalous
    if m.any():
        ax.scatter(np.array(dates)[m], soak.score[m], s=9, color="#c53030",
                   zorder=3, label=f"anomalous windows (n={int(m.sum())})")
    for j, a in enumerate(alerts):
        i0 = a["start_index"]
        i1 = a["clear_index"] if a["clear_index"] is not None else soak.n - 1
        ax.axvspan(dates[i0], dates[min(i1, soak.n - 1)], color="#f6ad55", alpha=0.45,
                   label="alert episode" if j == 0 else None)
        ax.annotate(f"ALERT {j+1}", (dates[a["alert_index"]], soak.score[a["alert_index"]]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=8, color="#7b341e")
    ax.set_ylabel("score")
    ax.set_title(f"Soak timeline — {soak.n} windows, {soak.exposure_s/86400:.2f} days, "
                 f"{len(alerts)} alert(s)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)

    axr.plot(dates, soak.regime, drawstyle="steps-post", lw=0.8, color="#2f855a")
    axr.set_ylabel("regime")
    axr.set_yticks(sorted(set(int(x) for x in soak.regime)))
    axr.grid(alpha=0.25)
    _fmt_time_axis(axr, soak.ts)
    fig.autofmt_xdate()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distributions(soak: Soak, stats: dict[int, dict], out: Path) -> Path:
    """Per-regime score histograms against their own thresholds, plus the
    pooled score/threshold ratio. The ratio panel is the honest cross-regime
    comparison; the raw panels are what you show someone who wants to see the
    actual distances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regs = sorted(stats)
    ncol = len(regs) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regs):
        m = soak.regime == r
        s = soak.score[m]
        thr = stats[r]["threshold_median"]
        ax.hist(s, bins=60, color="#2b6cb0", alpha=0.8)
        ax.axvline(thr, color="#c53030", ls="--", lw=1.5, label=f"threshold {thr:.1f}")
        ax.axvline(stats[r]["score_p99"], color="#2f855a", ls=":", lw=1.4,
                   label=f"p99 {stats[r]['score_p99']:.1f}")
        ax.set_title(f"regime {r} — {stats[r]['windows']} w "
                     f"({stats[r]['occupancy']*100:.0f} %)\n"
                     f"headroom {stats[r]['headroom_pct']:.0f} %", fontsize=9)
        ax.set_xlabel("Mahalanobis score")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    axf = axes[-1]
    ratio = soak.score / np.where(soak.threshold > 0, soak.threshold, np.nan)
    axf.hist(ratio[np.isfinite(ratio)], bins=80, color="#805ad5", alpha=0.85)
    axf.axvline(1.0, color="#c53030", ls="--", lw=1.5, label="threshold")
    axf.set_title("all regimes: score / threshold", fontsize=9)
    axf.set_xlabel("ratio (1.0 = at the line)")
    axf.legend(fontsize=7)
    axf.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regime_occupancy(soak: Soak, out: Path, bin_hours: float = 1.0) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres, series = regime_occupancy_series(soak, bin_hours)
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 3.8),
                                  gridspec_kw={"width_ratios": [3, 1]})
    if len(centres):
        dates = _to_dates(centres)
        regs = sorted(series)
        vals = np.vstack([np.nan_to_num(series[r]) for r in regs])
        ax.stackplot(dates, vals, labels=[f"regime {r}" for r in regs], alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"fraction of each {bin_hours:g} h bin")
        ax.set_title("Regime occupancy over time — did the machine use what we learned?")
        ax.legend(loc="upper right", fontsize=8, ncol=len(regs))
        _fmt_time_axis(ax, centres)
        fig.autofmt_xdate()

        occ = [float((soak.regime == r).mean()) for r in regs]
        axb.bar([str(r) for r in regs], occ, color="#2f855a")
        axb.set_ylim(0, 1)
        axb.set_xlabel("regime")
        axb.set_title("overall occupancy", fontsize=9)
        for i, v in enumerate(occ):
            axb.text(i, v + 0.02, f"{v*100:.0f} %", ha="center", fontsize=8)
        axb.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_persist_sweep(rec: dict, target: float, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sw = rec["sweep"]
    x = [r["persist_minutes"] for r in sw]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, [r["rate_per_node_week"] for r in sw], "o-", color="#2b6cb0",
            label="observed rate")
    ax.plot(x, [r["rate_upper_bound"] for r in sw], "s--", color="#805ad5",
            label="95 % upper bound")
    ax.axhline(target, color="#c53030", ls="--", label=f"Gate 3 target ({target:g}/node-week)")
    ax.axvline(rec["current_minutes"], color="#718096", ls=":", label="current setting")
    if rec["feasible"]:
        ax.axvline(rec["recommended_minutes"], color="#2f855a", lw=2,
                   label=f"recommended {rec['recommended_minutes']:g} min")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
    ax.set_xlabel("persist_minutes (also = detection latency on a real fault)")
    ax.set_ylabel("false alarms per node-week")
    ax.set_title("Cost of the persistence gate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
# 8. Report rendering
# ============================================================================

def _t(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _verdict(summary: dict) -> str:
    if summary["passes_upper"]:
        return "**PASS** — even the 95 % upper confidence limit is within the gate."
    if summary["passes_point"]:
        return ("**PASS (point estimate only)** — the observed rate meets the gate, "
                "but the run is too short to rule out a worse true rate. "
                "Keep soaking.")
    return "**FAIL** — the observed rate is above the gate."


def render_markdown(soak: Soak, summary: dict, alerts: list[dict],
                    stats: dict[int, dict], overall: dict, rec: dict,
                    gaps: list[dict], figs: dict, persist_minutes: float,
                    clear: int) -> str:
    L: list[str] = []
    a = L.append
    a("# Soak report — false alarms per node-week")
    a("")
    a(f"*Generated {_t(datetime.now(tz=timezone.utc).timestamp())} from "
      f"`{soak.db_path}`.*")
    a("")
    a("Every alert counted here is assumed to be a FALSE alarm: the whole point "
      "of a soak is to run on a machine known to be healthy (Gate 3 of "
      "the execution plan (not in this public copy)). If the machine was not healthy for the whole "
      "run, this report is invalid — say so and re-run.")
    a("")

    # ---- headline ----
    a("## 1. Headline")
    a("")
    ext = (" **(EXTRAPOLATED — the run is shorter than a week)**"
           if summary["extrapolated"] else "")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| **False alarms per node-week** | **{summary['rate_per_node_week']:.2f}**{ext} |")
    a(f"| Alerts actually raised | {summary['alerts']} |")
    a(f"| Monitored exposure | {summary['exposure_days']:.2f} days "
      f"({summary['exposure_weeks']:.3f} node-weeks, {soak.n} windows) |")
    a(f"| 95 % upper confidence limit | {summary['rate_upper_bound']:.2f} per node-week |")
    a(f"| Gate 3 target | ≤ {summary['target']:g} per node-week |")
    a(f"| Verdict | {_verdict(summary)} |")
    a("")
    if summary["extrapolated"]:
        a(f"> The point estimate is `{summary['alerts']} alerts × 7 days ÷ "
          f"{summary['exposure_days']:.2f} days`. Linear extrapolation from a "
          f"short run is a **weak** claim — with {summary['alerts']} events the "
          f"sampling noise dominates. The upper confidence limit "
          f"({summary['rate_upper_bound']:.2f}) is the number to quote until the "
          f"full 7 days are in.")
        a("")
    elif summary["alerts"] == 0:
        a(f"> Zero alerts over {summary['exposure_days']:.1f} days. That is a "
          f"result, but not proof of zero: the data are consistent with any true "
          f"rate up to {summary['rate_upper_bound']:.2f} per node-week at 95 % "
          f"confidence. Headroom (§4) is the better early-warning signal.")
        a("")

    a(f"Gate replayed with `persist_minutes = {persist_minutes:g}` "
      f"(= {persist_minutes_to_windows(persist_minutes, soak.window_s)} consecutive "
      f"windows of {soak.window_s:g} s) and `clear = {clear}` windows of "
      f"hysteresis, matching `firmware/inference.py:AlertGate`.")
    a("")

    # ---- run integrity ----
    a("## 2. Run integrity")
    a("")
    a(f"- Window length (inferred): **{soak.window_s:g} s**")
    a(f"- First window: {_t(soak.ts[0])}")
    a(f"- Last window: {_t(soak.ts[-1])}")
    a(f"- Wall-clock span: {soak.span_s/86400:.2f} days")
    a(f"- Coverage (monitored ÷ span): **{soak.coverage*100:.1f} %**")
    a(f"- Gaps longer than 1.5 windows: **{len(gaps)}**")
    a("")
    if gaps:
        a("| from | to | missing windows | minutes lost |")
        a("|---|---|---|---|")
        for g in sorted(gaps, key=lambda g: -g["seconds"])[:10]:
            a(f"| {_t(g['from_ts'])} | {_t(g['to_ts'])} | {g['missed_windows']} | "
              f"{g['seconds']/60:.1f} |")
        a("")
        a("> Each gap is a node that was not listening. Investigate before "
          "trusting the alert count: `journalctl -u acoustic-monitor --since ...` "
          "(see the operations runbook (not in this public copy) §Collecting logs). A restart also resets the "
          "in-RAM alert streak, so a node that reboots every 20 minutes can "
          "*never* fire a 30-minute persistence gate.")
        a("")
    for n in soak.notes:
        a(f"> ⚠ {n}")
        a("")

    # ---- alerts ----
    a("## 3. Alert episodes")
    a("")
    if not alerts:
        a("None. ")
    else:
        a("| # | episode start | alert fired | cleared | peak score | threshold | peak ratio |")
        a("|---|---|---|---|---|---|---|")
        for i, al in enumerate(alerts, 1):
            i0, i1 = al["start_index"], (al["clear_index"] or soak.n) - 1
            sl = slice(i0, max(i0 + 1, i1 + 1))
            peak = float(np.nanmax(soak.score[sl]))
            thr = float(np.nanmedian(soak.threshold[sl]))
            a(f"| {i} | {_t(soak.ts[i0])} | {_t(soak.ts[al['alert_index']])} | "
              f"{_t(soak.ts[min(i1, soak.n-1)])} | {peak:.1f} | {thr:.1f} | "
              f"{peak/thr:.2f} |")
    a("")
    a(f"Per-window anomalous rate: **{soak.anomalous.mean()*100:.2f} %** "
      f"({int(soak.anomalous.sum())} of {soak.n} windows). ")
    a("This is the quantity the persistence gate converts into an alert count; "
      "it is high-variance and by itself says nothing about customer experience.")
    a("")

    # ---- distribution + headroom ----
    a("## 4. Score distribution, per regime, and headroom")
    a("")
    a("`ratio = score / threshold`; 1.0 is exactly at the line. "
      "`headroom` = 100 × (1 − p99 ratio): the margin at the 99th percentile of "
      "ordinary operation.")
    a("")
    a("| regime | windows | occupancy | hours | p50 | p95 | p99 | max | above thr | headroom |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r, st in sorted(stats.items()):
        a(f"| {r} | {st['windows']} | {st['occupancy']*100:.1f} % | {st['hours']:.1f} | "
          f"{st['ratio_p50']:.2f} | {st['ratio_p95']:.2f} | {st['ratio_p99']:.2f} | "
          f"{st['ratio_max']:.2f} | {st['exceed_fraction']*100:.2f} % | "
          f"{st['headroom_pct']:.1f} % |")
    a(f"| **all** | {overall['n']} | 100 % | {soak.exposure_s/3600:.1f} | "
      f"{overall['ratio_p50']:.2f} | {overall['ratio_p95']:.2f} | "
      f"{overall['ratio_p99']:.2f} | {overall['ratio_max']:.2f} | "
      f"{overall['exceed_fraction']*100:.2f} % | {overall['headroom_pct']:.1f} % |")
    a("")
    worst = min(stats.items(), key=lambda kv: kv[1]["headroom_pct"]) if stats else None
    if worst:
        r, st = worst
        a(f"**Tightest regime: {r}**, headroom {st['headroom_pct']:.1f} % at p99, "
          f"worst single window {st['ratio_max']:.2f}× threshold.")
        if st["headroom_pct"] < 10:
            a("")
            a("> ⚠ Under 10 % headroom. The alert count this week is not the "
              "risk; the risk is that this regime is one warm afternoon away "
              "from sitting above the line. Either the learn period undersampled "
              "this regime (check its window count above — a regime learned from "
              "<20 windows has a badly estimated covariance) or the machine has "
              "drifted since. Retrain before the pilot, not during it.")
    a("")
    if figs.get("distributions"):
        a(f"![score distributions]({figs['distributions']})")
        a("")

    # ---- regimes ----
    a("## 5. Regime occupancy — did the machine use what we learned?")
    a("")
    rare = [r for r, st in stats.items() if st["occupancy"] < 0.02]
    a(f"- Regimes observed during the soak: {sorted(stats)}")
    if rare:
        a(f"- ⚠ Regimes with <2 % occupancy: {rare}. Either the learn period "
          f"promoted a transient state to a regime (harmless, but it means k is "
          f"inflated and that regime's Gaussian is fitted on very few windows), "
          f"or the machine only rarely does something you barely learned — which "
          f"is exactly where false alarms come from.")
    a("- A regime whose occupancy *pattern changes* partway through the run is "
      "a machine that started doing something new. If your alerts cluster there, "
      "the fix is a retrain (`baseline.py --retrain`), not a longer gate.")
    a("")
    if figs.get("occupancy"):
        a(f"![regime occupancy]({figs['occupancy']})")
        a("")

    # ---- persist recommendation ----
    a("## 6. Recommended `persist_minutes` — computed from your data")
    a("")
    t = rec["tail_model"]
    a(f"**Current setting: {rec['current_minutes']:g} min.**")
    if rec["feasible"]:
        a(f"**Recommendation: `persist_minutes: {rec['recommended_minutes']:g}` "
          f"({rec['recommended_windows']} windows).**")
    else:
        a(f"**No usable recommendation ≤ {rec['max_minutes']:g} min "
          f"(the arithmetic asks for {rec['recommended_minutes']:g} min).**")
        a("")
        a("> A persistence gate this long is not a condition monitor, it is a "
          "slow smoke alarm — and it delays the detection of a *real* fault by "
          "the same amount. When the gate cannot be tuned to the target, the "
          "baseline is the problem: look at §5 for a regime the learn period "
          "missed, then extend the learn period or retrain with feedback.")
    a("")
    a(f"Basis: {rec['reason']}")
    a("")
    if t.get("applicable"):
        a("Observed transient (excursion) statistics — consecutive anomalous windows:")
        a("")
        a(f"- excursions: **{t['n_runs']}** ({t['runs_per_week']:.1f} per node-week)")
        a(f"- length: median {t['median_run_windows']:.0f} w, mean "
          f"{t['mean_run_windows']:.1f} w, p95 {t['p95_run_windows']:.0f} w, "
          f"max **{t['max_run_windows']} w** "
          f"(= {t['max_run_windows']*soak.window_s/60:.1f} min)")
        a(f"- fitted geometric continuation probability q = {t['q_continue']:.3f}; "
          f"predicted rate at the recommended gate: "
          f"{t['predicted_rate_at_need']:.3f} per node-week")
        a("")
        a("> The tail model exists because the empirical sweep can only see "
          "excursions that happened. You sampled "
          f"{t['n_runs']} of them; the next one may be longer than all of them. "
          "The model is geometric (constant per-window probability of "
          "continuing), which is mildly optimistic for real acoustic events — "
          "treat its answer as a floor.")
        a("")
    a("| persist_minutes | windows | alerts in this run | per node-week | 95 % upper | meets target |")
    a("|---|---|---|---|---|---|")
    for row in rec["sweep"]:
        mark = "✅" if row["passes_upper"] else ("~" if row["passes_point"] else "❌")
        a(f"| {row['persist_minutes']:g} | {row['need_windows']} | {row['alerts']} | "
          f"{row['rate_per_node_week']:.2f} | {row['rate_upper_bound']:.2f} | {mark} |")
    a("")
    a("`✅` = 95 % upper bound meets the target; `~` = point estimate only; "
      "`❌` = fails. Remember the cost column that is not in this table: "
      "**detection latency**. A 60-minute gate means a real bearing fault is "
      "reported an hour after it becomes detectable. That is fine for bearings "
      "(they degrade over days) and unacceptable for, say, a dry-running pump.")
    a("")
    if figs.get("sweep"):
        a(f"![persist sweep]({figs['sweep']})")
        a("")

    # ---- timeline ----
    a("## 7. Timeline")
    a("")
    if figs.get("timeline"):
        a(f"![timeline]({figs['timeline']})")
        a("")

    # ---- what to do ----
    a("## 8. What to do with this")
    a("")
    a("1. If the verdict is PASS: put the headline number, the upper bound and "
      "the timeline figure in `RESULTS.md` and in the deck. Quote it as "
      "\"N false alarms in a X-day soak on a healthy <machine>, 95 % upper "
      "bound Y per node-week\" — the qualifiers are what make it credible.")
    a("2. If it is FAIL: work down this list *in order*, because the first two "
      "are free and the third costs sensitivity —")
    a("   1. **Missing regime** (§5): retrain with the soak data, or extend the "
      "learn period and refit. `python firmware/baseline.py --retrain`.")
    a("   2. **Feedback**: mark the alert episodes \"this was normal\" in the "
      "dashboard and retrain; that is what the feedback loop is for.")
    a("   3. **Only then** raise `persist_minutes` to §6's recommendation.")
    a("3. Either way, record the headroom number. Next month's soak against "
      "this month's headroom is how you find drift before a customer does.")
    a("")
    return "\n".join(L)


# ============================================================================
# 9. CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soak_report.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, required=True,
                   help="device state DB (firmware/state.py schema). Opened read-only.")
    p.add_argument("--outdir", type=Path, default=Path("output/soak"),
                   help="directory for soak_report.md + PNGs")
    p.add_argument("--persist-minutes", type=float, default=None,
                   help="gate setting to evaluate. Default: read from --config, "
                        "else 30 (config.yaml default).")
    p.add_argument("--config", type=Path, default=None,
                   help="firmware/config.yaml, to read persist_minutes and window seconds")
    p.add_argument("--clear-windows", type=int, default=DEFAULT_CLEAR_WINDOWS,
                   help="AlertGate re-arm hysteresis (must match the firmware)")
    p.add_argument("--window-seconds", type=float, default=None,
                   help="override the inferred window length")
    p.add_argument("--target", type=float, default=DEFAULT_TARGET,
                   help="false alarms per node-week gate (Gate 3 = 1.0)")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="confidence level for the Poisson upper bound")
    p.add_argument("--max-persist-minutes", type=float, default=120.0,
                   help="refuse to recommend a gate longer than this")
    p.add_argument("--since", type=float, default=None, help="unix ts lower bound")
    p.add_argument("--until", type=float, default=None, help="unix ts upper bound")
    p.add_argument("--bin-hours", type=float, default=1.0,
                   help="bin width for the regime-occupancy plot")
    p.add_argument("--no-plots", action="store_true",
                   help="markdown + JSON only (no matplotlib)")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the full result as JSON (default: <outdir>/soak_report.json)")
    return p


def analyse(soak: Soak, persist_minutes: float, clear: int, target: float,
            conf: float, max_persist: float) -> dict:
    """Everything except I/O and plots. Returns a plain dict so it can be
    JSON-dumped, diffed between runs, and asserted on in tests."""
    need = persist_minutes_to_windows(persist_minutes, soak.window_s)
    alerts = replay_gate(soak.anomalous, need, clear)
    summary = rate_summary(len(alerts), soak.exposure_s, target, conf)
    stats = per_regime_stats(soak)
    overall = headroom_stats(soak.score, soak.threshold)
    rec = recommend_persist_minutes(soak, persist_minutes, clear, target, max_persist)
    gaps = find_gaps(soak.ts, soak.window_s)
    return {"need_windows": need, "alerts": alerts, "summary": summary,
            "per_regime": stats, "overall": overall, "recommendation": rec,
            "gaps": gaps}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    persist = args.persist_minutes
    window_s = args.window_seconds
    if args.config and args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        persist = persist if persist is not None else cfg["anomaly"]["persist_minutes"]
        window_s = window_s or float(cfg["window"]["seconds"])
    if persist is None:
        persist = 30.0

    soak = load_soak(args.db, args.since, args.until, window_s)
    log.info("loaded %d windows (%.2f days exposure, window %.0f s)",
             soak.n, soak.exposure_s / 86400, soak.window_s)

    res = analyse(soak, persist, args.clear_windows, args.target,
                  args.confidence, args.max_persist_minutes)

    args.outdir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, str] = {}
    if not args.no_plots:
        figs["timeline"] = plot_timeline(soak, res["alerts"],
                                         args.outdir / "soak_timeline.png").name
        figs["distributions"] = plot_distributions(
            soak, res["per_regime"], args.outdir / "soak_distributions.png").name
        figs["occupancy"] = plot_regime_occupancy(
            soak, args.outdir / "soak_regimes.png", args.bin_hours).name
        figs["sweep"] = plot_persist_sweep(
            res["recommendation"], args.target,
            args.outdir / "soak_persist_sweep.png").name

    md = render_markdown(soak, res["summary"], res["alerts"], res["per_regime"],
                         res["overall"], res["recommendation"], res["gaps"],
                         figs, persist, args.clear_windows)
    md_path = args.outdir / "soak_report.md"
    md_path.write_text(md)

    json_path = args.json or (args.outdir / "soak_report.json")
    payload = {
        "db": str(args.db),
        "window_seconds": soak.window_s,
        "n_windows": soak.n,
        "exposure_days": soak.exposure_s / 86400.0,
        "coverage": soak.coverage,
        "persist_minutes_evaluated": persist,
        "summary": res["summary"],
        "per_regime": {str(k): v for k, v in res["per_regime"].items()},
        "overall": res["overall"],
        "gaps": res["gaps"],
        "notes": soak.notes,
        "recommendation": {k: v for k, v in res["recommendation"].items()
                           if k != "runs"},
        "excursion_run_lengths": res["recommendation"]["runs"],
        "alerts": res["alerts"],
        "figures": figs,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float))

    s = res["summary"]
    print(f"\n{'='*70}")
    print(f"FALSE ALARMS PER NODE-WEEK : {s['rate_per_node_week']:.2f}"
          + ("   (EXTRAPOLATED)" if s["extrapolated"] else ""))
    print(f"  alerts / exposure        : {s['alerts']} in {s['exposure_days']:.2f} days")
    print(f"  95% upper bound          : {s['rate_upper_bound']:.2f}")
    print(f"  Gate 3 (<= {s['target']:g})           : "
          f"{'PASS' if s['passes_upper'] else ('PASS (point only)' if s['passes_point'] else 'FAIL')}")
    print(f"  headroom @ p99           : {res['overall']['headroom_pct']:.1f} %")
    print(f"  recommended persist_min  : {res['recommendation']['recommended_minutes']:g}"
          f" (current {persist:g})"
          + ("" if res["recommendation"]["feasible"] else "   [NOT FEASIBLE — fix the baseline]"))
    print(f"{'='*70}")
    print(f"report : {md_path}")
    print(f"json   : {json_path}")
    for f in figs.values():
        print(f"figure : {args.outdir / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
