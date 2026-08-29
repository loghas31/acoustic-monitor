# Alerting — how we avoid crying wolf

Companion to the system overview (not in this public copy) §5.
Code: `firmware/inference.py` (`AlertGate`), `firmware/main.py`,
`backend/alerts.py`, `tools/soak_report.py`.

---

## The commercial premise

A customer who receives three false alarms stops believing the box and
eventually unplugs it. An unplugged box is a cancelled subscription. So the
governing metric of this product is not sensitivity — it is

> **false alarms per node-week**, target ≤ 1

Detection is easy to demo; trust is what you actually sell.

## Defence 1 — regimes

Handled in [DOC_DETECTOR.md](DOC_DETECTOR.md). Mode changes are normal, not
anomalies.

## Defence 2 — the persistence gate

A window is *anomalous* if its score exceeds the regime threshold. An **alert**
requires the score to stay above threshold **continuously for
`persist_minutes`** (default 30, i.e. 60 consecutive 30 s windows).

The reasoning is pure physics-of-the-situation: a forklift passing, a door
slamming, a wash-down cycle — seconds to minutes. A spalled bearing does not
heal; it will still be there in an hour, and in a week. **Time is the cheapest
feature that separates the two**, and it costs almost nothing in detection
because real faults develop over days.

Arithmetic: at a 7 % per-window false-positive rate, requiring 60 consecutive
independent excursions is astronomically unlikely. Real excursions are not
independent, which is exactly why `soak_report.py` measures the empirical run
lengths rather than trusting this calculation.

## Defence 3 — one alert per episode, with hysteresis

`AlertGate` latches when it fires and re-arms only after `clear` consecutive
normal windows (default 4). Without hysteresis, a score oscillating around the
threshold would machine-gun alerts. One problem produces one alert.

```python
gate = AlertGate(need=60, clear=4)
gate.feed(anomalous)   # -> True exactly once per episode
```

## Defence 4 — "this was normal"

Every alert in the dashboard carries two buttons: **This was normal** and
**Real problem**. Pressing the first records the verdict, de-escalates the
device from red to amber, and sends `mark_normal` to the node so those windows
join the baseline at the next retrain.

This turns the customer's annoyance into training data. It is also, long-term,
the most valuable asset the company accumulates: a cross-customer library of
labelled machine sound.

## Delivery paths

| Path | Works without internet? | Notes |
|---|---|---|
| LAN webhook | **Yes** | fires straight from the device; covers a siren relay or a local bridge |
| MQTT → cloud → email | No | queues on-device and replays on reconnect (QoS 1, bounded deque) |
| MQTT → cloud → webhook | No | Slack, Teams, PagerDuty all accept an HTTP POST |
| Dashboard bell | No | |

Core detection and local alerting are required to survive total internet loss —
that is a hard constraint in the spec, and why `local_webhook` is called
directly from the firmware loop and can never raise.

## Health tiers

**Corrected 2026-08-18 (backlog T1.7 / self-review F5). The previous version of
this section was wrong, and the measurement that shows it is below.**

It used to say: green < 70 % of threshold, amber 70–100 %, red = alert raised.
Both a band in score *magnitude*. Measured against the repo baseline:

| Population | old amber band (0.7–1.0× threshold) |
|---|---|
| 200 fresh **healthy** windows | **16.5 %** |
| 40 windows of a fault ramped from severity 0.002 → 0.05 | **12.5 %** |

The healthy score distribution's own upper tail lives inside the band —
median **0.580×** threshold, p95 **0.762×**, max **1.034×** — while a
developing fault crosses it in about one severity doubling. So amber was a
"watch this one" badge *more likely on a healthy machine than on a failing
one*. That is worse than the dead UI F5 predicted: it actively teaches the
customer that colour on this dashboard means nothing.

The tiers are now defined on **state**, which is what the pipeline already
knows (`firmware/reporting.py: tier_from`):

| Tier | Meaning | Notifies? | Healthy windows | Ramp windows |
|---|---|---|---|---|
| green | below the regime threshold | no | 99.5 % | 60 % |
| amber | **above** threshold, persistence gate not yet satisfied | **no** | **0.5 %** | **40 %** |
| red | an alert episode is live | yes | — | — |

Amber still never notifies, so it costs no trust — the difference is that it
now carries information. Every transient produces one (the forklift lights the
badge amber for a minute and then it clears, which is exactly the story we want
the dashboard to tell), and a developing fault sits amber for the whole
persistence window before the alert.

Verified end to end, `firmware/main.py --simulate --fast --persist-minutes 2`:
the single-window severity-0.5 transient at w04 reads **amber, streak 1/4, no
alert**; the growing fault reads amber at w12–w14 (streak 1→3) and flips to
**red at w15 with ALERT #1**. One alert, unchanged from before this work.

## The display index

The raw Mahalanobis distance is not showable: over severity 0 → 0.5 it moves
**2.46 decades** (4.63 → 1340). `ScoreReporter.report()` returns a bounded
0–100 `index`, log-linear in score/threshold and pinned to **70 exactly at the
threshold**, so 70 means "at this machine's own learned limit" on every unit
and in every regime regardless of mic sensitivity or mounting. The same sweep
becomes **47.0 → 91.3**. Healthy windows span 31.6–70.1 (median 53.5, IQR
49.0–57.6), so the fleet view has ~9 points of honest day-to-day movement while
green.

A calibrated *probability* was the obvious choice here and was built first. It
saturates: median healthy percentile **100.0000**, because the χ² fit is made
on in-sample learn distances which are biased low. It is still reported as
`percentile` for the cases where it is informative, but it does not drive the
display. See the note in `firmware/reporting.py`.

## Severity trending

A Mahalanobis distance of 1340 is not a physical quantity, so it cannot answer
"is my machine worse than last week". `physical_severity()` returns quantities
that can, all monotone in defect size (Spearman ρ = +1.000 over the sweep):

| Metric | severity 0 → 0.5 | On the wire? |
|---|---|---|
| band-limited RMS in the demodulation band | −23.8 → **−6.0 dB** (17.8 dB) | yes, `severity_band_rms_db` |
| envelope-spectrum peak height (`env_peak_db`) | 19.8 → **64.8 dB** (45.0 dB) | **no** — see below |
| envelope peak / background (`env_peak_ratio`) | **3.6× → 582×** over the demo ramp | yes, `severity_env_peak_ratio` |
| envelope-fluctuation energy, dB re learn period | −0.0 → **+20.9 dB** | yes, `severity_env_db_re_learn` |
| detected repetition rate | 121 Hz (noise) → **152.5 Hz**, locked from severity 0.02 | yes, `severity_env_peak_hz` |

**Read the third row's units carefully.** `env_peak_ratio` is a linear ratio,
not dB, and the dashboard plots it as "×" on a log axis for that reason. It is
*exactly* invariant to overall gain — measured 23.19 at gains 1, 2, 4 and 8 —
whereas band RMS moves 6.02 dB per doubling. That is why both are charted:
level and contrast are independent, and a microphone knocked slightly closer
to the machine moves one and not the other.

`env_peak_db`, which `physical_severity`'s own docstring calls "the trendable
one", is computed but **not** in the telemetry payload. Recorded rather than
silently fixed (T1.11): band RMS and the ratio between them span level and
contrast, so it adds nothing the two published fields cannot reconstruct.

The repetition rate is the one to watch on real data: we never *name* it as
BPFO to the customer (the system overview (not in this public copy) §3), but if next week's peak is at the same
frequency and taller, that is one defect getting worse rather than a new
problem — and in the simulation it tracks the shaft correctly across regime
changes (152.5 Hz at f_r = 50, 91.6 Hz at f_r = 30).

## Measuring it: `tools/soak_report.py`

Point it at a device's SQLite state after a week on a healthy machine:

```bash
python tools/soak_report.py --db /var/lib/acoustic-monitor/state.db --outdir report/
```

It reports false alarms per node-week, a **95 % Poisson upper bound**, score
distributions per regime, regime occupancy over time, headroom (how close
healthy operation came to the threshold), and a **recommended
`persist_minutes` computed from the observed transient statistics**.

The Poisson bound matters and is the intellectually honest part: observing
**0 alerts in 7 days does not demonstrate a rate below 1 per week** — the 95 %
upper bound on that observation is about 3 per week. The tool says so instead
of declaring victory.

Verified output on a synthetic 7-day, 20 160-window soak containing three
injected transients:

```
FALSE ALARMS PER NODE-WEEK : 0.00
  alerts / exposure        : 0 in 7.00 days
  95% upper bound          : 3.00
  Gate 3 (<= 1)            : PASS (point only)
  headroom @ p99           : 20.7 %
  recommended persist_min  : 2 (current 30)
```

All three transients were correctly suppressed by the gate.
