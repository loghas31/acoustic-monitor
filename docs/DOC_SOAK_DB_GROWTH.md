# Database growth + SD-wear audit — T4.2

"Measure bytes/day at the real telemetry rate, confirm retention pruning
holds, and estimate SD write endurance. Recommend settings." —
the task backlog (not in this public copy) T4.2. Companion to `docs/DOC_SOAK_MEMORY.md` (T4.1, RAM
rather than disk). Tool: `tools/db_growth_audit.py`; fast regression tests
(including the real finding below) in `tests/test_db_growth_audit.py`.

## Method

Drives the real `firmware/state.py`'s `StateDB.record_window` directly —
not a reimplementation of its SQL — with synthetic feature vectors of the
exact shape and rounding precision `record_window` itself writes (37-dim,
`round(v, 5)`). Real feature extraction was skipped on purpose: database
growth depends on row STRUCTURE and SIZE, not on whether the numbers came
from a real signal, and skipping it turns a would-be multi-hour run (T4.1
measured ~230-270 ms/window for real extraction) into one that finishes in
under a minute. 1% of windows were marked anomalous — a round, deliberately
pessimistic stand-in for a healthy machine's real false-alarm rate (not this
project's own measured number, which needs H4's real soak), chosen so the
estimate isn't flattered by assuming zero anomalies.

One piece of care needed explaining: `state.py`'s retention pruning
(`_trusted_prune_ts`, from T4.3's NTP-jump fix) is *deliberately* suspicious
of a timestamp that advances faster than real elapsed time — simulating 18
days inside one real minute is exactly that shape of jump, and the guard
would (correctly) refuse to prune almost anything if left alone. The audit
tool's fake clock advances `time.monotonic()` in lockstep with the simulated
timestamps for the duration of the process only (restored before exit), so
the guard sees real-looking elapsed time. Full reasoning in
`tools/db_growth_audit.py`'s own docstring.

## What was actually measured

18 simulated days at the shipped 30 s window (2,880 windows/day), shipped
7-day retention:

| day | size (kB) | readings rows | anomaly rows |
|---|---|---|---|
| 1  | 1,356 | 2,880  | 31  |
| 7  | 9,300 | 20,160 | 211 |
| 8  | 9,324 | 20,161 | 240 |
| 18 | 9,376 | 20,161 | 547 |

**Retention pruning works as documented.** Readings-row count climbs for
exactly 7 days, then plateaus at 20,161 rows (7 × 2,880, off by one at the
boundary) for the remaining 11 simulated days — one full retention window's
worth, steady state, exactly the "at most one window lost, old rows pruned
on every insert" contract `docs/DOC_FIRMWARE.md` describes for the readings
table.

**Growth rate, measured, not estimated:**
- **Pre-retention (days 1–7, nothing pruned yet):** 1,324 kB/day.
- **Post-retention (days 8–18, steady state):** +5.2 kB/day — small, not
  zero (see the real finding below for why).
- **Measured bytes/reading-row (day 1, before any DELETE): 482.1 B.** The
  module docstring in `firmware/state.py` (frozen — not edited for this)
  estimates "~400 B/row"; measured is about **20% higher**. Recorded here
  rather than in the frozen file itself, per the frozen-file rule (a
  hand-estimated comment being 20% low is not, by itself, a failing test's
  worth of "genuine bug").

482.1 B/row × 2,880 rows/day ≈ 1,388 kB/day, matching the measured 1,324
kB/day pre-retention rate closely (the small gap is SQLite page-boundary
and index overhead, which does not scale linearly with row count).

## A real finding: the `anomalies` table has no retention policy at all

`docs/DOC_FIRMWARE.md` describes local-state retention in a way a reasonable
reader would take as covering the whole database ("old rows are pruned on
every insert (SD-card wear is a genuine field failure mode)"). Reading
`state.py`'s actual SQL: only `DELETE FROM readings WHERE ts < ?` exists.
There is no equivalent statement for `anomalies`. **Confirmed both by the
18-day audit (anomaly-row count only ever climbs — 31 → 240 → 547, never
drops) and by a small, fast, direct regression test**
(`tests/test_db_growth_audit.py::test_anomalies_table_is_not_pruned_by_retention__real_finding`):
an anomaly row written on day 0 is still present after retention has long
since pruned its corresponding reading.

At the realistic scale this matters at — real alerts, not every
locally-anomalous window, since only a *persisted* episode reaches a
customer — this is a slow, low-severity gap: `anomalies` rows are written
whenever a single window scores over threshold (T1.6's own held-out
measurement: up to 3.8% of windows on the worst calibration tested, likely
much lower deployed, since the 30-minute persistence gate on
top means almost none of those become a real alert episode). It is NOT
zero over a multi-year unattended deployment, and it is currently
**undocumented and unbounded**, which is worse than being small.
**Not fixed here** — `state.py` is frozen and this is a real, but not
urgent, gap rather than the kind of active-harm bug (T4.3's NaN-poisoned
baseline, the NTP wipe) the frozen-file exception exists for; recorded as a
"Not done" row in `docs/DOC_STATUS.md`, and `docs/DOC_FIRMWARE.md`'s
retention line corrected to say plainly which table the pruning contract
actually covers.

## SD write endurance

The write-volume number that matters for flash wear is **not** the
plateaued file size (9.3 MB, capped by retention) — it is the *cumulative*
bytes written over the card's life, which keeps climbing even after file
size stops, because old rows are constantly replaced by new ones. At the
measured **1,324 kB/day** logical write rate: **~483 MB/year**, plus a small
amount from the unbounded anomalies table (+5.2 kB/day × 365 ≈ 1.9 MB/year
at this audit's pessimistic 1% rate — smaller still at a realistic alert
rate). **This is a lower bound**, not the true flash-write figure: SQLite's
default rollback-journal mode writes a journal alongside each transaction's
page writes, and the filesystem/SD controller add their own overhead (wear
levelling, erase-block granularity) — none of that is visible from
measuring the `.db` file's own size, and this audit does not claim to
measure it. Treat the number below as directionally right, not exact, and
settle it for real the way every other x86-vs-Pi number in this repo gets
settled: measure it on the actual hardware (H4).

Real published endurance figures, current as of this writing (search
results, cited below — not this project's own measurement):

| Card class | Example | TBW rating |
|---|---|---|
| Plain consumer microSD | (typical) | usually **unpublished** — not a number to design around |
| Surveillance/"high endurance" consumer | Kingston, 128 GB | ~117 TBW |
| Surveillance/"high endurance" consumer | Lexar, 128 GB | ~135 TBW |
| Surveillance/"high endurance" consumer | Samsung Pro Endurance, 128 GB | ~820 TBW |
| Industrial SLC | Kingston/SanDisk, 64 GB | ~1,920 TBW, 30K P/E cycles |
| Industrial (pSLC-mode) | ATP | up to 25,000 TBW |

**Even the lowest cited figure (unpublished consumer cards aside) is ~117
TB.** Against this audit's ~483 MB/year lower-bound estimate, that is
headroom on the order of **100,000+ years** even before accounting for the
real flash-write-amplification factor being unmeasured — several orders of
magnitude of margin large enough that getting the amplification factor
wrong by 10× or 100× would not change the conclusion.

## Recommendation

**The card the build specifies — an official Raspberry Pi 32 GB A2-class
microSD — is more than adequate for this application's own write load.** No upgrade to an industrial or
surveillance-rated card is warranted by this workload; the write volume
this software generates is not the risk. the parts list (not in this public copy)'s existing
warning ("you will corrupt one eventually, but not in week 1") is about
power-loss corruption and general SD unreliability on a Pi, a real and
separate risk this audit does not address or change — buying a spare, which
the parts list (not in this public copy) already recommends, remains the right mitigation for that,
not a higher TBW rating. Two things worth doing that this audit's numbers do
support: (1) fix the `docs/DOC_FIRMWARE.md` retention claim to say plainly
that `anomalies` is not pruned (done, this run); (2) if a future run wants
to close the "Not done" gap properly, add `anomalies` retention with either
the same 7-day window or a longer one (an audit trail of past alerts is
plausibly worth keeping longer than the raw readings) — a product decision,
not something this audit makes unilaterally.

## Reproduce

```bash
python tools/db_growth_audit.py --db /tmp/soak.db --days 18 --retention-days 7 \
    --anomaly-rate 0.01 --seed 17 --out-json /tmp/soak.json
```

Sources for the SD endurance figures:
- [Industrial SD cards: Key factors to consider](https://www.atpinc.com/blog/industrial-sd-cards-factors-requirements-to-consider)
- [Kingston Industrial microSD datasheet](https://www.kingston.com/datasheets/SDCIT2_us.pdf)
- [SanDisk Industrial High-Endurance whitepaper](https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/whitepaper/white-paper-industrial-high-endurance-video-cards.pdf)
- [Best Endurance microSD Cards for 24/7 Operations](https://www.faceofit.com/best-endurance-microsd-cards-for-24-7-operations/)
- [Reliability of microSD Endurance Cards Compared](https://mightygadget.com/reliability-of-microsd-endurance-cards-compared/)
