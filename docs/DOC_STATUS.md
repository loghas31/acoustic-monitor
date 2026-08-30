# Status — what is proven, what is assumed, what is missing

Companion to the system overview (not in this public copy) §7. **Keep this file honest.** It is
the difference between an engineering project and a demo.

Last updated: 2026-08-30, T1.16 #1. **A claim in this project's own tooling
moved from "warned about" to "actually detectable".** `cold_start_screen.py`'s
clipping guard was blind to every lossy recording — i.e. to every phone
recording, which is the only real input this project has. Measured through a
real ffmpeg AAC round trip: `data/normal.wav`, a HEALTHY machine scoring 5.5,
driven into clipping reports **51.7 at 49.75 Hz**, above the ~35 this tool's
own documentation calls a real fault, with the flat-top test reading 0.00063
and staying silent. Fixed by adding true-peak (dBTP, ITU-R BS.1770-4) as a
second, independent test — a codec destroys the sample-level evidence and
*creates* the inter-sample overshoot, so the two fail in opposite conditions.
Zero false positives on all ten real/simulated recordings in `data/`
(−0.74 to −27.72 dBTP against a 0.0 threshold). Suite 680 → **694**.
Full detail: the task backlog (not in this public copy) T1.16 #1 / Run log, `docs/DOC_SELF_REVIEW.md`
F28. **Residual, stated plainly:** the guard is 3-6 dB clear on broadband
audio but only **0.05 dB** clear on a clipped pure tone — narrowband signals
gain almost no overshoot from clipping — so tonal clipping is caught only by
the original flat-top test, and only on un-transcoded WAV. Not closed, not
hidden.

Previous update: 2026-08-29, F25. `tools/fridge_scan.py` cleared its per-stem
`data/_scan_work/<stem>` working directory with `shutil.rmtree()`, which this
project's sandboxed dev environment refuses to execute at all — confirmed
directly (a plain shell `rm -rf` on the identical directory fails the same
way; `mv`/`os.rename` of it succeeds). Fixed by retiring the directory via
rename into `output/_attic/scan_work/` instead of deleting it, plus a
per-stem `fcntl.flock` for a separate, narrower, real-machine-only
concurrency risk. `tests/test_fridge_scan.py` now 12/12 green (was 6/12
failing). Full detail: the task backlog (not in this public copy) F25 / Run log, `docs/DOC_SELF_REVIEW.md`
F25 (includes a mid-run retraction of this fix's own first, overstated
"two concurrent sessions raced" theory). **Residual, stated plainly:** this
run's suite baseline was only 564/676 tests re-verified green before the
sandbox hit a documented ENOSPC/`useradd`-failed wedge; the remaining ~112
(alphabetically from `test_severity_calibration.py` on) were not re-run this
session and should not be assumed green until a future run confirms them —
none exercise `tools/fridge_scan.py`, so there is no code-path reason to
expect this fix broke them, but that is an inference, not a measurement.

Update before that: 2026-08-28, T5.3 (pitch deck). No detection/physics claim moved
— the pitch material (not in this public copy) is a packaging task, not a measurement — but worth a
line here because building it exercised this file directly: line 717's
instruction ("the pitch deck should quote the ratio to control, not the
absolute number") was followed, so the deck's physics slide quotes **4.5×
separation from the healthy control (5.8× at BPFO vs 1.3× healthy)**, the
CWRU-surrogate figure, not `verify_signals.py`'s flattering 56.7× headline.
Every other number on the ten slides is likewise sourced from an existing doc
(the cost model (not in this public copy), the original brief (not in this public copy) §5, the funding notes (not in this public copy), the task backlog (not in this public copy)'s gate table),
with the deck's own honesty slide restating "no sensor has ever been attached"
and the three undecided questions verbatim from the original brief (not in this public copy) §5, so a reader of
the deck alone gets the same caveats a reader of this file would. Full detail
in the task backlog (not in this public copy) T5.3.

Previous update: 2026-08-27, T1.14 Part 2 CLOSED (F20), paperwork finished this
run for a fix that landed in the working tree via a concurrent agent session
during this same run (verified independently below, not redone; a genuine
mid-flight retraction in `docs/DOC_SELF_REVIEW.md` F20 is worth reading —
an unreproducible throwaway-script number was caught by rebuilding the
measurement as a committed tool, `tools/sweep_crest_margin.py`, before it was
published). `CREST_FLOOR_MARGIN` in `firmware/baseline.py` raised **0.3 →
0.7**, calibration statistic p99 → `max` over the learn period's healthy
crests (the `max` change alone does almost nothing — floor 7.073 → 7.089, FPR
unchanged at 0.107 — the larger margin does the work). Deployed floor
**7.073 → 7.488572112684821** (re-derived independently this run from a fresh
`firmware/baseline.py --simulate --windows 48`, bit-identical across 3
repeats). Re-measured directly against the real deployed `firmware/
baseline.npz`: `deployed_threshold_fpr` **0.107 → 0.000**, TPR/AUC unchanged
at 1.000/1.000, gating counts unchanged (0 transient, 1 persistent). **Not a
trade**, per the committed sweep (`tools/sweep_crest_margin.py --margins 0.3
0.7 1.0 --severity 0.35 0.20 0.10`): F19 recovery is identical (6/6, 4/6, 0/6
at the three severities) at every margin tested — the FPR drop costs nothing
measurable. **A separate, complementary check this run** (different question:
calibration itself vs. the pre-T1.13 flat constant, not which margin to use
within calibration): a fresh 14-machine reproduction, `select_demodulation_band`
scored against each machine's own calibrated floor vs. the flat
`DEFAULT_CREST_FLOOR=10.0`, gives raw-constant recovery **3/14** and calibrated
(margin 0.7, `max`) recovery **11/14** on the same severity-0.20/1600 Hz
scenario — independent evidence that per-machine calibration (T1.13's actual
contribution) is where the real gain is, not the specific margin value T1.14
tuned. Band selection itself is still per-window and still unstable — this
raises the floor until instability stops mattering *on these signals*, it
does not fix the mechanism — so the real fix (per-machine margin calibration
or a persistence-count band switch) is filed as **T1.15**, still open. Full
suite re-verified this run in 2 chunks (207+3 skipped / 393, `test_api.py`
primed in both, 13 shared = 590) plus the 4 new `tests/test_sweep_crest_
margin.py` tests confirmed green separately = **594 passed, 3 skipped, 594
collected**, matching README. No frozen file touched this run (the fix was
already in the working tree before this run started; the task backlog (not in this public copy), this
file and the commit log (not in this public copy) are what this run's own paperwork pass
edited). Full detail in `docs/DOC_SELF_REVIEW.md` F20 and the task backlog (not in this public copy)
T1.14/T1.15.

Previous update: 2026-08-26, T1.12 (calibrate the two severity scales, F18). New
`ml/sensitivity/calibrate_severity_scales.py` measured, rather than assumed,
what `ml/simulate.py`'s and `ml/realdata/synth_phone_recording.py`'s
`severity` knobs actually mean physically (band RMS in dB re each
generator's own healthy floor). Neither of the two hypotheses this task was
opened to test held up: it is not a constant ~10x rescaling (the dB offset
between the curves drifts by ~6 dB across the tested range), and it is not
that a pink noise floor masks faults a white one does not (at matched
absolute dB, mic-only detection margin was as good or better on the pink
generator). The real mechanism is `synth_phone_recording.make_pair`'s own
`shared_knock_ring`, which puts fault-band energy into ITS healthy signal by
design — something `ml/simulate.py`'s healthy signal structurally cannot
have — so low nominal "severity" values there are swamped by an in-band
floor `ml/simulate.py` never faces, not by noise colour. **New "proven by
execution" row below**; **new "Assumed, not proven" caveat below**
narrowing "detection down to severity 0.02" to `ml/simulate.py`'s own
generator, since it does not transfer to any generator whose healthy signal
carries its own fault-band energy. 10 new tests
(`tests/test_severity_calibration.py`), suite 575 → 585 passing, no frozen
file touched, no regression. Full detail in `docs/DOC_SENSITIVITY.md` §T1.12
and the task backlog (not in this public copy) T1.12.

Previous update: 2026-08-26, T1.14 Part 1 (pin the F20 regression) + Part 2
investigation. `deployed_threshold_fpr`/TPR/AUC/regime-false-alarm/gating
counts are now pinned against the REAL deployed `firmware/baseline.npz` by
`tests/test_evaluate_pinned.py` (7 tests; `ml/evaluate.py` refactored to
expose `compute_metrics()`, a pure function, CLI output verified bit-identical
before/after). Corrected a stale row below (FPR "0.07" → the actual **0.1071**
T1.13 produces). **Part 2 (fix band instability, not just assert it) was
investigated, not shipped**: Option A (hold the band chosen during the learn
period) is mechanically disproved — the calibrated `crest_floor` is BY
CONSTRUCTION built so a healthy learn period never selects anything but
`DEFAULT_BAND` (`tests/test_crest_floor_calibration.py` already pins this), so
holding it forever is bit-for-bit equivalent to reverting T1.13 entirely:
measured 0/6 F19 recovery. Option B (single-window amplitude-margin
hysteresis, prototyped in a throwaway script, not committed) has a real
tunable trade-off, not a free win: margin 1.5 → 6/6 F19 recovery but FPR
unchanged at 0.107; margin 2.0 → 4/6 recovery, FPR 0.036; margin 3.0 → 0/6
recovery, FPR 0.000. No single global margin satisfies T1.14's own bar ("6/6
AND ~0"). Next step for T1.14 Part 2, not yet tried: per-machine calibrated
margin (mirroring how `crest_floor` itself is calibrated) or a persistence
requirement (N consecutive winning windows before switching, mirroring
`AlertGate`) rather than a single-window amplitude margin. No frozen file
touched (`ml/evaluate.py` is not on the frozen list); full suite re-verified,
575 collected (was 568), matching README. See the task backlog (not in this public copy) T1.14 and
`docs/DOC_SELF_REVIEW.md` F20 for the measured numbers.

Previous update: 2026-08-23, T7.3 (backlog paperwork only — no code changed this
run). Two "proven by execution" rows added above, for work that was already
built and already described in the commit log (not in this public copy) but had never been
entered in the task backlog (not in this public copy): **F17**, the iPhone Voice Memos default (lossy
AAC) deleting everything above ~10 kHz (**−78.6 dB**) — dangerous rather than
merely lossy because machine resonances live at 1–20 kHz, so a real fault
above the cliff would read back as a quiet healthy machine; and the
phone-upload route (`POST /recordings`) calling the same `phone_monitor.
analyse()` the CLI uses, with no second signal-processing path to drift out
of step. This run reinstalled deps (`python-multipart` newly required,
`ffmpeg` present), re-ran the full suite in 2 chunks (551 passed, 3 skipped,
554 collected — matching README), and reproduced bit-identical baseline
thresholds (8.074/9.381) and STAGE 3 GATE: PASS. Full detail in
the task backlog (not in this public copy) T7.3.

Previous update: 2026-08-23, T3.4 (real MQTT connect/reconnect coverage). New
"proven by execution" row added above: `MqttUplink.start()`'s real
`connect_async`/`loop_start` path — never exercised by anything before this
run — now has 6 tests against a real (if minimal, hand-rolled) MQTT 3.1.1
broker, `tools/fake_mqtt_broker.py`, closing the last-named T3.4 gap. Real
bug found and fixed in the TEST HARNESS, not production code: the broker's
first `kick()` (drop the connection to simulate a blip) used plain
`conn.close()`, which measurably does not deliver a TCP FIN while another
thread is blocked reading the same socket — fixed with `shutdown(SHUT_RDWR)`
first, verified non-vacuous by reverting it and watching the reconnect test
fail as expected. `mqtt_client.py` itself needed no change. Suite 536 → 542
passing, no regression (STAGE 3 GATE PASS, bit-identical thresholds). Full
detail in the task backlog (not in this public copy) T3.4.

Previous update: 2026-08-22, T2.1/T2.2 (deployment kit). No detection/physics
claim moved — this is an ops-tooling pair, not a measurement — but it closes
the last piece of the "the day the Pi boots" kit alongside T2.3
(`dev_up.sh`). **T2.1** (`scripts/provision_pi.sh` + `check_provision.sh`)
was found already fully built with its own test file
(`tests/test_provision_scripts.py`, 9 tests) but never ticked off in
the task backlog (not in this public copy) — the same orphaned-paperwork shape T0.1 and T7.2 already
hit this session. Verified rather than redone: 9/9 pass. **T2.2**
(`scripts/deploy_node.sh`) was genuinely missing and is this run's real
contribution — rsyncs `firmware/`+`ml/` to a provisioned Pi over ssh,
excluding the device's own `*.db`/`baseline.npz` from the `--delete` pass so
a routine code push can never erase a unit's learned state, then restarts
the systemd service and tails its journal. **Neither script can be run
against a real Pi in this sandbox** (no network target, no root, no
raspi-config — confirmed directly). What can be, and was: `deploy_node.sh`
was run as a REAL subprocess against fake `ssh`/`rsync` executables placed
first on PATH that log their own argv instead of opening a connection
(`tests/test_deploy_node.py`, 9 tests) — this proves the actual rsync
excludes/paths and ssh command sequence execute exactly as documented, a
stronger check than `bash -n` alone gives `provision_pi.sh`. New "Not
proven" row below: the systemd restart and journal-tail steps are
untested against a real `sshd`/`systemd` — that is H2's job. Suite
514 → **523 passing** (0 failed, verified in two chunks). `README.md`'s
test-count claims corrected to match (514 → 523). Full detail in
the task backlog (not in this public copy) T2.1/T2.2.

Previous update: 2026-08-20, T4.4 (Pi performance harness). New
`tools/pi_perf_harness.py` times every real stage `extract_features` calls
(30 reps/stage/case, healthy + bearing-fault). New "proven by execution" row
below: **`select_demodulation_band` alone is 58% of the whole ~153 ms
extraction time** — nearly 6× the next most expensive stage — not
previously broken out anywhere the ~150 ms/window figure gets quoted.
`MahalanobisScorer.score` confirmed negligible (0.01 ms), matching
`README.md`'s existing claim. At this repo's standing 8–10× A53 slowdown
assumption (not measured this run): 1,227–1,534 ms against the 2,000 ms
stage-2 gate — passes, with real but not huge margin (~24% at the high end)
flagged plainly rather than called comfortable. **A second discrepancy
found and left unreconciled, not silently picked one number:**
`tracemalloc`'s measured Python-level peak (~25.6 MB, itself a documented
lower bound) is markedly higher than `README.md`'s existing hand-estimated
"~15 MB" feature-extraction-peak line; both are x86 numbers of different
things, and neither has been checked against the Pi's real allocator —
`README.md`'s table footnoted to say so rather than either number quietly
overwriting the other. Full detail in `docs/DOC_PI_PERF.md` and
the task backlog (not in this public copy) T4.4.

Previous update: 2026-08-20, T4.2 (database growth + SD-wear audit). New
`tools/db_growth_audit.py` drives the real `StateDB.record_window` for 18
simulated days at the shipped 30 s window / 7-day retention. New "proven by
execution" row below: retention pruning genuinely plateaus the readings
table after 7 days; measured bytes/reading-row **482.1 B** (state.py's own
docstring estimate, "~400 B", was ~20% low — corrected in the docs). **A
real, previously-undocumented finding, new "Not done" row below:** the
`anomalies` table has NO retention policy at all — only `readings` is
pruned — confirmed by the 18-day audit and a fast direct regression test
(`tests/test_db_growth_audit.py`). `docs/DOC_FIRMWARE.md`'s retention claim,
which read as covering the whole database, corrected to say which table it
actually describes. **SD-wear conclusion, with real published TBW figures
cited via web search:** ~483 MB/year lower-bound write estimate against even
the lowest cited endurance rating (~117 TBW) gives roughly 100,000+ years of
headroom — the official Pi 32 GB A2 card the parts list (not in this public copy) already
specifies is adequate; no purchasing change recommended. Explicitly flagged
as a lower bound, not exact — SQLite's journal + filesystem write
amplification is not visible from measuring the `.db` file's own size. Full
detail in `docs/DOC_SOAK_DB_GROWTH.md` and the task backlog (not in this public copy) T4.2.

Previous update: 2026-08-20, T4.1 (memory-leak soak). New `tools/memory_soak.py`
drives the exact functions `firmware/main.py`'s real loop calls (not a
reimplementation) for **3,200 simulated windows** (26.7 simulated hours),
sampling RSS every 5th window, and reports the trend per continuous process
rather than across process restarts — the naive whole-run comparison mixes
real memory growth together with measured ~10 MB inter-process allocator
noise at process start alone. New "proven by execution" row below: **no
chunk showed the same-signed, compounding growth a real leak produces** —
three of six ~500-600-window chunks trended up, three trended down, no
relationship to window count; RSS never exceeded 165.2 MB against the
shipped 350 MB systemd cap. **Honest limit, in a new "Assumed, not proven"
row:** this soak can rule out a GROSS leak and cannot, on its own, rule out
a SLOW one smaller than the measured ~150-300 MB/week noise floor — that
floor is a sandbox artefact (each chunk is a separate OS process, because
this agent's shell caps a single call well under the wall time thousands of
windows takes), not a property of the code, and a genuinely continuous run
(a laptop overnight, or H4's real 7-day Pi soak) would resolve it more
tightly. Full detail in `docs/DOC_SOAK_MEMORY.md` and the task backlog (not in this public copy) T4.1.

Previous update: 2026-08-20, T2.4 (the operations runbook (not in this public copy)). No proven/assumed claim
moved — this is an operations doc, not a measurement — but every command in
it was actually run this session: `baseline.py --simulate` for starting a
learn period, a real `dev_up.sh` server hit with `curl` for bringing up and
reading the dashboard (`/dashboard/summary` and `/devices/{id}/status`
responses quoted verbatim), the real `baseline.py --retrain` CLI against a
freshly-built feedback episode for the retrain section (mean ratio
**531.92x → 0.62x**, matching T3.4's own regression test), and the real
`main.py --simulate` demo sequence with its actual log lines quoted for the
video-recording section. **New "Not done" row below, found while writing
it:** `config.yaml`'s comment promises a 24–72 h "production target" learn
period, but `window.learn_windows: 96` — what `baseline.py --windows`
actually defaults to — is 96 × 30 s = **48 minutes**; nothing currently
orchestrates a real multi-day learn period. Flagged in the runbook, here,
and with a corrective footnote on `README.md`'s "Customer setup" section
rather than silently changed, since which default is right is a product
decision this task was not asked to make. `journalctl` log-collection
commands are given as the documented standard for the shipped systemd unit
but explicitly labelled as **not executed** — no systemd exists in this
sandbox to run them against. Full detail in the task backlog (not in this public copy) T2.4.

Previous update: 2026-08-20, T2.3 (`scripts/dev_up.sh` — backend without
Docker). New "proven by execution" row below: `scripts/dev_up.sh` +
`tests/test_dev_up.py` bring up a real uvicorn process over a real socket
(every other backend test uses FastAPI's in-process `TestClient`, which
never binds one) and run a full register-user → register-device → POST a
reading → read it back round trip against it. Three real bugs, all found by
running the script rather than reading it: (1) the first version defaulted
its SQLite file under `backend/` — the network-mounted repo directory — and
hit the same `disk I/O error` `tests/test_api.py` already documents; fixed
to default under `$TMPDIR`. (2) that default filename was not uid-qualified,
the identical shape of bug a concurrent run's audit (F12 in
the commit log (not in this public copy)) had just fixed elsewhere in the suite for a shared
`/tmp/test_acoustic.db` — fixed the same way, `acoustic-monitor-dev-$EUID.db`.
(3) fixing that exposed an ordering bug: computing the uid and the script's
own directory both shell out to external commands (`id`, `dirname`) that ran
*before* the python3-on-PATH check, so a badly broken PATH died two lines
early with a confusing `dirname: command not found` instead of the script's
intended message — fixed by using bash's builtin `$EUID` and moving the
builtin `command -v python3` check to the first line. **Used the
opportunity to close a real methodology gap**: every full-suite
"verification" so far this session (see every "Previous update" below) has
been a chunked, run-file-by-file re-run, because the bash tool's own
per-call timeout is shorter than a true single-process `pytest tests/`
normally takes. This run finally ran the whole suite as one process —
**487 passed, 0 failed, 158.34s** — which is the only way F12-shaped
cross-file interaction bugs (shared engines, shared fixture paths) can
actually be caught; chunked-by-file runs cannot see them by construction.
An interim attempt showed 76 spurious `FileExistsError`s, traced (by
grepping every erroring file for hardcoded `/tmp` paths — none found beyond
the already-fixed `test_severity_persistence.py`) to a likely second,
concurrent agent process sharing this same sandbox's `/tmp`, not a bug in
this repo; not reproduced on the clean re-run. Suite 484 → 487 passing.
Full detail in the task backlog (not in this public copy) T2.3.

Previous update: 2026-08-20, T1.4 (sensitivity study — how bad can reality
be?). `ml/sensitivity/sweep.py` + `tests/test_sensitivity.py` (21 tests)
already existed from a concurrent run; this run executed the actual 20-point
sweep (SNR, resonance frequency, mounting attenuation, interfering
machinery — 5 values each) and wrote `docs/DOC_SENSITIVITY.md` +
`ml/artifacts/sensitivity.{json,png}`. New rows below and in "Assumed, not
proven": **SNR is the one axis that visibly broke in range** (AUC
1.000 → 0.878, TPR → 0.67 at −5 dB, 0/5 mildest-severity fault windows
caught); **mounting attenuation held to 24 dB tested but its safety margin
collapsed ~54×** (140.6× → 2.6× threshold), the axis judged most likely to
matter first on a real machine and least conclusively cleared;
interference showed no measured degradation to 4× the primary hum;
resonance showed no measured degradation but is **confounded by
`ml/simulate.py`'s own clamp** (`min(resonance_hz, 0.4*fs)` caps the accel
channel at 2560 Hz for every tested value ≥4500 Hz — roughly the top half
of that axis does not test what its label claims). All of this is small-n
(5-15 fault windows, 12 healthy per point) — a sensitivity MAP, not a
calibrated false-alarm-rate spec. Full detail in `docs/DOC_SENSITIVITY.md`
and the task backlog (not in this public copy) T1.4.

Previous update: 2026-08-20, T3.6 (frontend <-> backend live integration
check). `mock.js`'s own header comment promises its shapes "mirror the v2
FastAPI responses exactly" — nothing checked that before this. New
`tests/test_frontend_backend_integration.py` seeds 3 real devices through
the real FastAPI `TestClient` (healthy, faulty-with-an-acknowledged-alert,
legacy-firmware) and new `frontend/src/api/contract.test.mjs` (node, no
browser) checks that every field this repo's own React components actually
read — grepped out of `Overview.jsx`/`DeviceDetail.jsx`/`AlertConfig.jsx`/
`Onboarding.jsx`/`lib/trend.js`, not guessed — is present in both the
mock's response and the real backend's. **Honestly scoped:** this is a
JSON-contract check, not a rendered-pixel one — no headless browser exists
in this sandbox, so the "Frontend never exercised against a live backend in
a browser" row below is UNCHANGED and still open; this closes a narrower,
different gap (does the DATA shape agree, not does it RENDER correctly).
Verified non-vacuous: deleting a field from a copy of `mock.js` makes the
check fail, naming the exact component that would break. **A real, minor
finding:** `api.alertLog()` is called by `client.js` but zero `.jsx` files
call it — dead frontend API surface, recorded not fixed. Suite 483 → 484
passing. Full detail in the task backlog (not in this public copy) T3.6.

Previous update: 2026-08-20, T3.7 (fingerprint the baseline against the code
that produced it). This closes a gap flagged repeatedly since T1.8: a
firmware change that preserves the feature vector's DIMENSION but shifts
the DISTRIBUTION underneath it (measured then: 100% of fresh healthy
windows scoring a median 138.4x threshold) used to be indistinguishable
from a real fault. `firmware/baseline.py` now stores a fingerprint of the
learn period's own score/threshold ratio distribution
(`startup_ratio_median`/`startup_ratio_p95`); `main.py`'s startup sequence
checks the unit's first 8 real windows against it and raises
`BaselineMismatchError` — a distinct refusal with the retrain command, not
an ordinary persistence alert — if they look implausible. New "proven by
execution" row below reproduces the T1.8-class bug directly and end to end
through the real `main.py --simulate` CLI. **A design correction made
mid-run, worth recording:** the check was first written INSIDE
`MahalanobisScorer.score()` itself, triggered automatically by call count;
the full-suite re-run caught that this broke `test_state_feedback.py` and
`test_sensitivity.py`, both of which call `score()` directly against
curated anomalous windows for unrelated reasons (feedback retraining,
threshold behaviour) that have nothing to do with a real unit's first 8
windows. Reworked so `score()` stays pure and the one-shot check is an
explicit method main.py calls itself — the kind of bug only a full
regression run catches, not code review. the handover notes (not in this public copy) updated: T3.7
was explicitly flagged there as "still open"; now describes what the check
does and does not catch (a gross ~5x mismatch on nearly all of the first 8
windows, not a partial drift — retraining after any upstream feature
change is still the real fix). Suite 475 → 483 passing. Full detail in
the task backlog (not in this public copy) T3.7.

Previous update: 2026-08-20, T3.5 (config validation on startup). New
"proven by execution" row below: `firmware/config_schema.py` catches every
missing section/key, wrong type, and out-of-range value in the real
`firmware/config.yaml`'s schema in one pass, plus a physics-grounded check
that an `accelerometer.sample_rate` below ~2.2 kHz cannot see the 1–20 kHz
resonance band the IIS3DWB was chosen for. `main.py`/`baseline.py` now
refuse a malformed config with a named, one-line `ConfigError` instead of
a bare `KeyError` several frames deep — reproduced the old failure first
(renaming `window:` to `windowz:` crashed with `KeyError: 'window'`, no
filename, no hint) and pinned the fix with an end-to-end subprocess test
against the real, fixed CLIs. `firmware/train.py`'s `anomaly.sigma_k` is a
newly-discovered gap: read by that module but absent from the shipped
config and NOT added to the schema, because `train.py` is untested,
unreferenced v1.5 autoencoder code (confirmed via
`grep -rln "train.py" tests/*.py docs/*.md`, no hits) — recorded under
"Not done" below rather than silently worked around. Suite 449 → 475
passing. Full detail in the task backlog (not in this public copy) T3.5.

Previous update: 2026-08-20, T3.4 (feedback-loop test coverage, partial).
`tests/test_state_feedback.py` closes a real gap — nothing previously
connected `state.py`'s `mark_normal`/`feedback_vectors` to `baseline.py
--retrain` under test. New "proven by execution" row below: a realistic
10-window feedback episode drops its own mean score/threshold ratio
**531.92x → 0.62x** on retrain, while an unrelated one-off transient that
was never fed back stays flagged at **1.60x** — the fold-in learns the
specific reported pattern, not "loud transients in general." A secondary,
non-obvious finding: a SINGLE loud window fed back does not fully
desensitise the model, because T1.6's contamination guard (correctly)
distrusts a lone outlier and deploys the safer threshold instead of
stretching to fit it — the feedback loop only works as designed for
realistic multi-window episodes, which is what production always produces
(the 30-minute persistence gate means no episode is ever one window). Left
`[~]` at the time: `mqtt_client.py`'s "fake broker" ask was not done. **Closed
2026-08-23** — see the "Last updated" entry at the top of this file and the
new proven-by-execution row above; full detail in the task backlog (not in this public copy) T3.4.

Previous update: 2026-08-20, T3.3 (`RESULTS.md` template). No proven/assumed
claims moved — this is a lab-notebook template, not a measurement — but
worth recording here because writing it caught two commands elsewhere in the
repo that don't do what a reasonable person would guess: `tools/accel_axis_report.py`
has no CLI at all (simulate-only, no way to point it at a real recording),
and `tools/ingest.py --stem` silently only applies when ingesting one file at
a time. Neither is a bug in the tool itself — both are documented in
`RESULTS.md` directly so Logan doesn't rediscover them mid-experiment. Full
detail in the task backlog (not in this public copy) T3.3.

Previous update: 2026-08-20, T3.1 (GitHub Actions CI). `.github/workflows/ci.yml`
added — pytest + the STAGE 3 physics/detector gate in one job, `npm ci && npm
run build` in another. It has never run on GitHub's own infrastructure (the
agent cannot push or trigger Actions from its sandbox), so every command in
it was instead run locally against clean, from-scratch copies of the relevant
directories first. That caught two real bugs before they could fail silently
on a fresh contributor's first PR: `ml/evaluate.py` reads `firmware/baseline.npz`
by default, which is gitignored and therefore absent on a clean checkout — a
workflow that wrote its scratch baseline anywhere else would have made
`evaluate.py` silently evaluate nothing; and `firmware/requirements.txt`
pulls in the hardware-only `sounddevice`/`spidev` packages, which this whole
session's 443-test suite ran clean without, so CI does not install them
either. Full detail in the task backlog (not in this public copy) T3.1.

Previous update: 2026-08-20, T7.2 (phone-recording quickstart). Found
`tools/phone_monitor.py`, its 8 tests, and a mic-only extension to
`firmware/capture.FileSource` already built and uncommitted with no paperwork
— verified them (`--self-test` PASS: 0.0 % band-selector fire rate, 0.0 %
speed-estimate reliability, 0.0 % of windows above threshold, correct
learn-period contamination flag) rather than redoing them. This run's own
addition, `ml/realdata/synth_phone_recording.py` (an independent third
pink-noise generator, resonance placed outside `DEFAULT_BAND` on purpose),
answers the "is the band-selector fallback severity-dependent or a permanent
blind spot" question the earlier pink-noise finding (below) left open: it is
**severity-dependent** — crest 5–8.5 (fallback, the fault is missed) below
roughly severity 0.7–0.9 on this generator's noise floor, crest 13–20 (auto-
locates the true resonance) above it. Also found, by trying the obvious wrong
way to score a phone recording: a mic-only feature vector scored against the
DEPLOYED audio+accel baseline reads ~11,800x threshold on healthy AND faulty
alike (the zeroed accel channel dominates, not the machine) — T4.3's
dead-channel guard working as designed, not a new bug, but a real trap for
future code that is now documented in `docs/PHONE_RECORDING.md` and pinned by
a test. Full detail in the task backlog (not in this public copy) T7.2.

Previous update: 2026-08-20. Two things closed that run, neither requiring new
detector work: **F10 / backlog T0.1** (`channel_stats` computing RMS/crest/
kurtosis on the raw signal instead of the DC-removed one) was fixed in code by
a concurrent run on 2026-08-19 whose own backlog/status update never landed —
this run found `firmware/features.py` already carrying the fix and five
regression tests already in `tests/test_features.py`, confirmed the deployed
baseline is unaffected (retrained: unchanged at **8.069 / 9.380**; 60-window
replay via `tools/sim_trace.py`: 0 of 11 healthy windows above threshold), and
closed the paperwork. Full detail and the DC/gravity measurements are in
`docs/DOC_SELF_REVIEW.md` F10 (marked ✅ FIXED). **The fix is verified neutral
on synthetic data only** — the simulator has no DC offset and no gravity, so
this cannot be validated here, only failed to be contradicted; it remains a
prediction about hardware until H2. Second, **T7.1** (the handover notes (not in this public copy)) was
written — the prose document for Logan alone, once the agent's subscription
ends, covering how to run the pipeline, five likely hardware bring-up
failures, which numbers are synthetic, the three open questions the hardware
sprint settles, and the week-1 bring-up order. Every command and number in it
was re-executed this run, not recalled.

Previous update: 2026-08-19, the fault-injection audit (T4.3). Two genuine bugs
of the F2 shape — code that fails quietly into a plausible-looking wrong
result rather than saying it doesn't know — were found by executing all five
named failure scenarios (corrupt baseline file, NTP clock jump, disk full,
broker unreachable for days, sensor unplugged mid-run) and fixed: a
NaN-poisoned `baseline.npz` used to load and score without error (a
corrupted threshold makes `score > threshold` False by IEEE754 definition,
i.e. "never anomalous", forever); a Pi's NTP-forward clock step used to wipe
the *entire* readings table in one prune call rather than losing at most one
window. Full detail in §What the fault-injection audit taught us below.

Previous update: 2026-08-18, the run that measured which feature blocks actually
earn their place in detection (T1.10 / self-review F9). Every block tested
detects at least one of two fault kinds (bearing outer race, imbalance) at
held-out AUC > 0.85 — including the two band-ILR blocks F9 found to be near
one-dimensional on healthy data. Low rank is not the same as no information:
the envelope block is the mirror-image counterexample, full rank under an
imbalance fault and still at chance-level AUC for it.

Previous update: 2026-08-18, the run that stopped the detector inventing operating
regimes out of sensor noise (T1.9). The mic-only build — the *recommended*
build — was splitting one unchanging machine into two regimes on essentially
every learn period, at 6.3× the false alarms; so was any build whose audio and
accelerometer levels move together. Self-review finding F7 confirmed as a real
bug, though not the one it described.

Previous update: 2026-08-18, the run that gave the simulated accelerometer
three genuinely different axes (T1.8). Self-review finding F6 is the first one
to be **confirmed** rather than disproved; the accelerometer half of the
feature vector is now capable of being wrong, and the repo baseline had to be
retrained because a simulator change silently invalidated it.

---

## Proven by execution

Every row below was run and its output observed. Re-run any of them.

| Claim | Command | Result |
|---|---|---|
| Clipping survives a lossy codec as an inter-sample peak, and is detectable that way | `pytest tests/test_cold_start_screen.py` (14 new tests, one gated on ffmpeg) | Real AAC 128k round trip of ADC-clipped audio: flat-top **0.00063, silent**; true peak **+3.27 dBTP**, warned. Healthy `normal.wav` driven into clipping scores **51.7 at 49.75 Hz** on old code with no warning (healthy baseline 5.5); `bearing_outer` goes **94.1 at 99.75 Hz** against a true 152.25. Zero false positives: six real phone recordings **-2.65 to -27.72 dBTP**, `normal`/`bearing_outer`/`bearing_inner`/`imbalance` **-0.74 to -0.91**, threshold 0.0 |
| The clipping guard is 3-6 dB clear on broadband audio and 0.05 dB clear on a tone | same, plus the measurements in `true_peak_dbtp`'s docstring | broadband clipped **+6.31 dBTP**, real fan audio clipped **+3.27**, clipped 137 Hz sine **+0.05** against a clean full-scale sine at **-0.00**. Tonal clipping is covered by `clipped_fraction`, not by this — stated as a live blind spot, not a solved case |
| RESULTS.md Experiment 0 is reproducible after the `fan_experiment.load` signature change | `python tools/fan_experiment.py "data/Healthy fan take 1.wav" "data/Card in fan.wav" "data/Healthy fan take 2 (after card).wav" --predict-hz 26.1` | **bit-identical** to the recorded run: 1.9 / 20.4 / 5.3 at 9.2 / 25.8 / 15.0 Hz |
| Envelope analysis beats raw spectrum | `python ml/verify_signals.py` | 2.2× raw vs **56.7×** envelope |
| Features behave on synthetic data | `python firmware/features.py` | 37-dim vector; fault selects 3866–5420 Hz, crest 95.4; healthy falls back, crest 6.0 |
| No feature block is singular by construction | `pytest tests/test_compositional.py` | The one exactly-dependent block (6 raw envelope fractions, null direction = uniform vector to \|cos\| **1.0000**) is gone: sv ratio **6.5e-3 → 0.34** on identical windows. ILR verified invertible, scale-invariant, isometric, and NaN-free on a dead channel |
| Removing the redundancy did not change detection | 300 bootstrap learn/holdout splits, same 192 windows both ways | held-out healthy FPR **0.0492 ± 0.0301 → 0.0345 ± 0.0224**, paired difference −0.0147, 95 % CI **[−0.083, +0.048]** — favourable direction, not distinguishable from zero. AUC 1.000 both |
| The score does not lean on the ill-conditioned directions | same, F4's test re-run after T1.5 | top-5 tightest eigendirections give **6.1 %** of d² on healthy windows, down from 31.6 %, even though cond(precision) rose 8.1 → 18.8 |
| A stale baseline is refused legibly, not cryptically | `pytest tests/test_compositional.py::test_stale_baseline_is_refused_with_a_readable_message` | 40-dim baseline + 37-dim firmware now raises a named error with the retrain command; previously `ValueError: operands could not be broadcast together with shapes (37,) (40,)` from inside `score()` |
| Regimes are found | `python firmware/baseline.py --simulate --windows 48` | k=2, 24/24 split, thresholds **8.069 / 9.380** — bit-identical to the deployed `baseline.npz` after T1.9, so the clustering change is a no-op on the shipped configuration |
| The silhouette threshold was unsafe below 2 effective dimensions | `python tools/regime_miconly_cost.py --stage null` | On single-cluster noise with NO regimes, the pre-T1.9 rule invented them in **98.8 %** of 1-D clouds and **98.5 %** of collinear 2-column clouds, against **0.0 %** in 2-D and 3-D. Null silhouette median **0.584** (1-D) vs **0.283** (3-D), maximum **0.702** — above the 0.5 threshold that was tuned in 3-D |
| A mic-only node split one unchanging machine into two regimes | `python tools/regime_miconly_cost.py --stage cost` | 48 healthy mic-only windows → k=2, counts **[30, 18]**, a regime boundary through noise **0.0002 decades** wide. Over 100 bootstrap learn periods the old rule chose k>1 **100/100** (k=2 66×, k=3 32×, k=4 2×) |
| …and it cost 6.3× the false alarms | same | held-out healthy FPR **0.1358 ± 0.1445** (old rule) vs **0.0217 ± 0.0290** (k forced to 1, the oracle here). AUC **1.000 in every arm** — the damage is entirely false alarms. It also fired T1.6's contamination warning on **14 of 200** clean fits |
| The fix reproduces the oracle exactly | same | with `MIN_REGIME_SEPARATION` + the dimension-aware silhouette floor: k=1 in **100/100**, FPR **0.0217 ± 0.0290**, i.e. identical to forced k=1; **0.000** invented regimes across all four null cloud types |
| Dropping the dead dimension — F7's prescribed fix — does nothing | `pytest tests/test_regimes_miconly.py -k no_op` | A constant column is exactly 0 after standardisation, so k-means labels and silhouette are **identical** with it present or removed. The defect was one *live* dimension, not one dead one |
| Genuine regimes still survive without an accelerometer | same file | the repo's 50/30 Hz two-speed learn schedule gives k=2, 24/24, centroid gap **10.0** (10× the gate) in **both** the full and mic-only builds. Level-only regimes are recovered from **0.1 decade** apart and merged below it |
| The old threshold estimator was set by one learn window | `pytest tests/test_threshold.py` | p99.5/max = **0.989** at n=24 (0.964 even at n=480). On the pre-fix code one bad window of 24 moved the threshold **1.47x at 12 σ, 2.16x at 25 σ, 3.30x at 50 σ** — unbounded and monotone in the outlier size |
| χ² with *p* dof is the **wrong** analytic fix for this feature vector | same | effective rank **13.7 of 37**, so d² concentrates near the effective dimensionality: mean(d²) = **28.5** vs p = 37, χ²₃₇ rejected at KS p = **3e-90**. Deployed at n=24 it gave **11.0 %** held-out FPR vs 3.8 % for the estimator it was meant to replace. F3's preferred option, disproved |
| A robustly-fitted scaled χ² does describe them | same | KS p = **0.62**; reproduces the pooled 99.5th percentile to **0.7 %** (6.866 fitted vs 6.914 measured over 480 healthy windows) |
| The new estimator is exactly indifferent to outlier magnitude | `pytest tests/test_threshold.py::test_estimator_itself_is_exactly_insensitive_to_outlier_size` | threshold **bit-identical** at 12 / 25 / 50 / 100 σ, because only the median and 75th percentile are read. End to end the deployed threshold moves 1.39x / 1.24x / 1.01x / 0.80x — bounded and no longer increasing |
| The fix costs nothing on a clean learn period | `python firmware/baseline.py --simulate --windows 48` | thresholds **8.348 / 9.882**, identical to before; `min()` selects the empirical value, ratios 0.798 / 0.757, not flagged |
| A contaminated learn period no longer blinds the unit | end-to-end, 1 of 48 learn windows with a loud external event (audio ×6) | regime-1 threshold **21.504 → 8.426** (clean value 8.391); detection of a severity-0.02 bearing fault **0.375 → 0.833**; held-out FPR 0.000 both; **flagged**, ratio 2.55 |
| Contaminated learn periods are detected, clean ones are not falsely flagged | 200 resamples per condition | false-flag on clean **1.0 %** (n=24) / **0.5 %** (n=48); catches **99.2 %** of 12 σ and **100 %** of 25 σ contamination at ratio 1.25 |
| One outlying learn window used to become its own regime with threshold 0.0 | `pytest tests/test_threshold.py::test_one_outlying_operating_point_does_not_become_its_own_regime` | before: k=2, counts **[47, 1]**, thresholds **[7.4658, 0.0]** — LedoitWolf fitted to a single sample, so every window later assigned there alarms unconditionally. `MIN_REGIME_WINDOWS = 8` fixes it; genuine two-regime periods still give k=2 |
| The simulated accelerometer was three copies of one signal | `python tools/accel_axis_report.py` (before T1.8) | inter-axis r **+0.9988 / +0.9964 / +0.9952** on every signal kind; the 12 per-axis accel statistics spanned effective rank **3.75 of 12** with a 4-dimensional near-null space (sv ratio 1.3e-3). F6 confirmed — 12 features carried ~4 |
| It now has three axes that can disagree | same, after T1.8 | r **+0.04 / −0.68 / +0.51**; effective rank **9.32 of 12** healthy (8.11 on a fault ramp), sv ratio 1.3e-3 → **1.8e-2**. Median R² of a y/z statistic on the x block **0.175**, 1 of 8 above 0.95 |
| Decorrelating the axes needed the PHASE, not just different resonances | `pytest tests/test_accel_axes.py` | giving each axis its own housing mode + independent sensor noise still left r(x,z) = **0.904**, because the shaft hum is 20 dB above everything and all three saw it in phase. Adding a per-axis phase relative to the rotating imbalance vector is what took it to 0.04–0.68 |
| Axis 0 is bit-identical, so audio, accel band-ILR and `estimate_fr` provably did not move | `pytest tests/test_accel_axes.py -k bit_identical` | `np.array_equal` against the old single-axis generator on all four signal kinds; the audio channel too. Any change in detection is attributable to the 8 y/z statistics alone |
| One shaft and one defect still give one fault frequency | same | demodulating each axis **in its own band** (x 2304–2816, y 1579–2210, z 768–1536 Hz) recovers BPFO at **152.62 Hz on all three** (+0.02 % of the true 152.60), at 42.9× / 40.3× / 36.2× background |
| Three real axes changed detection by nothing measurable | `python tools/accel_axis_compare.py` — 200 paired bootstrap splits, 96 healthy + 24 fault windows at severity 0.02 | held-out healthy FPR **0.0292 ± 0.0360 → 0.0319 ± 0.0340**, paired difference +0.0027, 95 % CI **[−0.084, +0.104]**; AUC **1.000 both**. Expected: the simulator was already at AUC 1.000 and had no room to improve. The gain is testability, not accuracy |
| A simulator change silently invalidated the deployed baseline | scored 40 fresh healthy windows against the pre-T1.8 `baseline.npz` | median **138.4×** threshold, max 143.2×, **100 %** of healthy windows above threshold. The 37-dimension contract check added in T1.5 passed throughout, because the dimension did not change — only the distribution. Retrained: thresholds **8.348 / 9.882 → 8.069 / 9.380** |
| `channel_stats` now removes DC before computing RMS/crest/kurtosis (F10) | `pytest tests/test_features.py -k dc`; injected DC offsets scored against the deployed baseline | a healthy window with a 10 % DC offset scored **2.62×** threshold pre-fix (permanent false alarm), **172.9×** at 100 % offset; post-fix, a DC offset does not move the score (pinned by test). Accelerometer `logrms` moved **0.008** for a 4× vibration increase pre-fix (measuring gravity, not vibration) vs **0.60** post-fix. Deployed baseline retrained and **unchanged** at 8.069 / 9.380; 60-window replay puts 0/11 healthy windows above threshold. **Cannot be validated beyond this** — the simulator has no DC and no gravity, so this is a prediction about hardware, not a closed loop |
| The band-ILR blocks are near one-dimensional on healthy data but not uninformative | `python tools/feature_block_report.py`; `pytest tests/test_feature_blocks.py` | effective rank **2.31 of 7** (audio) / **1.53 of 7** (accel) on the healthy 2-speed learn period, corroborating F9's raw-fraction measurement on a different representation. Despite that, a per-block Mahalanobis distance trained on healthy-only windows detects a bearing fault at AUC **0.993 / 0.997** and an unrelated imbalance fault at AUC **0.965 / 0.907** |
| The envelope block is fault-specific, not universally useful | same | AUC **0.996** detecting a bearing fault (it is *built* to see impact periodicity) but AUC **0.447** — chance — detecting an imbalance fault, even though its effective rank stays **6.63 of 7** (not rank-suppressed). High rank does not imply informative, mirroring T1.5's "low rank does not imply uninformative" from the other side |
| Every feature block detects at least one of two tested fault kinds | same, all 5 blocks × 2 fault kinds | worst case audio_band_ilr on imbalance (0.965) and accel_band_ilr on imbalance (0.907); no block scores below 0.9 for at least one fault. The per-channel statistics blocks (audio, accel) are the most reliably informative, AUC ≥ 0.968 on both fault kinds tried |
| `tools/phone_monitor.py` runs the real learn→score pipeline on one recording, mic-only, with no bearing geometry | `python tools/phone_monitor.py --self-test`; `pytest tests/test_phone_monitor.py` (8 tests) | PASS: 0.0 % of windows above threshold, mic-only speed reliability 0.0 % (by design), learn-period contamination correctly flagged when present. SYNTHETIC fixture (pink noise + mains hum, independent of `ml/simulate.py`) |
| The protrugram's pink-noise fallback (found on accel data, T1.1) reproduces on the mic/phone domain **and is severity-gated, not permanent** | `pytest tests/test_phone_recording.py`; `python ml/realdata/synth_phone_recording.py --self-test` | at moderate fault severity, crest **5–8.5** (below `crest_floor=10`) → falls back to 3–6 kHz, missing a resonance placed at 1600 Hz; Gate 2 fails 1.7×/1.0× (forcing the true band recovers a real but sub-gate 3.0×/1.6×). At higher severity, crest **13–20** → finds the true band unaided, Gate 2 passes 12–18×/6–11×. SYNTHETIC — an independent third pink-noise generator, not `ml/simulate.py` or `validate_public_dataset._pink` |
| Mic-only speed estimate can be badly wrong, not just unconfirmed | same | `estimate_fr` mic-only branch: 16.6 Hz found vs 24.17 Hz true (31 % off), `reliable=False` throughout, exactly as the function's docstring promises |
| Scoring a mic-only capture against the DEPLOYED audio+accel baseline is the wrong tool | `pytest tests/test_phone_recording.py -k full_sensor_baseline` | healthy and "faulty" synthetic windows both score **~11,800×** threshold (score ≈95,580 vs threshold 8.07) — the zeroed accelerometer channel dominates, not the machine. T4.3's dead-channel guard behaving as designed, not a new bug; `phone_monitor.py` avoids it by training its own mic-only baseline from the recording itself |
| Detector separates fault from healthy | `python ml/evaluate.py` (`tests/test_evaluate_pinned.py` pins this) | **ROC AUC 1.000**, deployed-threshold FPR **0.000** (T1.14 Part 2's calibrated `crest_floor=7.488572112684821`) / TPR 1.00. **This row moved twice**: "FPR 0.07" (stale, pre-T1.13) → **0.1071** (T1.13's per-machine floor, 2026-08-26) → **0.000** (T1.14 Part 2, `CREST_FLOOR_MARGIN` 0.3→0.7, 2026-08-27), each time re-measured and pinned rather than asserted from memory. **Not a trade**: the committed `tools/sweep_crest_margin.py` shows F19 recovery identical (6/6, 4/6, 0/6 across three severities) at margins 0.3/0.7/1.0 — an earlier throwaway-script claim that this cost fault recovery did not reproduce and was retracted (see F20 in `docs/DOC_SELF_REVIEW.md`). Separately, this run confirmed calibration itself (vs. the flat pre-T1.13 constant) is where the real gain is — 3/14 vs 11/14 recovery on a 14-machine sample. Band selection is still per-window and unstable; the mechanism fix (per-machine margin or persistence-based switching) is filed as **T1.15**, still open |
| Regime switches do not alert | same | **0** false alarms |
| Transients do not alert; faults do | `python firmware/main.py --simulate …` | **1** alert, 0 for a severity-0.5 transient |
| Soak analysis is sound | `python tools/soak_report.py` on 7-day DB | 0/week, 95 % upper bound 3.0, 3 transients suppressed |
| Fault-frequency maths | `python ml/realdata/fault_frequencies.py --bearing 6204 --rpm 2850` | BPFO 144.97 + BPFI 235.03 = 380.00 = N·f_r ✓ |
| The identity holds for **every** table entry at every speed | `pytest tests/test_realdata.py` | BPFO + BPFI = N·f_r to rtol 1e-12 across 8 bearings × 5 speeds; BPFO = N·FTF and BPFI − BPFO = N·f_r·γ likewise |
| Our formulae agree with an **independent authority** | same | CWRU's published multipliers reproduced from CWRU's published inches to **5e-4 absolute** (worst: 6203 BPFO 3.05312 vs 3.0530). Not the "4 d.p." the source claimed — the residual is CWRU's own 4-s.f. rounding of D, and is 0.05 Hz at 152 Hz, ~70× inside the ±2 % slip window |
| N is recoverable from the published outer multiplier | same | N = 9 (6205) and N = 8 (6203) to within 0.01 by inverting BPFO_mult = (N/2)(1−γ) |
| Shaft-harmonic masking is what stops a harmonic being called a fault | same | Same hand-built spectrum, `fr_hz=None` → **100.0x at 150.00 Hz** (−1.70 % "slip"); `fr_hz=50` → **12.0x at 152.60 Hz**. The guard width matters: patched from `3*df` to 1 bin, the search picks a leakage skirt at **149.80 Hz (40x)** |
| An unresolvable fault line returns INCONCLUSIVE, not FAIL | same, on `data/normal.wav` + `data/bearing_outer.wav` | A bearing with N=8, d/D=0.25 has BPFO = 3.00·f_r at *every* speed; gate returns `inconclusive=True`, `passed=False`, ratio NaN |
| Gate 2 passes on the repo's own pair (SYNTHETIC) | same | envelope **61.3x** faulty vs **2.2x** healthy → contrast **28.1x**; raw 2.6x → demod gain **23.8x**; comb at 1x/2x/3x = 60.7 / 9.8 / 3.2, all within 0.1 % of prediction |
| Rectify and Hilbert demodulators give the same verdict | same | Same PASS, peak location agrees to <0.5 Hz — so features.py's ~6× faster rectifier costs nothing |
| Bench tools degrade safely | `python firmware/bench/selftest.py` | friendly "NO HARDWARE", exit 2, no tracebacks |
| Detector survives a 12 kHz accelerometer-only dataset in CWRU format | `python ml/realdata/validate_public_dataset.py --make-surrogate /tmp/surr` | runs end to end; AUC **0.9889** — **SYNTHETIC SURROGATE, see below** |
| The reportable layer reaches the dashboard, firmware → chart JSON | `python tools/e2e_severity_trend.py` | 40 simulated windows published, **40 of 40** carry a `display_index` through ingest; band RMS **−21.99 → −3.00 dB**, envelope contrast **49.5× → 405×**, envelope energy re learn **+2.59 → +26.84 dB**, index **54.8 → 94.1**, all rising; fleet sparkline switches to `display_index`. SYNTHETIC source |
| Severity really does need one line **per regime** | same run, adjacent windows at equal fault severity | regime 0 (f_r = 30 Hz, BPFO 91.6) shows **434.6×** contrast where regime 1 (f_r = 50 Hz, BPFO 152.5) shows **192.9×** — a 2.3× difference from operating mode alone. A single line for the machine would sawtooth every mode change and bury the trend |
| `env_peak_ratio` is a gain-invariant ratio, not a level in dB | `pytest tests/test_reporting.py -k gain_invariant` | ratio **23.19 at gains 1/2/4/8** (rel 1e-6); `band_rms_db` **−23.26 / −17.24 / −11.22 / −5.20**, exactly 6.02 dB per doubling. The dashboard labels it "×" on a log axis because it spans **3.6× → 582×** over one fault ramp |
| The severity columns can be added to a **populated** database | `pytest tests/test_severity_persistence.py` | `create_all` alone adds none of them (pinned); `add_missing_columns()` adds all 7, is idempotent (second call returns `[]`), preserves the existing row, and the migrated DB then accepts a full firmware payload |
| Resampling 12 kHz → 16/6.4 kHz does not change the answer | same, `--fs-mode native` | AUC 0.9826 native vs 0.9889 device (Δ 0.006) |
| CWRU `.mat` variable layout (`X097_DE_time`, `X097RPM`) is read correctly | same (surrogate files are written with the real CWRU naming via `scipy.io.savemat`) | DE, FE and RPM recovered from all 13 files |
| Regime clustering finds the CWRU load structure | same | k=3 from 4 load points (1797/1772/1750/1730 rpm) |
| An awkward real-world file survives conversion | `python tools/ingest.py --self-test` | 4 kHz carrier at 44.1 kHz stereo int16 → 16 kHz mono: carrier recovered to **0.000 %**, 137 Hz modulation to **0.00 %**, DC +0.05 → +2.7e-7, 40/40 features |
| Ingest does not change the Week-2 answer | 44.1 kHz stereo **int32** phone-like pair → `ingest.py` → `analyse_recording.py` | BPFO envelope **37.1x** (vs **35.8x** on the original 16 kHz files), peak 152.50 Hz (−0.06 % slip) in both, contrast 25.9x, **Gate 2 PASS** |
| Bandwidth really is the binding constraint | same faulty signal band-limited to 8 kHz, then ingested | BPFO envelope **35.8x → 1.2x**; Gate 2 fails all three checks. The `!!` warning is not decorative |
| One common gain preserves relative level | `ingest.py` on a 0.35/0.55-scaled pair | ingested RMS ratio **1.560** vs the 1.571 applied — 0.7 % (the residual is real: the round trip filters the faulty file's top octave harder) |
| The old amber tier was **not** dead UI — it was worse | 200 fresh healthy windows + a 40-window severity ramp, scored against the repo baseline | The 0.7–1.0× band fired on **16.5 %** of healthy windows and only **12.5 %** of ramp windows: *more likely on a healthy machine than a failing one*. Healthy score/threshold: min 0.283, median **0.580**, p95 **0.762**, max **1.034**. **This disproves F5's prediction that the band could never fire** |
| The state-based tier separates | same populations, `reporting.tier_from` | amber on **0.5 %** of healthy windows vs **40 %** of ramp windows |
| The display index is bounded, monotone and legible | `pytest tests/test_reporting.py`; severity sweep 0 → 0.5 | raw score **4.63 → 1340.01** (2.46 decades) becomes index **47.0 → 91.3**; exactly 70.0 at threshold in every regime; healthy windows span **31.6–70.1** (median 53.5, IQR 49.0–57.6) |
| A calibrated *probability* cannot drive the display | same | median healthy percentile **100.0000**, p95 100.0000 — saturates, because the χ² fit uses in-sample learn distances which are biased low. Built, measured, rejected; still reported as `percentile` |
| Physical severity is monotone in defect size, unlike the score | severity sweep 0 → 0.5, 3 seeds/point | Spearman ρ = **+1.000** for band RMS (**−23.8 → −6.0 dB**), envelope peak height (**19.8 → 64.8 dB**) and envelope energy re learn (**−0.0 → +20.9 dB**); ρ = +0.976 for peak ratio (3.8 → 206.9) |
| The detected repetition rate locks onto the fault line and tracks speed | same, and `main.py --simulate` across a regime switch | 121 Hz (noise) when healthy → **152.5 Hz** from severity 0.02 upward at f_r = 50; **91.6 Hz** at f_r = 30 — both BPFO for the simulated bearing |
| Reporting cannot change the alert decision | `pytest tests/test_reporting.py::test_reporting_does_not_change_the_alert_decision`; `main.py --simulate --fast --minutes 30 --persist-minutes 2` | score dicts bit-identical with the reporter in the loop; **1 alert**, transient at w04 stays amber (streak 1/4), fault flips red at w15 — unchanged from before T1.7 |
| A 24-window learn period is NOT enough for 37 features | same bootstrap, `--n-learn 24` | held-out healthy FPR **0.55–0.59 for both representations** — 12 windows per regime cannot fit even a Ledoit-Wolf covariance in 37 dimensions. At 48 (24 per regime, what `baseline.py --simulate --windows 48` ships) it is 0.03. Do not let a customer's learn period be halved |
| Whole codebase holds together | `pytest tests/` | **392 passed** in 37.7 s (378 before T1.10; +14 for `test_feature_blocks.py`) |
| T1.6 did not regress the pipeline | `python ml/evaluate.py`; `python firmware/main.py --simulate --fast --minutes 90 --no-mqtt` | AUC **1.000**, FPR 0.00 / TPR 1.00, 0 regime false alarms, STAGE 3 GATE PASS; **1** alert over 90 simulated minutes; extraction 148–156 ms/window |
| Dashboard builds | `npm run build` | ✓ 3.3 s |
| API works end to end | live uvicorn + SQLite | register → ingest → anomaly → feedback → health red→amber |
| A corrupt/truncated/NaN-poisoned `baseline.npz` is refused, not scored | `pytest tests/test_fault_injection.py -k baseline` | garbage bytes, a truncated write, a missing-field file, a NaN in `thresholds`/`means`/`precisions`/`global_mean`/`op_centroids`, and a zero `global_std` all raise `ValueError` naming the retrain command; a well-formed baseline is unaffected |
| An NTP forward clock step does not wipe retention | `pytest tests/test_fault_injection.py -k ntp_forward` | before the fix: a 2-year forward step in `ts` deleted every row in `readings`; after: ≥5/5 rows written before the step survive it, and the step's own row keeps the corrected wall time |
| Retention still prunes normally when the clock is NOT jumping | `pytest tests/test_fault_injection.py -k normal_operation_still_prunes` | 0-day retention still drops a reading once real elapsed time has passed |
| Disk-full mid-window is all-or-nothing, not a partial write | `pytest tests/test_fault_injection.py -k disk_full` | a simulated `OperationalError` on the reading insert, the anomaly insert, or the retention delete each leave prior committed windows untouched and commit nothing from the failing call; the device recovers on its own once writes succeed again |
| A failed write no longer leaks into a LATER, unrelated commit | `pytest tests/test_fault_injection.py::test_disk_full_on_anomaly_insert_does_not_leak_into_a_later_commit` | before the fix: a fault window whose anomaly insert failed left its reading pending, and the next successful window's `commit()` flushed it to disk with no anomaly record attached (measured: `[2000.0, 2030.0]` on disk when only `[2030.0]` should be there); after: `rollback()` on the exception path gives `[2030.0]` only |
| MQTT offline behaviour matches its own module contract | `pytest tests/test_fault_injection.py -k "telemetry_is_dropped or anomalies_queue or overflow"` | telemetry dropped (not queued) while disconnected; anomalies queue and replay in FIFO order on reconnect; a full offline queue keeps accepting new anomalies (evicting the oldest) rather than raising — now logged on eviction, previously silent |
| `MqttUplink.start()`'s REAL connect/reconnect path (not a stub) works end to end | `pytest tests/test_mqtt_live.py`, against `tools/fake_mqtt_broker.py` (a real listening socket, not `client.publish` stubbed) | real CONNECT→CONNACK→retained `status`→`cmd` subscribe; real QoS 0 telemetry / QoS 1 anomaly delivery; a real broker-pushed downlink reaches `on_command`; a refused CONNACK (rc≠0) does not mark `connected`; and the headline case — a severed TCP connection is detected by paho's own callback in **~0.5 s**, and the offline queue drains fully once paho's own automatic reconnect (no test code touches `_on_connect`/`_on_disconnect`) re-establishes the session. 6/6 passing |
| An iPhone Voice Memos recording (lossy AAC, default settings) can silently delete the very frequencies the detector depends on (F17) | `pytest tests/test_check_phone_audio.py` | ffmpeg AAC at a 32 kbps-class bitrate is flat to ±1.1 dB up to ~10 kHz then falls **−78.6 dB** at 10–16 kHz — a deletion, not attenuation, and invisible to a human listener; the system overview (not in this public copy) places machine resonances anywhere in 1–20 kHz, so a real fault above the cliff would read back as a quiet healthy machine. `check_phone_audio.lossy_cutoff_hz()` detects the cliff and the upload route refuses the recording naming the fix (Settings → Voice Memos → Audio Quality → Lossless); full-bandwidth audio and a single notched band are both confirmed NOT flagged, so the check discriminates rather than rejecting everything |
| The phone-upload route (`POST /recordings`) runs the SAME analysis code the CLI does, not a second path | `pytest tests/test_recordings_upload.py` | `backend/recordings.py` calls `phone_monitor.analyse()` directly; tests assert on `phone_monitor`'s own summary-dict keys as the contract, so the two cannot silently drift apart; upload size cap, empty-upload rejection, per-device recording isolation (404 on another device's id, not a permissions leak that reveals existence) and the `learn_period_too_short` flag below 48 windows are all exercised against the real FastAPI app, not mocked |
| A sensor dying MID-RUN is not silently reported healthy | `pytest tests/test_fault_injection.py -k dying_mid_run` | real simulated signals through the real scorer (not a synthetic fixture): zeroing the accelerometer or the microphone partway through a run flips every subsequent window anomalous (0/1 stray false positive tolerated on the ≤10 live windows beforehand) and drops `fr_reliable`; score is finite (no NaN/Inf) throughout |
| The "this was normal" feedback loop actually desensitises the model for a realistic episode | `pytest tests/test_state_feedback.py::test_retrain_folds_feedback_into_a_new_baseline_and_desensitises_it` | the REAL `firmware/baseline.py --retrain` CLI, run as a subprocess against a 10-window recurring-transient episode: mean score/threshold ratio **531.92x → 0.62x**. A never-fed-back one-off transient of a different kind stayed flagged at **1.60x** — the fold-in learns the specific reported pattern, not "loud in general" |
| A SINGLE outlier fed back does not fully desensitise — the contamination guard is doing its job, not failing | same file, measured while building the test | one loud window folded into a 24-window regime triggered T1.6's contamination flag and left the window still scored anomalous (ratio improved but stayed > 1.0); the SAME kind of transient repeated across 10 windows did not trigger contamination and folded in cleanly. The feedback loop only works as designed for multi-window episodes — which is what production always produces, since no alert episode is ever raised in under `persist_minutes` (30 min = 60 windows) |
| Retraining with no feedback recorded leaves the deployed baseline untouched, byte for byte | `pytest tests/test_state_feedback.py::test_retrain_with_no_feedback_says_so_and_does_not_touch_the_file` | `firmware/baseline.py --retrain` on a state DB with zero feedback prints "no feedback windows recorded" and exits before writing; the `.npz` file's bytes are identical before and after |
| A malformed `config.yaml` fails loudly and legibly, not with a bare `KeyError` | `pytest tests/test_config_schema.py::test_a_malformed_config_used_to_fail_illegibly_now_fails_with_a_named_error` | before the fix: `window:` renamed to `windowz:` crashed `main.py --simulate` with a raw `KeyError: 'window'` three frames deep, naming neither the file nor a fix; after: `config error: <path> failed validation (1 problem): 1. missing section [window]` on stderr, exit 1, same fix verified for `baseline.py` |
| The real, shipped `firmware/config.yaml` validates cleanly against the new schema | `pytest tests/test_config_schema.py::test_the_repos_own_config_yaml_validates_cleanly` | passes with no changes to config.yaml itself — the schema was built by grepping actual field usage, not guessed |
| An accelerometer sample rate too slow for the resonance band is caught before deployment, not discovered on a real machine | `pytest tests/test_config_schema.py -k bandwidth or 800` | `accelerometer.sample_rate: 800` (ADXL345-class) is rejected by name ("provably blind" to the 1–20 kHz band); the shipped 6400 Hz clears it |
| A T1.8-class firmware change (feature dimension unchanged, distribution shifted) is refused at startup, not scored as if the machine had just broken | `pytest tests/test_baseline_fingerprint.py::test_a_systematically_shifted_feature_contract_now_refuses_loudly_instead_of_scoring_silently` and `::test_main_py_refuses_a_mismatched_baseline_legibly_end_to_end` | a systematic shift on accelerometer-derived columns only, dimension unchanged: `MahalanobisScorer.score()` still scores every window (stays pure), but `check_startup_fingerprint()` on the first 8 ratios raises `BaselineMismatchError` naming "a different feature-generation contract"; end to end, the real `main.py --simulate` CLI against a real baseline with shifted `means` exits non-zero with `baseline mismatch: ...` on stderr and never reaches an ordinary `ALERT #` |
| The startup check does not cry wolf on a genuinely healthy startup or a single early transient | `pytest tests/test_baseline_fingerprint.py -k "healthy_startup or single_early_transient or score_itself_never_raises"` | 8-20 genuinely healthy windows never trip it; one loud window among the first 8 (the same false-positive shape T1.6's contamination guard distrusts elsewhere) doesn't either; `score()` itself never raises no matter what it's fed, even 20 windows shifted 50 units — only the explicit `check_startup_fingerprint()` call can |
| `mock.js`'s "shapes mirror the v2 FastAPI responses exactly" claim is checked, not just asserted in a comment | `pytest tests/test_frontend_backend_integration.py` | 3 real devices (healthy, faulty-with-an-acknowledged-alert, legacy-firmware) seeded through the real FastAPI `TestClient`; every field `Overview.jsx`/`DeviceDetail.jsx`/`AlertConfig.jsx`/`Onboarding.jsx`/`lib/trend.js` actually read (grepped, not guessed) is present in both the mock's and the real backend's response for `/dashboard/summary`, `/devices/{id}/status`, `/devices/{id}/readings`, `/devices/{id}/anomalies`, `/anomalies/{id}/feedback`, `/devices/register`, `/alerts/log/{id}`. Verified non-vacuous: deleting `anomalies_7d` from a copy of `mock.js` fails the check, naming `lib/trend.js#sparklineCaption` |
| The detector's headline numbers (AUC/TPR/FPR) hold across a range of signal quality, not just the simulator's one default point | `docs/DOC_SENSITIVITY.md`; reproduce with `python ml/sensitivity/sweep.py run --axis <axis> --value <v> --out ...` then `combine` | 20-point sweep (SNR, resonance, mounting attenuation, interference, 5 values each) through the real `firmware/baseline.py`/`inference.py`. SNR is the one axis that visibly breaks in range: AUC 1.000 → **0.878**, TPR → **0.67** at −5 dB. Attenuation held (TPR/AUC 1.000) to 24 dB tested but the score/threshold margin fell **140.6× → 2.6×** (~54× collapse) — the axis judged most likely to break first on a real machine. Interference and resonance showed no in-range degradation (resonance confounded above ~4500 Hz by `ml/simulate.py`'s own `min(resonance_hz, 0.4*fs)` clamp — stated in the report, not hidden) |
| The backend runs without Docker, over a real socket, not just FastAPI's in-process `TestClient` | `bash scripts/dev_up.sh` then `pytest tests/test_dev_up.py` | real `uvicorn` process, register user → register device → POST a reading over HTTP → read it back via `/dashboard/summary`; clean `SIGTERM` shutdown ("Application shutdown complete" in the log, no "disk I/O error"); `--fresh` removes the script's own default DB file for real |
| The whole suite holds together as ONE process, not just chunked file-by-file | `TMPDIR=/tmp python3 -m pytest tests/ -q -p no:cacheprovider --basetemp=/tmp/pt` | **487 passed, 0 failed, 158.34s.** First true single-process run this session (every earlier "suite re-verified" claim in this file's history was a chunked, run-file-by-file re-run — see T2.3 in the task backlog (not in this public copy) for why that matters and what it can miss). **PARTLY DISPROVED 2026-08-29 (T3.1/F27) — read this before trusting a green whole-suite run again.** A single-process `pytest tests/` is necessary but not sufficient: it cannot see a test that fails at *collection*, because such a test never reports as failed, it silently does not run and the summary line merely gets shorter. Measured on this repo the same day: `pytest tests/` gave **676 collected, 0 errors** while `pytest tests/ --ignore=tests/test_api.py` gave **662 collected, 1 error**. Whole-suite greenness was a property of `test_api.py` sorting first alphabetically. The row above is still true as far as it goes — chunking really does miss cross-file state bugs — but "one process, green" now means *no ordering bug reachable in that one ordering*, not *no ordering bug*. `tests/test_import_isolation.py` covers the gap by running pytest in subprocesses over deliberately hostile subsets |
| A test cannot silently vanish from the suite via import shadowing | `pytest tests/test_import_isolation.py` | **4/4 pass (3.6 s).** Two tests spawn pytest subprocesses (`--ignore=tests/test_api.py`; and the hostile pair `test_evaluate_pinned.py` + `test_frontend_backend_integration.py`) and assert clean collection — subprocesses because an in-process assertion runs after collection has already finished and so is structurally blind to this class of bug, the same blindness that hid F12 and F14. One static guard forbids any bare `import main` in `tests/` (both `firmware/main.py` and `backend/main.py` are importable as `main`); one forward-looking guard fails if a *new* duplicate module basename appears across `firmware/`/`ml/`/`backend/`, with `main.py` allowlisted. **Verified non-vacuous**: dropped unchanged into a pristine pre-fix copy of the repo it gives 3 failed / 1 passed |
| The real firmware loop's memory does not grow without bound over thousands of windows | `python tools/memory_soak.py ...` then `--summary --chunk-size <N>`; full method and numbers in `docs/DOC_SOAK_MEMORY.md` | **3,200 windows (26.7 simulated hours), 6 continuous-process chunks: no chunk showed the same-signed, compounding growth a real leak produces** — 3 up, 3 down, no relationship to window count. Peak RSS **165.2 MB** against the shipped 350 MB systemd cap. Caveat measured, not asserted: RSS at fresh-process start alone varies ~10 MB run to run, which sets a ~150-300 MB/week-equivalent noise floor this sandboxed, chunked method cannot resolve below — rules out a gross leak, not a slow one |
| Retention pruning genuinely bounds the local state database's size | `python tools/db_growth_audit.py --db ... --days 18 --retention-days 7`; full method and numbers in `docs/DOC_SOAK_DB_GROWTH.md` | **readings-row count climbs for exactly 7 simulated days then plateaus at 20,161 rows (one retention window) for the remaining 11.** Measured bytes/reading-row **482.1 B** (state.py's own "~400 B" docstring estimate was ~20% low). Pre-retention growth **1,324 kB/day**, matching 482.1 B × 2,880 windows/day. SD-wear headroom against this workload's ~483 MB/year lower-bound write estimate: **roughly 100,000+ years even at the lowest cited real endurance rating (~117 TBW)** — sources in `docs/DOC_SOAK_DB_GROWTH.md` |
| Where the ~150 ms/window feature-extraction time actually goes | `python tools/pi_perf_harness.py --reps 30 --baseline ...`; full method and numbers in `docs/DOC_PI_PERF.md` | **`select_demodulation_band` is 58% of the whole 153.4 ms extraction** (88.9 ms) — nearly 6× the next stage (`estimate_fr`, 16.3 ms) — never previously broken out. `MahalanobisScorer.score` confirmed negligible (0.01 ms). A53 estimate at this repo's standing 8–10× slowdown: **1,227–1,534 ms against the 2,000 ms stage-2 gate — passes, ~24% margin at the high end**, called out plainly rather than "comfortable". A second, unreconciled discrepancy found: `tracemalloc`'s measured ~25.6 MB Python-level peak (a documented lower bound) vs `README.md`'s existing hand-estimated "~15 MB" — both x86, neither checked against the Pi, footnoted rather than either number silently overwriting the other |
| `scripts/provision_pi.sh` and `scripts/check_provision.sh` are syntactically valid and internally consistent with each other | `pytest tests/test_provision_scripts.py` | **9/9 pass**: `bash -n` on both; `provision_pi.sh` uses `set -euo pipefail`, `check_provision.sh` deliberately does not (a health check must run every test, not abort on the first failure); every file `provision_pi.sh` reads from the checked-out repo actually exists; both scripts agree the systemd unit's `ExecStart` must point at the venv's python3, not the bare system one; an existing deployed `config.yaml` is never overwritten |
| `scripts/deploy_node.sh` invokes rsync and ssh with exactly the arguments it documents, in the right order | `pytest tests/test_deploy_node.py` | **9/9 pass**, run as a REAL subprocess against fake `ssh`/`rsync` executables placed first on PATH (they log their own argv and exit 0, no network call is made): confirms two rsync calls (`firmware/` with `--exclude='*.db' --exclude='baseline.npz'`, then `ml/`) land at `<user>@<host>:/opt/acoustic-monitor/{firmware,ml}/`; default ssh user is `pi`, overridable as a second positional argument; `ssh -t ... systemctl restart acoustic-monitor` and `ssh -t ... journalctl -u acoustic-monitor -f` fire by default and are independently skippable via `--no-restart`/`--no-tail`; zero arguments or an unrecognised flag exit non-zero with a usage message |
| `ml/simulate.py`'s and `synth_phone_recording.py`'s `severity` knobs are NOT the same physical scale, and pink noise is not what makes the phone generator's faults hard to detect | `python ml/sensitivity/calibrate_severity_scales.py curves` / `detect-both`; `pytest tests/test_severity_calibration.py` | band RMS re healthy floor at the "10x" pairing (sim 0.02 / phone 0.20, sim 0.20 / phone 2.0) gives dB offsets of 3.4 and 9.7 — not constant, so no single rescaling factor exists. At matched absolute dB, mic-only detection margin was **as good or better on the pink generator** (19.7×/45.0× threshold at ~7/12 dB) than the white one (3.0×/15.5× at ~4/10 dB) — rules out "pink masks faults" as the mechanism. The actual cause: `synth_phone_recording.make_pair`'s `shared_knock_ring` (fixed amplitude 0.15) puts fault-band energy into its OWN healthy signal by design; `ml/simulate.py`'s healthy signal has none. Detection is near-chance (TPR 0.0-0.17) while fault-ring amplitude is below that 0.15 floor and recovers sharply (TPR 1.0) above it — matches T7.2's independently-found "severity-gated band-selector fallback" crossover |

## Assumed, not proven

| Assumption | Risk if wrong | How to settle it |
|---|---|---|
| The IIS3DWB register map is right | accelerometer returns garbage | `check_accel.py` — WHO_AM_I + gravity |
| The I2S overlay exposes an ALSA capture device | no audio at all | `check_audio.py` |
| Machine resonance is near 3–6 kHz | **detector goes quietly deaf** — now quantified: removing two thirds of that band takes BPFO envelope contrast from 35.8x to **1.2x** | `check_mount.py` tap test |
| Magnet mounting couples adequately above 3 kHz | weak/erratic signal | `check_mount.py` repeatability |
| Python can drain the FIFO at rate | frequency axis silently stretches | `check_accel.py` measured ODR |
| Real faults look like our simulation | AUC collapses. **T1.4 (2026-08-20) narrows this to a ranked list of what to check first**, still entirely within-simulator: if the real bench disagrees, mounting attenuation (safety margin fell ~54× over 0-24 dB in-sim, breaking point untested beyond that) and low SNR (AUC visibly degrades below 0 dB in-sim) are the most likely explanations, ahead of resonance frequency or a second machine's interference, both of which showed no in-sim degradation in range | week-2 bench experiment |
| "Detection down to severity 0.02" (the SNR/attenuation tables in `docs/DOC_SENSITIVITY.md`) generalises to any realistic noise floor | **Narrowed by T1.12 (2026-08-26).** That claim is measured and correct for `ml/simulate.py`'s own severity scale specifically — a generator whose healthy signal has ZERO energy in the fault's resonance band. It does not transfer to `ml/realdata/synth_phone_recording.py`'s severity scale (which detects only from roughly severity 0.2-0.5 upward), not because pink noise is worse, but because that generator's healthy signal already carries fault-band energy of its own (the `shared_knock_ring`, amplitude 0.15) by design. Any FUTURE generator built from a real recording will have the same property — a real room is never silent at the bearing housing's resonance — so "severity 0.02" must not be quoted as a general early-warning capability without re-deriving the equivalent physical dB threshold for whatever noise floor is actually in play | H2/H3: once a real recording exists, run `ml/sensitivity/calibrate_severity_scales.py` (or its method) against it to find where the fault's own contribution starts to dominate the room/mount's own resonance-band noise — that dB level, not a severity index, is the real early-warning limit |
| The 14 band-energy features carry INDEPENDENT information from each other | **They mostly do not, on simulated data: measured 1.03 of 8 dimensions (audio), 1.01 of 8 (accel)** (F9), confirmed again on the assembled ILR blocks by T1.10 (2.31 of 7, 1.53 of 7). Most of the 14 columns are redundant with each other. **This is a narrower claim than "these features are useless"** — T1.10 measured that the one dominant direction each block does have separates a bearing fault (AUC 0.993/0.997) and an imbalance fault (AUC 0.965/0.907) from healthy at AUC > 0.9. On real data these features may carry more independent information than the simulator allows, or the same one dominant direction may simply move — either way it is unverified until hardware | `band_fractions` on the first real recording — a real machine in a real room should span materially more than 1.03 of 8. `tools/feature_block_report.py` recomputes the AUC table too |
| The three accelerometer axes on a REAL machine look anything like `capture.ACCEL_AXES` | T1.8 replaced copies with a plausible structure — two radial modes and a softer, more damped axial path, with a per-axis phase — but every constant in it is invented. If a real housing puts all three modes in the same band, or if magnet mounting swamps the axial axis entirely, the 12 accel statistics could collapse back towards 4. The model is now falsifiable, which is the improvement; it is not verified | H2 bring-up: `firmware/bench/check_mount.py`'s tap test gives f0 and Q per axis directly, then `python tools/accel_axis_report.py` on a real capture recomputes the same four numbers. Replace the constants with measured ones — do not leave a plausible invention in the code |
| A firmware or sensor change that preserves the feature COUNT is safe to deploy | **Partially addressed by T3.7 (2026-08-20).** The GROSS version of this — a change that leaves nearly every one of the first 8 windows anomalous at ≥5× the learn period's own tail, the T1.8 shape (100% of healthy windows at 138× threshold) — is now caught automatically at startup and refused with `BaselineMismatchError`, not silently scored. A SMALLER or partial drift (some windows shift, not all; a shift too gentle to clear the 5× bound in 8 windows) can still slip under it — the check is deliberately conservative to avoid crying wolf on a real fast-onset fault or one loud startup transient. The underlying rule is unchanged and still the safest one: retrain and score a batch of fresh healthy windows after ANY change upstream of the feature vector, don't rely on the check alone | H4's soak, and any future firmware change: watch whether the startup check ever fires on a real unit, and whether a change that should have been caught (by hindsight) was gentle enough to slip under the 5× bound — that would mean the bound needs retuning against real data, not simulated |
| Real machines have ≤ 4 clean regimes | false alarms on mode changes | week-3 soak |
| A real machine's within-regime wander is small compared to a regime change | T1.9's two criteria are calibrated in absolute physical units (5 % of speed, 0.1 decade of level) against a **simulated** within-regime scatter of ~0.001 in those units — essentially zero. A real machine drifting with load, ambient temperature or time of day could wander a good fraction of 0.1 decade inside one operating mode, in which case the absolute gate stops discriminating and only the 0.75 silhouette floor is holding the line. The failure would be visible: regimes appearing and disappearing between retrains | H4's 7-day soak. Log the operating point per window and plot its spread within a mode against the 0.1-decade / 5 %-speed scale. If within-mode wander is a sizeable fraction of that, `MIN_REGIME_SEPARATION` must be re-derived from the measurement instead of from the definition |
| The learn period's **Gaussian** is robust to contamination | T1.6 fixed the threshold estimator, not the model under it. A contaminated window still inflates the mean, the standardising std and the covariance, which moves every distance. Measured residual at n=24 on rank-deficient data: the deployed threshold still wanders **up to 1.4×** — bounded and no longer tracking the outlier's size, but not zero. A robust covariance (MinCovDet) is not available at n=24 with d=37 | H4's 7-day soak: if the contamination flag fires on real learn periods, this channel matters; if it never fires, it does not |
| The nominal 99.5 % quantile delivers a 0.5 % false-alarm rate | It does not, on our own simulator: **held-out healthy FPR was 3.8 % per window at n=24**, 7.6× the design point, because held-out windows come from a different draw than the learn period. The 4-window persistence gate is doing more work than the threshold is | H4 soak — the real false-alarms-per-node-week number, which is the only version of this that matters commercially |
| ~90 MB steady-state RAM on the Pi | OOM restarts | run it on the Pi |
| `provision_pi.sh` and `deploy_node.sh` actually work against a real Pi | Neither script has been run against a real `sshd`, real `systemd`, real `raspi-config`, or a real `/boot/firmware/config.txt` — both are verified only by syntax and by faking the parts of the environment that don't exist in this sandbox (`tests/test_provision_scripts.py`, `tests/test_deploy_node.py`). A real Pi could disagree in ways a fake `ssh` executable cannot catch: passwordless sudo not configured for `systemctl`/`journalctl` (the scripts assume it or an interactive prompt), a different Raspberry Pi OS release changing `raspi-config`'s non-interactive flag syntax or moving `config.txt` back to `/boot/`, or an actual PEP-668/venv interaction the `sed`-patched `ExecStart` doesn't anticipate | H2 week-1 bring-up: run `provision_pi.sh` for real, then `check_provision.sh` to audit the result, then a real `deploy_node.sh acoustic1.local` after a trivial code change and confirm the service actually restarts with the new code (e.g. a version string bump) |
| Physical severity **trends usably on a real machine** | T1.7's severity metrics are monotone in the *simulator's* severity parameter, which is a knob we turn, not a defect that grows. Load, temperature and speed also move band RMS, and none of that is in the simulation. If they dominate, "your machine is 3 dB worse than last week" is noise | H4's 7-day soak: plot band RMS and envelope-peak height per regime over a week on a healthy machine. If the healthy week-to-week spread is comparable to the 17.8 dB the simulator produces across the whole severity range, the trend metric is not usable as-is |
| The 30 s-window display index is what a customer should see | Partly settled by T1.11: the fleet sparkline now aggregates it hourly and the device page plots it per window with the alert line at 70. But the aggregate is a **mean**, chosen for one SQL expression that works on SQLite and Postgres alike, and a mean over 120 windows is exactly the statistic one loud minute distorts — the same argument that made T1.6 use the median. Nobody has looked at an hour of real windows to see whether it matters | H4's 7-day soak: compare hourly mean and hourly median of `display_index` on a real healthy machine. If they diverge, the sparkline needs a percentile, which SQLite cannot do in one query |
| No slow memory leak exists in the firmware loop | T4.1's soak (`docs/DOC_SOAK_MEMORY.md`) ran 3,200 windows in 6 separate OS processes, because this agent's sandbox cannot run one continuous process that long in a single shell call. That chunking sets a measured ~150-300 MB/week-equivalent noise floor (inter-process allocator variation alone), below which a real slow leak would be invisible to this method. What WAS ruled out: a gross leak, the kind that would OOM the 350 MB systemd cap within days to weeks — no chunk showed compounding, same-signed growth | H4's real 7-day soak, one continuous `systemd` process on a real Pi, has none of this noise by construction. Short of that: run `tools/memory_soak.py --windows 20000` as ONE uninterrupted local process (not chunked) overnight on any laptop — outside this agent's shell, nothing stops that |

## Not done

- **Headless Wi-Fi provisioning (T5.1) is implemented and unit-tested, not
  hardware-tested.** `tools/wifi_provision.py` (NetworkManager control:
  `is_connected`/`scan_networks`/`start_ap`/`stop_ap`/`connect_to_network`,
  a `tick()` state machine, a `--once`/`--max-iterations`-testable daemon
  loop) and `tools/wifi_portal_app.py` (the captive-portal FastAPI page)
  are both verified against a FAKE `nmcli` on PATH and FastAPI's
  `TestClient` — 35 new tests, all green — the same evidence class
  `scripts/deploy_node.sh` (T2.2) used for rsync/ssh. **None of it has run
  against a real wireless interface**: no `nmcli` binary exists in this
  sandbox at all (checked directly), so whether `nmcli device wifi
  hotspot` actually works on the Pi Zero 2 W's onboard chip, whether a
  phone's OS auto-detects the captive portal without the DNS-hijacking
  layer (deliberately not built — see `docs/WIFI_PROVISIONING.md`'s "What's
  designed but NOT built" section), and the systemd wiring to run any of
  this unattended, are all open until H2. Full design rationale, the three
  alternatives considered and why each was rejected, and the exact
  boundary between "implemented" and "designed only" are in
  `docs/WIFI_PROVISIONING.md`.
- **`firmware/state.py`'s `anomalies` table has no retention policy —
  it grows unbounded for the life of the device.** Found by T4.2's
  database-growth audit (`docs/DOC_SOAK_DB_GROWTH.md`) and confirmed by a
  fast direct regression test
  (`tests/test_db_growth_audit.py::test_anomalies_table_is_not_pruned_by_retention__real_finding`):
  `record_window`'s retention prune (`DELETE FROM readings WHERE ts < ...`)
  has no equivalent statement for `anomalies`, so an anomaly row survives
  indefinitely even after its own reading has been pruned. Slow at any
  realistic alert rate — the T4.2 audit's own pessimistic 1% per-window
  anomaly rate added only ~5.2 kB/day at steady state — but genuinely
  unbounded, and `docs/DOC_FIRMWARE.md` used to describe retention in a way
  that read as covering the whole database rather than just `readings`
  (corrected this run). Not fixed here: `state.py` is frozen, and this is a
  slow gap rather than the kind of active-harm bug (a silently-invalid
  baseline, an NTP jump wiping the readings table) the frozen-file exception
  exists for. If closed, the natural design is a SEPARATE, probably longer,
  retention window for `anomalies` — an audit trail of past alerts is
  plausibly worth keeping longer than the raw 30 s readings — which is a
  product decision, not one this audit makes unilaterally.
- **The shipped learn period is 48 minutes, not the 24–72 h `config.yaml`'s
  own comment promises.** Found writing T2.4's the operations runbook (not in this public copy).
  `window.learn_windows: 96` at the shipped 30 s window is what
  `firmware/baseline.py` defaults `--windows` to when none is given, and
  96 × 30 s = 48 minutes. Nothing currently orchestrates a real multi-day
  learn period automatically — an operator who wants the 24–72 h the comment
  describes has to pass `--windows` explicitly (e.g. `--windows 5760` for
  48 h) or edit `learn_windows` before deploying. Not fixed here: which
  default is actually right is a product decision (a shorter learn period
  demos faster but sees less real-world variation) rather than a
  documentation bug, so it is flagged rather than silently changed.
- **`firmware/train.py` reads a config key (`anomaly.sigma_k`) that does not
  exist in the shipped `firmware/config.yaml`.** Found while building T3.5's
  config schema; deliberately NOT added to the schema, because `train.py` is
  untested, unreferenced v1.5 autoencoder code (`grep -rln "train.py"
  tests/*.py docs/*.md` returns nothing) and is not part of the v1 Mahalanobis
  product's config contract. If `train.py` is ever revived, it needs its own
  schema entry — or `sigma_k` added to `config.yaml` — before it can run.
- **No sensor has ever been attached.** `HardwareSource` and `IIS3DWB` are
  datasheet transcriptions; unconfirmed constants are marked `UNVERIFIED`.
- **No real machine data.** Every number above comes from synthetic signals.
  In particular, `tools/ingest.py` has converted *simulated* recordings that
  were deliberately degraded to look like phone captures (44.1 kHz, stereo,
  int32, DC offset, 8 kHz band-limited). It has **never seen a real
  microphone**. What is proven is the conversion arithmetic and the audit, not
  that a real phone recording of a real motor carries a usable signal.
- **`docker-compose` never run** — no Docker daemon in the development
  sandbox. Verified with uvicorn + SQLite instead.
- **Frontend never exercised against a live backend in a browser.** T3.6
  (2026-08-20) closed the narrower, adjacent gap — does the real backend's
  JSON match what the frontend's own code (mock.js and the components) reads
  — via `tests/test_frontend_backend_integration.py` +
  `frontend/src/api/contract.test.mjs`. This row is about something that
  check cannot see: whether it actually RENDERS — CSS, layout, click
  handlers, whether recharts draws the axis it's given. Still unverified,
  still needs a browser this sandbox does not have.
- **Frontend severity chart never rendered in a browser.** T1.11 stored the
  fields, exposed them on the API and wrote the recharts panels, and the data
  transforms behind them are executed by `frontend/src/lib/trend.test.mjs`
  (18 checks, run inside pytest by `tests/test_frontend_trend.py`). The
  *pixels* are unverified: this sandbox has no browser and no DOM, and
  `npm run build` only proves the bundle compiles. First person with a screen
  should open the device page and confirm the two panels look like the numbers
  in the table below.
- **`env_peak_db` is computed but never published.** `physical_severity`
  calls it "the trendable one", yet `firmware/main.py`'s telemetry sends
  `env_peak_ratio` instead. Measured 2026-08-18, this is acceptable rather
  than a bug: the ratio is *exactly* gain-invariant (23.19 at gains 1/2/4/8)
  while `band_rms_db` tracks gain 1:1 (−23.26 → −5.20 dB over the same
  gains), so the two published fields between them span level and contrast
  and determine `env_peak_db`. Recorded so nobody re-derives it. Pinned by
  `test_env_peak_ratio_is_a_gain_invariant_ratio_not_a_level`.
- **Public real-bearing dataset validation: TOOL DONE, DATA NOT YET RUN.**
  `ml/realdata/validate_public_dataset.py` exists, is tested (14 tests) and
  runs the full `extract_features → fit_baseline → MahalanobisScorer` path
  from a directory of CWRU `.mat` files. It has **never seen the real CWRU
  data**, because the agent sandbox has no outbound DNS — `getent hosts
  engineering.case.edu` fails, and only the pip proxy resolves. Everything
  below the line labelled SURROGATE was measured on a synthetic dataset this
  file generates itself.

  **This is a five-minute job for Logan and it is the highest-value software
  task left:** `python ml/realdata/validate_public_dataset.py --help-download`
  prints the recipe, then one command produces the first number this project
  owns that did not come from its own simulator.

  Licence note recorded: the CWRU Bearing Data Center publishes the files
  openly but states **no licence** — the footer carries only "© Case Western
  Reserve University". So: acknowledge the source in the report and the deck,
  and **never commit the `.mat` files** (`.gitignore` covers `data/`).
- **`scripts/`** (Pi provisioning, deployment, dev-up) not written.
- **v1.5 cloud autoencoder** written but never trained end to end.

## What the surrogate run taught us anyway (2026-08-16)

The surrogate is not real data and proves no physics. But it is an
*independently written* signal model — pink noise floor, 60 Hz mains, two
structural resonances, double-exponential impacts, load-zone AM, and it does
not import `ml/simulate.py`. Running our detector against someone else's
assumptions, even our own other assumptions, moved two rows on this page.

| Finding | Number | Why it matters |
|---|---|---|
| **The protrugram never fired.** `select_demodulation_band`'s `crest_floor = 10` was not reached on *any* surrogate file (best crest 4.6–8.0), so every window fell back to the default 3–6 kHz band. | crest 5.2–7.8 vs floor 10.0 | On a pink noise floor the envelope-spectrum crest is much lower than on the band-limited white noise `simulate.py` uses. The fallback happened to contain the 3.4 kHz resonance, so nothing broke — **on a real machine whose resonance sits outside 3–6 kHz it would go quietly deaf.** `check_mount.py`'s tap test is now load-bearing, and `crest_floor` should be re-tuned on the first real recording, not before. |
| **Envelope contrast collapses on a harder signal.** | **5.8×** at BPFO (0.007″ outer race) vs **56.7×** on `verify_signals.py` — and **1.3×** on the healthy control | The 56.7× headline is a property of the simulation's SNR, not of envelope analysis. Expect single-digit contrast on the bench. It is still a 4.5× separation from the healthy control, which is what actually matters, but the pitch deck should quote the ratio to control, not the absolute number. |
| **False positives at 1 s windows are not negligible.** | FPR 0.125 on 16 held-out healthy windows (2 windows), TPR 0.967 | Small sample, so treat as "order 10 %", not "12.5 %". Production uses 30 s windows and a 30-minute persistence gate, which is exactly the defence this number argues for. Do not quote a per-window FPR as a product number. |

**Follow-up, T7.2 (2026-08-20):** the "it would go quietly deaf" prediction
above is now measured, not just predicted, on the microphone/phone domain —
`ml/realdata/synth_phone_recording.py` puts a resonance outside 3–6 kHz on
purpose and the fallback does exactly this at moderate severity. It is
**severity-gated**, not permanent: crest crosses the floor and the true band
is found unaided once the fault is strong enough (crest 13–20). See
`docs/PHONE_RECORDING.md` and the new rows in the proven table above.

## What building the ingest tool taught us (2026-08-17)

`tools/ingest.py` was written for T1.2 and its own tests immediately found
three defects. All three are the same species: **code that appears to work and
silently corrupts a number.** None was hypothetical; each is quoted with the
measurement that exposed it.

| Finding | Number | Why it matters |
|---|---|---|
| **A 6-decimal-place `t_s` column IS a sample-rate error.** `recording_io` and `firmware/capture.FileSource` both recover the accelerometer rate as `1/median(diff(t_s))`, so the printed precision of that column *is* the rate. At 6400 Hz the step is 0.00015625 s; written as `%.6f` it becomes `0.000156` and reads back as **6410.26 Hz**. | **+0.16 %** stretch of the accelerometer frequency axis | `ingest.py` now writes 9 dp and the round-trip rate is exactly 6400.000000 Hz (asserted in `tests/test_ingest.py`). **`ml/simulate.py`'s `export_accel_csv` has the same 6-dp habit, so every file in `data/*_accel.csv` loads at 6410.3 Hz today.** 0.16 % changes no published result — accel features are eight band-energy *ratios* plus four statistics — so `simulate.py` is left frozen and this is recorded rather than fixed. Fix it if an accel-side frequency ever gets quoted to better than 0.5 %. |
| **Trial-and-error CSV parsing eats data.** The obvious `skiprows=1`-then-fall-back idiom *succeeds* on a headerless numeric file and discards the first row. | 1 sample in every headerless CSV, silently | Header presence is now decided by trying to parse the first line as floats. Regression-tested. Worth remembering when the first dataset export arrives. |
| **A clipping check needs to know what full scale is.** `|x| ≥ 0.999` is right for an integer wav and meaningless for a CSV in g: the repo's own `data/bearing_outer_accel.csv` peaks at 1.674 g and was reported as **20.0 % clipped**. | 20.0 % → 0.002 % after the fix | A warning that cries wolf on good data trains you to ignore it, and clipping is the one artefact that *manufactures* a bearing fault (a flat top is a broadband impulse). Physical-unit inputs are now tested for many samples at the same extreme value instead. |

And one measurement worth keeping, because it converts an assumption into a
number: the **"machine resonance is near 3–6 kHz"** row below is usually argued
for from the datasheet. Band-limiting the *same* faulty recording to an 8 kHz
phone rate — which leaves only a third of the 3–6 kHz demodulation band —
collapses the BPFO envelope ratio from **35.8x to 1.2x** and fails all three
Gate-2 checks. Losing part of the resonance band does not degrade detection
gracefully; it removes it. That is why `check_mount.py`'s tap test is on the
critical path, and why `ingest.py` refuses to be quiet about a narrowband
source.

## What testing the fault-frequency maths taught us (2026-08-17)

`tests/test_realdata.py` was written for T1.3. Writing it disproved one claim
in the code and produced an error budget that changes what you should worry
about at the bench.

| Finding | Number | Why it matters |
|---|---|---|
| **A guessed pitch diameter is nearly free.** `fault_frequencies.py`'s `6205` entry claimed the (bore+OD)/2 estimate costs "~1.4 % in BPFO". It does not. The 1.4 % is in **D**; BPFO carries (1 − γ), so the sensitivity is γ/(1 − γ) = **0.255**, not 1. | 1.38 % in D → **0.358 % in BPFO** = **0.64 Hz** at f_r = 50 Hz (predicted 0.353 %, measured 0.358 %) | 0.64 Hz is ~6× *inside* the ±2 % (±3.6 Hz) slip window the analysis already searches. Estimated geometry will **not** be what makes Week 2 inconclusive, so do not spend bench time hunting a manufacturer's drawing. BPFI is even less sensitive (γ/(1 + γ) = 0.169). Comment corrected in the source. |
| **The ball count is the parameter that hurts.** BPFO scales linearly with N. | miscount 8 as 9 → **12.5 %**, **19 Hz** at f_r = 50 Hz | Four times the slip window — a completely different peak. **Count the balls through the seal gap; it takes ten seconds.** Shaft speed is the other 1:1 term, which is why `analyse_recording.py` refuses to run without `--rpm` and says "MEASURE it". |
| **A plausible bearing can be permanently unmeasurable.** N = 8 with d/D = 0.25 gives BPFO = (N/2)(1 − γ) = **3.00 · f_r exactly, at every speed** — the outer-race line and the 3rd shaft harmonic are the same line. | γ = 0.25 and N = 8 are both squarely normal for a deep-groove ball bearing | The report's standard advice for INCONCLUSIVE — "re-run 10 % faster so the lines separate" — is **wrong in this case**, because the ratio is speed-independent. Check `--bearing X --rpm Y` for a near-integer BPFO multiplier *before* seeding a defect. The repo's own 6204 is fine: 3.0519, i.e. 2.6 Hz clear at f_r = 50. |
| **The 3-bin guard is load-bearing, not a magic number.** | patched from `3*df` to 1 bin, the search returns a leakage skirt at **149.80 Hz (40x)** instead of the real line at 152.60 Hz (12x) | Masking only the peak bin leaves the Hann main lobe's skirts, which are still taller than an early fault. Verified by patching the constant and re-running, not by argument. |

## What fixing the compositional dependency taught us (2026-08-17)

T1.5 replaced three blocks of energy fractions with isometric log-ratio
coordinates (40 dims → 37). The change is sound and the tests are worth having,
but **the honest headline is that it bought no measurable detection
improvement**, and the investigation was worth far more than the fix.

| Finding | Number | Why it matters |
|---|---|---|
| **The prescribed fix would not have worked.** The backlog and F1 both said "CLR or drop one band". CLR coordinates sum to zero by construction, so a CLR-transformed block is still exactly rank D−1 in D columns. | CLR block sv ratio **< 1e-12**; ILR **0.50** on the same data | The famous transform is not always the right one. Pinned by a test so nobody "simplifies" the code back into the bug. |
| **F1 mislocated the defect.** The band features are `log10` of fractions, so their constraint is non-linear; there was **one** exactly-singular block, and it was the one F1 never mentioned (the 6 raw envelope fractions). | env null direction = uniform vector, \|cos\| **1.0000**; band blocks' null direction = mean-fraction weights, \|cos\| **1.0000** | Same symptom, two different mechanisms. Diagnose before prescribing — the fix depends on which one you have. |
| **"Effective rank" is not an information measure.** After the fix, effective rank *fell* (17.4/40 → 9.0/37) and the condition number *rose* (3324 → 8359). | information unchanged; AUC 1.000 → 1.000 | Eight separately-noisy log-fractions spread variance over eight directions while carrying about one direction of signal. F1's central statistic was counting noise. **The metric that motivated the task could not have detected whether the task succeeded.** |
| **A condition number is a summary, not a diagnosis.** cond(precision) rose 8.1 → 18.8, and yet the score's reliance on the tightest directions *fell*. | top-5 tightest directions: **31.6 % → 6.1 %** of d² on healthy windows | The property F4 actually cared about improved 5× while the number usually quoted for it got worse. Re-measure the property, not the proxy. |
| **Changing a feature contract is a deployment hazard.** A pre-T1.5 baseline against post-T1.5 firmware produced a raw numpy broadcast error from inside `score()`. | `ValueError: operands could not be broadcast together with shapes (37,) (40,)` | Failing safe is not enough when the person reading the log is a student at a customer site. `MahalanobisScorer` now names both dimensions and prints the retrain command. |

And the finding that outweighs all of the above, recorded as **F9**: chasing the
residual degeneracy showed that `simulate.py`'s band composition spans **1.03 of
8 dimensions**, so the 14 band-energy features carry roughly 2 dimensions of
information between them. That is ~6 dimensions per block, against the 1 per
block that F1 was about. It is the third independent sign that the simulator has
fewer degrees of freedom than the feature vector claims to measure — after F6
(accelerometer axes that are copies) and F5 (a score with no amber zone).
**The feature vector cannot be evaluated on this simulator.** Only hardware
settles it.

**Updated 2026-08-18.** T1.8 removed one of the three signs: the accelerometer
axes are no longer copies (effective rank 3.75 → **9.32 of 12**), so the count
of features with no demonstrated independent contribution falls from ~26 of 37
to **14 of 37**, all of them band-energy features. The conclusion is unchanged
in kind — 14 is still more than a third of the vector, and enriching the
simulator would move the number without making it true. F9 stands; it is now
the *only* remaining instance, and the thing that settles it is still one real
recording. Per-group status is tabulated in `docs/DOC_PIPELINE.md`.

## What giving the accelerometer three real axes taught us (2026-08-18)

T1.8 acted on self-review finding **F6**. It is the first finding to survive
being tested — F1, F3 and F5 were all disproved by the work they motivated —
and confirming one turns out to teach as much as refuting one.

| Finding | Number | Why it matters |
|---|---|---|
| **F6 was right, to two significant figures.** It claimed the 12 accelerometer features carried "roughly 4 features of information". | measured **effective rank 3.75 of 12**, with the smallest four singular values at 1.3e-3 of the largest — a 4-dimensional near-null space, exactly what duplicating one axis twice produces | The review process works when the reviewer measures. F6 quoted a correlation matrix; F1, F3 and F5 quoted arguments or single-seed points, and all three were wrong. **Quote a measurement.** |
| **The finding named the wrong file.** F6 and the backlog both said `ml/simulate.py` was where the copying lived, and the task text carried a note about needing a frozen-file exception. | the three lines were in `firmware/capture.py`, which is **not frozen**; `ml/simulate.py` is untouched by T1.8 | Diagnose before prescribing — the second time this exact lesson has been recorded (T1.5 found F1 had mislocated its defect too). The frozen-file rule cost nothing here only because the note was checked rather than obeyed. |
| **Different resonances did not decorrelate the axes. Different PHASE did.** The first implementation gave each axis its own housing mode (f0, Q, gain) and independent sensor noise. | the axes still correlated at **r(x,z) = 0.904** | On a healthy machine the shaft hum sits 20 dB above everything else, so correlation is set by the hum, not by the fault. To decorrelate a signal you must decorrelate whatever dominates its **variance** — which is not the thing you are trying to detect. Fixed with a per-axis phase relative to the rotating imbalance vector. |
| **Fixing it changed detection by nothing measurable.** | held-out healthy FPR **0.0292 → 0.0319**, paired 95 % CI [−0.084, +0.104]; AUC **1.000 → 1.000** | Exactly what T1.5 found for the compositional fix, and for the same reason: the simulator is too easy to register an improvement. **The value is that the accelerometer features can now be wrong.** A test that cannot fail is not evidence, and before T1.8 no accelerometer claim could have failed. |
| **A simulator change silently invalidated the shipped baseline.** The feature vector stayed 37-dimensional, so T1.5's dimension guard — added precisely to catch this class of bug — passed. | 40 fresh **healthy** windows scored median **138.4×** their threshold; **100 %** above threshold | This is a deployment bug, not a simulation artefact: a firmware update that changes the sensor path while keeping the feature count would make every unit in the field alarm continuously, and the only guard we have would say nothing. Backlog **T3.7**. The immediate fix was to retrain: thresholds 8.348 / 9.882 → **8.069 / 9.380**. |
| **A 24-window learn period is not viable.** Discovered by accident, running the comparison at the wrong operating point. | held-out FPR **0.55–0.59** at 12 windows per regime, vs **0.03** at 24 per regime | 37 features need more than 12 samples per regime, Ledoit–Wolf shrinkage notwithstanding. If a customer's learn period is cut short, or a regime is rare enough to collect only a handful of windows, that regime alarms on everything. Related to the `MIN_REGIME_WINDOWS = 8` floor T1.6 added — 8 is enough to avoid a singular fit, and nowhere near enough to be *useful*. |

## What checking mic-only regime clustering taught us (2026-08-18)

T1.9 acted on self-review finding **F7**, which called the problem "not
fatal". It was the most expensive finding in the file.

| Finding | Number | Why it matters |
|---|---|---|
| **A model-selection threshold is only valid in the dimensionality it was tuned in.** `SILHOUETTE_MIN = 0.5` was chosen where a real split scores 0.8+, in 3 dimensions. | on data with **no regimes at all**, median best silhouette is **0.584** in one effective dimension and **0.283** in three; maximum over 1500 trials **0.702** | The threshold was below the noise floor of the statistic it thresholds. Nothing in the code was wrong in 3-D — the number simply stopped meaning what it meant when a channel died. Any constant tuned against one data geometry needs re-checking when the geometry changes, and the cheapest check is to run the estimator on data you know has no structure. |
| **The prescribed fix was a no-op.** F7 proposed dropping dead dimensions before clustering. | k-means labels and silhouette **bit-identical** with the dead column present or removed — a constant column contributes exactly 0 to every distance | Third time a self-review finding has prescribed a fix that measurement rejected (F1's CLR, F3's χ²(p), now F7's dimension-dropping). The pattern is always the same: the finding identifies a real smell and then guesses at the mechanism. Write the experiment before writing the fix. |
| **The bug was not confined to mic-only.** | two live channels correlated at 0.98 are **one** effective dimension and split noise **98.5 %** of the time — the same as mic-only | Our simulator's accel level is independent of its audio level (r = **0.11**); a real machine's will not be, because both track load. The recommended build looked safe only because of an artefact of the simulator. **Ask what makes the passing case pass** before concluding the failure is a corner case. |
| **"Not fatal" cost 6.3× the false alarms.** | held-out healthy FPR **0.1358 ± 0.1445** vs **0.0217 ± 0.0290** at the correct k=1; k > 1 chosen in **100 of 100** bootstrap learn periods | Churn risk #1 is false alarms, and this was a defect in the recommended build that fired on essentially every unit. A finding's own severity estimate is a guess until someone measures it — F7's was wrong by a factor of six in the direction that mattered. |
| **One defect manufactured a second, false diagnostic.** | T1.6's learn-period-contamination warning fired on **14 of 200** perfectly clean fits, purely because the data had been split | Diagnostics inherit the assumptions of everything upstream. A warning that fires on clean data trains its reader to ignore it, which is worse than not having it. |
| **Two criteria, because each missed a case.** An absolute physical gate (0.1 decade / 5 % speed) and a dimension-aware silhouette floor (0.75). | the deployed failure passes the relative test (1.5σ of the noise) and fails the absolute one (0.0002 decades); a genuinely wandering machine does the reverse | Written down because the temptation was to ship the first criterion — it fixed the observed failure completely, and the FPR number was already the headline. It took deliberately constructing the mirror-image case to find that it was half a fix. **The bug you measured is not the whole bug class.** |

## What quantifying feature-block dimensionality taught us (2026-08-18)

T1.10 acted on self-review finding **F9**, which measured that the audio and
accel band-ILR blocks span roughly one dimension each on simulated healthy
data and left an open question: does that mean those 14 columns are along for
the ride? `tools/feature_block_report.py` and `tests/test_feature_blocks.py`
answer it directly rather than by further argument, by training a per-block
Mahalanobis distance on healthy-only windows and scoring it against held-out
healthy and faulty windows — for two unrelated fault kinds, not just the
bearing fault F9 was measured against.

| Finding | Number | Why it matters |
|---|---|---|
| **Low rank is not the same as no information.** Both band-ILR blocks stay near one-dimensional on healthy data (effective rank 2.31/7, 1.53/7), matching F9. | AUC **0.993 / 0.997** detecting a bearing fault, **0.965 / 0.907** detecting an unrelated imbalance fault, trained on healthy-only data | F9's number is still true and still means most of the 14 columns are redundant with each other — but the one dominant direction each block does have turns out to be exactly the direction that moves when a fault appears, for two different fault mechanisms. "Unproven independence between columns" and "no detection value" are different claims, and only the first one was actually measured before now. |
| **High rank is not the same as informative, either — the mirror image.** The envelope block, which does most of the bearing-fault detection work, was tested against an imbalance fault it was never designed to see. | effective rank **6.63 of 7** under an imbalance ramp (not rank-suppressed) yet AUC **0.447** — chance | Envelope crest and the envelope-band ILR coordinates measure impact *periodicity*; an imbalance fault has none, it is a smoothly growing 1x tone. The block varies plenty (hence full rank) without any of that variance being about the fault. This is T1.5's "effective rank is not an information measure" confirmed from the opposite direction: T1.5 found low rank did not mean low information, T1.10 finds high rank does not mean high information either. Rank measures how variance is spread; it never measures what the variance is *of*. |
| **No block tested is universally along for the ride — but no block is universally useful either.** | worst per-fault AUC for any block was 0.907 (accel_band_ilr on imbalance); the envelope block's 0.447 was the only near-chance result anywhere in the 5×2 table | Which block does the detection work depends entirely on which fault you ask about. A bearing fault and an imbalance fault are structurally different failure modes (impacts vs. a growing tone), and the feature vector's five blocks divide the work between them rather than all pulling for every fault. That is an argument FOR keeping all five blocks, not against — the fact that F9's low-rank blocks turned out useful for two different fault types is a point in their favour, not a discharge of F9's concern about within-block redundancy. |
| **The per-channel statistics blocks are the steadiest performers.** | audio_stat and accel_stat: AUC ≥ 0.968 on both fault kinds tested | These four-numbers-per-channel features need no band selection, no periodicity assumption, nothing about geometry — they are the simplest features in the vector and, on this evidence, the hardest to fool. Consistent with the system overview (not in this public copy) §5's framing that detection is the easy part; what varies block to block is which *kind* of deviation each one is built to catch. |

**What this does not settle.** Two fault kinds, one simulator, one set of
invented physical constants. A real bearing fault, a real imbalance, and a
real misalignment may excite these blocks in combinations this simulator does
not model — in particular, `ml/simulate.py`'s imbalance signal is a pure
growing sinusoid with no harmonics or envelope structure of its own, which is
why it lands so cleanly on the per-channel statistics and band blocks and not
at all on the envelope block. A real unbalanced rotor may behave differently.
Re-run `tools/feature_block_report.py` on the first real triaxial capture
alongside `tools/accel_axis_report.py`.

## What the fault-injection audit taught us (2026-08-19)

T4.3 was promoted to top priority by the 2026-08-19 backlog override: with
hardware indefinitely blocked, the highest-value remaining engineering work
is guaranteeing that when something goes wrong, the software says so rather
than returning a plausible wrong number — the F2 lesson (a dead
accelerometer returning a believable 10 Hz shaft speed) made systematic
across five named scenarios. `tests/test_fault_injection.py` executes each
one for real rather than reasoning about it.

| Finding | Number | Why it matters |
|---|---|---|
| **A corrupted baseline could make a unit permanently silent.** `MahalanobisScorer` loaded a `baseline.npz` with a NaN in it (a bad write, a failing SD card) without error. | `score > threshold` is IEEE754-`False` whenever either operand is NaN | A NaN threshold or precision matrix does not make the score NaN-and-obviously-wrong — it makes `anomalous` **False**, silently, on every window, forever. This is the F2 failure shape arriving through file corruption instead of a dead sensor: a plausible-looking result produced by data that measured nothing. Fixed by validating schema and finiteness at load time; garbage/truncated files now raise `ValueError` naming the retrain command instead of a raw `BadZipFile`/`KeyError`. |
| **An NTP forward step could delete an entire fleet unit's history in one call.** `StateDB` pruned retention using the current window's raw wall-clock timestamp. | a simulated 2-year forward step (a Pi with no RTC syncing NTP after boot) deleted **100 %** of rows written seconds earlier | DOC_FIRMWARE.md documents "at most one window is lost because state lives in SQLite" as the crash-recovery contract. A clock step breaks that assumption in a completely different way — no crash at all, just silent, total data loss on the next successful call. Fixed by bounding the prune cutoff with `time.monotonic()`-tracked elapsed time, which a wall-clock step does not touch, while the row itself keeps the corrected wall time for the dashboard. |
| **A failed write used to leak into a later, unrelated commit.** Found testing disk-full, not one of the five named scenarios but the same shape: `record_window`'s statements shared one SQLite implicit transaction with no `rollback()` on the exception path. | measured directly: a fault window whose anomaly insert failed, followed by an ordinary successful window, left **`[2000.0, 2030.0]`** on disk when only `[2030.0]` should be there | The failed fault window's reading was not gone, it was *pending* — the very next successful call's `commit()` flushed it too, reappearing as a bare reading with no anomaly record: a real bearing fault silently reclassified as unremarkable because the disk happened to be full for one write. Fixed with an explicit `rollback()` on the exception path, which makes "one call, one atomic unit" true regardless of what the caller does with the exception (crash-and-restart today, some future retry loop tomorrow). |
| **Three of the five scenarios needed no code change — verified, not assumed.** | disk-full (once the rollback fix above is in): all-or-nothing, confirmed by three separate injection points (reading insert, anomaly insert, retention delete). Broker-unreachable-for-days: telemetry dropped, anomalies queued and replayed in FIFO order, overflow evicts the oldest without raising — confirmed against `mqtt_client.py`'s own module docstring rather than re-read. Sensor dying mid-run: a REAL simulated signal path (not a synthetic Gaussian fixture) through the real scorer, with the accelerometer or microphone zeroed partway through a run, reliably flips the score anomalous within one window of the channel dying, and `fr_reliable` drops with it. | "Cannot predict what breaks" cuts both ways — three of five plausible failure modes turned out to already be handled correctly by design decisions made for other reasons (SQLite's transaction model, the bounded offline queue, the dead-channel sentinel in `channel_stats`). Recording that a scenario was tested and found already-safe is exactly as valuable as recording a bug, and cheaper to act on: nothing to fix, one fewer unknown at the bench. |
| **One silent gap was hardened without changing behaviour.** `mqtt_client.py`'s offline anomaly queue is a `deque(maxlen=500)`; appending past capacity silently evicts the oldest entry. | not frozen, so no exception was needed | 500 anomaly events is unrealistic given the persistence gate (at most a handful of alerts per unit per year), but a multi-day broker outage is exactly the T4.3 scenario, and a silent drop is the one outcome the whole audit exists to catch. Added one `log.warning` on eviction; the eviction itself (oldest-first, unbounded acceptance of new events) is unchanged, because it is already the documented, correct trade-off. |

**What this does not settle.** All five scenarios were exercised with
mocked I/O failures or synthetic clock/signal manipulation, not a real disk,
a real NTP daemon, or real hardware coming unplugged — the sandbox has none
of those. The fixes address the mechanism each scenario demonstrated
(transaction atomicity, clock-step bounding, load-time validation), which is
sound regardless of exactly how a real failure is triggered, but "degrades
safely on the bench" is still an open question for H2–H4.

## How to read "ROC AUC 1.000"

It means **the simulation is easy**, not that the product works. The simulator
was written by the same project that wrote the detector, using the same
assumed physics; it cannot falsify those assumptions. Real bearings are
noisier, real mounting is imperfect, real factories have forklifts.

Expect the first real number to be materially worse. **That is not failure —
that is the experiment finally being real.** The version of this table
produced after week 2 is the one worth showing anybody.

## The three questions that decide the project

1. **Week 2:** does the envelope signature appear on a real seeded fault?
2. **Week 3:** are false alarms ≤ 1 per node-week on a real healthy machine?
3. **Month 3:** will anyone host a sensor for free?

Kill criteria for each are in the execution plan (not in this public copy).

## Updating this file

After every bench session, move rows from "assumed" to "proven" **with the
measured number attached**, or record the failure. A project whose status doc
only ever gains green ticks is a project that has stopped testing itself.
