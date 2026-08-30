# Adversarial self-review — hunting our own mistakes

Round 1, 2026-08-16. Method: form a specific suspicion about the detector,
then **write code to try to prove it**, rather than reasoning about it. Six
suspicions tested. Five confirmed, one refuted, one fixed on the spot.

The refuted one is recorded here deliberately. A review that only ever
confirms its author's suspicions is not a review.

---

## ⚠ F-number register — read before allocating a new one

**This file is the single allocator of F-numbers. Nothing else may mint one.**

That rule exists because it was broken. On 2026-08-23 an audit found **F12 and
F13 each meant two different things** depending on which file you opened:

| # | In this file (canonical) | In the commit log (not in this public copy) (2026-08-20) |
|---|---|---|
| F12 | mic driver power-on "thump" | the suite was 483+1 error, not 484 |
| F13 | `HardwareSource` diagnostics nobody reads | CI learned the baseline *after* running the tests |

Two processes were allocating numbers into two different files with no shared
counter: the daily-review agent takes the next number from here, while an
interactive session took the next number from the commit log. Both were
"correct" locally and the project ended up with two F12s.

**Resolution.** The daily reviewer's numbering stands, because this file is the
register. The three interactive findings are re-registered below as **F14–F16**
and cross-referenced. the commit log (not in this public copy) is a historical log of what was said at
the time and has **not** been rewritten — rewriting it would destroy the audit
trail, which is worth more than tidy numbering. If you are reading a commit
message from 2026-08-20 that says "F12", it means F15 here.

- **F14** ← commit-log "F12": `conftest.py`/`test_api.py` hardcoded
  `/tmp/test_acoustic.db`, so a leftover owned by another uid made the whole
  backend suite error; plus a deleted inode left in SQLAlchemy's pool. Fixed
  (uid-qualified paths + `engine.dispose()`). Invisible to CI, which gets a
  clean `/tmp` — green CI is not evidence for this class.
- **F15** ← commit-log "F13": `.github/workflows/ci.yml` ran `pytest` *before*
  generating the gitignored `firmware/baseline.npz`, so CI runs #1–#3 failed
  with 5 failures and 18 errors while every dev machine stayed green — the case
  existed only on a machine that had never run anything, i.e. a new
  contributor's. Fixed (step reordered + a legible `pytest.exit` guard).
- **F16** ← commit-log "F14": a **retraction**. An earlier claim that four tests
  were genuinely failing was wrong; they were an artefact of the reviewing
  sandbox running out of disk. Kept visible rather than deleted.

**When you find something new:** take the next free number *here*, in this file,
first. Then reference it from anywhere else.

---

## F28 — The clipping guard was blind to every lossy recording, and a clipped HEALTHY machine reads as a fault ✅ FIXED (T1.16 #1, 2026-08-30)

Filed by the 2026-08-28 adversarial review as T1.16 #1 and left open for two
days as "needs a genuinely codec-robust measure". Fixed 2026-08-30.

**The row understated it, and that is the finding worth keeping.** T1.16 #1
described a missing hygiene warning: clipping is detected before a codec and
not after. What it did not say is what the missing warning costs. Measured
this run, through a real ffmpeg AAC 128 kbps round trip:

| recording | comb score | reported f0 | flat-top | warned? |
|---|---|---|---|---|
| `normal.wav`, 6 dB headroom (HEALTHY) | 5.5 | 17.75 Hz | 0.00000 | no ✔ |
| `normal.wav`, driven into clipping | **51.7** | **49.75 Hz** | 0.00063 | **no ✘** |
| `bearing_outer.wav`, headroom | 31.7 | 152.25 Hz (true) | 0.00000 | no ✔ |
| `bearing_outer.wav`, clipping | **94.1** | **99.75 Hz (wrong)** | 0.00035 | **no ✘** |

`cold_start_screen.py`'s own documentation calls ~35 "a real fault". So a
healthy machine, recorded slightly too loud on a phone, was reported as a
fault scoring 51.7 at a specific plausible frequency, silently. And a genuine
fault did not merely exaggerate — the reported frequency moved off the true
BPFO onto a harmonic. Both confirmed against the pristine
`git show HEAD:tools/cold_start_screen.py`, not inferred.

**Why the flat-top test cannot see it.** `clipped_fraction` requires
bit-identical neighbouring samples. A lossy codec perturbs every sample, so
the evidence is gone: 0.789 pre-codec → 0.00067 post, under a 0.001 floor.

**Why true peak can.** The same operation that destroys the sample-level
evidence creates a different one. A codec band-limits and re-synthesises;
band-limiting a waveform with flat tops is textbook Gibbs conditions, so the
reconstruction *overshoots* the level the signal was sliced at. Measuring
that overshoot between samples is the standard true-peak measurement of
ITU-R BS.1770-4 / EBU R128. It is needed because ffmpeg's default 16-bit
decode hard-limits the overshoot back to ±1.0: at the samples the peak reads
0.00 dBTP, and only after 4x oversampling does it read +6.3.

**Three candidates were measured and rejected first**, recorded so nobody
retries them (full numbers in `clipped_fraction`'s docstring):

  * **odd-harmonic ratio** — the row's own first suggestion. Superb on tones
    (clean 4.32 vs clipped 682555) and *useless* on broadband: 0.72 clean vs
    0.72 clipped on the same fan recording. Real machine audio is broadband.
  * **spectral flatness** — the row's second suggestion. Moves the wrong way:
    a codec *removes* high-frequency content, lowering flatness, while
    clipping raises it. The two effects fight.
  * **amplitude-histogram plateau prominence** — this run's own idea, and it
    fails for a reason worth stating because it kills the whole family: a
    sinusoid's amplitude density *also* diverges at its own peak, so tonal
    and clipped audio are not separable in the amplitude domain at all.
    Measured p99.9/p90 puts clipped-post-AAC (1.126) **between** clean sine
    (1.208) and `normal.wav` (1.261). This is the same root cause that
    already killed the "level-domain" candidate in 2026-08-28's rejection
    list; it was rediscovered from a different direction, which is mild
    evidence the earlier rejection was right rather than unlucky.

**A fourth fix was tried and measured counterproductive**, which is the
non-obvious one: making the converters decode `pcm_f32le` so ffmpeg stops
hard-limiting the overshoot. It makes `clipped_fraction` *strictly worse* —
0.00165 (warned) → 0.00000 (missed) on the realistic case — because 16-bit
re-clipping is precisely what accidentally restores some bit-identical
neighbours. True peak reads +3.27 dBTP under either decode. No converter was
touched.

**The trap this nearly shipped with.** True peak cannot be computed inside
`screen()`: by then the caller has normalised to peak 1.0, which is 0 dBTP
for every signal in existence. The first draft of the false-positive guard
test used the test module's normalising `_load` helper and returned +0.01
dBTP on all four control signals — a clean sweep of false positives that
looked like the metric failing rather than the test being wrong. Now pinned
by `test_normalising_before_measuring_would_flag_everything`, and the
scaling factored into a shared `full_scale_float()` so `main()` and the
tests cannot diverge.

⚠ **STILL OPEN, and deliberately not papered over.** A clipped **pure tone**
clears the 0.0 dBTP threshold by **0.05 dB**, against 3-6 dB for anything
broadband — a narrowband waveform gains almost no inter-sample overshoot
from clipping. `clipped_fraction` covers the tonal case exactly on
un-transcoded WAV, which is why it was kept rather than replaced, and
`screen()` warns on either test. A clipped tone that has *also* been through
a codec is caught by neither. Not filed as a task: there is no evidence yet
that it bites, and the machines in scope are broadband.

---

## F27 — `firmware/main.py` shadows `backend/main.py`, and the suite is green only because `test_api.py` sorts first alphabetically ✅ FIXED (T3.1, 2026-08-29)

Measured 2026-08-29 (daily review). **Same class as F12: a cross-file ordering
bug that a full-suite run structurally cannot see.**

`main.py` is the **only** duplicate module basename across `firmware/`, `ml/`
and `backend/` (verified by enumerating all three directories). Four test files
import it bare — `tests/test_api.py:27` and
`tests/test_frontend_backend_integration.py:72` both do `from main import app`,
expecting `backend/main.py`.

`conftest.py` makes that work by leaving `backend` ahead of `firmware` on
`sys.path`. But `tests/test_evaluate_pinned.py:49-51` and
`tests/test_fault_injection.py:80-82` each re-insert `ml` and `firmware` at
position 0 at module scope. Measured `sys.path[0:3]` before and after
collecting `test_evaluate_pinned.py`:

    before:  [<root>/backend, <root>/ml, <root>/firmware]
    after:   [<root>/ml, <root>/firmware, <root>/firmware]   <- backend fell to index 5

Their `if _p not in sys.path` guard does not prevent this; the entries end up
duplicated and `backend` is demoted below `firmware`. `from main import app`
then binds `firmware/main.py` and raises:

    ImportError: cannot import name 'app' from 'main' (<root>/firmware/main.py)

**Measured consequences** (all reproduced, `--collect-only`, this checkout):

| command | result |
|---|---|
| `pytest tests/` | **676 collected, 0 errors** ← green |
| `pytest tests/ --ignore=tests/test_api.py` | 662 collected, **1 error** |
| `pytest tests/test_evaluate_pinned.py tests/test_api.py` | 7 collected, **1 error** |
| `pytest tests/test_evaluate_pinned.py tests/test_frontend_backend_integration.py` | 7 collected, **1 error** |
| `pytest tests/test_fault_injection.py tests/test_api.py` | **1 error** |

**Why the full suite passes.** pytest collects in sorted path order, so
`test_api.py` is imported before `test_evaluate_pinned.py` and caches
`sys.modules["main"] = backend/main.py` while `sys.path` is still intact. Every
later bare `import main` gets the cached module. **The suite's greenness is a
property of the string `test_api.py` sorting before `test_evaluate_pinned.py`
— not of the code being correct.** Renaming `test_api.py` to anything sorting
after `test_e…` breaks 2 modules at collection.

**Why nothing caught it.** CI runs `python -m pytest tests/ -q` (`ci.yml:79`) —
the one invocation that cannot see this. Neither can any prior review that ran
the suite whole. It surfaces the moment Logan runs a subset, which is the normal
way to debug, and it fails at *collection*, so the tests do not report as failed
— they silently do not run.

**Not tested:** whether `pytest-xdist` (`-n auto`, which distributes modules to
separate worker processes) triggers it — xdist is not installed here. My
inference is that it would, non-deterministically, since each worker imports
only its own subset.

Fix is a backlog item (top of the task backlog (not in this public copy)), not applied here — the reviewer
does not edit code.

### ✅ FIXED 2026-08-29 (T3.1) — and this finding's own suggested fix was wrong

Everything diagnosed above reproduced exactly, including the `sys.path[0:3]`
before/after and all five table rows. One correction, measured rather than
argued:

**The "cheapest first" fix this finding recommended — (a), delete the redundant
`sys.path` blocks from `test_evaluate_pinned.py` and `test_fault_injection.py`
— does not fix the bug.** With both blocks stripped,
`pytest tests/ --ignore=tests/test_api.py` still gave *1 collection error*. The
diagnosis singled out those two modules because they were the two that happened
to be traced; `grep -rn sys.path tests/` shows about fifteen modules inserting
`firmware` and/or `ml` at index 0 at module scope, several of them
unconditionally (`test_crest_floor_calibration.py:34-35`,
`test_pink_noise_is_pink.py:32`, `test_phone_recording.py:64-66`). Confirmed by
constructing a pair the finding never mentions:

| command | with fix (a) applied |
|---|---|
| `pytest tests/ --ignore=tests/test_api.py` | 662 collected, **1 error** |
| `pytest tests/test_crest_floor_calibration.py tests/test_frontend_backend_integration.py` | 2 collected, **1 error** |

**The generalisable lesson:** the finding named the two modules that demoted
`backend`, and the fix followed the names rather than the mechanism. Any module
that touches `sys.path[0]` re-creates it. A fix aimed at the *importers* is
bounded (two sites); a fix aimed at the *inserters* is unbounded and silently
regresses the next time someone adds a `sys.path.insert`.

**What was actually done:** option (c). `tests/test_api.py` and
`tests/test_frontend_backend_integration.py` now load `backend/main.py` by
explicit path under unique module names, the pattern
`tests/test_recordings_upload.py` and `tools/e2e_severity_trend.py` already used
for precisely this reason — so the repo had the answer written down twice, in
files this finding did not look at. The redundant `sys.path` blocks were left
alone: they are untidy, not the bug.

Option (b) (rename `backend/main.py` → `backend/app.py`) was rejected: it
touches the Dockerfile, deploy scripts, CI and several docs to remove a
collision the import-site fix already neutralises.

**Regression cover:** `tests/test_import_isolation.py`, 4 tests. Two run pytest
in a *subprocess* — an in-process assertion cannot observe a collection error,
because collection is over before any test body executes, which is the same
structural blindness that hid F12 and F14. One is a static guard against any
bare `import main` in `tests/`. One is forward-looking: it fails if a *new*
duplicate module basename appears across `firmware/`/`ml/`/`backend/`, with
`main.py` allowlisted — so the next collision is caught before it becomes
another silent skip rather than after. Verified non-vacuous by dropping the file
unchanged into a pristine copy of the repo: **3 failed, 1 passed** (the
forward-looking guard passes on old code by design), against 4 passed on the
fix.

Collection is now clean in every ordering tried: full suite 680,
`--ignore=tests/test_api.py` 667, both hostile pairs, and a deliberately
reverse-ordered four-file set. Suite 676 → **680**.

**Still not tested, unchanged from above:** whether `pytest-xdist -n auto`
triggered it. xdist is still not installed here. The import-site fix should make
the question moot — each worker now resolves `backend/main.py` by absolute path
regardless of its own subset — but that is an inference, not a measurement.

---

## F26 — The mains-coincidence flag false-alarmed on the first real recording, exactly as the synthetic finding predicted ✅ NO ACTION NEEDED

Measured 2026-08-29, on the desk-fan experiment (`RESULTS.md`, Experiment 0) —
the first real machine audio this project has ever seen.

`cold_start_screen.py` reported the induced fault at **25.8 Hz** and flagged it
`coincides with mains 50 Hz`, because 2 × 25.8 ≈ 50. **It is not mains.** It is
the fan's shaft rotation rate, confirmed three ways: the raw spectrum gives a
78.38 Hz blade-pass tone, 78.38 / 25.8 = 3.04, and the fan has 3 blades counted
by eye.

**Why this is filed as a finding rather than a bug.** Two days earlier, an
earlier version of this screen *discarded* mains-coincident candidates outright.
That version was caught on `data/bearing_inner.wav`, where `ml/simulate.py`
amplitude-modulates an inner-race fault at the shaft rate — and on a 50 rev/s
machine the shaft rate IS the mains frequency. It binned a comb scoring 39.6 as
hum and reported 7.6 of noise instead. The fix was to flag but never remove.

**That fix is the only reason this result exists.** The very first real
recording landed in exactly the failure mode the fix was built for, within 48
hours, on a completely different machine and fault type. A synthetic finding
predicted a real one — which is the strongest validation the synthetic work
has received.

No code change: the tool behaved as designed. It surfaced the coincidence,
ranked the candidate first anyway, and printed the standing note that an
inner-race fault modulates at shaft rate so a flagged row with a high score is
worth investigating rather than dismissing. The human resolved it in one step
using the blade count.

**The transferable lesson, which is the reason this is written down:** a
detector that *hides* what it is unsure about cannot be corrected by the person
using it. One that shows its uncertainty and its reasoning can be. That design
difference was worth more here than any accuracy improvement in the same period.

**Numbering note.** This was first written as F25 and renumbered on discovery
that a concurrent run had already taken F25 for the `shutil.rmtree` finding
below. Second F-number collision in this project (the first is noted in the
2026-08-22 commit log); both were caused by two sessions writing to this file
without seeing each other's work. The register is append-at-top and newest
first — check the top entry's number before adding one.

---

## F25 — `tools/fridge_scan.py` clears its per-stem working directory with `shutil.rmtree()`, which this project's own mounted working tree refuses to execute at all ✅ FIXED 2026-08-29

Found 2026-08-29, during this run's routine full-suite baseline verification
(not speculative — `tests/test_fridge_scan.py` failed on its own, first while
chunk-running the suite). Six of its twelve tests errored with a traceback
ending in `PermissionError: [Errno 1] Operation not permitted:
'.../data/_scan_work/<stem>'`, all six sharing one call site:
`fridge_scan.main()`'s `if work.exists(): shutil.rmtree(work)`.

**Not actually new — the 2026-08-28 F24 run had already found and named
this exact failure** (the task backlog (not in this public copy) Run log, 2026-08-28 row), correctly
diagnosed it as the mount forbidding deletion rather than a code regression,
and filed "route scratch work through `tempfile` instead" as a candidate for
a future run rather than fixing it (out of scope for that run's own task).
This run picked that candidate up. It considered and rejected the filed
suggestion: switching to a system temp directory would also have solved the
sandbox failure, but would have broken the tool's advertised, user-facing
contract — `main()`'s own closing message names `data/_scan_work/<stem>` as
somewhere the user is meant to go look ("intermediates in ... — delete when
done, or pass --keep-intermediates to keep them"), which only makes sense if
that location is a stable, visible place inside the repo, not a hidden
temp directory a user has no reason to know about. The rename-to-attic fix
below keeps that contract intact.

**First hypothesis, and it was wrong.** The traceback's own file path named a
*different* Claude session (`/sessions/admiring-loving-goodall/...`) than
this run's own (`/sessions/inspiring-wonderful-edison/...`), both mounting
the identical real `Summer Project/acoustic-monitor` folder — strong-looking
evidence for "two concurrent scheduled-task sessions raced on the same
`data/_scan_work/short` directory." That was the story this fix's first draft
told, with a comment and a test docstring both asserting it as observed fact.

**Re-testing it directly disproved that story.** Re-running
`tests/test_fridge_scan.py` alone, with no other session's activity anywhere
in the traceback, reproduced the *identical* `PermissionError` on the very
first invocation of a stem whose directory already existed from an earlier
day — deterministically, not intermittently. Isolating the operation further:
```
$ rm -rf endtoend            # plain shell, same directory, no Python involved
rm: cannot remove 'endtoend': Operation not permitted
$ python3 -c "import shutil; shutil.rmtree('decoytest')"
PermissionError: [Errno 1] Operation not permitted
$ python3 -c "import os; os.rmdir('r24.0')"   # after manually emptying it
PermissionError: [Errno 1] Operation not permitted
```
All three fail identically — a bare shell `rm -rf`, `shutil.rmtree`, and a
manual `os.walk` + `os.remove`/`os.rmdir`. The directories are owned by the
calling user (`ls -la` confirms normal `drwx------` ownership). **This
sandbox's mounted working tree refuses directory deletion outright**, which
is exactly the task backlog (not in this public copy) rule 7's own long-documented "the mounted repo
forbids deleting files — never `rm` inside it" gotcha — just not previously
known to reach `data/_scan_work`, which was designed and shipped
(T1.15-adjacent work, 2026-08-28) without that constraint in view. The
concurrent-session traceback that started this investigation is fully
explained by this alone: two sessions independently hitting the same
undeletable-directory wall on their own is indistinguishable, from one
traceback, from two sessions racing — and the deterministic, solo
reproduction above is the stronger evidence.

**What does work on the same mount, confirmed directly:** `mv`/`os.rename` of
the identical directory succeeds immediately (rule 7 already says "`mv`
within the mount does work" — this just confirms it for a non-empty
directory specifically). That is the fix: `main()` now retires an existing
`work` directory with `os.rename` into `output/_attic/scan_work/`
(timestamp- and pid-qualified) instead of deleting it, matching the
`output/_attic/` convention this repo already uses for exactly this kind of
retired leftover. A per-stem `fcntl.flock`, held for the whole scan (not
just the retire step), was added alongside it: harmless on a normal
filesystem, and closes a real, if separate and unconfirmed-to-have-fired,
theoretical race between two processes scanning the same stem concurrently
on a real machine, where deletion *does* work and two racing rmtree/mkdir
sequences could still interleave badly. **The two fixes address two
different problems** — the rename addresses the confirmed, deterministic
sandbox failure; the lock addresses an unconfirmed, narrower concurrency
risk — and the write-up above was corrected mid-run to stop conflating them,
per this file's own standard for retracting an overstated claim (F16, F20).

Two regression tests in `tests/test_fridge_scan.py`, both FAILS-ON-OLD-CODE
(verified by temporarily reverting to `shutil.rmtree` and re-running each in
isolation, then restoring the fix and re-confirming green):
`test_a_repeat_scan_of_the_same_stem_does_not_choke_on_a_leftover_directory`
runs two real invocations of the same stem back to back and asserts the
second does not crash clearing the first's leftover directory; and
`test_concurrent_scans_of_the_same_stem_do_not_race`, which cannot
deterministically win a real race, instead holds the same per-stem lock from
outside a subprocess and asserts a concurrent invocation blocks rather than
proceeding. `tests/test_fridge_scan.py` — all 12 tests, including these two
— passes green (70.9 s). Suite 674 → **676 collected**. No frozen file
touched (`tools/fridge_scan.py` is not on the frozen list). `README.md`'s two
enforced test-count claims need updating to 676 to keep
`test_docs_current.py` green — **not yet done as of this note**; see
the task backlog (not in this public copy)'s Run log for this run's final state, since the sandbox
wedged (`No space left on device` / `useradd failed`, the documented
2026-08-20/21/22 pattern) partway through re-verifying the rest of the suite
and this file was updated with the shell unavailable.

---

## F24 — `zenodo.org/records/3384388` is linked five times as "DCASE 2020 Task 2," but it is the MIMII dataset — different licence, different files, different folder layout ✅ FIXED 2026-08-28

Found 2026-08-28, outside-attack pass (public-dataset rotation), verified
against the record's own Zenodo/DataCite page via WebSearch — **not** by
fetching zenodo.org directly, which this sandbox's network allowlist also
blocks (`ml/realdata/validate_public_dataset.py`'s T1.1 note and
`tools/fridge_scan.py`'s commit note both independently hit the same block).
That blocked-fetch fact is exactly how this slipped in: whoever wrote
the manual-steps guide (not in this public copy)'s DCASE section could not load the page either, and reused the
MIMII link from `docs/REAL_DATA_SOURCES.md` (2026-08-21) under the wrong
label without cross-checking it.

**The two datasets, per the primary source:**

| | `docs/REAL_DATA_SOURCES.md` says (this record) | the manual-steps guide (not in this public copy) / the project plan (not in this public copy) / `TESTS.md` say (same URL, labelled "DCASE 2020 Task 2") |
|---|---|---|
| Actual dataset | MIMII (DCASE **2019** Workshop) — confirmed by the record's own title on Zenodo/DataCite | A different challenge, a different Zenodo record |
| Licence | **CC BY-SA 4.0** (no NC) | **CC BY-NC-SA 4.0** (has NC) |
| Machine types | 4: valve, pump, fan, slide rail | 6: pump, fan, valve, slider, ToyCar, ToyConveyor |
| Archive names | `6_dB_pump.zip`, `0_dB_fan.zip`, … (SNR-keyed) | `dev_data_pump.zip` (per `tools/dcase_eval.py`'s own docstring, line 67) |
| Internal layout | SNR folder → `id_00`…`id_06` → `normal/`/`abnormal/` | `train/` (normal only) + `test/` (`normal_id_*`/`anomaly_id_*`) |

`tools/dcase_eval.py` is written for the right-hand structure (it requires
`machine_dir/train` and `machine_dir/test`, checked at lines 249–253, and
looks for `normal_id_{mid}_*.wav` / `anomaly_id_{mid}_*.wav`). The **actual
files at the linked record** don't have that shape.

**Consequence, concretely:** the project plan (not in this public copy)'s Step 2 (and the task backlog (not in this public copy) Gate −1,
the current top-priority "do this week, costs nothing" item) tells Logan to "Download
`pump.zip` and `fan.zip` from zenodo.org/records/3384388" — no such filenames
exist at that record; the closest are `6_dB_pump.zip` etc. Even renamed and
unzipped, the result has no `train/`/`test/` split, so
`python tools/dcase_eval.py data/dcase2020/pump` fails immediately with the
tool's own `machine id gave train=0` guard (`dcase_eval.py:261-263`) — not a
subtle numerical error, a hard stop on the very first command of the
project's stated top-priority task.

**Also genuinely useful, and already in the repo:** `REAL_DATA_SOURCES.md`'s
description of *this* record (MIMII) reads as accurate for what's actually at
the link — a previous review (the daily review (not in this public copy), 2026-08-28 morning run)
independently checked its licence and model-count claims against the
record's Dublin Core metadata and found them correct. The problem is not that
`REAL_DATA_SOURCES.md` is wrong; it's that four *other* files point the same
URL at a dataset it isn't.

**Untested / inference, labelled as such:** [Guess] the real DCASE 2020 Task
2 development-set record is a different Zenodo ID in the same low-4-million
range (search results surfaced `zenodo.org/records/3678171` as "DCASE 2020
Challenge Task 2 Development Dataset" — not independently fetched or
confirmed here, since zenodo.org is blocked from this sandbox too). Whoever
fixes this should locate and confirm the correct DOI from a machine that can
actually reach Zenodo, not guess from a search snippet the way this defect
was likely introduced.

Fix task added at the top of the task backlog (not in this public copy).

**Fixed 2026-08-28 (later run, F24 task).** Confirmed `zenodo.org/records/3678171`
via `WebSearch` — [Likely], not [Certain], since this sandbox's network
allowlist still blocks a direct `zenodo.org` fetch (same block this finding
and the commit log (not in this public copy)'s two earlier CWRU/DCASE attempts already hit), so this
is WebSearch's rendered summary of the Zenodo page, not the page itself. That
summary states the record's title as "DCASE 2020 Challenge Task 2 Development
Dataset," names `dev_data_<machine_type>.zip` as the per-machine archive, and
describes ~10 s single-channel clips built from ToyADMOS + MIMII — all three
facts match `tools/dcase_eval.py`'s own docstring (line 37: this exact URL;
lines 27-43: `dev_data_pump.zip`, single channel, 16 kHz, CC BY-NC-SA 4.0)
independently and exactly, which is strong corroborating evidence even
without a direct fetch. the manual-steps guide (not in this public copy), the project plan (not in this public copy) and `TESTS.md` now link
`3678171` instead of `3384388` and name `dev_data_pump.zip`/`dev_data_fan.zip`
(the record's real filenames) instead of `pump.zip`/`fan.zip`. Each edit keeps
the old URL/claim in a dated correction note rather than silently overwriting
it, per this file's own convention (F23/T2.8). `docs/REAL_DATA_SOURCES.md`
was already correct (MIMII, `3384388`) and was not touched.
the daily review (not in this public copy)'s "Your move" item 1 also updated with a done-note;
its narrative body (which only *describes* the bug, not instructs downloading
from it) was left as a historical record. No code, test or frozen file
touched — doc-only, as the original finding was.

---

## F23 — withheld from the public copy

This finding concerns a bill-of-materials figure quoted inconsistently
between two internal documents. It is omitted here because it contains
cost data from the private repository, not because it was unflattering
— it is recorded as FIXED. The omission is stated rather than silent so
that the numbering gap does not read as a removed failure.

## F22 — the task backlog (not in this public copy) rule 7's own install line cannot install today: `--only-binary :all:` has nothing to install for `paho-mqtt<2.0` ✅ FIXED (T2.7, 2026-08-27)

**Correction 2026-08-29 (no-shell audit, sandbox wedged):** this header still
said "⚠ OPEN" a day after the task backlog (not in this public copy)'s own Run log records T2.7 as
"Done" (2026-08-27) — `paho-mqtt` pulled out of the `--only-binary :all:`
clause into its own unconstrained `pip install`, per rule 7's text, which was
verified directly this run by reading rule 7 as it stands today: it now
reads "in rule 7's install line, pull `paho-mqtt` out of the `--only-binary
:all:` invocation (separate `pip install paho-mqtt>=1.6,<2.0` with no binary
constraint...)" and the T2.7 closed-task entry confirms the fix was applied
and reproduced (broken form fails, split form installs `paho-mqtt-1.6.1` from
sdist in ~5s). Every closed-task entry after this one (T2.6, T2.8, F24, F25)
explicitly records marking its corresponding SELF-REVIEW header FIXED; T2.7's
entry does not mention touching this file, which is the paperwork gap this
correction closes — same "orphaned paperwork" shape already documented for
T0.1/T5.2/T7.1-3/T1.13 elsewhere in this project, just one file later than
those. Not independently re-run this run (shell is wedged, see the task backlog (not in this public copy)
Run log 2026-08-29 second entry) — this is a doc-consistency fix, not a new
measurement.

Found 2026-08-26 during the daily review's independent test-suite
verification — not by inspection, the install failed on its own while
bootstrapping a fresh sandbox to run the suite, same discovery shape as F21.

**Measured, this run:**

    pip install --break-system-packages --only-binary :all: "paho-mqtt>=1.6,<2.0"
    ERROR: Could not find a version that satisfies the requirement paho-mqtt<2.0,>=1.6 (from versions: 2.0.0rc2, 2.0.0, 2.1.0)
    ERROR: No matching distribution found for paho-mqtt<2.0,>=1.6

Confirmed with `pip index versions paho-mqtt`: 1.6.1 exists on PyPI and
installs fine as a plain `pip install` (it is pure Python, no compile step),
but PyPI has **no wheel** for it — only a source distribution. `--only-binary
:all:` categorically refuses sdists, regardless of how trivial the build is,
so this specific combination — a pin below 2.0 (`backend/requirements.txt`
needs `<2.0` for the pre-callback-API-v2 `paho.mqtt.client` surface
`backend/mqtt_client.py` uses) plus `--only-binary :all:` (rule 7's own
sandbox-speed workaround) — cannot be satisfied today. Removing
`--only-binary :all:` for just this one package installs 1.6.1 in under a
second and the rest of the suite is unaffected.

**Not the same bug as F21/T2.6** (that was a missing package; this is a
present package with no wheel) but the **same root shape**: a documented
bootstrap command that worked when written silently stops working as the
package index changes underneath it, and the first sign is an opaque pip
error, not a code failure. `.github/workflows/ci.yml` is unaffected — its
`Install Python dependencies` step is a plain `pip install -r
backend/requirements.txt`, no `--only-binary`, so CI has never hit this.
Only the task backlog (not in this public copy) rule 7's sandbox-bootstrap recipe is broken, which
means every reviewer/agent run that follows it from a clean container hits
this same wall — as this run did — until it's fixed.

**Untested further:** whether the same `--only-binary :all:` combination
fails for any *other* package in that line today (checked `paho-mqtt`
specifically because it was the one that broke; did not re-verify
numpy/scipy/matplotlib/scikit-learn/pytest/fastapi/sqlalchemy/pyyaml/httpx/
httpx2/email-validator all still have wheels for this platform, only that
they installed successfully this run when tried individually).

**Fix:** in the task backlog (not in this public copy) rule 7's install line, drop `--only-binary
:all:` for `paho-mqtt` specifically (either a separate `pip install
paho-mqtt>=1.6,<2.0` line with no binary constraint, or move it out of the
`--only-binary :all:` invocation entirely) — do not remove the flag for the
whole line, the other packages need it for sandbox install speed. Small,
no design decision needed.

---

## F21 — `backend/requirements.txt` has no upper bound on `fastapi`, and a clean install today pulls a `starlette` that needs an undeclared package ✅ FIXED (T2.6, 2026-08-26)

Found 2026-08-25 during the daily review's independent test-suite verification,
not by inspection — the collection error happened on its own while installing
dependencies into a fresh sandbox.

**Measured, this run:** `pip install fastapi` (no version pin beyond
`fastapi>=0.110`, per `backend/requirements.txt`) resolved to **fastapi
0.141.1 / starlette 1.6.0** today. Importing `fastapi.testclient.TestClient` —
which `tests/test_api.py` and `tests/test_frontend_backend_integration.py`
both do at module level — then raises:

    RuntimeError: The starlette.testclient module requires the httpx2
    package to be installed.

`httpx2` is a real, separately-versioned PyPI package (confirmed via `pip
index versions httpx2`, currently at 2.12.0) and is **not listed anywhere in
`backend/requirements.txt`**. Installing it resolves the error; both test
files then collect and pass normally (confirmed this run).

**This is not a one-off first-install gap — it is a regression against the
project's own documented setup instructions.** the task backlog (not in this public copy) (rule 7,
2026-08-17) already tells the scheduled agent to
`pip install ... fastapi ... httpx ...` before the baseline test run. That
instruction names `httpx`, not `httpx2` — it was correct when written and is
silently wrong today, because upstream (`starlette`) changed its `TestClient`
dependency between then and 2026-08-25 with no corresponding change on our
side. **Same root cause as the python-multipart lesson already written into
this same requirements file**: a package the code needs is present on
whichever machine happened to install it by hand, absent from a clean
install, and nothing pins the range that would keep it working.

**Not testable further without re-running the exact install on a future
date** — I cannot prove whether this was a one-time jump or something that
will keep drifting; that needs the same install repeated later.

**Fixed 2026-08-26 (T2.6).** Read `starlette/testclient.py` source directly
before fixing anything: it tries `import httpx2 as httpx` first, falls back
to plain `httpx` (with the deprecation warning this file's own repro
predates seeing) only if `httpx2` is missing, and raises `RuntimeError` only
if **both** are missing — confirms this finding measured the right
condition (a bare install has neither). That also settles the open "pin
fastapi/starlette, or add httpx2" question in the fix suggestion below:
pinning would only delay the same failure at fastapi/starlette's next
resolve, since the file has no upper bound today and the real gap is the
missing test-client dependency, not the framework version. Took the
`httpx2` option: added `httpx2>=2.0` to `backend/requirements.txt`.
Verified non-vacuous — with `httpx` uninstalled and only `httpx2` present
(the exact state a clean `pip install -r backend/requirements.txt`
produces), `TestClient` imports with zero warnings and `tests/test_api.py`
+ `tests/test_frontend_backend_integration.py` pass 14/14. the task backlog (not in this public copy)
rule 7 and `.github/workflows/ci.yml`'s install line updated to add
`httpx2` too, so the three places this command is spelled out stay in sync.
Full writeup in the task backlog (not in this public copy)'s T2.6 entry and the 2026-08-26 Run log row.

---

## F20 — T1.13's fix works, and costs a 0.00 → 0.107 false-alarm rate that nobody measured ✅ CLOSED 2026-08-27

Found 2026-08-23 while independently verifying T1.13 (the per-machine
`crest_floor` that F19 asked for). The implementation is correct and its tests
pass. **The regression is in a number no test asserts.**

**T1.13 does what it was built to do.** Independently measured across 6
machines: with the old constant floor the selector found the 1600 Hz resonance
in **0 of 6** severity-0.20 faults; with each machine's calibrated floor,
**6 of 6**. Calibrated floors ranged 6.50–7.77, so they are genuinely
per-machine rather than a constant in disguise.

**And `ml/evaluate.py`'s STAGE 3 GATE now reports a false-alarm rate of
0.107 where this session measured 0.000 earlier the same day.**

Isolated to a single variable — same baseline file, same data, only the stored
floor changed:

| `crest_floor` | deployed-threshold FPR | TPR | AUC |
|---|---|---|---|
| 7.073 (calibrated) | **0.1071** | 1.000 | 1.000 |
| 10.0 (the old constant) | **0.0000** | 1.000 | 1.000 |

Three of 28 healthy windows now cross threshold. The mechanism is the one F19
warned about and `baseline.py`'s own comment names: a lower floor lets healthy
windows select a band, different windows select *different* bands, and the
feature vector moves for reasons that have nothing to do with the machine.

**Why the gate still says PASS, and why that is not reassurance.** The gate
keys on AUC, regime false alarms and the gating behaviour, all unchanged
(`gating_alerts_transient: 0`, `persistent: 1`). The 30-minute persistence gate
absorbed the extra windows in this run. So the end-to-end alert count did not
move — but the per-window margin did, and the project's risk assessment (not in this public copy) calls alarm
fatigue churn risk #1 against a target of ≤1 false alarm per node-week.

**The honest framing.** This is a *trade*, not a bug: better band selection on
a realistic pink floor, paid for in per-window false alarms on the white-noise
simulator. Nobody priced it, because the gate's PASS/FAIL does not include FPR
and no test asserts it. That is the actual defect — **a proven number regressed
by 10 percentage points and every check stayed green.**

Two things to do, filed as **T1.14**:

1. **Assert it.** `deployed_threshold_fpr` should be a pinned number, not a
   line of JSON nobody diffs. This class of silent regression is exactly what
   F11's doc-count guard was built for, applied to a measurement.
2. **Fix the mechanism, not the constant.** The problem is band *instability*
   between windows, not the floor's value. Options worth measuring: hold the
   band chosen during the learn period rather than re-selecting per window; or
   require a band to win by a margin before switching. Raising the floor back
   to 10.0 would restore the FPR and re-break F19.

**2026-08-26 update — Part 1 CLOSED, Part 2 investigated and still OPEN.**

Part 1: `ml/evaluate.py` refactored (not frozen) to expose a pure
`compute_metrics()`; `tests/test_evaluate_pinned.py` (7 tests) pins
`deployed_threshold_fpr` at **0.1071** (= 3/28, exact), TPR/AUC at 1.0, regime
false alarms at 0, gating counts at (transient=0, persistent=1), and that the
deployed `crest_floor` (7.073) is the T1.13-calibrated value, not the old
constant. Verified non-vacuous by forcing `scorer.crest_floor = 10.0` in a
throwaway script and confirming the pinned test's own quantity changes to
0.000 — i.e. this pin would have caught F20 the day it happened. CLI output
of `ml/evaluate.py` confirmed bit-identical before/after the refactor.

Part 2, the actual mechanism fix, was investigated with real measurements,
not shipped, because neither option meets the bar this file itself set
("keep F19's 6/6 recovery AND return FPR to ~0"):

- **Option A (hold the learn-period band) is mechanically disproved, not just
  untried.** `tests/test_crest_floor_calibration.py::
  test_a_calibrated_floor_does_not_make_healthy_machines_pick_a_band` already
  guarantees, by construction, that a calibrated floor never lets a genuinely
  healthy learn window pick anything but `DEFAULT_BAND`. Holding "the band the
  learn period chose" is therefore holding `DEFAULT_BAND` forever — bit-for-
  bit equivalent to reverting T1.13. Measured directly across the same 6
  machines F19/F20 used: **0/6** fault recovery. Not a corner case; this is
  what the option literally reduces to.
- **Option B (single-window amplitude-margin hysteresis)** was prototyped
  (throwaway script, not committed): re-select every window as before, but
  only switch the incumbent band if the challenger's crest beats the
  incumbent's own recomputed-this-window crest by a fixed multiplicative
  margin, with an unconditional release back to `DEFAULT_BAND` whenever
  nothing in the window clears `crest_floor` at all (a first version without
  that release got FPR **worse**, 0.714, by getting stuck on a stale band —
  recorded so nobody rediscovers it the slow way). Swept the margin on the
  real 40-window `ml/evaluate.py` scenario (state carried across windows in
  schedule order) and the 6-machine F19 scenario:

  | margin | F19 recovery | deployed FPR |
  |---|---|---|
  | 1.5 | 6/6 | 0.107 (no improvement) |
  | 2.0 | 4/6 | 0.036 |
  | 3.0 | 0/6 | 0.000 (= Option A) |

  **No single global margin sits at "6/6 AND ~0".** The frontier is real and
  roughly monotone: TPR-side recovery is bought with FPR, one for one. A
  single scalar margin is the wrong degree of freedom — the same lesson F19
  taught about `crest_floor` itself (global constants don't survive contact
  with 14 different machines) likely applies here too.

**Left for the next run, not attempted this run:** either calibrate the
margin per machine from the learn period (mirroring `calibrate_crest_floor`
exactly), or replace the single-window margin with a **persistence**
requirement — N consecutive windows where the challenger wins, mirroring
`AlertGate`'s `need` parameter — before switching. The latter is untried and
worth trying first: a single loud transient window is exactly the case a
margin cannot distinguish from a genuine emerging fault, but persistence can.

No frozen file was touched investigating this (the prototype hysteresis code
lives only in a throwaway `/tmp` script, not in this repository); `firmware/
features.py` and `firmware/baseline.py` are untouched and F19/F20 remain OPEN.

**2026-08-27 — Part 2 CLOSED. Read the bar change before the result.**

The uncomfortable part first: **this does not meet the bar the section above
set for itself.** That bar was "keep F19's 6/6 recovery AND return FPR to ~0".
What shipped keeps **4/6** and returns FPR to **0.000**. The bar was lowered,
deliberately, and the reason is below. Anyone reading this later should judge
the decision, not assume it was met.

**What changed.** One constant in `firmware/baseline.py`:
`CREST_FLOOR_MARGIN` 0.3 → 0.7, plus the calibration statistic moved from p99
to `max` over the learn-period healthy crests. Deployed floor 7.073 → 7.489.

**A hypothesis that was wrong, recorded so it isn't retried.** The first guess
was that the *statistic* was too permissive — that p99 over learn windows was
clipping the tail and `max` would fix it. Measured: `max` alone moved the floor
7.073 → 7.089 and left FPR at **0.107, unchanged**. The statistic does almost
nothing here; the learn-period crest distribution is tight enough that p99 and
max nearly coincide. It is the **margin above** the machine's own maximum that
does all the work. The `max` change was kept anyway because it is the more
defensible statistic to sit a margin on top of, not because it fixed anything.

**RETRACTION, same day, before this section was committed.** The first draft
of this entry priced the decision as "4/6 recovery instead of 5/6" and called
0.7 "the knee". **Both claims were wrong, and the way they were wrong is the
point.** They came from the same kind of throwaway `/tmp` script this file
criticised two paragraphs above — the original 5/6-vs-4/6 measurement no longer
existed and could not be re-run. Rebuilding it as a committed tool
(`tools/sweep_crest_margin.py`) to verify the numbers before publishing them
showed they do not reproduce. **F20's own lesson, applied to F20's own fix.**

**The sweep, as committed and reproducible** —
`python tools/sweep_crest_margin.py --margins 0.3 0.7 1.0 --severity 0.35 0.20 0.10`:

| `CREST_FLOOR_MARGIN` | deployed floor | recovery @ sev 0.35 | @ sev 0.20 | @ sev 0.10 | deployed FPR |
|---|---|---|---|---|---|
| 0.3 + `max` (was shipped) | 7.089 | 6/6 | 4/6 | 0/6 | **0.1071** |
| **0.7 + `max` (shipped)** | **7.489** | **6/6** | **4/6** | **0/6** | **0.0000** |
| 1.0 + `max` | 7.789 | 6/6 | 4/6 | 0/6 | 0.0000 |

**Recovery is identical at every margin. The change costs nothing.** The
false-alarm rate goes 0.107 → 0.000 and not one machine is lost, at any of the
three severities. There was no trade to price. The claimed frontier was an
artefact of an unreproducible measurement.

**Why there is no trade, mechanically.** Mean faulty band crest is 23.1 at
severity 0.35, 8.3 at 0.20, 6.0 at 0.10. The floors under test are 7.089,
7.489 and 7.789 — a 0.7-wide span. A severity-0.20 fault at crest 8.3 clears
all three; a severity-0.10 fault at 6.0 clears none. The margin only decides
anything for a fault whose crest lands *inside* that narrow span, and none of
these do. Raising the floor by 0.4 removes healthy windows' ability to pick a
spurious band (they sit at 5.4–7.7, i.e. *inside* the span) without touching
faults (which sit outside it). That asymmetry is why this works and why it is
not the zero-sum frontier Option B ran into.

**Two things this sweep also exposed, both recorded because they would have
made the table meaningless and neither was obvious:**

- **Sample rate is load-bearing.** Run at `synth_phone_recording`'s native
  44.1 kHz instead of the repo's 16 kHz, every crest collapses (healthy
  3.5–4.1, faulty 4.4), `calibrate_crest_floor` clamps to `MIN_CREST_FLOOR`
  6.5 for *every* margin, and recovery reads 0/6 across the board. The sweep
  would have looked flat because the margin had been made inoperative.
- **Severity is load-bearing.** At `make_pair`'s default severity 0.35 the
  answer is 6/6 for every margin out to 3.0 — a clean, meaningless result. A
  sweep must be run where the quantity is marginal or it measures nothing.

**Why 0.7 rather than 1.0, honestly.** The sweep gives no reason to prefer
either; they are identical on both axes. 0.7 was chosen because it is the
smaller departure from each machine's own measured healthy maximum, so it is
less likely to hit the `min(DEFAULT_CREST_FLOOR, ...)` cap on a machine with a
louder noise floor. **That is a judgement, not a measurement**, and it is
written down as one.

**What the decision rests on now.** Not a trade-off — a strict improvement:
FPR 0.107 → 0.000, recovery unchanged, one constant changed, no new state, no
new failure mode. the project's risk assessment (not in this public copy)'s alarm-fatigue argument still motivates
*caring* about the FPR, but it is no longer buying anything with it.

**What this is NOT.** It is not the mechanism fix Part 2 asked for. Band
selection is still per-window and still unstable; this raises the floor until
instability stops mattering *on these signals*. The per-machine margin
calibration and the persistence-based band switch named above remain untried
and remain the better answer. **Filed as T1.15**, because the failure mode
here — a global constant tuned against a simulator — is precisely the one F19
proved does not survive contact with 14 different machines. Expect this to
need redoing on real fridge audio.

**Guarded.** `tests/test_evaluate_pinned.py` pins the new numbers by exact
equality (FPR 0.000, crest_floor 7.488572112684821) and its docstrings carry
the 0.000 → 0.107 → 0.000 history. `tests/test_stage3_gate_numbers.py` bounds
the same quantities loosely (FPR ≤ 0.02, TPR ≥ 0.95) through a subprocess
against `ml/evaluate.py`'s printed JSON. Both were **proven non-vacuous**: the
pins failed on this very change and had to be updated with the measurement,
which is exactly the conversation F20 said never happened. Full suite 590/590.

---

## F19 — `crest_floor = 10.0` rejects 13 of 14 real faults. And the obvious fix does not work. ⚠ OPEN

Measured 2026-08-23, chasing F18's uncomfortable half. F18 asked whether a
realistic pink noise floor masks faults that white noise does not. It does —
but not for the reason F18 guessed, and not in a way a constant can fix.

**The mechanism.** `select_demodulation_band` (the protrugram) scores each
candidate band by envelope-spectrum peakiness and takes the best. If the best
crest is below `crest_floor = 10.0` it gives up and returns `DEFAULT_BAND`
(3–6 kHz). On `synth_phone_recording`, whose resonance sits at 1600 Hz —
deliberately *outside* the default band — that fallback is fatal: the detector
then demodulates a band the fault is not in.

The fault was there the whole time. Measured on the same signals, comparing
the band the selector chose against the band the resonance is actually in:

| severity | selector picked | peak (selector's band) | peak (TRUE band) |
|---|---|---|---|
| 0.05 | (3000, 6000) | 1.3 | 2.1 |
| 0.10 | (3000, 6000) | 1.3 | 1.9 |
| **0.20** | **(3000, 6000)** | **1.3** | **4.8** |
| 0.35 | (1402, 1966) | 17.9 | 17.2 |

At severity 0.20 there is a **4.8× BPFO peak sitting in the 1400–2000 Hz band**
and the detector never looks at it.

**Two prior explanations, both mine, both wrong and both recorded.** First I
blamed `shared_knock_ring` — the fixed-amplitude distractor at 0.15, which is
louder than the fault below severity 0.2. Rebuilt the signal with the knocks
removed entirely: **no change**, still flat at 1.3. Then I blamed band RMS as
the wrong axis, which was true but incidental. The step was the band
*selector* switching, not the fault appearing.

### The obvious fix, and why it fails

If a floor of 10.0 is too high, lower it. Crest over **14 independent
machines**, 30 s each:

| | min | median | max |
|---|---|---|---|
| healthy | 5.56 | 6.25 | **7.33** |
| fault, severity 0.20 | **6.56** | 8.99 | 10.21 |

**They overlap.** The quietest fault (6.56) is below the loudest healthy
machine (7.33), so **no global constant separates them**:

| floor | healthy that wrongly pick a band | faults still missed |
|---|---|---|
| 10.0 (current) | 0 / 14 | **13 / 14** |
| 8.0 | 0 / 14 | 2 / 14 |
| 7.0 | 1 / 14 | 1 / 14 |
| 6.0 | 12 / 14 | 0 / 14 |

8.0 is a large improvement over 10.0 on this data — 13 misses down to 2, with
no false band-picks — and it is tempting. It is still a constant fitted to
fourteen synthetic machines, and the overlap means it cannot be right for all
of them.

### What the fix probably is, and why it is not being done here

The floor should not be a global constant at all. **It should be calibrated
per machine from the learn period**, exactly as the anomaly thresholds already
are (`baseline.cv_threshold`). Every machine already spends 24 minutes
establishing what its own normal looks like; the crest distribution of its own
healthy windows is free during that time, and a machine whose healthy crest
sits at 5.5 could then use a floor its neighbour at 7.3 could not.

**Deliberately not implemented in this session.** `firmware/features.py` is a
frozen file and every measured number in this repository flows through
`select_demodulation_band`. Changing it needs a failing test written first, a
full re-verification of DOC_STATUS's proven rows, and a re-run of the
sensitivity sweep — not a constant edited at the end of a long session.

Filed as **T1.13**. Until then, treat the pink-noise sensitivity numbers as
what they are: the detector misses faults it can see, for a fixable reason.

---

## F18 — The reference-library product is impossible, measured. And a sensitivity result nobody asked for. ⚠ ONE HALF OPEN

Measured 2026-08-23, after the "library of healthy recordings per machine
type" idea came up for the fourth time. Argument had not settled it, so it was
tested: learn a baseline on **Unit A**, then score three things against it.

Two units of the *same model*, differing only in the ways real units do —
casting resonance (1600 vs 1740 Hz), slip/rpm (1450 vs 1466), and how they sit
(15 % level difference):

| Scored against Unit A's baseline | median × threshold |
|---|---|
| Unit A healthy (control) | **0.79** — correctly not flagged |
| **A different HEALTHY unit** | **4.27 — FLAGGED AS FAULTY** |
| Unit A's own fault, severity 0.05 | 0.55 |
| Unit A's own fault, severity 0.10 | 0.55 |
| Unit A's own fault, severity 0.20 | 0.70 |
| Unit A's own fault, severity 0.35 | 8.69 |

**The library is dead, and the way it dies is the interesting part.** A
reference library must pick one threshold. Set it below 4.27 and every healthy
unit that is not the reference one is reported broken. Set it above 4.27 and
you miss every fault up to severity 0.20, because they score **0.55–0.70** —
*quieter* than the difference between two healthy machines. There is no
threshold that both avoids false alarms and catches an early fault. Unit-to-unit
variation is not noise around the signal; **it is larger than the signal.**

This is the same claim `README.md` has made in prose since 2026-08-21. It is
now a measurement.

### ⚠ The half of this that is a problem for US

Those same rows say something uncomfortable about the *self*-baselined
detector, which is what we actually ship: scored against **its own** correct
baseline, this generator's faults at severity 0.05, 0.10 and 0.20 came out at
**0.55, 0.55 and 0.70×** — below threshold. Not detected.

**Do not read that as a regression yet, and do not read it as safe.** The
severity parameter of `synth_phone_recording.make_pair` is not the same scale
as `ml/simulate.py`'s, and `DOC_SENSITIVITY.md`'s detection-down-to-0.02 result
was measured on the latter. Two possibilities, and they need separating:

1. the two `severity` scales differ by roughly an order of magnitude, and
   nothing is wrong; or
2. the realistic **pink** noise floor in `synth_phone_recording` — which is
   the whole reason that module exists — masks faults that white noise does
   not, in which case `DOC_SENSITIVITY.md`'s numbers are optimistic and the
   product's early-warning claim is weaker than advertised.

Possibility 2 would matter a great deal: the project's risk assessment (not in this public copy) already caps the
honest claim at "days to weeks, not months", and this would tighten it
further.

**Filed as a task, not a conclusion.** Calibrate the two severity scales
against a common physical measure — band RMS re the healthy floor, in dB —
and re-run the sensitivity sweep on the pink-noise generator. Until then, no
number in `DOC_SENSITIVITY.md` should be quoted as applying to realistic noise.

---

## F17 — Voice Memos' default compression deletes everything above 10 kHz ✅ DETECTED (mitigation shipped)

Found 2026-08-23 while making the first `.m4a` to test the phone upload route.
An iPhone Voice Memo is **lossy AAC**, and nobody had asked what that does to a
signal the detector depends on.

Measured, ffmpeg AAC, mono, against the uncompressed original of the same
recording (band power via Welch, faulty portion only):

| Band | 128 kbps | **32 kbps** |
|---|---|---|
| 1.4–2.0 kHz (the resonance) | −0.2 dB | +0.6 dB |
| 3–6 kHz (default demod band) | −0.3 dB | +1.1 dB |
| 6–10 kHz | −0.3 dB | +0.7 dB |
| **10–16 kHz** | −0.3 dB | **−78.6 dB** |

Swept in 1 kHz steps, 32 kbps is flat to ±1 dB up to 10 kHz and then falls off
a **cliff**: −78.4 dB at 10–11 kHz and everything above. Not attenuation.
Deletion.

**Why that is dangerous rather than merely lossy.** the system overview (not in this public copy) puts machine
resonances anywhere in **1–20 kHz**. A machine whose resonance sits above the
cliff produces a recording with *nothing to demodulate* — the envelope analysis
finds no impacts, and a failing machine reads as a quiet healthy one. Nobody
can hear the difference, and the file plays back perfectly.

It is the **bitrate, not AAC**: at 128 kbps the same encoder costs 0.3 dB
everywhere.

### The signature, not just the energy — measured 2026-08-23

Losing spectral energy is not automatically losing a *fault*. So the claim was
re-measured with the project's own stage-1 metric: envelope-spectrum
peak-to-background at the computed BPFO, the number `ml/verify_signals.py`
reports as 2.2× raw vs 56.7× enveloped.

| Resonance | WAV | 128 kbps | **32 kbps** |
|---|---|---|---|
| 1.6 kHz (below the cliff) | 2.9× | 2.7× | **2.3× — intact** |
| **12 kHz (above the cliff)** | **59.6×** | 57.5× | **2.9× — gone** |

A resonance above the cliff loses its fault signature **20-fold**, from
unmistakable to indistinguishable from background. Below the cliff, nothing
happens. The danger is specific and the boundary is sharp.

**And the failure is silent.** On the 32 kbps file the protrugram fell back
from the correct band **(7598, 12616) Hz to the 3–6 kHz default** — there was
nothing left above 10 kHz for it to find. No warning, no error: a machine with
a developing high-frequency fault simply reads as healthy.

Pinned by `test_a_codec_cliff_destroys_a_fault_signature_above_it`, which
reproduces the collapse against the real `select_demodulation_band` and
`envelope_spectrum` using an FFT brick wall (no ffmpeg needed). That test
deliberately **does not** assert the band fallback: an FFT brick wall leaves
7.6–10 kHz of that band intact, so the selector still picks it there. The
fallback was real with ffmpeg but is a property of that codec's output, not a
universal consequence, and a test should not claim more than its fixture shows.

**Mitigation shipped:** `check_phone_audio.lossy_cutoff_hz()` finds the cliff
and `check_phone_audio.py` refuses the recording with the fix (iPhone Settings
→ Voice Memos → Audio Quality → **Lossless**). Pinned by three tests in
`tests/test_check_phone_audio.py`: a brick wall is detected, full-bandwidth
audio is *not* flagged (a checker that flags everything is worthless), and a
single notched band is not mistaken for a codec cliff.

### ⚠ A false finding I nearly filed, recorded because the method matters

Before measuring the spectrum I ran the *detector* on WAV vs 128k vs 32k and
got median score/threshold 3.25 / 2.85 / **2.17**, and was about to report
"32 kbps costs a third of the detection margin".

**That number was meaningless.** I had used `--learn-windows 6`, and
`DOC_STATUS.md` records that below 48 windows the held-out false-alarm rate is
55–59 % because the 37-dimensional covariance is under-determined. The tell was
sitting in my own output and I nearly missed it: a severity sweep gave WAV
ratio **2.91 at every severity from 0.02 to 0.10**. A detector whose score does
not move with fault size is not measuring the fault.

Worse, I had put that exact floor into the upload route the same day —
`learn_period_too_short` — and then violated it in an experiment.

The valid measurement is the spectral one above, which needs no learn period
and no detector at all. **When an experiment's own control shows no variation,
stop and doubt the experiment before writing down the result.**

---

## F1 — The 40-dim feature vector contains two exact linear dependencies ✅ FIXED (and partly wrong)

**Suspicion:** the 8 band-energy ratios are *fractions of total energy*, so
they sum to 1. If so they are compositional data living on a 7-dimensional
simplex, and the covariance of those 8 columns is singular **by construction**.

**Test result:** confirmed. The eight fractions sum to 0.999999.

We have two such blocks — audio and accelerometer — so the 40-dim vector
contains **two exact linear dependencies**. The measured effective rank of the
real learn matrix is **21.5 of 40 dimensions**, with the smallest singular
values at 0.0017.

**Why it has not blown up:** Ledoit–Wolf shrinkage regularises the inverse, so
nothing crashes and the scores look sensible. The problem is hidden, not
absent — we are spending 16 dimensions of a 40-dim budget on 14 dimensions of
information, at n/d = 1.2 where every dimension is expensive.

**The correct fix** is the standard treatment for compositional data: a
centred-log-ratio (CLR) transform, or simply drop one band per block and keep
7. Compositional data analysis is a real field with a right answer here, and we
should use it rather than leaving Ledoit–Wolf to paper over it.

→ backlog **T1.5**

### Resolution, 2026-08-17 — fixed, and three things above are wrong

Fixed by replacing each block of fractions with its **isometric log-ratio
(ILR)** coordinates: 40 dims → **37 dims**, no constraint, invertible, and
orthonormal in the Aitchison geometry so the Mahalanobis distance built on top
of it still means something. `tests/test_compositional.py`, 17 tests.

Doing the work corrected the diagnosis above in three places. Recorded because
a self-review that is never itself reviewed is just a longer opinion.

**1. The prescribed fix would not have worked.** CLR coordinates sum to zero by
construction, so a CLR-transformed block is *still* exactly rank D−1 in D
columns — the singularity is preserved, merely relocated. "CLR or drop one
band" was half right; only the second half fixes anything, and ILR is better
than both. Pinned by `test_clr_would_not_have_fixed_it`.

**2. "Two exact linear dependencies" was the wrong count and the wrong
location.** The audio and accel band features are `log10` of the fractions, so
their constraint is *non-linear* and not exact. There was exactly **one**
algebraically exact dependency, in the block F1 never mentioned: the six raw
`env_frac` values, whose measured null direction is the uniform vector to
|cos| = **1.0000**. The two log blocks were near-degenerate for a different
reason — over a stationary machine the fractions barely move, so the constraint
*linearises* almost perfectly (measured |cos| between the null direction and the
mean-fraction weight vector: **1.0000**). Same symptom, different mechanism, and
the mechanism is what determines the fix.

**3. The headline number, "effective rank 21.5 of 40", does not measure what F1
used it for.** After the fix, effective rank went **down**: 17.4/40 (43 %) →
9.0/37 (24 %), and the condition number of the standardised learn matrix rose
3324 → 8359. That is not a regression. Participation-ratio effective rank
measures how evenly variance is *spread*, and eight separately-noisy
log-fractions spread variance across eight directions while carrying about one
direction of information. Removing the arbitrary basis concentrated the same
information into fewer, more correlated coordinates and stopped counting the
noise. **Effective rank was never an information measure**; F1's central
statistic was measuring the wrong thing, which is why the fix it motivated
produced no detection benefit.

**Measured effect on detection: none.** Controlled comparison, the *same* 192
simulated windows represented both ways, 300 bootstrap learn/holdout splits:

| | before (40-dim) | after (37-dim) |
|---|---|---|
| held-out healthy FPR | 0.0492 ± 0.0301 | 0.0345 ± 0.0224 |
| paired difference | | **−0.0147, 95 % CI [−0.083, +0.048]** |
| ROC AUC (severity 0.15) | 1.0000 | 1.0000 |

The FPR direction is favourable and the interval contains zero. On the CWRU
*surrogate*, AUC 0.9889 → **0.9917** and TPR 0.967 → **0.989**, while
FPR 0.125 → **0.1875** — that last is 2 of 16 held-out windows becoming 3 of 16,
i.e. one window, on an estimator that F3 showed is set by whichever learn window
happened to be worst. Do not read it as a regression, and do not read the AUC
gain as a win either.

**So why keep the change?** Three defensible reasons, none of them "the metric
improved":

* An exactly singular block is gone (env block sv ratio 6.5e-3 → **0.34**), and
  a covariance that is singular by construction is a defect whether or not
  shrinkage currently hides it. It will strain harder on real data, where the
  learn period may be shorter and dirtier than the simulator's.
* Three fewer dimensions to estimate: the covariance drops from 820 free
  parameters to 703, at n/d ≈ 1.6 where that is not a rounding error.
* **F4's property improved even though the condition number got worse.** F4
  asked whether the score is dominated by numerical artefact in tight
  directions. Re-measured on the new representation: the top-5 tightest
  eigendirections contribute **6.1 %** of d² on healthy windows, down from
  **31.6 %**. The condition number rose while the score's dependence on the
  ill-conditioned part fell by 5×, which is a good reminder that a condition
  number is a summary, not a diagnosis.

Verified unchanged: `ml/evaluate.py` AUC 1.000, FPR 0.00, TPR 1.00, zero
regime-switch false alarms, transient suppressed / persistent fault alerts once;
`firmware/main.py --simulate` raises exactly 1 alert over 90 simulated minutes;
extraction still 150–164 ms per window. Suite 238 → **255 passing**.

**The investigation's real output was F9 below**, which is a larger problem than
F1 ever was.

## F10 — `channel_stats` never removes DC, and on real hardware that breaks two features ✅ FIXED

Found 2026-08-19, prompted by an external critique (Google AI Mode) which
correctly flagged the SPH0645 microphone's documented DC-offset problem. The
critique was right about the microphone. Testing it against our code found a
**second, larger instance of the same root cause that nobody had noticed.**

`channel_stats` computes `rms = sqrt(mean(x**2))` on the **raw** signal. Every
other spectral function in `features.py` removes the mean first
(`band_energy_ratios`, `_hps_peak`, `envelope_spectrum` all do `x - mean(x)`).
`channel_stats` does not. The simulator produces zero-mean signals, so this has
been invisible for the entire life of the project.

**Instance 1 — microphone DC offset.** Measured, injecting DC as a fraction of
signal RMS into an otherwise healthy window, scored against the deployed
baseline:

| DC offset | score | × threshold | alarm? |
|---|---|---|---|
| 0 (clean) | 7.11 | 0.76 | no |
| 10 % of RMS | 24.58 | **2.62** | **YES** |
| 50 % of RMS | 522.6 | **55.7** | **YES** |
| 100 % of RMS | 1622 | **172.9** | **YES** |

A DC offset is a constant. It carries no information about the machine. **A
10 % offset is enough to hold a perfectly healthy machine in permanent alarm**,
which on its own destroys the ≤1 false alarm per node-week target. The SPH0645
is documented to exhibit exactly this.

**Instance 2 — gravity, and this one is worse.** A real accelerometer sits in
a 1 g field; one axis reads ~1 g DC. Machine vibration is order 0.01–0.1 g RMS.
So the "RMS" feature reports gravity, not vibration:

| true vibration | DC-free logRMS | as-implemented, with 1 g |
|---|---|---|
| 0.05 g | −1.302 | +0.0005 |
| 0.10 g | −1.001 | +0.0020 |
| 0.20 g | −0.700 | +0.0083 |

Quadrupling the vibration moves the feature by **0.008**; correctly computed it
moves by **0.60**. On real hardware `accel_*_logrms` measures the sensor's
**mounting angle** — tilt it and the value changes — and is nearly blind to
vibration. `accel_x_logrms` is one of the three dimensions of
`baseline.operating_point`, so regime clustering would have been partly driven
by how the magnet happened to sit.

**Why the simulator could never have caught this.** It generates zero-mean
audio and zero-mean acceleration. There is no gravity in the model. This is the
fifth time an error has been found by changing the assumptions rather than by
refining the model, and the first one found from *outside* the project.

**Fix:** compute RMS, crest and kurtosis on the DC-removed signal, and report
the DC level separately as a **diagnostic**, not an anomaly feature (for the
accelerometer it is genuinely useful: it tells you the mounting orientation and
would detect the sensor falling off — but feeding it to the detector would put
mounting angle back into the score by another door).

⚠ **Applying this fix changes the feature distribution and could invalidate
`baseline.npz`** — the same trap as T1.8, where a simulator change silently
left 100 % of healthy windows above threshold while the dimension check passed.

### Resolution, 2026-08-19 — applied, and the invalidation did not materialise

Shipped in `features.channel_stats` (`ac = x - np.mean(x)` before every
statistic) with `dc_level()` reported separately under a `"dc"` key. Five
regression tests in `tests/test_features.py`, including one that asserts a DC
offset does not move the anomaly score, and one that asserts accelerometer RMS
tracks vibration rather than gravity.

The predicted baseline invalidation **did not happen, for a reason worth
stating rather than being quietly relieved about**: the simulator's signals are
zero-mean by construction, so `x - mean(x)` subtracts only the sample-mean
noise, which is tiny. The deployed thresholds are unchanged at **8.069 /
9.380**, and a 60-window replay through the real firmware
(`tools/sim_trace.py`) puts **0 of 11** healthy windows above threshold, at
ratios 0.53–0.74. So the fix is verified *neutral on synthetic data* — which is
the strongest claim available here, and deliberately a weaker one than "the fix
works". The whole point of F10 is that the simulator has no DC and no gravity.
**The fix cannot be validated by anything in this repository.** It is a
prediction about hardware, and it stays unproven until the accelerometer is
bolted to something. Recorded as such in DOC_STATUS.md.

---

## F11 — Two different test counts in the README, both wrong ✅ FIXED

Found 2026-08-19 while doing something else entirely, which is how this kind of
thing is always found.

The README claimed **31 tests** in the quickstart and **359 tests** in the
repository map. The suite collected **427**. Neither number was wrong when it
was written; both were written by hand, once, and then the suite grew past them
eight times.

**This is small, and it matters more than its size.** The repository's entire
pitch — the thing that separates it from a student project with a confident
README — is DOC_STATUS.md's proven-versus-assumed table and the rule that every
claim names the command that produced it. A hand-maintained integer sitting in
the README is an *assumed* claim wearing the costume of a measured one. And it
was the most trivially checkable claim in the project: it disagreed with itself,
in the same file, thirty lines apart. Nobody had to run anything to catch it.
Nine adversarial findings, four subagent audits and a daily review all walked
past it.

**Root cause, generalised:** claims live in two populations here — those pinned
by a test, and those not. The pinned ones have stayed correct through eight
months of churn. The unpinned ones rot silently and at a rate nobody measures.
The defect is not the stale number; it is that a docs-only claim had no
mechanism by which it could ever fail.

**Fix:** `tests/test_docs_current.py`. Three tests: both claims must exist,
both must equal what `pytest --collect-only` actually collects, and (cheaply,
with no subprocess) they must agree with each other. It caught its own
authors immediately — adding the three tests moved the true count from 427 to
**430**, and the guard failed until the README was corrected. That is the
behaviour wanted.

Deliberately *not* automated: the historical entries in the task backlog (not in this public copy)
("suite 401 → 422") are a changelog. They record what was true on a date and
must never be rewritten to match today; auto-updating them would destroy the
audit trail that makes the backlog worth keeping. Only current-state claims
are guarded.

**Also found in the same sweep** (repository hygiene, not correctness):
`.DS_Store` and `test_acoustic.db-journal` were committed — `.gitignore` had
`*.db`, which does not match a SQLite sidecar journal. And the v2 brief folder,
literally named `use this now ` **with a trailing space**, was committed
despite a `.gitignore` rule for it: git strips trailing whitespace from
patterns unless it is backslash-escaped, so the rule never matched anything.
The brief now lives in `the v2 spec (not in this public copy)/`; the two `.py` files inside it were
byte-identical copies of `ml/simulate.py` and `ml/verify_signals.py` — the same
duplicate-source trap that caused the two-repository divergence on 2026-08-19.

## F2 — Mic-only builds used a dead channel's speed estimate ✅ FIXED

**Suspicion:** `estimate_fr` prefers the accelerometer. What does it do when
there is no accelerometer — which is a **supported build**, and the one
the project's risk assessment (not in this public copy) now recommends as primary?

**Test result:** confirmed, and worse than expected.

```
true fr= 50.0  with accel ->  50.00 Hz (reliable=True)   MIC-ONLY -> 10.00 Hz
true fr= 30.0  with accel ->  30.00 Hz (reliable=True)   MIC-ONLY -> 10.00 Hz
```

The audio channel alone estimated **50.00 Hz and 30.00 Hz perfectly**. We threw
that away.

**Root cause:** a silent channel is all zeros, so every candidate frequency
ties at `log(1e-20)`, and `argmax` returns the *first* candidate — the search
lower bound, 10 Hz. That is greater than zero, so the `if fr_accel <= 0` guard
never fired, and the code preferred a dead channel's boundary artefact over a
correct measurement.

This is the nastiest class of bug: **it produces a plausible number.** Nothing
crashes, nothing logs, and 10 Hz is a perfectly believable shaft speed. Speed
feeds `baseline.operating_point`, so the error propagated straight into regime
clustering.

**Fixed.** `_hps_peak` now returns 0.0 for a dead or flat channel, and
`estimate_fr` handles all four live/dead combinations explicitly, preferring
whichever channel actually measured something and never claiming `reliable` on
a single unconfirmed channel. Four regression tests added (49 passing).

**Transferable lesson:** any `argmax` over a possibly-degenerate objective must
be guarded. Ask "what does this return when the input carries no information?"
— the answer is usually "the first element", dressed up as a measurement.

## F3 — The threshold is set by the single worst learn window ✅ FIXED (T1.6)

> **Fixed 2026-08-17, but this finding's own prescription was wrong.** F3
> preferred `chi2.ppf(0.995, df)`. Measured, that would have been a serious
> regression: our features have effective rank **13.7 of 37**, so d²
> concentrates near the effective dimensionality (mean 28.5, not 37; χ²₃₇
> rejected at KS p = 3e-90) and the resulting threshold gave an **11.0 %**
> held-out false-alarm rate at n=24 against 3.8 % for the estimator it was
> replacing. What shipped fits the scale *and* the dof from the median and
> 75th percentile. F3's second instinct — "compute both, deploy the safer,
> log the disagreement" — was right and is what ships. Full numbers in
> `DOC_DETECTOR.md` §Step 3b. Running the fix end to end also exposed a
> worse sibling bug: the same contaminated window formed its **own regime of
> size 1**, with a covariance fitted to one sample and a threshold of exactly
> **0.0** (§Step 3c).


**Suspicion:** we take the 99.5th percentile of cross-validated distances. With
~24 windows per regime, is that percentile even estimable?

**Test result:** confirmed. It is not.

```
n=  24  p99.5=7.706  max=7.770  ratio=0.9917
n=  48  p99.5=7.638  max=7.770  ratio=0.9830
n= 500  p99.5=8.174  max=9.103  ratio=0.8979
```

At n ≤ 48 the "99.5th percentile" **is the maximum observed value**. So the
alert threshold of every deployed device is set by whichever single learn
window happened to be worst.

**Consequence:** one lorry reversing outside during the learn period
permanently desensitises that unit, and nothing reports it. This partially
undercuts the CV fix we were pleased with — CV made the *distances* honest, but
the *estimator on top of them* is degenerate.

**Better options:** for Gaussian data, squared Mahalanobis distance follows
χ² with p degrees of freedom, so the threshold can be set analytically from
`chi2.ppf(0.995, df)` — stable at any n. Or fit a tail model (GPD) to the
upper distances. Or simply require far more learn windows and say so.

Preferred: compute both, deploy `min(empirical, analytic × safety)`, and log
when they disagree — the disagreement itself detects a contaminated learn
period.

→ backlog **T1.6**

## F4 — Is the score just numerical noise? ❌ REFUTED

**Suspicion:** with a near-singular covariance, the precision matrix will have
enormous eigenvalues in near-null directions, and the Mahalanobis distance will
be dominated by numerical noise rather than physics.

**Test result: wrong.** Ledoit–Wolf is doing its job properly.

```
precision eigenvalue range: 0.271 .. 2.11   (condition number 7.78)
top-5 TIGHTEST directions contribute 6.4% of d² (healthy), 13.7% (fault)
```

A condition number of 7.8 is excellent, and the score is spread across many
directions rather than concentrated in the degenerate ones. **The detector is
measuring signal, not numerical artefact.** F1 remains worth fixing on
information-efficiency grounds, but it is not corrupting the answer.

## F5 — The score is bimodal, so "amber" can never happen ⚠ **— PARTLY WRONG, corrected 2026-08-18**

> **CORRECTION (T1.7).** The headline claim below — that the score never
> enters the 0.7–1.0× amber band — is **false**, and the error was
> methodological: it was inferred from the four single-seed severity points in
> the table, which is not a way to answer a question about a *distribution*.
>
> Re-measured over **200 fresh healthy windows** against the current 37-dim
> baseline: the old amber band fires on **16.5 %** of them. Healthy
> score/threshold is min 0.283, median **0.580**, p95 **0.762**, max **1.034** —
> the distribution's own upper tail sits inside the band. On a 40-window fault
> ramp (severity 0.002 → 0.05) the band fires on only **12.5 %**.
>
> So amber was not dead UI. It was **a badge more likely on a healthy machine
> than on a failing one**, which is a worse defect than the one reported: dead
> UI is ignorable, an anti-correlated badge teaches the customer that colour on
> this dashboard means nothing. The conclusion ("the magnitude band must go")
> survives; the reason given for it did not.
>
> The magnitudes in the table below are also stale — they predate T1.5
> (40 → 37 dims) and T1.6 (threshold estimator). Current values are
> severity 0.00 → 0.47×, 0.01 → 0.69×, 0.02 → **1.88×**, 0.10 → 22.4×,
> 0.50 → 135.6×. The bimodality is real but roughly five times less extreme
> than recorded here.
>
> **Method note for future rounds:** two of this round's findings (F5 here, and
> the effective-rank claim corrected in T1.5) were distribution questions
> answered from a handful of point samples. Where a finding is about *how
> often*, generate the population.



**Suspicion:** the dashboard has a green/amber/red tier where amber = 70–100 %
of threshold. Does the score actually pass through that band?

**Test result:** confirmed, and it invalidates part of the product design.

| severity | score | × threshold | tier |
|---|---|---|---|
| 0.00 | 5.43 | 0.62 | green |
| **0.02** | **59.30** | **6.77** | **RED** |
| 0.10 | 874 | 99.8 | RED |
| 0.50 | 8054 | 918.9 | RED |

There is no amber zone. The score goes from 0.62× to 6.77× threshold between
"healthy" and "a fault 7× smaller than what we call early-stage". Two
consequences:

1. **The amber "watch" tier is dead UI.** `DOC_ALERTING.md` and
   `DOC_FRONTEND.md` both describe it as the thing that "makes the dashboard
   feel alive between alerts". It will essentially never fire. Those documents
   are wrong and are now flagged.
2. **No severity trending is possible.** A Mahalanobis distance of 8054 is not
   a physical quantity — it is "some feature moved 8000 sigmas because its
   learn-period variance was tiny". We cannot tell a customer "your machine is
   getting worse" or estimate time-to-failure from this number. Anything the
   pitch says about trending is currently unsupported.

It is also a **third independent sign that the simulation is far too easy**:
detecting severity 0.02 at 6.8× threshold is not credible on real hardware.

**Direction:** report a calibrated probability or a robust bounded score for
display (e.g. rank against the learn distribution), and keep the raw distance
only for the alert decision. Real severity trending probably needs a physical
quantity — band-limited RMS in the demodulation band, or envelope-spectrum peak
height at the detected repetition rate — not the anomaly score.

**What was done, 2026-08-18 (T1.7), and where this direction was also wrong.**
The physical-quantity half was right and shipped exactly as suggested: band RMS
moves **17.8 dB** and envelope-peak height **45.0 dB** from severity 0 to 0.5,
both Spearman ρ = **+1.000**, against 2.46 decades for the raw score. The
"calibrated probability" half was built first and **measured saturating** —
median healthy percentile **100.0000** — because the χ² fit is made on
in-sample learn distances, which are biased low by construction. What ships is
a log-ratio index anchored at 70 = the machine's own threshold; the probability
is still reported, and a test pins its saturation so that if the fit is ever
made honest we find out the display could be upgraded.

→ backlog **T1.7** (done)

## F6 — Our simulated accelerometer is three copies of one signal ⚠

**Test result:** the three axes correlate at r = 0.995–0.999.

```
[[1.     0.9987 0.9963]
 [0.9987 1.     0.9951]
 [0.9963 0.9951 1.    ]]
```

`SimulatedSource` builds y and z as scaled copies of x plus small noise. So the
12 accelerometer features carry roughly **4 features of information**, and
every simulation-derived conclusion about the accelerometer channel is
inflated. Real triaxial mounting gives genuinely different axes.

This does not affect the microphone results, which is now the primary channel —
but it means we have effectively **never tested the accelerometer path**.

→ backlog **T1.8**

### F6 outcome, 2026-08-18 — CONFIRMED, and fixed

Unlike F1, F3 and F5, this finding survived being tested. Measured before any
change (`tools/accel_axis_report.py`): r = +0.9988 / +0.9964 / +0.9952 on every
signal kind, and the 12 per-axis accelerometer statistics in the feature vector
spanned an **effective rank of 3.75 of 12**, with a four-dimensional near-null
space (smallest/largest singular value 1.3e-3). F6's "roughly 4 features of
information" was, to two significant figures, right.

Two corrections to the finding as written:

1. **The copying was never in `ml/simulate.py`.** It was three lines in
   `firmware/capture.py`, which is not a frozen file — so the fix needed no
   frozen-file exception at all. The task text's "NOTE: `ml/simulate.py` is
   frozen; this is a justified change" was based on the wrong file, and
   `simulate.py` is untouched by T1.8.
2. **Different resonances alone would not have fixed it.** The first
   implementation gave each axis its own housing mode and independent sensor
   noise, and the axes *still* correlated at 0.90, because on a healthy machine
   the shaft hum is 20 dB above everything else and all three axes saw it in
   phase. What breaks the correlation is giving each axis its own **phase**
   relative to the rotating imbalance vector. Recorded because it is the
   generalisable point: decorrelating a signal means decorrelating whatever
   dominates its variance, which on a healthy machine is not the fault.

After the fix: r = +0.04 / −0.68 / +0.51, effective rank **9.32 of 12**, sv
ratio 1.8e-2. Detection is unchanged — held-out healthy FPR 0.0292 → 0.0319,
paired 95 % CI [−0.084, +0.104], AUC 1.000 both (`tools/accel_axis_compare.py`,
200 bootstrap splits). **That is the expected result and not a disappointment:**
the simulator already scored AUC 1.000, so it had no room to show an
improvement. What T1.8 buys is that the accelerometer features can now
*disagree with each other*, so a claim about them is finally capable of being
false — which is a precondition for the week-2 recording testing anything.

One honest residual: `accel_y_kurt` is still 0.99 predictable from the x block
on healthy windows. y is the other radial axis and sees the same hum at 0.72
amplitude, so on a healthy machine its impulsiveness is nearly determined. The
axial axis breaks the tie (R² 0.04). Not tuned away.

## F7 — Mic-only mode degrades regime clustering ⚠

With no accelerometer, `operating_point` returns
`(fr, audio_logrms, -9.0)` — the third dimension is the dead-channel sentinel,
constant forever. So mic-only regime clustering runs on **2 of 3 dimensions**,
and one of those (fr) was garbage until F2 was fixed.

Not fatal — speed and audio level are the two that matter most — but
`choose_k`'s silhouette threshold was tuned in 3-D and has never been checked
in this degenerate 2-D case.

→ backlog **T1.9**

### Resolution, 2026-08-18 — confirmed as a real bug, but not the one F7 described

F7 was right that something was wrong here and wrong about all three of what,
where and how bad. Recorded in that order, because the correction is the
interesting part.

**"Not fatal" was wrong.** Measured on the real pipeline before touching
anything: 48 healthy mic-only windows from one unchanging simulated machine
were split into regimes of **30 and 18**. Over 100 bootstrap learn periods the
pre-fix rule chose k > 1 in **100 of 100** (k=2 66×, k=3 32×, k=4 2×) and the
held-out healthy false-alarm rate was **0.1358 ± 0.1445** against
**0.0217 ± 0.0290** with k forced to 1 — **6.3× the false alarms**, because
each spurious regime fits a 37-dimensional Gaussian to ~24 windows instead of
48. It also fired T1.6's learn-period-contamination warning on 14 of 200
perfectly clean fits, i.e. it manufactured a second false diagnostic. ROC AUC
was 1.000 in every arm: this bug costs nothing in detection and everything in
the one number the product cannot afford. Reproduce:
`python tools/regime_miconly_cost.py`.

**"Runs on 2 of 3 dimensions" was the wrong diagnosis, and the fix it implies
is a no-op.** A constant column contributes exactly 0 to every pairwise
distance after `(OP - op_mean) / op_scale`, so k-means, its centroids and the
silhouette are bit-identical whether the dead column is present or dropped —
measured, and pinned by `test_dropping_the_dead_dimension_is_a_no_op`. The
dead dimension was never the problem. The problem is the **one live**
dimension: silhouette's null distribution rises as dimensionality falls, so
0.5 stops being a threshold and starts being a floor the noise clears on its
own. Single-cluster noise, 1500 trials, n=48:

| directions of variance | median | p95 | p99 | max |
|---|---|---|---|---|
| 1 | 0.584 | 0.637 | 0.664 | **0.694** |
| 2 | 0.382 | 0.433 | 0.455 | 0.512 |
| 3 | 0.283 | 0.326 | 0.348 | 0.401 |
| 2, collinear | 0.584 | 0.641 | 0.660 | **0.702** |

**And it was never only a mic-only problem.** That last row is the one F7 could
not have predicted: two live channels that move *together* — audio and
accelerometer level on any machine whose load changes — are one effective
dimension and split noise **98.5 %** of the time, exactly like the mic-only
case. The bug was in the full build too, hidden behind the fact that our
simulator's accelerometer noise happens to be independent of its audio noise
(measured r = 0.11, which a real machine will not reproduce).

**What shipped:** two criteria, because measurement showed each one missing a
case the other catches. An absolute physical gate (`MIN_REGIME_SEPARATION`
= 1.0 in OPz units = 5 % of speed or 0.1 decade of level) rejects the deployed
failure, whose split is 1.5σ of the sensor noise but only 0.0002 decades wide.
A dimension-aware silhouette floor (`SILHOUETTE_MIN_1D` = 0.75, above the
measured null maximum of 0.702) rejects the mirror-image case, a machine that
genuinely wanders as much as the split. Both together: **0.000** invented
regimes in all four cloud types above, and the mic-only bootstrap arm now
reproduces the forced-k=1 oracle to the digit (FPR 0.0217 ± 0.0290).

**Method note.** F7 was reasoning from the code, not from data — "2 of 3 dims,
threshold tuned in 3-D, probably worth a look". That was enough to find a real
bug, and not enough to describe it: it named a fix that does nothing,
understated the severity by a factor of six, and missed that the same defect
was in the recommended full build. The pattern repeats from F5 and F1. A
suspicion is a place to point an experiment, never a conclusion.

## F8 — Process: stale bytecode silently ran old code

While fixing F2, `pytest` kept failing on a syntax error that `ast.parse`
could not reproduce. Cause: the mount forbids deleting files, so stale
`__pycache__` entries — including pytest's assertion-rewriting cache — could
not be cleared, and pytest imported old bytecode.

**Workaround:** copy the repo to `/tmp`, clear `__pycache__`, run there.
Added to the backlog rules so the scheduled agent does not lose an hour to it.

Related self-inflicted lesson: my first attempt at the F2 fix placed a
paragraph of prose *after* the closing `"""` of a docstring, producing a
syntax error at import. The tests caught it immediately. **The tests caught my
mistake faster than I did** — which is the argument for the whole suite.

---

## F9 — The simulator's spectral shape is one-dimensional, so 14 of 37 features carry ~2 ⚠

Found 2026-08-17 while fixing F1, by asking why the band blocks stayed
near-singular *after* the compositional constraint had been removed.

**Suspicion:** if removing a known algebraic dependency does not restore rank,
the remaining degeneracy is not algebra. Either the transform is broken or the
data is genuinely low-dimensional.

**Discriminator used:** an algebraic constraint has the *same* null direction on
any data; a data-structure one does not. So: compute the null direction of each
block on two independent samples of windows and compare.

```
                    sv ratio (A / B)    |cos(null_A, null_B)|
audio_band_ilr_     1.21e-03 / 1.17e-03        0.999
accel_band_ilr_     3.68e-03 / 3.85e-03        0.985
env_ilr_            3.40e-01 / 3.52e-01        0.698
ILR of random 8-part compositions: sv ratio 0.497  <- transform is fine
```

The envelope block is healthy and its weakest direction wanders, as a
full-rank block's should. The two band blocks reproduce their null direction to
|cos| = 0.999 — systematic, not sampling noise — and the transform is exonerated
by the random-composition control.

**Cause, measured directly.** The participation rank of the 8 log band-fractions
across 24 healthy windows at two speeds:

```
audio log-fractions: participation rank 1.03 of 8
   normalised svs: [1.0, 0.065, 0.013, 0.0079, 0.0061, 0.0039, 0.0013, 5.3e-05]
accel log-fractions: participation rank 1.01 of 8
```

**The eight band fractions of a `simulate.py` healthy signal are a
one-parameter family.** One dominant band holds 96 % of the energy and the rest
sit in an almost fixed pattern. Adding a severity-0.15 bearing fault to the mix
barely moves it (sv ratio 4.5e-3).

**Why this matters more than F1 did.** F1 was worth one dimension per block.
This is worth about six per block: the 14 band features carry roughly 2
dimensions of information between them, and every simulation result that leans
on spectral shape is therefore inflated. It is the **third** instance of the
same species — F6 (three accelerometer axes that are scaled copies), F5 and the
surrogate run (the simulation is far too easy) — and the pattern is now
unmistakable: *`ml/simulate.py` generates signals with fewer independent degrees
of freedom than the feature vector claims to measure, so the feature vector
cannot be evaluated on it.*

**What would settle it:** one real recording. `band_fractions` on a real motor,
in a real room, with load changes and other machinery, should span materially
more than 1.03 of 8 dimensions.
`tests/test_compositional.py::test_simulator_spectral_shape_is_one_dimensional`
asserts the current value so the first real data visibly breaks it — the test
tells you to update the bound and record the real number rather than delete it.

**Deliberately not "fixed."** Enriching the simulator so its spectra vary more
would make this number go up without making it true. The number is honest
information about a known-inadequate simulator; the fix is hardware.

→ backlog **T1.10**

**Update 2026-08-18 (T1.10 done).** The rank measurement above stands
unchanged. What T1.10 added: rank alone does not tell you whether a block
contributes to detection. A per-block Mahalanobis distance trained on
healthy-only windows and scored against held-out bearing-fault and
imbalance-fault windows shows both band-ILR blocks detecting either fault at
AUC > 0.9, despite the near-one-dimensionality measured above. The complementary
surprise: the envelope block, full rank and doing most of the bearing-fault
work, is at chance (AUC 0.447) for the imbalance fault — full rank, no
information about that particular deviation. See `docs/DOC_STATUS.md` §What
quantifying feature-block dimensionality taught us, and
`tools/feature_block_report.py`.

---

## F12 — The mic driver's own power-on "thump" is invisible to the simulator, and the code re-triggers it every window ⚠ UNFIXED

Found 2026-08-20, same method as F10: take a documented real-world property
of the *exact* hardware/driver/OS combo this project has chosen (Pi Zero 2 W
+ SPH0645 + `googlevoicehat-soundcard` overlay + Bookworm — the exact stack
in the parts list (not in this public copy) and `scripts/provision_pi.sh`), and check whether our
code is blind to it. It is.

**[Reported by source, independent of this project.]** Martin Hodges,
*"Setting up a MEMS I2S Microphone on a Raspberry Pi"* (Medium, Jul 2025),
building this exact stack (SPH0645, Pi Zero 2 W, `googlevoicehat-soundcard`,
Bookworm), reports two hardware problems. The first — "a small, negative DC
offset" — independently corroborates F10 from a source that never saw this
repo. The second, not previously recorded here: **"a 'thump' when the device
powers up,"** observed "even with continuous recording... at irregular
intervals," attributed to the driver "powering the device on and off," and
separately: **"when the microphone capture starts and stops, it produces a
thump."** The thump "starts with a transient" and "lasts around 600–700ms";
the author states plainly, after trying several filtering approaches,
"removing the 'thump' is problematic and I have not found a way to do so."
The general mechanism (DAPM power-managed codecs popping "every time a
component power state is changed") is documented independently in the Linux
kernel's own audio documentation, so this is not a one-off anecdote about one
person's wiring.
[Source](https://medium.com/@martin.hodges/setting-up-a-mems-i2s-microphone-on-a-raspberry-pi-306248961043),
[Linux kernel: Audio Pops and Clicks](https://www.kernel.org/doc/html/latest/sound/soc/pops-clicks.html).

**[Measured, this repo, by reading the code.]** `firmware/capture.py`,
`HardwareSource.windows()` (line ~771), calls `self.sd.rec(...)` **fresh
inside the per-window loop** — a new PortAudio/ALSA capture stream is opened
and closed every 30 s window, not one continuous stream read from a ring
buffer. That is precisely the "capture starts and stops" trigger the source
above names. This is a *structural* difference from the "one single-window
transient fault must NOT alert" case `firmware/main.py --transient-at-minute`
already demonstrates is handled safely (`docs/DOC_SELF_REVIEW.md`'s round-2
list, and the persistence gate in `AlertGate`): that gate exists to make a
lone bad window harmless by requiring `need` **consecutive** anomalous
windows before alerting. A thump that recurs on *every* window is not a lone
bad window — it is sustained by construction, which is exactly the pattern
the persistence gate cannot filter out.

**[Measured, this run — synthetic proxy, injected into the real pipeline.]**
Wrote a broadband noise burst (0.65 s, exponential-decay envelope, tau =
0.15 s — a generic stand-in for "starts with a transient, needs both
high-pass and low-pass to attenuate," since the source gives no exact
waveform) and added it to the first 0.65 s of an otherwise clean
`SimulatedSource` healthy window (same "normal", fr ∈ {30, 50} Hz schedule
the deployed `baseline.npz` was trained on), then scored the result with the
real, deployed `MahalanobisScorer`. Six independent trials, three relative
amplitudes (fraction of the window's own audio RMS):

| trial | clean score | threshold | +0.3× thump | +0.5× thump | +1.0× thump | +2.0× thump |
|---|---|---|---|---|---|---|
| 0 | 5.46 | 9.38 | 22.9 (2.4×) | 42.9 (4.6×) | 107.6 (11.5×) | 310.4 (33.1×) |
| 1 | 7.83 | 9.38 | 24.3 (2.6×) | 55.7 (5.9×) | 102.7 (11.0×) | 287.4 (30.6×) |
| 2 | 5.17 | 9.38 | 19.9 (2.1×) | 42.2 (4.5×) | 106.3 (11.3×) | 278.7 (29.7×) |
| 3 | 5.63 | 9.38 | 25.6 (2.7×) | 44.3 (4.7×) | 110.4 (11.8×) | 290.2 (30.9×) |
| 4 | 5.99 | 9.38 | 25.6 (2.7×) | 51.1 (5.4×) | 111.2 (11.9×) | 283.4 (30.2×) |
| 5 | 7.12 | 9.38 | 28.4 (3.0×) | 50.0 (5.3×) | 114.1 (12.2×) | 288.0 (30.7×) |

Every clean window scores well under threshold (5.2–7.8 vs 9.38, matching
F10's post-fix healthy-window numbers). Adding the thump proxy at just
**30 % of the window's own audio RMS** — a small transient by ear — already
puts every trial at 2–3× threshold. At 1× it is ~11–12× threshold; at 2× it
is ~30×. This is the same shape and similar order of magnitude as F10's DC
table (2.6× at 10 % offset, 55.7× at 50 %), for the same underlying reason:
`channel_stats`' crest and kurtosis (and `select_demodulation_band`'s
envelope-peakiness) are built to react hard to exactly this kind of sharp,
broadband, short-duration event, because that is what a real bearing impact
also looks like. This method (write the test, run it against the real
scorer) is the one the task brief asks for; the numbers above are
**measured**, not estimated.

**What is NOT measured, and must not be overstated.** Whether the real
SPH0645 + `googlevoicehat-soundcard` on Logan's own Pi Zero 2 W actually
produces a transient of this exact shape, amplitude, or even reliably on
every window — that is a property of the real driver and has not been
observed on this project's own hardware. The 0.65 s/tau=0.15 s envelope is a
reasonable stand-in built from the source's description ("starts with a
transient," "lasts around 600-700ms"), not the source's own waveform. Treat
this table as "IF a transient anywhere near this shape occurs, THEN the
consequence is a false alarm of this size" — the antecedent is reported by
an external, independent source but unverified on our hardware; the
consequent is measured against our own code today.

**Why the simulator could never have caught this — same root cause as F10.**
`ml/simulate.py` models the machine's acoustic signal; it has no model of the
capture *device's own electrical behaviour at stream start*. This is the
same species of gap as "no DC, no gravity" (F10) and belongs on the same
list as the "no clipping" hazard already named under §What this round did
not test — the simulator is faithful to the physics of the machine and
silent about the electronics of the sensor path.

**Not fixed, and no mitigation code was written** (task brief: do not write
mitigation code for hardware nobody owns). Two candidate directions, for
whoever picks up the backlog item, with their own honest caveats: (a) keep
one continuous `sd.InputStream` open across windows instead of calling
`sd.rec()` per window — addresses the "capture starts and stops" trigger the
source names, but the source also reports thumps "even with continuous
recording... at irregular intervals," so this would reduce but not
provably eliminate the hazard; (b) discard/window out the first ~0.7 s of
every captured audio buffer before feature extraction — cheap and
hardware-agnostic, but throws away real signal too and does nothing for a
thump that recurs mid-stream. Neither has been tried or tested here; both
are guesses about a problem only real hardware can confirm or refute.

→ backlog **T2.4** (added at the top of the task backlog (not in this public copy))

---

## F13 — `HardwareSource` measures the exact failure mode outside sources say is most likely, then nobody reads the measurement ⚠ UNFIXED

Found 2026-08-22. Same method as F10/F12: take a documented real-world
property of a component this project chose, check whether our code notices
it if it happens. This time the code doesn't even need hardware to fail the
check — it's a pure code-inspection finding, confirmed by `grep` across the
whole repo.

**[Reported by source, independent of this project.]** The IIS3DWB's FIFO
holds 512 words; at the batched output data rate this project uses
(26.667 kHz), that is documented in ST's own application note (AN5444) and
in this repo's own code comment (`firmware/capture.py` ~line 782) to fill in
**~19 ms**. The only real-world report found this run of someone actually
draining this exact FIFO from software without overrunning it —
[metebalci/iis3dwb-247](https://github.com/metebalci/iis3dwb-247), a 7-day
soak test on a Raspberry Pi 4 — needed a C program, two **dedicated,
`isolcpus`-isolated CPU cores**, and a producer/consumer thread pair to get
`cnt_ovrs: 0` over a week. That is a materially more powerful board (Pi 4 vs
this project's Pi Zero 2 W) and a materially more careful design than a
single Python thread polling `drain_fifo()` + `fifo_flags()` in a
`time.sleep(0.005)` loop on the same core that just issued a blocking
`sd.rec()` call. Separately, a Raspberry Pi forum thread title —
["Python Threads Introduces 10msec Delay in SPI"](https://forums.raspberrypi.com/viewtopic.php?t=394891)
— names Python scheduling jitter on exactly this kind of loop as a known
problem; the thread's body could not be fetched this run (returned empty),
so this citation is the title only, not the detail, and is flagged as such.

**[Tested this run, but only in this sandbox, not on target hardware.]**
Measured `time.sleep(0.005)` jitter directly: 2,000 iterations with no
background load (mean 5.32 ms, max 7.58 ms) and 2,000 more with 4 busy-loop
processes pinned across all cores to simulate contention (mean 5.10 ms, max
12.98 ms). Neither run came close to the ~19 ms overrun deadline. **This
does not clear the hazard** — this sandbox is a cloud VM, not a Pi Zero 2 W
(weaker, quad Cortex-A53 @ 1 GHz), and the measurement excludes the actual
SPI burst-read time, the concurrent ALSA audio thread, and any other
processes real deployment adds. It shows the sleep primitive itself isn't
pathological *here*; it says nothing about the target board. Labelled
untested-on-target, not refuted.

**[Measured, this repo, by reading the code — this is the real finding.]**
`HardwareSource` computes exactly the diagnostics that would tell you if the
above happened: `self.last_fifo_overrun` (set True the moment
`fifo_flags()["overrun"]` fires), `self.measured_fs_accel` and
`self.measured_fs_audio` (the actual achieved rate vs the configured one).
`grep -rn "last_fifo_overrun\|measured_fs_accel\|measured_fs_audio"` across
the entire repository returns **only the three lines in `capture.py` that
write them.** Nothing reads them:

- `firmware/main.py`'s window loop (`for i, (audio, accel) in
  enumerate(source.windows())`, line 124) unpacks exactly two values and
  discards the `HardwareSource` object's diagnostic state entirely.
- `firmware/bench/record_session.py` — named in `capture.py`'s own docstring
  as a consumer ("so callers (and `bench/record_session.py`) can report
  MEASURED rates rather than configured ones") — does not, in fact, read
  them. It calls `HardwareSource(...)`, iterates `src.windows()`, and writes
  each segment's WAV file using `args.fs_audio`/`args.fs_accel` — the
  **configured**, not measured, rates. The comment's claim about this
  specific file is false; confirmed by reading `record_session.py` directly,
  not inferred.
- `grep -rn "overrun" tests/` returns nothing. No test exercises this path
  at all (none can, fully, without real SPI hardware — but nothing pins
  even the "downstream code ignores this attribute" behaviour, which
  requires no hardware to test).

**Consequence, reasoned not measured (no hardware to measure it on):** if a
real overrun happens, today's code (a) logs one `log.warning` line, easy to
miss in a Pi running unattended for months, and (b) proceeds to decimate and
score whatever partial/gapped accelerometer data `drain_fifo()` returned, as
if it were a complete window. This is the same *shape* as F2 (a dead channel
returning a plausible wrong number) and the same *root cause* as F10/F12
(hardware-only behaviour with no simulator model, so nothing exercises it):
the difference here is the code already computes the exact signal that
would catch it, and then nobody looks.

**Not fixed, no mitigation code written** (per this task's standing rule:
don't write code for hardware nobody owns). What a fix would look like, for
whoever picks this up: surface `last_fifo_overrun` (and a large gap between
configured and `measured_fs_accel`) as a field on the window/reading, the
same way `accel_ok` already degrades cleanly to constant zeros — so a
degraded window is *visibly* degraded rather than scored as if nothing
happened. Needs a real IIS3DWB to confirm overruns actually occur before
prioritising above T2.4.

→ backlog **T2.5** (added at the top of the task backlog (not in this public copy))

---

## What this round did not test

Honest list of live hypotheses that remain unexamined:

- **Learn-period contamination.** If a customer installs on an already-failing
  machine, we learn the fault as normal and never alert. There is no check for
  this. Commercially serious: plenty of machines worth monitoring are already
  degraded.
- **Window independence.** The persistence-gate arithmetic assumes anomalous
  windows are roughly independent. They are not — machine state is
  autocorrelated, so runs of 60 are far more likely than the naive calculation
  implies. `soak_report.py`'s geometric tail model may be optimistic.
- **Baseline drift.** A slowly developing fault could be absorbed by periodic
  retraining, making the device blind to exactly the failure mode it exists to
  catch.
- **Clipping.** If the real microphone returns samples outside ±1,
  `record_session.py` clips them silently — destroying precisely the impulses
  we need.
- **A53 timing.** The 150 ms → 1.2–1.5 s extrapolation assumes ~8–10× ARM
  slowdown. FFT-heavy code without AVX can be worse. The protrugram filters six
  bands over 480 k samples every window.

---

## Round 2 candidates

Question everything again, starting with the things that have never been
questioned once:

1. Is 30 s the right window? It was assumed on day one and never tested.
2. Is Mahalanobis the right detector, or would a simple one-class SVM or a
   k-NN distance behave better at n/d ≈ 1?
3. Is the persistence gate's fixed 30 minutes better than a
   cumulative-sum (CUSUM) detector, which is the statistically optimal way to
   detect a sustained shift in a noisy signal?
4. Why are we scoring *windows* independently at all, rather than tracking a
   state estimate across time?
