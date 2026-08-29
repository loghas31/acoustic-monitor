# Contributing

This project was scoped for two people (physics / CS) and has been solo since
2026-08-18 — the ownership split in the original the execution plan (not in this public copy) is void,
do not reinstate it. This file replaces it: not a division of labour, but the
four things future-you will have forgotten by the time they matter — what
never gets committed, how to actually run the suite, why some files refuse
edits, and the branching habit worth keeping even solo. If a scheduled agent
is also working this repo (see the task backlog (not in this public copy)), the same rules bind it.

## Branch per experiment

`git checkout -b hardware-i2s`, merge to `main` when it works, delete the
branch. This repository's actual history so far is three commits straight
onto `main` (`git log --oneline`: `5feee91`, `21b0b9f`, `ce39239`) — the
convention has never been exercised, and this paragraph is honest about that
rather than implying otherwise. It matters more once hardware work starts:
`H2`/`H3`/`H4` (bring-up, seeded-fault recording, soak) each produce a
`baseline.npz` and a set of readings that the *next* experiment should not
silently sit on top of. A branch per hardware session, merged only once
`ml/evaluate.py` says `STAGE 3 GATE: PASS` against it, gives you a clean
place to abandon a bad session without touching `main`.

The scheduled agent cannot create branches or commit — the sandbox mount
forbids the file deletions git needs (see the task backlog (not in this public copy) rule 4b) — so
every agent run lands as uncommitted working-tree changes and a block in
the commit log (not in this public copy) for you to paste into GitHub Desktop. That means *you*
are the one who decides whether an agent's run becomes its own branch or
folds into whatever you're already doing; the agent has no opinion on it.

## What must never be committed

`.gitignore` already excludes the obvious things (`data/`, `*.db` and its
`-journal`/`-wal`/`-shm` siblings, `node_modules/`, `frontend/dist/`, `.env`,
`firmware/baseline.npz`, `ml/artifacts/`). Three rules worth keeping in your
head rather than trusting the file to enforce silently:

1. **Real recordings and datasets don't go in git, ever, licensed or not.**
   `data/` is ignored specifically so a phone recording or a downloaded CWRU
   `.mat` file can't be added by accident. CWRU's own bearing dataset
   publishes no redistribution licence — only "(c) Case Western Reserve
   University" — so this is a legal boundary, not just a repo-size one. If
   you need to share a real recording with a collaborator, use Drive/OneDrive
   and reference the link in `RESULTS.md`, per the execution plan (not in this public copy)'s original
   convention.
2. **Keys and broker credentials never go in git.** There is currently
   nothing to leak — no `.env.example`, no committed secret, checked directly
   (`find . -iname ".env*"` returns nothing tracked) — but the day MQTT/cloud
   credentials exist, `.env` is already ignored; keep using it rather than
   `config.yaml`, which is not.
3. **Adding a pattern to `.gitignore` does not remove a file already
   tracked.** `backend/.DS_Store`, `frontend/.DS_Store` and `ml/.DS_Store`
   are tracked *right now* despite `.DS_Store` being in `.gitignore` —
   verified with `git ls-files | grep DS_Store`, three hits. They were
   committed before the ignore rule existed, and the rule only stops new
   additions. To actually remove one: `git rm --cached <path>` (keeps the
   local file, drops it from git), then commit. This exact gap is also how
   `test_acoustic.db-journal` and a folder literally named `use this now `
   (trailing space — `.gitignore` silently strips trailing whitespace from
   patterns unless escaped) both ended up committed; both are documented as
   findings F11 in `docs/DOC_SELF_REVIEW.md`. If `git status` ever shows a
   tracked file you don't recognise, check whether it predates the ignore
   rule before assuming the rule is broken.

Also worth avoiding on principle, not because anything currently enforces it:
duplicate copies of source files kept "for reference" in a sibling folder.
The `use this now ` incident above was exactly this — two byte-identical
copies of `ml/simulate.py` and `ml/verify_signals.py` sitting outside the
tree they belong to, which is precisely the two-repository-divergence trap:
edit one copy, the other silently goes stale, and nothing tells you which
one ran.

## Running the suite

```bash
TMPDIR=/tmp python3 -m pytest tests/ -q -p no:cacheprovider --basetemp=/tmp/pt
```

Not plain `pytest tests/` — the mount this project's agents run in has no
POSIX temp-cleanup semantics, and pytest's own temp-directory cleanup crashes
without `TMPDIR=/tmp` and `-p no:cacheprovider`. `--basetemp` must be a path
that does not already exist; reusing one makes every `tmp_path`-based test
fail with `FileExistsError` (the task backlog (not in this public copy) records 22 spurious failures
from exactly this). Re-run this file's command verbatim before trusting the
result: this run reproduced it clean at **443 passed** in ~114 s.

**The stale-bytecode trap.** `__pycache__/` cannot be deleted on the mount
(the mount forbids the deletions git and `rm -rf` both need for tracked
paths), so pytest can import *old* compiled bytecode for a file whose current
source is fine, and report a confusing syntax or import error that has
nothing to do with what you just edited. If test output stops making sense —
an error in a line that doesn't exist any more, or a `SyntaxError` that
`python3 -c "import ast; ast.parse(open(f).read())"` disagrees with — copy
the repo to a fresh location and run there instead of trying to debug the
mount in place:

```bash
cd /tmp && rm -rf trepo && mkdir trepo
cp -r firmware ml tests backend tools conftest.py trepo/
cp firmware/baseline.npz trepo/firmware/ 2>/dev/null
find trepo -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
cd trepo && TMPDIR=/tmp python3 -m pytest tests/ -q -p no:cacheprovider --basetemp=/tmp/pt2
```

**Correction to this exact recipe, found and fixed while writing this file**
(the version previously in the task backlog (not in this public copy) §7 omitted `tools/`): once
`tools/ingest.py` and `tools/phone_monitor.py` existed, `tests/test_ingest.py`
and `tests/test_phone_monitor.py` import them directly, and a copy that
excludes `tools/` fails to *collect* — two `ImportError`s that look exactly
like a stale-bytecode syntax error but aren't. Reproduced both ways this run:
without `tools/`, `2 errors during collection, no tests ran`; with it added,
collection succeeds and the suite runs. the task backlog (not in this public copy) §7 has been
corrected to match. Two more failures are *expected* from this scratch copy
and are not bugs: `test_docs_current.py` (needs `README.md`, which the recipe
above deliberately doesn't copy) and `test_frontend_trend.py` (needs
`frontend/`). Copy `README.md` and `frontend/` alongside the directories
above if you want a fully faithful check instead of the fast stale-bytecode
check.

## The frozen-file rule

```
ml/simulate.py            ml/verify_signals.py
firmware/features.py      firmware/baseline.py      firmware/inference.py
firmware/state.py         firmware/main.py
tests/*.py  (add to these freely; do not weaken existing assertions)
```

These files carry the project's actual evidence — AUC 1.000, the envelope
contrast numbers in `RUN_IT.md`, the gating behaviour `main.py --simulate`
prints. Editing one for style, a "cleaner" refactor, or because it looked
mergeable with something else risks quietly invalidating a result that took
a full run to establish and is now cited in `docs/DOC_STATUS.md`,
the handover notes (not in this public copy) or a backlog entry as settled. `firmware/capture.py` is
**deliberately not on this list** — several backlog entries (T1.8 among them)
found and fixed real bugs there, and it currently has uncommitted changes
from this session's phone-recording work. Check which file actually holds
the code in question before invoking this rule on it; `capture.py` and
`features.py` sit right next to each other in the tree and are easy to
confuse.

The one legitimate way through: **write a failing test that demonstrates the
bug first**, then fix the frozen file, then re-run `ml/evaluate.py` and the
full suite and report whether the numbers moved. the task backlog (not in this public copy)'s run log
has several examples of this done properly — T1.6 (threshold estimator),
T1.8 (accelerometer axes), T1.9 (mic-only regime clustering), F10/T0.1
(`channel_stats` DC removal) — each one names the failing test it started
from and the before/after numbers it produced. A frozen-file edit with no
failing test and no re-measurement is the one thing this project cannot
recover from silently: T1.8's own history is the proof — a firmware change
that left the feature *dimension* unchanged but the *distribution* invalid
put 100% of fresh healthy windows at a median 138.4× their alarm threshold,
and the dimension-only contract check in place at the time passed the whole
way through. Re-measure, don't just re-check the shape.
