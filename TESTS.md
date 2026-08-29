# TESTS.md — run the detector on your fridge tonight

**One command, one recording, no hardware, no purchases.**

```bash
python tools/fridge_scan.py ~/Downloads/fridge.m4a
```

Everything below is how to get to that line and how to read what comes out.

---

## Read this first — it changes what you record

**You cannot hold your phone to the fridge for a minute and get a verdict.**

This is not a tuner that listens and names the note. It is an anomaly
detector: it learns what *your* fridge normally sounds like, then flags
departures from that. One short recording gives it nothing to compare against.

So a single recording has to be long enough to contain **both** the learning
and the scoring:

| You record | Spent learning | Actually scored | Verdict worth having? |
|---|---|---|---|
| 10 min | — | — | refused, and it tells you why |
| 25 min | 24 min | 1 min (2 windows) | technically runs, proves nothing |
| **40 min** | 24 min | **16 min (32 windows)** | **yes — record this** |
| 60 min | 24 min | 36 min (72 windows) | better |

**Record 40 minutes.** Put the phone down and go and do something else.

**The fridge should be healthy while it learns — and if you can't be sure, run
the cold-start screen too.** The main detector has no idea what "healthy" means
in the abstract; it only knows what this machine did during the learn period. A
fridge that is already faulty gets its fault learned as normal, and the detector
stays silent forever.

Since you probably *can't* know your fridge is healthy, run this as well — it
needs no baseline and no history:

```bash
python tools/cold_start_screen.py --self-test     # see what it can do
python tools/cold_start_screen.py data/_scan_work/your_recording.wav
```

It looks for a periodic impact train — the signature of a bearing or gear
fault — directly in the recording. Measured: it catches an advanced fault 6/6
and correctly reports the true fault frequency (73.65 Hz BPFO) without being
told the bearing type. It does **not** catch early faults (0/6 at moderate
severity) and is blind to non-impacting problems like a tired compressor.

So: a hit means investigate. A clean result is weak evidence, not a clean bill
of health. Full reasoning and the measurements in `docs/COLD_START.md`.

The 24-minute learning floor is not me being cautious. Below 48 learn windows
the measured held-out false-positive rate is **55–59 %** — a coin flip that
prints confident-looking numbers. `--learn-windows` will let you go lower and
will warn you every time; that setting is for debugging, not for impatience.

---

## Step 0 — Pre-flight (2 min, do it before you record)

Open **Terminal** (⌘-Space, type "Terminal"). Paste these two lines:

```bash
cd "/path/to/acoustic-monitor"
python tools/fridge_scan.py --preflight
```

It checks the maths libraries, checks ffmpeg, and then runs the genuine
learn→score pipeline on a known-healthy synthetic signal to confirm it behaves.
You want:

```
Ready. Go and record 40 minutes.
```

If it says anything else, it tells you the exact fix. **Do this before
recording, not after** — finding out your ffmpeg is missing when you already
have a 400 MB file is annoying; finding out after you've returned the fridge to
normal is worse.

(If `python` isn't found, try `python3`.)

## Step 1 — Set the phone to Lossless (2 min, do it once)

**Settings → Apps → Voice Memos → Audio Quality → Lossless**

(Older iOS: Settings → Voice Memos → Audio Quality.)

Skip this and the test is worthless. The default is compressed AAC, which
deletes everything above a hard cutoff around 10 kHz — and bearing and
compressor fault signatures live from roughly 1–20 kHz. The file will play
back sounding completely normal. The evidence is simply gone.

Both settings produce a `.m4a`, so the filename won't tell you which you got.
Sanity check: lossless is ~10 MB/min, compressed ~2 MB/min. A 40-minute
lossless recording should be around 400 MB. If yours is 80 MB, you recorded
the wrong thing.

## Step 2 — Record 40 minutes (40 min, mostly waiting)

1. Phone **flat on top of the fridge**, screen up, not held, not leaning.
2. Voice Memos → record → leave it.
3. **Don't move it, and don't move the fridge.** Don't open the door.
4. Note the time you started and roughly what the kitchen was doing.

Quiet house is better, but ordinary background is fine and arguably more
honest — a real deployment has background too.

## Step 3 — Get it onto the laptop (2 min)

Voice Memos → tap the recording → **⋯** → **Share** → **AirDrop** → your Mac.
It lands in `~/Downloads`.

## Step 4 — Run it (5–10 min)

**You do not upload the recording anywhere.** Not to GitHub, not to me, not to
any website. It stays on your laptop and the analysis runs there. You don't
need VS Code or any editor open — this is a Terminal command and nothing else.

In Terminal:

```bash
cd "/path/to/acoustic-monitor"
python tools/fridge_scan.py ~/Downloads/"New Recording 12.m4a"
```

Replace the filename with your actual one. **Quote it** — phone filenames have
spaces in them, and without quotes the shell reads them as three arguments.

Easiest way to get the path right: type `python tools/fridge_scan.py ` (with
the trailing space), then **drag the file from Finder into the Terminal
window**. It pastes the correct quoted path for you.

That's it. The script converts the file, checks it's usable, learns your
fridge's normal, scores the rest, and prints the table.

**Do not put the recording in GitHub.** `data/` is gitignored precisely so
audio of your kitchen can never be committed. Uploading it would achieve
nothing anyway — GitHub stores files, it doesn't run the analysis.

---

## Step 5 — Reading the result

You get a row per scored window and three summary answers.

```
   w       band Hz  crest fired   fr Hz  rel reg    score     thr  x thr  flag
  48  3000-6000      6.8    no    10.4    n   0     6.13    7.41   0.83
  57  3000-6000      5.7    no    50.0    n   1     7.36    5.17   1.42    !!
```

- **`score` vs `thr`** — how far this window sits from normal, against the
  threshold learned for its regime.
- **`x thr`** — the ratio. Under 1.0 is normal. Over 1.0 is flagged `!!`.
- **`reg`** — which learned regime (operating state) it was matched to.

**`Q3` is the headline:** the percentage of scored windows above threshold.

### What each outcome means

**Q3 = 0 %.** The detector did not fire on your fridge. That is the expected
result for a working fridge and it is worth having — but be precise about what
it proves: *it did not false-alarm*. It does **not** show it would catch a
fault, because you never showed it one.

**Q3 = a few %, flags scattered.** Usually the fridge changed state — see the
trap below. Look at the `reg` column on the flagged rows.

**Q3 = high, or flags clustered together in time.** Something genuinely
changed during the recording. Most likely the kitchen, not the fridge. What
did you do at that point?

### The trap you will probably hit: fridges cycle

**A fridge compressor switches on and off.** That gives it two genuinely
different normal states, and the detector learns them as separate regimes
(you'll see `k=2`). If the compressor ran for 30 of your 40 minutes, the
"off" regime is learned from very few windows, its threshold is estimated
badly, and ordinary windows in that state get flagged.

I hit exactly this while testing the pipeline: **2 of 16 windows flagged on a
signal I knew was healthy**, both in the under-learned regime.

So: **if flagged rows share a `reg` value that's rare in the table, suspect the
regime, not the fridge.** The fix is a longer recording that covers both states
properly. This is a real limitation of the current detector, not a mistake you
made — it's why the recording length matters more than anything else here.

---

## Step 6 — Then make it prove something (optional, +30 min)

Everything above shows the detector doesn't cry wolf. It doesn't show it can
detect anything. For that you have to give it a fault to find.

`docs/FRIDGE_TEST.md` Part B has the protocol — the safe version is loading the
machine (fill it, leave the door ajar so the compressor runs hard) rather than
damaging it. Record another 40 minutes in that state and compare.

**This is the result worth putting in a dissertation.** "It stayed quiet on a
healthy machine and fired on a loaded one" is a real claim. "It stayed quiet"
alone is half an experiment.

---

## The other free test: DCASE (30 min, mostly downloading)

Real industrial machines — pumps, fans, valves — recorded by other people,
labelled normal and anomalous. Machines you didn't control, which is exactly
what makes it worth more than the fridge.

**I could not download this for you** — Zenodo is outside the network allowlist
I'm permitted to fetch from. You'll need to grab it yourself:

**[zenodo.org/records/3678171](https://zenodo.org/records/3678171)**

*(Corrected 2026-08-28, F24: this used to link `zenodo.org/records/3384388` —
that's MIMII, a different dataset with a different licence and no `train`/`test`
split. `dev_data_pump.zip`/`dev_data_fan.zip` below don't exist there. See
`docs/REAL_DATA_SOURCES.md §1` if you want MIMII instead — it's real too, just
a different download and a different tool path.)*

Download `dev_data_pump.zip` or `dev_data_fan.zip` (start with those — they're
rotating machines, closest to what this detector is built for). Unzip into
`data/`, then:

```bash
python tools/dcase_eval.py data/dcase2020/pump
```

You can confirm the tool works before downloading anything:

```bash
python tools/dcase_eval.py --self-test
```

That proves the plumbing — folder walking, clip joining, labelling. It says
nothing about real pumps. **Nothing in this repo has been run against the real
DCASE data yet**, so you'll be the first to see that number. It may not be
flattering; that's the point of running it.

**Licence — this one matters.** DCASE 2020 Task 2 is **CC BY-NC-SA 4.0**. The
**NC** is NonCommercial:

- ✅ Fine: validating the detector, results in your dissertation, a figure in a
  funding application.
- ❌ Not fine: training a model you sell or that ships in a paid product.

Keep it in `data/` and never let it become training data for anything
commercial.

---

## Where files go

**`acoustic-monitor/data/`** — recordings, downloads, everything.

It already exists and is **gitignored** (`.gitignore` line 4), so:

- Nothing you put there is ever committed or made public. Record your kitchen
  freely.
- **It is not backed up either.** If the laptop dies, those recordings are
  gone. Copy anything you care about somewhere else yourself.

The script writes working files to `data/_scan_work/`. Delete it whenever.

---

## If something goes wrong

| It says | Do this |
|---|---|
| `TOO SHORT` | Record longer. It shows the arithmetic. Don't lower `--learn-windows`. |
| `ffmpeg is not installed` | `brew install ffmpeg` |
| `No such file` | Quote the path. Check `~/Downloads`. |
| `below the documented floor` | You passed `--learn-windows` under 48. Results are debugging aids only. |

Each step prints the exact command it ran, so you can copy any single one and
re-run it on its own to see the full error.

---

## What this whole exercise is and isn't

It converts "works on signals I generated myself" into "works on a machine I
didn't design". That is a genuine step up and it's the claim that carries
weight in a dissertation.

It is **not** validation of the product. That needs a machine with a known,
deliberately-introduced fault — the task backlog (not in this public copy) H3, which needs hardware
that's currently out of stock. The fridge test is the best available evidence
that needs nothing you don't already own.

Related: the manual-steps guide (not in this public copy) (every manual step in the project),
`docs/FRIDGE_TEST.md` (the full experimental protocol and its limits),
`docs/RUN_IT.md` (the synthetic pipeline, no recording needed).
