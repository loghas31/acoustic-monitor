# The fridge test — your first real data

The first experiment in this project that uses a real machine. It costs an
afternoon and nothing else.

## What it can and cannot prove

**It cannot test "good vs bad frequencies".** Your fridge is healthy, so there
is no bad recording to compare against, and the detector does not hold a
library of what a healthy fridge sounds like — it learns *this* fridge and
watches for change (`README.md` §What this is and what it is not).

What it can settle, and these are the two open questions that matter most:

1. **Does it false-alarm on a real machine?** Every false-alarm number in this
   repo is from synthetic audio. the project's risk assessment (not in this public copy) names alarm fatigue as
   churn risk #1. A fridge running normally for an hour is the cheapest
   possible test of the thing most likely to kill the product.
2. **Does it survive a real duty cycle?** A fridge compressor switches on and
   off. That is exactly the multi-regime case `baseline.choose_k` exists for.
   Compressor-on and compressor-off should become **two regimes**, not an
   anomaly. If a normal compressor start trips an alert, the persistence gate
   and regime clustering do not work on real machines, and better to know now.

And with one piece of card you can manufacture a genuine third test — see
Part B.

---

## Before you start (10 minutes, do not skip)

1. **iPhone Settings → Voice Memos → Audio Quality → Lossless.** The default
   is 32 kbps AAC, which deletes everything above 10 kHz — finding F17. On a
   compressed recording a high-frequency fault is simply absent.
2. **Run the AGC distance test** (`docs/PHONE_RECORDING.md`). 40 seconds. If
   automatic gain control is on, level features describe your phone, not your
   fridge.
3. **Start the server and get your API key** — the phone deployment guide (not in this public copy) Step 0.
   Or skip the phone entirely and AirDrop the file to the laptop; for a first
   experiment that is simpler and there is no shame in it.

---

## Part A — the healthy baseline (~60 minutes)

**Position matters more than anything else here.** The detector compares the
machine to itself, so if the phone moves, the machine appears to change. Prop
it against the fridge side or on top, **touching the cabinet**, and do not
touch it again until the recording ends. Tape it if you have to.

- Kitchen door shut. No dishwasher, no kettle, no radio, nobody cooking.
- Record **at least 60 minutes**. The learn period alone is 24 minutes
  (48 × 30 s), and you want the compressor to switch at least once — most
  fridges cycle every 30–45 minutes.
- Note the time whenever you hear the compressor start or stop. That is your
  ground truth for the regime question, and nothing else will give it to you.

Then:

```bash
python tools/check_phone_audio.py fridge.m4a          # is the recording usable?
python tools/ingest.py fridge.m4a --mic-only --out-dir data/real
python tools/phone_monitor.py data/real/fridge.wav --learn-windows 48
```

### Reading it

| Result | What it means |
|---|---|
| `windows_above_threshold_pct` **0 %** | The best outcome. A real machine, an hour, no false alarms. |
| **1–10 %** | Plausible and worth investigating — check whether the flagged windows line up with the compressor times you wrote down. |
| **>20 %** | A real problem, and the most valuable thing you could find. It means the synthetic false-alarm numbers do not transfer to real machines. |
| `baseline_k_regimes` **2** | The compressor cycle became two regimes. This is the design working on a real duty cycle for the first time. |
| `baseline_k_regimes` **1** with flagged windows at compressor starts | Regime clustering did not separate the states, and the persistence gate is carrying the whole load. Record it. |

**Whatever happens, write the numbers into `RESULTS.md`** — including a boring
result. "Zero false alarms on a real fridge over an hour" is the first
real-machine evidence this project has ever had.

---

## Part B — manufacture a fault (~15 minutes)

This is how you get a "bad" recording without a broken fridge.

Tape a small piece of stiff card so it rests lightly against a vibrating panel
— it will buzz while the compressor runs. That is not a bearing fault, but it
**is** a genuine, physical, sustained change in what the machine sounds like,
and that is precisely what the detector claims to catch.

Do not stop the recording, do not move the phone. Let it run 20 more minutes
with the card in place.

Then analyse the *whole* file: the learn period comes from the clean part, the
card section gets scored.

- **Flagged during the card section, not before** → the detector responds to a
  real acoustic change on a real machine. That is a genuine result and the
  first end-to-end demonstration outside simulation.
- **Not flagged** → either the change was too quiet, or it fell outside the
  demodulation band. Check `band Hz` in the per-window table before concluding
  anything.

⚠ **Be honest about what Part B is.** A rattle is broadband and easy. A real
early bearing fault is a faint impulse train buried under the machine's own
noise. Passing Part B does **not** mean the product works; failing it means
something is badly wrong. It is a smoke test, not proof.

---

## What this still will not have proved

No bearing fault, no CWRU or MIMII data, no accelerometer, no Pi, no
long-duration false-alarm rate over a week. `docs/REAL_DATA_SOURCES.md` and
the hardware plan cover those.

But it converts this project from "verified on synthetic data" to "has heard a
real machine", and that is the single sentence missing from every funding
application and customer conversation you have.
