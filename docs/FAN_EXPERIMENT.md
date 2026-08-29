# FAN_EXPERIMENT.md — a real machine, a real fault, one afternoon

The strongest result you can produce without buying anything or waiting for
access to a fridge. About 45 minutes of your time.

**What it establishes:** that the detector finds a genuine mechanical fault on
a real machine, at the frequency the geometry predicts, and stops finding it
when the fault is removed. That is a physics result, not a software demo.

---

## What you need

- A desk fan or table fan. Any speed, any size. Two speed settings is a bonus.
- A stiff card — a playing card, a business card, a folded index card.
- Tape.
- Your phone.

---

## Step 0 — Check the tool works (2 minutes, do this first)

**`tools/fan_experiment.py` has never been run.** The sandbox it was written in
ran out of disk before it could be executed once. Everything else in this repo
was verified end to end; this was not.

```bash
cd "/path/to/acoustic-monitor"
python tools/fan_experiment.py --self-check
```

It builds three synthetic recordings of known and unequal length and confirms
the tool reaches the right conclusion. **If it does not print PASS, stop and
tell me** — otherwise you will do the experiment properly and get a wrong
answer from the analysis, which is the worst of both.

---

## Step 1 — Make the prediction BEFORE you record (5 minutes)

This is the step that turns a demo into an experiment. A number predicted
afterwards is not a prediction.

**Count the blades.** Look at the fan. Write it down.

**Measure the shaft speed.** Film the blades on your phone in **slow-mo
(240 fps)**, then step through and count how many frames one full rotation
takes. Mark one blade with a dot of tape to make it followable.

$$\text{rpm} = \frac{240 \times 60}{\text{frames per revolution}}$$

So 11 frames per revolution → $240 \times 60 / 11 \approx 1310$ rpm.

**Compute the blade-pass frequency** — the rate at which blades strike the card:

$$f_{\text{blade}} = \frac{\text{blades} \times \text{rpm}}{60}$$

Three blades at 1310 rpm → **65.5 Hz**. That is your prediction. Write it in
your notes now, before any recording exists.

*(Don't trust the number on the fan's box. Measure it.)*

---

## Step 2 — Set up

1. Fan on a table, running at a **fixed speed**. Don't use oscillation.
2. Phone flat on the table, **10–20 cm from the fan body**, screen up. Not in
   the airflow — wind noise on the microphone will swamp everything. Off to the
   side, near the motor housing.
3. **Mark the phone's position with tape.** It must go back in exactly the same
   place for all three recordings. Distance changes level, and level changes
   the numbers.
4. **Settings → Apps → Voice Memos → Audio Quality → Lossless.** The default is
   compressed AAC, which deletes content above ~10 kHz and can silently remove
   the evidence.

---

## Step 3 — Record three times, 5 minutes each

Three separate Voice Memos. Aim for 5:00 each — the tool trims them to equal
length, but starting close means it discards less.

| # | Condition | What to do |
|---|---|---|
| 1 | **before** | Fan running normally. Don't touch anything. |
| 2 | **during** | Stop the fan. Tape the card so blades just clip it. Restart. Record. |
| 3 | **after** | Stop the fan. Remove the card. Restart. Record. |

**The card:** it should be struck by each blade with a light, audible tick —
not a heavy grinding. If the fan slows or strains, it's too far in. You want a
clean repeating impact, which is what a bearing defect produces.

**Keep a log with times.** Every event: "14:02 fan stopped", "14:04 card
fitted", "14:31 someone came in". When a window flags, the log is the only way
to distinguish "the machine did something" from "the room did something".

**Recording 3 is the one people skip. Don't.** Without it, a rise in score
could be the room warming, traffic outside, the fan ageing. It's what makes the
result a controlled comparison instead of an anecdote.

---

## Step 4 — Transfer and run

AirDrop all three to the Mac (they land in `~/Downloads`), then:

```bash
cd "/path/to/acoustic-monitor"
python tools/fan_experiment.py \
    ~/Downloads/before.m4a ~/Downloads/during.m4a ~/Downloads/after.m4a \
    --blades 3 --rpm 1310
```

Use your own blade count and measured rpm. Easiest way to get the paths right:
type the command up to `fan_experiment.py `, then drag each file from Finder
into the Terminal window in order.

---

## Step 5 — What the result means

You get a table and three verdicts.

**The pattern that means it worked:**

| condition | score |
|---|---|
| before | low |
| **during** | **clearly higher** |
| after | back to roughly `before` |

plus the reported peak frequency landing near your predicted blade-pass rate.

**If the peak is 2× or 3× your prediction**, that is not a failure — harmonics
are expected in an impact train, and the tool says so. Re-check your blade
count; a 3-blade fan reporting 131 Hz against a 65.5 Hz prediction is reporting
the second harmonic, which is still the right machine.

**If the score rises but the frequency is wrong**, be suspicious. Something
changed, but not necessarily what you think — check your log for events at that
time.

**If nothing separates**, report that. A null result you understand is worth
more than a positive one you cannot explain, and this project's whole record
(see `DOC_SELF_REVIEW.md`) is built on writing down the measurements that
disappointed.

---

## Step 6 — Two upgrades worth 20 extra minutes

**Two fan speeds.** Repeat recording 2 at the fan's other speed setting. A real
mechanical fault frequency **scales with rpm**; background noise does not. If
your detected peak moves proportionally when the speed changes, you have shown
the thing you found is mechanically real rather than an artefact. This is a
stronger control than most undergraduate projects have and it costs one extra
recording.

**Exercise the main detector too.** The three-recording design above uses the
baseline-free screen. To also test the self-baselined path — the actual product
— do one continuous 40-minute recording: 24 minutes healthy (it learns), 8 more
minutes healthy (checks false alarms), then 8 minutes with the card (checks
detection). Note the exact moment you fit the card; the detector should fire in
the windows after it and not before.

```bash
python tools/fridge_scan.py ~/Downloads/continuous.m4a
```

---

## Step 7 — Write it down

Put the numbers in `RESULTS.md`: what you predicted, what you measured, what it
does and does not establish. That document is what makes the repo worth reading
— see the publication notes (not in this public copy).

State the limits plainly. This shows the detector finds an *impulsive* fault on
*one* machine in *one* room. It does not establish sensitivity to early faults,
performance across machine types, or a false-alarm rate over weeks. Saying so
precisely is what makes the positive claim credible.
