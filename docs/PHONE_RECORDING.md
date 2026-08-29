# Phone-recording quickstart

Backlog T7.2. This is the only real-world signal available with zero
purchases: record a fridge compressor, an extractor fan, a washing machine —
anything with a motor and a bearing — on your phone, and get a real answer
about whether the detector's pipeline runs on real air instead of
`ml/simulate.py`'s signals. It cannot answer the actual product question
(does the detector separate a healthy machine from a faulty one — that needs
`H3`, a seeded fault on the bench motor) but it answers a cheaper and still
useful one: does this code, running on something that was never a synthetic
fixture, behave the way the design says it should?

## The two tools, and why there are two

**`tools/phone_monitor.py`** runs the actual product pipeline —
`firmware.baseline.fit_baseline` then `firmware.inference.MahalanobisScorer`,
the same functions the firmware calls — on ONE recording. It learns a
baseline from the first part of the recording and scores the rest against it.
This is the tool that answers T7.2's three questions and the one to run first.

**`ml/realdata/analyse_recording.py`** answers a different, narrower
question: given a HEALTHY and a FAULTY recording of the same machine and a
known (or guessed) bearing geometry, does the envelope spectrum show a line at
the predicted BPFO? That needs two recordings and a seeded fault, which a
phone quickstart on an undamaged appliance does not have. Use it later, at
`H3`, once there is an actual before/after pair.

Do not score a phone recording against the deployed `firmware/baseline.npz`
directly (e.g. by calling `MahalanobisScorer` on it yourself). That baseline
was learned with a real three-axis accelerometer; `tools/ingest.py --mic-only`
correctly leaves the accelerometer channel absent, and `firmware/capture.py`'s
`FileSource` (extended for T7.2, see below) fills it with zeros — which is
exactly the "dead channel" shape `firmware/inference.py` was hardened against
in T4.3. Measured this run: a synthetic mic-only healthy window scored
**95,580** against a threshold of **8.07** — over 10,000x — against the
deployed audio+accel baseline, and an equally "faulty" one scored the same,
because the score was dominated by the missing accelerometer channel, not by
the machine. That is not a bug; it is T4.3's guard doing exactly its job. It
means the deployed baseline is simply the wrong file to score a phone
recording against. `phone_monitor.py` sidesteps this correctly by training a
fresh, mic-only baseline from the recording itself, so what it learns from and
what it scores are always the same kind of feature vector.

## ⚠ First: automatic gain control, which could invalidate the whole recording

**Added 2026-08-21. This page did not mention AGC at all, and it is the most
likely way a phone recording silently lies to you.**

Phones are built to record voices, not instruments. Their capture chain
typically applies **automatic gain control** and noise suppression: when the
sound gets louder the phone turns the gain down, and vice versa. For speech
that is helpful. For this project it is potentially fatal, for a specific
reason:

`audio_logrms` is not just one of the 37 features — it is **one of the three
dimensions of `baseline.operating_point`**, the space the regime clustering
runs in. If AGC is riding the gain, then:

- absolute level no longer means "how loud the machine is", it means "what the
  phone decided to do", so **regimes may cluster on AGC behaviour rather than
  machine state**; and
- worse, **a machine getting louder — which is the fault you are looking for —
  is exactly what AGC compensates away.**

This is the same shape as finding F10 (`channel_stats` measuring gravity
instead of vibration): a feature that looks fine, is computed correctly, and is
measuring the wrong thing. The simulator cannot show it, because simulated
audio has no AGC.

[Likely, not certain] iOS applies processing on the voice-recording path, and
it varies by app, iOS version and settings. **Do not take my word for it —
measure it**, in two minutes:

> **The distance test — now automated.** Record ~20 s of a steady noise source
> (a running tap, a fan) held at ~10 cm. Without stopping, move to ~40 cm and
> hold for another ~20 s. Then:
>
> ```bash
> python tools/check_phone_audio.py my_test.m4a --distance-test
> ```
>
> Sound pressure falls roughly with $1/r$, so quadrupling the distance should
> cost about **12 dB**. The tool measures the drop and gives one of three
> verdicts — AGC on, ambiguous, or AGC off — and in the same pass checks sample
> rate, clipping and DC offset, which are the other three ways a phone
> recording is quietly useless.
>
> It is verified against signals where the answer is known by construction:
> `python tools/check_phone_audio.py --self-test` (clean → 12.0 dB and passes;
> the same signal through a block normaliser → −0.0 dB and is flagged), plus
> `tests/test_check_phone_audio.py`, 7 tests. A tool that claims to detect AGC
> without ever having been shown AGC would be an opinion.
>
> **Do the check before the long recording, not after.** It takes 40 seconds
> and decides whether the next hour is worth spending.

If AGC is on, you have three options, in order of preference:

1. **Find a recorder app that exposes raw/measurement mode** and disables
   processing. Search for "no AGC" or "raw audio" in the description.
2. **Run mic-only and accept that level features are untrustworthy** — the
   envelope and band-ratio features are *ratios* and survive a slowly-varying
   gain far better than absolute RMS does. Say so when reporting any result.
3. **Use it for spectral shape only**, not for anomaly scoring.

Record the answer in `RESULTS.md` either way. "AGC was on/off on this phone"
is a real finding and it determines how much anything else in this file is
worth.

## Proof the path works, end to end — run it yourself in two minutes

Executed 2026-08-23. Until this date the phone path had **never been run at
realistic length**, because generating a 24-minute learn period took about
eight minutes and usually ran out of memory. Both causes are fixed (see
the commit log (not in this public copy), pink-noise vectorisation and `--fs`).

```bash
# 28 min of realistic phone-like audio: mains hum + BPFO impacts + PINK floor.
# --fs 16000 because that is the rate the detector runs at; 44.1 kHz needs
# ~2.5 MB per second and a 4 GB box will OOM on a full learn period.
python ml/realdata/synth_phone_recording.py --out-dir /tmp/ph --duration-s 1700 --fs 16000 --seed 11

python tools/ingest.py /tmp/ph/phone_healthy.wav --mic-only --out-dir /tmp/real
python tools/phone_monitor.py /tmp/real/phone_healthy.wav --learn-windows 48
```

**Healthy recording — measured:** 48 learn windows, k=1 regime, threshold
7.903. Eight scored windows at **0.54–0.84× threshold, 0 % flagged.** It does
not cry wolf.

Then the case that matters — learn on healthy, let the machine degrade
(splice the healthy file's first 1440 s onto the faulty file's last 260 s):

**Degrading recording — measured:** the same 48 healthy learn windows, then
**100 % of scored windows flagged at 10.8–11.3× threshold.**

The detail worth more than the pass/fail: on the faulty windows the
**protrugram band selector fired on 100 % of them and locked onto
1402–1966 Hz** — the 1600 Hz resonance this generator deliberately places
*outside* the 3–6 kHz default band — while on healthy windows it fell back to
the default. `DOC_STATUS.md` had recorded that the selector had never been
tested on realistic pink noise with the resonance outside the default band.
Now it has been, and it found it.

⚠ **Still synthetic.** The signals come from `ml/realdata/synth_phone_recording.py`,
which is a realistic-noise proxy, not a phone and not a machine. This proves
the *pipeline* end to end. It says nothing about your fridge, and it cannot —
that needs your phone and an hour.

## Recording

Use the phone's voice memo app, default quality (44.1 or 48 kHz is fine —
`tools/ingest.py` resamples). Hold or prop the phone 5–30 cm from the machine
housing, as still as possible; handling noise and wind on the mic look exactly
like the broadband impacts the detector is built to find. Record for at
**least 30 minutes**, ideally 60+: `phone_monitor.py`'s learn period needs 48
windows of 30 s (24 minutes) as an absolute floor — `docs/DOC_STATUS.md`
measured that fewer than that gives a 55–59 % held-out false-alarm rate for
this 37-dimensional feature vector, which is not a measurement of anything,
it is noise. If you can, record through at least one natural cycle (a fridge
compressor switching on and off, a washing machine changing programme stage)
so the regime-clustering logic has something real to be tested against.
**Measure the rpm if you can** (a phone strobe app, or count revolutions by
eye over 30 s) — you will want it for `analyse_recording.py` later, at `H3`;
`phone_monitor.py` itself does not need it.

## Running it

```bash
python tools/ingest.py my_recording.m4a --mic-only --out-dir data/real \
    --stem healthy_fridge --note "kitchen fridge compressor, phone on top of casing"

python tools/phone_monitor.py data/real/healthy_fridge.wav --learn-windows 48
```

`ingest.py` prints a verification block — resample ratio, clipping, DC offset,
and the demodulation-band bandwidth check — before writing the canonical wav;
read it, it catches the failure modes (wrong sample rate, an 8 kHz-limited
recording, a clipped phone mic) that would otherwise silently produce a
confident wrong answer downstream. `phone_monitor.py` then prints one row per
scored 30 s window and, at the end, the three honest answers this task asks
for by name.

## What "good" looks like — proven this run, on synthetic data

`python tools/phone_monitor.py --self-test` runs the same code against a
from-scratch synthetic signal (pink noise + 50 Hz mains hum, written
independently of `ml/simulate.py` so a pass is not circular) with no periodic
fault in it. Executed this run:

```
windows         : 64 total (48 learn + 16 scored)
regimes learned : k=1, thresholds=[8.16]

Q1  demod band selector fired (found a periodic band) on 0.0% of scored
    windows (fell back to the 3-6 kHz default the rest of the time)
Q2  shaft-speed estimate was reliable on 0.0% of scored windows
    (expected: 0% in mic-only mode by design)
Q3  0.0% of scored windows were above their regime's threshold
    (GOOD — behaves like a healthy machine, if this recording was one)

PASS: tool runs end to end; a healthy synthetic signal with no periodic
fault stays below threshold; mic-only correctly reports the speed estimate
as unreliable.
```

This is what a genuinely healthy recording should look like: Q1 near 0 %
(nothing periodic to find), Q2 exactly 0 % (mic-only never confirms speed —
see below, this is by design, not a bug to chase), Q3 near 0 % (stays below
its own learned threshold). It tells you the CODE works. It says nothing
about your fridge, and the report says so in those words.

## The three questions, answered honestly, with numbers

**Q1 — did the demodulation band selector fire, or fall back to the 3–6 kHz
default?** This is the one to distrust the most, and this run quantified why.
`features.select_demodulation_band` requires an envelope-spectrum crest of at
least 10 before it trusts a band over the default; below that it returns
`DEFAULT_BAND` on purpose, so a healthy machine's windows don't band-hop for
no reason. `docs/DOC_STATUS.md` already found this floor is never reached on
a *pink*-noise surrogate in the accelerometer domain (crest 5.2–7.8). This run
extended that finding to the microphone/phone domain and, unlike the earlier
finding, put a real fault behind a real resonance **outside** the default
band (`ml/realdata/synth_phone_recording.py`, 1600 Hz on purpose) to see what
it costs. Measured: at a moderate fault severity the crest on this generator's
pink floor sits at **5–8.5**, still below the floor — the selector falls back
to 3–6 kHz and misses the real resonance entirely, and Gate 2 fails outright
(1.7x/1.0x — forcing the true band by hand recovers a real, if still
sub-threshold, line: 3.0x absolute, 1.6x contrast). At a stronger severity the
crest crosses **13–20** and the selector finds the true band unaided; Gate 2
then passes cleanly (12–18x absolute, 6–11x contrast). **The practical
consequence for your recording: if `phone_monitor.py` reports Q1 near 0 % on
a machine you suspect is already faulty, do not read that as "no fault" — it
may mean the fault is real but too quiet for the automatic band selector yet,
and `ml/realdata/analyse_recording.py --band <your tap-test frequency>`
(forced, not auto-selected) is the next thing to try, not a shrug.**

**Q2 — was the shaft-speed estimate reliable?** No, and it is not supposed to
be, in mic-only mode. `features.estimate_fr`'s own docstring is explicit: a
single live channel (the microphone, with no accelerometer to cross-check
against) returns `reliable=False` unconditionally — "one unconfirmed estimate
is a working assumption, not a measurement." This run's synthetic proxy (a
recording with no shaft-rate tone in the audio at all, which is plausible —
plenty of real machines don't put one there) measured just how wrong that
unconfirmed number can be: the harmonic-product-spectrum estimate locked onto
16.6 Hz against a true 24.17 Hz, 31 % off. **Never trust the auto speed
estimate from a phone-only recording for anything that needs the number — if
you later want to run `analyse_recording.py`'s geometry-based Gate-2 check,
measure the rpm yourself (strobe, or eyes-and-a-stopwatch) rather than reading
it off this tool's output.**

**Q3 — did a (presumed) healthy machine stay below its own threshold?** This
is the one `phone_monitor.py` gets right by construction: because it learns
its OWN baseline from the first part of your recording and scores the rest
against that (not against the deployed `firmware/baseline.npz`), a stationary
healthy recording should read close to 0 % above threshold — as the self-test
above shows. If your real recording reads well above 0 %, the two most likely
causes, in order of likelihood, are (a) the learn period itself was not
representative — a fridge that only reached steady-state hum halfway through
your recording will show its own "warm-up" as a false regime — and (b) a
genuinely unusual recording, which is either your machine or your recording
technique (handling noise, position change) and the tool cannot tell you
which. `phone_monitor.py` also flags learn-period contamination directly
(it did, correctly, in the self-test run above — see `!! learn-period
looks CONTAMINATED` — triggered by the self-test's own randomness, not a
bug); read that line before concluding anything about Q3.

## What this run did NOT prove

Nothing here is evidence about a real machine. Every number above comes from
one of two synthetic signal generators (`phone_monitor.py`'s own self-test
fixture, and `ml/realdata/synth_phone_recording.py`), neither of which has
ever been near a real bearing, a real room, or a real phone microphone. What
this run proves is narrower and still worth having: the code that Logan will
run on a real recording does not crash, produces the fields the three
questions need, and behaves the way its own design documents (`DOC_STATUS.md`,
`features.py`'s docstrings) say it should on noise that is at least more
adversarial than `ml/simulate.py`'s band-limited white floor. The first real
phone recording is still the first real test.

## Tests

`tests/test_phone_monitor.py` (8 tests) covers `firmware/capture.FileSource`'s
mic-only support and `tools/phone_monitor.py`'s `analyse()` end to end,
including the CLI `--self-test` path run for real. `tests/test_phone_recording.py`
(5 tests) covers the severity/crest crossover above and the "wrong baseline"
pitfall, using the independent `synth_phone_recording.py` generator so the
crossover is not just asserted, it is reproduced on demand.
