# RUN_IT — using this code on your own laptop, with no hardware

Nothing in this repository depends on Claude, on a Raspberry Pi, or on an
internet connection. It is Python. This page gets it running on your machine in
about ten minutes, and shows you the **exact output you should see**, so you can
tell "it worked" apart from "it printed something".

If your output differs from what is printed below, that is a finding, not a
mistake — write it down.

> Everything here is **synthetic**. It proves the software works. It proves
> nothing about a real bearing. See `DOC_STATUS.md`.

---

## 0. What you need

- Python 3.10 or newer (`python3 --version`)
- About 400 MB of disk for the packages
- No hardware, no Pi, no network after the install

```bash
git clone https://github.com/loghas31/acoustic-monitor.git acoustic-monitor
cd acoustic-monitor
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt
```

`(.venv)` should now appear in your prompt. **You must run the `source` line
again every time you open a new terminal**, or you will get
`ModuleNotFoundError: No module named 'numpy'`.

---

## 1. Make the synthetic machine signals

```bash
python ml/simulate.py --outdir data
```

```
fr = 50.0 Hz | BPFO = 152.6 Hz | BPFI = 247.4 Hz | resonance = 4500.0 Hz
wrote normal.wav  (10.0s @ 16000 Hz)
wrote bearing_outer.wav  (10.0s @ 16000 Hz)
wrote bearing_inner.wav  (10.0s @ 16000 Hz)
wrote imbalance.wav  (10.0s @ 16000 Hz)
wrote bearing_outer_accel.csv
```

Those are playable `.wav` files. **Listen to `normal.wav` and then
`bearing_outer.wav`.** You will probably not hear a difference, and that is the
entire reason this project needs signal processing rather than a microphone and
a human.

`BPFO = 152.6 Hz` is the ball-pass frequency of the outer race: with $N$ balls,
shaft speed $f_r$, ball diameter $d$, pitch diameter $D$ and contact angle
$\phi$,

$$\mathrm{BPFO} = \frac{N}{2} f_r \left(1 - \frac{d}{D}\cos\phi\right)$$

Remember 152.6 Hz. It shows up again in step 6 without anyone telling the
detector to look for it.

---

## 2. The physics evidence

```bash
python ml/verify_signals.py
```

```
wrote fig1_spectrograms.png
wrote fig2_envelope.png
BPFO peak-to-background ratio — raw: 2.2x | envelope: 56.7x
PASS: envelope analysis recovers the fault signature; raw spectrum does not.
```

**This is the single most important number in the project.** Open
`fig2_envelope.png`. In the raw spectrum the fault is 2.2× above the noise —
invisible in practice. After envelope demodulation it is **56.7×**. A bearing
impact does not make a tone at 152.6 Hz; it makes a *bang* that rings the
casing at ~4.5 kHz, 152.6 times a second. You have to demodulate to see the
repetition rate. If this step ever stops printing PASS, stop and find out why
before doing anything else.

---

## 3. Feature extraction

```bash
python firmware/features.py
```

```
  demod band: 3000-6000 Hz (crest 6.0) | fr = 50.00 Hz (reliable=True)
  env_crest = 0.709  audio_kurt = -1.510

[bearing fault sev 0.15]  extraction: 152 ms
  demod band: 3866-5420 Hz (crest 95.4) | fr = 50.00 Hz (reliable=True)
  env_crest = 1.984  audio_kurt = -1.441

OK: 37-dim feature vector, names aligned.
```

Note what changed by itself: on the healthy window the band selector shrugs and
takes a default 3–6 kHz band (crest 6.0); on the faulty one it locks onto
**3866–5420 Hz with crest 95.4** — it found the resonance. Nobody told it where
to look.

`extraction: 152 ms` is the compute budget. A Pi Zero's A53 core is roughly
8–10× slower, so ~1.2–1.5 s per 30 s window. That is the number that makes the
whole thing fit on a low-cost single-board computer.

---

## 4. Learn what "normal" sounds like

```bash
python firmware/baseline.py --simulate --windows 48 \
    --out firmware/baseline.npz --db /tmp/state.db
```

Ends with:

```
  "empirical_over_analytic": [0.983, 0.999],
  "learn_period_contaminated": [false, false],
  "saved": "firmware/baseline.npz"
```

This is the 24-minute learn period, compressed. It clustered the windows into
**2 operating regimes** — the simulated machine runs at 50 Hz and at 30 Hz, and
those are two different normals, not an anomaly — and fitted a threshold to
each.

`baseline.npz` is **gitignored**. It is per-machine, learned, and never shared;
a fresh clone has none, which is why this step comes before the next two.

---

## 5. Does it actually detect anything?

```bash
python ml/evaluate.py
```

```
  "auc": 1.0,
  "deployed_threshold_fpr": 0.0,
  "deployed_threshold_tpr": 1.0,
  "regime_switch_false_alarms": 0,
  "gating_alerts_transient": 0,
  "gating_alerts_persistent": 1
STAGE 3 GATE: PASS
```

**Do not be impressed by `auc: 1.0`.** It is a synthetic fault in a synthetic
machine with no factory around it. A real machine will be much harder, and if
this number were anything less than ~1.0 on data this clean, the design would
be broken. The two lines worth reading are `regime_switch_false_alarms: 0`
(changing speed does not cry wolf) and `gating_alerts_transient: 0` alongside
`gating_alerts_persistent: 1` — a one-off bang is ignored, a sustained fault is
not.

---

## 6. Watch it run

```bash
python firmware/main.py --simulate --no-mqtt --fast --minutes 10 \
    --persist-minutes 2
```

```
w11 regime=1 score=  5.66 thr= 9.38 idx= 54.7 green sev= +0.0 dB @ 235.8 Hz streak=0/4
w12 regime=0 score=415.05 thr= 8.07 idx= 87.1 amber sev=+14.8 dB @  91.7 Hz streak=1/4
w13 regime=0 score=469.04 thr= 8.07 idx= 87.6 amber sev=+16.2 dB @  91.6 Hz streak=2/4
w14 regime=0 score=522.18 thr= 8.07 idx= 88.1 amber sev=+17.5 dB @  91.5 Hz streak=3/4
w15 regime=0 score=577.29 thr= 8.07 idx= 88.5 red   sev=+18.6 dB @  91.5 Hz streak=4/4
ALERT #1  score=577.29 (thr 8.07) persisted 2 min
w16 regime=1 score=493.47 thr= 9.38 idx= 87.2 red   sev=+15.0 dB @ 152.6 Hz streak=5/4

done: 1 alert(s) raised
```

This is the whole product in twelve lines. Three things to notice:

1. **`streak=1/4 … 4/4`.** The fault is obvious at w12 but nothing happens until
   w15. That delay is the feature, not a bug — it is what stops a slammed door
   from generating a support call.
2. **`@ 91.5 Hz` then `@ 152.6 Hz`.** The detected repetition rate changes when
   the simulated shaft slows from 50 Hz to 30 Hz. Both are **3.05× the shaft
   speed**, and 152.6 Hz is the BPFO from step 1. The detector was never told
   the bearing geometry. It measured it.
3. **`ALERT #1`, once.** Not once per window for the next twenty minutes.

For a prettier version of the same run: `open tools/sim_dashboard.html`.

---

## 7. The tests

```bash
pytest tests/
```

```
430 passed in 58s
```

If a number in the README disagrees with reality,
`tests/test_docs_current.py` fails on purpose. That test exists because the
README once claimed 31 tests and 359 tests, in the same file, and both were
wrong.

---

## 8. Optional — the cloud half

Only needed if you want the fleet dashboard. Everything above works without it.

```bash
cd backend && docker-compose up --build     # API :8000, MQTT :1883
cd frontend && npm install && npm run dev   # dashboard, demo mode if no backend
```

⚠️ `docker-compose` has **never been run** in this project's development
environment — no Docker daemon was available. The config is written but
unproven; expect to fix something the first time. This is recorded in
`DOC_STATUS.md` and is not a surprise.

---

## Where this fits

| You are here | Next |
|---|---|
| Software runs on your laptop, synthetic data | Record a real motor with your phone → `tools/ingest.py` |
| Real audio through the same pipeline | Validate against the public CWRU bearing dataset |
| Both pass | Buy hardware (the parts list (not in this public copy)) and build (the build guide (not in this public copy)) |

The order matters. Every stage above is free to run and can invalidate the one
after it.
