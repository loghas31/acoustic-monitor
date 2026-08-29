# Firmware — what runs on the Pi

Companion to the system overview (not in this public copy) §6. Directory: `firmware/`.

---

## Files

| File | Role |
|---|---|
| `main.py` | the loop: capture → features → score → gate → store → publish → alert |
| `capture.py` | three interchangeable signal sources behind one interface |
| `features.py` | 37-dim feature vector — see [DOC_PIPELINE.md](DOC_PIPELINE.md) |
| `baseline.py` | learn period: regimes + CV thresholds + retrain |
| `inference.py` | `MahalanobisScorer` + `AlertGate` (+ optional cloud AE) |
| `state.py` | SQLite: readings, anomalies, feedback, meta |
| `mqtt_client.py` | TLS uplink, offline queueing, command downlink |
| `config.yaml` | the only file a deployment edits |
| `acoustic-monitor.service` | systemd unit with a 350 MB memory cap |
| `bench/` | hardware bring-up — see [DOC_BENCH.md](DOC_BENCH.md) |

## The loop

```python
for audio, accel in source.windows():          # 30 s windows
    feats = extract_features(audio, fs_a, accel, fs_v)
    op    = operating_point(feats["vector"], feats["fr_hz"])
    score = scorer.score(feats["vector"], op)
    db.record_window(score, feats)
    uplink.publish_telemetry({...})
    if gate.feed(score["anomalous"]):
        uplink.publish_anomaly(event)
        local_webhook(cfg["local_alert"]["webhook_url"], event)
```

Everything else is error handling, and the error handling is the point: the
loop must never die. MQTT failures queue. Webhook failures log and continue.
systemd restarts on crash with `RestartSec=10`, and at most one window is lost
because state lives in SQLite.

## Three signal sources, one interface

`capture.py` exposes `windows()` yielding `(audio, accel)` from:

- **`SimulatedSource`** — calls `ml/simulate.py` live. Takes a `schedule(i)`
  callback so tests and demos can script regime changes, transients and
  growing faults.
- **`FileSource`** — replays recorded `.wav` + `.csv`. This is how real
  recordings enter the pipeline in week 2.
- **`HardwareSource`** — INMP441 over I2S/ALSA + IIS3DWB over SPI.

**Nothing hardware-related is imported at module load**, so `import capture`
works on a laptop with no spidev, no sounddevice and no ALSA. Every hardware
failure becomes a `HardwareUnavailable` carrying a *remedy sentence*, which the
bench tools print as friendly text. A student never sees a raw traceback for a
missing sensor.

**Degraded mode:** `HardwareSource(require_accel=False)` (the default) runs
mic-only if the accelerometer is missing or mis-wired. The microphone carries
the resonance band on its own, so you lose the cross-check, not the physics.
The execution plan explicitly allows shipping the sprint this way.

⚠ **`HardwareSource` and `IIS3DWB` have never touched a bench.** Register
constants are transcribed from the datasheet and anything unconfirmed is
marked `UNVERIFIED`. Run `firmware/bench/selftest.py` before believing any
field data.

## Local state (`state.py`)

| Table | Contents |
|---|---|
| `readings` | one row per window incl. the **37-dim feature vector** |
| `anomalies` | alert episodes |
| `feedback` | windows the customer marked normal, awaiting retrain |
| `meta` | key/value |

Storing the feature vector is what makes the feedback loop real — retraining
needs the actual vectors, not just the fact that something happened. At
**482 B/row measured** (T4.2; the ~400 B estimate above was ~20% low) and
7-day retention that is a few tens of MB, and old **`readings`** rows are
pruned on every insert (SD-card wear is a genuine field failure mode — T4.2
found the write volume itself is not a real risk against any published SD
endurance rating, see `docs/DOC_SOAK_DB_GROWTH.md`). **Only `readings` is
pruned this way — `anomalies` currently has no retention policy at all**
(T4.2, confirmed by both an 18-simulated-day audit and a fast regression
test) and grows for the life of the device. Slow at any realistic alert
rate, but unbounded and, until this correction, undocumented; recorded as a
"Not done" gap in `docs/DOC_STATUS.md` rather than fixed under the
frozen-file exception, since it is a slow gap rather than the kind of
active-harm bug that exception is for.

## Configuration

`config.yaml` holds device identity, sample rates, window length, learn length,
`persist_minutes`, MQTT settings, the LAN webhook, and storage paths.

**Note what is absent: bearing geometry.** Asking a facilities manager for a
ball pitch diameter would break the zero-knowledge install that makes the
product sellable.

## Running it

```bash
# learn a baseline (simulated two-regime machine)
python firmware/baseline.py --simulate --windows 48 --out firmware/baseline.npz

# ten simulated minutes: regime switches + a transient + a growing fault
python firmware/main.py --simulate --no-mqtt --fast --minutes 10 \
    --persist-minutes 2 --fault-at-minute 6 --transient-at-minute 2
```

Expected: `done: 1 alert(s) raised` — one for the persistent fault, none for
the transient, none for the regime switches.

## Resource budget (Pi Zero 2W, 512 MB)

| Component | Estimate |
|---|---|
| Python + numpy/scipy resident | ~60 MB |
| scikit-learn (learn/retrain only) | ~35 MB |
| 30 s buffers | ~8 MB |
| Feature extraction peak | ~15 MB |
| Scoring | negligible |
| **Steady state** | **~90 MB** |

The systemd unit caps at 350 MB: if anything leaks past that, die and restart
clean rather than dragging the OS into swap.
