# The physics — why bearings sound the way they do

Companion to the system overview (not in this public copy) §2. Code: `ml/simulate.py`,
`ml/verify_signals.py`, `ml/realdata/fault_frequencies.py`.

---

## Impacts, not tones

A healthy rotating machine produces a tone at the shaft rate f_r (50 Hz for a
3000 rpm motor), weak harmonics at 2f_r and 3f_r from residual imbalance and
misalignment, and broadband noise.

A bearing with a spall produces something structurally different. Each time a
rolling element passes over the defect it delivers an **impulse**. An impulse
is broadband — it excites every resonance it can reach. The bearing housing
answers by ringing at its natural frequencies (~1–20 kHz), decaying over a few
milliseconds.

So the fault produces a train of high-frequency bursts whose **repetition
rate** carries the diagnostic information. The rate is low (~100–300 Hz); the
carrier is high (~kHz). This is amplitude modulation, and it is why you cannot
find bearing faults by looking for a peak at the fault frequency.

## The four fault frequencies

For N rolling elements, ball diameter d, pitch diameter D, contact angle φ,
shaft frequency f_r, and γ = (d/D)·cos φ:

| Defect | Frequency |
|---|---|
| Outer race (BPFO) | (N/2)·f_r·(1 − γ) |
| Inner race (BPFI) | (N/2)·f_r·(1 + γ) |
| Ball (BSF) | (D/2d)·f_r·(1 − γ²) |
| Cage (FTF) | (f_r/2)·(1 − γ) |

Useful identity, and a free correctness check on any implementation:

**BPFO + BPFI = N·f_r** — the γ terms cancel exactly.

`ml/realdata/fault_frequencies.py` prints this check on every run, and
`tests/` asserts it.

## Outer vs inner race: the sideband tell

- An **outer-race** defect is stationary relative to the load zone. Every
  impact is equally loud. The envelope spectrum shows BPFO and its harmonics,
  unmodulated.
- An **inner-race** defect rotates with the shaft, passing in and out of the
  load zone once per revolution. The impacts are therefore amplitude-modulated
  at f_r, which puts **sidebands at BPFI ± f_r** in the envelope spectrum.

The original project brief had this backwards (it put sidebands on the outer
race). `ml/simulate.py` models it correctly — outer race unmodulated, inner
race load-zone-modulated.

## Slip: why real fault trains are not periodic

Rolling elements slip by roughly 1–2 %, so the impact interval wobbles. The
simulator models this with 1.5 % timing jitter. This matters: a perfectly
periodic train would make detection artificially easy, and it is precisely why
naive comb-filter or exact-frequency-matching detectors disappoint in the
field.

## Envelope analysis — the core technique

```
raw signal ──► band-pass around the resonance ──► |analytic signal|
           ──► remove DC ──► FFT ──► envelope spectrum
```

The band-pass isolates the carrier. The magnitude of the analytic signal (or
rectify-and-smooth, which is what the firmware uses because it is far cheaper)
recovers the modulating envelope. Its FFT reveals the repetition rate.

**Measured on our synthetic early-stage fault (severity 0.15):**

| Spectrum | Peak-to-background at BPFO |
|---|---|
| Raw | 2.2× — indistinguishable from noise |
| Envelope | **56.7×** — unmistakable |

Reproduce: `python ml/verify_signals.py` (writes `fig1_spectrograms.png`,
`fig2_envelope.png`).

## Choosing the band: why not spectral kurtosis

The textbook method for choosing the demodulation band is the **kurtogram** —
find the band with the highest spectral kurtosis, since impulsive signals are
strongly super-Gaussian.

**We measured it failing on our own signals.** At BPFO ≈ 153 Hz with a
resonance decay of τ = Q/(πf₀) ≈ 2 ms, consecutive rings overlap into a
quasi-continuous amplitude-modulated carrier. A modulated sine is *sub*-Gaussian
— negative kurtosis. Spectral kurtosis read ≈ 0 even at severity 0.5 and
selected no band at all.

The fix is a **protrugram**: choose the band whose *envelope spectrum* is
peakiest (max/median). That measures the quantity actually of interest —
periodicity of the burst rhythm — and works at any impact rate.
Implementation: `features.select_demodulation_band`.

This is a real, checkable result and would make a respectable section of a
final-year report.

## Shaft speed without a tachometer

The features need f_r, but customers will not install a tacho. Two attempts:

- **Cepstrum** (the spec'd method): take the spectrum of the log spectrum, and
  a harmonic series produces a peak at quefrency 1/f_r. We measured this
  failing — with only 2–3 weak harmonics the cepstral comb is too sparse to
  rise above noise. The detected peaks were noise.
- **Harmonic product spectrum** (what we use): score each candidate f by the
  summed log-power at f, 2f, 3f. A true fundamental *with* harmonics beats a
  lone loud peak — which matters, because 50 Hz mains hum is exactly a lone
  loud peak.

We then require the audio and accelerometer estimates to agree within 5 %. If
they disagree, the reading is flagged `fr_reliable = False` and reported as
"speed unknown" rather than guessed. Mains hum is acoustic-electrical and
largely absent from the accelerometer, so the cross-check specifically defeats
the most likely failure mode.

## Further reading

The standard references for this material are Randall & Antoni's work on
rolling-element bearing diagnostics and the review literature on envelope
analysis; the Case Western Reserve University bearing data centre is the
classic public dataset for validating against real seeded faults.
