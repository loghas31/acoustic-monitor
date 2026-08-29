# The feature pipeline — sound in, 37 numbers out

Companion to the system overview (not in this public copy) §4. Code: `firmware/features.py`.
Entry point: `extract_features(audio, fs_audio, accel, fs_accel)`.

---

## Design constraint

Everything here runs on a Raspberry Pi Zero 2W: quad-core Cortex-A53, **512 MB
RAM**. Dependencies are **numpy and scipy only**. No librosa (it pulls in
numba/llvmlite, which is painful on ARM and memory-hungry); the 20-line Mel
filterbank is ours. No TensorFlow anywhere on the device.

Budget: 30 s of audio must be processed in well under 30 s. Measured ~150 ms
per window on an x86 development machine, which is roughly 1.2–1.5 s on an
A53 — comfortably inside the 2 s gate.

---

## The 37 features

| Index | Group | Count | What it captures |
|---|---|---|---|
| 0–3 | Audio statistics | 4 | log RMS, kurtosis, crest factor, skew |
| 4–15 | Accelerometer statistics | 12 | same four, per axis (x radial, y radial 90° round, z axial) |
| 16–22 | Audio band energies | 7 | ILR coordinates of 8 log-spaced band energy fractions, 10 Hz–8 kHz |
| 23–29 | Accel band energies | 7 | same, on the primary axis |
| 30–36 | Envelope features | 7 | log total envelope energy, ILR of 6 band fractions (5), envelope crest |

Names are in `features.FEATURE_NAMES`; a test asserts the vector length and
name list never drift apart.

**Was 40 features until 2026-08-17 (backlog T1.5).** The three band-fraction
blocks are *compositions* — sets of energy fractions that sum to 1 — so each
carried one fewer free number than it had columns, and handing all of them to a
Gaussian gave it a covariance that was singular by construction. Each block now
enters as **isometric log-ratio (ILR) coordinates**: D parts → D−1 coordinates,
unconstrained, invertible (nothing is discarded), and orthonormal in the
Aitchison geometry, which is what makes a Mahalanobis distance on top of them
meaningful. The long rationale, including why the more famous CLR transform
would *not* have fixed it, is the comment block above `ilr()` in
`firmware/features.py`.

`band_energy_ratios()` still exists and still returns the eight plain
log-fractions — "band 4 holds 3 % of the energy" is the sentence you want in a
report. It is simply no longer what the model consumes.

**A baseline trained before T1.5 is not loadable against this code.**
`inference.MahalanobisScorer` compares the stored width against
`FEATURE_NAMES` and refuses a mismatch with an explicit retrain instruction;
before that check the failure was `ValueError: operands could not be broadcast
together with shapes (37,) (40,)` thrown from inside `score()`.

**Why these:**

- **Kurtosis and crest factor** are the classic impact detectors. Gaussian
  noise gives kurtosis ≈ 0 (Fisher) and crest ≈ 4. Impacts push both up hard.
- **RMS** tracks overall severity — this is what ISO 10816-style vibration
  standards use.
- **Band energy ratios** capture spectral *shape*: a machine whose energy
  migrates between bands sounds different regardless of cause. Being ratios
  they are also immune to the microphone's absolute gain, and the ILR transform
  preserves that exactly. **Caveat measured 2026-08-17:** on *simulated* signals
  these blocks carry far less information than their column count suggests —
  the simulator's band composition spans **1.03 of 8** dimensions. See
  DOC_SELF_REVIEW F9; treat any simulation result that leans on the band
  features as unproven until a real recording says otherwise.
- **Envelope crest** (max/median of the envelope spectrum, 5–500 Hz) is the
  geometry-free replacement for named BPFO matching. *Any* strong periodic
  impacting spikes it, whatever frequency it sits at. This single feature does
  most of the bearing-detection work.

## Which feature groups are actually exercised, and which are not

Kept current deliberately. A feature that cannot vary independently on the data
you tested it with has not been tested, however many columns it occupies.

| Features | Status on SIMULATED data | Measured |
|---|---|---|
| 0–3 audio statistics | exercised | effective rank **3.70 of 4** on the healthy 2-speed learn period; AUC **0.972–1.000** detecting either fault kind tested (T1.10) |
| 4–15 accel statistics (12) | **exercised since 2026-08-18 (T1.8)** | effective rank **9.32–9.46 of 12** (two independent measurements agree), inter-axis r +0.04 / −0.68 / +0.51. Until T1.8 these were 3.75 of 12 at r = 0.995–0.999, i.e. one axis counted three times. `accel_y_kurt` is still 0.99 predictable from the x block on healthy windows — one residual, recorded not hidden |
| 16–22 audio band ILR (7) | **low-rank, NOT uninformative (T1.10)** | near one-dimensional on healthy data (effective rank **2.31 of 7**, sv ratio **8.9e-4** — corroborates F9's raw-fraction measurement of 1.03 of 8 on a different representation), but its dominant direction detects a bearing fault at AUC **0.993** and an imbalance fault at AUC **0.965**. Low rank is not the same as no signal — see below |
| 23–29 accel band ILR (7) | **low-rank, NOT uninformative (T1.10)** | effective rank **1.53 of 7**, sv ratio **3.6e-3** (F9: 1.01 of 8), computed from axis 0 only so T1.8 changed nothing here. AUC **0.997** (bearing) / **0.907** (imbalance) |
| 30–36 envelope features (7) | exercised, but **fault-specific (T1.10)** | carries most of the bearing-fault detection: AUC **0.996**, envelope crest separates 56.7× vs 2.2× raw. But it is FULL RANK and near-USELESS for imbalance (effective rank **6.63 of 7**, AUC **0.447**, chance) — it measures impact *periodicity*, and imbalance has none. High rank does not imply the block is informative about a given fault, any more than low rank implies it isn't |

**Revised by T1.10, 2026-08-18** (`tools/feature_block_report.py`,
`tests/test_feature_blocks.py`, 14 tests): F9's rank measurement stands, but
"unproven" should not be read as "contributing nothing". Every block tested
detects at least one of the two fault kinds tried (bearing outer race,
imbalance) at AUC > 0.85, trained on healthy-only windows and scored
held-out — including both band-ILR blocks, despite each spanning barely one
effective dimension on healthy data. What F9's number means precisely is that
**most of the 14 band-energy columns are redundant with each other**, not that
they carry zero information; a single dominant direction inside a near-1D
block can still separate healthy from faulty cleanly. The one block that
*fails* to detect a fault it was tested against is the envelope block on
imbalance — the mirror image of the low-rank-but-informative story, and a
second confirmation (after T1.5) that effective rank is not an information
measure in either direction.

Still true and unmoved by this: it is one simulator, two synthetic fault
types, and every AUC above rests on `ml/simulate.py`'s assumed physics. The
thing that would settle it for real is one real recording — `band_fractions`
on a real motor in a real room should span materially more than 1.03 of 8, and
a real imbalance or misalignment fault may excite the envelope band in ways
this simulator does not model. `tools/accel_axis_report.py` recomputes the
accelerometer half of this table in one command and
`tools/feature_block_report.py` recomputes all five blocks including the
AUC-by-fault-kind table; run both on the first real triaxial capture.

## The five stages

**1. STFT and log-Mel spectrogram.** 1024-point Hann, 50 % overlap, 64 Mel
bins from 20 Hz to 8 kHz. Mel spacing allocates fine resolution to low
frequencies (shaft harmonics) and coarse to high (resonance bursts) — the
right trade for machine sound. The Mel spectrogram is used for the dashboard
heatmap and the optional cloud autoencoder; it is *not* the anomaly input.

**2. Demodulation band selection (protrugram).** Six log-spaced candidate
bands from 1 kHz upward; for each, compute the envelope spectrum and its
crest. Pick the peakiest. If nothing is periodic anywhere (crest < 10, i.e. a
healthy machine), fall back to a fixed default so healthy vectors do not
band-hop — hopping would inflate the baseline covariance for no information.

See [DOC_PHYSICS.md](DOC_PHYSICS.md) for why spectral kurtosis was rejected here.

**3. Envelope extraction.** Band-pass → rectify → low-pass at 600 Hz →
decimate ×8. Rectify-and-smooth rather than the Hilbert analytic magnitude:
for a narrowband carrier the two are equivalent for our purposes, and this
version needs zero full-length FFTs. Measured ~6× faster, which matters on
an A53.

**4. Statistics and band ratios.** Straightforward, with one important guard:
a dead or disconnected channel returns fixed finite values rather than NaN.
A single NaN would propagate through the covariance and silently destroy every
score thereafter.

**5. Shaft speed estimate.** Harmonic product spectrum on each channel plus a
5 % agreement check, returning `(fr_hz, fr_reliable)`. Never a silent guess.

## Output contract

```python
{
  "vector":       np.ndarray,   # (37,)  -> the anomaly model
  "mel":          np.ndarray,   # (64, n_frames) -> dashboard heatmap
  "band":         (lo, hi),     # which band was demodulated
  "band_crest":   float,        # how periodic that band was
  "fr_hz":        float,
  "fr_reliable":  bool,
}
```

## Try it

```bash
python firmware/features.py     # healthy vs severity-0.15 fault, with timings
```

Expected: the fault window selects a band centred on the simulated 4.5 kHz
resonance with a high crest, while the healthy window falls back to the
default band with a low crest.

## Tests

`tests/test_features.py` (12 tests) covers the Mel filterbank shape and
coverage, Gaussian reference values for the statistics, band-ratio
normalisation, f_r accuracy at two speeds, the unreliable-disagreement path,
band selection on fault and healthy, **BPFO recovery at severity 0.15**, and
the performance gate.
