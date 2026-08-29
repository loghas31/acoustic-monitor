# The detector — regimes, Mahalanobis distance, thresholds

Companion to the system overview (not in this public copy) §3–4.
Code: `firmware/baseline.py` (learn), `firmware/inference.py` (score).

---

## The problem

We have 37 numbers per 30 s window and **no labelled faults** — and neither
will any customer on day one. So this must be unsupervised: learn what normal
looks like, flag departures.

## Step 1 — regimes

A machine that idles, then runs loaded, then runs fast occupies several
distinct islands in feature space. Fitting one Gaussian over all of them puts
the mean in empty space between islands, and then **every legitimate mode
change looks anomalous**. That is the fastest way to lose a customer.

So the learn period is clustered, k ∈ 1…4 chosen by silhouette score, and one
Gaussian is fitted per cluster.

**Where we cluster matters.** We cluster in a 3-dimensional *physical
operating-point* space:

```
(shaft frequency, audio log-RMS, accelerometer log-RMS)
```

not in the full 37 dimensions. We measured why: with ~50 learn windows in 37
standardised dimensions, every model-selection statistic we tried was blind —
silhouette 0.118, resampling stability ARI 0.36 versus 0.34 for the
single-regime null. The true two-regime structure was recoverable by k-means
but *undecidable* by any criterion for choosing k. In three physical
dimensions the same split is unambiguous.

The scaling is also physical rather than statistical: 5 % of shaft speed,
0.1 decades of level. Sample-standard-deviation scaling inflates whichever
dimension happens to be constant into unit-scale noise, which buried the real
split (silhouette collapsed from ~0.99 to 0.37).

**Lesson worth keeping:** domain knowledge beat a statistic we could not
estimate at our sample size.

**A silhouette alone is not enough (T1.9, 2026-08-18).** Silhouette measures
the *shape* of a partition, and its value on data with no regimes at all
depends on how many directions the cloud varies in. Measured on single-cluster
noise, 1500 trials: median best silhouette **0.584** in one effective
dimension against **0.283** in three, with a maximum of **0.702** — above the
0.5 threshold this file used to describe. A mic-only node on a fixed-speed
machine has exactly one live dimension (speed constant, accelerometer level
pinned at the dead-channel sentinel), and so did **any** build whose audio and
accelerometer levels move together. Measured consequence, before the fix: 48
healthy windows of one unchanging simulated machine were split into regimes of
30 and 18, and across 100 bootstrap learn periods k > 1 was chosen **every
time**, costing **6.3×** the held-out false alarms (0.1358 vs 0.0217).

Two criteria now have to be met, because each one misses a case the other
catches:

* **Absolute** — `MIN_REGIME_SEPARATION`: the regime centroids must be at
  least 1.0 apart in the standardised operating space, which by construction
  means 5 % of shaft speed or 0.1 decade of level. The failure above splits a
  cloud 0.0002 decades wide, so it fails this.
* **Relative** — the silhouette floor rises to **0.75** when the cloud has one
  effective dimension (singular values, not column counts, so two collinear
  channels count as one). A machine that genuinely wanders 0.1 decade inside
  one operating mode is the mirror-image failure: wide enough to pass the
  absolute test, not clean enough to pass this one.

The honest cost: in one dimension a genuine split needs a centroid gap of
roughly 6× the within-regime scatter to be recognised, so nearly-overlapping
operating points get merged. Merging is the safe direction — one slightly wide
Gaussian, rather than two Gaussians each fitted to half the learn period.
Reproduce with `python tools/regime_miconly_cost.py`.

## Step 2 — Mahalanobis distance

Within a regime, the anomaly score is

d(x) = √((x − μ)ᵀ Σ⁻¹ (x − μ))

This is the statistically correct "how many standard deviations away is this,
accounting for correlations". Features that naturally move together (RMS and
band energies do) are not double-counted.

Σ is estimated with **Ledoit–Wolf shrinkage**. With ~30 samples and 37
dimensions the sample covariance is ill-conditioned and its inverse explodes;
Ledoit–Wolf shrinks toward a scaled identity with an analytically optimal
weight and no hyperparameter to tune on-device.

**Regime assignment uses the operating point, not the full vector.** A
developing fault distorts envelope and kurtosis features but barely moves
speed or overall level — so a faulty window still gets compared against the
regime it genuinely belongs to, instead of escaping to whichever regime makes
it look most normal.

## Step 3 — the threshold, and a bug worth understanding

The threshold is the 99.5th percentile of learn-period distances. The obvious
implementation computes those distances on the same windows the Gaussian was
fitted to — and **it is badly wrong**.

In-sample Mahalanobis distances are biased low: the fitted Gaussian hugs the
very points that defined it, and with n ≈ d the optimism is severe. We
measured the consequence: an in-sample threshold flagged **79 % of held-out
healthy windows**.

The fix is 5-fold cross-validation — score each fold against a Gaussian fitted
without it. That took the held-out false-positive rate from 79 % to **7 % per
window**, which after the 4-window persistence gate is ≈ 2 × 10⁻⁵.

This is the single most important bug found during development, and it was
only found by *running* the thing on held-out data. It is a good cautionary
tale for any report: the code was "correct" and the maths was standard, and it
was still 79 % wrong.

### Step 3b — and the estimator on top of those distances was degenerate too

Cross-validation made the *distances* honest. It left the *estimator* broken.
A learn period is 24–96 windows, and `np.percentile(d, 99.5)` of 24 numbers
is arithmetically the largest of them (measured p99.5/max = **0.989** at
n = 24). So every deployed unit's alert threshold was set by whichever single
learn window happened to be worst — and it scaled without limit with how bad
that window was: **1.47× at 12 σ, 2.16× at 25 σ, 3.30× at 50 σ**.

The textbook fix does not work here. Squared Mahalanobis distance is χ² with
*p* degrees of freedom only when the data has *p* independent directions.
Ours does not: the feature vector's effective rank is **13.7 of 37**, so d²
concentrates near the effective dimensionality, not the nominal one —
measured mean(d²) = **28.5** against p = 37, with χ²₃₇ rejected by a KS test
at p = 3e-90. Deployed on 24-window learn periods, `chi2.ppf(0.995, 37)`
produced an **11.0 %** held-out false-alarm rate against 3.8 % for the
estimator it was meant to replace.

What ships instead keeps the χ² *shape* but fits both its scale and its
degrees of freedom from the **median and 75th percentile** of the distances.
Nothing in the tail is read, so no amount of tail contamination can move the
fit — the estimator's output is bit-identical at 12, 25, 50 and 100 σ. Over
480 healthy windows the fitted model is not rejected (KS p = 0.62) and
reproduces the pooled 99.5th percentile to 0.7 % (6.866 vs 6.914).

The deployed threshold is **min(empirical, fitted)**. `min` is deliberate: it
guarantees the change can never make a unit less sensitive than the code it
replaces. On a clean learn period the empirical value is the lower one and
nothing changes at all — the repository's own baseline retrained to the same
thresholds, 8.348 / 9.882.

**The disagreement is the diagnostic.** empirical/fitted > 1.25 means the
sample tail is heavier than the fitted body can explain, which is what a
contaminated learn period looks like from the inside. False-flag rate on
clean learn periods 1.0 % (n = 24) / 0.5 % (n = 48); catches 99.2 % of
12 σ contamination and 100 % of 25 σ.

End to end on the real pipeline, with one of 48 learn windows carrying a loud
external event (a vehicle passing outside, audio × 6):

| | clean learn period | with the lorry |
|---|---|---|
| old threshold (regime 1) | 8.391 | **21.504** |
| new deployed threshold | 8.391 | **8.426** |
| detection, severity-0.02 bearing fault, old | 1.000 | **0.375** |
| detection, same fault, new | 1.000 | **0.833** |
| held-out healthy false-alarm rate | 0.017 | 0.000 |
| contamination flagged | no | **yes** (ratio 2.55) |

### Step 3c — the same contamination also invented a regime

Running that case end to end exposed a second, worse bug. The lorry window's
operating point sat far from every other, so k-means gave it **its own
cluster of size 1**. LedoitWolf was then fitted to a single sample, whose
distance to its own mean is exactly zero, so that regime's threshold came out
at **0.0** — after which every window ever assigned to it alarms
unconditionally. Measured before the fix: k=2, counts [47, 1], thresholds
[7.4658, **0.0**].

`choose_k` now rejects any k that produces a cluster smaller than
`MIN_REGIME_WINDOWS = 8`. A cluster that small is an outlier, not an
operating mode. Genuine two-regime learn periods are unaffected.

## Step 4 — the feedback loop

When a customer presses **"this was normal"**, the cloud sends the device a
`mark_normal` command with the episode's time range. The device copies those
windows' stored feature vectors into a `feedback` table, and
`baseline.py --retrain` folds them into the next fit. The unit therefore
adapts to that specific site instead of being permanently wrong about it.

## What runs on the device

Scoring is one 37×37 matrix–vector product per regime — microseconds. **No ML
runtime is needed on the Pi at all** for v1. scikit-learn is used only during
the learn/retrain step (KMeans, LedoitWolf).

The optional v1.5 convolutional autoencoder (`ml/model.py`) is trained
**server-side** and pushed down as a quantised `.tflite` for inference only:
TFLite cannot train, and full TensorFlow does not fit in 512 MB. That split is
architectural, not a preference.

## Evidence

```bash
python ml/evaluate.py
```

On a 70/30 healthy/faulty synthetic set across two regimes:

| Metric | Result |
|---|---|
| ROC AUC | 1.000 |
| FPR / TPR at deployed threshold | 0.07 / 1.00 |
| Alerts caused by regime switches | 0 |
| Alerts from a single-window transient | 0 |
| Alerts from a persistent fault | 1 |

**Read AUC 1.0 sceptically.** It means the simulation is easy. Real machines
will be worse, and the honest version of this table only exists after week 2.

## Tests

`tests/test_baseline.py` (9 tests): k selection on two-regime and
single-regime data, fit/score round-trip, operating-point-driven regime
assignment, the CV-threshold-wider-than-in-sample property, and four
alert-gate behaviours.
