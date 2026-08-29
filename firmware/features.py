"""
features.py — signal-processing front end (v2, geometry-free).

Per 30 s window we compute one fixed-length feature VECTOR for the on-device
Mahalanobis anomaly model, plus a Mel spectrogram for telemetry/the dashboard
heatmap (and the optional v1.5 cloud autoencoder).

What changed from v1 and why (matches launch_prompt_v2):

* No named BPFO/BPFI indicators. They need bearing geometry the customer
  cannot supply (zero-knowledge install is a hard constraint). The envelope
  spectrum is still computed — fault periodicity raises its band energies and
  crest factor, which the anomaly model sees WITHOUT knowing which bearing
  line it is. "Indicative, not diagnostic."
* Audio at 16 kHz, Mel 64 bins to 8 kHz: bearing impacts ring housing
  resonances in the 1–20 kHz region; the old 4 kHz ceiling was provably blind
  (ml/verify_signals.py, fig1).
* Envelope band chosen by ENVELOPE-SPECTRUM PEAKINESS per band (a
  "protrugram"), not spectral kurtosis. We measured SK failing here, and the
  reason is physics: at BPFO ~150 Hz with ring decay tau = Q/(pi*f0) ~ 2 ms,
  consecutive bursts overlap into a near-continuous AM carrier — which is
  SUB-Gaussian (negative kurtosis). Kurtosis-based selection (the classic
  kurtogram) only works for sparse impacts; envelope periodicity strength
  works at any impact rate. Default fallback 3–6 kHz when nothing is periodic
  (healthy machine) so healthy feature vectors don't band-hop.
* fr estimated by HARMONIC PRODUCT SPECTRUM on both channels + agreement
  check, with an explicit "unreliable" flag. The spec asked for cepstrum, but
  a machine with only 2–3 weak harmonics gives a cepstral comb too sparse to
  detect (measured: cepstral peaks were noise). HPS is the few-harmonics
  analogue of the same idea. Naive single-peak picking locks onto 50 Hz mains;
  requiring audio/accel agreement defends against that — mains hum is
  acoustic-electrical and nearly absent in the accelerometer.

* Energy fractions enter the vector as ISOMETRIC LOG-RATIO (ILR) coordinates,
  not as raw or log fractions. Any set of D fractions that sums to 1 carries
  only D-1 free numbers, so handing all D to a Gaussian model gives it a
  covariance that is singular by construction. See the long note above `ilr()`.
  This is why the vector is 37-dim and not 40-dim (backlog T1.5 / self-review
  F1).

Dependency policy unchanged: numpy + scipy only (no librosa/numba on a Pi).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfilt, stft, welch
from scipy.stats import kurtosis, skew

# ----------------------------------------------------------------------------
# STFT + Mel (telemetry + v1.5 autoencoder input)
# ----------------------------------------------------------------------------

N_FFT, HOP, N_MELS = 1024, 512, 64
FMIN, FMAX = 20.0, 8000.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """(n_mels, n_fft//2+1) triangular filters on the Mel scale. Mel allocates
    resolution densely at low frequency (shaft harmonics) and coarsely at high
    frequency (resonance bursts) — the right summary for machine sound."""
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, fs / 2)), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_pts) / fs).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        c = min(max(c, l + 1), n_bins - 1)
        r = min(max(r, c + 1), n_bins)
        fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def stft_mag(x: np.ndarray, fs: float):
    """|STFT| (n_bins, n_frames) — computed once, reused by Mel AND spectral
    kurtosis so we never pay for two transforms."""
    f, t, Z = stft(x, fs, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                   boundary=None, padded=False)
    return f, np.abs(Z)


def log_mel(mag: np.ndarray, fs: float) -> np.ndarray:
    fb = mel_filterbank(N_MELS, N_FFT, fs, FMIN, FMAX)
    return np.log1p(fb @ mag)


# ----------------------------------------------------------------------------
# Demodulation band selection (protrugram) -> envelope spectrum
# ----------------------------------------------------------------------------

DEFAULT_BAND = (3000.0, 6000.0)
ENV_DECIM = 8          # envelope decimation: 16 kHz -> 2 kHz (content < 500 Hz)
ENV_LP_HZ = 600.0

# T1.13 / SELF-REVIEW F19: the global default, now overridable per machine.
# `select_demodulation_band` and `extract_features` both default to this so
# every EXISTING caller (most of tests/, ml/, tools/) is bit-identical unless
# it explicitly opts into a calibrated floor. `baseline.calibrate_crest_floor`
# computes the per-machine value from the learn period and is the only
# intended source of anything else. See its docstring for the measurement.
DEFAULT_CREST_FLOOR = 10.0


def envelope(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[np.ndarray, float]:
    """Demodulate one band: bandpass -> rectify -> lowpass -> decimate.

    Rectify-and-smooth instead of the Hilbert analytic magnitude: for a
    narrowband carrier they are equivalent for our purpose, and this version
    needs zero full-length FFTs (Hilbert needs two of 480k points each —
    measured 6x slower; the Pi's 500 ms budget cares).
    Returns (envelope, envelope_sample_rate)."""
    lo = max(band[0], 1.0)
    hi = min(band[1], fs / 2 * 0.98)
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    e = np.abs(sosfilt(sos, x))
    sos_lp = butter(4, ENV_LP_HZ, btype="low", fs=fs, output="sos")
    return sosfilt(sos_lp, e)[::ENV_DECIM], fs / ENV_DECIM


def envelope_spectrum(x: np.ndarray, fs: float,
                      band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    env, fs_env = envelope(x, fs, band)
    env = env - env.mean()
    mag = np.abs(rfft(env * np.hanning(len(env))))
    return rfftfreq(len(env), 1.0 / fs_env), mag


def _env_crest(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Peakiness of the envelope spectrum in 5–500 Hz: max / median.
    ANY strong periodic impacting spikes this, regardless of which fault
    frequency it is — the geometry-free core of v1 detection."""
    freqs, mag = envelope_spectrum(x, fs, band)
    sel = (freqs >= 5.0) & (freqs <= 500.0)
    return float(np.max(mag[sel]) / (np.median(mag[sel]) + 1e-12))


def select_demodulation_band(x: np.ndarray, fs: float, n_bands: int = 6,
                             lo: float = 1000.0,
                             crest_floor: float = DEFAULT_CREST_FLOOR) -> tuple[tuple[float, float], float]:
    """Protrugram: pick the band whose ENVELOPE SPECTRUM is most peaky.

    Why not spectral kurtosis (the textbook kurtogram)? Measured failure on
    our own signals: at BPFO ~150 Hz the resonance bursts (decay ~2 ms)
    overlap into a quasi-continuous AM carrier whose kurtosis is NEGATIVE
    (sub-Gaussian, like a sine). SK selects nothing. Envelope-spectrum
    peakiness measures the actual quantity of interest — periodicity of the
    burst rhythm — at any impact rate.

    Healthy machines have no periodic envelope anywhere: every band's crest is
    low and which-band-wins is noise. Below crest_floor we return DEFAULT_BAND
    so healthy feature vectors don't band-hop (band-hopping would inflate the
    baseline covariance for no information gain).
    Returns ((lo, hi), best_crest)."""
    edges = np.geomspace(lo, fs / 2 * 0.95, n_bands + 1)
    best_band, best_crest = DEFAULT_BAND, -np.inf
    for i in range(n_bands):
        band = (float(edges[i]), float(edges[i + 1]))
        c = _env_crest(x, fs, band)
        if c > best_crest:
            best_crest, best_band = c, band
    if best_crest < crest_floor:
        return DEFAULT_BAND, best_crest
    return best_band, best_crest


ENV_BANDS = np.array([5.0, 15.0, 35.0, 75.0, 150.0, 300.0, 500.0])  # 6 log-ish bands


def envelope_fractions(x: np.ndarray, fs: float,
                       band: tuple[float, float]) -> tuple[np.ndarray, float, float]:
    """(6 envelope-band energy fractions, total energy, spectrum crest).

    The fractions are the composition: they sum to 1 by construction (measured
    0.999849–0.999997 across a learn period — the residual is the single
    500 Hz bin that the half-open band edges exclude). Total energy and crest
    are ordinary scalars and are NOT part of it."""
    freqs, mag = envelope_spectrum(x, fs, band)
    p = mag ** 2
    sel = (freqs >= ENV_BANDS[0]) & (freqs <= ENV_BANDS[-1])
    total = float(np.sum(p[sel])) + 1e-12
    fracs = []
    for i in range(len(ENV_BANDS) - 1):
        m = (freqs >= ENV_BANDS[i]) & (freqs < ENV_BANDS[i + 1])
        fracs.append(float(np.sum(p[m])) / total)
    crest = float(np.max(mag[sel]) / (np.median(mag[sel]) + 1e-12))
    return np.array(fracs), total, crest


def envelope_features(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """7 geometry-free numbers summarising 'is there periodic impacting':

    [0]    log total envelope-fluctuation energy (5–500 Hz) — the SIZE of the
           composition, which the fractions deliberately throw away, so it is
           carried separately.
    [1..5] ILR coordinates of the 6 envelope-band energy fractions — a coarse
           'where is the rhythm', in 5 numbers rather than 6 because 6
           fractions summing to 1 only ever contained 5.
    [6]    envelope-spectrum crest (max/median, 5–500 Hz): ANY strong
           periodicity spikes this, whatever its frequency. This is the
           feature that replaces named BPFO matching in v1.

    Was 8 numbers before T1.5. The 6 raw fractions were the one EXACTLY
    singular block in the vector: their measured null direction was the uniform
    vector to |cos| = 1.0000, i.e. textbook compositional degeneracy.
    """
    fracs, total, crest = envelope_fractions(x, fs, band)
    return np.concatenate([[np.log10(total)], ilr(fracs), [np.log10(crest + 1.0)]])


# ----------------------------------------------------------------------------
# Compositional data: energy fractions are NOT ordinary numbers
# ----------------------------------------------------------------------------
#
# THE PROBLEM (measured, backlog T1.5, self-review F1)
# ----------------------------------------------------------------------------
# Several of our features are *energy fractions*: "what share of the total sits
# in this band". D such fractions always sum to 1, so they live on a
# (D-1)-dimensional simplex, not in D-dimensional space. Feed all D to a
# Gaussian and its covariance matrix is singular BY CONSTRUCTION — there is a
# direction in feature space along which the data provably cannot vary.
#
# Nothing crashes: Ledoit-Wolf shrinkage regularises the inverse and the scores
# look sensible (self-review F4 confirmed the detector is measuring signal, not
# numerical noise). The cost is subtler — we pay for D dimensions and receive
# D-1, at n/d ~ 1.6 where every dimension is expensive.
#
# Measured on a 14-window healthy learn matrix, as smallest/largest singular
# value of the STANDARDISED block (i.e. of the matrix the Mahalanobis model
# actually sees):
#
#     audio band fractions (D=8)   6.8e-3     <- degenerate
#     accel band fractions (D=8)   2.9e-3     <- degenerate
#     envelope fractions   (D=6)   6.5e-3     <- degenerate
#     audio statistics     (D=4)   2.8e-1     <- healthy, for comparison
#     random full-rank reference   2.1e-1 median, never below 5e-2 in 200 draws
#
# Note the log10 in `band_energy_ratios` does NOT rescue this. It makes the
# constraint non-linear in principle, but over a stationary machine the
# fractions barely move, so the constraint LINEARISES almost perfectly: the
# measured null direction of the audio band block matches the mean-fraction
# weight vector to |cos| = 1.0000. A dependency you cannot see is still a
# dependency you are paying for.
#
# THE FIX
# ----------------------------------------------------------------------------
# Compositional data analysis (Aitchison, 1986) has a right answer. Work with
# log-RATIOS between parts — the only functions of a composition that ignore
# the arbitrary total — and choose an ORTHONORMAL basis for them. That is the
# isometric log-ratio (ILR) transform: D parts -> D-1 real coordinates, no
# constraint, full rank, and Euclidean distance between coordinate vectors
# equals Aitchison distance between compositions. The last property is what
# makes it the correct choice HERE specifically: our detector is a Mahalanobis
# distance, so the geometry of the coordinate space is not a detail.
#
# WHY NOT CLR, WHICH THE BACKLOG ORIGINALLY PROPOSED
# ----------------------------------------------------------------------------
# The centred log-ratio is the more famous transform and it does NOT fix this.
# CLR returns D coordinates that sum to zero by construction, so the block is
# still exactly rank D-1 in D columns — the singularity is preserved, merely
# relocated from the simplex constraint to a sum-to-zero constraint. `clr()` is
# kept below because ILR is defined in terms of it and because it is the right
# thing for *plots* (its coordinates map one-to-one onto the original bands and
# so stay interpretable). It is deliberately not what enters the vector.
# `tests/test_compositional.py::test_clr_would_not_have_fixed_it` pins this.
#
# WHY NOT JUST DROP ONE BAND
# ----------------------------------------------------------------------------
# Also full rank, also cheap, and it was the other option in the backlog. But
# it privileges an arbitrary reference band, and the resulting coordinates are
# correlated with each other by construction, which is precisely the sort of
# hidden structure this exercise exists to remove. ILR costs one `cumsum`.

ZERO_FLOOR = 1e-9   # multiplicative zero replacement, see clr()


def clr(comp: np.ndarray, floor: float = ZERO_FLOOR) -> np.ndarray:
    """Centred log-ratio: D parts -> D coordinates, log(x_i / geometric mean).

    Zero replacement: a band can legitimately contain exactly zero energy (a
    dead channel gives zero everywhere), and log(0) is not a number. We floor
    the fractions at `floor` BEFORE renormalising — the standard multiplicative
    replacement. `floor` matches the 1e-9 already used inside
    `band_energy_ratios`, so the two descriptions of the same spectrum agree
    about what counts as "no energy here".

    Returns coordinates that sum to zero — which is exactly why this is not
    what goes into the feature vector. See the note above.
    """
    c = np.maximum(np.asarray(comp, dtype=float), floor)
    c = c / c.sum()
    L = np.log(c)
    return L - L.mean()


def ilr(comp: np.ndarray, floor: float = ZERO_FLOOR) -> np.ndarray:
    """Isometric log-ratio: D parts -> D-1 UNCONSTRAINED coordinates.

    Uses the standard Helmert (sequential-binary-partition) basis:

        y_i = sqrt(i / (i+1)) * log( geomean(x_1..x_i) / x_{i+1} ),  i = 1..D-1

    Read y_i as "how much louder, in log terms, is everything up to band i than
    band i+1 is" — a contrast between a group of bands and the next one up.
    The sqrt weight makes the basis orthonormal, so no coordinate is louder
    than another merely because it summarises more bands.

    Properties that matter downstream, each pinned by a test:
      * D-1 outputs -> the redundant dimension is REMOVED, not shrunk away.
      * scale invariant -> doubling the microphone gain moves nothing.
      * invertible (`ilr_inverse`) -> no information is discarded.
      * isometric -> Euclidean distance here == Aitchison distance there, so
        the Mahalanobis distance built on top of it is meaningful.
      * all-zero input -> all-zero output, finite. A disconnected sensor must
        give a stable sentinel, never a NaN: one NaN in this vector poisons
        every anomaly score the device will ever produce.
    """
    c = np.maximum(np.asarray(comp, dtype=float), floor)
    c = c / c.sum()
    L = np.log(c)
    D = len(L)
    # cumulative sum of logs -> the log of the running geometric mean, for free
    csum = np.cumsum(L)
    i = np.arange(1, D)                       # 1 .. D-1
    return np.sqrt(i / (i + 1.0)) * (csum[i - 1] / i - L[i])


def ilr_inverse(y: np.ndarray) -> np.ndarray:
    """Recover the composition from its ILR coordinates (tests + debugging).

    Rebuilds the CLR coordinates by summing the Helmert basis vectors, then
    exponentiates and renormalises. Only used off-device."""
    y = np.asarray(y, dtype=float)
    D = len(y) + 1
    z = np.zeros(D)
    for k in range(D - 1):
        i = k + 1                              # basis vector index, 1-based
        w = np.sqrt(i / (i + 1.0))
        z[:i] += y[k] * w / i
        z[i] -= y[k] * w
    c = np.exp(z - z.max())
    return c / c.sum()


# ----------------------------------------------------------------------------
# Band energy ratios + per-channel statistics
# ----------------------------------------------------------------------------

def band_fractions(x: np.ndarray, fs: float, n_bands: int = 8,
                   lo: float = 10.0) -> np.ndarray:
    """Raw energy FRACTION in each of n_bands log-spaced bands up to ~Nyquist.

    Split out from `band_energy_ratios` so that the composition itself — the
    thing that is compositional — exists as an object, and both the human-
    readable log-fraction view and the ILR coordinates that enter the model are
    computed from one definition rather than two that can drift apart."""
    freqs = rfftfreq(len(x), 1.0 / fs)
    p = np.abs(rfft(x - np.mean(x))) ** 2
    edges = np.geomspace(lo, fs / 2 * 0.95, n_bands + 1)
    total = float(np.sum(p[(freqs >= edges[0]) & (freqs <= edges[-1])])) + 1e-12
    out = []
    for i in range(n_bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out.append(float(np.sum(p[m])) / total)
    return np.array(out)


def band_energy_ilr(x: np.ndarray, fs: float, n_bands: int = 8,
                    lo: float = 10.0) -> np.ndarray:
    """The n_bands-1 numbers that actually enter the feature vector."""
    return ilr(band_fractions(x, fs, n_bands, lo))


def band_energy_ratios(x: np.ndarray, fs: float, n_bands: int = 8,
                       lo: float = 10.0) -> np.ndarray:
    """log10 energy fraction in n_bands log-spaced bands up to ~Nyquist.
    Captures broad spectral shape: a machine whose energy migrates between
    bands sounds different, whatever the cause.

    KEPT FOR REPORTING, NOT FOR THE MODEL. "band 4 holds 3 % of the energy" is
    the sentence you want in a report or on a plot axis, and these are those
    numbers. They are a composition, though, so what enters the Mahalanobis
    vector is `band_energy_ilr` — see the compositional-data note above."""
    return np.log10(band_fractions(x, fs, n_bands, lo) + 1e-9)


def dc_level(x: np.ndarray) -> float:
    """Mean of the channel — the DC component. Reported as a DIAGNOSTIC, never
    used as an anomaly feature.

    For the accelerometer it is the projection of gravity onto that axis, so it
    measures mounting orientation: a step change means the sensor has moved or
    fallen off. For the microphone it should be ~0; a large value means the
    ADC or the I2S alignment is wrong.

    Deliberately NOT in the feature vector. On a fixed installation it is a
    constant, and F7/T1.9 showed that a constant column in a clustering input
    is worse than useless — it invited invented regimes."""
    return float(np.mean(x))


def channel_stats(x: np.ndarray) -> np.ndarray:
    """RMS, kurtosis, crest, skew — all computed on the DC-REMOVED signal.

    Kurtosis/crest are the classic impact detectors: ~0 / ~4 for Gaussian
    operation, rising fast with impulsive faults. RMS tracks ISO-10816-style
    overall severity.

    ------------------------------------------------------------------------
    WHY THE MEAN IS REMOVED (self-review F10, 2026-08-19)
    ------------------------------------------------------------------------
    This function previously computed sqrt(mean(x**2)) on the RAW signal while
    every other function in this file removed the mean first. The simulator
    generates zero-mean signals, so the defect was invisible for the entire
    life of the project. On real hardware it breaks two things:

    1. MICROPHONE DC OFFSET. The SPH0645 is documented to carry one. Measured:
       an offset of just 10 % of signal RMS moved a healthy window from 0.76x
       to 2.62x threshold — permanent false alarm. At 100 % it was 173x. A DC
       offset is a constant and carries no information about the machine; if it
       moves the anomaly score, the detector is measuring the sensor.

    2. GRAVITY. A real accelerometer sits in a 1 g field. Machine vibration is
       order 0.01-0.1 g RMS, so raw RMS reports ~1.0 g regardless. Measured:
       quadrupling true vibration (0.05 -> 0.20 g) moved this feature by 0.008
       in log10; DC-free it moves 0.60. `accel_x_logrms` is one of the three
       dimensions of `baseline.operating_point`, so regime clustering was
       partly tracking the magnet's mounting ANGLE rather than vibration.

    Crest factor is affected the same way — a DC pedestal raises max|x| and RMS
    together and compresses the ratio (measured 4.72 -> 1.24 with 1 g present).

    Dead-channel guard: a disconnected/flat sensor axis must yield finite,
    stable numbers — NaNs here would silently poison the whole feature vector
    and with it every anomaly score after install day.
    """
    x = np.asarray(x, dtype=float)
    ac = x - np.mean(x)                       # <-- F10: the whole fix
    rms = float(np.sqrt(np.mean(ac ** 2)))
    if rms < 1e-9 or float(np.std(x)) < 1e-12:
        return np.array([-9.0, 0.0, 0.0, 0.0])
    return np.array([
        np.log10(rms),
        float(kurtosis(ac)),
        float(np.max(np.abs(ac)) / rms),
        float(skew(ac)),
    ])


# ----------------------------------------------------------------------------
# Running-frequency estimate (cepstrum + cross-check, never a silent guess)
# ----------------------------------------------------------------------------

def _hps_peak(x: np.ndarray, fs: float, lo: float, hi: float,
              n_harm: int = 3) -> float:
    """Harmonic product spectrum: score each candidate fundamental f by the
    summed log-power at f, 2f, 3f. A fundamental WITH harmonics beats a lone
    peak — so 50 Hz mains hum (no mechanical harmonics in the accel channel)
    doesn't win just by being loudest. The few-harmonics-robust cousin of the
    cepstrum (which we measured failing on a 3-harmonic machine: the cepstral
    comb is too sparse to stand above noise).

    Returns 0.0 for a channel carrying no usable signal. THIS MATTERS: a dead
    or disconnected channel is all zeros, every candidate then ties at
    log(1e-20), and `argmax` silently returns the FIRST candidate — i.e. the
    search lower bound. That is a plausible-looking number (10 Hz) produced by
    a channel that measured nothing, and it caused a real bug: mic-only builds
    discarded a perfectly good 50 Hz audio estimate in favour of a dead
    accelerometer's boundary artefact. Always distinguish "no signal" from
    "signal at the lowest frequency I was allowed to consider".
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 16 or float(np.std(x)) < 1e-12:
        return 0.0                                        # dead / silent channel
    nper = int(min(len(x), 8 * fs))                       # 0.125 Hz resolution
    f, p = welch(x - np.mean(x), fs, nperseg=nper)
    logp = np.log(p + 1e-20)
    cand = np.flatnonzero((f >= lo) & (f <= hi))
    if len(cand) == 0:
        return 0.0
    score = np.full(len(cand), -np.inf)
    for j, bi in enumerate(cand):
        s = 0.0
        for h in range(1, n_harm + 1):
            if bi * h < len(f):
                s += logp[bi * h]
        score[j] = s
    # A flat score surface means nothing stood out — report "unknown" rather
    # than the arbitrary bin argmax happens to land on.
    if not np.isfinite(score).any() or float(np.ptp(score)) < 1e-9:
        return 0.0
    return float(f[cand[np.argmax(score)]])


def estimate_fr(audio: np.ndarray, fs_audio: float,
                accel: np.ndarray, fs_accel: float,
                lo: float = 10.0, hi: float = 120.0) -> tuple[float, bool]:
    """Returns (fr_hz, reliable).

    HPS on each channel independently, then:
      * both live and agreeing within 5 %  -> accel value, reliable=True
        (accel is mechanically coupled and immune to acoustic mains hum)
      * both live but disagreeing          -> accel value, reliable=False
      * only ONE channel live              -> that channel's value,
                                              reliable=False
      * neither live                       -> 0.0, reliable=False

    The single-live-channel case is not a corner case: **mic-only is a
    supported build** (see `capture.HardwareSource(require_accel=False)`), and
    an earlier version of this function preferred the accelerometer
    unconditionally, so a mic-only node threw away a correct 50 Hz audio
    estimate in favour of a dead channel's 10 Hz boundary artefact. Speed feeds
    `baseline.operating_point`, so that error propagated straight into regime
    clustering. Tested by `test_fr_mic_only_uses_audio`.

    `reliable=False` on a single channel is deliberate: one unconfirmed
    estimate is a working assumption, not a measurement. Downstream shows
    "speed unknown" rather than a confident wrong number."""
    fr_audio = _hps_peak(audio, fs_audio, lo, hi)
    ax = accel[:, 0] if accel.ndim == 2 else accel
    fr_accel = _hps_peak(ax, fs_accel, lo, hi)

    audio_live, accel_live = fr_audio > 0, fr_accel > 0
    if audio_live and accel_live:
        reliable = abs(fr_audio - fr_accel) / fr_accel < 0.05
        return fr_accel, reliable
    if accel_live:
        return fr_accel, False
    if audio_live:
        return fr_audio, False
    return 0.0, False


# ----------------------------------------------------------------------------
# The one call the firmware loop makes per window
# ----------------------------------------------------------------------------

N_BANDS = 8          # spectral bands per channel -> N_BANDS-1 ILR coordinates
N_ENV_BANDS = len(ENV_BANDS) - 1     # 6 envelope bands -> 5 ILR coordinates

FEATURE_NAMES = (
    [f"audio_stat_{s}" for s in ("logrms", "kurt", "crest", "skew")]
    + [f"accel_{a}_{s}" for a in "xyz" for s in ("logrms", "kurt", "crest", "skew")]
    + [f"audio_band_ilr_{i}" for i in range(N_BANDS - 1)]
    + [f"accel_band_ilr_{i}" for i in range(N_BANDS - 1)]
    + ["env_log_total"] + [f"env_ilr_{i}" for i in range(N_ENV_BANDS - 1)] + ["env_crest"]
)  # 4 + 12 + 7 + 7 + 7 = 37 dims
#
# WAS 40 DIMS until T1.5. Three blocks of energy fractions each lost exactly one
# column, because each was a composition carrying one fewer free number than it
# had columns. Nothing was discarded: `ilr` is invertible. The 37-dim vector
# holds the same information in a representation that is not rank-deficient,
# which matters at n/d ~ 1.6 where the covariance is estimated from barely more
# windows than there are features.
#
# NOTE FOR ANYONE UPDATING A BASELINE: this changes the feature contract, so a
# baseline.npz trained before T1.5 is not loadable against this code. Retrain:
#   python firmware/baseline.py --simulate --windows 48
# `inference.MahalanobisScorer` checks the stored dimension and refuses a
# mismatch rather than scoring garbage.


def extract_features(audio: np.ndarray, fs_audio: float,
                     accel: np.ndarray, fs_accel: float,
                     crest_floor: float = DEFAULT_CREST_FLOOR) -> dict:
    """Everything downstream needs. accel: (n,) or (n, 3).

    crest_floor (T1.13 / F19): passed straight to `select_demodulation_band`.
    Defaults to the original global constant, so every caller that does not
    pass it explicitly is bit-identical to before this parameter existed.
    Callers that HAVE a calibrated per-machine floor (main.py's live loop,
    via `MahalanobisScorer.crest_floor`) should pass it here."""
    if accel.ndim == 1:
        accel = accel[:, None]
    # pad to 3 axes so the vector length never changes (simulation is 1-axis,
    # IIS3DWB hardware is 3-axis)
    while accel.shape[1] < 3:
        accel = np.column_stack([accel, accel[:, -1] * 0.0])

    band, band_crest = select_demodulation_band(audio, fs_audio, crest_floor=crest_floor)
    _, mag = stft_mag(audio, fs_audio)

    vec = np.concatenate([
        channel_stats(audio),
        np.concatenate([channel_stats(accel[:, i]) for i in range(3)]),
        band_energy_ilr(audio, fs_audio),
        band_energy_ilr(accel[:, 0], fs_accel),
        envelope_features(audio, fs_audio, band),
    ])
    fr_hz, fr_reliable = estimate_fr(audio, fs_audio, accel, fs_accel)

    return {
        "vector": vec,                          # (37,) Mahalanobis input
        "mel": log_mel(mag, fs_audio),          # (64, n_frames) heatmap / v1.5 AE
        "band": band,                           # demodulation band used
        "band_crest": band_crest,               # how periodic the best band was
        "fr_hz": fr_hz,
        "fr_reliable": fr_reliable,
        # DIAGNOSTIC ONLY — never an anomaly feature (F10). Accelerometer DC is
        # the gravity projection, so a step change means the sensor moved or
        # fell off; microphone DC should be ~0 and a large value means the I2S
        # alignment or ADC is wrong. Both are things a field engineer needs to
        # see and neither belongs in a clustering input (see F7).
        "dc": {
            "audio": dc_level(audio),
            "accel_x": dc_level(accel[:, 0]),
            "accel_y": dc_level(accel[:, 1]),
            "accel_z": dc_level(accel[:, 2]),
        },
    }


# ----------------------------------------------------------------------------
# Self-demonstration: `python firmware/features.py`
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
    from simulate import SimConfig, bearing_fault_signal, normal_signal

    cfg = SimConfig(duration_s=30.0)
    rng = np.random.default_rng(0)
    print(f"generating 30 s windows (audio {cfg.fs_audio} Hz, accel {cfg.fs_accel} Hz)...")
    cases = {
        "healthy": (normal_signal(cfg, cfg.fs_audio, rng),
                    normal_signal(cfg, cfg.fs_accel, rng)),
        "bearing fault sev 0.15": (
            bearing_fault_signal(cfg, cfg.fs_audio, rng, 0.15, "outer"),
            bearing_fault_signal(cfg, cfg.fs_accel, rng, 0.15, "outer")),
    }
    for name, (audio, accel) in cases.items():
        t0 = time.perf_counter()
        out = extract_features(audio, cfg.fs_audio, accel, cfg.fs_accel)
        dt = time.perf_counter() - t0
        v = out["vector"]
        print(f"\n[{name}]  extraction: {dt*1000:.0f} ms")
        print(f"  vector shape {v.shape}  mel shape {out['mel'].shape}")
        print(f"  demod band: {out['band'][0]:.0f}-{out['band'][1]:.0f} Hz "
              f"(crest {out['band_crest']:.1f}) | "
              f"fr = {out['fr_hz']:.2f} Hz (reliable={out['fr_reliable']})")
        print(f"  env_crest = {v[FEATURE_NAMES.index('env_crest')]:.3f}  "
              f"audio_kurt = {v[FEATURE_NAMES.index('audio_stat_kurt')]:.3f}")
    assert len(FEATURE_NAMES) == 37 == len(v), "feature vector length drifted"
    print("\nOK: 37-dim feature vector, names aligned.")
