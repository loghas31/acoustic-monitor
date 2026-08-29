"""
inference.py — v2 on-device scoring: Mahalanobis distance against the nearest
learned regime. Pure NumPy at runtime; one matrix-vector product per regime
(37x37 · 37) — microseconds, no ML runtime needed on the Pi at all for v1.

The optional v1.5 path (cloud-trained autoencoder pushed down as .tflite for
INFERENCE ONLY) is kept behind CloudAEScorer below. It is not used by default
and the device must never depend on it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# T3.7 startup fingerprint check — see BaselineMismatchError and the check in
# MahalanobisScorer.score() below. Tuned, not guessed: STARTUP_CHECK_WINDOWS
# matches MIN_REGIME_WINDOWS's use elsewhere in this codebase as "enough to
# not be one bad window"; the ratio bound is expressed as a MULTIPLE of the
# learn period's own p95 ratio (usually close to 1 by construction) so it
# scales with how tight or loose a given baseline's threshold already is,
# rather than a single magic number copied across baselines.
STARTUP_CHECK_WINDOWS = 8
STARTUP_CHECK_MIN_FRACTION_ANOMALOUS = 0.75
STARTUP_CHECK_RATIO_MULTIPLE = 5.0


class BaselineMismatchError(RuntimeError):
    """Raised by MahalanobisScorer.score() when the first STARTUP_CHECK_WINDOWS
    windows look implausible under the loaded baseline's own learn-period
    fingerprint (see fit_baseline's `startup_ratio_median`/`startup_ratio_p95`
    in baseline.py).

    WHY THIS EXISTS: T1.8 measured a firmware change (a different simulated
    accelerometer model) that left the feature vector's DIMENSION unchanged —
    so T1.5's dimension-mismatch check never fired — but silently invalidated
    the deployed baseline: 100% of fresh HEALTHY windows scored a median
    138.4x their threshold. Nothing distinguished that, at the time, from a
    real fault appearing the instant the unit booted; main.py's ordinary
    persistence gate would eventually have raised a normal-looking fault
    alert. That is the wrong diagnosis and the wrong fix (a customer who
    thinks their compressor is failing calls a mechanic; a customer whose
    firmware silently drifted from its baseline needs a retrain).

    Deliberately NOT a silent auto-retrain — a unit that quietly relearns
    during a real fault learns the fault as normal. The caller (main.py)
    must stop and ask a human to run --retrain (or a fresh learn period)."""


class MahalanobisScorer:
    """Scores one 37-dim feature vector per 30 s window.

    score(x, op):
      1. assign regime: nearest learned centroid in the standardised 3-dim
         OPERATING-POINT space (fr, audio RMS, accel RMS)
      2. standardise the full 37-dim vector with the learn-period mean/std
      3. Mahalanobis distance to that regime's Gaussian; anomalous if above
         that regime's own threshold

    Regime assignment uses the operating point, not the full vector, on
    purpose: a developing fault distorts envelope/kurtosis features but
    barely moves speed and overall level — so a faulty window is still
    compared against the regime it actually belongs to, instead of escaping
    to whichever regime makes it look most normal.

    'Nearest regime' is what makes legitimate operating-mode changes free:
    idle -> loaded moves the window between learned islands; only sound that
    matches NO learned regime scores high.
    """

    def __init__(self, baseline_path: Path):
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"{baseline_path} missing — run firmware/baseline.py first (learn period)")

        # T4.3 fault-injection audit. `baseline.npz` is written once by a
        # learn period and then read on every boot for months; in that time
        # it can be truncated by a power loss mid-`np.savez_compressed`,
        # corrupted by a failing SD card, or simply be the wrong file. Before
        # this the failure surfaced as whatever `np.load` or a bare
        # dict-index happened to raise — `zipfile.BadZipFile`, `EOFError`, a
        # `KeyError` naming one field — none of which tell a student at a
        # customer site what to do. Convert all of them to one message with
        # the retrain command, same policy as the dimension-mismatch check
        # below.
        RETRAIN = ("Retrain it:\n    python firmware/baseline.py --simulate "
                  "--windows 48\n(or re-run the learn period on the machine).")
        from baseline import load_baseline
        try:
            b = load_baseline(baseline_path)
        except Exception as e:                            # noqa: BLE001
            raise ValueError(
                f"{baseline_path} could not be read as a baseline file "
                f"({type(e).__name__}: {e}). It is corrupt, truncated, or was "
                f"not produced by firmware/baseline.py. {RETRAIN}") from e

        required = ("global_mean", "global_std", "op_mean", "op_scale",
                   "op_centroids", "means", "precisions", "thresholds", "k")
        missing = [k for k in required if k not in b]
        if missing:
            raise ValueError(
                f"{baseline_path} is missing field(s) {missing} — wrong "
                f"schema/version, or a write that was cut short. {RETRAIN}")

        try:
            self.g_mean = b["global_mean"]
            self.g_std = b["global_std"]
            self.op_mean = b["op_mean"]
            self.op_scale = b["op_scale"]
            self.op_centroids = b["op_centroids"]  # (k, 3)
            self.means = b["means"]                # (k, d)
            self.precisions = b["precisions"]      # (k, d, d)
            self.thresholds = b["thresholds"]      # (k,)
            self.k = int(b["k"])
        except Exception as e:                            # noqa: BLE001
            raise ValueError(
                f"{baseline_path} has unreadable field(s) ({type(e).__name__}: "
                f"{e}) — it is corrupt. {RETRAIN}") from e

        # NUMERIC SANITY CHECK. A baseline that LOADS but is not finite is
        # worse than one that fails to load: `score > threshold` with a NaN
        # anywhere is IEEE754-guaranteed False, so a corrupted precision
        # matrix or threshold makes the unit silently report "not anomalous"
        # forever — on exactly the device that most needs attention. A
        # non-positive std has the same shape of danger the other way:
        # z = (x - mean) / std divides by ~0 and manufactures a score that
        # LOOKS like a real extreme reading rather than a broken file. This
        # is the F2 failure shape again — a plausible number produced by
        # data that measured nothing — arriving through file corruption
        # instead of a dead sensor.
        numeric = {"global_mean": self.g_mean, "op_mean": self.op_mean,
                  "op_scale": self.op_scale, "op_centroids": self.op_centroids,
                  "means": self.means, "precisions": self.precisions,
                  "thresholds": self.thresholds}
        bad = [name for name, arr in numeric.items()
              if not np.all(np.isfinite(np.asarray(arr, dtype=float)))]
        if not np.all(np.isfinite(self.g_std)) or np.any(np.asarray(self.g_std) <= 0):
            bad.append("global_std")
        if bad:
            raise ValueError(
                f"{baseline_path} contains non-finite or non-positive values "
                f"in {sorted(bad)} — the file is corrupt (bad write, disk "
                f"error, or partial save). Scoring against it would silently "
                f"under- or over-report anomalies rather than fail. {RETRAIN}")

        # FEATURE-CONTRACT CHECK. A baseline stores the learn period's mean and
        # std, so its length IS the feature-vector length the model was trained
        # for. If the firmware's feature vector has changed since (T1.5 moved it
        # from 40 to 37 dims by removing three compositional redundancies), the
        # stored model is meaningless against the new vector.
        #
        # Without this check the failure surfaced as
        #     ValueError: operands could not be broadcast together with
        #                 shapes (37,) (40,)
        # thrown from inside score(), on a device, possibly hours after an
        # update, with nothing in the message naming the cause or the fix.
        # Failing safe is not sufficient; it has to fail legibly, because the
        # person reading this log is a student at a customer site.
        from features import FEATURE_NAMES
        stored, current = int(self.g_mean.shape[0]), len(FEATURE_NAMES)
        if stored != current:
            raise ValueError(
                f"{baseline_path} was trained on {stored}-dim feature vectors "
                f"but this firmware produces {current} dims. The feature "
                f"contract changed — the baseline must be retrained, not "
                f"migrated:\n    python firmware/baseline.py --simulate "
                f"--windows 48\n(or re-run the learn period on the machine)."
            )

        # T3.7 startup fingerprint. Baselines saved before this field existed
        # (b has no "startup_ratio_p95") make check_startup_fingerprint() a
        # no-op rather than raising on a missing key — a baseline that
        # predates the feature is not itself evidence of a mismatch.
        self.startup_ratio_p95: float | None = (
            float(b["startup_ratio_p95"]) if "startup_ratio_p95" in b else None)

        # T1.13 / SELF-REVIEW F19: per-machine calibrated crest_floor for
        # `features.select_demodulation_band`, computed from this baseline's
        # own learn period (see baseline.calibrate_crest_floor). Same
        # backward-compat pattern as startup_ratio_p95 above — a baseline
        # saved before this field existed deploys the original constant,
        # unchanged. The caller (main.py's window loop) is responsible for
        # passing this into extract_features(..., crest_floor=...); it is
        # not used inside score() itself, which only ever sees an already
        # -extracted vector.
        from features import DEFAULT_CREST_FLOOR
        self.crest_floor: float = (
            float(b["crest_floor"]) if "crest_floor" in b else DEFAULT_CREST_FLOOR)

    def score(self, vector: np.ndarray, op: np.ndarray) -> dict:
        opz = (op - self.op_mean) / self.op_scale
        regime = int(np.argmin(np.linalg.norm(self.op_centroids - opz, axis=1)))
        z = (vector - self.g_mean) / self.g_std
        delta = z - self.means[regime]
        score = float(np.sqrt(max(float(delta @ self.precisions[regime] @ delta), 0.0)))
        threshold = float(self.thresholds[regime])
        return {
            "score": score,
            "regime": regime,
            "threshold": threshold,
            "anomalous": score > threshold,
        }

    def check_startup_fingerprint(self, ratios: list[float]) -> None:
        """T3.7. Call ONCE, explicitly, after scoring a unit's first
        STARTUP_CHECK_WINDOWS real windows since it booted — pass their
        score/threshold ratios, in order. Deliberately NOT wired into
        score() itself: score() is used against curated/replayed windows
        all over this codebase's own test suite (feedback retraining,
        threshold behaviour, sensitivity sweeps) where "the first 8 calls"
        does not mean "the first 8 windows a real unit ever measured", and
        those uses must stay pure scoring with no hidden one-shot side
        effect. main.py's real startup loop is the only caller that means
        it literally.

        Raises BaselineMismatchError if the ratios look like a firmware/
        baseline mismatch rather than a genuine startup (see that class's
        docstring for the T1.8 measurement this defends against). A no-op
        if this baseline predates the fingerprint fields."""
        if self.startup_ratio_p95 is None or not ratios:
            return
        frac_anomalous = sum(r > 1.0 for r in ratios) / len(ratios)
        median_ratio = float(np.median(ratios))
        bound = STARTUP_CHECK_RATIO_MULTIPLE * max(self.startup_ratio_p95, 1.0)
        if frac_anomalous >= STARTUP_CHECK_MIN_FRACTION_ANOMALOUS and median_ratio >= bound:
            raise BaselineMismatchError(
                f"{sum(r > 1.0 for r in ratios)} of {len(ratios)} of this "
                f"unit's first windows are anomalous (median score/threshold "
                f"= {median_ratio:.1f}x; the learn period's own p95 ratio "
                f"was {self.startup_ratio_p95:.2f}x) — this looks like the "
                f"baseline was fit on a different feature-generation "
                f"contract, not a fault that appeared in the instant this "
                f"unit booted. Retrain it:\n    python firmware/baseline.py "
                f"--simulate --windows 48\n(or re-run the learn period on "
                f"the machine).")


class CloudAEScorer:
    """v1.5 (optional): tflite autoencoder inference on Mel patches. The model
    is trained SERVER-SIDE on uploaded learn-period spectrograms and pushed to
    the device; tflite-runtime cannot train and full TF does not fit in 512 MB,
    so this split is architectural, not a preference. Unused in v1."""

    def __init__(self, model_path: Path, scaler: dict):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]["index"]
        self._out = self.interp.get_output_details()[0]["index"]
        self.scaler = scaler

    def reconstruction_errors(self, patches: np.ndarray) -> np.ndarray:
        lo, hi = self.scaler["lo"], self.scaler["hi"]
        X = np.clip((patches - lo) / (hi - lo + 1e-9), 0, 1)[..., None].astype(np.float32)
        errs = np.empty(len(X))
        for i, x in enumerate(X):
            self.interp.set_tensor(self._in, x[None])
            self.interp.invoke()
            errs[i] = float(np.mean((self.interp.get_tensor(self._out)[0] - x) ** 2))
        return errs


class AlertGate:
    """Persistence gating with one-alert-per-episode semantics.

    feed(anomalous) -> True exactly once, when the streak first reaches
    `need` consecutive anomalous windows. The episode then latches; it
    re-arms only after `clear` consecutive normal windows (hysteresis, so a
    score wobbling around threshold doesn't machine-gun alerts)."""

    def __init__(self, need: int, clear: int = 4):
        self.need, self.clear = max(1, need), clear
        self.streak = 0
        self.normal_streak = 0
        self.in_episode = False

    def feed(self, anomalous: bool) -> bool:
        if anomalous:
            self.streak += 1
            self.normal_streak = 0
            if not self.in_episode and self.streak >= self.need:
                self.in_episode = True
                return True
        else:
            self.streak = 0
            self.normal_streak += 1
            if self.in_episode and self.normal_streak >= self.clear:
                self.in_episode = False
        return False
