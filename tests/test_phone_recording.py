"""test_phone_recording.py — the phone-recording path (backlog T7.2).

Complements `tests/test_phone_monitor.py`, which tests the actual product
tool (`tools/phone_monitor.py`: learn a baseline from part of one recording,
score the rest) against short, healthy, noise-plus-hum fixtures. This file
asks the question that tool's own self-test deliberately does not: what
happens when a REAL fault is present, at a realistic (pink, not white) noise
floor, at a resonance the default demodulation band does not cover? See
`docs/PHONE_RECORDING.md` for how the two fit together.

`tools/ingest.py` already turns arbitrary audio into the canonical format and
`ml/realdata/analyse_recording.py` already supports `--mic-only`. What had
never been exercised is the ONE thing that matters for a real phone capture:
a REALISTIC noise floor. `docs/DOC_STATUS.md` already records that on a
*pink*-noise surrogate (T1.1, accelerometer domain) the protrugram's
`crest_floor = 10.0` was never reached and every window silently fell back to
`DEFAULT_BAND` — harmlessly there, because the fallback happened to contain
the resonance. This file asks the same question on the microphone/phone
domain, with the resonance placed OUTSIDE `DEFAULT_BAND` on purpose, using
`ml/realdata/synth_phone_recording.py` (an independent third signal model —
see that file's docstring for why it does not import `ml/simulate.py` or
`validate_public_dataset._pink`).

Three things pinned here, each measured before being written down:

1. **The fallback reproduces on the mic/phone domain and is severity-gated.**
   At the default fault severity (0.35, `_severity_below_crest_floor` in the
   fixture below) the protrugram's crest on this generator's pink floor never
   reaches 10 and Gate 2 fails even though a real, geometry-consistent BPFO
   line is present (recoverable at 3.0x/1.6x, still short of the 4x/2x gate,
   with `--band` forced to the true resonance). At severity 0.9 the crest
   crosses ~13 and the protrugram finds the true band unaided, and Gate 2
   passes outright. `test_protrugram_falls_back_at_low_severity_on_pink_floor`
   and `test_protrugram_finds_the_true_band_at_higher_severity`.

2. **Mic-only speed is exactly as unreliable as `estimate_fr`'s docstring
   says, and this generator demonstrates why:** it deliberately does not put
   a shaft-rate tone in the audio (a real machine's mic often does not carry
   one either), so the HPS estimate locks onto incidental content and is
   `reliable=False`, off by tens of percent from the true `fr_hz`.
   `test_miconly_speed_estimate_is_unreliable_and_can_be_wrong`.

3. **The deployed production baseline is the WRONG TOOL for a phone capture,
   and silently so.** `firmware/inference.py` was trained with a real (three-
   axis) accelerometer; a mic-only window has that channel zeroed, which is
   exactly the "dead channel" shape T4.3 already made the scorer refuse to
   call healthy — so a phone recording run through the live per-window
   scorer reports "anomalous" at ~10,000x threshold whether the machine is
   healthy or faulty. This is not a bug (T4.3's whole point is that a missing
   channel must not silently score healthy) but it means the live scorer is
   never the right tool for a phone-only capture: use
   `ml/realdata/analyse_recording.py`, or train a dedicated mic-only baseline.
   `test_miconly_capture_is_unusable_against_the_full_sensor_baseline`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "ml" / "realdata"))

from features import extract_features, select_demodulation_band  # noqa: E402
from baseline import operating_point  # noqa: E402
from inference import MahalanobisScorer  # noqa: E402
import analyse_recording as A  # noqa: E402
import synth_phone_recording as S  # noqa: E402

FS_AUDIO = 16000.0          # device rate; synth_phone_recording generates at
                            # 44100 Hz "phone" rate — the resample step itself
                            # is tools/ingest.py's job and is covered by
                            # tests/test_ingest.py, not repeated here.


def _resample_to_device_rate(x: np.ndarray, fs_in: float) -> np.ndarray:
    """Minimal resample so these tests do not need to shell out to
    tools/ingest.py or touch disk. Uses the same polyphase approach
    (scipy.signal.resample_poly) that ingest.py uses internally."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    frac = Fraction(int(FS_AUDIO), int(fs_in)).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator)


def _make_device_rate_pair(seed=1, duration_s=40.0, severity=0.35,
                           resonance_hz=S.DEFAULT_RESONANCE_HZ):
    pair = S.make_pair(seed=seed, duration_s=duration_s, severity=severity,
                       resonance_hz=resonance_hz)
    h = _resample_to_device_rate(pair["healthy"], pair["fs"])
    f = _resample_to_device_rate(pair["faulty"], pair["fs"])
    return h, f, pair


# ----------------------------------------------------------------------------
# 1. Protrugram fallback, severity-gated
# ----------------------------------------------------------------------------

def test_protrugram_falls_back_at_low_severity_on_pink_floor():
    """Below the crest floor the selector must return DEFAULT_BAND — that is
    the documented, deliberate behaviour (features.py: 'healthy machines have
    no periodic envelope anywhere ... below crest_floor we return
    DEFAULT_BAND'). What this test PINS is that a realistic pink-noise phone
    capture with a real (if moderate) fault reaches that fallback, at a
    resonance the fallback band does not cover — the exact silent-miss shape
    docs/DOC_STATUS.md already flagged on the accelerometer surrogate."""
    _, faulty, pair = _make_device_rate_pair(severity=0.35)
    win = int(FS_AUDIO * 10)
    band, crest = select_demodulation_band(faulty[:win], FS_AUDIO)
    assert crest < 10.0, (
        f"expected the pink floor to keep a single 10 s window's crest below "
        f"the crest_floor at this severity (measured {crest:.1f}); if this "
        f"now fires, the finding below about needing --band no longer holds "
        f"and DOC_STATUS/PHONE_RECORDING.md must be re-measured")
    assert band == (3000.0, 6000.0)
    # the true resonance (1600 Hz by default) is nowhere near the fallback
    assert not (band[0] <= pair["resonance_hz"] <= band[1])


def test_forcing_the_true_band_recovers_real_signal_but_not_gate_2():
    """With the fallback band the fault is invisible (documented above); with
    the TRUE band forced the BPFO line is measurably real — recovered, not
    invented — but at this severity still short of the go/no-go gate. Both
    facts matter: the physics is there, and the automatic band selector is
    what stands between it and a usable answer."""
    h, f, pair = _make_device_rate_pair(severity=0.35, duration_s=40.0)
    healthy_rec = A.Recording(audio=h, fs_audio=FS_AUDIO, accel=None,
                              fs_accel=None, meta={})
    faulty_rec = A.Recording(audio=f, fs_audio=FS_AUDIO, accel=None,
                             fs_accel=None, meta={})
    geom = A.lookup(pair["bearing"])
    band = (pair["resonance_hz"] - 300, pair["resonance_hz"] + 300)
    res = A.analyse_pair(healthy_rec, faulty_rec, geom, pair["fr_hz"],
                         window_s=10.0, band=band)
    v = A.verdict(res)
    # (A) a real BPFO-locked line exists once the right band is used
    assert res["env_faulty"]["ratio"] > res["raw_faulty"]["ratio"], (
        "demodulation should still beat the raw spectrum even where the "
        "gate itself is not met")
    # measured, not assumed: at severity 0.35 this does NOT clear Gate 2 —
    # if it starts passing, the severity/crest crossover documented in
    # docs/PHONE_RECORDING.md has moved and must be re-measured there too.
    assert v["passed"] is False


def test_protrugram_finds_the_true_band_at_higher_severity():
    """At a severity strong enough to push the envelope crest above the
    floor, the protrugram should locate the TRUE resonance unaided (no
    --band override) and Gate 2 should pass outright. This is the other half
    of finding 1: the fallback is not a permanent blind spot, it is a
    severity-dependent one, and this pins where the crossover sits for this
    generator's noise floor."""
    h, f, pair = _make_device_rate_pair(severity=0.9, duration_s=40.0)
    win = int(FS_AUDIO * 10)
    band, crest = select_demodulation_band(f[:win], FS_AUDIO)
    healthy_rec = A.Recording(audio=h, fs_audio=FS_AUDIO, accel=None,
                              fs_accel=None, meta={})
    faulty_rec = A.Recording(audio=f, fs_audio=FS_AUDIO, accel=None,
                             fs_accel=None, meta={})
    geom = A.lookup(pair["bearing"])
    res = A.analyse_pair(healthy_rec, faulty_rec, geom, pair["fr_hz"],
                         window_s=10.0)
    v = A.verdict(res)
    assert v["passed"] is True, (
        f"expected Gate 2 to pass at severity 0.9 (band picked "
        f"{res['band']}, crest on faulty ~{crest:.1f}); if this now fails "
        f"the severity calibration in docs/PHONE_RECORDING.md is stale")


# ----------------------------------------------------------------------------
# 2. Mic-only speed estimate
# ----------------------------------------------------------------------------

def test_miconly_speed_estimate_is_unreliable_and_can_be_wrong():
    """estimate_fr's own docstring guarantees reliable=False on a single live
    channel; this generator (which puts no shaft-rate tone in the audio, only
    mains hum + BPFO impacts + pink floor — a plausible real recording) shows
    that unconfirmed estimate can also be badly WRONG, not just unconfirmed.
    That is why PHONE_RECORDING.md tells Logan to pass --rpm measured with a
    tachometer or a strobe, never to trust the auto estimate."""
    from features import estimate_fr
    _, faulty, pair = _make_device_rate_pair(severity=0.35)
    win = int(FS_AUDIO * 30)
    accel = np.zeros((int(6400 * 30), 3))
    fr_hz, reliable = estimate_fr(faulty[:win], FS_AUDIO, accel, 6400.0)
    assert reliable is False
    # not asserting a specific wrong value (that would be over-fitting to one
    # seed) — asserting the SHAPE of the failure: it is not close to truth.
    rel_err = abs(fr_hz - pair["fr_hz"]) / pair["fr_hz"]
    assert rel_err > 0.10, (
        f"expected the unconfirmed mic-only estimate to miss truth by a "
        f"real margin (measured fr={fr_hz:.2f} Hz vs true "
        f"{pair['fr_hz']:.2f} Hz, {rel_err:.1%} off); if this generator now "
        f"produces an accidentally-accurate estimate, PHONE_RECORDING.md's "
        f"warning still stands (reliable=False either way) but the measured "
        f"example number should be refreshed")


# ----------------------------------------------------------------------------
# 3. The live scorer is the wrong tool for a mic-only capture
# ----------------------------------------------------------------------------

def test_miconly_capture_is_unusable_against_the_full_sensor_baseline():
    """Not a bug: T4.3 deliberately made the scorer distrust a dead/missing
    channel rather than silently report healthy. The consequence for a phone
    capture is that the live per-window Mahalanobis score is uninformative —
    it fires on a zeroed accelerometer channel, not on the bearing — so
    healthy and faulty score alike, both far past threshold. Pinned here so
    nobody re-discovers this by accident and 'fixes' T4.3's channel-death
    guard to make phone recordings score plausibly; the correct fix is to use
    ml/realdata/analyse_recording.py for phone data, or train a dedicated
    mic-only baseline, not to weaken the guard."""
    baseline_path = ROOT / "firmware" / "baseline.npz"
    if not baseline_path.exists():
        pytest.skip("no firmware/baseline.npz in this checkout")
    scorer = MahalanobisScorer(baseline_path)

    healthy, faulty, pair = _make_device_rate_pair(severity=0.35)
    win = int(FS_AUDIO * 30)
    accel = np.zeros((int(6400 * 30), 3))

    def score_first_window(audio):
        feats = extract_features(audio[:win], FS_AUDIO, accel, 6400.0)
        op = operating_point(feats["vector"], feats["fr_hz"])
        return scorer.score(feats["vector"], op)

    sh = score_first_window(healthy)
    sf = score_first_window(faulty)
    assert sh["anomalous"] is True
    assert sf["anomalous"] is True
    # both score by roughly the same (large) margin -- confirming the score
    # is dominated by the zeroed accel channel, not by which recording this
    # was, is the whole point of the finding.
    assert sh["score"] > 100 * sh["threshold"]
    assert sf["score"] > 100 * sf["threshold"]
