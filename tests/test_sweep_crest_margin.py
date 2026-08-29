"""Guards for `tools/sweep_crest_margin.py` — F20's retraction.

The F20 write-up now rests on three measured claims. All three were things
that, had they been wrong, would have produced a clean-looking table that
meant nothing. So they are tested rather than trusted:

  1. The harness DISCRIMINATES — recovery genuinely responds to fault
     severity. A harness that returned the same answer regardless would have
     "confirmed" any hypothesis put to it.
  2. Severity 0.35 SATURATES it. This is why the sweep is not run at
     `make_pair`'s default, and why an earlier run that looked like a clean
     6/6 across the board was worthless.
  3. The 44.1 kHz phone rate DISABLES the margin entirely via
     `MIN_CREST_FLOOR` clamping. This is the trap that produced a spurious
     0/6-everywhere reading while the code under test was fine.

Deliberately NOT tested here: `deployed_fpr()`. It refits a 48-window baseline
per margin and takes minutes; `tests/test_evaluate_pinned.py` already pins the
shipped point cheaply against the real deployed baseline. Adding it here would
buy nothing and make the suite slow enough that people skip it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "firmware", ROOT / "ml" / "realdata"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_crest_margin import f19_recovery   # noqa: E402

# 3 machines and 18 s rather than the sweep's 6 and 20 s: enough to show the
# effect, fast enough to belong in the suite. 18 s is a FLOOR, not a taste --
# f19_recovery chunks the learn signal into 2 s windows and refuses to
# calibrate on fewer than MIN_CREST_SAMPLES (8) of them, so anything below
# 16 s raises rather than measuring. 12 s was tried first and did exactly that.
MACHINES = 3
DURATION = 18.0


def test_the_harness_actually_discriminates_between_severities():
    """The non-vacuity check. If a loud fault and a near-silent one produce
    the same recovery count, the harness measures nothing and every number in
    F20's table is meaningless — including the ones that support the fix."""
    loud, _ = f19_recovery(0.7, n_machines=MACHINES, duration_s=DURATION,
                           severity=0.35)
    quiet, _ = f19_recovery(0.7, n_machines=MACHINES, duration_s=DURATION,
                            severity=0.10)
    assert loud > quiet, (
        f"recovery is {loud}/{MACHINES} at severity 0.35 and {quiet}/"
        f"{MACHINES} at 0.10 — the harness is not responding to the fault at "
        f"all, so F20's sweep table cannot be trusted")


def test_default_severity_saturates_and_is_therefore_the_wrong_place_to_sweep():
    """Documents WHY the sweep defaults to severity 0.20. At 0.35 the faulty
    band crest is ~23 against floors of ~7, so every margin recovers
    everything and the sweep looks clean while proving nothing."""
    rec, tot = f19_recovery(3.0, n_machines=MACHINES, duration_s=DURATION,
                            severity=0.35)
    assert rec == tot, (
        "severity 0.35 no longer saturates even at margin 3.0 — the sweep's "
        "choice of 0.20 may need revisiting, and F20's table re-measuring")


def test_the_phone_sample_rate_disables_the_margin_via_min_floor_clamping():
    """The trap that produced a spurious 0/6-everywhere reading.

    At 44.1 kHz the band crests collapse (healthy ~3.5-4.1, faulty ~4.4)
    because the extra bandwidth dilutes the envelope peak across more
    candidate bands. `calibrate_crest_floor` then clamps to MIN_CREST_FLOOR
    for every margin, so nothing clears the floor and recovery is 0 whatever
    the margin is. Asserted so that a future change to MIN_CREST_FLOOR or to
    the band grid surfaces here with an explanation attached, rather than as
    a confusing sweep result someone has to re-derive.
    """
    at_phone_rate = f19_recovery(0.7, n_machines=MACHINES, duration_s=DURATION,
                                 severity=0.35, fs=44100.0)[0]
    at_repo_rate = f19_recovery(0.7, n_machines=MACHINES, duration_s=DURATION,
                                severity=0.35, fs=16000.0)[0]
    assert at_phone_rate == 0
    assert at_repo_rate > at_phone_rate, (
        "the 16 kHz and 44.1 kHz results no longer differ — either the crest "
        "scaling changed or MIN_CREST_FLOOR moved; re-read f19_recovery's "
        "docstring before trusting any sweep run at a non-default rate")


def test_the_margin_argument_actually_does_something():
    """T1.16 #3. An adversarial review replaced `CREST_FLOOR_MARGIN = margin`
    with a hardcoded 0.7 inside `f19_recovery` and all four tests still passed:
    nothing in a file whose entire purpose is a MARGIN SWEEP compared two
    different margins.

    Asserting on recovery would not fix it — recovery is genuinely constant
    across the useful margin range (that is F20's finding: there is no trade).
    So this asserts on the quantity the margin provably moves, the calibrated
    floor itself, which is what the margin is added to by construction.
    """
    floors = {m: _calibrated_floor(m) for m in (0.3, 0.7, 1.0)}
    assert floors[0.3] < floors[0.7] < floors[1.0], (
        f"the margin is not moving the calibrated floor: {floors}. Either it "
        f"is being ignored, or the min/max clamps are saturating for every "
        f"value tested — both make this whole script meaningless.")
    assert floors[0.7] - floors[0.3] == pytest.approx(0.4, abs=0.05), (
        "the floor should rise by exactly the margin difference; if it does "
        "not, something other than the margin is moving it")


def _calibrated_floor(margin: float) -> float:
    """The deployed floor for one machine at a given margin — the smallest
    quantity that proves the margin is wired through at all."""
    import numpy as np
    import baseline as baseline_mod
    from synth_phone_recording import make_pair
    from features import select_demodulation_band

    pair = make_pair(seed=1, duration_s=20.0, fs=16000.0)
    rate = pair["fs"]
    win = int(rate * 2.0)
    crests = [select_demodulation_band(pair["healthy"][i:i + win], rate,
                                       crest_floor=0.0)[1]
              for i in range(0, len(pair["healthy"]) - win + 1, win)]
    old = baseline_mod.CREST_FLOOR_MARGIN
    try:
        baseline_mod.CREST_FLOOR_MARGIN = margin
        return baseline_mod.calibrate_crest_floor(np.asarray(crests))
    finally:
        baseline_mod.CREST_FLOOR_MARGIN = old


def test_refuses_to_calibrate_on_too_few_learn_windows():
    """The harness must fall over loudly rather than quietly calibrating on a
    handful of windows — the exact mistake F17 was, measured with 6 learn
    windows when the floor was 48."""
    with pytest.raises(RuntimeError, match="learn windows"):
        f19_recovery(0.7, n_machines=1, duration_s=4.0, severity=0.20)
