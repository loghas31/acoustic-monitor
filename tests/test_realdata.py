"""Tests for ml/realdata/ — the Week-2 measurement chain.

Scope (backlog T1.3). Two files, two very different kinds of risk:

  * `fault_frequencies.py` is ALGEBRA. It cannot fail loudly — a wrong gamma or
    a transposed diameter still returns a plausible number in Hz, and you would
    then hunt for a peak that was never going to be there and conclude the
    sensor does not work. So it is tested against (a) an identity that must hold
    for any geometry, and (b) an INDEPENDENT authority (CWRU's own published
    multipliers), not against itself.

  * `analyse_recording.py` decides the go/no-go. Its two subtle parts are the
    shaft-harmonic masking in `peak_to_background` — which already fired once as
    a real bug, reporting a 175x "fault" that was a shaft harmonic in BOTH
    recordings — and the three-part verdict, whose whole value is that it can
    return INCONCLUSIVE rather than being forced into PASS/FAIL.

Deliberately NOT covered here: the ingest/resample path (36 tests in
tests/test_ingest.py) and the dataset classification/split logic (14 tests in
tests/test_validate_public_dataset.py). Duplicating them would add runtime and
no information.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "realdata"))

import analyse_recording as A  # noqa: E402
import fault_frequencies as F  # noqa: E402

DATA = ROOT / "data"


# ===========================================================================
# 1. The algebra: identities that must hold for EVERY geometry
# ===========================================================================
#
# These are not "does the code do what it did yesterday" tests. Each one is a
# statement about rolling-element kinematics that would remain true if the file
# were rewritten from scratch, which is what makes them worth having.

ALL_KEYS = sorted(F.BEARINGS)
SPEEDS_HZ = [10.0, 29.95, 47.5, 50.0, 120.0]


@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize("fr", SPEEDS_HZ)
def test_bpfo_plus_bpfi_equals_n_times_fr(key, fr):
    """THE identity: BPFO + BPFI = N * fr, for any gamma.

        BPFO + BPFI = (N/2) fr (1 - g) + (N/2) fr (1 + g) = N fr

    The gamma terms cancel exactly, so this holds whatever the ball and pitch
    diameters are. It is therefore a free correctness check on any geometry you
    look up or transcribe: get d/D wrong and BPFO and BPFI both move, but their
    SUM does not — unless you have also broken the formulae.

    Asserted to floating-point equality (rtol 1e-12), not to a few decimals,
    because there is no physics in the residual — only rounding.
    """
    g = F.BEARINGS[key]
    assert g.bpfo(fr) + g.bpfi(fr) == pytest.approx(g.n_elements * fr, rel=1e-12)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_bpfo_is_exactly_n_times_the_cage_frequency(key):
    """BPFO = N * FTF. A stationary outer-race defect is struck once per ball
    per cage revolution — that is the whole derivation, and it means the cage
    frequency is not an independent number to get wrong."""
    g = F.BEARINGS[key]
    assert g.bpfo(50.0) == pytest.approx(g.n_elements * g.ftf(50.0), rel=1e-12)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_bpfi_minus_bpfo_is_n_fr_gamma(key):
    """BPFI - BPFO = N * fr * gamma. The DIFFERENCE isolates the geometry that
    the sum cancels, so the two identities together pin gamma completely: pass
    both and your d/D is right, not just self-consistent."""
    g = F.BEARINGS[key]
    fr = 47.5
    assert g.bpfi(fr) - g.bpfo(fr) == pytest.approx(
        g.n_elements * fr * g.gamma, rel=1e-12)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_cage_is_always_slower_than_half_shaft_speed(key):
    """FTF = (fr/2)(1 - gamma) with 0 < gamma < 1, so the cage ALWAYS turns at
    less than half shaft speed and never backwards. A cage frequency above
    fr/2 means a diameter has been swapped for a radius."""
    g = F.BEARINGS[key]
    assert 0.0 < g.ftf(100.0) < 50.0


@pytest.mark.parametrize("key", ALL_KEYS)
def test_gamma_is_in_the_physically_sensible_range(key):
    """gamma = (d/D)cos(phi) is 0.20-0.25 for essentially every deep-groove ball
    bearing (the balls have to fit between the rings, and the rings have to have
    somewhere to be). The docstring's own rule of thumb: outside ~0.10-0.35 you
    have mixed up a diameter with a radius, or mm with inches — a factor-of-two
    error that produces a perfectly plausible-looking BPFO."""
    assert 0.10 < F.BEARINGS[key].gamma < 0.35


@pytest.mark.parametrize("key", ALL_KEYS)
def test_frequencies_scale_linearly_with_shaft_speed(key):
    """Every defect frequency is proportional to fr, which is what makes the
    published 'multiplier' form meaningful. Doubling the speed must double
    every line — if it does not, an fr has been hard-coded somewhere."""
    g = F.BEARINGS[key]
    a, b = g.all_frequencies(31.0), g.all_frequencies(62.0)
    for name in ("FTF", "BPFO", "BPFI", "BSF", "BSF_2x"):
        assert b[name] == pytest.approx(2.0 * a[name], rel=1e-12)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_multipliers_are_the_frequencies_at_unit_speed(key):
    """`multipliers()` is the speed-independent catalogue form: multiply by fr
    in Hz to get Hz. Also check `fr` itself is not in the dict — a multiplier of
    1.0 labelled 'fr' would silently become a fifth 'defect frequency' in any
    loop over the dict."""
    g = F.BEARINGS[key]
    m = g.multipliers()
    assert "fr" not in m
    for name, mult in m.items():
        assert g.all_frequencies(37.0)[name] == pytest.approx(37.0 * mult, rel=1e-12)


def test_ball_defect_line_is_twice_ball_spin():
    """A ball defect contacts BOTH races per ball revolution, so the observed
    line is usually 2*BSF. Several published tables (CWRU's included) quote the
    2x value under a 'rolling element' heading, which is exactly the sort of
    convention mismatch that makes you look 100 Hz away from your evidence."""
    g = F.lookup("6205-2RS-JEM")
    assert g.bsf_2x(29.95) == pytest.approx(2.0 * g.bsf(29.95), rel=1e-12)


def test_contact_angle_raises_bpfo():
    """gamma = (d/D)cos(phi), so a non-zero contact angle SHRINKS gamma and
    therefore RAISES BPFO = (N/2)fr(1 - gamma). Sign errors here are easy and
    invisible; an angular-contact bearing analysed as phi=0 puts the predicted
    line a few percent low, which is the same size as the slip window."""
    base = dict(designation="x", n_elements=9, ball_diameter_mm=7.94,
                pitch_diameter_mm=39.04)
    straight = F.BearingGeometry(**base, contact_angle_deg=0.0)
    angled = F.BearingGeometry(**base, contact_angle_deg=30.0)
    assert angled.gamma < straight.gamma
    assert angled.bpfo(50.0) > straight.bpfo(50.0)
    assert angled.gamma == pytest.approx(straight.gamma * math.cos(math.radians(30.0)))


def test_rpm_to_hz():
    """'The single most common unit error in this whole subject.' 2850 rpm is
    47.5 Hz. Pass rpm into a formula expecting Hz and your predicted BPFO is out
    by a factor of 60 — off the end of the spectrum, so you would see nothing at
    all and blame the microphone."""
    assert F.rpm_to_hz(2850.0) == pytest.approx(47.5)
    assert F.rpm_to_hz(1797.0) == pytest.approx(29.95)


def test_the_repos_own_synthetic_default_is_reproducible():
    """ml/simulate.py's default machine is a 6204 at fr = 50 Hz, and every
    synthetic result in the repo (verify_signals.py's 56.7x, the Gate-2 runs)
    is quoted at BPFO = 152.6 Hz. Pin it, so a future edit to the 6204 entry
    cannot silently invalidate published numbers.

    Note the second assertion: the predicted line sits 2.6 Hz from the 3rd shaft
    harmonic at 150 Hz. That closeness is not incidental — it is the reason
    `peak_to_background` needs harmonic masking at all (section 4 below).
    """
    g = F.lookup("6204")
    assert g.bpfo(50.0) == pytest.approx(152.597, abs=0.001)
    assert g.bpfi(50.0) == pytest.approx(247.403, abs=0.001)
    assert abs(g.bpfo(50.0) - 3 * 50.0) == pytest.approx(2.597, abs=0.001)


# ===========================================================================
# 2. Agreement with an INDEPENDENT authority (CWRU's published multipliers)
# ===========================================================================
#
# Everything in section 1 is internal consistency: the file agreeing with
# itself. This section is the only place the formulae are checked against
# numbers this project did not compute.
#
# CWRU publishes BOTH the geometry (in inches) AND the resulting multipliers on
# its Bearing Information page. That redundancy is a gift: we compute the
# multipliers from their inches and compare with their published values.

# Source: CWRU Bearing Data Center, "Bearing Information".
CWRU_PUBLISHED = {
    "6205-2RS-JEM": {"BPFI": 5.4152, "BPFO": 3.5848, "FTF": 0.39828,
                     "BSF_2x": 4.7135},
    "6203-2RS-JEM": {"BPFI": 4.9469, "BPFO": 3.0530, "FTF": 0.3817,
                     "BSF_2x": 3.9874},
}


@pytest.mark.parametrize("key,published", sorted(CWRU_PUBLISHED.items()))
def test_matches_cwru_published_multipliers(key, published):
    """Reproduce CWRU's published multipliers from CWRU's published inches.

    TOLERANCE, AND WHY IT IS 5e-4 AND NOT 1e-4
    ------------------------------------------
    fault_frequencies.py's module comment claims agreement 'to 4 decimal
    places'. Measured, it is not quite that good, and the discrepancy is
    physical rather than a bug:

        6203-2RS-JEM  BPFO   computed 3.05312   published 3.0530   (2.9e-4 rel)
        6203-2RS-JEM  BSF_2x computed 3.98768   published 3.9874   (7.0e-5 rel)
        6205-2RS-JEM  BPFO   computed 3.58478   published 3.5848   (5.6e-6 rel)
        6205-2RS-JEM  FTF    computed 0.39831   published 0.39828  (7.5e-5 rel)

    CWRU publishes the pitch diameter to four significant figures (1.537 in,
    1.122 in). A 4-s.f. D carries ~3e-4 relative uncertainty, which propagates
    into gamma and is then multiplied by N/2 — about 3e-4 absolute on the outer
    multiplier. That is precisely the size of the residual, so the two sources
    agree to within the precision CWRU actually published.

    The practical consequence, and the reason this is not just pedantry: a 3e-4
    relative error on BPFO is 0.05 Hz at 152 Hz, which is a hundred times
    smaller than the 2 % slip window the analysis searches. The geometry is not
    the limiting uncertainty. The RPM is.
    """
    m = F.BEARINGS[key].multipliers()
    for name, value in published.items():
        assert m[name] == pytest.approx(value, abs=5e-4), (
            f"{key} {name}: computed {m[name]:.6f} vs CWRU {value}")


@pytest.mark.parametrize("key", sorted(CWRU_PUBLISHED))
def test_ball_count_is_recoverable_from_the_published_outer_multiplier(key):
    """CWRU publishes the multipliers but NOT the ball count. The table's N was
    recovered by inverting BPFO_mult = (N/2)(1 - gamma).

    That inversion landing on an integer is a genuine consistency check — it
    uses the published multiplier and the published diameters, two numbers that
    only agree if both are right. Assert it lands within 0.01 of an integer:
    N = 9 for the 6205, N = 8 for the 6203.
    """
    g = F.BEARINGS[key]
    n_recovered = 2.0 * CWRU_PUBLISHED[key]["BPFO"] / (1.0 - g.gamma)
    assert n_recovered == pytest.approx(g.n_elements, abs=0.01)


def test_boundary_dimension_rule_reproduces_the_cwru_6203_pitch_exactly():
    """The 'estimated' entries infer D ~ (bore + OD)/2. That rule is only
    trustworthy because it can be checked against a published case: the CWRU
    6203 is a 17x40 mm bearing, so the rule gives 28.5 mm against a published
    1.122 in = 28.499 mm. Agreement to 0.01 mm."""
    est_6203 = F.from_boundary_dimensions(17.0, 40.0, 8, ball_diameter_mm=6.746)
    assert est_6203.pitch_diameter_mm == pytest.approx(
        F.BEARINGS["6203-2RS-JEM"].pitch_diameter_mm, abs=0.01)


def test_a_pitch_diameter_error_is_ATTENUATED_in_bpfo_not_passed_through():
    """The 6205 is the case that keeps the rule honest — and it corrected the
    source comment while this test was being written.

    (bore + OD)/2 gives D = 38.50 mm against CWRU's published 39.04 mm: 1.38 %
    low. fault_frequencies.py's `6205` entry said that costs '~1.4 % in BPFO'.
    Measured, it costs 0.358 %. The comment has been corrected; here is why it
    was wrong, because the reason is useful:

        BPFO  = (N/2) fr (1 - gamma),  gamma = d/D
        dBPFO/BPFO = -dgamma/(1 - gamma) = (gamma/(1 - gamma)) * (dD/D)

    The sensitivity is not 1, it is gamma/(1 - gamma) = 0.255 for this bearing.
    A pitch-diameter error is ATTENUATED roughly fourfold on its way into BPFO,
    because BPFO depends on (1 - gamma) and gamma is small. Predicted 0.353 %,
    measured 0.358 %.

    BPFI carries (1 + gamma) instead, so its sensitivity is gamma/(1 + gamma) =
    0.169 — attenuated even harder. Both are asserted below.

    WHY THIS MATTERS PRACTICALLY: a guessed pitch diameter puts BPFO 0.64 Hz off
    at fr = 50 Hz, well inside the +/-2 % (+/-3.6 Hz) slip window the analysis
    already searches. So estimated geometry is NOT the thing that will make
    Week 2 inconclusive. The ball count N is — BPFO scales linearly with it, so
    miscounting 8 balls as 9 is a 12.5 % error, four times the slip window.
    Count the balls through the seal gap; do not bother agonising over D.
    """
    est = F.from_boundary_dimensions(25.0, 52.0, 9, ball_diameter_mm=7.94)
    pub = F.BEARINGS["6205-2RS-JEM"]
    g = pub.gamma

    d_pitch_pct = 100.0 * abs(est.pitch_diameter_mm - pub.pitch_diameter_mm) \
        / pub.pitch_diameter_mm
    assert d_pitch_pct == pytest.approx(1.38, abs=0.05)

    d_bpfo_pct = 100.0 * abs(est.bpfo(50.0) - pub.bpfo(50.0)) / pub.bpfo(50.0)
    assert d_bpfo_pct == pytest.approx(d_pitch_pct * g / (1.0 - g), rel=0.02)
    assert d_bpfo_pct == pytest.approx(0.358, abs=0.01)

    d_bpfi_pct = 100.0 * abs(est.bpfi(50.0) - pub.bpfi(50.0)) / pub.bpfi(50.0)
    assert d_bpfi_pct == pytest.approx(d_pitch_pct * g / (1.0 + g), rel=0.02)

    # In Hz at the repo's reference speed: sub-hertz, inside the slip window.
    assert abs(est.bpfo(50.0) - pub.bpfo(50.0)) < 1.0


def test_the_ball_count_is_the_parameter_that_actually_hurts():
    """The companion to the test above. N is the one parameter the boundary-
    dimension rules cannot guess, and BPFO scales linearly with it: one extra
    ball is a 12.5 % error on an 8-ball bearing — four times the slip window,
    and it moves BPFO 19 Hz at fr = 50 Hz. That is a completely different peak.

    This is why the table's 'estimated' entries say VERIFY N BY EYE."""
    eight = F.from_boundary_dimensions(20.0, 47.0, 8, ball_diameter_mm=7.94)
    nine = F.from_boundary_dimensions(20.0, 47.0, 9, ball_diameter_mm=7.94)
    err_pct = 100.0 * (nine.bpfo(50.0) - eight.bpfo(50.0)) / eight.bpfo(50.0)
    assert err_pct == pytest.approx(12.5, abs=0.01)
    assert nine.bpfo(50.0) - eight.bpfo(50.0) > 15.0     # Hz — a different line


# ===========================================================================
# 3. Geometry construction: the input errors a tired student will actually make
# ===========================================================================

def test_lookup_is_case_insensitive_and_understands_aliases():
    assert F.lookup("6204") is F.BEARINGS["6204"]
    assert F.lookup("  6205-2rs-jem ") is F.BEARINGS["6205-2RS-JEM"]
    assert F.lookup("6205-CWRU") is F.BEARINGS["6205-2RS-JEM"]
    assert F.lookup("6203-2rs") is F.BEARINGS["6203-2RS-JEM"]


def test_unknown_bearing_error_lists_what_is_available():
    """A KeyError saying '6210' is useless at the bench. The message must name
    the alternatives and the escape hatch (explicit geometry)."""
    with pytest.raises(KeyError) as e:
        F.lookup("6210")
    msg = str(e.value)
    assert "6204" in msg and "--bore-mm" in msg


@pytest.mark.parametrize("bore,od,n", [
    (0.0, 47.0, 8),      # zero bore
    (-20.0, 47.0, 8),    # negative bore
    (47.0, 20.0, 8),     # OD and bore transposed — the realistic mistake
    (20.0, 20.0, 8),     # degenerate
    (20.0, 47.0, 2),     # a "bearing" with two balls
])
def test_boundary_dimensions_reject_impossible_input(bore, od, n):
    with pytest.raises(ValueError):
        F.from_boundary_dimensions(bore, od, n)


def test_boundary_dimensions_ball_rule_when_diameter_unknown():
    """If you do not know d either, the fallback is d ~ 0.6 * (OD - bore)/2:
    the balls fill roughly 60 % of the radial gap. Check the rule is what is
    implemented, and that the result is still a physically sensible gamma —
    an implausible fallback would poison every prediction made without a
    ball diameter."""
    g = F.from_boundary_dimensions(20.0, 47.0, 8)
    assert g.ball_diameter_mm == pytest.approx(0.6 * 0.5 * (47.0 - 20.0))
    assert 0.10 < g.gamma < 0.35


def test_estimated_geometries_are_labelled_estimated():
    """`confidence` is carried alongside the numbers on purpose: in Week 2 you
    need to know whether you are betting the go/no-go on a drawing or on a
    guess. An 'estimated' geometry silently labelled 'published' would turn a
    2 % near-miss into an apparent failure of the physics."""
    assert F.from_boundary_dimensions(20.0, 47.0, 8).confidence == "estimated"
    assert F.BEARINGS["6205-2RS-JEM"].confidence == "published"
    assert F.BEARINGS["6206"].confidence == "estimated"


def test_report_prints_the_identity_and_it_balances():
    """format_report SHOWS the identity rather than asserting it, so a student
    can do the check in their head. Make sure the two printed sums actually
    agree — a report that prints a failing check is worse than no check."""
    g = F.lookup("6204")
    text = F.format_report(g, 50.0)
    assert "BPFO + BPFI" in text and "must match" in text
    assert f"{g.bpfo(50.0) + g.bpfi(50.0):.3f}" in text
    assert f"{g.n_elements * 50.0:.3f}" in text


# ===========================================================================
# 4. peak_to_background — slip windows and shaft-harmonic masking
# ===========================================================================
#
# This is the function that already produced a wrong answer with total
# confidence. Its docstring records the incident: a +/-2 % slip window at
# 152.6 Hz is +/-3.05 Hz wide and therefore SWALLOWS the 150 Hz third shaft
# harmonic, which was duly reported as a 175x bearing fault in the healthy
# recording as well as the faulty one.
#
# The tests below build spectra by hand — a flat floor with tones dropped into
# known bins — because that is the only way to know what the right answer is.


def _spectrum(df: float = 0.1, fmax: float = 600.0, floor: float = 1.0):
    freqs = np.arange(0.0, fmax, df)
    return freqs, np.full_like(freqs, floor)


def _put(freqs, mag, hz, height):
    """Drop a tone of the given height into the nearest bin."""
    mag[int(np.argmin(np.abs(freqs - hz)))] = height
    return mag


def test_finds_a_clean_line_and_reports_its_height_over_the_floor():
    freqs, mag = _spectrum()
    _put(freqs, mag, 152.6, 12.0)
    r = A.peak_to_background(freqs, mag, target_hz=152.6, fr_hz=None)
    assert r["ratio"] == pytest.approx(12.0, rel=1e-3)
    assert r["peak_hz"] == pytest.approx(152.6, abs=0.05)
    assert r["slip_pct"] == pytest.approx(0.0, abs=0.05)
    assert r["confounded"] is False


def test_slip_is_measured_not_assumed():
    """Real bearings slide 1-2 %, so the line sits BELOW the computed BPFO. The
    point of reporting `slip_pct` rather than silently accepting anything in the
    window: 2 % low is physics, 20 % away means the rpm or the geometry is
    wrong, and only the reported number distinguishes them."""
    freqs, mag = _spectrum()
    _put(freqs, mag, 150.3, 9.0)          # 1.5 % below a 152.6 Hz target
    r = A.peak_to_background(freqs, mag, target_hz=152.6, tol_pct=2.0, fr_hz=None)
    assert r["peak_hz"] == pytest.approx(150.3, abs=0.05)
    assert r["slip_pct"] == pytest.approx(-1.5, abs=0.05)


def test_a_line_outside_the_slip_window_is_not_found():
    """The window is a commitment: 3 % away is not 'nearly right', it is a
    different line. Widening --tol-pct until the peak appears is how you talk
    yourself into a result."""
    freqs, mag = _spectrum()
    _put(freqs, mag, 157.5, 50.0)         # +3.2 %, outside a +/-2 % window
    r = A.peak_to_background(freqs, mag, target_hz=152.6, tol_pct=2.0, fr_hz=None)
    assert r["ratio"] == pytest.approx(1.0, rel=0.05)   # floor only


def test_background_is_a_median_so_the_fault_line_cannot_inflate_it():
    """Background is the MEDIAN over 20-500 Hz, not the mean. The tall things in
    that range — the shaft harmonics and the fault line itself — are exactly
    what we are measuring, so letting them raise the floor they are compared
    against would shrink every ratio in the report.

    Here: a floor of 1.0 with a handful of 500x spikes. Median stays 1.0; the
    mean would be several times higher and the reported ratio correspondingly
    smaller."""
    freqs, mag = _spectrum()
    for hz in (40.0, 80.0, 120.0, 200.0, 300.0):
        _put(freqs, mag, hz, 500.0)
    _put(freqs, mag, 152.6, 10.0)
    r = A.peak_to_background(freqs, mag, target_hz=152.6, fr_hz=None)
    assert r["background"] == pytest.approx(1.0, rel=1e-6)
    assert r["ratio"] == pytest.approx(10.0, rel=1e-3)


def test_an_unmasked_shaft_harmonic_IS_reported_as_a_bearing_fault():
    """THE REGRESSION TEST. This reproduces the original bug on purpose.

    Spectrum: a 100x shaft harmonic at 150 Hz (3 x fr, fr = 50) and a genuine
    but modest 12x bearing line at 152.6 Hz. With `fr_hz=None` — i.e. no
    harmonic masking — the search returns the 150.00 Hz harmonic at 100x and
    calls it BPFO, because it is the tallest thing in the slip window.

    Measured: ratio 100.0x, peak at 150.00 Hz, "slip" -1.70 %. Every one of
    those numbers looks like a textbook result. The healthy recording would
    produce the identical number, which is what eventually gave the game away.
    """
    freqs, mag = _spectrum()
    _put(freqs, mag, 150.0, 100.0)
    _put(freqs, mag, 152.6, 12.0)
    bad = A.peak_to_background(freqs, mag, target_hz=152.6, tol_pct=2.0, fr_hz=None)
    assert bad["peak_hz"] == pytest.approx(150.0, abs=0.05)
    assert bad["ratio"] == pytest.approx(100.0, rel=1e-3)
    assert bad["slip_pct"] == pytest.approx(-1.70, abs=0.05)
    assert bad["excluded_hz"] == []


def test_masking_the_shaft_harmonic_recovers_the_real_bearing_line():
    """Same spectrum, `fr_hz=50` supplied. The 150 Hz guard band is excised and
    the search returns the real line: 12x at 152.60 Hz, zero slip. The masked
    harmonic is reported in `excluded_hz` so the report can say what it did
    rather than quietly deleting data."""
    freqs, mag = _spectrum()
    _put(freqs, mag, 150.0, 100.0)
    _put(freqs, mag, 152.6, 12.0)
    good = A.peak_to_background(freqs, mag, target_hz=152.6, tol_pct=2.0, fr_hz=50.0)
    assert good["peak_hz"] == pytest.approx(152.6, abs=0.05)
    assert good["ratio"] == pytest.approx(12.0, rel=1e-3)
    assert good["confounded"] is False
    assert good["excluded_hz"] == [150.0]


def test_guard_band_is_wider_than_one_bin_but_far_narrower_than_the_slip_window():
    """guard = max(3 * df, 0.35 Hz). Three bins covers the main-lobe leakage of
    a Hann-windowed tone — mask only the peak bin and the tone's skirts survive
    to be picked as the maximum. The two bounds matter in opposite directions:
    too narrow and leakage is mistaken for the fault; too wide and the guard
    eats the slip window (see the next test)."""
    freqs, mag = _spectrum(df=0.1)
    # A leaky harmonic: peak at 150.0 with skirts either side, all taller than
    # the genuine 152.6 Hz line.
    for off, h in ((-0.2, 40.0), (-0.1, 70.0), (0.0, 100.0), (0.1, 70.0), (0.2, 40.0)):
        _put(freqs, mag, 150.0 + off, h)
    _put(freqs, mag, 152.6, 12.0)
    r = A.peak_to_background(freqs, mag, target_hz=152.6, tol_pct=2.0, fr_hz=50.0)
    assert r["peak_hz"] == pytest.approx(152.6, abs=0.05), \
        "leakage skirts survived the guard band and were picked as the fault"
    assert r["ratio"] == pytest.approx(12.0, rel=1e-3)


def test_target_sitting_on_a_shaft_harmonic_is_confounded_not_reported():
    """When the predicted fault frequency lands ON a shaft harmonic the two are
    the same line at this resolution and NOTHING honest can be reported. The
    function returns NaN and says so, rather than a number indistinguishable
    from a shaft harmonic.

    An unresolvable measurement is a fact about the experiment, not a result to
    be rounded off."""
    freqs, mag = _spectrum()
    _put(freqs, mag, 150.0, 100.0)
    r = A.peak_to_background(freqs, mag, target_hz=150.0, tol_pct=2.0, fr_hz=50.0)
    assert r["confounded"] is True
    assert math.isnan(r["ratio"]) and math.isnan(r["peak_hz"])
    assert "not separable" in r["note"]
    assert r["excluded_hz"] == [150.0]


def test_guard_bands_can_swallow_the_whole_slip_window():
    """The other degenerate case, and it is NOT the same as confounded: the
    target is far enough from the harmonic to be distinguishable in principle,
    but at this frequency resolution every bin inside the (narrow) slip window
    falls in a guard band. Reachable, and it must not be silently reported as a
    zero or a floor-level ratio.

    Constructed: 1 Hz bins => guard = 3 Hz. Target 153.4 Hz is 3.4 Hz from the
    150 Hz harmonic, so not confounded. A 0.33 % slip window is +/-0.5 Hz and
    contains exactly the 153 Hz bin, which is 3.0 Hz from the harmonic and so
    masked. Nothing survives; the note tells you the fix is a tighter
    --tol-pct, not a different machine."""
    freqs, mag = _spectrum(df=1.0, fmax=300.0)
    r = A.peak_to_background(freqs, mag, target_hz=153.4,
                             tol_pct=100.0 * 0.5 / 153.4, fr_hz=50.0)
    assert r["confounded"] is False
    assert math.isnan(r["ratio"])
    assert "entirely covered" in r["note"] and "tol-pct" in r["note"]


def test_harmonics_beyond_the_search_limit_are_not_masked():
    """The mask only extends to `n_shaft_harmonics` (default 12). Above that a
    shaft harmonic is invisible to the guard and would be picked as the fault.

    This is a documented LIMIT, not a bug — 12 x fr is 600 Hz on a 50 Hz shaft,
    comfortably past any first-order bearing line. But it is worth pinning,
    because a slow machine (fr = 10 Hz, BPFO above 130 Hz) walks straight into
    it, and the fix is one keyword argument."""
    freqs, mag = _spectrum()
    _put(freqs, mag, 130.0, 80.0)          # 13 x fr for fr = 10 Hz
    default = A.peak_to_background(freqs, mag, target_hz=130.0, fr_hz=10.0)
    assert default["confounded"] is False and default["ratio"] == pytest.approx(80.0, rel=1e-3)

    widened = A.peak_to_background(freqs, mag, target_hz=130.0, fr_hz=10.0,
                                   n_shaft_harmonics=13)
    assert widened["confounded"] is True


def test_target_outside_the_analysed_range_says_so():
    freqs, mag = _spectrum(fmax=200.0)
    r = A.peak_to_background(freqs, mag, target_hz=850.0, fr_hz=None)
    assert math.isnan(r["ratio"])
    assert "outside" in r["note"]


def test_non_positive_target_is_a_programming_error():
    freqs, mag = _spectrum()
    for bad in (0.0, -152.6):
        with pytest.raises(ValueError):
            A.peak_to_background(freqs, mag, target_hz=bad)


# ===========================================================================
# 5. verdict — the three-part gate and the third outcome
# ===========================================================================
#
# The verdict is built from four measurement dicts, so it can be tested without
# any signal processing at all. That is deliberate: the gate logic is a policy
# decision and should be readable and testable on its own.


def _res(env_f: float, env_h: float, raw_f: float, *, confounded: bool = False,
         note: str = "") -> dict:
    """Minimal `analyse_pair` result carrying only what `verdict` reads."""
    def m(ratio, conf=False, nte=""):
        return {"ratio": ratio, "confounded": conf, "note": nte,
                "peak_hz": float("nan"), "slip_pct": float("nan"),
                "excluded_hz": []}
    return {
        "target_name": "BPFO",
        "env_faulty": m(env_f, confounded, note),
        "env_healthy": m(env_h),
        "raw_faulty": m(raw_f),
    }


def test_all_three_checks_pass():
    v = A.verdict(_res(env_f=40.0, env_h=2.0, raw_f=2.5))
    assert v["passed"] is True and v["inconclusive"] is False
    assert [ok for _, ok, _ in v["checks"]] == [True, True, True]
    assert v["contrast"] == pytest.approx(20.0, rel=1e-6)
    assert v["envelope_gain"] == pytest.approx(16.0, rel=1e-6)


def test_check_A_fails_when_there_is_no_line_at_all():
    """Absolute: a 3x line is not evidence of anything, however good the
    contrast looks."""
    v = A.verdict(_res(env_f=3.0, env_h=0.5, raw_f=1.0), min_ratio=4.0)
    assert v["passed"] is False
    tags = {tag: ok for tag, ok, _ in v["checks"]}
    assert tags["A absolute"] is False
    assert tags["B contrast"] is True      # contrast alone would have passed


def test_check_B_is_the_control_condition_most_projects_skip():
    """A big envelope ratio on the faulty recording proves nothing if the
    healthy recording has the same one — that is a property of the machine, the
    room or the mounting, not of the fault.

    Here both recordings show 40x. Check A passes handsomely. The gate still
    fails, and it fails for the right reason."""
    v = A.verdict(_res(env_f=40.0, env_h=40.0, raw_f=2.0), min_contrast=2.0)
    tags = {tag: ok for tag, ok, _ in v["checks"]}
    assert tags["A absolute"] is True
    assert tags["B contrast"] is False
    assert v["passed"] is False
    assert v["contrast"] == pytest.approx(1.0, rel=1e-6)


def test_check_C_fails_when_demodulation_adds_nothing():
    """Method: if the raw spectrum shows the line just as well, we have not
    reproduced the project's central claim — and the whole argument for a
    16 kHz microphone over a cheap low-rate accelerometer is that claim."""
    v = A.verdict(_res(env_f=10.0, env_h=1.0, raw_f=25.0))
    tags = {tag: ok for tag, ok, _ in v["checks"]}
    assert tags["A absolute"] is True and tags["B contrast"] is True
    assert tags["C method"] is False
    assert v["passed"] is False


def test_an_unmeasurable_raw_ratio_counts_as_raw_told_us_nothing():
    """The raw measurement can be confounded while the envelope one is fine —
    the raw spectrum is where BPFO sits next to a shaft harmonic, and the whole
    point of demodulating is that the envelope spectrum does not have that
    problem. So a NaN raw ratio must PASS check C ('demodulation did the work'),
    not fail it, and the reported gain is infinite."""
    res = _res(env_f=40.0, env_h=2.0, raw_f=float("nan"))
    v = A.verdict(res)
    assert v["passed"] is True
    assert math.isinf(v["envelope_gain"])
    assert "UNMEASURABLE" in v["checks"][2][2]


def test_confounded_fault_line_gives_INCONCLUSIVE_not_FAIL():
    """The third outcome, and the reason it exists: collapsing 'the fault line
    is on top of a shaft harmonic' into FAIL would send two students chasing a
    mounting problem they do not have.

    The fix for INCONCLUSIVE is to change the EXPERIMENT — run the rig 10 %
    faster so BPFO moves off the harmonic — not to change the analysis."""
    v = A.verdict(_res(env_f=float("nan"), env_h=float("nan"), raw_f=float("nan"),
                       confounded=True, note="target 150.00 Hz is within 0.60 Hz"))
    assert v["inconclusive"] is True
    assert v["passed"] is False           # inconclusive is never a pass
    assert v["checks"] == []
    assert "150.00" in v["reason"]


def test_verdict_flags_are_plain_python_bools():
    """`analyse_pair` compares numpy scalars, so the checks would naturally hold
    np.bool_. The code wraps them in bool() on purpose: np.bool_ is not JSON
    serialisable, and this verdict is what gets written into the Week-2
    lab-notebook entry."""
    v = A.verdict(_res(env_f=40.0, env_h=2.0, raw_f=2.5))
    assert type(v["passed"]) is bool
    for _, ok, _ in v["checks"]:
        assert type(ok) is bool


@pytest.mark.parametrize("min_ratio,min_contrast,expected", [
    (4.0, 2.0, True),
    (50.0, 2.0, False),     # raise the absolute bar above the measurement
    (4.0, 30.0, False),     # raise the contrast bar above the measurement
])
def test_thresholds_are_honoured(min_ratio, min_contrast, expected):
    v = A.verdict(_res(env_f=40.0, env_h=2.0, raw_f=2.5), min_ratio, min_contrast)
    assert v["passed"] is expected


# ===========================================================================
# 6. End to end on the repo's own simulated pair
# ===========================================================================
#
# Sections 1-5 test the pieces. This runs the actual Gate-2 chain — load, pick a
# demodulation band on the faulty record, average envelope spectra, mask
# harmonics, apply the gate — on data/normal.wav and data/bearing_outer.wav.
#
# SYNTHETIC. These files come from ml/simulate.py. Passing here says the
# measurement chain is wired up correctly; it says nothing about real bearings.


@pytest.fixture(scope="module")
def sim_pair():
    from recording_io import load_recording
    if not (DATA / "normal.wav").exists():
        pytest.skip("data/normal.wav not present")
    return (load_recording(DATA / "normal.wav"),
            load_recording(DATA / "bearing_outer.wav"))


@pytest.fixture(scope="module")
def sim_result(sim_pair):
    healthy, faulty = sim_pair
    return A.analyse_pair(healthy, faulty, F.lookup("6204"), fr_hz=50.0,
                          window_s=5.0, tol_pct=2.0)


def test_gate2_passes_on_the_simulated_pair(sim_result):
    """Observed 2026-08-17: envelope ratio at BPFO 61.3x on faulty vs 2.2x on
    healthy (contrast 28.1x), raw 2.6x, so demodulation gains 23.8x. Gate PASS.

    Bounds are deliberately loose — this is a smoke test of the chain, not a
    published number, and tightening it would make an unrelated change to the
    band selection look like a regression."""
    v = A.verdict(sim_result)
    assert v["inconclusive"] is False
    assert v["passed"] is True
    assert sim_result["env_faulty"]["ratio"] > 20.0
    assert v["contrast"] > 5.0
    assert v["envelope_gain"] > 5.0


def test_the_fault_line_lands_where_the_geometry_predicts(sim_result):
    """The falsifiable prediction: BPFO = 152.60 Hz for a 6204 at 50 Hz. The
    simulator has no slip, so the peak should land essentially on it. On real
    data expect 1-2 % low — that is the number to watch in Week 2."""
    assert sim_result["target_hz"] == pytest.approx(152.597, abs=0.001)
    assert sim_result["env_faulty"]["peak_hz"] == pytest.approx(152.6, abs=0.5)
    assert abs(sim_result["env_faulty"]["slip_pct"]) < 1.0


def test_the_third_shaft_harmonic_is_masked_in_every_spectrum(sim_result):
    """150 Hz is 2.6 Hz from BPFO and inside the +/-2 % window, so it must be
    excluded from all four measurements — including the healthy ones, which is
    where the original bug was most obvious."""
    for key in ("raw_healthy", "raw_faulty", "env_healthy", "env_faulty"):
        assert sim_result[key]["excluded_hz"] == [150.0], key


def test_the_defect_rings_as_a_harmonic_comb(sim_result):
    """A real outer-race defect produces a comb at 1x, 2x, 3x BPFO — noise does
    not put peaks at exactly twice and three times your predicted frequency.
    This is the strongest confirmation available short of taking the bearing
    apart. Observed: 60.7x / 9.8x / 3.2x, each within 0.1 % of prediction."""
    combs = sim_result["harmonics"]
    assert len(combs) == 3
    for i, h in enumerate(combs, start=1):
        assert np.isfinite(h["ratio"]), f"BPFOx{i} unmeasurable"
        assert abs(h["slip_pct"]) < 1.0, f"BPFOx{i} landed {h['slip_pct']:.2f} % away"
    assert combs[0]["ratio"] > combs[2]["ratio"], \
        "the fundamental should be the tallest tooth of the comb"


def test_the_healthy_recording_does_not_pass_as_its_own_fault(sim_pair):
    """The sharpest control available: run the gate with the HEALTHY recording
    in both slots. Contrast is 1.0 by construction, so check B must fail and the
    gate must not pass. If this ever passes, the gate is measuring the machine
    rather than the fault."""
    healthy, _ = sim_pair
    res = A.analyse_pair(healthy, healthy, F.lookup("6204"), fr_hz=50.0,
                         window_s=5.0)
    v = A.verdict(res)
    assert v["passed"] is False
    assert v["contrast"] == pytest.approx(1.0, rel=1e-6)


def test_a_bearing_whose_bpfo_is_exactly_3x_shaft_speed_is_INCONCLUSIVE(sim_pair):
    """Not a contrivance for the test's sake — this is a bearing that could sit
    on a real bench. N = 8 with gamma = d/D = 0.25 gives

        BPFO = (N/2) fr (1 - gamma) = 4 fr x 0.75 = 3.00 fr

    exactly, at EVERY shaft speed. gamma = 0.25 and N = 8 are both squarely in
    the normal range for a deep-groove ball bearing, so such a bearing is
    permanently unmeasurable by this method: the outer-race line and the third
    shaft harmonic are the same line.

    The gate must return INCONCLUSIVE and say why. Changing the speed does NOT
    help here (the ratio is speed-independent) — which is the one case where the
    report's standard advice is wrong and the answer is to instrument a
    different bearing or use order tracking. Worth knowing before Week 2 rather
    than during it."""
    healthy, faulty = sim_pair
    geom = F.BearingGeometry(designation="contrived N=8, d/D=0.25",
                             n_elements=8, ball_diameter_mm=8.0,
                             pitch_diameter_mm=32.0, source="test",
                             confidence="published")
    assert geom.multipliers()["BPFO"] == pytest.approx(3.0, rel=1e-12)

    res = A.analyse_pair(healthy, faulty, geom, fr_hz=50.0, window_s=5.0)
    assert res["target_hz"] == pytest.approx(150.0, rel=1e-12)
    v = A.verdict(res)
    assert v["inconclusive"] is True and v["passed"] is False
    assert "not separable" in v["reason"]


def test_a_forced_band_is_used_verbatim(sim_pair):
    """--band overrides the protrugram. It is the first thing to try when the
    gate fails, so it must actually bypass band selection rather than being
    treated as a hint."""
    healthy, faulty = sim_pair
    res = A.analyse_pair(healthy, faulty, F.lookup("6204"), fr_hz=50.0,
                         window_s=5.0, band=(3000.0, 6000.0))
    assert res["band"] == (3000.0, 6000.0)
    assert math.isnan(res["band_crest"])


def test_rectify_and_hilbert_demodulators_agree(sim_pair):
    """features.py rectifies because Hilbert needs two full-length FFTs (~6x
    slower, and the Pi has a 500 ms budget). That choice is only free if the two
    demodulators give the same answer on a narrowband carrier.

    Assert the same verdict and the same peak location. The RATIOS are allowed
    to differ — the two demodulators have different noise floors — but if they
    disagreed about whether there is a fault, the Pi would be running a
    different experiment from the one we validated."""
    healthy, faulty = sim_pair
    kw = dict(fr_hz=50.0, window_s=5.0, band=(3865.0, 5420.0))
    r_rect = A.analyse_pair(healthy, faulty, F.lookup("6204"), method="rectify", **kw)
    r_hilb = A.analyse_pair(healthy, faulty, F.lookup("6204"), method="hilbert", **kw)
    assert A.verdict(r_rect)["passed"] == A.verdict(r_hilb)["passed"] is True
    assert r_hilb["env_faulty"]["peak_hz"] == pytest.approx(
        r_rect["env_faulty"]["peak_hz"], abs=0.5)


def test_unknown_demodulation_method_is_rejected(sim_pair):
    healthy, faulty = sim_pair
    with pytest.raises(ValueError):
        A.analyse_pair(healthy, faulty, F.lookup("6204"), fr_hz=50.0,
                       window_s=5.0, method="wavelet-magic")


# ===========================================================================
# 7. CLI contracts
# ===========================================================================
#
# Exit codes matter: these scripts get run from shell scripts and the Week-2
# session is a tired evening in a lab. 0 = pass, 1 = the gate failed, 2 = you
# gave me bad input.


def test_fault_frequencies_cli_prints_a_balanced_identity(capsys):
    assert F.main(["--bearing", "6204", "--rpm", "3000"]) == 0
    out = capsys.readouterr().out
    assert "400.000" in out          # 8 balls x 50 Hz
    assert "BPFO" in out and "152.597" in out


def test_fault_frequencies_cli_json_round_trips(capsys):
    import json
    assert F.main(["--bearing", "6205-2RS-JEM", "--fr", "29.95", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["n_elements"] == 9
    assert d["confidence"] == "published"
    assert d["frequencies_hz"]["BPFO"] + d["frequencies_hz"]["BPFI"] == \
        pytest.approx(9 * 29.95, rel=1e-12)


def test_fault_frequencies_cli_lists_the_table(capsys):
    assert F.main(["--list"]) == 0
    out = capsys.readouterr().out
    for key in ALL_KEYS:
        assert key in out


@pytest.mark.parametrize("argv", [
    ["--bearing", "6204"],                       # no speed
    ["--rpm", "2850"],                           # no geometry
    ["--bearing", "9999", "--rpm", "2850"],      # unknown bearing
    ["--bearing", "6204", "--rpm", "0"],         # zero speed
    ["--n", "9", "--ball-mm", "50", "--pitch-mm", "39", "--rpm", "1797"],
])
def test_fault_frequencies_cli_rejects_bad_input_with_code_2(argv, capsys):
    assert F.main(argv) == 2
    assert capsys.readouterr().err.startswith("error")


def test_analyse_recording_demo_runs_and_exits_zero(capsys):
    """--demo is the self-test: it builds a synthetic pair with ml/simulate.py
    and runs the whole chain. It must label itself synthetic in the output —
    a Week-2 report that could be mistaken for real data is worse than none."""
    assert A.main(["--demo", "--no-figure"]) == 0
    out = capsys.readouterr().out
    assert "DEMO MODE" in out and "NOT evidence about real machines" in out
    assert "WEEK-2 GATE: PASS" in out


@pytest.mark.parametrize("argv", [
    [],                                                    # no recordings
    ["--healthy", "nope.wav", "--faulty", "nope2.wav", "--bearing", "6204"],
    ["--healthy", "nope.wav", "--faulty", "nope2.wav", "--rpm", "2850"],
])
def test_analyse_recording_cli_rejects_bad_input_with_code_2(argv, capsys):
    assert A.main(argv) == 2
    assert "error" in capsys.readouterr().err


def test_analyse_recording_reports_a_missing_file_without_a_traceback(capsys):
    assert A.main(["--healthy", str(DATA / "does_not_exist.wav"),
                   "--faulty", str(DATA / "normal.wav"),
                   "--bearing", "6204", "--rpm", "3000"]) == 2
    err = capsys.readouterr().err
    assert "no such recording" in err
    assert "Traceback" not in err
