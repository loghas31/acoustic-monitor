"""
fault_frequencies.py — bearing defect-frequency calculator.

WHY THIS EXISTS (and why it is not in firmware/)
------------------------------------------------
The shipping detector is deliberately GEOMETRY-FREE: it flags "this machine no
longer sounds like its own normal" without ever being told what bearing is
inside. That is a product decision (a gym manager cannot tell you a ball pitch
diameter) and firmware/features.py holds the line on it.

But in Week 2 *we* are not the customer — we are the experimenters. We bolted a
known bearing to a known motor and seeded a known defect. Knowing where the peak
*should* land turns a vague "there's a bump in the envelope spectrum" into a
falsifiable prediction: "there will be a peak at 107.4 Hz, and if there isn't,
the sensing chain is wrong." That is the difference between evidence and
wishful thinking, and it is the whole point of Gate 2.

So: this module is a MEASUREMENT AID, not part of the detector. Nothing in
firmware/ imports it.


THE PHYSICS
-----------
A rolling-element bearing is a small planetary gearbox. The inner ring (bore)
is clamped to the shaft and turns at the shaft rate fr. The outer ring sits in
the housing and does not turn. Between them, N rolling elements are held at
even spacing by the cage.

Assume PURE ROLLING (no sliding) at both contacts. The cage — and every ball
with it — orbits the shaft axis at the mean of the two ring surface speeds:

    FTF = (fr / 2) * (1 - (d/D) cos(phi))          [Fundamental Train Freq]

where
    d   = rolling-element (ball) diameter
    D   = pitch diameter (diameter of the circle through the ball centres)
    phi = contact angle (0 for a deep-groove ball bearing under pure radial
          load; ~15-40 deg for angular-contact / loaded thrust bearings)

The ratio (d/D) cos(phi) is the only geometry that matters. Call it `gamma`.

Now count impacts. A defect on the OUTER race is stationary. Each of the N
balls sweeps past it once per cage revolution:

    BPFO = N * FTF = (N/2) * fr * (1 - gamma)      [Ball Pass Freq, Outer]

A defect on the INNER race rides the shaft, so what matters is the ball train's
speed RELATIVE to the inner ring, (fr - FTF):

    BPFI = N * (fr - FTF) = (N/2) * fr * (1 + gamma)   [Ball Pass Freq, Inner]

A defect on a BALL strikes a race once per ball rotation about its own axis:

    BSF = (D / (2d)) * fr * (1 - gamma^2)          [Ball Spin Frequency]

NOTE the classic trap: a ball defect usually contacts BOTH races each ball
revolution, so the observed line is often at 2*BSF, not BSF. Several published
tables (including CWRU's) quote the 2x value under a "rolling element" heading.
We return the true BSF and expose `bsf_2x` separately rather than silently
picking one convention.


THE IDENTITY WORTH REMEMBERING
------------------------------
    BPFO + BPFI = (N/2) fr (1 - gamma) + (N/2) fr (1 + gamma) = N * fr

The gamma terms cancel exactly. So the two race frequencies always sum to
(number of balls) x (shaft speed), whatever the geometry. Two consequences:

  * It is a free correctness check on any geometry you look up or any table you
    copy. tests/test_realdata.py asserts it.
  * If you know N and fr you know the SUM, so measuring one race frequency
    pins the other. Handy when a table is missing.

Also useful: BPFO = N * FTF exactly, and BPFI - BPFO = N * fr * gamma.


WHICH INPUT ERRORS ACTUALLY MOVE THE PREDICTED PEAK
---------------------------------------------------
Differentiating BPFO = (N/2) fr (1 - gamma) with gamma = (d/D)cos(phi):

    dBPFO/BPFO  =  dN/N  +  dfr/fr  -  dgamma/(1 - gamma)

and since dgamma/gamma = dd/d - dD/D, a relative error in either diameter
enters BPFO scaled by gamma/(1 - gamma) ~ 0.26, and BPFI by gamma/(1 + gamma)
~ 0.17. So, worst first:

    N   sensitivity 1.0    miscount 8 balls as 9 -> 12.5 %, 19 Hz at fr = 50.
                           This is the killer. Count them through the seal gap.
    fr  sensitivity 1.0    an assumed nameplate rpm on a loaded induction motor
                           is easily 2-3 % out. MEASURE it.
    D   sensitivity 0.26   the (bore+OD)/2 estimate is 1.38 % out on a 6205 and
                           costs 0.358 % in BPFO — 0.64 Hz, inside the slip
                           window. Measured; see tests/test_realdata.py.
    d   sensitivity 0.26   likewise forgiving.

The moral is not intuitive and is worth stating plainly: guessing the pitch
diameter is nearly free, guessing the ball count or the shaft speed is not.


WHAT WILL BITE YOU ON REAL DATA
-------------------------------
1. SLIP. Pure rolling is an idealisation. Real bearings slide by ~1-2 %, so the
   observed peak sits a bit BELOW the computed BPFO and wanders. Never search
   for the peak in a +/-0.1 Hz bin; use a +/-1-2 % tolerance window.
   (analyse_recording.py does exactly this, --tol-pct.)
2. fr IS NOT THE NAMEPLATE RPM. An induction motor slips under load. CWRU's
   "1750 rpm" nameplate cases actually run at 1750 rpm *because* they are
   loaded — the no-load case is 1797 rpm. Measure fr, don't assume it.
3. BPFO IS OFTEN NEAR A SHAFT HARMONIC. With the repo's synthetic default,
   BPFO = 152.6 Hz sits 2.6 Hz from the 3rd shaft harmonic at 150 Hz. Naive
   peak-picking in a raw spectrum confuses them. This is a big part of why the
   project demodulates instead.
4. THE PEAK IS IN THE ENVELOPE SPECTRUM, NOT THE RAW ONE. An early defect puts
   almost no energy at BPFO itself; it puts energy at the housing resonance
   (kHz), amplitude-modulated at BPFO. See ml/verify_signals.py.


CLI
---
    python ml/realdata/fault_frequencies.py --bearing 6204 --rpm 2850
    python ml/realdata/fault_frequencies.py --list
    python ml/realdata/fault_frequencies.py --n 9 --ball-mm 7.94 --pitch-mm 39.04 --rpm 1797
    python ml/realdata/fault_frequencies.py --bore-mm 25 --od-mm 52 --n 9 --rpm 1797
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BearingGeometry:
    """Everything the defect-frequency formulae need.

    Units are millimetres and degrees. Only the RATIO d/D matters, so any
    consistent length unit works — mm is used because that is how bearing
    catalogues are written outside the USA.

    `source` and `confidence` are carried with the numbers on purpose. A
    geometry you guessed and a geometry you read off the manufacturer's
    drawing predict different peak locations, and in Week 2 you need to know
    which of the two you are betting the go/no-go gate on.

    confidence:
      "published"  — N, d and D all come from a citable authority.
      "estimated"  — at least one number is inferred (typically the pitch
                     diameter from boundary dimensions, or the ball count).
                     Treat the predicted frequency as +/- a few percent and
                     widen your search window accordingly.
    """

    designation: str
    n_elements: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0
    source: str = ""
    confidence: str = "estimated"

    # -- the single geometric ratio everything depends on ---------------------
    @property
    def gamma(self) -> float:
        """gamma = (d/D) * cos(phi). Typically 0.20-0.25 for deep-groove ball
        bearings. If you compute a gamma outside ~0.10-0.35 for a ball bearing,
        you have mixed up a diameter with a radius, or mm with inches."""
        return (self.ball_diameter_mm / self.pitch_diameter_mm) * math.cos(
            math.radians(self.contact_angle_deg)
        )

    # -- the four defect frequencies, all in Hz for fr in Hz -------------------
    def ftf(self, fr_hz: float) -> float:
        """Fundamental train (cage) frequency. Always < fr/2."""
        return 0.5 * fr_hz * (1.0 - self.gamma)

    def bpfo(self, fr_hz: float) -> float:
        """Ball pass frequency, outer race = N * FTF."""
        return 0.5 * self.n_elements * fr_hz * (1.0 - self.gamma)

    def bpfi(self, fr_hz: float) -> float:
        """Ball pass frequency, inner race = N * (fr - FTF)."""
        return 0.5 * self.n_elements * fr_hz * (1.0 + self.gamma)

    def bsf(self, fr_hz: float) -> float:
        """Ball spin frequency: rotation rate of a ball about its own axis.
        A ball defect typically shows at 2*BSF (it hits both races) — see
        `bsf_2x`."""
        ratio = self.pitch_diameter_mm / (2.0 * self.ball_diameter_mm)
        return ratio * fr_hz * (1.0 - self.gamma**2)

    def bsf_2x(self, fr_hz: float) -> float:
        """The line a ball defect usually produces. This is what CWRU's table
        calls 'Rolling Element'."""
        return 2.0 * self.bsf(fr_hz)

    def all_frequencies(self, fr_hz: float) -> dict[str, float]:
        return {
            "fr": fr_hz,
            "FTF": self.ftf(fr_hz),
            "BPFO": self.bpfo(fr_hz),
            "BPFI": self.bpfi(fr_hz),
            "BSF": self.bsf(fr_hz),
            "BSF_2x": self.bsf_2x(fr_hz),
        }

    def multipliers(self) -> dict[str, float]:
        """Defect frequencies expressed as multiples of running speed — the
        form bearing catalogues and the CWRU site publish, because they are
        speed-independent."""
        return {k: v for k, v in self.all_frequencies(1.0).items() if k != "fr"}


# ----------------------------------------------------------------------------
# Lookup table
# ----------------------------------------------------------------------------
#
# Every entry cites where its numbers came from. Read the `confidence` field
# before you trust a prediction to within a few percent.
#
# On the two CWRU entries: these are transcribed from the data centre's own
# "Bearing Information" page, which publishes ball and pitch diameter in inches
# AND the resulting defect-frequency multipliers. That redundancy lets us
# verify our formulae against an independent authority — see
# tests/test_realdata.py, which reproduces CWRU's published multipliers to
# 4 decimal places from the geometry alone. Note CWRU does NOT publish the ball
# count; we recover N = 9 (and N = 8 for the 6203) by inverting
# BPFO_multiplier = (N/2)(1 - gamma), which lands on an integer to 4 d.p. That
# is a satisfying consistency check in its own right.
#
# On the "estimated" entries: the pitch diameter of a deep-groove ball bearing
# is very close to the mean of its boundary diameters, D ~ (bore + OD)/2, since
# the ball centres sit near the mid-plane of the annulus. That rule reproduces
# the CWRU 6203 pitch diameter EXACTLY (28.50 mm) and the 6205 to within 1.4 %
# (38.50 mm estimated vs 39.04 mm published). So: good enough to know which
# 10 Hz to look in, not good enough to claim a 0.5 Hz match. Ball counts and
# ball diameters for the estimated entries follow the common ISO 15:2017
# 62-series values quoted across bearing catalogues and vibration-analysis
# tables; a student should confirm against the actual bearing where possible
# (count the balls through the seal gap — it takes ten seconds and removes all
# doubt about N).

BEARINGS: dict[str, BearingGeometry] = {
    # ---- CWRU test rig (authoritative; used by validate_public_dataset.py) --
    "6205-2RS-JEM": BearingGeometry(
        designation="6205-2RS JEM SKF (CWRU drive end)",
        n_elements=9,
        ball_diameter_mm=0.3126 * 25.4,   # 7.940 mm  (published in inches)
        pitch_diameter_mm=1.537 * 25.4,   # 39.040 mm (published in inches)
        contact_angle_deg=0.0,
        # Source: CWRU Bearing Data Center, "Bearing Information",
        # https://engineering.case.edu/bearingdatacenter/bearing-information
        # Published size table: ID 0.9843", OD 2.0472", thickness 0.5906",
        # ball dia 0.3126", pitch dia 1.537".
        # Published multipliers: inner 5.4152, outer 3.5848, cage 0.39828,
        # rolling element 4.7135 (= 2*BSF).
        # N=9 recovered from 3.5848 = (N/2)(1 - 0.3126/1.537).
        source="CWRU Bearing Data Center, Bearing Information page",
        confidence="published",
    ),
    "6203-2RS-JEM": BearingGeometry(
        designation="6203-2RS JEM SKF (CWRU fan end)",
        n_elements=8,
        ball_diameter_mm=0.2656 * 25.4,   # 6.746 mm
        pitch_diameter_mm=1.122 * 25.4,   # 28.499 mm
        contact_angle_deg=0.0,
        # Source: CWRU Bearing Data Center, "Bearing Information" (same page).
        # Published multipliers: inner 4.9469, outer 3.0530, cage 0.3817,
        # rolling element 3.9874. N=8 recovered from the outer multiplier.
        source="CWRU Bearing Data Center, Bearing Information page",
        confidence="published",
    ),
    # ---- common small-motor bearings (62-series, deep groove) ---------------
    "6202": BearingGeometry(
        designation="6202 deep-groove ball (15x35x11 mm)",
        n_elements=8,
        ball_diameter_mm=5.953,           # 0.2344 in = 15/64"
        pitch_diameter_mm=25.0,           # (15 + 35)/2
        contact_angle_deg=0.0,
        source="boundary dims ISO 15 (15/35/11); D from (bore+OD)/2; "
               "d = common 62-series value. VERIFY N by eye.",
        confidence="estimated",
    ),
    "6203": BearingGeometry(
        designation="6203 deep-groove ball (17x40x12 mm)",
        n_elements=8,
        ball_diameter_mm=6.746,
        pitch_diameter_mm=28.5,           # (17 + 40)/2 — matches CWRU exactly
        contact_angle_deg=0.0,
        source="boundary dims ISO 15 (17/40/12); d and N cross-checked against "
               "the CWRU 6203-2RS JEM entry above, which agrees to <0.01 mm.",
        confidence="published",
    ),
    "6204": BearingGeometry(
        designation="6204 deep-groove ball (20x47x14 mm)",
        n_elements=8,
        ball_diameter_mm=7.94,            # 5/16"
        pitch_diameter_mm=33.5,           # (20 + 47)/2
        contact_angle_deg=0.0,
        source="boundary dims ISO 15 (20/47/14); D from (bore+OD)/2. These are "
               "the same numbers used as the default in ml/simulate.py, so the "
               "synthetic BPFO of 152.6 Hz at fr=50 Hz is reproducible here.",
        confidence="estimated",
    ),
    "6205": BearingGeometry(
        designation="6205 deep-groove ball (25x52x15 mm)",
        n_elements=9,
        ball_diameter_mm=7.94,
        pitch_diameter_mm=38.5,           # (25 + 52)/2
        contact_angle_deg=0.0,
        source="boundary dims ISO 15 (25/52/15); D from (bore+OD)/2. Compare "
               "with the '6205-2RS-JEM' entry (published D = 39.04 mm) to see "
               "how much the estimate costs you: 1.38 % in D but only 0.358 % "
               "in BPFO (0.64 Hz at fr = 50 Hz). Measured 2026-08-17 in "
               "tests/test_realdata.py, which corrected an earlier claim of "
               "'~1.4 % in BPFO' here. The error is attenuated because BPFO "
               "carries (1 - gamma), giving sensitivity gamma/(1 - gamma) = "
               "0.255, not 1. See the note under THE IDENTITY above.",
        confidence="estimated",
    ),
    "6206": BearingGeometry(
        designation="6206 deep-groove ball (30x62x16 mm)",
        n_elements=9,
        ball_diameter_mm=9.525,           # 3/8"
        pitch_diameter_mm=46.0,           # (30 + 62)/2
        contact_angle_deg=0.0,
        source="boundary dims ISO 15 (30/62/16); D from (bore+OD)/2; "
               "d = common 62-series value. VERIFY N by eye.",
        confidence="estimated",
    ),
    "6208": BearingGeometry(
        designation="6208 deep-groove ball (40x80x18 mm)",
        n_elements=9,
        ball_diameter_mm=11.906,          # 15/32"
        pitch_diameter_mm=60.0,           # (40 + 80)/2
        contact_angle_deg=0.0,
        source="boundary dims ISO 15 (40/80/18); D from (bore+OD)/2; "
               "d = common 62-series value. VERIFY N by eye.",
        confidence="estimated",
    ),
}

# Friendly aliases so `--bearing 6205-CWRU` and similar do the obvious thing.
_ALIASES = {
    "6205-CWRU": "6205-2RS-JEM",
    "6205-2RS": "6205-2RS-JEM",
    "6203-CWRU": "6203-2RS-JEM",
    "6203-2RS": "6203-2RS-JEM",
}


def lookup(designation: str) -> BearingGeometry:
    """Case-insensitive lookup. Raises KeyError with a helpful message."""
    key = designation.strip().upper()
    key = _ALIASES.get(key, key)
    if key in BEARINGS:
        return BEARINGS[key]
    raise KeyError(
        f"bearing {designation!r} is not in the table. "
        f"Known: {', '.join(sorted(BEARINGS))}. "
        f"Supply geometry explicitly with --n/--ball-mm/--pitch-mm, or "
        f"approximate it with --bore-mm/--od-mm/--n."
    )


def from_boundary_dimensions(bore_mm: float, od_mm: float, n_elements: int,
                             ball_diameter_mm: float | None = None,
                             contact_angle_deg: float = 0.0,
                             designation: str = "custom") -> BearingGeometry:
    """Build an APPROXIMATE geometry from the numbers stamped on the bearing.

    Every bearing has its boundary dimensions printed on the seal or shown in
    any catalogue: bore x outside diameter x width. The pitch diameter is very
    nearly the mean of the first two, because the ball centres lie near the
    mid-plane of the annulus:

        D ~ (bore + OD) / 2

    If you do not know the ball diameter either, a serviceable rule for a
    deep-groove ball bearing is that the balls fill roughly 60 % of the radial
    gap between the rings:

        d ~ 0.6 * (OD - bore) / 2

    Both rules are good to a few percent, which is good enough to know which
    part of the envelope spectrum to look at — and NOT good enough to claim a
    precise match. Count the balls if you possibly can: N is the one parameter
    the rules cannot guess, and BPFO scales linearly with it.
    """
    if bore_mm <= 0 or od_mm <= bore_mm:
        raise ValueError("need 0 < bore_mm < od_mm")
    if n_elements < 3:
        raise ValueError("n_elements must be >= 3")
    pitch = 0.5 * (bore_mm + od_mm)
    ball = ball_diameter_mm if ball_diameter_mm else 0.6 * 0.5 * (od_mm - bore_mm)
    return BearingGeometry(
        designation=designation,
        n_elements=n_elements,
        ball_diameter_mm=ball,
        pitch_diameter_mm=pitch,
        contact_angle_deg=contact_angle_deg,
        source=f"approximated from boundary dimensions {bore_mm:g}x{od_mm:g} mm",
        confidence="estimated",
    )


def rpm_to_hz(rpm: float) -> float:
    """Shaft speed: revolutions per minute -> revolutions per second (Hz).

    The single most common unit error in this whole subject. 2850 rpm is
    47.5 Hz, not 2850 Hz. Every formula above wants Hz."""
    return rpm / 60.0


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def format_report(geom: BearingGeometry, fr_hz: float, tol_pct: float = 2.0,
                  n_harmonics: int = 3) -> str:
    """Human-readable block: the frequencies, the search windows to use, and
    the sanity identity."""
    f = geom.all_frequencies(fr_hz)
    m = geom.multipliers()
    out: list[str] = []
    a = out.append

    a(f"Bearing      : {geom.designation}")
    a(f"Source       : {geom.source}")
    a(f"Confidence   : {geom.confidence.upper()}"
      + ("   <-- predicted frequencies are +/- a few percent; widen your "
         "search window" if geom.confidence != "published" else ""))
    a("")
    a(f"Geometry     : N = {geom.n_elements} elements, "
      f"d = {geom.ball_diameter_mm:.3f} mm, D = {geom.pitch_diameter_mm:.3f} mm, "
      f"contact angle = {geom.contact_angle_deg:g} deg")
    a(f"               gamma = (d/D)cos(phi) = {geom.gamma:.5f}")
    a("")
    a(f"Shaft speed  : {fr_hz:.4f} Hz  ({fr_hz * 60:.1f} rpm)")
    a("")
    a("  Defect frequency        multiple of fr        Hz")
    a("  " + "-" * 52)
    for name, label in (("BPFO", "BPFO  outer race"),
                        ("BPFI", "BPFI  inner race"),
                        ("BSF", "BSF   ball spin"),
                        ("BSF_2x", "2xBSF ball defect line"),
                        ("FTF", "FTF   cage")):
        a(f"  {label:<24}{m[name]:>10.4f}   {f[name]:>12.3f}")
    a("")

    # The identity, shown rather than asserted — it is the check the student
    # should be able to do in their head.
    a(f"  Check: BPFO + BPFI = {f['BPFO']:.3f} + {f['BPFI']:.3f} = "
      f"{f['BPFO'] + f['BPFI']:.3f} Hz")
    a(f"         N * fr      = {geom.n_elements} * {fr_hz:.4f} = "
      f"{geom.n_elements * fr_hz:.3f} Hz   (must match: the gamma terms cancel)")
    a("")

    # Search windows: what analyse_recording.py will actually hunt in.
    a(f"  Where to look in the ENVELOPE spectrum (+/-{tol_pct:g} % for slip):")
    for name in ("BPFO", "BPFI"):
        base = f[name]
        for h in range(1, n_harmonics + 1):
            centre = h * base
            half = centre * tol_pct / 100.0
            a(f"    {name} x{h}: {centre:8.2f} Hz   window "
              f"[{centre - half:7.2f}, {centre + half:7.2f}] Hz")
    a("")
    a("  Reminder: these are ENVELOPE-spectrum locations. An early defect puts")
    a("  almost no energy at BPFO in the RAW spectrum — it modulates a kHz")
    a("  housing resonance at BPFO. Demodulate first (ml/verify_signals.py).")
    return "\n".join(out)


def format_table() -> str:
    """`--list`: every known bearing and its speed-independent multipliers."""
    out = ["Known bearing geometries (multipliers are per Hz of shaft speed):", ""]
    out.append(f"  {'key':<16}{'N':>3}  {'d mm':>7} {'D mm':>7}  "
               f"{'BPFO':>8}{'BPFI':>8}{'BSF':>8}{'FTF':>8}  confidence")
    out.append("  " + "-" * 90)
    for key in sorted(BEARINGS):
        g = BEARINGS[key]
        m = g.multipliers()
        out.append(
            f"  {key:<16}{g.n_elements:>3}  {g.ball_diameter_mm:>7.3f} "
            f"{g.pitch_diameter_mm:>7.3f}  {m['BPFO']:>8.4f}{m['BPFI']:>8.4f}"
            f"{m['BSF']:>8.4f}{m['FTF']:>8.4f}  {g.confidence}"
        )
    out.append("")
    out.append("  Multiply any column by the shaft speed IN HZ (rpm/60) to get Hz.")
    out.append("  'estimated' rows infer pitch diameter from boundary dimensions;")
    out.append("  see each entry's `source` field in the code for the details.")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fault_frequencies.py",
        description="Compute bearing defect frequencies (BPFO/BPFI/BSF/FTF) "
                    "from a bearing designation or explicit geometry.",
        epilog=(
            "examples:\n"
            "  %(prog)s --list\n"
            "  %(prog)s --bearing 6204 --rpm 2850\n"
            "  %(prog)s --bearing 6205-2RS-JEM --rpm 1797\n"
            "  %(prog)s --n 9 --ball-mm 7.94 --pitch-mm 39.04 --rpm 1797\n"
            "  %(prog)s --bore-mm 25 --od-mm 52 --n 9 --rpm 1797\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true",
                   help="list the built-in bearing table and exit")

    g = p.add_argument_group("bearing selection (choose ONE route)")
    g.add_argument("--bearing", metavar="DESIGNATION",
                   help="look up a known bearing, e.g. 6204, 6205-2RS-JEM")
    g.add_argument("--n", "--n-elements", dest="n_elements", type=int,
                   metavar="N", help="number of rolling elements")
    g.add_argument("--ball-mm", type=float, metavar="D",
                   help="rolling-element (ball) diameter, mm")
    g.add_argument("--pitch-mm", type=float, metavar="D",
                   help="pitch diameter, mm")
    g.add_argument("--bore-mm", type=float, metavar="D",
                   help="bore (inner) diameter, mm — used with --od-mm to "
                        "APPROXIMATE the pitch diameter")
    g.add_argument("--od-mm", type=float, metavar="D",
                   help="outside diameter, mm")
    g.add_argument("--contact-angle", type=float, default=0.0, metavar="DEG",
                   help="contact angle in degrees (default 0, correct for a "
                        "radially loaded deep-groove ball bearing)")

    s = p.add_argument_group("operating point")
    s.add_argument("--rpm", type=float, metavar="RPM",
                   help="shaft speed in revolutions per minute")
    s.add_argument("--fr", type=float, metavar="HZ",
                   help="shaft speed in Hz (alternative to --rpm)")

    o = p.add_argument_group("output")
    o.add_argument("--tol-pct", type=float, default=2.0, metavar="PCT",
                   help="slip tolerance for the printed search windows "
                        "(default 2.0 %%)")
    o.add_argument("--harmonics", type=int, default=3, metavar="K",
                   help="how many harmonics of BPFO/BPFI to print (default 3)")
    o.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON instead of the report")
    return p


def _resolve_geometry(args) -> BearingGeometry:
    """Turn CLI arguments into one BearingGeometry, or raise ValueError with a
    message a human can act on."""
    if args.bearing:
        return lookup(args.bearing)

    if args.n_elements and args.ball_mm and args.pitch_mm:
        if args.ball_mm >= args.pitch_mm:
            raise ValueError("--ball-mm must be smaller than --pitch-mm "
                             "(a ball has to fit inside the pitch circle)")
        return BearingGeometry(
            designation="custom (explicit geometry)",
            n_elements=args.n_elements,
            ball_diameter_mm=args.ball_mm,
            pitch_diameter_mm=args.pitch_mm,
            contact_angle_deg=args.contact_angle,
            source="supplied on the command line",
            confidence="published",
        )

    if args.bore_mm and args.od_mm and args.n_elements:
        return from_boundary_dimensions(
            args.bore_mm, args.od_mm, args.n_elements,
            ball_diameter_mm=args.ball_mm,
            contact_angle_deg=args.contact_angle,
        )

    raise ValueError(
        "not enough geometry. Use one of:\n"
        "  --bearing 6204                              (table lookup)\n"
        "  --n 9 --ball-mm 7.94 --pitch-mm 39.04       (exact geometry)\n"
        "  --bore-mm 25 --od-mm 52 --n 9               (approximate)\n"
        "Run with --list to see the built-in table."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(format_table())
        return 0

    try:
        geom = _resolve_geometry(args)
    except (ValueError, KeyError) as e:
        # No raw traceback: this is a user-input problem, not a crash.
        print(f"error: {e}".replace("\\n", "\n"), file=sys.stderr)
        return 2

    if args.rpm is None and args.fr is None:
        print("error: give a shaft speed with --rpm (or --fr in Hz).",
              file=sys.stderr)
        return 2
    fr_hz = args.fr if args.fr is not None else rpm_to_hz(args.rpm)
    if fr_hz <= 0:
        print("error: shaft speed must be positive.", file=sys.stderr)
        return 2
    if fr_hz > 1000:
        print(f"warning: fr = {fr_hz:.1f} Hz is {fr_hz * 60:.0f} rpm — did you "
              f"pass rpm to --fr by mistake?", file=sys.stderr)

    if args.json:
        import json
        print(json.dumps({
            "bearing": geom.designation,
            "source": geom.source,
            "confidence": geom.confidence,
            "n_elements": geom.n_elements,
            "ball_diameter_mm": geom.ball_diameter_mm,
            "pitch_diameter_mm": geom.pitch_diameter_mm,
            "contact_angle_deg": geom.contact_angle_deg,
            "gamma": geom.gamma,
            "fr_hz": fr_hz,
            "frequencies_hz": geom.all_frequencies(fr_hz),
            "multipliers": geom.multipliers(),
        }, indent=2))
    else:
        print(format_report(geom, fr_hz, args.tol_pct, args.harmonics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
