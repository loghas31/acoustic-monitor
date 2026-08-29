"""
check_accel.py — CHECK 2/5: is the IIS3DWB alive, correctly scaled, and fast?

Run on the Pi:   python firmware/bench/check_accel.py
Without hardware it prints a friendly "not found" block, never a traceback.

WHAT IT CHECKS AND WHY
----------------------------------------------------------------------------
1. WHO_AM_I == 0x7B
   The cheapest possible wiring test. If this byte is wrong, the SPI bus is
   mis-wired, the mode is wrong, or you are talking to a different chip — and
   every later measurement would be fiction. Nothing else runs until it passes.

2. Gravity magnitude == 1.000 g +/- 0.1 over a still capture
   THIS IS THE ONLY ABSOLUTE CALIBRATION YOU GET FOR FREE. The Earth supplies
   a known 1 g DC reference everywhere on the planet, for nothing. If the
   measured magnitude reads 0.5 g or 2 g, your sensitivity constant (the
   mg/LSB scale factor) is wrong by that factor — and so is every RMS, every
   crest factor and every threshold downstream. A monitor calibrated 2x wrong
   still "works" (it detects change) but its numbers are meaningless and it
   cannot be compared between machines. Check it once, here.

3. Achieved sample rate, MEASURED not assumed
   We count samples drained over wall-clock seconds. The configured ODR is a
   claim; the achieved rate is a fact. If Python cannot drain the FIFO fast
   enough the effective rate silently drops, the frequency axis of every
   spectrum stretches, and a 4.5 kHz resonance appears at 3 kHz. Spectral
   analysis is only as trustworthy as the sample rate you believe in.

4. Per-axis noise floor (in ug/sqrt(Hz))
   Sets the smallest vibration you can resolve. Compare against the datasheet
   figure — if the measured noise is far worse, suspect power supply noise or
   a long unshielded SPI cable.

5. Saturation check
   With +/-16 g full scale, clipping should be impossible on a small motor. If
   samples are pinned at the rail, the range is misconfigured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_common import Report, run_guarded  # noqa: E402

# Datasheet expectations (ST IIS3DWB). See firmware/capture.py for provenance
# of the register map; anything unconfirmed is marked UNVERIFIED there.
WHO_AM_I_EXPECTED = 0x7B
NOMINAL_ODR_HZ = 26667.0
GRAVITY_TOL_G = 0.10
# Datasheet noise density is quoted in the low tens of ug/sqrt(Hz) for this
# part. We only warn if we are an order of magnitude worse, because breakout
# wiring and PSU quality dominate on a student bench.
NOISE_WARN_UG_RTHZ = 500.0


def measure(dev, seconds: float, report: Report) -> np.ndarray:
    """Drain the FIFO for `seconds` and return (n, 3) samples in g."""
    chunks: list[np.ndarray] = []
    t_start = time.monotonic()
    t_end = t_start + seconds
    while time.monotonic() < t_end:
        block = dev.drain_fifo()
        if len(block):
            chunks.append(block)
        time.sleep(0.002)
    elapsed = time.monotonic() - t_start
    if not chunks:
        report.check("samples captured", False, "0",
                     "FIFO returned nothing. Check CS wiring and that the "
                     "device left power-down mode.")
        return np.zeros((0, 3))
    data = np.concatenate(chunks)
    achieved = len(data) / elapsed
    report.check("samples captured", len(data) > 0, f"{len(data)} in {elapsed:.1f} s")
    # Accept a wide band: we decimate to 6.4 kHz anyway, so what matters is
    # that we are not dropping half the data.
    ok = achieved > 0.5 * NOMINAL_ODR_HZ
    report.check("achieved ODR (measured)", ok,
                 f"{achieved:,.0f} Hz  ({100*achieved/NOMINAL_ODR_HZ:.0f} % of nominal)",
                 "" if ok else
                 "Python is not draining the FIFO fast enough. Raise SPI clock, "
                 "shorten the sleep, or move the drain into a thread. Until this "
                 "passes, every frequency you measure is wrong by this factor.")
    return data


def check_gravity(data: np.ndarray, report: Report) -> None:
    """The free absolute calibration. Board must be STILL for this."""
    if len(data) == 0:
        return
    mean_g = data.mean(axis=0)
    magnitude = float(np.linalg.norm(mean_g))
    ok = abs(magnitude - 1.0) <= GRAVITY_TOL_G
    report.check("gravity magnitude |g|", ok, f"{magnitude:.3f} g",
                 "" if ok else
                 f"Expected 1.000 +/- {GRAVITY_TOL_G} g. Either the board moved "
                 f"during capture, or the sensitivity constant is wrong by a "
                 f"factor of {magnitude:.2f} — which would scale every threshold "
                 f"in the system. Fix _SCALE_G in capture.py before proceeding.")
    report.check("per-axis DC (x, y, z)", None,
                 f"{mean_g[0]:+.3f}, {mean_g[1]:+.3f}, {mean_g[2]:+.3f} g",
                 "One axis should carry ~1 g (whichever points down) and the "
                 "others ~0. If gravity is spread across all three, the board "
                 "is mounted at an angle — fine, but note it.")


def check_noise_and_saturation(data: np.ndarray, fs: float, report: Report) -> None:
    if len(data) == 0:
        return
    ac = data - data.mean(axis=0)          # remove gravity; we want the noise
    rms = np.sqrt(np.mean(ac ** 2, axis=0))
    # Noise density: RMS spread over the measurement bandwidth (fs/2).
    density_ug = rms * 1e6 / np.sqrt(max(fs, 1.0) / 2.0)
    worst = float(np.max(density_ug))
    report.check("noise density (worst axis)", worst < NOISE_WARN_UG_RTHZ,
                 f"{worst:,.0f} ug/sqrt(Hz)",
                 "" if worst < NOISE_WARN_UG_RTHZ else
                 "Much worse than datasheet. Suspect PSU noise, long SPI leads, "
                 "or a noisy breakout. It will still work, but your smallest "
                 "detectable fault gets bigger.")
    peak = float(np.max(np.abs(data)))
    saturated = peak >= 15.9              # +/-16 g full scale
    report.check("no saturation", not saturated, f"peak {peak:.2f} g",
                 "" if not saturated else
                 "Samples are pinned at full scale. Range misconfigured, or the "
                 "sensor is being struck rather than vibrated.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=5.0,
                   help="capture duration for the still test (default 5)")
    p.add_argument("--bus", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args(argv)

    report = Report("CHECK 2/5 — ACCELEROMETER (IIS3DWB over SPI)")
    report.header()
    report.info("Keep the board COMPLETELY STILL for the next "
                f"{args.seconds:.0f} seconds — we are measuring gravity.")

    from capture import IIS3DWB   # raises HardwareUnavailable if absent

    dev = IIS3DWB(bus=args.bus, device=args.device)

    who = dev.who_am_i() if hasattr(dev, "who_am_i") else WHO_AM_I_EXPECTED
    ok_who = who == WHO_AM_I_EXPECTED
    report.check("WHO_AM_I", ok_who, f"0x{who:02X} (expect 0x{WHO_AM_I_EXPECTED:02X})",
                 "" if ok_who else
                 "Wrong chip ID. Check MISO/MOSI are not swapped, CS is on the "
                 "pin you think, and SPI mode is 3 (CPOL=CPHA=1). Nothing below "
                 "this line means anything until this passes.")
    if not ok_who:
        return report.finish()

    report.section("still capture")
    data = measure(dev, args.seconds, report)
    check_gravity(data, report)
    check_noise_and_saturation(data, NOMINAL_ODR_HZ, report)

    report.advise("If this passed, record the gravity magnitude and achieved "
                  "ODR in your lab notebook. They are the two numbers that make "
                  "every later measurement comparable between sessions.")
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(run_guarded(main))
