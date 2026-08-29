"""
check_mount.py — CHECK 3/5: find the machine's real resonance by tapping it.

Run on the Pi, with the node stuck to the machine:
    python firmware/bench/check_mount.py --taps 5
Then tap the housing firmly with a screwdriver handle when prompted.

WHY THIS IS THE MOST SCIENTIFICALLY IMPORTANT SCRIPT IN THE BENCH SET
----------------------------------------------------------------------------
The whole detector rests on envelope demodulation: band-pass the frequency
region where bearing impacts ring, then look at the *rhythm* of that ringing.
Everything depends on choosing the right band.

Right now that band is a GUESS inherited from simulation (3-6 kHz, because
ml/simulate.py models a 4.5 kHz resonance). A real motor's end-shield has its
own resonances set by its geometry, mass and how you clamped the sensor on.
Guessing wrong means demodulating a band with no fault energy in it, and the
detector goes deaf while looking perfectly healthy.

An impulse (a tap) excites ALL resonances at once — that is what makes impulse
testing the standard technique in experimental modal analysis. The structure
answers by ringing at its own natural frequencies. So: hit it, record, and
read the answer off the spectrum. This is the cheapest possible modal test and
it takes thirty seconds.

The bearing impacts you are hunting will ring these SAME resonances, because
they are the same structure. So the peaks found here are exactly where the
fault energy will appear.

WHAT YOU GET
    - the dominant resonance frequency and its Q (sharpness)
    - a recommended demodulation band to put in config.yaml
    - a mounting-quality verdict from tap-to-tap repeatability: a solidly
      mounted sensor gives the same answer every time; a rocking magnet does
      not, and a rocking magnet will generate false alarms forever.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_common import Report, run_guarded  # noqa: E402

MIN_RESONANCE_HZ = 800.0     # below this we are in shaft-harmonic territory
REPEATABILITY_TOL = 0.15     # 15 % spread across taps = acceptable mounting


def find_taps(x: np.ndarray, fs: float, n_expected: int) -> list[tuple[int, int]]:
    """Locate impulse events: samples far above the running noise level.

    Threshold at 8x the median absolute value — median, not mean, so the taps
    themselves don't inflate the threshold that is meant to find them."""
    env = np.abs(x)
    thresh = 8.0 * float(np.median(env) + 1e-12)
    above = env > thresh
    if not above.any():
        return []
    edges = np.flatnonzero(np.diff(above.astype(int)) == 1)
    win = int(0.05 * fs)                       # 50 ms of ring-down per tap
    segments = []
    last_end = -1
    for e in edges:
        if e < last_end:                        # same tap, still ringing
            continue
        end = min(e + win, len(x))
        if end - e > 0.005 * fs:                # ignore specks
            segments.append((int(e), int(end)))
            last_end = end
    return segments[:n_expected * 2]


def analyse_ring(seg: np.ndarray, fs: float) -> tuple[float, float]:
    """Return (resonance_hz, q_estimate) for one tap's ring-down.

    Q from the -3 dB bandwidth of the dominant peak: Q = f0 / bandwidth.
    A high-Q resonance rings for a long time and is ideal for demodulation —
    it concentrates each impact's energy into a narrow, findable band."""
    w = np.hanning(len(seg))
    mag = np.abs(np.fft.rfft(seg * w))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
    band = freqs >= MIN_RESONANCE_HZ
    if not band.any():
        return 0.0, 0.0
    idx = np.flatnonzero(band)[np.argmax(mag[band])]
    f0 = float(freqs[idx])
    peak = mag[idx]
    half = peak / np.sqrt(2.0)                  # -3 dB points
    lo = idx
    while lo > 0 and mag[lo] > half:
        lo -= 1
    hi = idx
    while hi < len(mag) - 1 and mag[hi] > half:
        hi += 1
    bw = float(freqs[hi] - freqs[lo])
    q = f0 / bw if bw > 0 else 0.0
    return f0, q


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taps", type=int, default=5, help="how many taps to average")
    p.add_argument("--seconds", type=float, default=15.0, help="recording length")
    p.add_argument("--channel", choices=["audio", "accel"], default="audio",
                   help="which sensor to listen on (audio works mic-only)")
    p.add_argument("--fs", type=float, default=16000.0)
    args = p.parse_args(argv)

    report = Report("CHECK 3/5 — MOUNT & RESONANCE (tap test)")
    report.header()
    report.info(f"Recording {args.seconds:.0f} s. Tap the machine housing firmly "
                f"{args.taps} times, near the sensor, with a screwdriver handle.")
    report.info("Tap the METAL of the bearing housing — not the sensor itself.")

    if args.channel == "audio":
        from capture import HardwareSource
        src = HardwareSource(window_s=args.seconds, fs_audio=int(args.fs),
                             require_accel=False)
        audio, _ = next(iter(src.windows()))
        x, fs = np.asarray(audio, dtype=float), args.fs
    else:
        from capture import IIS3DWB
        dev = IIS3DWB()
        import time
        chunks, t_end = [], time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            b = dev.drain_fifo()
            if len(b):
                chunks.append(b)
            time.sleep(0.002)
        data = np.concatenate(chunks) if chunks else np.zeros((1, 3))
        x, fs = data[:, 0] - data[:, 0].mean(), 26667.0

    report.section("tap detection")
    segs = find_taps(x, fs, args.taps)
    ok_count = len(segs) >= max(3, args.taps // 2)
    report.check("taps detected", ok_count, f"{len(segs)}",
                 "" if ok_count else
                 "Too few clear impulses. Tap harder, closer to the sensor, and "
                 "make sure the machine is switched OFF during the tap test.")
    if not ok_count:
        return report.finish()

    results = [analyse_ring(x[a:b], fs) for a, b in segs]
    f0s = np.array([r[0] for r in results if r[0] > 0])
    qs = np.array([r[1] for r in results if r[0] > 0])
    if len(f0s) == 0:
        report.check("resonance found", False, "none")
        return report.finish()

    f0 = float(np.median(f0s))
    q = float(np.median(qs))
    spread = float(np.std(f0s) / (np.mean(f0s) + 1e-12))

    report.section("modal result")
    report.check("dominant resonance", f0 > MIN_RESONANCE_HZ, f"{f0:,.0f} Hz")
    report.check("Q (sharpness)", None, f"{q:.0f}",
                 "Higher Q = each impact rings longer and is easier to "
                 "demodulate. Q below ~5 means a heavily damped structure; "
                 "detection will be harder and you should lean on the mic.")
    ok_mount = spread <= REPEATABILITY_TOL
    report.check("mounting repeatability", ok_mount, f"{100*spread:.0f} % spread",
                 "" if ok_mount else
                 "The resonance moves between taps, which means the sensor is "
                 "rocking rather than rigidly coupled. Clean the surface, use a "
                 "thin smear of grease under the magnet, or move to a flatter "
                 "spot. A loose mount is a false-alarm generator.")

    # Recommend a band: centred on the resonance, wide enough to survive the
    # resonance shifting with temperature and load, narrow enough to exclude
    # the shaft harmonics at the bottom of the spectrum.
    lo = max(MIN_RESONANCE_HZ, f0 * 0.7)
    hi = min(f0 * 1.4, fs / 2 * 0.95)
    report.section("recommended configuration")
    report.check("demodulation band", None, f"{lo:,.0f} - {hi:,.0f} Hz")
    report.advise(
        f"Put this in firmware/config.yaml as the demodulation band, or pass it "
        f"to features.select_demodulation_band as a prior. The current default "
        f"(3000-6000 Hz) came from SIMULATION, not from your machine. If your "
        f"measured resonance is far from that range, the default would have "
        f"demodulated a band with no fault energy in it.")
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(run_guarded(main))
