"""
bench_common.py — shared plumbing for the week-1 bring-up scripts.

Two jobs:

 1. A tiny PASS/FAIL reporter. Every check carries a measured NUMBER and the
    limit it was compared against. "Microphone OK" is an opinion; "achieved
    sample rate 15987 Hz (configured 16000, error 0.08 %, limit 1 %)" is a
    measurement. Only the second one survives contact with a bug.

 2. Turning missing hardware into a friendly, actionable message. These scripts
    are the first thing two students run on a Pi they have never used, at the
    point in the project where morale is most fragile. A Python traceback
    ending in `OSError: [Errno 2] No such file or directory: '/dev/spidev0.0'`
    is a correct diagnosis and a useless one.

Exit codes (used by selftest.py to build its summary):
    0  all checks passed
    1  ran fine, but at least one check FAILED   -> a real problem to fix
    2  hardware not present                      -> nothing to test yet
    3  the operator aborted (Ctrl-C)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np

# Make `firmware/` importable whether you run this as a script, a module, or
# from another directory. The bench tools deliberately import the SAME
# capture.py the firmware uses — a bring-up test that exercises a different
# code path from production tests the wrong thing.
FIRMWARE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = FIRMWARE_DIR.parent
for _p in (str(FIRMWARE_DIR), str(REPO_ROOT / "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXIT_OK, EXIT_FAIL, EXIT_NO_HARDWARE, EXIT_ABORT = 0, 1, 2, 3

WIDTH = 78


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class Report:
    """Collects PASS/FAIL lines and renders them at a fixed width.

    Deliberately dependency-free and deliberately boring: this output gets
    pasted into lab notebooks, commit messages and (eventually) RESULTS.md, so
    it must be greppable plain ASCII with no colour codes.
    """

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, str, str, str]] = []   # status, name, value, note
        self.notes: list[str] = []

    # -- emit ---------------------------------------------------------------

    def header(self) -> None:
        print("=" * WIDTH)
        print(self.title)
        print("=" * WIDTH)

    def info(self, text: str) -> None:
        print(f"     {text}")

    def section(self, text: str) -> None:
        print()
        print(f"--- {text} " + "-" * max(0, WIDTH - 5 - len(text)))

    def check(self, name: str, ok: bool | None, value: str, note: str = "") -> bool:
        """ok=None means "measured, not asserted" -> reported as INFO."""
        status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append((status, name, value, note))
        line = f"[{status}] {name:<34s} {value}"
        print(line)
        if note:
            for wrapped in textwrap.wrap(note, WIDTH - 7):
                print(f"       {wrapped}")
        return bool(ok) if ok is not None else True

    def advise(self, text: str) -> None:
        self.notes.append(text)

    # -- summarise ----------------------------------------------------------

    @property
    def failures(self) -> list[tuple[str, str, str, str]]:
        return [r for r in self.rows if r[0] == "FAIL"]

    @property
    def n_asserted(self) -> int:
        return len([r for r in self.rows if r[0] in ("PASS", "FAIL")])

    def finish(self) -> int:
        print()
        print("-" * WIDTH)
        n_fail = len(self.failures)
        n_pass = self.n_asserted - n_fail
        if n_fail == 0:
            print(f"RESULT: PASS  ({n_pass}/{self.n_asserted} checks)")
        else:
            print(f"RESULT: FAIL  ({n_pass}/{self.n_asserted} checks passed, "
                  f"{n_fail} failed)")
            for _, name, value, _ in self.failures:
                print(f"        - {name}: {value}")
        for note in self.notes:
            print()
            lines = textwrap.wrap(note, WIDTH - 6)
            for i, wrapped in enumerate(lines):
                print(f"NOTE: {wrapped}" if i == 0 else f"      {wrapped}")
        print("-" * WIDTH)
        return EXIT_OK if n_fail == 0 else EXIT_FAIL


# ---------------------------------------------------------------------------
# Friendly hardware-missing handling
# ---------------------------------------------------------------------------

def print_missing_hardware(what: str, message: str, remedy: str,
                           extra: str = "") -> None:
    """The whole point of this module. No traceback, ever."""
    print()
    print("=" * WIDTH)
    print(f"  HARDWARE NOT FOUND: {what}")
    print("=" * WIDTH)
    print()
    print("  What happened")
    for line in textwrap.wrap(message, WIDTH - 4):
        print(f"    {line}")
    print()
    print("  What to do")
    for line in textwrap.wrap(remedy or "See the hardware design notes (not in this public copy).", WIDTH - 4):
        print(f"    {line}")
    if extra:
        print()
        for line in extra.strip("\n").split("\n"):
            print(f"    {line}")
    print()
    print("  This is not a crash. Nothing is broken in the software — there is")
    print("  simply no sensor attached yet. To see what a good result looks")
    print("  like on synthetic data, re-run the same command with --simulate.")
    print("=" * WIDTH)


def run_guarded(fn, *args, **kwargs) -> int:
    """Call a bench main(); convert HardwareUnavailable / Ctrl-C into exit
    codes and friendly text. Anything else is a genuine bug and is allowed to
    raise, because hiding real bugs from students teaches the wrong lesson."""
    from capture import HardwareUnavailable
    try:
        return fn(*args, **kwargs)
    except HardwareUnavailable as e:
        print_missing_hardware(
            {"audio": "microphone (INMP441 over I2S)",
             "accel": "accelerometer (IIS3DWB over SPI)"}.get(e.what, e.what),
            str(e), e.remedy)
        return EXIT_NO_HARDWARE
    except KeyboardInterrupt:
        print("\n\nAborted by operator (Ctrl-C). Nothing was written.")
        return EXIT_ABORT


# ---------------------------------------------------------------------------
# Small shared DSP helpers
# ---------------------------------------------------------------------------

def dbfs(x: float) -> float:
    """Amplitude ratio -> dB relative to full scale (1.0). Audio people think
    in dB because the ear and the dynamic range of the sensor both span ~6
    decades; a linear RMS of 0.003 is meaningless at a glance, -50 dBFS is
    immediately recognisable as 'quiet but alive'."""
    return 20.0 * np.log10(max(abs(float(x)), 1e-12))


def dominant_peak(x: np.ndarray, fs: float, fmin: float = 20.0,
                  fmax: float | None = None) -> tuple[float, float]:
    """(frequency_hz, magnitude) of the largest FFT bin, refined by fitting a
    parabola through the peak bin and its two neighbours.

    WHY THE PARABOLIC FIT: an N-point FFT has bin spacing fs/N. For a 5 s
    record at 16 kHz that is 0.2 Hz, fine — but for a 1 s record it is 1 Hz,
    which is already 0.1 % of a 1 kHz tone and eats a tenth of our 1 % budget
    for free. Interpolating the log-magnitude parabola across three bins
    recovers the true peak to a small fraction of a bin for any smooth window
    function. This is standard practice and costs three multiplies.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    hi = fs / 2 if fmax is None else fmax
    sel = np.flatnonzero((freqs >= fmin) & (freqs <= hi))
    if len(sel) == 0:
        return 0.0, 0.0
    k = sel[int(np.argmax(spec[sel]))]
    if 0 < k < len(spec) - 1:
        a, b, c = (np.log(spec[k - 1] + 1e-20), np.log(spec[k] + 1e-20),
                   np.log(spec[k + 1] + 1e-20))
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-30 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    return float(freqs[k] + delta * df), float(spec[k])


def noise_floor_dbfs(x: np.ndarray, fs: float, frame: int = 1024) -> float:
    """Median short-term RMS, in dBFS.

    MEDIAN, not mean: the noise floor is 'what it sounds like when nothing is
    happening'. A single door slam or one tap of the housing would drag a mean
    upward by tens of dB. The median of per-frame RMS is the level exceeded
    half the time, which is exactly the quantity you want when asking 'is this
    microphone alive but quiet, or dead?'.
    """
    x = np.asarray(x, dtype=np.float64)
    n = (len(x) // frame) * frame
    if n < frame:
        return dbfs(np.sqrt(np.mean(x ** 2)))
    frames = x[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return dbfs(float(np.median(rms)))


def synth_room_tone(n: int, fs: float, rng, level: float = 0.01,
                    tone_hz: float | None = None,
                    tone_level: float = 0.2) -> np.ndarray:
    """Plausible fake microphone signal for --simulate: pink-ish broadband
    plus optional pure tone plus a small DC offset (real MEMS mics have one).

    This exists so the ANALYSIS code is exercised end-to-end without hardware.
    It is not a model of a machine — ml/simulate.py is that.
    """
    from scipy.signal import lfilter
    white = rng.standard_normal(n)
    # One-pole lowpass gives a -6 dB/octave tilt: closer to real room noise
    # than white, and it makes the noise-floor number look like a real one.
    a = 0.95
    pink = lfilter([1.0 - a], [1.0, -a], white)
    x = level * pink / (np.std(pink) + 1e-12)
    if tone_hz:
        t = np.arange(n) / fs
        x = x + tone_level * np.sin(2 * np.pi * tone_hz * t)
    return x + 0.002        # small DC offset, like a real MEMS mic
