"""
capture.py — signal acquisition (v2). One interface, three sources:

    SimulatedSource — wraps ml/simulate.py live (demo, CI, development)
    FileSource      — replays exported .wav/.csv recordings
    HardwareSource  — INMP441 (I2S/ALSA) + IIS3DWB (SPI)

Every source yields (audio, accel) windows: audio (n,) float at fs_audio,
accel (m, 3) float at fs_accel.

Sensor change from v1: ADXL345 is GONE. Its ~800 Hz usable bandwidth cannot
see the 1–20 kHz resonance bands where bearing-impact energy lives
(ml/verify_signals.py fig1 is the proof). The IIS3DWB is ST's
vibration-monitoring part: ±16 g, 6.3 kHz flat band.

HARDWARE STATUS: the IIS3DWB driver below follows the datasheet register map
but has NOT touched a bench. Verify WHO_AM_I, SPI mode, and the FIFO drain
rate with a logic analyser before trusting field data. Marked TODO(bench).

Bring-up tooling lives in `firmware/bench/` — run those FIRST on new hardware:
    python firmware/bench/selftest.py
They measure (not assume) sample rate, gravity magnitude, WHO_AM_I and the
structural resonance. Nothing in this file should be trusted on a given board
until `bench/selftest.py` passes on that board.

DEGRADED MODE: `HardwareSource(require_accel=False)` (the default) runs
mic-only if the IIS3DWB is missing or mis-wired. the execution plan (not in this public copy) week 1
explicitly allows this — the microphone channel alone carries the resonance
band, so you lose the accel cross-check, not the physics. See the long comment
on `HardwareSource._open_accel` for why we emit exact zeros rather than fake
noise, and why that is numerically safe downstream.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ml"))


def default_schedule(i: int) -> dict:
    return {"kind": "normal", "severity": 0.0, "fr": 50.0}


# ----------------------------------------------------------------------------
# The simulated triaxial accelerometer (T1.8, self-review finding F6)
# ----------------------------------------------------------------------------
#
# WHAT WAS WRONG
# --------------
# Until 2026-08-18 this file built the accelerometer as
#
#       accel = [ax, 0.6*ax + noise, 0.35*ax + noise]
#
# — axes y and z were scaled copies of x. Measured inter-axis correlation was
# r(x,y) = 0.9988, r(x,z) = 0.9964, r(y,z) = 0.9952, and the 12 per-axis accel
# statistics in the feature vector spanned an effective rank of **3.75 of 12**,
# with the smallest four singular values at 1.3e-3 of the largest. In other
# words 12 features carried about 4 features of information, and the
# accelerometer half of the sensor had never actually been tested: nothing
# `accel_y_kurt` reports can disagree with `accel_x_kurt` when y = 0.6*x.
#
# WHAT REPLACES IT
# ----------------
# Three axes that share a mechanical *cause* and differ in their *path*:
#
#   * ONE shaft and ONE defect, so the impulse train is generated once and
#     shared. A ball rolling over a spall excites all three axes within about
#     0.1 ms — far less than one sample at 6.4 kHz — so the impacts really are
#     simultaneous, and pretending otherwise would be a different lie.
#
#   * Each axis then sees that impact through its own structural transfer
#     function. The housing has different mode shapes in different directions:
#     the radial mode aligned with the load zone is the stiffest and least
#     damped; the orthogonal radial mode is lower and softer; the axial path
#     runs through a bolted end shield, which is softer and much more damped
#     again. Modelled as a 2nd-order resonance per axis with its own f0, Q and
#     gain (the same `_resonance_filter` the frozen simulator uses).
#
#   * Each axis sees the rotating imbalance vector at its own PHASE. The radial
#     pair is 90 degrees apart on the housing, so y lags x by a quarter turn.
#     The axial response is not a projection of the radial force at all — it
#     comes from rocking of the rotor on its bearings, through a different mode
#     with its own phase — so z is given a third, non-quadrature angle. This
#     matters more than it sounds: on a HEALTHY machine the shaft hum is 20 dB
#     above everything else, so if all three axes saw it in phase they would
#     still correlate at 0.90 no matter how different the resonances were
#     (measured, before the phase term was added). Implemented with an exact
#     Hilbert quadrature, which rotates the 1x, 2x and 3x harmonics together —
#     a fixed time delay could not.
#
#   * Sensor self-noise is drawn INDEPENDENTLY per axis. This is the term that
#     genuinely decorrelates them at high frequency, and it is physically the
#     honest one: three sensing elements have three noise processes.
#
#   * A few per cent of transverse (cross-axis) sensitivity is added, because a
#     real MEMS triaxial part has it and a model with exactly zero cross-talk
#     would be its own kind of fiction.
#
# DELIBERATE ASYMMETRY, so you are not surprised by it
# ----------------------------------------------------
# The cross-axis mixing is LOWER TRIANGULAR: x leaks into y and z, but y and z
# do not leak back into x. That keeps axis 0 bit-identical to the old
# single-axis signal, which is worth more than the missing term: `features.py`
# computes the accelerometer band-ILR block and `estimate_fr` from channel 0
# ONLY, so an unchanged channel 0 proves those are unaffected and makes any
# change in detection attributable solely to the eight y/z statistics.
# The omitted feedback term would be `cross_axis` (3 %) of a signal already
# ~3x smaller than x, i.e. under 1 % of axis 0 — smaller than the sensor noise
# that was there before. Pinned by
# `tests/test_accel_axes.py::test_audio_and_axis_x_are_bit_identical_to_the_old_simulator`.
#
# HONEST LIMITS
# -------------
# These constants are a plausible structure, not a measurement. Nobody has put
# a triaxial accelerometer on a motor for this project. What the change buys is
# that the accelerometer features can now *differ from each other*, so a claim
# about them is capable of being wrong; it does not make the simulator true.
# The first real recording (H2/H3) should replace these numbers with measured
# ones — `firmware/bench/check_mount.py`'s tap test gives f0 and Q per axis
# directly.
ACCEL_AXES = {
    # gain: how much of the radial impact energy reaches this sensing element
    # f0_ratio / q_ratio: this axis's housing mode, relative to the nominal
    #   resonance in SimConfig (applied AFTER the Nyquist clamp, so the axes
    #   stay distinct at any sample rate)
    # hum_gain: response to the rotating imbalance vector
    # hum_phase_deg: where this axis stands relative to that rotating vector.
    #   0 = the reference radial axis; 90 = the orthogonal radial axis; the
    #   axial value is neither, because axial motion is rocking, not a
    #   projection of the radial force.
    "x": dict(gain=1.00, f0_ratio=1.00, q_ratio=1.00, hum_gain=1.00, hum_phase_deg=0.0),
    "y": dict(gain=0.55, f0_ratio=0.74, q_ratio=0.60, hum_gain=0.72, hum_phase_deg=90.0),
    "z": dict(gain=0.22, f0_ratio=0.45, q_ratio=0.30, hum_gain=0.28, hum_phase_deg=145.0),
    "cross_axis": 0.03,          # transverse sensitivity, x -> y and x -> z
}


def _quadrature(x: np.ndarray) -> np.ndarray:
    """Exact 90 degree phase lag at every frequency present.

    `imag(hilbert(x))` is the Hilbert transform, which maps sin(t) -> -cos(t),
    i.e. a quarter-cycle lag, and does so for the 1x, 2x and 3x shaft
    harmonics at once — which a fixed time delay could not. The FFT behind it
    assumes periodicity, so the first and last few samples carry a small edge
    artefact; over a 30 s window at 50 Hz that is 2 edges against 1500 cycles.
    """
    from scipy.signal import hilbert
    return np.imag(hilbert(x))


def simulated_accel_axes(kind: str, severity: float, fr: float, fs: int,
                         window_s: float, rng, fs_audio: int = 16000,
                         snr_db: float | None = None) -> np.ndarray:
    """Build one window of triaxial accelerometer data, (n, 3).

    Shared by `SimulatedSource` and `tools/simulate_soak.py` so the soak
    numbers stay comparable with everything else in the repo — they were two
    copies of the same three lines before T1.8, which is how models drift.

    `rng` is consumed in EXACTLY the draw order `ml/simulate.py`'s generators
    use (machine hum, then noise floor, then impulse train) so that axis 0 is
    bit-identical to the old single-axis output; the extra per-axis noise is
    drawn afterwards.
    """
    from scipy.signal import sosfilt
    from simulate import (SimConfig, _impulse_train, _machine_hum,
                          _noise_floor, _resonance_filter, _time)

    cfg = SimConfig(duration_s=window_s, fr=fr, fs_audio=fs_audio, fs_accel=fs)
    if snr_db is not None:
        cfg.snr_db = snr_db          # tools/simulate_soak.py varies this per window
    t = _time(cfg, fs)
    n = len(t)

    # --- the shared mechanical causes, in simulate.py's own draw order ------
    hum = _machine_hum(t, fr, rng)
    noise_x = _noise_floor(n, fs, cfg.snr_db, rng)

    train = None
    if kind in ("bearing_outer", "bearing_inner"):
        if kind == "bearing_outer":
            f_fault, mod = cfg.bearing.bpfo(fr), None
        else:
            f_fault, mod = cfg.bearing.bpfi(fr), fr
        train = _impulse_train(t, fs, f_fault, cfg.slip_jitter, rng,
                               modulate_at=mod)
    elif kind == "imbalance":
        pass
    elif kind != "normal":
        raise ValueError(f"unknown kind {kind!r}")

    # Growing 1x tone for the imbalance case (simulate.imbalance_signal draws
    # no randomness for this, so it costs nothing from the rng).
    growth = (np.linspace(0.0, severity, n) * np.sin(2 * np.pi * fr * t)
              if kind == "imbalance" else None)

    # Quadrature companions, computed once. `cos(phi)*s + sin(phi)*quad(s)`
    # rotates every harmonic of `s` by phi degrees at once.
    hum_q = _quadrature(hum)
    growth_q = _quadrature(growth) if growth is not None else None

    def rotate(s, s_q, phi_deg):
        if phi_deg == 0.0:
            return s                       # exact, so axis x stays bit-identical
        c, sn = np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))
        return c * s + sn * s_q

    # The frozen simulator normalises the burst train to unit variance and
    # then scales by severity. Do that ONCE, using axis x, so that the per-axis
    # gains below survive: normalising each axis separately would erase exactly
    # the difference we are trying to model.
    f0_x = min(cfg.resonance_hz, 0.4 * fs)
    burst_scale = None
    if train is not None:
        bx = sosfilt(_resonance_filter(fs, f0_x, cfg.resonance_q), train)
        burst_scale = float(np.std(bx) + 1e-12)

    # Terms are accumulated in `simulate.py`'s own order — hum, noise floor,
    # then the fault term — because floating-point addition is not associative
    # and axis x must come out bit-identical.
    axes = []
    for a in ("x", "y", "z"):
        m = ACCEL_AXES[a]
        phi = m["hum_phase_deg"]
        sig = m["hum_gain"] * rotate(hum, hum_q, phi)
        # independent sensor self-noise; axis x reuses the draw made above so
        # that it reproduces the old signal bit for bit
        sig = sig + (noise_x if a == "x"
                     else _noise_floor(n, fs, cfg.snr_db, rng))
        if growth is not None:
            sig = sig + m["hum_gain"] * rotate(growth, growth_q, phi)
        if train is not None:
            f0 = m["f0_ratio"] * f0_x
            q = max(2.0, m["q_ratio"] * cfg.resonance_q)
            b = sosfilt(_resonance_filter(fs, f0, q), train) / burst_scale
            sig = sig + m["gain"] * severity * b
        axes.append(sig)

    # transverse sensitivity, lower triangular — see the note above
    k = ACCEL_AXES["cross_axis"]
    axes[1] = axes[1] + k * axes[0]
    axes[2] = axes[2] + k * axes[0]
    return np.column_stack(axes)


class SimulatedSource:
    """Generates windows on demand from ml/simulate.py.

    `schedule(i)` -> {"kind": "normal"|"bearing_outer"|"bearing_inner"|
    "imbalance", "severity": float, "fr": Hz} — lets the caller script demos
    and tests: regime switches (change fr), transients, growing faults.

    Accelerometer: `ml/simulate.py` generates one channel, so the three axes
    are synthesised here by `simulated_accel_axes` — same impacts, different
    structural path per axis. See the long note above that function for the
    physics and for why axis 0 is deliberately left bit-identical."""

    def __init__(self, window_s: float = 30.0, fs_audio: int = 16000,
                 fs_accel: int = 6400, schedule=None, realtime: bool = False,
                 seed: int = 1234):
        self.window_s, self.fs_audio, self.fs_accel = window_s, fs_audio, fs_accel
        self.schedule = schedule or default_schedule
        self.realtime = realtime
        self.seed = seed

    def _generate(self, kind: str, severity: float, fr: float, rng) -> tuple:
        from simulate import (SimConfig, bearing_fault_signal, imbalance_signal,
                              normal_signal)
        cfg = SimConfig(duration_s=self.window_s, fr=fr,
                        fs_audio=self.fs_audio, fs_accel=self.fs_accel)
        gens = {
            "normal": lambda fs, r: normal_signal(cfg, fs, r),
            "bearing_outer": lambda fs, r: bearing_fault_signal(cfg, fs, r, severity, "outer"),
            "bearing_inner": lambda fs, r: bearing_fault_signal(cfg, fs, r, severity, "inner"),
            "imbalance": lambda fs, r: imbalance_signal(cfg, fs, r, severity),
        }
        gen = gens[kind]
        audio = gen(self.fs_audio, rng)
        # The accelerometer is NOT `gen(self.fs_accel, rng)` copied three
        # times any more (T1.8 / F6) — but it consumes the rng in the same
        # order, so axis 0 is the signal that call would have produced.
        accel = simulated_accel_axes(kind, severity, fr, self.fs_accel,
                                     self.window_s, rng, self.fs_audio)
        return audio, accel

    def windows(self):
        i = 0
        while True:
            sched = self.schedule(i)
            rng = np.random.default_rng(self.seed + i)
            t0 = time.monotonic()
            audio, accel = self._generate(sched["kind"], sched.get("severity", 0.0),
                                          sched.get("fr", 50.0), rng)
            if self.realtime:
                time.sleep(max(0.0, self.window_s - (time.monotonic() - t0)))
            yield audio, accel
            i += 1


class FileSource:
    """Replays one exported recording window by window, then stops.

    ----------------------------------------------------------------------
    MIC-ONLY (T7.2, 2026-08-20): csv_path may be None
    ----------------------------------------------------------------------
    `tools/ingest.py --mic-only` (the Week-1 fallback, and the only path a
    phone recording ever takes — a phone has no accelerometer CSV to hand
    over) writes no accelerometer file at all. Before this, the only
    consumer of this class required one and would crash on `np.loadtxt`
    with a real mic-only file. There is nothing bespoke to build for that
    case: `features.channel_stats`'s dead-channel guard and
    `features.estimate_fr`'s mic-only branch (see its docstring and
    `test_fr_mic_only_uses_audio`) both already treat an all-zero
    accelerometer channel as "sensor absent" and degrade correctly — mic-only
    is a supported build, not a special case bolted on here. So when
    `csv_path` is None this synthesises exactly that sentinel: an all-zero
    channel at `fs_accel_default`, the same shape `capture.HardwareSource`
    would hand downstream with `require_accel=False`.
    """

    def __init__(self, wav_path: Path, csv_path: Path | None = None,
                window_s: float = 30.0, fs_accel_default: int = 6400):
        from scipy.io import wavfile
        self.fs_audio, pcm = wavfile.read(wav_path)
        if pcm.ndim > 1:            # canonical ingest.py output is mono, but
            pcm = pcm.mean(axis=1)  # tolerate a raw stereo file too
        self.audio = pcm.astype(np.float64) / 32767.0
        if csv_path is not None:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            t = data[:, 0]
            cols = data[:, 1:4] if data.shape[1] >= 4 else data[:, 1:2]
            self.accel = cols
            self.fs_accel = round(1.0 / np.median(np.diff(t)))
        else:
            self.fs_accel = fs_accel_default
            n_accel = int(round(len(self.audio) * self.fs_accel / self.fs_audio))
            self.accel = np.zeros((n_accel, 1))
        self.window_s = window_s

    def windows(self):
        na, nv = int(self.window_s * self.fs_audio), int(self.window_s * self.fs_accel)
        for i in range(min(len(self.audio) // na, len(self.accel) // nv)):
            yield self.audio[i * na:(i + 1) * na], self.accel[i * nv:(i + 1) * nv]


# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
#
# NOTHING in this section is imported at module load time. `import capture`
# must succeed on a laptop with neither spidev nor sounddevice nor ALSA, because
# the test suite, ml/ scripts and the bench tools all import this module. Every
# hardware dependency is therefore imported inside the function that needs it,
# and every failure is converted into `HardwareUnavailable` with a message a
# student can act on — never a bare ImportError or OSError traceback.


class HardwareUnavailable(RuntimeError):
    """Raised when a sensor (or its host library) is missing or mis-wired.

    Carries `what` (a short machine-readable tag: "audio" / "accel") and
    `remedy` (a human sentence telling the student what to plug in or install).
    The bench scripts catch this and print a friendly block instead of a
    traceback; that is the whole reason the class exists.
    """

    def __init__(self, what: str, message: str, remedy: str = ""):
        self.what = what
        self.remedy = remedy
        super().__init__(message)


class IIS3DWB:
    """SPI driver for the ST IIS3DWB wideband vibration accelerometer.

    ------------------------------------------------------------------------
    WHY THIS PART
    ------------------------------------------------------------------------
    A bearing defect is not a tone, it is a series of *impacts*. Each impact
    rings whatever structural resonance it can reach — for a small motor
    end-shield that is typically 1–20 kHz. To see that ringing you need an
    accelerometer whose response is flat well past it. The IIS3DWB is flat to
    6.3 kHz and samples at a FIXED 26.667 kHz. An ADXL345-class part (~800 Hz
    usable) is provably blind to the band the fault energy lives in — see
    ml/verify_signals.py fig1.

    ------------------------------------------------------------------------
    REGISTER FACTS — provenance
    ------------------------------------------------------------------------
    Every constant below was checked against ST's own platform-independent
    driver header, which is the authoritative machine-readable version of the
    datasheet register map:
      https://github.com/STMicroelectronics/iis3dwb-pid/blob/master/iis3dwb_reg.h
      (mirrored at zephyrproject-rtos/hal_st .../iis3dwb_STdC/driver/iis3dwb_reg.h)
    and against ST's FIFO example:
      https://github.com/STMicroelectronics/STMems_Standard_C_drivers/blob/
      master/iis3dwb_STdC/examples/iis3dwb_fifo.c
    Datasheet: ST IIS3DWB, DS12718. Application note: AN5444.

    VERIFIED against that header:
      WHO_AM_I        0x0F, expected value 0x7B  (IIS3DWB_ID)
      CTRL1_XL        0x10  [7:5]=XL_EN, [3:2]=FS_XL, [1]=LPF2_XL_EN
      CTRL3_C         0x12  bit6=BDU, bit3=SIM, bit2=IF_INC, bit0=SW_RESET
      FIFO_CTRL3      0x09  [3:0]=BDR_XL
      FIFO_CTRL4      0x0A  [2:0]=FIFO_MODE
      FIFO_STATUS1/2  0x3A / 0x3B  (DIFF_FIFO is 10 bits, split across both)
      FIFO_DATA_OUT_TAG 0x78, then X_L..Z_H at 0x79..0x7E
      XL_EN code for 26.667 kHz = 5;  FIFO BDR_XL code for 26.667 kHz = 10
      FIFO_MODE code for Stream (continuous) = 6
      FIFO tag: sensor id is `tag_byte >> 3`; accelerometer data == 2

    ------------------------------------------------------------------------
    BUGS FIXED HERE relative to the first datasheet-only draft (2026-08)
    ------------------------------------------------------------------------
    1. FIFO_CTRL3 was written 0x06. BDR_XL=6 is *not* the 26.667 kHz batching
       code; ST's enum says IIS3DWB_XL_BATCHED_AT_26k7Hz = 10 (0x0A). With 0x06
       the accelerometer runs but NOTHING is batched into the FIFO, so
       drain_fifo() would have returned zero samples forever. This is exactly
       the class of bug bench/check_accel.py's measured-ODR test catches.
    2. The FIFO fill level was read from FIFO_STATUS1 only (8 bits). DIFF_FIFO
       is 10 bits: the low 8 in FIFO_STATUS1 and the top 2 in FIFO_STATUS2
       [1:0]. Reading one byte silently truncates any level ≥ 256 — you would
       lose samples in bursts and the measured ODR would come out low.
    3. Samples were read one 8-byte transfer at a time. At 26 667 samples/s
       that is 26 667 syscalls/s; CPython round-trip through spidev is tens of
       microseconds, so the drain could not keep up with the sensor and the
       FIFO would overrun. We now burst-read the whole FIFO in one transfer,
       relying on the documented address auto-wrap of the FIFO output block.
    4. The FIFO tag byte was discarded without checking. The FIFO can also
       carry temperature and timestamp words; unfiltered, those get decoded as
       enormous fake accelerations. We now keep only tag>>3 == 2.

    ------------------------------------------------------------------------
    UNVERIFIED (needs a bench, flagged honestly)
    ------------------------------------------------------------------------
    * The auto-increment WRAP of the FIFO output registers (0x78..0x7E then
      back to 0x78) is how ST's `iis3dwb_fifo_out_multi_raw_get()` reads N
      words in one burst, so it is strongly implied — but we have not seen it
      stated as a sentence in the datasheet and have not scoped it.
      `drain_fifo(burst=False)` falls back to one word per transfer if the
      burst path returns nonsense. check_accel.py compares the two.
    * Maximum SPI clock on real Pi wiring. Datasheet allows 10 MHz; we default
      to 8 MHz, which is conservative for dupont jumpers. UNVERIFIED.
    * spidev's per-transfer buffer limit (`/sys/module/spidev/parameters/
      bufsiz`, 4096 bytes by default on Pi OS) — we chunk to stay under it.
    """

    # --- register addresses (VERIFIED against ST iis3dwb_reg.h) -------------
    _WHO_AM_I, _WHO_AM_I_VAL = 0x0F, 0x7B
    _PIN_CTRL = 0x02
    _FIFO_CTRL1, _FIFO_CTRL2 = 0x07, 0x08
    _FIFO_CTRL3, _FIFO_CTRL4 = 0x09, 0x0A
    _CTRL1_XL, _CTRL3_C, _CTRL4_C = 0x10, 0x12, 0x13
    _CTRL6_C, _CTRL8_XL = 0x15, 0x17
    _STATUS_REG = 0x1E
    _OUTX_L_A = 0x28
    _FIFO_STATUS1, _FIFO_STATUS2 = 0x3A, 0x3B
    _INTERNAL_FREQ_FINE = 0x63
    _FIFO_DATA_OUT_TAG = 0x78

    # --- configuration byte values (derived from the bit layouts above) ----
    # CTRL3_C = BDU (bit6) | IF_INC (bit2). BDU stops a 16-bit sample being
    # torn across an update; IF_INC is what makes multi-byte bursts advance.
    _CTRL3_C_VAL = 0x44
    # CTRL1_XL = XL_EN=5 (<<5 = 0xA0) | FS_XL=1 (<<2 = 0x04) -> 0xA4.
    # NOTE the deliberately confusing full-scale encoding in ST's enum:
    #   0 = ±2 g, 1 = ±16 g, 2 = ±4 g, 3 = ±8 g.
    # ±16 g is code *1*, not 3. Getting this wrong scales every reading by
    # 8x — which is precisely what the 1 g gravity test in check_accel.py
    # exists to catch.
    _CTRL1_XL_VAL = 0xA4
    _FIFO_CTRL3_VAL = 0x0A       # BDR_XL = 10 = batch XL at 26.667 kHz
    _FIFO_CTRL4_VAL = 0x06       # FIFO_MODE = 6 = Stream (continuous)

    # ±16 g across a signed 16-bit word: 32 g / 65536 counts = 0.488 mg/LSB.
    # Matches ST's iis3dwb_from_fs16g_to_mg(). Arithmetically checkable, so
    # this one is not a guess.
    _SCALE_G = 32.0 / 65536.0    # = 0.00048828125 g per LSB

    ODR_NOMINAL_HZ = 26667.0     # datasheet nominal; the part's real ODR is
                                 # trimmed and readable via INTERNAL_FREQ_FINE
    FIFO_WORDS_MAX = 512         # 3 KB FIFO / 6 bytes of data per word.
                                 # ST's example caps the watermark at 511.
    FIFO_WORD_BYTES = 7          # 1 tag byte + 6 data bytes

    def __init__(self, bus: int = 0, device: int = 0, speed_hz: int = 8_000_000,
                 configure: bool = True):
        try:
            import spidev
        except ImportError as e:
            raise HardwareUnavailable(
                "accel", f"the 'spidev' Python module is not installed ({e})",
                "On the Pi:  pip3 install spidev   (and make sure "
                "'dtparam=spi=on' is in /boot/firmware/config.txt, then reboot)"
            ) from e

        self.spi = spidev.SpiDev()
        try:
            self.spi.open(bus, device)
        except (FileNotFoundError, OSError) as e:
            raise HardwareUnavailable(
                "accel", f"cannot open /dev/spidev{bus}.{device} ({e})",
                "Enable SPI: add 'dtparam=spi=on' to /boot/firmware/config.txt, "
                "reboot, then check that /dev/spidev0.0 exists."
            ) from e

        self.spi.max_speed_hz = speed_hz
        # SPI mode 3 = CPOL 1, CPHA 1. ST's parts idle the clock high and
        # sample on the rising (second) edge. Mode 0 reads plausible-looking
        # garbage on some boards, which is worse than reading nothing — the
        # WHO_AM_I check below is the only thing standing between you and a
        # week of debugging "noisy" data that was never real.
        self.spi.mode = 0b11

        self.who_am_i = self._read(self._WHO_AM_I)
        if self.who_am_i != self._WHO_AM_I_VAL:
            self.spi.close()
            raise HardwareUnavailable(
                "accel",
                f"IIS3DWB WHO_AM_I read back 0x{self.who_am_i:02X}, "
                f"expected 0x{self._WHO_AM_I_VAL:02X}",
                "0x00 or 0xFF usually means MISO/MOSI swapped, CS not wired to "
                "CE0 (pin 24), or the board unpowered. A plausible-but-wrong "
                "value usually means the SPI mode is wrong (this part needs "
                "mode 3). Check wiring against the hardware design notes (not in this public copy)."
            )

        if configure:
            self._write(self._CTRL3_C, self._CTRL3_C_VAL)
            self._write(self._CTRL1_XL, self._CTRL1_XL_VAL)
            self._write(self._FIFO_CTRL3, self._FIFO_CTRL3_VAL)
            self._write(self._FIFO_CTRL4, self._FIFO_CTRL4_VAL)

        # INTERNAL_FREQ_FINE (0x63) holds a signed trim of the internal
        # oscillator in units of +0.0015 %/LSB (datasheet). Reading it gives a
        # better nominal ODR than 26 667 Hz — but it is still the *sensor's*
        # opinion of its own clock, which is why check_accel.py additionally
        # measures the achieved rate against the Pi's wall clock.
        raw = self._read(self._INTERNAL_FREQ_FINE)
        trim = raw - 256 if raw > 127 else raw            # int8
        self.odr_hz = self.ODR_NOMINAL_HZ * (1.0 + 0.0015 * trim / 100.0)

    # -- low-level -----------------------------------------------------------

    def _read(self, reg: int) -> int:
        # Bit 7 of the address byte is the read flag on all ST MEMS SPI parts.
        return self.spi.xfer2([0x80 | reg, 0])[1]

    def _read_burst(self, reg: int, n: int) -> bytes:
        return bytes(self.spi.xfer2([0x80 | reg] + [0] * n)[1:])

    def _write(self, reg: int, val: int) -> None:
        self.spi.xfer2([reg & 0x7F, val])

    def fifo_level(self) -> int:
        """Number of unread FIFO words. DIFF_FIFO is 10 bits: low 8 in
        FIFO_STATUS1, top 2 in FIFO_STATUS2[1:0]. Reading only STATUS1 is the
        bug that used to be here — it wraps at 256 and silently loses data."""
        lo = self._read(self._FIFO_STATUS1)
        hi = self._read(self._FIFO_STATUS2) & 0x03
        return (hi << 8) | lo

    def fifo_flags(self) -> dict:
        """FIFO_STATUS2 status bits — overrun is the one that matters. If
        OVR is set your drain loop is too slow and you have lost samples;
        report it rather than quietly producing a short window."""
        s2 = self._read(self._FIFO_STATUS2)
        return {
            "watermark": bool(s2 & 0x80),
            "overrun": bool(s2 & 0x40),
            "full": bool(s2 & 0x20),
            "bdr_counter": bool(s2 & 0x10),
            "overrun_latched": bool(s2 & 0x08),
        }

    def read_one(self) -> np.ndarray:
        """Single (3,) sample in g straight from OUTX_L_A, bypassing the FIFO.
        Used by the gravity test: it removes the FIFO from the equation, so if
        gravity is right here but wrong through the FIFO you know the fault is
        in the FIFO decode, not the analog chain or the scale factor."""
        raw = self._read_burst(self._OUTX_L_A, 6)
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) * self._SCALE_G

    def drain_fifo(self, burst: bool = True, max_words: int | None = None) -> np.ndarray:
        """Read all queued accelerometer words -> (k, 3) array in g.

        Each FIFO word is 7 bytes: 1 tag + 6 data (X,Y,Z as little-endian
        int16). We read the whole backlog in one SPI transfer and rely on the
        FIFO output registers auto-wrapping 0x78..0x7E -> 0x78, which is how
        ST's own `iis3dwb_fifo_out_multi_raw_get()` works. Set burst=False to
        fall back to one transfer per word (slow, but unambiguous) if you
        suspect the wrap.
        """
        n = self.fifo_level()
        if max_words is not None:
            n = min(n, max_words)
        if n <= 0:
            return np.zeros((0, 3))

        if burst:
            # spidev's per-transfer buffer defaults to 4096 bytes on Pi OS
            # (/sys/module/spidev/parameters/bufsiz). 512 words * 7 = 3584 B,
            # which fits, but chunk anyway so a raised watermark can't break us.
            chunk_words = 4000 // self.FIFO_WORD_BYTES
            blobs = []
            remaining = n
            while remaining > 0:
                k = min(remaining, chunk_words)
                blobs.append(self._read_burst(self._FIFO_DATA_OUT_TAG,
                                              k * self.FIFO_WORD_BYTES))
                remaining -= k
            raw = b"".join(blobs)
        else:
            raw = b"".join(self._read_burst(self._FIFO_DATA_OUT_TAG,
                                           self.FIFO_WORD_BYTES)
                           for _ in range(n))

        words = np.frombuffer(raw, dtype=np.uint8).reshape(-1, self.FIFO_WORD_BYTES)
        # Tag byte layout: [7:3] sensor id, [2:1] counter, [0] parity.
        # Accelerometer data has sensor id 2; temperature (3) and timestamp (4)
        # words also appear in the stream if you enable them, and decoding
        # those as acceleration produces spectacular fake transients.
        is_xl = (words[:, 0] >> 3) == 2
        data = np.ascontiguousarray(words[is_xl, 1:])
        if data.size == 0:
            return np.zeros((0, 3))
        return data.view("<i2").astype(np.float64) * self._SCALE_G

    def close(self) -> None:
        try:
            self.spi.close()
        except Exception:                                  # noqa: BLE001
            pass


def open_audio(fs_audio: int = 16000, device=None, channels: int = 1):
    """Import sounddevice and validate an input device, or raise
    HardwareUnavailable with something a student can actually do.

    Kept separate from HardwareSource so bench/check_audio.py can use the exact
    same probe path the firmware uses — a bring-up test that exercises a
    different code path from production is a test of the wrong thing.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        # OSError here is the classic "PortAudio library not found" — pip
        # installs the Python wrapper but not the C library.
        raise HardwareUnavailable(
            "audio", f"sounddevice/PortAudio unavailable ({e})",
            "On the Pi:  sudo apt install -y libportaudio2 && pip3 install sounddevice"
        ) from e

    try:
        devices = sd.query_devices()
    except Exception as e:                                  # noqa: BLE001
        raise HardwareUnavailable(
            "audio", f"ALSA returned no device list ({e})",
            "No sound subsystem at all. On a Pi this means the I2S overlay is "
            "not loaded: check /boot/firmware/config.txt for 'dtparam=i2s=on' "
            "and 'dtoverlay=googlevoicehat-soundcard', then reboot."
        ) from e

    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) > 0]
    if not inputs:
        raise HardwareUnavailable(
            "audio", "ALSA is present but exposes no capture (input) devices",
            "Plug in the INMP441 and enable the I2S overlay — see "
            "the hardware design notes (not in this public copy) step 2. Verify with 'arecord -l' first; "
            "if arecord shows nothing, Python will not either."
        )
    return sd, inputs


class HardwareSource:
    """INMP441 (I2S/ALSA) + IIS3DWB (SPI), captured concurrently.

    TODO(bench): the timing of the two streams relative to each other is
    unverified. We start the audio recording, then busy-drain the accel FIFO
    for the same wall-clock interval. Sub-millisecond alignment is not claimed
    and is not needed: every feature in features.py is computed per channel
    over a 30 s window, so a few ms of skew is irrelevant. It WOULD matter if
    you ever add cross-channel coherence — note it before you do.

    MIC-ONLY FALLBACK (require_accel=False, the default): if the IIS3DWB is
    absent, this class logs once and continues with audio only. See
    `_open_accel` for the full rationale.
    """

    def __init__(self, window_s: float = 30.0, fs_audio: int = 16000,
                 fs_accel: int = 6400, alsa_device=None,
                 require_accel: bool = False, spi_bus: int = 0,
                 spi_device: int = 0):
        self.window_s = window_s
        self.fs_audio, self.fs_accel = fs_audio, fs_accel
        self.alsa_device = alsa_device
        self.require_accel = require_accel

        # Audio is not optional: with neither sensor there is nothing to do.
        self.sd, _ = open_audio(fs_audio, alsa_device)

        self.accel_dev = self._open_accel(spi_bus, spi_device)
        self.accel_ok = self.accel_dev is not None
        odr = self.accel_dev.odr_hz if self.accel_ok else IIS3DWB.ODR_NOMINAL_HZ
        self.decim = max(1, round(odr / fs_accel))          # 26.7 kHz -> ~6.4 kHz

        # Populated after each window so callers (and bench/record_session.py)
        # can report MEASURED rates rather than configured ones.
        self.measured_fs_audio: float | None = None
        self.measured_fs_accel: float | None = None
        self.last_fifo_overrun = False

    def _open_accel(self, bus: int, device: int):
        """Try to bring up the IIS3DWB; return None to run mic-only.

        WHY MIC-ONLY IS ACCEPTABLE (the execution plan (not in this public copy), week 1): the fault
        physics we detect is impacts exciting a structural resonance in the
        1–20 kHz region. A 16 kHz microphone reaches 8 kHz, which covers the
        band the envelope analysis actually uses. The accelerometer buys you
        immunity to airborne noise from neighbouring machines and a second
        opinion on shaft speed — valuable, but not load-bearing. Losing two
        weeks to SPI is worse than losing the cross-check.

        WHY WE EMIT EXACT ZEROS rather than synthetic noise: features.py is
        already hardened for a dead channel. `channel_stats` has an explicit
        guard that returns a fixed vector when RMS < 1e-9, `band_energy_ratios`
        returns a constant floor, and `estimate_fr` finds no accel agreement so
        it sets fr_reliable=False. The 20 accel-derived dimensions therefore
        become CONSTANT columns: identical in every learn window and every test
        window, so they contribute exactly zero to the Mahalanobis distance
        (LedoitWolf shrinkage keeps the covariance invertible). Fabricated
        noise, by contrast, would be indistinguishable from a real quiet
        accelerometer — and six months later nobody would remember that the
        vibration channel in that dataset was fiction. Zeros are honest and
        numerically inert; fake noise is neither.
        """
        try:
            dev = IIS3DWB(bus, device)
            log.info("IIS3DWB OK (WHO_AM_I=0x%02X, ODR=%.1f Hz)",
                     dev.who_am_i, dev.odr_hz)
            return dev
        except HardwareUnavailable as e:
            if self.require_accel:
                raise
            log.warning(
                "ACCELEROMETER UNAVAILABLE — running MIC-ONLY. %s\n"
                "  Fix: %s\n"
                "  The 20 accelerometer feature dimensions will be constant "
                "zeros and fr_reliable will be False. This is an allowed "
                "week-1 fallback (the execution plan (not in this public copy)); record it in your "
                "notes so nobody later mistakes this data for dual-channel.",
                e, e.remedy)
            return None

    def _empty_accel(self, n_samples: int) -> np.ndarray:
        return np.zeros((max(1, n_samples), 3))

    def windows(self):
        n_audio = int(self.window_s * self.fs_audio)
        while True:
            t0 = time.monotonic()
            rec = self.sd.rec(n_audio, samplerate=self.fs_audio, channels=1,
                              dtype="float32", device=self.alsa_device)

            if self.accel_ok:
                chunks = []
                overrun = False
                t_end = t0 + self.window_s
                while time.monotonic() < t_end:
                    chunks.append(self.accel_dev.drain_fifo())
                    if self.accel_dev.fifo_flags()["overrun"]:
                        overrun = True
                    # The 512-word FIFO at 26.667 kHz fills in 512/26667 =
                    # ~19 ms, NOT the ~5 ms the first draft of this comment
                    # claimed. Poll at ~5 ms for 4x headroom against the Pi's
                    # scheduler; anything slower than 19 ms guarantees overrun.
                    time.sleep(0.005)
                self.sd.wait()
                self.last_fifo_overrun = overrun
                if overrun:
                    log.warning("IIS3DWB FIFO overrun — samples were lost this "
                                "window. The drain loop is not keeping up.")
                accel_full = (np.concatenate(chunks) if chunks
                              else np.zeros((0, 3)))
            else:
                self.sd.wait()
                accel_full = np.zeros((0, 3))

            elapsed = time.monotonic() - t0
            self.measured_fs_audio = n_audio / elapsed if elapsed > 0 else None

            if len(accel_full) >= self.decim:
                self.measured_fs_accel = (len(accel_full) / self.decim) / elapsed
                k = (len(accel_full) // self.decim) * self.decim
                # Decimate by a boxcar mean. This is a crude anti-alias filter
                # (a 4-tap moving average has its first null at ODR/4 and only
                # ~13 dB of first-sidelobe rejection), but the part's own
                # analog chain is already rolling off above 6.3 kHz, so there
                # is little energy left to fold down. TODO(bench): compare
                # against a proper scipy.signal.decimate on real data.
                accel = accel_full[:k].reshape(-1, self.decim, 3).mean(axis=1)
            else:
                self.measured_fs_accel = 0.0 if self.accel_ok else None
                accel = self._empty_accel(int(self.window_s * self.fs_accel))

            yield rec[:, 0].astype(np.float64), accel

    def close(self) -> None:
        if self.accel_ok:
            self.accel_dev.close()


def make_source(cfg: dict, simulate: bool = False, schedule=None,
                realtime: bool = False, seed: int = 1234,
                require_accel: bool = False):
    w = cfg["window"]["seconds"]
    fa, fv = cfg["audio"]["sample_rate"], cfg["accelerometer"]["sample_rate"]
    if simulate:
        return SimulatedSource(w, fa, fv, schedule, realtime, seed)
    return HardwareSource(w, fa, fv, require_accel=require_accel)
