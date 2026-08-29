#!/usr/bin/env python3
"""
check_audio.py — bring-up test 1 of 5: is the microphone real?

    python firmware/bench/check_audio.py --list
    python firmware/bench/check_audio.py --seconds 10
    python firmware/bench/check_audio.py --tone 1000     # play 1 kHz from a phone
    python firmware/bench/check_audio.py --simulate      # no hardware needed

===========================================================================
WHY THIS SCRIPT EXISTS (read this before you run it)
===========================================================================

Everything downstream in this project is a frequency-domain measurement. The
detector selects a demodulation band, computes an envelope spectrum in that
band, and asks whether the envelope is periodic. Every one of those steps is
expressed in Hz. Hz are samples-per-second — so if you do not know the true
number of samples per second, you do not know any of your frequencies, and you
will not find out. The pipeline will keep running and keep producing 40-dim
feature vectors that look completely plausible.

Three specific failure modes this script catches, all of which are silent:

1. SAMPLE-RATE LIE. The googlevoicehat I2S overlay used for the INMP441 runs
   the hardware at a FIXED 48 kHz. When you ask ALSA (or sounddevice) for
   16 kHz you are actually getting 48 kHz resampled in software by the `plug`
   layer. That is usually fine — but if the resampler is misconfigured, or you
   accidentally open the raw `hw:` device, or the I2S clock divider lands
   somewhere unexpected, you get a stream whose real rate is not what you
   asked for. A 3x rate error moves a 4 kHz resonance to 1.33 kHz. Your
   demodulation band search would then select the wrong band, forever, and
   nothing would ever look wrong. So: we do NOT ask the driver what rate it
   used. We count the samples we received and divide by the wall-clock time
   they took to arrive. That number cannot lie.

2. BIT-DEPTH / ALIGNMENT BUGS. The INMP441 emits 24-bit samples inside 32-bit
   I2S frames. Getting the shift wrong gives you a signal that is scaled by a
   power of 256, or one that is byte-swapped into noise. The tone test catches
   the noise case immediately (no peak), and the RMS/clipping numbers catch
   the scaling case.

3. DEAD OR SATURATED CHANNEL. An unconnected I2S data line reads as digital
   silence or as a stuck rail. Both produce feature vectors — features.py has
   a dead-channel guard that returns a fixed vector — so the anomaly detector
   would happily learn "normal = dead microphone" and then never alert.

===========================================================================
WHAT IS MEASURED, AND WHAT THE NUMBER MEANS
===========================================================================

achieved sample rate  samples received / wall-clock seconds. Must match the
                      configured rate to better than 1 %. See above.
DC offset             mean of the signal. MEMS microphones have a small
                      genuine DC offset; a LARGE one means a broken decode.
                      It matters because features.channel_stats computes RMS
                      and crest factor on the raw signal: a DC term inflates
                      RMS, deflates crest, and does it by an amount that
                      drifts with temperature. (features.py removes the mean
                      before spectral work, but not before the statistics.)
RMS level             overall loudness in dBFS. Wants to be roughly
                      -60..-6 dBFS. Near 0 means clipping; below -80 means
                      dead or the gain is unusable.
clipping count        samples at full scale. Clipping generates broadband
                      harmonic splatter that looks EXACTLY like the impulsive
                      content we are trying to detect. A clipped healthy
                      machine can score as a bearing fault.
noise floor           median short-term RMS. This is the sensitivity limit of
                      the whole chain; a fault signature below it is invisible
                      no matter how clever the algorithm.
tone accuracy         with --tone, the dominant FFT peak must land within 1 %
                      of the frequency you played. This is an END-TO-END test
                      of the entire acquisition chain against an external
                      reference, which is the only kind of test that can catch
                      a systematic error.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    from bench_common import (EXIT_NO_HARDWARE, Report, dbfs, dominant_peak,
                              noise_floor_dbfs, run_guarded, synth_room_tone)
except ImportError:                                        # run as a module
    from firmware.bench.bench_common import (               # type: ignore
        EXIT_NO_HARDWARE, Report, dbfs, dominant_peak, noise_floor_dbfs,
        run_guarded, synth_room_tone)

from capture import HardwareUnavailable, open_audio        # noqa: E402


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def list_devices() -> int:
    """`arecord -l` for Python. Run this FIRST on a new Pi."""
    try:
        sd, inputs = open_audio()
    except HardwareUnavailable as e:
        from bench_common import print_missing_hardware
        print_missing_hardware("microphone (INMP441 over I2S)", str(e), e.remedy)
        return EXIT_NO_HARDWARE
    print("Capture (input) devices visible to ALSA/PortAudio:")
    print()
    for i, d in inputs:
        print(f"  [{i}] {d['name']}")
        print(f"        channels={d['max_input_channels']}  "
              f"default_samplerate={d['default_samplerate']:.0f} Hz")
    print()
    print("Pass the index or the name to --device. On a Pi with the")
    print("googlevoicehat overlay the INMP441 appears as")
    print("'snd_rpi_googlevoicehat_soundcar'. If nothing is listed here,")
    print("check 'arecord -l' at the shell first — if the shell cannot see it,")
    print("neither can Python, and the problem is the device-tree overlay.")
    return 0


def record(seconds: float, fs: int, device) -> tuple[np.ndarray, float, int]:
    """Return (samples, elapsed_wall_clock_s, requested_n).

    The clock is started immediately before the blocking read and stopped
    immediately after. `sd.rec` returns exactly the number of frames asked
    for, so the interesting quantity is how LONG they took: if the hardware is
    really running at 48 kHz and the resampler is producing 16 kHz correctly,
    16000*seconds frames take `seconds` of wall clock. If the achieved rate is
    wrong, they take proportionally longer or shorter.
    """
    sd, _ = open_audio(fs, device)
    n = int(seconds * fs)
    t0 = time.perf_counter()
    rec = sd.rec(n, samplerate=fs, channels=1, dtype="float32", device=device)
    sd.wait()
    elapsed = time.perf_counter() - t0
    return rec[:, 0].astype(np.float64), elapsed, n


def simulate(seconds: float, fs: int, tone_hz: float | None,
             seed: int = 7) -> tuple[np.ndarray, float, int]:
    """Synthetic stand-in. Deliberately includes a small rate error (0.2 %) so
    the achieved-rate check is exercised rather than trivially satisfied."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = synth_room_tone(n, fs, rng, level=0.01, tone_hz=tone_hz, tone_level=0.15)
    elapsed = seconds * 1.002
    return x, elapsed, n


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(x: np.ndarray, elapsed: float, n_requested: int, fs_cfg: int,
            tone_hz: float | None, rep: Report,
            rate_tol_pct: float = 1.0) -> None:
    n = len(x)

    # -- 1. the measured, not configured, sample rate -----------------------
    rep.section("sample rate (MEASURED, never trusted from the config)")
    fs_meas = n / elapsed if elapsed > 0 else 0.0
    err_pct = 100.0 * (fs_meas - fs_cfg) / fs_cfg
    rep.check("achieved sample rate",
              abs(err_pct) <= rate_tol_pct,
              f"{fs_meas:8.1f} Hz   (configured {fs_cfg} Hz, "
              f"error {err_pct:+.2f} %, limit +/-{rate_tol_pct:.1f} %)",
              "" if abs(err_pct) <= rate_tol_pct else
              "The stream is not arriving at the rate you asked for. Every "
              "frequency this project measures is therefore wrong by the same "
              "factor. Fix this before recording anything. Common cause: the "
              "I2S overlay runs the hardware at a fixed 48 kHz and you are "
              "opening the raw hw: device instead of plughw:.")
    rep.check("samples received", n == n_requested,
              f"{n} of {n_requested} requested",
              "" if n == n_requested else "Short read: the driver dropped "
              "frames. Usually an xrun from CPU starvation.")

    # -- 2. is there a signal at all? ---------------------------------------
    rep.section("signal integrity")
    dc = float(np.mean(x))
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if n else 0.0
    ac_rms = float(np.std(x)) if n else 0.0

    rep.check("channel alive", ac_rms > 1e-6,
              f"AC RMS {ac_rms:.3e} ({dbfs(ac_rms):6.1f} dBFS)",
              "" if ac_rms > 1e-6 else
              "Digital silence. The I2S data line (SD -> GPIO20, header pin "
              "38) is not delivering bits. Check that wire, check L/R is tied "
              "to GND, and check the overlay is loaded.")

    n_unique = len(np.unique(np.round(x, 9)))
    rep.check("not stuck / not constant", n_unique > 16,
              f"{n_unique} distinct sample values",
              "" if n_unique > 16 else
              "The stream is a small number of repeating values — a stuck bus "
              "or a bad bit alignment, not a microphone.")

    # DC offset: report in both absolute FS units and as a fraction of the AC
    # content, because 'is the DC big' only means anything relative to signal.
    dc_ratio = abs(dc) / (ac_rms + 1e-12)
    rep.check("DC offset", abs(dc) < 0.05,
              f"{dc:+.5f} FS  ({dc_ratio:.2f} x AC RMS)",
              "" if abs(dc) < 0.05 else
              "A large DC term inflates the RMS and suppresses the crest "
              "factor in features.channel_stats, and it drifts with "
              "temperature — so your 'normal' moves when the room warms up. "
              "Usually a sign-extension error on the 24-in-32-bit I2S frame.")

    rep.check("RMS level in usable window", -80.0 < dbfs(rms) < -3.0,
              f"{dbfs(rms):6.1f} dBFS   (want -80 .. -3)",
              "" if -80.0 < dbfs(rms) < -3.0 else
              "Too quiet to resolve a fault above the ADC floor, or so loud "
              "it is about to clip. Move the node or change the source level.")

    # Clipping: count samples within 0.1 % of full scale. float32 capture from
    # a 24-bit source can exceed 1.0 slightly after resampling, so >= 0.999.
    n_clip = int(np.count_nonzero(np.abs(x) >= 0.999))
    clip_pct = 100.0 * n_clip / max(n, 1)
    rep.check("clipping", n_clip == 0,
              f"{n_clip} samples at full scale ({clip_pct:.4f} %)",
              "" if n_clip == 0 else
              "Clipping folds broadband harmonics into the signal. Those look "
              "identical to bearing impacts to an envelope detector, so a "
              "clipped healthy machine can read as faulty. Reduce the source "
              "level or move the microphone away.")

    rep.check("peak headroom", peak < 0.95,
              f"peak {peak:.4f} FS ({dbfs(peak):6.1f} dBFS)")

    # -- 3. noise floor ------------------------------------------------------
    rep.section("noise floor")
    nf = noise_floor_dbfs(x, fs_cfg)
    rep.check("noise floor (median frame RMS)", nf < -20.0,
              f"{nf:6.1f} dBFS",
              "" if nf < -20.0 else
              "The floor is very close to full scale. Either the room is "
              "extremely loud or the gain path is broken.")
    rep.check("dynamic range above floor", None,
              f"{dbfs(peak) - nf:5.1f} dB peak-to-floor",
              "This is the whole budget you have for a fault signature. Early "
              "bearing faults sit tens of dB below the machine's own running "
              "noise, so more here is strictly better.")

    # Spectral noise density: useful for comparing two microphones or two
    # mounting positions honestly, independent of the analysis bandwidth.
    freqs = np.fft.rfftfreq(min(n, 16384), 1.0 / fs_cfg)
    if n >= 1024:
        seg = x[:min(n, 16384)]
        spec = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(len(seg))))
        # Normalise to amplitude spectral density: /sqrt(bin width * window ENBW)
        enbw = 1.5                                  # Hann equivalent-noise-bw
        binw = fs_cfg / len(seg)
        psd_amp = spec / (len(seg) / 2) / np.sqrt(binw * enbw)
        band = (freqs >= 1000.0) & (freqs <= min(6000.0, fs_cfg / 2 * 0.95))
        if band.any():
            rep.check("noise density 1-6 kHz", None,
                      f"{dbfs(float(np.median(psd_amp[band]))):6.1f} dBFS/sqrt(Hz)",
                      "Measured in the band the demodulation search uses. "
                      "Record this for each microphone and mounting position "
                      "you try; it is how you compare them objectively.")

    # -- 4. the external-reference test -------------------------------------
    if tone_hz:
        rep.section(f"tone test — expecting {tone_hz:.1f} Hz from an external source")
        f_peak, mag = dominant_peak(x, fs_cfg, fmin=20.0, fmax=fs_cfg / 2 * 0.98)
        err = 100.0 * (f_peak - tone_hz) / tone_hz
        rep.check("dominant peak frequency", abs(err) <= 1.0,
                  f"{f_peak:8.2f} Hz   (expected {tone_hz:.1f}, "
                  f"error {err:+.3f} %, limit +/-1 %)",
                  "" if abs(err) <= 1.0 else
                  f"The peak is off by {err:+.2f} %. If the ratio "
                  f"{f_peak / tone_hz:.3f} looks like a simple fraction "
                  "(0.333, 3.0, 0.5, 2.0) you have a sample-rate mismatch of "
                  "exactly that factor — almost always 48 kHz hardware being "
                  "labelled as 16 kHz or vice versa.")

        # Peak prominence: a peak that is only just above the floor is not
        # evidence of anything. 20 dB is a deliberately modest bar — a phone
        # speaker across a room easily clears it.
        spec_all = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
        prominence_db = 20 * np.log10((mag + 1e-20) /
                                      (np.median(spec_all) + 1e-20))
        rep.check("peak prominence over median bin", prominence_db > 20.0,
                  f"{prominence_db:5.1f} dB   (want > 20)",
                  "" if prominence_db > 20.0 else
                  "There is a peak, but it barely stands above the noise. "
                  "Turn the tone up, move the phone closer, or check that the "
                  "microphone port is not blocked.")

        # Harmonic check: a clean acquisition chain reproduces a sine as a
        # sine. Strong 2f/3f content that is NOT in the source means
        # nonlinearity — clipping, or a bad bit shift.
        if 3 * tone_hz < fs_cfg / 2:
            f_axis = np.fft.rfftfreq(n, 1.0 / fs_cfg)

            def bin_mag(f0):
                i = int(np.argmin(np.abs(f_axis - f0)))
                return float(np.max(spec_all[max(0, i - 2):i + 3]))

            thd_like = 20 * np.log10((bin_mag(2 * tone_hz) +
                                      bin_mag(3 * tone_hz) + 1e-20) /
                                     (bin_mag(tone_hz) + 1e-20))
            rep.check("harmonic distortion (2f+3f vs f)", thd_like < -20.0,
                      f"{thd_like:6.1f} dB   (want < -20)",
                      "" if thd_like < -20.0 else
                      "Large harmonics of a pure tone mean the chain is "
                      "nonlinear. Check clipping first, then the 24-in-32-bit "
                      "sample alignment. Note a phone speaker is itself "
                      "distorted, so re-test at a lower volume before blaming "
                      "the Pi.")
    else:
        rep.advise(
            "Run again with --tone 1000 while playing a 1 kHz sine from a "
            "phone (any tone-generator app, or a YouTube test tone). Without "
            "an external reference, the checks above can tell you the "
            "microphone is ALIVE but not that it is CORRECT. The tone test is "
            "the only one that validates the whole chain against something "
            "outside the Pi.")


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Bring-up test 1/5: verify the I2S microphone end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Typical week-1 sequence:\n"
               "  1. python firmware/bench/check_audio.py --list\n"
               "  2. python firmware/bench/check_audio.py --seconds 10\n"
               "  3. play a 1 kHz tone from a phone, then:\n"
               "     python firmware/bench/check_audio.py --tone 1000\n")
    p.add_argument("--list", action="store_true",
                   help="enumerate ALSA capture devices and exit")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="recording length (default 5). Longer = finer FFT "
                        "resolution and a better rate estimate.")
    p.add_argument("--rate", type=int, default=16000,
                   help="sample rate to request (default 16000, matching "
                        "firmware/config.yaml)")
    p.add_argument("--device", default=None,
                   help="ALSA device index or name (see --list)")
    p.add_argument("--tone", type=float, default=None, metavar="HZ",
                   help="assert the dominant peak is within 1%% of HZ. Play "
                        "this tone from a phone while the script records.")
    p.add_argument("--rate-tolerance", type=float, default=1.0, metavar="PCT",
                   help="allowed achieved-vs-configured rate error (default 1%%)")
    p.add_argument("--simulate", action="store_true",
                   help="run the analysis on synthetic data — no hardware, "
                        "shows what a PASS looks like")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if args.list:
        return list_devices()

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    rep = Report(f"CHECK 1/5 — AUDIO (INMP441 I2S microphone @ {args.rate} Hz)")
    rep.header()

    if args.simulate:
        rep.info("MODE: --simulate (synthetic signal; no hardware touched)")
        x, elapsed, n_req = simulate(args.seconds, args.rate, args.tone, args.seed)
    else:
        rep.info(f"Recording {args.seconds:.1f} s from "
                 f"device={device if device is not None else 'default'} ...")
        if args.tone:
            rep.info(f"Play a {args.tone:.0f} Hz tone NOW, near the microphone.")
        x, elapsed, n_req = record(args.seconds, args.rate, device)
        rep.info(f"done ({elapsed:.3f} s wall clock)")

    analyse(x, elapsed, n_req, args.rate, args.tone, rep, args.rate_tolerance)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(run_guarded(main))
