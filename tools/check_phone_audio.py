#!/usr/bin/env python3
"""
check_phone_audio.py — decide whether a phone recording is USABLE by this
detector, before you spend an hour recording with it.

The question this answers is not "is the machine faulty". It is: **does your
phone destroy the information the detector needs?** A phone is built to record
voices, and the processing that flatters speech is actively hostile here.

THE ONE THAT MATTERS: AUTOMATIC GAIN CONTROL
--------------------------------------------------------------------------
When the sound gets louder, a phone turns its gain down. For speech that is
helpful. For this project it is potentially fatal, because `audio_logrms` is
not merely one of 37 features — it is **one of the three dimensions of
`baseline.operating_point`**, the space the regime clustering runs in. With AGC
active:

  * absolute level stops meaning "how loud the machine is" and starts meaning
    "what the phone decided to do", so regimes may cluster on the phone's
    behaviour rather than the machine's; and
  * **a machine getting louder — the fault you are hunting — is exactly what
    AGC compensates away.**

Same shape as finding F10, where `channel_stats` measured gravity instead of
vibration: a feature computed correctly that measures the wrong thing. The
simulator cannot surface it, because simulated audio has no AGC.

TWO MODES
---------
**Distance test (decisive — use this one).** Record ~20 s of a steady source
(a running tap, a fan) at about 10 cm, then WITHOUT STOPPING move to about
40 cm and hold for another ~20 s. Sound pressure falls roughly as 1/r, so
quadrupling the distance should cost about 12 dB:

    python tools/check_phone_audio.py rec.m4a --distance-test

**Single file (weak, advisory).** With no distance step, all this can do is
report how much the level moves over the recording. A steady machine in a
quiet room genuinely produces a flat level, so a flat result here is NOT
evidence of AGC — it is merely uninformative. Never conclude "no AGC" from
this mode.

    python tools/check_phone_audio.py rec.m4a

Verify the tool itself against known-AGC and known-clean signals, no phone
needed:

    python tools/check_phone_audio.py --self-test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
for sub in ("firmware", "ml"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.append(p)

# Physics: sound pressure from a small source falls as 1/r, so 10 cm -> 40 cm
# is 20*log10(4) = 12.04 dB. Real rooms are reverberant and hands are not
# rulers, so the pass/fail bands below are deliberately wide.
EXPECTED_DROP_DB = 20.0 * np.log10(4.0)
AGC_SUSPECT_DB = 4.0      # below this, the level barely moved: AGC likely
AGC_CLEAR_DB = 8.0        # above this, the level tracked distance: AGC unlikely


def _load(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Read any phone format to mono float. Uses tools/ingest.py's reader when
    the file is already a WAV; otherwise shells out to ffmpeg for .m4a/.mp3,
    which is what a Voice Memo actually is."""
    from scipy.io import wavfile
    if path.suffix.lower() not in (".wav", ".wave"):
        tmp = Path(tempfile.mkdtemp()) / "conv.wav"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
               "-ac", "1", "-ar", str(target_sr), str(tmp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            raise SystemExit(
                "ffmpeg not found, and this file is not a WAV.\n"
                "  macOS:  brew install ffmpeg\n"
                "  Or export/convert the recording to WAV first — "
                "tools/ingest.py also accepts WAV directly.")
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"ffmpeg could not read {path.name}:\n"
                             f"{e.stderr.decode()[:400]}")
        path = tmp
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    return np.asarray(x, dtype=np.float64), int(sr)


def level_envelope(x: np.ndarray, sr: int, hop_s: float = 0.25) -> np.ndarray:
    """Short-term RMS in dB, one value per `hop_s`. DC-removed first — F10's
    lesson: an offset inflates RMS and would mask exactly what we measure."""
    n = max(1, int(hop_s * sr))
    trimmed = x[:len(x) - len(x) % n].reshape(-1, n)
    ac = trimmed - trimmed.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(ac ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def lossy_cutoff_hz(x: np.ndarray, sr: int = 16000,
                    drop_db: float = 40.0) -> float | None:
    """Find a lossy-codec brick wall, or None if the spectrum runs to Nyquist.

    Lossy encoders save bits by discarding the top of the spectrum entirely.
    Measured on ffmpeg AAC at Voice Memos' "Compressed" bitrate: flat to
    ±1 dB up to 10 kHz, then **−78 dB** above it. Not a gentle roll-off — a
    cliff. At 128 kbps the same file loses only 0.3 dB everywhere.

    That matters here because the system overview (not in this public copy) puts machine resonances
    anywhere in **1–20 kHz**. A machine whose resonance sits above the cliff
    is not merely attenuated, it is *absent*: the envelope analysis has
    nothing to demodulate and the recording looks like a quiet healthy
    machine. Nobody can hear the difference, which is what makes it dangerous.

    Returns the frequency where the cliff starts, in Hz.
    """
    f, p = welch(x, fs=sr, nperseg=8192)
    band = 1000.0
    edges = np.arange(1000.0, sr / 2, band)
    if len(edges) < 4:
        return None
    powers = []
    for lo in edges:
        m = (f >= lo) & (f < lo + band)
        powers.append(float(np.trapezoid(p[m], f[m])) if m.any() else 0.0)
    powers = np.asarray(powers)
    ref = float(np.median(powers[:max(2, len(powers) // 3)]))   # the flat part
    if ref <= 0:
        return None
    rel_db = 10.0 * np.log10(np.maximum(powers, 1e-30) / ref)
    dead = np.where(rel_db < -drop_db)[0]
    if len(dead) == 0:
        return None
    first = int(dead[0])
    # Require the cliff to persist — a single dead band could be a notch in the
    # machine's own spectrum rather than a codec limit.
    if not np.all(rel_db[first:] < -drop_db):
        return None
    return float(edges[first])


def distance_test(env: np.ndarray) -> dict:
    """Compare the first third against the last third, skipping the middle
    (that is where the hand is moving, and handling noise lives there)."""
    k = len(env) // 3
    if k < 4:
        raise SystemExit("recording too short for a distance test — "
                         "aim for ~20 s near and ~20 s far.")
    near, far = env[:k], env[-k:]
    drop = float(np.median(near) - np.median(far))
    return {"near_db": float(np.median(near)), "far_db": float(np.median(far)),
            "drop_db": drop, "expected_db": float(EXPECTED_DROP_DB)}


def report(x: np.ndarray, sr: int, do_distance: bool) -> int:
    env = level_envelope(x, sr)
    span = float(np.percentile(env, 95) - np.percentile(env, 5))
    dc = float(np.mean(x))
    rms = float(np.sqrt(np.mean((x - dc) ** 2)))
    clipped = float(np.mean(np.abs(x) >= 0.999))

    print(f"  duration        : {len(x)/sr:8.1f} s at {sr} Hz")
    print(f"  level p5-p95    : {span:8.1f} dB")
    print(f"  DC offset       : {dc:+8.5f}  ({dc/max(rms,1e-12):+.2%} of RMS)")
    print(f"  clipped samples : {clipped:8.2%}")

    verdict = 0
    if sr < 16000:
        print(f"\n  ✗ SAMPLE RATE TOO LOW. {sr} Hz gives you {sr/2:.0f} Hz of "
              f"bandwidth; the resonances this detector uses run 1-20 kHz. "
              f"Record at 44.1 or 48 kHz.")
        verdict = 1
    if clipped > 0.001:
        print(f"\n  ✗ CLIPPING at {clipped:.2%}. A clipped peak is a flat top, "
              f"and a flat top is a broadband impulse — indistinguishable from "
              f"the bearing impacts being hunted. Move further away or lower "
              f"the input level.")
        verdict = 1
    cutoff = lossy_cutoff_hz(x, sr)
    if cutoff is not None and cutoff < 0.45 * sr:
        print(f"\n  ✗ LOSSY COMPRESSION BRICK WALL at ~{cutoff/1000:.0f} kHz. "
              f"Everything above it is gone (measured -78 dB on 32 kbps AAC, "
              f"not attenuated — absent).")
        print(f"    Machine resonances run 1-20 kHz. If this machine's "
              f"resonance is above {cutoff/1000:.0f} kHz the recording cannot "
              f"show it, and a faulty machine will look quiet and healthy.")
        print(f"    Fix: iPhone Settings > Voice Memos > Audio Quality > "
              f"**Lossless**, or use a recorder app that writes WAV. At "
              f"128 kbps the same file loses only 0.3 dB; it is the low "
              f"bitrate, not AAC itself.")
        verdict = 1

    if abs(dc) > 0.05 * max(rms, 1e-12):
        print(f"\n  ⚠ DC offset is {dc/max(rms,1e-12):.1%} of RMS. Harmless here "
              f"(ingest.py removes it, and F10 made channel_stats DC-immune), "
              f"but it means the capture chain is not centred.")

    if do_distance:
        d = distance_test(env)
        print(f"\n  DISTANCE TEST")
        print(f"    near (first third) : {d['near_db']:7.1f} dB")
        print(f"    far  (last third)  : {d['far_db']:7.1f} dB")
        print(f"    measured drop      : {d['drop_db']:7.1f} dB")
        print(f"    expected (1/r, 4x) : {d['expected_db']:7.1f} dB")
        if d["drop_db"] < AGC_SUSPECT_DB:
            print(f"\n  ✗ AGC IS ALMOST CERTAINLY ON. The level barely moved "
                  f"when the distance quadrupled.")
            print(f"    Absolute level is meaningless in your recordings, so "
                  f"audio_logrms and the operating-point clustering built on "
                  f"it are unreliable.")
            print(f"    Options, best first: (1) a recorder app with a raw / "
                  f"measurement mode; (2) proceed but report every result with "
                  f"'AGC on' attached — the envelope and band-RATIO features "
                  f"survive a slowly-varying gain far better than absolute RMS "
                  f"does; (3) use spectral shape only, not anomaly scoring.")
            verdict = 1
        elif d["drop_db"] < AGC_CLEAR_DB:
            print(f"\n  ⚠ AMBIGUOUS. Some level tracking, but less than 1/r "
                  f"predicts. Could be partial AGC, could be a reverberant "
                  f"room (reflections fill in the far field). Repeat outdoors "
                  f"or in a soft-furnished room before trusting it.")
        else:
            print(f"\n  ✓ AGC APPEARS OFF. Level tracked distance close to "
                  f"1/r, so absolute level carries real information.")
    else:
        print(f"\n  No distance test requested, so nothing here can rule AGC "
              f"in or out.")
        if span < 2.0:
            print(f"  The level span is only {span:.1f} dB. That is consistent "
                  f"with AGC — and equally consistent with a steady machine in "
                  f"a quiet room. **Not evidence.** Run --distance-test.")
    return verdict


def _self_test() -> int:
    """Prove the AGC detector actually detects AGC, on signals where the
    answer is known. Without this the tool is just an opinion."""
    sr, dur = 16000, 60.0
    rng = np.random.default_rng(0)
    n = int(sr * dur)
    t = np.arange(n) / sr
    # A steady source that gets 12 dB quieter half way through: exactly what
    # moving 10 cm -> 40 cm does.
    base = rng.normal(0, 1, n) * 0.05
    gain = np.where(t < dur / 2, 1.0, 10 ** (-EXPECTED_DROP_DB / 20.0))
    clean = base * gain

    # The same signal through a crude AGC: normalise every 0.5 s block back to
    # a target level. This is what "the phone turns the gain down" looks like.
    blk = int(0.5 * sr)
    agc = clean.copy()
    for i in range(0, n - blk, blk):
        seg = agc[i:i + blk]
        r = np.sqrt(np.mean(seg ** 2))
        if r > 1e-9:
            agc[i:i + blk] = seg * (0.05 / r)

    ok = True
    for name, sig, should_flag in (("clean (no AGC)", clean, False),
                                   ("AGC applied", agc, True)):
        d = distance_test(level_envelope(sig, sr))
        flagged = d["drop_db"] < AGC_SUSPECT_DB
        good = flagged == should_flag
        ok &= good
        print(f"  {name:16s} drop {d['drop_db']:6.1f} dB  "
              f"-> {'AGC FLAGGED' if flagged else 'no AGC'}  "
              f"{'OK' if good else 'WRONG'}")
    print("\n  self-test PASSED" if ok else "\n  self-test FAILED")
    print("  (synthetic signals — this checks the detector, not your phone.)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, nargs="?")
    ap.add_argument("--distance-test", action="store_true",
                    help="the recording is near-then-far; compare the halves")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return _self_test()
    if a.recording is None:
        ap.error("give a recording, or use --self-test")
    if not a.recording.exists():
        print(f"FAIL: {a.recording} not found", file=sys.stderr)
        return 1

    x, sr = _load(a.recording)
    print(f"\n{a.recording.name}")
    return report(x, sr, a.distance_test)


if __name__ == "__main__":
    raise SystemExit(main())
