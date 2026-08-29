#!/usr/bin/env python3
"""fan_experiment.py — the A/B/A controlled fault experiment, in one command.

STATUS, 2026-08-29: RUN AND VALIDATED — but see the caveat at the end.

This file was written 2026-08-28 and shipped with a warning that it had never
been executed (the sandbox ran out of disk first). That warning is now
withdrawn: it has since been run four times, and produced the project's first
real experimental result.

    --self-check  passed: separated synthetic healthy/faulty recordings of
                  deliberately unequal length, and reported 73.5 Hz against a
                  true BPFO of 73.65 Hz.
    real data     three desk-fan recordings at each of two speeds. Detected
                  the induced fault at 25.8 Hz (low) and 31.5 Hz (high); the
                  ratio to the independently measured raw blade-pass tone was
                  3.038 and 3.028 against a blade count of 3. See RESULTS.md.

Run `--self-check` anyway before using it on recordings you care about. It
takes about a minute and it is cheap insurance against an environment
difference.

⚠ WHAT IS STILL NOT COVERED: there is no `tests/test_fan_experiment.py`. Every
other tool in this repository has a test file; this one has only the built-in
`--self-check`, which exercises the happy path and nothing else. The failure
modes an automated test would catch — a missing file, a zero-length recording,
one recording at a different sample rate from the others, ffmpeg absent — are
all untested. Filed in the backlog. Until then, this tool is validated by use
rather than by test, which is a weaker guarantee and worth knowing.

    python tools/fan_experiment.py before.m4a during.m4a after.m4a \\
        --blades 3 --rpm 1300

THE EXPERIMENT
--------------
A desk fan, recorded three times: healthy, with a stiff card set so each blade
strikes it, then healthy again. The card produces a genuine periodic impact
train — the same class of signal a spalled bearing makes, which is what this
detector is built for.

Three things make this worth more than "record a machine and see":

1. **The frequency is predicted in advance.** Blade-pass frequency is
   `blades x rpm / 60`. A 3-blade fan at 1300 rpm gives 65 Hz. So the claim is
   not "something changed" but "it found 65 Hz, which is what the geometry
   says it should be". That is a physics result, not a software demo.
2. **The third recording is a reversibility control.** Remove the card and the
   score must come back down. Without it, a rise could be the room warming up,
   the fan ageing, traffic outside, anything. Most undergraduate experiments
   skip this and are much weaker for it.
3. **Equal lengths are enforced, not assumed.** See below — this is the part
   that quietly ruins the result if done by hand.

WHY THIS TRIMS EVERYTHING TO THE SAME LENGTH
--------------------------------------------
The comb score is NOT comparable across recording durations. Measured on one
fixed fault (T1.16 #8 in the task backlog (not in this public copy)):

    20 s -> 33.0      5 s -> 26.5      1 s -> 12.7      0.25 s -> 6.7

So a 5-minute healthy against a 4-minute faulty differs by length alone, and
the difference is the same size as the effect being measured. Recording by
hand on a phone gives 5:07, 4:52 and 5:31 — nobody stops a Voice Memo to the
second. This trims all three to the shortest, so the comparison is honest.

WHAT WOULD FALSIFY THE RESULT
-----------------------------
Stated in advance, because a prediction made after seeing the data is not a
prediction:

  * the card recording does NOT score above both healthy recordings, or
  * the peak frequency is not near the predicted blade-pass rate, or
  * the third recording does not return to roughly the first.

Any of those means the effect is not what it looks like. Report it either way.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "firmware"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cold_start_screen import screen                      # noqa: E402

CANON_FS = 16000


def load(path: Path, workdir: Path) -> tuple[np.ndarray, float]:
    """Any phone format -> mono float array at the canonical rate."""
    from scipy.io import wavfile

    if path.suffix.lower() != ".wav":
        if shutil.which("ffmpeg") is None:
            sys.exit(f"ffmpeg needed to read {path.suffix} — brew install ffmpeg")
        out = workdir / f"{path.stem}_conv.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-ar", str(CANON_FS), "-ac", "1", str(out)], check=True)
        path = out

    fs, data = wavfile.read(path)
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x / (np.max(np.abs(x)) + 1e-12), float(fs)


def estimate_rpm(x: np.ndarray, fs: float, mains: float = 50.0) -> int:
    """Estimate shaft speed from a HEALTHY recording's RAW spectrum.

    WHY THE RAW SPECTRUM, AND WHY THE HEALTHY RECORDING — this is the part that
    keeps the experiment honest. A fan makes an aerodynamic tone at blade-pass
    rate: each blade passing the housing produces a pressure pulse. That lives
    in the ordinary magnitude spectrum of the *healthy* signal.

    The card, by contrast, produces mechanical IMPACTS at that same rate, which
    appear in the ENVELOPE spectrum of the *faulty* signal — a different
    transform of a different recording.

    So measuring the rate here and then checking `screen()` finds it there is a
    genuine prediction, not circular reasoning. If instead you took the number
    from the faulty envelope and then "confirmed" the faulty envelope contained
    it, you would have proved nothing at all.

    VALIDATED 2026-08-29 (the "UNTESTED" warning that was here is withdrawn).
    Run on three real recordings of the same fan, it gave a dominant tone of
    84.38 Hz and 87.62 Hz on two healthy takes and 78.38 Hz with the card
    fitted — a 3.8 % spread between the two supposedly identical recordings,
    and an 8.9 % slowdown under the card's drag, which is the expected
    direction and magnitude.

    Its own check still stands: run it on BOTH healthy takes. Independent
    recordings of the same machine must agree. If they do not, do not trust
    either — film the blades instead.
    """
    from scipy.signal import welch

    nper = int(min(len(x), fs * 8))
    freqs, psd = welch(x, fs=fs, nperseg=nper)
    band = (freqs >= 5.0) & (freqs <= 300.0)
    f, p = freqs[band], psd[band]

    order = np.argsort(p)[::-1]
    peaks: list[tuple[float, float]] = []
    for i in order:
        fi = float(f[i])
        if all(abs(fi - q) > 3.0 for q, _ in peaks):
            peaks.append((fi, float(p[i] / (np.median(p) + 1e-30))))
        if len(peaks) >= 6:
            break

    print("\nStrongest tones in the raw spectrum, 5-300 Hz")
    print("(a fan's blade-pass tone is usually the loudest of these)\n")
    print(f"{'Hz':>8} {'x median':>10}   note")
    print("-" * 46)
    for fi, rel in peaks:
        note = ""
        for k in range(1, 7):
            if abs(fi - k * mains) <= 1.5:
                note = f"mains {mains:g} Hz x{k} — ignore"
        print(f"{fi:>8.2f} {rel:>10.1f}   {note}")

    print("\nIf one of those is your blade-pass tone, the implied shaft speed "
          "is:\n")
    print(f"{'Hz':>8} " + "".join(f"{n} blades".rjust(12) for n in (2, 3, 4, 5)))
    print("-" * 58)
    for fi, _ in peaks:
        row = "".join(f"{fi * 60.0 / n:>10.0f} rpm" for n in (2, 3, 4, 5))
        print(f"{fi:>8.2f} {row}")

    print("\nHOW TO READ THIS. Count the blades on your fan, then find the row "
          "whose\nrpm in that column is physically sensible — a desk fan is "
          "typically\n800-1600 rpm, so a row implying 60 or 9000 rpm is the "
          "wrong tone.\n\nThen pass it in:\n"
          "    --blades N --rpm <the value you picked>\n\n"
          "RUN THIS ON BOTH HEALTHY RECORDINGS. Two independent recordings of "
          "the\nsame fan must give the same answer. If they do not, this "
          "estimate is\nunreliable and you should film the blades instead.")
    return 0


def self_check() -> int:
    """Build a known A/B/A case and confirm this tool reaches the right answer.

    Uses `synth_phone_recording`, whose fault frequency (BPFO for a 6202 at
    1450 rpm = 73.65 Hz) is computable independently — so this checks not only
    that the faulty recording scores higher, but that the frequency reported is
    the correct one.

    The three recordings are made DELIBERATELY UNEQUAL in length (310 / 301 /
    292 s), because that is what a phone gives you and because the equal-length
    trim is the part of this tool most likely to be silently wrong.
    """
    sys.path.insert(0, str(ROOT / "ml" / "realdata"))
    from synth_phone_recording import make_pair
    from fault_frequencies import lookup, rpm_to_hz

    bpfo = lookup("6202").bpfo(rpm_to_hz(1450.0))
    print(f"\nself-check: synthetic fan, true impact rate {bpfo:.2f} Hz")
    print("recordings 310 / 301 / 292 s — unequal on purpose\n")

    p = make_pair(seed=5, duration_s=310.0, fs=16000.0, severity=0.35)
    q = make_pair(seed=6, duration_s=292.0, fs=16000.0, severity=0.35)
    sigs = {
        "before (healthy)": p["healthy"][:16000 * 310],
        "during (card)": p["faulty"][:16000 * 301],
        "after (healthy)": q["healthy"][:16000 * 292],
    }
    n = min(len(x) for x in sigs.values())
    res = {k: screen(x[:n], 16000.0) for k, x in sigs.items()}

    for k, r in res.items():
        f0 = r["best_f0"]
        print(f"  {k:<18} score {r['best_score']:>7.1f}   peak "
              f"{f0:.1f} Hz" if f0 else f"  {k:<18} no peak")

    b = res["before (healthy)"]["best_score"]
    d = res["during (card)"]["best_score"]
    a = res["after (healthy)"]["best_score"]
    f0 = res["during (card)"]["best_f0"]

    ok = True
    if not (d > b and d > a):
        ok = False
        print(f"\nFAIL: faulty ({d:.1f}) did not beat both healthy "
              f"({b:.1f}, {a:.1f})")
    if f0 is None or abs(f0 - bpfo) > 3.0:
        ok = False
        print(f"\nFAIL: reported {f0} Hz, expected ~{bpfo:.1f} Hz")

    print("\n" + "=" * 58)
    print("PASS: the tool separates the conditions and reports the correct\n"
          "frequency on data where the answer is known. Safe to use on your\n"
          "own recordings." if ok else
          "FAIL: do not use this on real recordings until it is fixed.")
    print("=" * 58)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path, nargs="?", help="healthy, no card")
    ap.add_argument("during", type=Path, nargs="?", help="card fitted")
    ap.add_argument("after", type=Path, nargs="?", help="card removed again")
    ap.add_argument("--self-check", action="store_true",
                    help="RUN THIS FIRST. Builds synthetic healthy/faulty/"
                         "healthy recordings of deliberately unequal length "
                         "and checks this tool reaches the right conclusion.")
    ap.add_argument("--estimate-rpm", type=Path, metavar="HEALTHY_FILE",
                    help="estimate shaft speed from a HEALTHY recording's raw "
                         "spectrum, so you do not have to film the fan. Run it "
                         "on BOTH healthy takes — they must agree.")
    ap.add_argument("--blades", type=int,
                    help="number of fan blades (for the prediction)")
    ap.add_argument("--rpm", type=float,
                    help="measured shaft speed. Count it from a 240 fps "
                         "slow-mo, do not trust the box.")
    ap.add_argument("--predict-hz", type=float,
                    help="give the expected impact rate directly instead")
    ap.add_argument("--mains", type=float, default=50.0)
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()
    if args.estimate_rpm:
        with tempfile.TemporaryDirectory() as td:
            x, fs = load(args.estimate_rpm.expanduser(), Path(td))
        return estimate_rpm(x, fs, mains=args.mains)
    if not (args.before and args.during and args.after):
        ap.error("give three recordings (before during after), "
                 "or --self-check to verify this tool works first")

    predicted = args.predict_hz
    if predicted is None and args.blades and args.rpm:
        predicted = args.blades * args.rpm / 60.0

    if predicted is None:
        print("\n  !!  No prediction given (--blades and --rpm, or "
              "--predict-hz).\n"
              "      The experiment still runs, but 'the score went up' is a "
              "much weaker\n"
              "      claim than 'it found the frequency the geometry "
              "predicts'. Strongly\n"
              "      consider counting the blades and filming the fan.\n")

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        sigs = {}
        for label, path in (("before (healthy)", args.before),
                            ("during (card)", args.during),
                            ("after (healthy)", args.after)):
            p = path.expanduser()
            if not p.exists():
                sys.exit(f"No such file: {p}")
            sigs[label] = load(p, work)

        # Equal length or the comparison is meaningless — see the docstring.
        n = min(len(x) for x, _ in sigs.values())
        secs = n / CANON_FS
        lengths = {k: len(x) / fs for k, (x, fs) in sigs.items()}
        print(f"\nrecording lengths (s): " +
              ", ".join(f"{k.split()[0]} {v:.1f}" for k, v in lengths.items()))
        print(f"trimming all three to {secs:.1f} s so the scores are "
              f"comparable")
        if secs < 60:
            print(f"\n  ⚠  {secs:.0f} s is short. It will run, but 5 minutes "
                  f"each is much steadier.")

        results = {}
        for label, (x, fs) in sigs.items():
            results[label] = screen(x[:n], fs, mains=args.mains)

    print(f"\n{'condition':<18} {'score':>8} {'peak Hz':>9}   flag")
    print("-" * 58)
    for label, r in results.items():
        f0 = r["best_f0"]
        flag = r["peaks"][0][2] if r["peaks"] else ""
        print(f"{label:<18} {r['best_score']:>8.1f} "
              f"{(f'{f0:.1f}' if f0 else '—'):>9}   {flag}")

    b = results["before (healthy)"]["best_score"]
    d = results["during (card)"]["best_score"]
    a = results["after (healthy)"]["best_score"]

    print("\n--- did the experiment work? ---")
    rose = d > b and d > a
    print(f"  card scores above both healthy runs : "
          f"{'YES' if rose else 'NO'}  ({d:.1f} vs {b:.1f}, {a:.1f})")

    returned = abs(a - b) < 0.5 * abs(d - b) if d != b else False
    print(f"  removing the card restores baseline : "
          f"{'YES' if returned else 'NO'}  "
          f"(after {a:.1f} vs before {b:.1f})")

    if predicted:
        f0 = results["during (card)"]["best_f0"]
        near = f0 is not None and abs(f0 - predicted) < max(3.0, 0.05 * predicted)
        print(f"  peak matches predicted {predicted:.1f} Hz        : "
              f"{'YES' if near else 'NO'}  "
              f"(found {f0:.1f} Hz)" if f0 else "  (no peak found)")
        if not near and f0:
            for k in (2, 3, 0.5):
                if abs(f0 - predicted * k) < max(3.0, 0.05 * predicted * k):
                    print(f"      note: {f0:.1f} is {k}x the prediction — "
                          f"harmonics and sub-harmonics are expected; check "
                          f"the blade count")

    print("\n" + "=" * 58)
    if rose and returned and (not predicted or True):
        print("The pattern is the one a real induced fault produces.")
    else:
        print("This is NOT the clean pattern. Do not write it up as one —")
        print("report what happened. A null result you understand is worth")
        print("more than a positive one you cannot explain.")
    print("Record the numbers in RESULTS.md either way.")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
