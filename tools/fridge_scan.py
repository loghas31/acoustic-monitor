#!/usr/bin/env python3
"""fridge_scan.py — phone recording in, verdict out. One command.

    python tools/fridge_scan.py ~/Downloads/fridge.m4a

That is the whole thing. It converts the phone's file, runs the real
learn->score pipeline, and prints a verdict in English.

WHAT THIS ACTUALLY DOES, so you can distrust it properly
--------------------------------------------------------
Three existing tools in a trench coat. It runs, in order:

  1. `ffmpeg`            — .m4a (or .mp3/.caf/anything) -> 16 kHz mono .wav.
                           Skipped if you hand it a .wav already.
  2. `tools/ingest.py`   — into this project's canonical format, with the
                           audit checks (clipping, DC, silence) that catch a
                           bad recording BEFORE you draw conclusions from it.
  3. `tools/phone_monitor.py`
                         — learn what this machine normally sounds like from
                           the first N windows, then score the rest against
                           it. Mic-only: no accelerometer, no bearing
                           geometry, no shaft speed needed.

Every one of those can be run by hand, and if something looks wrong you
should — this wrapper prints each command as it runs it, so you can copy any
step and re-run it in isolation.

THE ONE THING PEOPLE GET WRONG
-------------------------------
**This cannot judge a machine from a short recording.** It is not a tuner that
listens for a moment and tells you the note. It is an anomaly detector: it
learns *this specific machine's* normal, then flags departures from it. That
means a single recording has to be long enough to contain BOTH.

Default is 48 learn windows of 30 s = **24 minutes just to learn**, and it
scores whatever is left over. So:

    30 minutes of audio  ->  24 min learning, 6 min scored (12 windows)
    45 minutes of audio  ->  24 min learning, 21 min scored (42 windows)
    20 minutes of audio  ->  REFUSED, and it will tell you why

The 48-window floor is not arbitrary and is not a knob to turn down when you
are impatient: below it the measured held-out false-positive rate is 55-59 %,
i.e. the thing becomes a coin flip that sounds confident. `--learn-windows`
exists for debugging and will warn you loudly if you drop below the floor.

WHAT A CLEAN RESULT PROVES
---------------------------
Less than you would like. "Nothing anomalous" on a healthy fridge means the
detector did not fire on a machine that was fine — it does NOT mean it would
have caught a fault, because you have not shown it one. That is what Part B of
`FRIDGE_TEST.md` (deliberately loading the machine) is for. Read
`TESTS.md` before drawing conclusions.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON_FS = 16000            # firmware/config.yaml audio.sample_rate
WINDOW_S = 30.0             # phone_monitor default
LEARN_WINDOWS = 48          # the documented floor; see module docstring


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    print(f"\n\033[1m[{what}]\033[0m " + " ".join(str(c) for c in cmd),
          flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"\n{what} failed (exit {proc.returncode}). The command above "
                 f"is runnable on its own — re-run it to see the full error.")
    return proc


def to_wav(src: Path, out: Path) -> Path:
    """Phone formats -> canonical-rate mono WAV, via ffmpeg."""
    if src.suffix.lower() == ".wav":
        return src
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not installed, and it is needed to read "
                 f"'{src.suffix}' files.\n\n    brew install ffmpeg\n\n"
                 "(Or export/convert the recording to .wav yourself and pass "
                 "that instead.)")
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
          "-ar", CANON_FS, "-ac", 1, out], "convert")
    return out


def check_duration(wav: Path, learn_windows: int) -> float:
    """Fail EARLY and in English if the recording is too short to say anything.

    Deliberately before ingest and scoring rather than after: discovering at
    the end of a pipeline that there were no windows left to score is the kind
    of thing that makes people quietly lower --learn-windows instead of going
    back to the fridge for a longer recording.
    """
    # scipy, not the stdlib `wave` module: `wave` raises
    # "unknown format: 3" on 32-bit float WAV, which is what Audacity and
    # Logic export by default. This is the only entry point aimed at someone
    # who will not read the source, so a raw traceback here defeats the point.
    from scipy.io import wavfile
    rate, data = wavfile.read(wav)
    seconds = len(data) / float(rate)

    total = int(seconds // WINDOW_S)
    scored = total - learn_windows
    mins = seconds / 60.0

    print(f"\nRecording is {mins:.1f} min = {total} windows of {WINDOW_S:.0f}s.")
    if scored < 1:
        need = (learn_windows + 1) * WINDOW_S / 60.0
        sys.exit(
            f"\nTOO SHORT — nothing would be left to score.\n\n"
            f"  This recording:  {mins:.1f} min ({total} windows)\n"
            f"  Learning needs:  {learn_windows} windows "
            f"({learn_windows * WINDOW_S / 60.0:.0f} min)\n"
            f"  Left to score:   {max(scored, 0)} windows\n\n"
            f"You need at least {need:.0f} minutes, and {need + 15:.0f}+ is "
            f"much better.\n\nGo back and record for longer. Do NOT lower "
            f"--learn-windows to make this run: below {learn_windows} learn "
            f"windows the measured false-positive rate is 55-59%, so it would "
            f"produce a confident-looking answer that means nothing.")
    if scored < 6:
        print(f"\n  ⚠  Only {scored} windows will be scored. That is enough to "
              f"run,\n     but too few to conclude much. 20+ is comfortable.")
    return seconds


def preflight() -> int:
    """Prove this laptop can do the whole job, before 40 minutes are spent.

    Deliberately checks the things that actually go wrong in this order:
    maths libraries (a broken scipy is the usual one), then ffmpeg (needed
    only for phone formats, so its absence is a warning not a failure), then
    the real analysis pipeline end to end via `phone_monitor --self-test`.

    The last one matters most: it is not a smoke test of imports, it runs the
    genuine learn->score path on a known-healthy synthetic signal and checks
    it stays below threshold. If that passes, the only remaining variable is
    your recording.
    """
    ok = True

    print("\n1/3  maths libraries")
    try:
        import numpy, scipy, sklearn                      # noqa: F401
        print("     ok — numpy, scipy, scikit-learn all import")
    except ImportError as e:
        ok = False
        print(f"     MISSING: {e.name}\n"
              f"     fix:  pip install -r ml/requirements.txt")

    print("\n2/3  ffmpeg (needed to read .m4a from the phone)")
    if shutil.which("ffmpeg"):
        print("     ok")
    else:
        print("     NOT FOUND — you can still analyse .wav files, but not the\n"
              "     .m4a your phone produces.\n"
              "     fix:  brew install ffmpeg")

    print("\n3/3  the real analysis pipeline (this takes ~1 min)")
    proc = subprocess.run(
        [sys.executable, "tools/phone_monitor.py", "--self-test"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode == 0 and "PASS" in proc.stdout:
        print("     ok — learned a baseline and scored against it correctly")
    else:
        ok = False
        print("     FAILED. Last output:\n" +
              "\n".join(f"     {ln}" for ln in
                        (proc.stdout + proc.stderr).strip().splitlines()[-6:]))

    print("\n" + "=" * 62)
    if ok and shutil.which("ffmpeg"):
        print("Ready. Go and record 40 minutes.")
    elif ok:
        print("Almost — install ffmpeg, then you're ready to record.")
    else:
        print("Not ready. Fix the above before recording, not after.")
    print("=" * 62)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, nargs="?",
                    help="whatever came off your phone (.m4a, .wav, ...)")
    ap.add_argument("--preflight", action="store_true",
                    help="check this machine can run the analysis at all, "
                         "BEFORE you spend 40 minutes recording. Takes ~1 min.")
    ap.add_argument("--machine", default="fridge",
                    help="what you recorded, for the saved metadata")
    ap.add_argument("--learn-windows", type=int, default=LEARN_WINDOWS,
                    help=f"windows to learn from before scoring "
                         f"(default {LEARN_WINDOWS}; read the docstring "
                         f"before lowering this)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep the converted/canonical wavs for inspection")
    args = ap.parse_args(argv)

    if args.preflight:
        return preflight()
    if args.recording is None:
        ap.error("give me a recording, or --preflight to check the setup first")

    src = args.recording.expanduser()
    if not src.exists():
        sys.exit(f"No such file: {src}\n\nIf you AirDropped it, look in "
                 f"~/Downloads. Quote the path if it has spaces in it.")

    if args.learn_windows < LEARN_WINDOWS:
        print(f"\n  ⚠  --learn-windows {args.learn_windows} is below the "
              f"documented floor of {LEARN_WINDOWS}.\n     Held-out false-"
              f"positive rate below that floor was measured at 55-59%.\n"
              f"     Treat any result from this run as a debugging aid, not "
              f"evidence.")

    # A working directory PER RUN, not one shared by every run ever. The
    # shared version accumulated files indefinitely (57 MB of test leftovers
    # were sitting in the repo when this was found) and, worse, let one run's
    # output be picked up by a later run with a similar filename. Clearing it
    # at the start also means "what is in here" always describes this run.
    stem = src.stem.replace(" ", "_")
    work = ROOT / "data" / "_scan_work" / stem
    work.parent.mkdir(parents=True, exist_ok=True)

    # Two hardenings to the clear-at-start step above, both found the same
    # way: by executing this project's own test suite twice against a
    # working tree that already had a previous run's directories in it,
    # which is exactly what re-running the suite in a fresh sandbox container
    # does every time.
    #
    # (1) DELETING it can fail outright. Measured directly in this project's
    # own sandbox: `shutil.rmtree()`, plain `os.rmdir`/`os.unlink`, and a
    # shell `rm -rf` on the SAME directory all raise/fail with
    # `PermissionError: [Errno 1] Operation not permitted` — not a race, not
    # a permissions-bits problem (the directory is owned by the calling
    # user), but the mounted working tree refusing deletion outright. `mv`/
    # `os.rename` of the identical directory succeeds immediately. So
    # "cleared" now means MOVED aside, never deleted, which works everywhere
    # rename works (this sandbox, and a normal filesystem alike) and, as a
    # side effect, never destroys a previous run's intermediates — they land
    # in `output/_attic/scan_work/`, the same attic convention this repo
    # already uses elsewhere for retired scan-work leftovers.
    #
    # (2) Two invocations analysing files that happen to share a stem (a
    # phone camera roll reusing "recording.m4a"; this project's own test
    # suite reusing "r"/"short"/"decoytest") both compute the identical
    # `work` path. Nothing serialised the retire-then-recreate sequence
    # above, so a second process could retire the FIRST process's still-in-
    # use directory out from under it. An flock, keyed by stem, is held for
    # the WHOLE scan below (not just the retire step — a narrower lock would
    # still let a second process retire the directory mid-ingest), and is
    # released automatically when this process exits, including a crash or
    # Ctrl-C, so a killed scan can never leave a stale lock behind.
    lock_path = work.parent / f"{stem}.lock"
    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)

    if work.exists():
        attic = ROOT / "output" / "_attic" / "scan_work"
        attic.mkdir(parents=True, exist_ok=True)
        work.rename(attic / f"{stem}_{int(time.time())}_{os.getpid()}")
    work.mkdir(parents=True, exist_ok=True)

    wav = to_wav(src, work / f"{stem}_{CANON_FS}.wav")
    check_duration(wav, args.learn_windows)

    _run([sys.executable, "tools/ingest.py", wav,
          "--out-dir", work, "--stem", stem, "--mic-only",
          "--machine", args.machine, "--label", "unknown"], "ingest")

    # ANALYSE THE FILE THE USER GAVE US, OR STOP. The previous version fell
    # back to `sorted(work.glob(f"{stem}*.wav"))[-1]`, which is wrong twice
    # over: the glob is a prefix match, so for a recording called `r.wav` it
    # also matches `r25.wav` from an earlier run, and `[-1]` then picks the
    # OTHER one. Silently analysing a different recording than the one handed
    # over is close to the worst thing a tool like this can do — every number
    # downstream is real, internally consistent, and about the wrong machine.
    #
    # The per-run working directory (above) makes cross-run collisions
    # impossible; this makes the remaining case loud instead of clever.
    canonical = work / f"{stem}.wav"
    if not canonical.exists():
        produced = sorted(p.name for p in work.glob("*.wav"))
        sys.exit(
            f"\ningest did not produce the expected file.\n\n"
            f"  expected: {canonical.name}\n"
            f"  found:    {produced or 'nothing'}\n\n"
            f"Refusing to guess which of these is your recording — analysing "
            f"the wrong file would give real-looking numbers about the wrong "
            f"machine. Re-run the ingest command printed above on its own to "
            f"see what it did.")

    # --window-s is passed EXPLICITLY, not left to match by luck. check_duration
    # computes its arithmetic with WINDOW_S; phone_monitor has its own default.
    # An adversarial review changed phone_monitor's default to 60 s and this
    # wrapper sailed past its own length check, then died inside the analysis
    # with "need MORE than --learn-windows 48" AFTER the full ingest had run —
    # exactly the late failure check_duration exists to prevent.
    _run([sys.executable, "tools/phone_monitor.py", canonical,
          "--learn-windows", args.learn_windows,
          "--window-s", WINDOW_S], "analyse")

    # Both detectors, always, on one recording. They answer DIFFERENT questions
    # and neither subsumes the other: the learn->score path above finds a
    # machine DEPARTING from its own normal, which is useless if the machine
    # was already faulty when it learned. The cold-start screen below needs no
    # history at all, but only finds impulsive faults. Running one and not the
    # other leaves an obvious hole, and asking the user to remember which
    # intermediate file to point the second tool at is a good way to have it
    # never run.
    print("\n" + "=" * 74)
    print("SECOND OPINION — baseline-free screen (does not assume the machine")
    print("was healthy while it learned). See docs/COLD_START.md.")
    print("=" * 74)
    _run([sys.executable, "tools/cold_start_screen.py", canonical],
         "cold-start screen")

    print("\n" + "=" * 74)
    print("Read the three 'honest answers' above, then TESTS.md section 5.")
    print("A quiet result means the detector did not fire on this machine.")
    print("It does NOT mean it would catch a fault — you have not shown it one.")
    print("=" * 74)

    if not args.keep_intermediates:
        print(f"\n(intermediates in {work.relative_to(ROOT)} — "
              f"delete when done, or pass --keep-intermediates to keep them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
