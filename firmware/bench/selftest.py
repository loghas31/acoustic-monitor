"""
selftest.py — CHECK 5/5: run the whole bring-up sequence and print one verdict.

    python firmware/bench/selftest.py            # on the Pi, sensor attached
    python firmware/bench/selftest.py --skip-mount   # bench test, no machine

This is the script you run when hardware arrives, and again every time you
touch the wiring. It is deliberately the ONLY thing the hardware design notes (not in this public copy)
asks a student to remember.

ORDER MATTERS AND IS NOT ARBITRARY:
  1. audio     — can we hear anything at the right rate?
  2. accel     — is the chip alive and correctly scaled? (gravity check)
  3. mount     — where does THIS machine actually resonate?
Each stage's assumptions depend on the previous one passing. A failure stops
the sequence rather than cascading confusing errors: if the sample rate is
wrong, the resonance you measure afterwards is wrong too, and being told
"resonance = 3.1 kHz" when it is really 4.5 kHz is worse than being told
nothing.

Exit codes: 0 all passed, 1 something failed, 2 no hardware present,
3 aborted. Suitable for use in a shell script or a CI runner on the Pi.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

from bench_common import (WIDTH, EXIT_OK, EXIT_FAIL,  # noqa: E402
                          EXIT_NO_HARDWARE, EXIT_ABORT)

STAGES = [
    ("audio", "check_audio.py", []),
    ("accel", "check_accel.py", []),
    ("mount", "check_mount.py", []),
]


def run_stage(script: str, extra: list[str]) -> int:
    proc = subprocess.run([sys.executable, str(BENCH / script), *extra])
    return proc.returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-mount", action="store_true",
                   help="skip the tap test (no machine attached yet)")
    p.add_argument("--skip-accel", action="store_true",
                   help="skip the accelerometer (mic-only build)")
    p.add_argument("--keep-going", action="store_true",
                   help="run all stages even after a failure")
    args = p.parse_args(argv)

    skip = set()
    if args.skip_mount:
        skip.add("mount")
    if args.skip_accel:
        skip.add("accel")

    print("=" * WIDTH)
    print("ACOUSTIC MONITOR — HARDWARE BRING-UP SELF TEST")
    print("=" * WIDTH)
    print("Runs every bench check in dependency order. Stop and fix the first")
    print("failure before trusting anything below it.")
    print()

    results: dict[str, int] = {}
    for name, script, extra in STAGES:
        if name in skip:
            results[name] = -1
            print(f"\n>>> SKIPPED: {name}\n")
            continue
        print(f"\n>>> STAGE: {name}\n")
        rc = run_stage(script, extra)
        results[name] = rc
        if rc != EXIT_OK and not args.keep_going:
            why = ("no sensor attached" if rc == EXIT_NO_HARDWARE
                   else "aborted" if rc == EXIT_ABORT else "checks failed")
            print(f"\n>>> STOPPING: '{name}' did not pass ({why}, exit {rc}).")
            print(">>> Fix this before running the later stages — their results")
            print(">>> depend on it and would mislead you.")
            break

    print()
    print("=" * WIDTH)
    print("SELF TEST SUMMARY")
    print("=" * WIDTH)
    for name, _, _ in STAGES:
        rc = results.get(name)
        status = {None: "not run", -1: "skipped", EXIT_OK: "PASS",
                  EXIT_NO_HARDWARE: "NO HARDWARE (nothing attached)",
                  EXIT_ABORT: "aborted"}.get(rc, f"FAIL ({rc})")
        print(f"  {name:<10s} {status}")
    print("=" * WIDTH)

    # "No hardware" is a different situation from "hardware is broken", and
    # conflating them would send a student debugging code that is fine.
    missing = [n for n, rc in results.items() if rc == EXIT_NO_HARDWARE]
    failed = [n for n, rc in results.items()
              if rc not in (EXIT_OK, EXIT_NO_HARDWARE, -1, None)]
    if failed:
        print(f"\nFailed: {', '.join(failed)}. See the hardware design notes (not in this public copy).")
        return EXIT_FAIL
    if missing:
        print(f"\nNo sensor detected for: {', '.join(missing)}.")
        print("The software is fine — there is nothing plugged in yet.")
        print("Wiring guide: the hardware design notes (not in this public copy)")
        return EXIT_NO_HARDWARE
    print("\nAll attempted stages passed. You are cleared to record a")
    print("session:  python firmware/bench/record_session.py --help")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
