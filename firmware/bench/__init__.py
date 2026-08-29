"""Week-1 hardware bring-up toolkit.

Run these ON THE PI, in this order, before you trust a single recording:

    python firmware/bench/check_audio.py            # is the microphone real?
    python firmware/bench/check_audio.py --tone 1000
    python firmware/bench/check_accel.py            # is the accelerometer real?
    python firmware/bench/check_mount.py            # what does the MACHINE do?
    python firmware/bench/record_session.py --machine "motor-1" --label healthy
    python firmware/bench/selftest.py               # all of the above, one screen

Every script:
  * takes --help,
  * prints PASS/FAIL lines each carrying a NUMBER (a check with no number is
    an opinion, not a measurement),
  * exits 0 on pass, 1 on fail, 2 when the hardware simply is not present —
    and in that last case prints a plain-English "here is what to plug in"
    block instead of a traceback,
  * accepts --simulate, which runs the identical analysis on synthetic data so
    you can see what a PASS looks like (and so the DSP is testable in CI).
"""
