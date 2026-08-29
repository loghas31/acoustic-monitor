"""Guards for `tools/fridge_scan.py` — the one command Logan actually runs.

This wrapper is the only entry point in the repo aimed at someone who is not
going to read the source first. That makes its FAILURE modes more important
than its success mode: a confusing error here sends someone back to the fridge
for another 40 minutes, or worse, quietly produces a number that means nothing.

Both halves are tested, and the first half was missing until an adversarial
review proved it:

  * THE SUCCESS PATH. A real 26-minute recording through the real ingest, the
    real learn->score pipeline and the real cold-start screen, asserting the
    exit code. This was absent — the reviewer replaced `ingest.py` and
    `phone_monitor.py` with stubs that exit 3 and all seven tests still
    passed, in 1.25 s instead of 31 s. A wrapper whose only job is
    orchestrating two tools had zero coverage of the orchestration.
  * THE REFUSALS, which for this tool matter as much: too-short recordings
    refused BEFORE any analysis with the arithmetic shown, because the
    tempting fix (lower --learn-windows) is the wrong one; dropping below the
    48-window floor warns loudly rather than silently producing a
    confident-looking verdict at a measured 55-59 % FPR; a missing file says
    where to look; a stale file from an earlier run can never be analysed in
    place of the one handed over.

Yes, the success-path test takes ~40 s. That is the correct price. The
alternative was a suite that passed while the tool did nothing.
"""

from __future__ import annotations

import fcntl
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "fridge_scan.py"


def _wav(path: Path, minutes: float, fs: int = 16000) -> Path:
    """A silent-ish WAV of a given length. Content is irrelevant — every test
    here is about the length check, which runs before anything reads samples."""
    n = int(minutes * 60 * fs)
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(n) * 1000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data.tobytes())
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_the_whole_pipeline_actually_runs_and_produces_a_verdict(tmp_path):
    """THE MISSING TEST. An adversarial review replaced `ingest.py` and
    `phone_monitor.py` with stubs that print SABOTAGE and exit 3 — and all
    seven tests in this file still passed, in 1.25 s instead of 31 s. Every
    test here checked a refusal; nothing checked that the tool WORKS.

    `test_the_boundary_is_where_the_docstring_says_it_is[25.0-True]` looked
    like a success-path test but only asserted `"TOO SHORT" not in output`,
    which a crashed run also satisfies.

    So: a real 26-minute recording, the real ingest, the real learn->score
    pipeline, the real cold-start screen. Assert the exit code and that each
    stage actually reported. Slow (~40 s) and worth it — this is the one path
    a user runs.
    """
    wav = _wav(tmp_path / "endtoend.wav", 26)
    proc = _run(str(wav))
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"pipeline failed:\n{out[-3000:]}"
    assert "[ingest]" in out and "[analyse]" in out
    assert "[cold-start screen]" in out, (
        "the baseline-free second opinion did not run — it is the half that "
        "still works when the machine was already faulty during learning")
    assert "windows" in out and "learn" in out


def test_stale_files_from_an_earlier_run_cannot_be_analysed(tmp_path):
    """Analysing the WRONG recording is the worst failure this tool can have:
    every number downstream is real, self-consistent, and about a different
    machine. The old code globbed `{stem}*.wav` and took the last match, so a
    recording called `r.wav` could be silently analysed as `r25.wav` left by an
    earlier run.

    Tested behaviourally rather than by grepping the source — the first attempt
    at this test asserted the glob string was absent from the file, and failed
    because the COMMENT explaining the old bug contains it. Checking behaviour
    cannot be fooled by prose.

    Plant a decoy in the per-run working directory, run, and confirm it is
    gone: the directory is cleared at the start of every run, so no earlier
    file can survive to be picked up.
    """
    wav = _wav(tmp_path / "decoytest.wav", 26)
    work = ROOT / "data" / "_scan_work" / "decoytest"
    work.mkdir(parents=True, exist_ok=True)
    decoy = work / "decoytest_STALE.wav"
    _wav(decoy, 1)
    assert decoy.exists()

    proc = _run(str(wav))
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    assert not decoy.exists(), (
        "a file from a previous run survived into this one; that is how the "
        "wrong recording gets analysed")


def test_a_too_short_recording_is_refused_before_any_analysis(tmp_path):
    """The important one. 10 minutes cannot answer anything: learning alone
    needs 24. It must say so in minutes, not fail somewhere downstream with an
    empty-array error that looks like a bug in the detector."""
    proc = _run(str(_wav(tmp_path / "short.wav", 10)))
    assert proc.returncode != 0
    assert "TOO SHORT" in proc.stdout + proc.stderr


def test_a_repeat_scan_of_the_same_stem_does_not_choke_on_a_leftover_directory(tmp_path):
    """FAILS-ON-OLD-CODE (against `shutil.rmtree()`). The first scan of a
    stem creates `data/_scan_work/<stem>`; every later scan of the SAME stem
    must clear it before reusing it. The old code did that with
    `shutil.rmtree()`, which raises `PermissionError: Operation not
    permitted` on this project's own mounted working tree — confirmed
    directly, not guessed: a plain shell `rm -rf` on the identical directory
    fails the exact same way, so this is a real property of the mount, not a
    Python quirk or a race between two processes (an earlier version of this
    fix assumed the latter from a single observed traceback; re-running it
    deterministically, alone, showed the failure every time, with no second
    process involved). `os.rename` of the same directory succeeds
    immediately, which is what main() now does instead of deleting — the
    retired directory lands in `output/_attic/scan_work/` rather than being
    destroyed.

    Two real invocations, same stem, back to back: the first creates the
    directory, the second must clear (by retiring) it without crashing.
    """
    stem = "repeatscan"
    wav = _wav(tmp_path / f"{stem}.wav", 10)  # too short -> fast, both times

    first = _run(str(wav))
    assert first.returncode != 0
    assert "TOO SHORT" in first.stdout + first.stderr

    work = ROOT / "data" / "_scan_work" / stem
    assert work.exists(), "first scan should have left its working directory behind"

    second = _run(str(wav))
    out = second.stdout + second.stderr
    assert "Operation not permitted" not in out, (
        "second scan of the same stem choked clearing the first scan's "
        f"leftover directory:\n{out[-1500:]}")
    assert second.returncode != 0
    assert "TOO SHORT" in out


def test_concurrent_scans_of_the_same_stem_do_not_race(tmp_path):
    """FAILS-ON-OLD-CODE (against the pre-lock main()). Two processes both
    computing `work = .../data/_scan_work/<stem>` with no lock between them
    could retire/recreate that directory out from under one another — a
    second, narrower risk from the same root cause as the leftover-directory
    test above, not proven to have fired for real on its own (the one
    concurrent-session traceback this project saw is fully explained by the
    deletion failure that test reproduces deterministically), but cheap to
    close given main() already has to touch this directory carefully.

    A true race is non-deterministic and depends on filesystem/OS timing a
    CI runner will not reproduce reliably, so this test does not try to win
    one. It takes the SAME lock fridge_scan.py now takes before touching
    `work`, holds it from outside the subprocess, and checks that a
    concurrent invocation blocks instead of barging in. Against the old
    code (no lock at all) the subprocess would run to completion regardless
    of this held lock, so the "still running" assertion below is the part
    that actually distinguishes fixed from broken.
    """
    stem = "racetest"
    work = ROOT / "data" / "_scan_work" / stem
    work.parent.mkdir(parents=True, exist_ok=True)
    lock_path = work.parent / f"{stem}.lock"

    wav = _wav(tmp_path / f"{stem}.wav", 10)  # too short -> fast refusal path

    held = open(lock_path, "w")
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT), str(wav)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        time.sleep(2.0)  # give it time to reach, and block on, the lock
        assert proc.poll() is None, (
            f"fridge_scan.py finished while this test still held "
            f"{stem}.lock exclusively -- it is not taking the lock at all, "
            f"so two concurrent scans of the same stem can still race")
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()

    out, err = proc.communicate(timeout=30)
    assert proc.returncode != 0
    assert "TOO SHORT" in out + err


def test_the_refusal_shows_its_arithmetic_and_names_the_wrong_fix(tmp_path):
    """A bare 'too short' sends someone back with no idea how long is enough,
    or tempts them into --learn-windows 6. The message must pre-empt both."""
    proc = _run(str(_wav(tmp_path / "short.wav", 10)))
    out = proc.stdout + proc.stderr
    assert "24 min" in out, "must say how much learning needs"
    assert "learn-windows" in out, "must name the tempting wrong fix"
    assert "55-59" in out, "must give the measured reason it is wrong"


def test_dropping_below_the_learn_floor_warns_loudly(tmp_path):
    """`--learn-windows` is legitimate for debugging, so it is not blocked --
    but it must never pass silently, because the output looks identical to a
    trustworthy run."""
    proc = _run(str(_wav(tmp_path / "r.wav", 6)), "--learn-windows", "4")
    out = proc.stdout + proc.stderr
    assert "below the documented floor" in out
    assert "55-59" in out


def test_a_missing_file_says_where_to_look(tmp_path):
    """AirDrop puts things in ~/Downloads and phone filenames contain spaces.
    Both are predictable, so both are answered in the error."""
    proc = _run(str(tmp_path / "nope.m4a"))
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "No such file" in out
    assert "Downloads" in out and "spaces" in out


def test_a_32bit_float_wav_is_readable(tmp_path):
    """Audacity and Logic export float32 by default, and the stdlib `wave`
    module raises "unknown format: 3" on it. check_duration used `wave`; a raw
    traceback in the one entry point aimed at non-programmers defeats its
    purpose."""
    import numpy as np
    from scipy.io import wavfile
    sys.path.insert(0, str(ROOT / "tools"))
    from fridge_scan import check_duration
    path = tmp_path / "float32.wav"
    wavfile.write(path, 16000,
                  (np.random.default_rng(0).standard_normal(16000 * 60 * 26)
                   * 0.1).astype(np.float32))
    assert check_duration(path, 48) == pytest.approx(26 * 60, abs=1.0)


def test_a_wav_is_not_sent_through_ffmpeg(tmp_path):
    """`to_wav` must pass .wav straight through — re-encoding a file that is
    already canonical is a lossy no-op waiting to happen."""
    sys.path.insert(0, str(ROOT / "tools"))
    from fridge_scan import to_wav
    src = _wav(tmp_path / "already.wav", 1)
    assert to_wav(src, tmp_path / "converted.wav") == src
    assert not (tmp_path / "converted.wav").exists()


@pytest.mark.parametrize("minutes,expect_ok", [(24.0, False), (25.0, True)])
def test_the_boundary_is_where_the_docstring_says_it_is(tmp_path, minutes,
                                                        expect_ok):
    """48 learn windows x 30 s = exactly 24 min, which leaves zero to score.
    So 24 must fail and 25 must get past the length check. Pinned because the
    numbers in the refusal message are quoted to the user as fact."""
    proc = _run(str(_wav(tmp_path / f"r{minutes}.wav", minutes)))
    refused = "TOO SHORT" in (proc.stdout + proc.stderr)
    assert refused != expect_ok
