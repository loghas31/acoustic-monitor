"""
tests/test_import_isolation.py — backlog T3.1 / SELF-REVIEW F27 regression.

WHAT WENT WRONG
----------------------------------------------------------------------------
`firmware/main.py` and `backend/main.py` are both top-level modules literally
called `main`. Which one `import main` resolves to is decided entirely by
`sys.path` order at the moment of the import:

  * `conftest.py` inserts `firmware`, `ml`, `backend` each at index 0, so the
    final order is backend, ml, firmware — `backend` wins, and
    `from main import app` gets the FastAPI app, which is what the backend
    test modules want.
  * But roughly fifteen test modules ALSO insert `firmware` and/or `ml` at
    `sys.path[0]` at module scope (`tests/test_evaluate_pinned.py:49-51`,
    `tests/test_fault_injection.py:80-82`,
    `tests/test_crest_floor_calibration.py:34`,
    `tests/test_phone_recording.py:64` and others). Any one of those demotes
    `backend` from index 0, and a later bare `from main import app` then binds
    `firmware/main.py`, which has no `app`.

Measured on this repo, 2026-08-29, before the fix:

    pytest tests/                         -> 676 collected, 0 errors
    pytest tests/ --ignore=tests/test_api.py
                                          -> 662 collected, 1 ERROR
      ImportError: cannot import name 'app' from 'main' (.../firmware/main.py)

The plain `pytest tests/` invocation was green purely because `test_api.py`
sorts alphabetically before every module that re-inserts `firmware` — it
imported `main` while `sys.path` was still intact and pinned the correct
module in `sys.modules` for everyone after it. That is a property of the
alphabet, not of the code.

WHY THIS FILE HAS TO USE SUBPROCESSES
----------------------------------------------------------------------------
The failure is a COLLECTION error, so the affected tests never report as
failures — they silently do not run, and the summary line just gets shorter.
An assertion inside the running suite therefore cannot see it: by the time
any test executes, collection has already finished and this process's
`sys.modules` is whatever it is. Only a *fresh interpreter*, collecting a
*different subset*, can observe the ordering dependency. That is also why
GitHub Actions (`.github/workflows/ci.yml:79`, plain `pytest tests/ -q`)
structurally cannot catch it — the same blind spot that hid F12 and F14.

THE FIX BEING GUARDED
----------------------------------------------------------------------------
`tests/test_api.py` and `tests/test_frontend_backend_integration.py` now load
`backend/main.py` from its path under a unique module name, the pattern
`tests/test_recordings_upload.py` and `tools/e2e_severity_trend.py` already
used for exactly this reason. Note that the backlog's own suggested cheapest
fix — deleting the redundant `sys.path` blocks from the two test modules F27
named — was tried first and MEASURED INSUFFICIENT: with both blocks removed,
`pytest tests/ --ignore=tests/test_api.py` still failed collection, because a
dozen other modules do the same insert. Removing the ambiguity at the import
site is what actually holds.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "firmware" / "baseline.npz"

# conftest.py's `pytest_collection_finish` hook aborts with returncode 1 if no
# learned baseline exists, which would make every subprocess here fail for an
# unrelated reason. The outer suite cannot even reach this file in that state,
# but a targeted `pytest tests/test_import_isolation.py` could, so be explicit.
pytestmark = pytest.mark.skipif(
    not BASELINE.exists(),
    reason="needs firmware/baseline.npz (docs/RUN_IT.md step 4)")


def _collect(*args: str) -> subprocess.CompletedProcess:
    """Collect-only pytest run in a clean interpreter, from the repo root.

    `-p no:cacheprovider` because the dev mount forbids the deletions pytest's
    cache directory does (backlog rule 7). `--collect-only` because collection
    is the whole failure surface — we deliberately do not execute the tests, so
    this stays fast enough to live in the normal suite (~2 s per call).
    """
    env = dict(os.environ)
    # Don't let a developer's PYTEST_ADDOPTS (e.g. `-x`, or an extra `-p`)
    # change what the subprocess collects and turn this into a flaky test.
    env.pop("PYTEST_ADDOPTS", None)
    env.setdefault("TMPDIR", "/tmp")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--collect-only", *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)


def _assert_clean(proc: subprocess.CompletedProcess, what: str) -> None:
    combined = proc.stdout + proc.stderr
    assert "during collection" not in combined, (
        f"{what} produced a COLLECTION error.\n"
        f"This is the T3.1/F27 shadowing bug: two modules named `main` on\n"
        f"sys.path, and a bare `import main` bound the wrong one. Load\n"
        f"backend/main.py by explicit path under a unique module name — see\n"
        f"tests/test_recordings_upload.py for the pattern.\n\n"
        f"--- pytest output ---\n{combined[-3000:]}")
    assert proc.returncode == 0, (
        f"{what} exited {proc.returncode}\n\n"
        f"--- pytest output ---\n{combined[-3000:]}")


def test_the_suite_collects_cleanly_without_test_api():
    """The exact command F27 used to fail on.

    `test_api.py` is the module whose alphabetical luck was masking the bug,
    so removing it is the minimal way to expose the real ordering dependency
    across the whole suite. Before the fix: 662 collected, 1 error.
    """
    proc = _collect("tests/", "--ignore=tests/test_api.py")
    _assert_clean(proc, "pytest tests/ --ignore=tests/test_api.py")


def test_a_firmware_module_collected_first_does_not_shadow_backends_main():
    """Two files, one process — the minimal reproduction.

    `test_evaluate_pinned.py` inserts `ml` and `firmware` at sys.path[0] at
    module scope; `test_frontend_backend_integration.py` then needs backend's
    `main`. pytest collects in the order given on the command line, so this
    pins the bad order deliberately rather than relying on the alphabet.
    """
    proc = _collect("tests/test_evaluate_pinned.py",
                    "tests/test_frontend_backend_integration.py")
    _assert_clean(proc, "evaluate_pinned + frontend_backend_integration")


def test_no_test_module_imports_the_ambiguous_main_module():
    """Static guard, so the next person gets told why rather than debugging it.

    Cheap (no subprocess) and it fails at the moment the bad line is written,
    not at the moment some unrelated module changes the collection order.
    """
    offenders = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("from main import", "import main")):
                offenders.append(f"{path.name}:{lineno}: {stripped}")
    assert not offenders, (
        "A test module imports the ambiguous top-level `main`:\n  "
        + "\n  ".join(offenders)
        + "\n\nBoth firmware/main.py and backend/main.py are called `main`, so"
          "\nwhich one you get depends on sys.path order at import time — see"
          "\nthis file's docstring. Load the one you want by explicit path:"
          "\n    spec = importlib.util.spec_from_file_location("
          "\n        'backend_main_<unique>', ROOT / 'backend' / 'main.py')")


def test_no_new_duplicate_module_basenames_appear_on_the_test_path():
    """Catch the NEXT collision before it becomes another silent skip.

    conftest.py puts firmware/, ml/ and backend/ all on sys.path, so any
    basename appearing in two of them is ambiguous in exactly the way `main`
    was. `main.py` is the known, deliberately-tolerated case (both are real
    entry points and renaming either churns deploy scripts, CI and docs); it
    is safe only because nothing imports it bare any more, which the test
    above enforces. Anything NEW here should be renamed rather than tolerated.
    """
    KNOWN_AMBIGUOUS = {"main.py"}

    seen = {}
    for sub in ("firmware", "ml", "backend"):
        for path in sorted((ROOT / sub).glob("*.py")):
            seen.setdefault(path.name, []).append(sub)

    duplicates = {name: dirs for name, dirs in seen.items()
                  if len(dirs) > 1 and name not in KNOWN_AMBIGUOUS}
    assert not duplicates, (
        "New duplicate module basename(s) across firmware/ ml/ backend/: "
        + ", ".join(f"{n} ({'/'.join(d)})" for n, d in sorted(duplicates.items()))
        + "\n\nAll three directories are on sys.path (conftest.py), so `import"
          "\n<name>` now resolves by sys.path order — a bug that hides as a"
          "\nCOLLECTION error, which means the affected tests silently do not"
          "\nrun instead of failing. Rename one, or load it by explicit path."
          "\nSee this file's docstring (T3.1 / SELF-REVIEW F27).")
