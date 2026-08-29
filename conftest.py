"""Pytest path setup: the repo uses flat script-style packages (firmware/, ml/,
backend/) per the project layout, so tests import them via sys.path."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("firmware", "ml", "backend"):
    sys.path.insert(0, str(ROOT / sub))

# Backend reads DATABASE_URL at import time — point it at a throwaway SQLite
# file BEFORE any test imports backend modules.
#
# The uid suffix is load-bearing, and its absence has now bitten this repo
# twice. SQLite has to live under /tmp (the dev mount lacks the POSIX locks it
# needs), but /tmp is shared and NOT cleared between agent containers, which
# run under different uids. A leftover `/tmp/test_acoustic.db` owned by another
# user is mode 644 — readable, not writable — so `create_all` fails with the
# thoroughly unhelpful "attempt to write a readonly database" and every backend
# test errors, with nothing in the repository having changed.
#
# `tests/test_severity_persistence.py` hit exactly this on 2026-08-18 and fixed
# it there (5 errors, suite 333 -> 328). The same fix was never applied to the
# shared default here or in `tests/test_api.py`, so on 2026-08-20 it happened
# again: 13 errors in test_api.py plus a cross-module knock-on into
# test_frontend_backend_integration.py, on a checkout whose own agent had
# recorded the suite as fully green — because that agent owned the file.
#
# Note this failure mode is INVISIBLE to GitHub Actions: a hosted runner gets a
# clean /tmp every time, so CI stays green while a real machine fails. Green CI
# is not evidence that this is fixed.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:////tmp/test_acoustic_{os.getuid()}.db")


def pytest_collection_finish(session):
    """Fail fast and legibly if the learned baseline is missing.

    `firmware/baseline.npz` is gitignored on purpose — it is a per-machine
    LEARNED artefact, not source. 23 tests load it as a fixture, so a fresh
    clone that has never run the learn period cannot pass the suite, and what
    it produces instead is 5 failures and 18 errors of raw
    `FileNotFoundError` from three frames inside `np.load`, repeated 23 times.
    That is a lot of noise for one missing command.

    This was invisible for the entire life of the project because every
    development machine has a baseline on disk from an earlier run. It first
    appeared on 2026-08-20, in CI runs #1-#3, on the only machine that had
    never run anything before — which is exactly the machine a new contributor
    is. The workflow now learns a baseline before invoking pytest; this hook is
    the seatbelt for everyone who runs `pytest` by hand on a clean checkout.

    Deliberately NOT auto-generating it: the learn period takes real time and
    writes into the working tree, and a test run that silently produces build
    artefacts is how you end up unable to tell which baseline your numbers came
    from. Name the command, let the human run it.
    """
    npz = ROOT / "firmware" / "baseline.npz"
    if npz.exists():
        return
    # `pytest.exit`, NOT `raise SystemExit`: raising out of a hook makes pytest
    # print the message wrapped in an INTERNALERROR> traceback, which reads
    # like the tooling broke rather than like an instruction. Verified both
    # ways before choosing.
    import pytest
    rule = "=" * 72
    pytest.exit(
        "\n" + rule + "\n"
        "No learned baseline at firmware/baseline.npz — 23 tests need one.\n\n"
        "It is gitignored because it is learned per machine, not source, so a\n"
        "fresh clone never has it. Learn one (about 30 s), then re-run:\n\n"
        "    python firmware/baseline.py --simulate --windows 48 \\\n"
        "        --out firmware/baseline.npz --db /tmp/state.db\n\n"
        "This is step 4 of docs/RUN_IT.md.\n"
        + rule,
        returncode=1)
