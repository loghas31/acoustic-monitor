"""
Guard against documentation drifting away from the code it describes.

Why this file exists (finding F11). The README carried two different test
counts — "31 tests" in the quickstart and "359 tests" in the repository map —
while the suite actually collected 427. Neither number was wrong when it was
written; both were written by hand and then never touched again. Every other
claim in this repository is pinned by a test, which is precisely why the
untested ones rotted: there was no mechanism by which a stale number could
announce itself. DOC_STATUS.md exists to separate "proven" from "assumed", and
a hand-maintained integer in a README is assumed, forever, by construction.

So the count is now a tested claim like any other. If you add tests and do not
update the README, this file fails and tells you both numbers.

Scope, deliberately narrow: this checks the CURRENT-STATE claims in README.md
only. Historical entries in the task backlog (not in this public copy) ("suite 401 -> 422") are a
changelog — they record what was true on a given date and must NOT be
rewritten to match today. Auto-updating them would destroy the audit trail
that makes the backlog worth keeping.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

# The two current-state claims, each with the regex that finds its integer.
# Keyed by a human-readable location so a failure says WHERE to edit.
_CLAIM_PATTERNS = {
    # Both patterns are deliberately loose about the WORDING around the
    # integer and strict about where it lives, because this same test file
    # ships in the public split of this repository, whose README states the
    # same two counts in shorter prose ('# 680 tests' and '| 680 tests, run
    # on every push |'). The original patterns required the private
    # README's exact phrasing - 'full suite (N tests)' and 'N tests across'
    # - so the public repo failed this guard on its first CI run despite
    # having both claims present and correct. A guard that fires on a
    # correct README teaches whoever hits it to delete the guard.
    "quickstart (`pytest tests/` comment)":
        re.compile(r"pytest\s+tests/\s*#\s*(?:\d+\.\s*full suite \()?(\d+) tests"),
    "repository map (`tests/` row)":
        re.compile(r"\|\s*`tests/`\s*\|\s*(\d+) tests"),
}


def _collected_test_count() -> int:
    """Ask pytest how many tests it collects, in a fresh process.

    A subprocess rather than the running session's `session.items`: this must
    give the same answer whether the outer invocation was `pytest tests/`,
    `pytest tests/test_docs_current.py`, or filtered with `-k`. Reading the
    live session would make the assertion depend on how the developer happened
    to invoke pytest, which is exactly the kind of flakiness that gets a guard
    test deleted rather than fixed.

    `--collect-only` imports the test modules but runs nothing, so there is no
    recursion: this function's own subprocess collects this test without
    executing it.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", str(REPO / "tests")],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    # pytest exit code 5 == "no tests collected", which is itself a failure
    # worth surfacing loudly rather than reporting as "0 tests".
    if proc.returncode not in (0,):
        pytest.fail(
            "collection sub-process exited {}; cannot verify doc counts.\n"
            "stdout tail:\n{}\nstderr tail:\n{}".format(
                proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:]))

    m = re.search(r"(\d+) tests? collected", proc.stdout)
    if not m:
        pytest.fail(
            "could not parse a collected-test count out of pytest's output. "
            "The summary format may have changed in this pytest version.\n"
            "stdout tail:\n{}".format(proc.stdout[-2000:]))
    return int(m.group(1))


def test_readme_states_a_test_count_in_both_places():
    """Both claims must exist. A silently-deleted claim is not a pass."""
    text = README.read_text(encoding="utf-8")
    missing = [where for where, pat in _CLAIM_PATTERNS.items()
               if pat.search(text) is None]
    assert not missing, (
        "README.md no longer contains the expected test-count claim(s): "
        + "; ".join(missing)
        + ". If you moved or reworded them, update _CLAIM_PATTERNS in this "
          "file so the guard keeps guarding."
    )


def test_readme_test_counts_match_the_suite():
    """The headline numbers in README.md equal what pytest actually collects."""
    text = README.read_text(encoding="utf-8")
    actual = _collected_test_count()

    wrong = {}
    for where, pat in _CLAIM_PATTERNS.items():
        m = pat.search(text)
        if m is None:
            continue                     # reported by the test above
        claimed = int(m.group(1))
        if claimed != actual:
            wrong[where] = claimed

    assert not wrong, (
        "README.md is out of date: the suite collects {} tests, but it "
        "claims {}. Edit README.md, do not edit this test.".format(
            actual,
            ", ".join("{} in {}".format(n, where)
                      for where, n in sorted(wrong.items())))
    )


def test_readme_counts_agree_with_each_other():
    """Cheap consistency check that runs without spawning a subprocess.

    F11's most embarrassing detail is that the README disagreed with ITSELF —
    31 in one place, 359 in another — which needed no knowledge of the suite
    at all to notice. That check should be instant and unconditional.
    """
    text = README.read_text(encoding="utf-8")
    found = {where: int(m.group(1))
             for where, pat in _CLAIM_PATTERNS.items()
             if (m := pat.search(text)) is not None}
    assert len(set(found.values())) <= 1, (
        "README.md contradicts itself about the test count: {}".format(
            ", ".join("{} says {}".format(w, n) for w, n in sorted(found.items())))
    )
