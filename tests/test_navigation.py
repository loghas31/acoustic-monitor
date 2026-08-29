"""Every document is reachable, every tool is alive, no dead files.

WHY THIS EXISTS
---------------
Same reasoning as F11's doc-count guard, applied to structure instead of
numbers. This repo has 42 documents and 21 tools. Nobody notices when one stops
being linked from anywhere — it does not break, it just quietly becomes
invisible, and then someone rewrites it from scratch because they could not
find it. Ten documents were unreachable from the README when this was written,
including `DOC_SELF_REVIEW.md`, which is arguably the most useful file here.

The failure mode is not "the build breaks". It is "the project slowly becomes
navigable only by the person who wrote it", which for a solo project is
invisible right up until it matters — a supervisor, a collaborator, a funder,
or yourself in six months.

WHAT IS DELIBERATELY *NOT* CHECKED
----------------------------------
That every file is *referenced* by something. That would pass trivially — a
file mentioning itself, or one dead file citing another. What is checked is
reachability from the README specifically, because that is the only entry point
a stranger has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# Root-level docs are their own entry points and need no inbound link.
ROOT_DOCS = {"README.md", "TESTS.md", "the project plan (not in this public copy)", "RESULTS.md", "CONTRIBUTING.md"}


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_every_doc_is_reachable_from_the_readme():
    """A document nobody can find is a document nobody reads."""
    unlinked = [p.name for p in sorted((ROOT / "docs").glob("*.md"))
                if p.name not in _readme()]
    assert not unlinked, (
        "these documents are not linked from README.md, so nothing points at "
        "them:\n  " + "\n  ".join(unlinked) +
        "\n\nAdd them to a table in README.md. If one is genuinely obsolete, "
        "move it to output/_attic/ rather than leaving it orphaned — the mount "
        "forbids rm, and a deleted-but-not-really file is worse than either.")


def test_every_tool_is_referenced_by_code_a_test_or_a_doc():
    """An orphaned tool is dead weight that still has to be read and
    maintained. Note the search includes other tools — `accel_axis_legacy.py`
    is referenced only by `accel_axis_compare.py`, is deliberately kept, and
    an earlier version of this check wrongly flagged it by not looking there."""
    searchable = []
    for pat in ("tests/*.py", "docs/*.md", "tools/*.py", "ml/**/*.py",
                "firmware/*.py", "*.md"):
        searchable += [p for p in ROOT.glob(pat) if p.is_file()]
    blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in searchable)

    orphans = []
    for tool in sorted((ROOT / "tools").glob("*.py")):
        # Discount the file's own text so a tool cannot vouch for itself.
        own = tool.read_text(encoding="utf-8", errors="ignore")
        if blob.replace(own, "").count(tool.stem) == 0:
            orphans.append(tool.name)

    assert not orphans, (
        "these tools are referenced by nothing at all:\n  " +
        "\n  ".join(orphans) + "\n\nEither wire them in, document them, or "
        "move them to output/_attic/.")


def test_the_three_entry_points_link_to_each_other():
    """A newcomer lands on README; someone about to test lands on TESTS.md;
    someone deciding what to do next lands on the project plan (not in this public copy). Each has to lead to the
    others or the reader gets stranded in whichever one they opened."""
    readme = _readme()
    assert "TESTS.md" in readme and "the project plan (not in this public copy)" in readme

    tests_md = (ROOT / "TESTS.md").read_text(encoding="utf-8")
    plan_md = (ROOT / "the project plan (not in this public copy)").read_text(encoding="utf-8")
    assert "the manual-steps guide (not in this public copy)" in tests_md or "FRIDGE_TEST.md" in tests_md
    assert "TESTS.md" in plan_md, "the plan must point at the run sheet"


def test_no_readme_link_points_at_a_missing_file():
    """The other half of reachability: links that go nowhere. A broken link is
    worse than no link, because the reader assumes the document exists and
    that they are looking in the wrong place."""
    broken = []
    for target in re.findall(r"\]\(([^)#:]+\.md)\)", _readme()):
        if not (ROOT / target).exists():
            broken.append(target)
    assert not broken, "README links to files that do not exist:\n  " + \
                       "\n  ".join(broken)


@pytest.mark.parametrize("name", sorted(ROOT_DOCS - {"README.md"}))
def test_the_root_level_docs_exist(name):
    """These are named in the README and in each other; if one is renamed
    without updating the rest, the entry points break silently."""
    assert (ROOT / name).exists(), f"{name} is referenced but missing"
