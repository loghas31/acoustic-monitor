"""
tests/test_memory_soak.py — backlog T4.1, `tools/memory_soak.py`'s own
mechanics. The actual multi-thousand-window soak (`docs/DOC_SOAK_MEMORY.md`)
takes tens of minutes and was run by hand, chunked across several separate
invocations because this project's own agent sandbox caps a single shell
call well under what "thousands of windows" needs in wall time (see that
doc, and the module's own docstring) — not something this fast test suite
should attempt to reproduce.

What IS worth pinning here, fast: the one piece of logic in the tool that is
easy to get subtly wrong and would silently invalidate every soak run built
on it — that chunk N of the soak generates BIT-IDENTICAL windows to what an
unbroken single run would have produced at the same global index, via the
seed/schedule shift in `main()`. If that drifts, "windows 3000-3599" in a
later chunk would silently stop meaning what it says.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "tools"), str(ROOT / "firmware")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import memory_soak as ms                              # noqa: E402
from capture import SimulatedSource                    # noqa: E402


def test_soak_schedule_is_the_same_two_speed_pattern_baseline_py_learns_from():
    # 24-window regimes, exactly baseline.py --simulate's own schedule.
    frs = [ms.soak_schedule(i)["fr"] for i in range(0, 96, 24)]
    assert frs == [50.0, 30.0, 50.0, 30.0]
    for i in range(0, 24):
        assert ms.soak_schedule(i)["kind"] == "normal"


def test_soak_schedule_injects_one_transient_every_200_windows_only():
    hits = [i for i in range(0, 1000) if ms.soak_schedule(i)["kind"] != "normal"]
    assert hits == [200, 400, 600, 800]          # NOT window 0 (i > 0 guard)
    for i in hits:
        sched = ms.soak_schedule(i)
        assert sched["kind"] == "bearing_outer"
        assert sched["severity"] == 0.5


def test_chunking_produces_bit_identical_windows_to_an_unbroken_run():
    """The correctness of every chunked soak run rests on this: a fresh
    SimulatedSource built with `seed=BASE+start_index` and
    `schedule=lambda j: soak_schedule(start_index+j)` — what memory_soak.py's
    main() constructs for chunk 2 onward — must yield exactly the windows an
    unbroken run from index 0 would have produced at the same global index,
    or "chunk 2 continues chunk 1" is false and every reported trend is
    measuring the wrong windows."""
    BASE_SEED = 4242
    unbroken = SimulatedSource(30.0, 16000, 6400, schedule=ms.soak_schedule,
                               realtime=False, seed=BASE_SEED)
    gen = unbroken.windows()
    unbroken_windows = [next(gen) for _ in range(30)]

    start_index = 20
    chunked = SimulatedSource(30.0, 16000, 6400,
                              schedule=lambda j: ms.soak_schedule(start_index + j),
                              realtime=False, seed=BASE_SEED + start_index)
    gen2 = chunked.windows()
    chunk_windows = [next(gen2) for _ in range(10)]   # covers global 20..29

    for j in range(10):
        u_audio, u_accel = unbroken_windows[start_index + j]
        c_audio, c_accel = chunk_windows[j]
        assert np.array_equal(u_audio, c_audio), f"audio differs at global index {start_index + j}"
        assert np.array_equal(u_accel, c_accel), f"accel differs at global index {start_index + j}"


def test_rss_kb_reads_a_positive_sane_number():
    rss = ms._rss_kb()
    assert 1_000 < rss < 50_000_000     # between 1 MB and 50 GB — sanity, not precision


def test_fit_slope_recovers_a_known_synthetic_slope():
    # Construct RSS that grows EXACTLY 2 kB/window and confirm the fit
    # recovers it, and the MB/week extrapolation is arithmetically right.
    idx = np.arange(0, 1000, 5, dtype=float)
    rss = 140_000.0 + 2.0 * idx
    slope, growth_mb_per_week = ms.fit_slope_mb_per_week(idx, rss, window_s=30.0)
    assert slope == pytest.approx(2.0, abs=1e-6)
    expected_mb_per_week = 2.0 * (7 * 24 * 3600 / 30.0) / 1024
    assert growth_mb_per_week == pytest.approx(expected_mb_per_week, rel=1e-6)


def test_fit_slope_on_flat_rss_is_zero():
    idx = np.arange(0, 500, 5, dtype=float)
    rss = np.full_like(idx, 145_000.0)
    slope, growth = ms.fit_slope_mb_per_week(idx, rss)
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert growth == pytest.approx(0.0, abs=1e-9)


def test_summarise_cli_reports_per_chunk_trend_on_a_synthetic_jsonl(tmp_path):
    """Build a tiny synthetic samples.jsonl with a KNOWN, DIFFERENT slope in
    each of two chunks, and check --chunk-size actually separates them —
    the whole point of that flag (see its own --help text) is that a single
    whole-run slope would average the two together and hide exactly this."""
    import json
    out = tmp_path / "samples.jsonl"
    rows = []
    for i in range(0, 100, 5):            # chunk 1: flat
        rows.append({"i": i, "rss_kb": 140_000.0, "t": 0.0})
    for i in range(100, 200, 5):          # chunk 2: rising fast
        rows.append({"i": i, "rss_kb": 140_000.0 + 50.0 * (i - 100), "t": 0.0})
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    buf = io.StringIO()
    with redirect_stdout(buf):
        ms._summarise(out, chunk_size=100)
    text = buf.getvalue()
    assert "chunk     0-   99" in text
    assert "chunk   100-  199" in text
    # chunk 1 (flat) should report ~0 slope; chunk 2 should report +50 kB/window
    lines = [l for l in text.splitlines() if "chunk " in l and "slope" in l]
    assert len(lines) == 2
    assert "+0.0000" in lines[0] or "-0.0000" in lines[0]
    assert "+50.0000" in lines[1]
