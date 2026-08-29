"""
tests/test_pi_perf_harness.py — backlog T4.4, `tools/pi_perf_harness.py`'s
own mechanics.

The real profiling run (`docs/DOC_PI_PERF.md`) uses `--reps 30` per stage
per case and takes tens of seconds — fine to run by hand, not something to
repeat inside the normal fast suite. What IS worth pinning here, fast and
with tiny `reps`: that the harness actually calls each real stage function
(not a stand-in), that its timing/statistics arithmetic is right, and that
the A53 estimate and gate-margin logic compute what they claim to.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "tools"), str(ROOT / "firmware"), str(ROOT / "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pi_perf_harness as harness                       # noqa: E402
from simulate import SimConfig, normal_signal            # noqa: E402


@pytest.fixture(scope="module")
def one_window():
    cfg = SimConfig(duration_s=30.0)
    rng = np.random.default_rng(0)
    audio = normal_signal(cfg, cfg.fs_audio, rng)
    accel = normal_signal(cfg, cfg.fs_accel, rng)
    return cfg, audio, accel


def test_rss_kb_reads_a_positive_number():
    assert harness._rss_kb() > 1_000


def test_time_stage_calls_the_real_function_reps_times():
    calls = []

    def fn(x):
        calls.append(x)

    times = harness._time_stage(fn, (42,), reps=4)
    assert len(times) == 4
    assert calls == [42, 42, 42, 42]
    assert all(t >= 0 for t in times)


def test_profile_window_covers_every_real_stage_and_matches_reps(one_window):
    cfg, audio, accel = one_window
    stages = harness.profile_window(audio, cfg.fs_audio, accel, cfg.fs_accel,
                                    scorer=None, reps=2)
    expected_stages = {
        "select_demodulation_band", "stft_mag", "channel_stats(audio)",
        "channel_stats(accel_x)", "band_energy_ilr(audio)",
        "band_energy_ilr(accel)", "envelope_features", "estimate_fr",
        "extract_features (whole)",
    }
    assert expected_stages <= stages.keys()
    assert "scorer.score" not in stages, "no baseline given, scorer stage must be skipped"
    for name, times in stages.items():
        assert len(times) == 2, f"{name} should report exactly `reps` timings"
        assert all(t > 0 for t in times), f"{name} reported a non-positive time"


def test_profile_window_includes_scorer_stage_when_given_one(one_window, tmp_path):
    """Real MahalanobisScorer, built the same way baseline.py --simulate
    builds one — not a stub — so this stage genuinely measures scorer.score,
    the actual downstream consumer of extract_features's output."""
    import subprocess
    out = tmp_path / "baseline.npz"
    db = tmp_path / "learn.db"
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"), "--simulate",
         "--windows", "48", "--out", str(out), "--db", str(db)],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr

    from inference import MahalanobisScorer
    scorer = MahalanobisScorer(out)
    cfg, audio, accel = one_window
    stages = harness.profile_window(audio, cfg.fs_audio, accel, cfg.fs_accel,
                                    scorer=scorer, reps=2)
    assert "scorer.score" in stages
    assert len(stages["scorer.score"]) == 2


def test_summarise_reports_a53_estimate_arithmetic_correctly(capsys):
    # Synthetic, exact numbers so the printed A53 range can be checked to
    # the millisecond, independent of any real machine's actual speed.
    stages = {"extract_features (whole)": [100.0, 100.0, 100.0]}
    harness.summarise(stages)
    out = capsys.readouterr().out
    low = 100.0 * harness.A53_SLOWDOWN_LOW
    high = 100.0 * harness.A53_SLOWDOWN_HIGH
    assert f"{low:.0f}-{high:.0f} ms" in out
    assert "PASSES with margin" in out          # 1000 ms high estimate < 2000 ms gate


def test_summarise_flags_at_risk_when_a53_estimate_exceeds_the_gate(capsys):
    # A whole-extraction time large enough that even the LOW end of the A53
    # estimate blows the 2000 ms stage-2 gate must say so, not "passes".
    stages = {"extract_features (whole)": [300.0] * 3}   # 300*8=2400 > 2000
    harness.summarise(stages)
    out = capsys.readouterr().out
    assert "AT RISK" in out


def test_memory_profile_runs_without_error_and_reports_both_numbers(one_window, capsys):
    cfg, audio, accel = one_window
    harness.memory_profile(audio, cfg.fs_audio, accel, cfg.fs_accel, reps=2)
    out = capsys.readouterr().out
    assert "tracemalloc peak" in out
    assert "RSS:" in out
