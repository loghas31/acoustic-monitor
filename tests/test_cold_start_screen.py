"""Guards for `tools/cold_start_screen.py` — the baseline-free fault screen.

What has to stay true for this tool to be worth having:

  1. It finds a LOUD fault with no baseline and no bearing geometry, and the
     fundamental it reports is the true BPFO — not a coincidence, the actual
     fault frequency computed from the bearing.
  2. It FLAGS mains and shaft coincidences without removing them. Both halves
     matter and they pull against each other. `analyse_recording.py` records a
     150 Hz shaft harmonic once reported as a bearing fault in the healthy AND
     faulty recording — so coincidences must be marked. But an inner-race
     fault is amplitude-modulated at the shaft rate, and on a direct-drive
     machine that rate IS the mains frequency — so they must never be dropped.
     An earlier version dropped them and reported 7.6 of noise on
     `bearing_inner` in place of the true 39.6.
  3. It does NOT claim to find early faults. The measured limit (0/6 at
     severity 0.20) is pinned deliberately: if someone "improves" the scoring
     and this starts passing, that is a real result worth investigating, not a
     test to delete quietly.

Point 3 is the unusual one. Pinning a FAILURE feels wrong until you consider
the alternative — a future tweak that appears to fix severity 0.20 while
actually just overfitting the six synthetic machines would otherwise ship
silently, and this tool's whole purpose is telling someone whether their
machine is broken.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "firmware", ROOT / "ml" / "realdata"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cold_start_screen import is_mains_related, screen   # noqa: E402
from fault_frequencies import lookup, rpm_to_hz          # noqa: E402
from synth_phone_recording import make_pair              # noqa: E402

BPFO = lookup("6202").bpfo(rpm_to_hz(1450.0))            # 73.65 Hz


def _pair(seed: int, severity: float):
    return make_pair(seed=seed, duration_s=20.0, fs=16000.0, severity=severity)


def test_a_loud_fault_is_found_with_no_baseline_at_all():
    """The reason this tool exists. No learn period, no history, no geometry —
    just one recording — and the fault must still stand out."""
    beat = 0
    for seed in range(1, 7):
        p = _pair(seed, 0.35)
        h = screen(p["healthy"], p["fs"])["best_score"]
        f = screen(p["faulty"], p["fs"])["best_score"]
        beat += f > h
    assert beat >= 5, f"only {beat}/6 faulty recordings scored above healthy"


def test_the_reported_frequency_is_the_real_bpfo_not_a_coincidence():
    """Separation alone could be luck. The fundamental it reports must be the
    bearing's actual outer-race fault frequency, computed independently from
    the geometry — which the screen is never told."""
    for seed in range(1, 4):
        f0 = screen(_pair(seed, 0.35)["faulty"], 16000.0)["best_f0"]
        assert f0 is not None
        assert abs(f0 - BPFO) < 2.0, (
            f"reported {f0:.1f} Hz, BPFO is {BPFO:.1f} Hz — the screen is "
            f"separating healthy from faulty for some OTHER reason, which "
            f"makes it unsafe to trust on a real machine")


def test_early_faults_are_not_found_and_that_limit_is_pinned():
    """Pinned failure — see the module docstring. At severity 0.20 the comb is
    not reliably above the noise, matching F18's finding that even the
    self-baselined detector misses these. If this ever starts passing,
    investigate before celebrating."""
    beat = sum(screen(_pair(s, 0.20)["faulty"], 16000.0)["best_score"] >
               screen(_pair(s, 0.20)["healthy"], 16000.0)["best_score"]
               for s in range(1, 7))
    assert beat <= 3, (
        f"severity 0.20 now separates {beat}/6, up from the measured 2/6 "
        f"(which is itself a coin-flip - the medians are indistinguishable). "
        f"Either the generator changed or the scoring did — find out which "
        f"before updating this number.")


@pytest.mark.parametrize("f0", [50.0, 100.0, 150.0, 25.0])
def test_mains_and_its_sub_harmonics_are_rejected(f0):
    """Both directions. 50 Hz is obvious. 25 Hz matters because its second
    harmonic IS the mains line — a comb built out of mains energy wearing a
    lower fundamental as a disguise."""
    assert is_mains_related(f0, 50.0)


@pytest.mark.parametrize("f0", [73.65, 87.0, 119.7])
def test_genuine_fault_frequencies_are_not_rejected_as_mains(f0):
    """The rejection must not be so greedy it eats the signal. BPFO (73.65)
    and BPFI (119.7) for this bearing must survive 50 Hz rejection."""
    assert not is_mains_related(f0, 50.0)


def test_a_shaft_harmonic_is_rejected_when_the_speed_is_known():
    """The failure `analyse_recording.py` already documents: a shaft harmonic
    confidently reported as a bearing fault. If the user supplies --fr, its
    harmonics must be excluded."""
    fr = 24.17
    assert is_mains_related(3 * fr, fr, tol=1.5), "3x shaft must be rejected"


@pytest.fixture(scope="session")
def sim_wavs(tmp_path_factory) -> Path:
    """Generate `ml/simulate.py`'s signals into a temp dir, once per session.

    GENERATED, NOT READ FROM `data/`, DELIBERATELY. The copies in `data/` are
    gitignored (that directory holds unredistributable public datasets), so a
    fresh clone and every CI run would find them absent and SKIP these tests —
    silently, and precisely on the checks that caught the mains-rejection
    defect. A test that only runs on one laptop is not a test.

    Generating them takes about a second and makes these run everywhere.
    """
    out = tmp_path_factory.mktemp("simwavs")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "ml" / "simulate.py"), "--outdir", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"ml/simulate.py failed:\n{proc.stdout}\n{proc.stderr}")
    return out


def _load(directory: Path, name: str):
    """One of `ml/simulate.py`'s signals, normalised.

    These matter because they come from a DIFFERENT generator than the one
    this screen was developed against (`synth_phone_recording`): white noise
    floor rather than pink, different bearing geometry, different fault
    synthesis. They are the closest thing to independent test data in the repo,
    and they are what exposed the mains-rejection defect.
    """
    from scipy.io import wavfile
    path = directory / f"{name}.wav"
    if not path.exists():
        pytest.fail(f"ml/simulate.py did not produce {path.name}")
    fs, d = wavfile.read(path)
    x = d.astype(np.float64)
    return x / (np.max(np.abs(x)) + 1e-12), float(fs)


@pytest.mark.parametrize("name,floor", [("bearing_outer", 20.0),
                                        ("bearing_inner", 20.0)])
def test_real_bearing_faults_score_high(sim_wavs, name, floor):
    """Independent-generator check. Measured 33.0 (outer) and 39.6 (inner)."""
    x, fs = _load(sim_wavs, name)
    assert screen(x, fs)["best_score"] > floor


@pytest.mark.parametrize("name", ["normal", "imbalance"])
def test_healthy_and_non_impulsive_faults_stay_quiet(sim_wavs, name):
    """`normal` is healthy; `imbalance` is a real fault this screen is
    DESIGNED to miss (it grows the 1x shaft component, it does not impact).
    Both must stay low, and imbalance staying low is correct behaviour, not a
    bug — it is the self-baselined detector's job."""
    x, fs = _load(sim_wavs, name)
    assert screen(x, fs)["best_score"] < 10.0


def test_an_inner_race_fault_at_the_mains_frequency_is_not_discarded(sim_wavs):
    """THE REGRESSION. `ml/simulate.py` amplitude-modulates an inner-race
    fault at the shaft rate (real physics: the defect passes through the load
    zone once per revolution). Shaft rate is 50 Hz, which IS the mains
    frequency. An earlier version of this screen rejected mains coincidences
    outright and so reported 7.6 of noise instead of the true 39.6 — it
    discarded the strongest evidence of a genuine fault.

    Coincidences must be FLAGGED, never removed."""
    x, fs = _load(sim_wavs, "bearing_inner")
    r = screen(x, fs)
    f0, score, flag = r["peaks"][0]
    assert score > 20.0, "the fault must still be the top-ranked candidate"
    assert abs(f0 - 50.0) < 3.0, f"expected the ~50 Hz modulation, got {f0}"
    assert flag, "it should be flagged as a mains coincidence"
    assert any(p[2] for p in r["peaks"]), "flags must survive into the output"


def test_flagging_never_removes_a_candidate_from_the_ranking(sim_wavs):
    """Structural guard on the fix: whatever is flagged is still present and
    still ranked by score, so no future 'tidy-up' can quietly filter again."""
    x, fs = _load(sim_wavs, "bearing_inner")
    peaks = screen(x, fs)["peaks"]
    scores = [s for _, s, _ in peaks]
    assert scores == sorted(scores, reverse=True)
    assert len(peaks) >= 2


FS = 16000.0
N = int(FS * 5)
T = np.arange(N) / FS


def test_non_finite_samples_are_refused_not_analysed():
    """NaN propagates to the score, and `nan > threshold` is False — so a
    corrupted recording would report 'nothing found', the most dangerous
    possible answer. It must refuse loudly instead."""
    x = np.concatenate([np.random.default_rng(0).standard_normal(N), [np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        screen(x, FS)


def test_clipping_is_detected_because_it_manufactures_harmonics():
    """Clipping is uniquely dangerous HERE: a flat-topped wave is a square
    wave is a harmonic series, which is exactly the evidence this tool looks
    for. Measured — hard-clipped sine scores 2904, a real fault about 35."""
    clipped = np.clip(np.sin(2 * np.pi * 137 * T) * 1.4, -1, 1)
    assert screen(clipped, FS)["clipping_suspected"]


def test_a_clean_tone_is_not_mistaken_for_clipping():
    """The false positive that would train the user to ignore the warning. A
    sinusoid dwells near its own peak, so a naive near-peak count reads 2.8 %
    on clean audio. Flat-top detection must read zero."""
    assert not screen(np.sin(2 * np.pi * 137 * T) * 0.5, FS)["clipping_suspected"]


@pytest.mark.parametrize("name", ["normal", "bearing_outer", "bearing_inner"])
def test_real_recordings_are_not_falsely_flagged_as_clipped(sim_wavs, name):
    """The guards must not fire on the data the tool is meant to work on."""
    x, fs = _load(sim_wavs, name)
    r = screen(x, fs)
    assert not r["clipping_suspected"]
    assert not r["degenerate_band"]


def test_an_empty_demodulation_band_is_caught():
    """A 137 Hz tone sits below the 1 kHz demodulation floor, so the band holds
    only leakage and the score becomes noise over noise — measured at 13431,
    two orders of magnitude above any real fault. The score is still returned
    (so the number can be inspected) but must be marked degenerate."""
    r = screen(np.sin(2 * np.pi * 137 * T) * 0.5, FS)
    assert r["degenerate_band"]
    assert r["band_energy_ratio"] < 1e-3


def test_the_degenerate_threshold_has_real_margin(sim_wavs):
    """Pinned so the guard cannot be tightened into firing on real audio.
    Measured: degenerate tone 2.7e-4, `normal` 3.0e-2, faults 1.9e-1."""
    for name in ("normal", "bearing_outer"):
        x, fs = _load(sim_wavs, name)
        assert screen(x, fs)["band_energy_ratio"] > 1e-2


def test_screen_returns_a_usable_shape_on_pure_noise():
    """No crash, and no confident answer, on something with no structure."""
    rng = np.random.default_rng(0)
    r = screen(rng.standard_normal(16000 * 5), 16000.0)
    assert "best_score" in r and "peaks" in r
    assert r["best_score"] >= 0.0
