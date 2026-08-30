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

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "firmware", ROOT / "ml" / "realdata"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scipy.io import wavfile                             # noqa: E402

from cold_start_screen import (                          # noqa: E402
    CLIP_FRAC_WARN, TRUE_PEAK_WARN_DBTP, clipped_fraction, full_scale_float,
    is_mains_related, screen, true_peak_dbtp)
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


# ---------------------------------------------------------------------------
# T1.16 #1 — a lossy codec defeats flat-top clipping detection, and lossy is
# the normal input. Fixed 2026-08-30 by adding `true_peak_dbtp`. These tests
# were verified FAILS-ON-OLD-CODE before the fix went in.
# ---------------------------------------------------------------------------

def _lossy(x: np.ndarray) -> np.ndarray:
    """Emulate the one thing every lossy codec does, without needing ffmpeg.

    A codec band-limits and re-synthesises. Band-limiting a signal that has
    flat tops is textbook Gibbs conditions, so the reconstruction overshoots
    the level the original was sliced at while perturbing every sample — which
    is exactly the pair of effects that kills `clipped_fraction` (it needs
    bit-identical neighbours) and creates the inter-sample peak
    `true_peak_dbtp` measures.

    Verified faithful against a real ffmpeg AAC 128 kbps round trip on
    2026-08-30: on ADC-clipped broadband noise, real AAC gave flat-top
    0.00046 / +6.31 dBTP and this emulation gives 0.00000 / +4.78 dBTP — same
    verdict from both tests, same direction, same order of magnitude. A real
    round trip is also exercised, in the ffmpeg-gated test below.
    """
    from scipy.signal import resample_poly
    return resample_poly(resample_poly(x, 1, 2), 2, 1)[:len(x)]


def test_true_peak_catches_clipping_that_the_flat_top_test_loses_to_a_codec():
    """THE BUG, in one assertion. Every phone recording arrives codec-damaged,
    so the flat-top test — which needs bit-identical neighbours — is blind on
    the normal input. Measured on this signal: flat-top 0.236 before the
    codec, 0.000 after (warning floor 0.001), while the true peak stays at
    +4.8 dBTP because band-limiting a flat top MAKES the overshoot."""
    rng = np.random.default_rng(11)
    clipped = np.clip(2.5 * rng.standard_normal(N), -1.0, 1.0)

    assert clipped_fraction(clipped) > CLIP_FRAC_WARN, "pre-codec: caught"

    lossy = _lossy(clipped)
    assert clipped_fraction(lossy) <= CLIP_FRAC_WARN, (
        "the flat-top test is expected to be blind here — that is the bug "
        "this test exists for; if it started catching it, re-read the fix")

    tp = true_peak_dbtp(lossy)
    assert tp >= TRUE_PEAK_WARN_DBTP, f"true peak {tp:+.2f} dBTP should warn"
    r = screen(lossy, FS, true_peak=tp)
    assert r["clipping_suspected"]
    assert r["true_peak_clipping_suspected"]
    assert not r["flat_top_clipping_suspected"]


def test_codec_surviving_clipping_turns_a_HEALTHY_machine_into_a_fault(sim_wavs):
    """Why the warning is worth having, in the only terms that matter.

    `normal.wav` is a HEALTHY machine and scores about 5. Drive it into
    clipping and put it through a codec and it scores 59.7 at 49.75 Hz —
    above the ~35 that `cold_start_screen`'s own documentation calls a real
    fault. The flat-top test reads 0.00000 and says nothing. Measured through
    a real ffmpeg AAC round trip too: 51.7 at 49.75 Hz, flat-top 0.00063,
    silent (the 0.001 floor missed it by a factor of 1.6).

    So this is not a hygiene warning. Without it the tool tells someone their
    working machine is broken, with a plausible number and a plausible
    frequency. The same experiment on `bearing_outer` (true answer 152.25 Hz)
    returns 94.1 at 99.75 Hz — the score inflates AND the frequency moves, so
    a genuine fault is misreported rather than merely exaggerated."""
    x, fs = _load(sim_wavs, "normal")
    healthy = screen(_lossy(np.clip(0.5 * x, -1.0, 1.0)), fs)
    assert healthy["best_score"] < 20.0, "sanity: normal.wav is healthy"

    loud = _lossy(np.clip(8.0 * x, -1.0, 1.0))
    tp = true_peak_dbtp(loud)
    r = screen(loud / np.max(np.abs(loud)), fs, true_peak=tp)

    assert r["best_score"] > 35.0, (
        "the danger itself: clipping manufactures a fault-sized score on a "
        "healthy machine")
    assert not r["flat_top_clipping_suspected"], (
        "the flat-top test is blind here — that is the bug")
    assert r["clipping_suspected"], "true peak must catch what flat-top lost"


def test_a_codec_damaged_recording_with_headroom_is_not_flagged():
    """The other half, and the one that decides whether the warning is worth
    having. Same codec damage, same broadband signal, but recorded with 6 dB
    of headroom as it should be. Measured -2.0 dBTP; it must stay silent."""
    rng = np.random.default_rng(11)
    quiet = _lossy(0.15 * rng.standard_normal(N))
    tp = true_peak_dbtp(quiet)
    assert tp < TRUE_PEAK_WARN_DBTP, f"true peak {tp:+.2f} dBTP must not warn"
    assert not screen(quiet, FS, true_peak=tp)["clipping_suspected"]


@pytest.mark.parametrize("name", ["normal", "bearing_outer", "bearing_inner",
                                  "imbalance"])
def test_true_peak_does_not_fire_on_the_repo_s_own_recordings(sim_wavs, name):
    """A warning that fires on good audio is worse than no warning — the
    stated reason three earlier candidate metrics were rejected. Measured
    2026-08-30: normal -0.85, bearing_inner -0.91 dBTP, and the six real
    phone recordings behind RESULTS.md Experiment 0 run -11.0 to -27.7.

    Reads the file RAW, through the same `full_scale_float` helper `main()`
    uses — deliberately not through `_load`, which normalises. The first
    version of this test did use `_load` and all four signals came back at
    +0.01 dBTP, a clean sweep of false positives, because a peak-normalised
    signal is at full scale BY CONSTRUCTION. Kept as a comment because it is
    the single easiest way to get this measurement wrong."""
    fs, data = wavfile.read(sim_wavs / f"{name}.wav")
    tp = true_peak_dbtp(full_scale_float(data))
    assert tp < TRUE_PEAK_WARN_DBTP, f"{name}: {tp:+.2f} dBTP false positive"
    x, fs2 = _load(sim_wavs, name)
    assert not screen(x, fs2, true_peak=tp)["clipping_suspected"]


def test_full_scale_float_gets_each_wav_dtype_right():
    """A silent, total failure if wrong: an int16 recording read without
    dividing by 32768 reports about +90 dBTP and flags everything; one read
    with the wrong divisor reports -90 and flags nothing."""
    assert true_peak_dbtp(full_scale_float(
        np.full(1024, 32767, dtype=np.int16)), oversample=1) == pytest.approx(0.0, abs=0.01)
    assert true_peak_dbtp(full_scale_float(
        np.full(1024, 16384, dtype=np.int16)), oversample=1) == pytest.approx(-6.02, abs=0.05)
    assert true_peak_dbtp(full_scale_float(
        np.full(1024, 255, dtype=np.uint8)), oversample=1) == pytest.approx(0.0, abs=0.1)
    assert true_peak_dbtp(full_scale_float(
        np.full(1024, 0.5, dtype=np.float32)), oversample=1) == pytest.approx(-6.02, abs=0.05)


def test_an_unmeasured_true_peak_reports_as_unmeasured_not_as_clean():
    """`screen()` cannot compute this itself: by the time a signal reaches it
    the caller has usually normalised to peak 1.0, which is 0 dBTP whatever
    the recording was. So callers that have not measured it pass None, and the
    result must say so rather than implying a clean bill of health."""
    r = screen(np.sin(2 * np.pi * 137 * T) * 0.5, FS)
    assert r["true_peak_dbtp"] is None
    assert r["true_peak_clipping_suspected"] is False


def test_normalising_before_measuring_would_flag_everything():
    """Pins the ordering bug this fix is one line away from. If `main()` ever
    measures the true peak AFTER its peak normalisation, every recording ever
    passed in reads 0.00 dBTP and warns. Both directions asserted on the same
    signal, so the test cannot pass vacuously."""
    rng = np.random.default_rng(5)
    quiet = 0.05 * rng.standard_normal(N)
    assert true_peak_dbtp(quiet) < TRUE_PEAK_WARN_DBTP
    assert true_peak_dbtp(quiet / np.max(np.abs(quiet))) >= TRUE_PEAK_WARN_DBTP


def test_true_peak_is_measured_between_samples_not_just_at_them():
    """The reason for the 4x oversampling of ITU-R BS.1770-4, and the reason
    this survives ffmpeg's default 16-bit decode: that decode hard-limits the
    decoded overshoot back to +-1.0, so the peak sample alone reads exactly
    0.00 dBTP and the evidence would be gone."""
    rng = np.random.default_rng(11)
    hard_limited = np.clip(_lossy(np.clip(2.5 * rng.standard_normal(N),
                                          -1.0, 1.0)), -1.0, 1.0)
    assert np.max(np.abs(hard_limited)) <= 1.0
    at_samples = true_peak_dbtp(hard_limited, oversample=1)
    oversampled = true_peak_dbtp(hard_limited, oversample=4)
    assert at_samples <= 0.0 + 1e-9
    assert oversampled > at_samples + 1.0, (
        f"oversampling recovered only {oversampled - at_samples:.2f} dB")


def test_true_peak_is_unchanged_when_the_signal_is_processed_in_blocks():
    """The block loop exists so a 25-minute recording does not allocate
    ~800 MB on a Pi. It must not change the answer or lose a peak that lands
    on a block seam."""
    rng = np.random.default_rng(2)
    x = _lossy(np.clip(2.0 * rng.standard_normal(N), -1.0, 1.0))
    whole = true_peak_dbtp(x, block=1 << 20)
    chunked = true_peak_dbtp(x, block=4096)
    assert abs(whole - chunked) < 0.05, f"{whole:+.3f} vs {chunked:+.3f}"


def test_true_peak_handles_empty_and_silent_input_without_crashing():
    assert true_peak_dbtp(np.array([])) == float("-inf")
    assert true_peak_dbtp(np.zeros(N)) == float("-inf")
    assert not screen(np.sin(2 * np.pi * 137 * T) * 0.5, FS,
                      true_peak=float("-inf"))["true_peak_clipping_suspected"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="needs ffmpeg for a real AAC round trip")
def test_a_real_aac_round_trip_behaves_as_the_emulation_predicts(tmp_path):
    """The emulation above is only trustworthy if the real codec agrees. This
    is the same experiment through a genuine ffmpeg AAC 128 kbps encode and
    decode, at ffmpeg's DEFAULT 16-bit output — the exact command
    `fridge_scan.py`, `fan_experiment.py` and `check_phone_audio.py` all use."""
    from scipy.io import wavfile
    rng = np.random.default_rng(11)
    clipped = np.clip(2.5 * rng.standard_normal(int(FS * 8)), -1.0, 1.0)
    src, enc, dec = (tmp_path / "s.wav", tmp_path / "s.m4a", tmp_path / "d.wav")
    wavfile.write(src, int(FS), clipped.astype(np.float32))
    for cmd in (["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-c:a", "aac", "-b:a", "128k", str(enc)],
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc),
                 "-ar", str(int(FS)), "-ac", "1", str(dec)]):
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)

    fs_out, data = wavfile.read(dec)
    x = data.astype(np.float64) / 32768.0

    assert clipped_fraction(x) <= CLIP_FRAC_WARN, (
        "real AAC is expected to defeat the flat-top test — that is the bug")
    tp = true_peak_dbtp(x)
    assert tp >= TRUE_PEAK_WARN_DBTP, f"real AAC: {tp:+.2f} dBTP should warn"
    assert screen(x, float(fs_out), true_peak=tp)["clipping_suspected"]
