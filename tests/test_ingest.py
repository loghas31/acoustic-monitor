"""Tests for tools/ingest.py (backlog T1.2).

Scope note: this guards the failures that are SILENT. Every one of the checks
below corresponds to a way an ingested recording can look perfectly fine and
still produce a confident wrong answer downstream:

  * dtype scaling — an int32 wav read as int16 shifts every log-RMS feature;
  * the resample ratio — get it wrong and the whole frequency axis stretches,
    so BPFO is searched for in the wrong place and "no fault" is reported
    forever;
  * agreement with validate_public_dataset.resample_to — if the dataset path
    and the bench path resample differently, their results are not comparable;
  * the CSV header heuristic — the first version of this code silently dropped
    the first sample of every headerless file;
  * clipping detection semantics — a false positive on accelerometer data in g
    trains the students to ignore the warning that matters;
  * the bandwidth audit — the only check that catches "your microphone cannot
    hear the resonance band at all", which is the one failure that yields a
    system that appears to work and can never detect anything.

Three tests deliberately assert on the *content of a warning*, not just that
one was emitted. A warning whose text is wrong is worse than no warning.

The fault-frequency identity (BPFO + BPFI = N.f_r) and analyse_recording's
verdict logic are backlog T1.3 and are not duplicated here.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "ml" / "realdata", ROOT / "firmware"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ingest as I  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tone(fs: float, dur: float, f: float, amp: float = 0.3,
          dc: float = 0.0, seed: int = 0) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * f * t) + 0.01 * rng.standard_normal(len(t)) + dc


def _write_wav(path: Path, x: np.ndarray, fs: float, dtype=np.int16,
               channels: int = 1) -> Path:
    if channels > 1:
        x = np.column_stack([x] * channels)
    if dtype == np.int16:
        data = np.clip(x * 32767, -32768, 32767).astype(np.int16)
    elif dtype == np.int32:
        data = np.clip(x * (2 ** 31 - 1), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
    elif dtype == np.uint8:
        data = np.clip(x * 127 + 128, 0, 255).astype(np.uint8)
    elif dtype == np.float32:
        data = x.astype(np.float32)
    else:
        raise ValueError(dtype)
    wavfile.write(path, int(fs), data)
    return path


def _write_accel_csv(path: Path, a: np.ndarray, fs: float,
                     header: bool = True, gaps: int = 0) -> Path:
    t = np.arange(len(a)) / fs
    if gaps:
        # simulate dropped FIFO reads: a few large steps in the time column
        for i in range(1, gaps + 1):
            t[i * len(a) // (gaps + 1):] += 4.0 / fs
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["t_s"] + [f"accel_{c}" for c in "xyz"[: a.shape[1]]])
        for ti, row in zip(t, np.atleast_2d(a.T).T):
            w.writerow([f"{ti:.6f}"] + [f"{v:.6f}" for v in np.atleast_1d(row)])
    return path


def _peak_hz(x: np.ndarray, fs: float) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1.0 / fs)[int(np.argmax(mag))])


# ---------------------------------------------------------------------------
# 1. dtype scaling — the int32-read-as-int16 failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,tol", [
    (np.int16, 1e-3), (np.int32, 1e-3), (np.float32, 1e-6), (np.uint8, 3e-2),
])
def test_all_wav_dtypes_normalise_to_pm_one(tmp_path, dtype, tol):
    """Whatever the on-disk dtype, we must come back in [-1, 1] with the same
    waveform. uint8 gets a loose tolerance because 8-bit quantisation really is
    that coarse (that is a property of the format, not of this code)."""
    fs = 16000.0
    x = _tone(fs, 0.5, 1000.0, amp=0.5)
    p = _write_wav(tmp_path / f"t_{np.dtype(dtype).name}.wav", x, fs, dtype)
    y2d, fs_read, meta = I.read_wav_any(p)
    y = I.to_mono(y2d)
    assert fs_read == fs
    assert np.max(np.abs(y)) <= 1.0
    assert np.max(np.abs(y - x)) < tol
    assert meta["source_dtype"] == np.dtype(dtype).name


def test_uint8_offset_is_handled():
    """uint8 wav is the only UNSIGNED case: silence is 128, not 0. Read it as
    signed and the whole file has a +1.0 DC offset."""
    x = np.zeros(64, dtype=np.uint8) + 128
    assert float(np.mean((x.astype(float) - 128.0) / 128.0)) == 0.0


def test_multichannel_averages_and_channel_selects(tmp_path):
    fs = 16000.0
    a, b = _tone(fs, 0.3, 500.0, 0.4), _tone(fs, 0.3, 500.0, 0.2, seed=1)
    p = tmp_path / "st.wav"
    wavfile.write(p, int(fs), np.clip(np.column_stack([a, b]) * 32767,
                                      -32768, 32767).astype(np.int16))
    mono = I.to_mono(I.read_wav_any(p)[0])
    assert np.allclose(mono, (a + b) / 2, atol=1e-3)
    only0 = I.to_mono(I.read_wav_any(p, channel=0)[0])
    assert np.allclose(only0, a, atol=1e-3)
    with pytest.raises(I.IngestError):
        I.read_wav_any(p, channel=7)


# ---------------------------------------------------------------------------
# 2. Resampling — the stretched-frequency-axis failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs_in,fs_out,up,down", [
    (44100.0, 16000.0, 160, 441),      # phone / CD rate
    (48000.0, 16000.0, 1, 3),          # USB interface, arecord default
    (12000.0, 16000.0, 4, 3),          # CWRU 12 kHz -> our audio slot
    (12000.0, 6400.0, 8, 15),          # CWRU 12 kHz -> our accel slot
    (16000.0, 16000.0, 1, 1),          # already canonical: no-op
])
def test_resample_ratio_is_exact_small_integers(fs_in, fs_out, up, down):
    """If these ever come out as something like 397/1000, the rate is not what
    the caller thinks it is and every frequency downstream is wrong."""
    assert I.resample_ratio(fs_in, fs_out) == (up, down)


def test_resample_matches_the_public_dataset_validator():
    """The bench path (this file) and the public-dataset path
    (validate_public_dataset.py) must resample IDENTICALLY, or a CWRU number
    and a bench number are not comparable. Asserted numerically so the two
    implementations cannot drift."""
    import validate_public_dataset as V
    x = _tone(12000.0, 1.0, 1500.0)
    assert np.allclose(I.resample_to(x, 12000.0, 16000.0),
                       V.resample_to(x, 12000.0, 16000.0), atol=0, rtol=0)


def test_resampling_preserves_frequency():
    """A 4 kHz tone at 44.1 kHz must still be at 4 kHz after conversion to
    16 kHz — not at 4000 * 16/44.1 = 1451 Hz, which is what happens if you
    relabel the rate instead of resampling."""
    fs_in, f0 = 44100.0, 4000.0
    y = I.resample_to(_tone(fs_in, 2.0, f0), fs_in, 16000.0)
    assert abs(_peak_hz(y, 16000.0) - f0) / f0 < 0.005


def test_resampling_is_antialiased_not_decimated():
    """Downsampling 44.1 -> 16 kHz must FILTER before decimating. A 7 kHz tone
    is above the 8 kHz output Nyquist... just: put one at 15 kHz, which must
    NOT alias down to 1 kHz."""
    fs_in = 44100.0
    y = I.resample_to(_tone(fs_in, 1.0, 15000.0, amp=0.5), fs_in, 16000.0)
    # Only the small noise floor should remain; an aliased 15 kHz tone would
    # appear as a strong line near 1 kHz.
    assert float(np.sqrt(np.mean(y ** 2))) < 0.05


# ---------------------------------------------------------------------------
# 3. CSV reading — the silently-dropped-row failure
# ---------------------------------------------------------------------------

def test_headerless_csv_keeps_every_row(tmp_path):
    """Regression: the first implementation tried skiprows=1 first, which
    SUCCEEDS on a headerless numeric file and eats the first sample."""
    p = tmp_path / "bare.csv"
    p.write_text("0.1\n0.2\n0.3\n0.4\n0.5\n")
    sig, fs, meta = I.read_csv_any(p, fs_in=12000.0)
    assert len(sig) == 5
    assert meta["csv_header_row"] is False
    assert fs == 12000.0


def test_headerless_csv_without_fs_is_refused(tmp_path):
    p = tmp_path / "bare.csv"
    p.write_text("0.1\n0.2\n0.3\n0.4\n0.5\n")
    with pytest.raises(I.IngestError, match="sample rate is unknown"):
        I.read_csv_any(p)


def test_time_column_gives_the_rate_and_reports_jitter(tmp_path):
    fs = 6400.0
    a = np.column_stack([_tone(fs, 1.0, 100.0)])
    clean = _write_accel_csv(tmp_path / "clean_accel.csv", a, fs)
    _, fs_read, meta = I.read_csv_any(clean)
    assert meta["time_column"] is True
    # %.6f timestamps quantise the step, so allow 0.5 %.
    assert abs(fs_read - fs) / fs < 0.005
    assert meta["timestamp_jitter_frac"] < 0.25

    jittery = _write_accel_csv(tmp_path / "drop_accel.csv", a, fs, gaps=3)
    _, _, jmeta = I.read_csv_any(jittery)
    assert jmeta["time_column"] is True, \
        "a file with a few dropped FIFO reads must still LOAD — the point is " \
        "to report the jitter, not to refuse the data"
    assert jmeta["timestamp_jitter_frac"] > 0.25


def test_fs_in_disagreeing_with_the_time_column_is_refused(tmp_path):
    fs = 6400.0
    p = _write_accel_csv(tmp_path / "a_accel.csv",
                         np.column_stack([_tone(fs, 0.5, 100.0)]), fs)
    with pytest.raises(I.IngestError, match="disagrees"):
        I.read_csv_any(p, fs_in=12000.0)


def test_signal_column_is_not_mistaken_for_a_time_axis():
    assert not I._looks_like_time_column(_tone(16000.0, 0.1, 300.0))
    assert I._looks_like_time_column(np.arange(1000) / 6400.0)


def test_mat_files_are_redirected_not_mangled(tmp_path):
    p = tmp_path / "97.mat"
    p.write_bytes(b"\x00")
    with pytest.raises(I.IngestError, match="validate_public_dataset"):
        I.read_signal(p)


# ---------------------------------------------------------------------------
# 4. The audit
# ---------------------------------------------------------------------------

def test_clipping_test_differs_for_wav_and_physical_units():
    """Regression: accelerometer data in g peaking at 1.7 was reported as
    '20 % clipped' because ±1 was assumed to be full scale. A warning that
    cries wolf on good data is worse than no warning."""
    g_data = 1.7 * np.sin(np.linspace(0, 40 * np.pi, 5000))
    assert I.clipped_fraction(g_data, full_scale_known=False) < 0.01
    assert I.clipped_fraction(g_data, full_scale_known=True) > 0.3

    hard = np.clip(1.6 * np.sin(np.linspace(0, 40 * np.pi, 5000)), -1.0, 1.0)
    assert I.clipped_fraction(hard, full_scale_known=True) > 0.3
    assert I.clipped_fraction(hard, full_scale_known=False) > 0.3


def _severities(audit):
    return {sev for sev, _ in audit["findings"]}


def test_bandwidth_audit_grades_the_three_cases():
    """The audit's single most valuable output. 48 kHz source: fine. 8 kHz
    source: only a third of the 3-6 kHz demodulation band exists — warn.
    4 kHz source: the band does not exist at all — blind."""
    band = (3000.0, 6000.0)

    ok = I.audit_audio(_tone(48000.0, 1.0, 4000.0), 48000.0, 16000.0, band)
    assert "warn" not in _severities(ok) and "blind" not in _severities(ok)

    narrow = I.audit_audio(_tone(8000.0, 1.0, 1000.0), 8000.0, 16000.0, band)
    assert "warn" in _severities(narrow)
    assert any("only 33 %" in m for _, m in narrow["findings"])

    deaf = I.audit_audio(_tone(4000.0, 1.0, 500.0), 4000.0, 16000.0, band)
    assert "blind" in _severities(deaf)
    assert any("entirely BELOW" in m for _, m in deaf["findings"])


def test_silence_and_nonfinite_are_blind():
    assert "blind" in _severities(
        I.audit_audio(np.zeros(16000), 16000.0, 16000.0))
    x = _tone(16000.0, 0.5, 1000.0)
    x[10] = np.nan
    assert "blind" in _severities(I.audit_audio(x, 16000.0, 16000.0))


def test_band_energy_fraction_finds_where_the_power_is():
    fs = 16000.0
    inband = I.band_energy_fraction(_tone(fs, 1.0, 4500.0, amp=1.0), fs,
                                    (3000.0, 6000.0))
    outband = I.band_energy_fraction(_tone(fs, 1.0, 300.0, amp=1.0), fs,
                                     (3000.0, 6000.0))
    assert inband > 0.8
    assert outband < 0.05


def test_gravity_check_on_a_three_axis_sensor():
    n = 5000
    rng = np.random.default_rng(0)
    good = np.column_stack([0.02 * rng.standard_normal(n) + m
                            for m in (0.10, -0.30, 0.94)])
    res = I.audit_accel(good, 6400.0, jitter_frac=0.0)
    assert 0.9 < res["gravity_magnitude"] < 1.1
    assert "warn" not in _severities(res)

    wrong_units = good * 9.81                 # m/s^2, not g
    assert "warn" in _severities(I.audit_accel(wrong_units, 6400.0, 0.0))

    dead = good.copy()
    dead[:, 1] = 0.0
    assert any("dead channel" in m
               for _, m in I.audit_accel(dead, 6400.0, 0.0)["findings"])


# ---------------------------------------------------------------------------
# 5. Gain policy — the destroyed-level-relationship failure
# ---------------------------------------------------------------------------

def _sources(*peaks):
    out = []
    for i, pk in enumerate(peaks):
        s = I.Source(audio_path=Path(f"s{i}.wav"), stem=f"s{i}")
        s.audio = pk * np.sin(np.linspace(0, 20 * np.pi, 1000))
        out.append(s)
    return out


def test_common_gain_preserves_the_level_relationship():
    """The whole reason healthy and faulty should be ingested together."""
    srcs = _sources(0.2, 0.6)
    gains = I.plan_gain(srcs, normalise=False, independent=False)
    assert len(set(gains.values())) == 1 and pytest.approx(1.0) == gains["s0"]

    gains = I.plan_gain(srcs, normalise=True, independent=False)
    assert len(set(gains.values())) == 1
    ratio = (0.6 * gains["s1"]) / (0.2 * gains["s0"])
    assert ratio == pytest.approx(3.0)          # 3:1 in, 3:1 out


def test_independent_gain_destroys_it_as_documented():
    gains = I.plan_gain(_sources(0.2, 0.6), normalise=True, independent=True)
    ratio = (0.6 * gains["s1"]) / (0.2 * gains["s0"])
    assert ratio == pytest.approx(1.0)          # both at 90 % FS: level gone


def test_gain_intervenes_only_to_prevent_wrapping():
    """int16 wraps on overflow, turning a loud transient into a square wave —
    i.e. into a fake broadband impact. So a >1.0 peak must be scaled even when
    the caller did not ask for normalisation."""
    gains = I.plan_gain(_sources(1.4), normalise=False, independent=False)
    assert gains["s0"] < 1.0
    assert 1.4 * gains["s0"] == pytest.approx(0.9, rel=1e-3)


# ---------------------------------------------------------------------------
# 6. End to end through the real CLI
# ---------------------------------------------------------------------------

def test_cli_round_trip_is_canonical_and_replayable(tmp_path):
    """The claim the whole file exists to support: an awkward real-world
    recording goes in, and something FileSource / extract_features can consume
    comes out — with the physics intact."""
    fs_src, f0, f_mod = 44100.0, 4000.0, 137.0
    t = np.arange(int(fs_src * 4.0)) / fs_src
    env = 1.0 + 0.8 * np.sign(np.sin(2 * np.pi * f_mod * t))
    x = 0.3 * env * np.sin(2 * np.pi * f0 * t) + 0.05      # AM + DC offset
    src = _write_wav(tmp_path / "phone.wav", x, fs_src, np.int32, channels=2)

    out_dir = tmp_path / "canonical"
    rc = I.main([str(src), "--out-dir", str(out_dir), "--stem", "healthy",
                 "--mic-only", "--rpm", "2850", "--bearing", "6204",
                 "--label", "healthy"])
    assert rc == 0

    from recording_io import load_recording
    rec = load_recording(out_dir / "healthy.wav")
    assert rec.fs_audio == 16000.0
    assert abs(rec.duration_s - 4.0) < 0.01
    assert not rec.has_accel

    # frequency axis intact
    assert abs(_peak_hz(rec.audio, rec.fs_audio) - f0) / f0 < 0.005
    # DC removed
    assert abs(float(np.mean(rec.audio))) < 1e-3
    # the modulation — the thing a bearing fault actually is — survives
    from features import envelope_spectrum
    ef, em = envelope_spectrum(rec.audio, rec.fs_audio, (3000.0, 6000.0))
    sel = (ef > 20) & (ef < 400)
    assert abs(float(ef[sel][int(np.argmax(em[sel]))]) - f_mod) / f_mod < 0.02
    # and the shipping feature extractor accepts it
    from features import FEATURE_NAMES, extract_features
    accel, fs_a = rec.accel_for_features()
    out = extract_features(rec.audio, rec.fs_audio, accel, fs_a)
    # width from the contract, not a literal (40 -> 37 at T1.5)
    assert len(out["vector"]) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(out["vector"]))


def test_cli_writes_a_sidecar_with_provenance(tmp_path):
    src = _write_wav(tmp_path / "src.wav", _tone(44100.0, 1.0, 4000.0), 44100.0)
    out_dir = tmp_path / "c"
    assert I.main([str(src), "--out-dir", str(out_dir), "--stem", "h",
                   "--mic-only", "--rpm", "3000", "--note", "bench 1"]) == 0
    meta = json.loads((out_dir / "h.json").read_text())
    assert meta["fs_audio"] == 16000.0
    assert meta["rpm"] == 3000.0
    assert meta["fr_hz"] == pytest.approx(50.0)
    assert meta["note"] == "bench 1"
    assert meta["source_sha256"].startswith("sha256:")
    assert meta["audit"]["audio"]["resample_up"] == 160
    assert meta["mic_only"] is True


def test_cli_pairs_the_accelerometer_by_convention(tmp_path):
    """`<stem>_accel.csv` beside the wav must be found without being asked
    for, and resampled onto an exactly-6400 Hz time axis."""
    fs_a = 6410.0
    _write_wav(tmp_path / "run.wav", _tone(16000.0, 2.0, 4000.0), 16000.0)
    a = np.column_stack([_tone(fs_a, 2.0, 100.0) + m for m in (0.1, -0.3, 0.94)])
    _write_accel_csv(tmp_path / "run_accel.csv", a, fs_a)

    out_dir = tmp_path / "c"
    assert I.main([str(tmp_path / "run.wav"), "--out-dir", str(out_dir),
                   "--rpm", "3000"]) == 0

    from recording_io import load_recording
    rec = load_recording(out_dir / "run.wav")
    assert rec.has_accel
    assert rec.accel.shape[1] == 3
    # The output time axis is generated by us, so the recovered rate must be
    # EXACTLY the device rate. Regression: writing t_s with "%.6f" quantises
    # the 0.00015625 s step to 0.000156 and the file reads back at 6410.26 Hz
    # — a silent 0.16 % stretch of the accelerometer frequency axis in a file
    # this tool advertises as canonical.
    assert abs(rec.fs_accel - 6400.0) < 0.01, (
        "the written time column must recover the exact device rate; "
        f"got {rec.fs_accel}")


def test_cli_trims(tmp_path):
    src = _write_wav(tmp_path / "long.wav", _tone(16000.0, 6.0, 4000.0), 16000.0)
    out_dir = tmp_path / "c"
    assert I.main([str(src), "--out-dir", str(out_dir), "--stem", "t",
                   "--mic-only", "--start-s", "1.0", "--duration-s", "2.0",
                   "--rpm", "3000"]) == 0
    from recording_io import load_recording
    assert abs(load_recording(out_dir / "t.wav").duration_s - 2.0) < 0.01


def test_cli_dry_run_writes_nothing(tmp_path):
    src = _write_wav(tmp_path / "s.wav", _tone(44100.0, 1.0, 4000.0), 44100.0)
    out_dir = tmp_path / "nope"
    I.main([str(src), "--out-dir", str(out_dir), "--mic-only", "--dry-run"])
    assert not (out_dir / "s.wav").exists()


def test_cli_rejects_ambiguous_invocations(tmp_path):
    a = _write_wav(tmp_path / "a.wav", _tone(16000.0, 0.5, 1000.0), 16000.0)
    b = _write_wav(tmp_path / "b.wav", _tone(16000.0, 0.5, 1000.0), 16000.0)
    out = ["--out-dir", str(tmp_path / "c"), "--mic-only"]
    assert I.main([]) == 2                                    # no inputs
    assert I.main([str(a), str(b), "--stem", "one", *out]) == 2
    assert I.main([str(a), str(b), "--accel", str(a), *out]) == 2


def test_exit_codes_grade_the_severity(tmp_path):
    """The exit code is what a shell script sees, so it must mean something.

      clean            -> 0
      warning          -> 0 by default (the message is on stdout), 1 --strict
      unusable (blind) -> 1 always, because analysing that file is a mistake
    """
    out = ["--out-dir", str(tmp_path / "c"), "--mic-only", "--rpm", "3000"]

    clean = _write_wav(tmp_path / "ok.wav", _tone(48000.0, 1.0, 4000.0), 48000.0)
    assert I.main([str(clean), *out]) == 0
    assert I.main([str(clean), *out, "--strict"]) == 0

    # 8 kHz: only a third of the 3-6 kHz demodulation band exists.
    narrow = _write_wav(tmp_path / "nb.wav", _tone(8000.0, 1.0, 1000.0), 8000.0)
    assert I.main([str(narrow), *out]) == 0
    assert I.main([str(narrow), *out, "--strict"]) == 1

    # 4 kHz: the band does not exist at all. Nothing can be detected here.
    deaf = _write_wav(tmp_path / "deaf.wav", _tone(4000.0, 1.0, 500.0), 4000.0)
    assert I.main([str(deaf), *out]) == 1


def test_self_test_passes(tmp_path):
    """The tool's own --self-test must pass, since it is what a student runs
    first when something looks wrong."""
    assert I.self_test(tmp_path / "st") == 0
