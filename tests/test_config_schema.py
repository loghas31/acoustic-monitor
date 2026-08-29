"""
tests/test_config_schema.py — backlog T3.5, config validation on startup.

`main.py`, `baseline.py` and `train.py` each did a bare
`yaml.safe_load(path.read_text())` and indexed straight into the result.
Measured before this file existed: `python firmware/main.py --config
<config missing the [window] section> --simulate --no-mqtt --fast --minutes 1
--db /tmp/x.db` crashed with

    File ".../firmware/main.py", line 103, in run
        window_s = cfg["window"]["seconds"]
    KeyError: 'window'

three frames deep, naming neither the file nor what a fix looks like — the
exact "fails, but not legibly" shape `docs/DOC_SELF_REVIEW.md`/T4.3 already
hardened five other failure modes against. `test_a_malformed_config_used_to_fail_illegibly_now_fails_with_a_named_error`
below reproduces that crash on purpose and pins that it no longer happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "firmware"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_schema import ConfigError, load_config, validate_config  # noqa: E402

REAL_CONFIG = ROOT / "firmware" / "config.yaml"


def _load_real() -> dict:
    return yaml.safe_load(REAL_CONFIG.read_text())


# ----------------------------------------------------------------------------
# 1. The repo's own config.yaml must always validate — this is the schema's
#    own regression test against the file it's meant to protect.
# ----------------------------------------------------------------------------

def test_the_repos_own_config_yaml_validates_cleanly():
    validate_config(_load_real(), source=str(REAL_CONFIG))    # must not raise


def test_load_config_reads_and_validates_the_real_file():
    cfg = load_config(REAL_CONFIG)
    assert cfg["device"]["id"] == "dev-0001"


# ----------------------------------------------------------------------------
# 2. Missing sections / keys
# ----------------------------------------------------------------------------

def test_missing_section_is_named_in_the_error():
    cfg = _load_real()
    del cfg["window"]
    with pytest.raises(ConfigError, match=r"missing section \[window\]"):
        validate_config(cfg)


def test_missing_key_within_a_present_section_is_named():
    cfg = _load_real()
    del cfg["mqtt"]["host"]
    with pytest.raises(ConfigError, match=r"missing mqtt\.host"):
        validate_config(cfg)


def test_several_problems_are_all_reported_in_one_error_not_one_at_a_time():
    """The whole point of validating in one pass: a hand-edited config is
    likely to have more than one thing wrong."""
    cfg = _load_real()
    del cfg["window"]
    del cfg["mqtt"]["host"]
    cfg["audio"]["sample_rate"] = -1
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg)
    msg = str(exc.value)
    assert "missing section [window]" in msg
    assert "missing mqtt.host" in msg
    assert "audio.sample_rate" in msg
    assert "3 problems" in msg


# ----------------------------------------------------------------------------
# 3. Wrong types / bad values
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("path,bad_value", [
    ("audio.sample_rate", "sixteen thousand"),
    ("audio.sample_rate", -16000),
    ("audio.sample_rate", 0),
    ("window.seconds", 0),
    ("window.learn_windows", 0.5),           # must be an int, not a float
    ("anomaly.persist_minutes", -30),
    ("mqtt.port", "8883"),                    # string, not int
    ("mqtt.tls", "true"),                     # string, not bool
    ("storage.retention_days", 0),
    ("device.id", ""),                        # empty string
    ("device.id", None),
])
def test_a_bad_value_is_rejected_with_the_dotted_path_named(path, bad_value):
    cfg = _load_real()
    section, key = path.split(".")
    cfg[section][key] = bad_value
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        validate_config(cfg)


def test_empty_strings_are_allowed_where_the_shipped_config_documents_them_as_blank():
    """mqtt.api_key and local_alert.webhook_url are "" in the repo's own
    config.yaml until registration/onboarding fills them in — the schema
    must not treat the documented default as an error."""
    cfg = _load_real()
    cfg["mqtt"]["api_key"] = ""
    cfg["local_alert"]["webhook_url"] = ""
    validate_config(cfg)          # must not raise


def test_a_section_that_is_not_a_mapping_is_rejected():
    cfg = _load_real()
    cfg["mqtt"] = "not a mapping"
    with pytest.raises(ConfigError, match=r"\[mqtt\] must itself be a mapping"):
        validate_config(cfg)


def test_a_non_mapping_top_level_document_is_rejected():
    with pytest.raises(ConfigError, match="did not parse as a mapping"):
        validate_config(["this", "is", "a", "list"])
    with pytest.raises(ConfigError, match="did not parse as a mapping"):
        validate_config(None)


# ----------------------------------------------------------------------------
# 4. The one semantic (not just type) check: accelerometer bandwidth
# ----------------------------------------------------------------------------

def test_an_accelerometer_sample_rate_too_slow_for_the_resonance_band_is_flagged():
    """the system overview (not in this public copy) §2: the whole reason the part is an IIS3DWB and not
    an ADXL345-class chip is bandwidth into the 1-20 kHz resonance band. A
    config that type-checks cleanly but specifies an ADXL345-slow rate
    should not pass silently — it reproduces the exact mistake the hardware
    section exists to avoid."""
    cfg = _load_real()
    cfg["accelerometer"]["sample_rate"] = 800      # ADXL345-class, ~400 Hz usable
    with pytest.raises(ConfigError, match="provably blind"):
        validate_config(cfg)


def test_the_shipped_6400hz_rate_clears_the_bandwidth_check():
    cfg = _load_real()
    assert cfg["accelerometer"]["sample_rate"] == 6400
    validate_config(cfg)      # must not raise


# ----------------------------------------------------------------------------
# 5. load_config's file-level errors
# ----------------------------------------------------------------------------

def test_load_config_on_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_on_invalid_yaml_names_the_parse_error(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("device:\n  id: [unterminated")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


# ----------------------------------------------------------------------------
# 6. End to end: main.py / baseline.py now refuse a malformed config legibly
# ----------------------------------------------------------------------------

def _write_broken_config(tmp_path: Path) -> Path:
    cfg = _load_real()
    del cfg["window"]                     # the exact field main.py's run() hit first
    path = tmp_path / "broken_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_a_malformed_config_used_to_fail_illegibly_now_fails_with_a_named_error(tmp_path):
    """Reproduces the exact crash quoted in this file's module docstring
    (missing [window], hit at `cfg["window"]["seconds"]` three frames into
    main.py's run()) and pins that it is now a legible, one-shot refusal
    instead of a bare KeyError."""
    broken = _write_broken_config(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "main.py"),
         "--config", str(broken), "--simulate", "--no-mqtt", "--fast",
         "--minutes", "1", "--db", str(tmp_path / "state.db")],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert "KeyError" not in r.stderr, (
        f"must not be a bare KeyError any more:\n{r.stderr}")
    assert "missing section [window]" in r.stderr, r.stderr


def test_baseline_py_also_refuses_a_malformed_config_legibly(tmp_path):
    broken = _write_broken_config(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "baseline.py"),
         "--config", str(broken), "--simulate", "--windows", "8",
         "--out", str(tmp_path / "b.npz"), "--db", str(tmp_path / "state.db")],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert "KeyError" not in r.stderr, (
        f"must not be a bare KeyError any more:\n{r.stderr}")
    assert "missing section [window]" in r.stderr, r.stderr


def test_a_valid_config_still_runs_main_py_end_to_end(tmp_path):
    """The other half of the regression: validation must not reject the
    real, valid config it is meant to let through."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "firmware" / "main.py"),
         "--simulate", "--no-mqtt", "--fast", "--minutes", "1",
         "--db", str(tmp_path / "state.db")],
        cwd=str(ROOT / "firmware"), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "done:" in r.stdout
