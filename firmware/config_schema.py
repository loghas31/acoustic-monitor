"""
config_schema.py — validate config.yaml on load (backlog T3.5).

WHY THIS EXISTS
----------------------------------------------------------------------------
Before this file, `main.py`, `baseline.py` and `train.py` each did
`yaml.safe_load(path.read_text())` and then indexed straight into the
result: `cfg["window"]["seconds"]`, `cfg["mqtt"]["host"]`, and so on, six
call sites deep across three modules. A `config.yaml` with a typo'd section
name, a missing key, or a string where a number belongs did not fail at
startup — it failed the first time the specific line touching that field
ran, as a bare `KeyError` or `TypeError` with no context, possibly minutes
into a `--simulate` run or after MQTT had already connected. That is exactly
the failure shape this project has spent most of its self-review budget
hardening against elsewhere (`docs/DOC_SELF_REVIEW.md`, `T4.3`): code that
fails, but not legibly, to a student at a customer site with no Python
traceback experience.

`validate_config` checks the WHOLE file in one pass and raises ONE
`ConfigError` naming every problem found, not just the first — a config
edited by hand (this is the onboarding-flow-writes-it file today, but a
human will edit one by hand at the bench) is more likely to have several
things wrong than exactly one, and discovering them one crash at a time is
its own bad experience.

WHAT IS DELIBERATELY NOT VALIDATED
----------------------------------------------------------------------------
`accelerometer.driver` is read from config.yaml but, checked directly
against `capture.make_source`, is not actually branched on by any consumer
today — `HardwareSource` always talks to the real IIS3DWB regardless of its
value. Enforcing an enum on a field nothing reads would be presumptuous, so
this schema only requires it be a non-empty string, matching what the field
currently means: documentation, not configuration.

`firmware/train.py` (the unused, untested v1.5 cloud-autoencoder path — see
the task backlog (not in this public copy), no test file references it) reads `cfg["anomaly"]["sigma_k"]`,
which does NOT exist in the repository's own `firmware/config.yaml`. That is
a real latent bug in `train.py`, found while building this schema, but out
of scope for T3.5: `train.py` is not part of the v1 product's config
contract (v1 is Mahalanobis, not the isolation-forest/autoencoder train.py
describes) and adding `sigma_k` to this schema would validate a field the
shipped config correctly does not have. Recorded in the task backlog (not in this public copy) instead
of silently working around it.
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised by `validate_config`/`load_config`. `str(exc)` is a
    newline-joined, numbered list of every problem found — designed to be
    printed directly, not parsed."""


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _positive_number(path: str, v: Any, errors: list[str]) -> None:
    if not _is_number(v) or v <= 0:
        errors.append(f"{path} must be a positive number, got {v!r}")


def _positive_int(path: str, v: Any, errors: list[str]) -> None:
    if not (isinstance(v, int) and not isinstance(v, bool)) or v <= 0:
        errors.append(f"{path} must be a positive integer, got {v!r}")


def _non_empty_str(path: str, v: Any, errors: list[str]) -> None:
    if not isinstance(v, str) or not v.strip():
        errors.append(f"{path} must be a non-empty string, got {v!r}")


def _string(path: str, v: Any, errors: list[str]) -> None:
    """Allows an empty string — several fields (mqtt.api_key,
    local_alert.webhook_url) are documented as legitimately blank until
    registration/onboarding fills them in."""
    if not isinstance(v, str):
        errors.append(f"{path} must be a string, got {v!r}")


def _bool(path: str, v: Any, errors: list[str]) -> None:
    if not isinstance(v, bool):
        errors.append(f"{path} must be true or false, got {v!r}")


# section -> {key: (required, checker)}. Mirrors exactly what
# main.py / baseline.py / capture.py / mqtt_client.py index into today —
# grepped, not guessed (see the module docstring for the one field this
# deliberately excludes).
_SCHEMA: dict[str, dict[str, tuple[bool, Any]]] = {
    "device": {
        "id": (True, _non_empty_str),
        "name": (True, _non_empty_str),
    },
    "audio": {
        "sample_rate": (True, _positive_number),
        "channels": (True, _positive_int),
    },
    "accelerometer": {
        "driver": (True, _non_empty_str),
        "sample_rate": (True, _positive_number),
        "range_g": (True, _positive_number),
    },
    "window": {
        "seconds": (True, _positive_number),
        "learn_windows": (True, _positive_int),
    },
    "anomaly": {
        "persist_minutes": (True, _positive_number),
    },
    "mqtt": {
        "host": (True, _non_empty_str),
        "port": (True, _positive_int),
        "tls": (True, _bool),
        "ca_cert": (True, _string),
        "api_key": (True, _string),
        "base_topic": (True, _non_empty_str),
    },
    "local_alert": {
        "webhook_url": (True, _string),
    },
    "storage": {
        "sqlite_path": (True, _non_empty_str),
        "baseline_path": (True, _non_empty_str),
        "retention_days": (True, _positive_number),
    },
}


def validate_config(cfg: Any, source: str = "config.yaml") -> None:
    """Raises `ConfigError` naming every problem in `cfg`, or returns None.

    `source` is only used to make the error message name the file, since by
    the time this runs the caller's `Path` is out of scope."""
    errors: list[str] = []

    if not isinstance(cfg, dict):
        raise ConfigError(
            f"{source} did not parse as a mapping (got {type(cfg).__name__}) "
            f"— is the file empty, or is it a YAML list/scalar at the top "
            f"level instead of key: value sections?")

    for section, keys in _SCHEMA.items():
        if section not in cfg:
            errors.append(f"missing section [{section}]")
            continue
        block = cfg[section]
        if not isinstance(block, dict):
            errors.append(
                f"[{section}] must itself be a mapping of key: value pairs, "
                f"got {type(block).__name__}")
            continue
        for key, (required, checker) in keys.items():
            path = f"{section}.{key}"
            if key not in block:
                if required:
                    errors.append(f"missing {path}")
                continue
            checker(path, block[key], errors)

    # One cross-field sanity check worth having: the accelerometer sample
    # rate is what the whole hardware choice in the system overview (not in this public copy) §2 is FOR
    # (resonances live at 1-20 kHz; an accelerometer sampled below ~2.2 kHz
    # is provably blind to any of that, the same argument verify_signals.py
    # makes for the microphone). Catches a config that parses and type-checks
    # cleanly but would silently reproduce the exact ADXL345-class mistake
    # the hardware section of this project exists to avoid.
    accel = cfg.get("accelerometer", {})
    if isinstance(accel, dict) and _is_number(accel.get("sample_rate")):
        if accel["sample_rate"] < 2200:
            errors.append(
                f"accelerometer.sample_rate = {accel['sample_rate']} Hz is "
                f"below ~2.2 kHz — the whole point of the IIS3DWB "
                f"(the system overview (not in this public copy) §2) is bandwidth into the 1-20 kHz "
                f"resonance band a bearing fault rings at; a slower "
                f"accelerometer is provably blind to it, the same failure "
                f"mode verify_signals.py demonstrates for the microphone")

    if errors:
        numbered = "\n".join(f"  {i}. {e}" for i, e in enumerate(errors, 1))
        raise ConfigError(
            f"{source} failed validation ({len(errors)} problem"
            f"{'s' if len(errors) != 1 else ''}):\n{numbered}")


def load_config(path: Path | str) -> dict:
    """Reads and validates a config file in one call — the function
    `main.py`/`baseline.py` call instead of a bare `yaml.safe_load`."""
    import yaml
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise ConfigError(
            f"{path} does not exist. Copy firmware/config.yaml and edit it, "
            f"or pass --config pointing at the real one.") from None
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    validate_config(cfg, source=str(path))
    return cfg
