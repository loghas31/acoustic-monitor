"""
tests/test_provision_scripts.py — backlog T2.1,
`scripts/provision_pi.sh` + `scripts/check_provision.sh`.

Neither script can run end to end in this sandbox — no real Pi, no
`/boot/firmware/config.txt`, no root-owned `/opt`, `/etc/systemd/system`
(confirmed this run: `apt-get install` fails here with "Permission denied"
even attempting it, and there is no `raspi-config`). That is exactly the
scope the task backlog (not in this public copy) T2.1 sets for this pair: "Cannot be run in the
sandbox — lint with bash -n and pair it with scripts/check_provision.sh
that verifies a provisioned system." This file is that lint, automated —
syntax validity, and that every file path the scripts reference against the
REPO actually exists, so a future rename elsewhere in the repo doesn't
silently break provisioning without anyone noticing until they're on a real
Pi with no way to debug it easily.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROVISION = ROOT / "scripts" / "provision_pi.sh"
CHECK = ROOT / "scripts" / "check_provision.sh"


def _bash_n(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10)


def test_provision_pi_has_valid_bash_syntax():
    r = _bash_n(PROVISION)
    assert r.returncode == 0, r.stderr


def test_check_provision_has_valid_bash_syntax():
    r = _bash_n(CHECK)
    assert r.returncode == 0, r.stderr


def test_provision_pi_uses_strict_mode():
    text = PROVISION.read_text()
    assert "set -euo pipefail" in text, (
        "an idempotent provisioning script that silently continues past a "
        "failed step (a missing package, a bad path) is worse than one "
        "that stops loudly — see the task backlog (not in this public copy)'s own rule for scripts")


def test_check_provision_deliberately_does_not_use_dash_e():
    """The opposite design choice, on purpose: a HEALTH CHECK that aborts
    on its first failed test would hide every check after it. Pinned so a
    future edit doesn't 'clean up' this into set -e and silently break the
    'run every check, report all of them' contract."""
    text = CHECK.read_text()
    assert "set -uo pipefail" in text
    assert "set -euo pipefail" not in text


def test_provision_pi_references_real_files_in_this_repo():
    """Every file provision_pi.sh reads FROM the checked-out repo (not from
    the target Pi) must actually exist, or the script fails on line 1 of
    real use with no chance to fix it remotely."""
    for rel in ("firmware/main.py", "firmware/acoustic-monitor.service",
               "firmware/config.yaml", "firmware/requirements.txt"):
        assert (ROOT / rel).exists(), f"provision_pi.sh references {rel}, which does not exist"


def test_provision_pi_installs_into_a_venv_not_the_system_python():
    """The specific bug this script's own header comment says it caught:
    the checked-in systemd unit's ExecStart runs /usr/bin/python3, which
    cannot see anything installed in a venv. Pin that the installed unit's
    ExecStart is patched to point at the venv, not left as the checked-in
    file's system-python ExecStart."""
    text = PROVISION.read_text()
    assert "sed" in text and "ExecStart=" in text
    assert "$VENV_DIR/bin/python3" in text


def test_check_provision_checks_for_the_same_venv_fix():
    text = CHECK.read_text()
    assert "/opt/acoustic-monitor/venv/bin/python3" in text
    assert "ExecStart" in text


def test_provision_pi_never_overwrites_an_existing_deployed_config():
    text = PROVISION.read_text()
    assert '[ ! -f "$ETC_DIR/config.yaml" ]' in text, (
        "config.yaml holds this device's own id/api_key — re-running "
        "provisioning must not clobber it")


def test_check_provision_exits_nonzero_on_a_synthetic_failure(tmp_path):
    """Cannot exercise the real system checks (no root, no real Pi — see
    module docstring), but CAN pin the exit-code CONTRACT other tooling
    would rely on (a boot-time health check, CI) — that iterating a mix of
    pass/fail `check` calls sets a nonzero exit code without aborting
    early. Reproduced with a tiny synthetic script using the same check()/
    pass()/fail() shape, not the real script (which needs a real Pi's
    filesystem to run meaningfully)."""
    synthetic = tmp_path / "mini_check.sh"
    synthetic.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "FAILED=0\n"
        "pass() { echo \"PASS $1\"; }\n"
        "fail() { echo \"FAIL $1\"; FAILED=1; }\n"
        "check() { if eval \"$2\" >/dev/null 2>&1; then pass \"$1\"; else fail \"$1\"; fi; }\n"
        "check \"true always passes\" \"true\"\n"
        "check \"false always fails\" \"false\"\n"
        "check \"a check after a failure still runs\" \"true\"\n"
        "[ \"$FAILED\" -eq 1 ] && exit 1 || exit 0\n"
    )
    r = subprocess.run(["bash", str(synthetic)], capture_output=True, text=True, timeout=10)
    assert r.returncode == 1
    assert "PASS true always passes" in r.stdout
    assert "FAIL false always fails" in r.stdout
    assert "PASS a check after a failure still runs" in r.stdout, (
        "a failed check must not stop the rest from running")
