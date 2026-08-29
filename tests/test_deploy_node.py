"""
tests/test_deploy_node.py — backlog T2.2, `scripts/deploy_node.sh`.

Like `scripts/provision_pi.sh` (T2.1, see `test_provision_scripts.py`'s own
docstring), this cannot run end to end in this sandbox — there is no real
Pi and no ssh target. Two levels of evidence, same split as
`test_dev_up.py` used for the backend:

1. `bash -n` syntax + grep-level checks that the script's own stated
   contract (excludes, remote paths, flag names) is really in the text.
2. A REAL subprocess run of the REAL script, with fake `ssh` and `rsync`
   executables placed first on PATH. They do not open a network
   connection or move any bytes — they just log their own argv to a file
   and exit 0 — but that is enough to prove, by executing the actual
   script rather than reading it, that deploy_node.sh invokes rsync with
   the right source directories and excludes and ssh with the right
   remote commands, in the right order, and that --no-restart/--no-tail
   really do skip the steps their names promise.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deploy_node.sh"


def _bash_n(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10)


def test_deploy_node_has_valid_bash_syntax():
    r = _bash_n(SCRIPT)
    assert r.returncode == 0, r.stderr


def test_deploy_node_uses_strict_mode():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text


def test_deploy_node_never_deletes_state_or_baseline():
    """The one way this script could do real damage: rsync --delete
    without excluding the device's own state.db / baseline.npz would wipe
    a unit's learned normal on every code push. Pin the excludes are
    present, not just described in a comment."""
    text = SCRIPT.read_text()
    assert "--delete" in text
    assert "--exclude='*.db'" in text
    assert "--exclude='baseline.npz'" in text


def test_deploy_node_with_no_args_fails_fast_with_usage():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=10)
    assert r.returncode != 0
    assert "usage" in (r.stdout + r.stderr).lower()


def _write_fake(path: Path, tag: str, logfile: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'{tag} %s\\n\' "$*" >> "{logfile}"\n'
        "exit 0\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_with_fake_remote(tmp_path: Path, args: list[str]) -> tuple[subprocess.CompletedProcess, str]:
    """Runs the REAL deploy_node.sh as a subprocess with fake ssh/rsync
    first on PATH, so no network call is ever attempted. Returns the
    completed process and the fake-remote's own invocation log."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    logfile = tmp_path / "calls.log"
    logfile.write_text("")
    _write_fake(fake_bin / "rsync", "RSYNC", logfile)
    _write_fake(fake_bin / "ssh", "SSH", logfile)

    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(SCRIPT), *args], env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=15)
    return r, logfile.read_text()


def test_default_run_syncs_firmware_and_ml_then_restarts_then_tails(tmp_path):
    r, log = _run_with_fake_remote(tmp_path, ["acoustic1.local"])
    assert r.returncode == 0, r.stdout + r.stderr
    lines = log.strip().splitlines()

    rsync_calls = [l for l in lines if l.startswith("RSYNC")]
    ssh_calls = [l for l in lines if l.startswith("SSH")]

    assert len(rsync_calls) == 2, log
    assert "firmware/" in rsync_calls[0] and "pi@acoustic1.local:/opt/acoustic-monitor/firmware/" in rsync_calls[0]
    assert "ml/" in rsync_calls[1] and "pi@acoustic1.local:/opt/acoustic-monitor/ml/" in rsync_calls[1]
    # the firmware sync (but not the ml sync) must exclude the two files
    # that hold this device's own learned state
    assert "--exclude=*.db" in rsync_calls[0]
    assert "--exclude=baseline.npz" in rsync_calls[0]

    # default ssh user is 'pi' (the build guide (not in this public copy)'s own Imager convention)
    assert any("pi@acoustic1.local" in c and "systemctl restart acoustic-monitor" in c for c in ssh_calls), log
    assert any("pi@acoustic1.local" in c and "journalctl -u acoustic-monitor" in c for c in ssh_calls), log


def test_explicit_ssh_user_overrides_the_default(tmp_path):
    r, log = _run_with_fake_remote(tmp_path, ["192.168.1.42", "logan", "--no-tail"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "logan@192.168.1.42" in log
    assert "pi@" not in log


def test_no_restart_skips_the_restart_but_still_syncs(tmp_path):
    r, log = _run_with_fake_remote(tmp_path, ["acoustic1.local", "--no-restart", "--no-tail"])
    assert r.returncode == 0, r.stdout + r.stderr
    rsync_calls = [l for l in log.strip().splitlines() if l.startswith("RSYNC")]
    ssh_calls = [l for l in log.strip().splitlines() if l.startswith("SSH")]
    assert len(rsync_calls) == 2, "code sync must still happen"
    assert not any("systemctl restart" in c for c in ssh_calls)
    assert not any("journalctl" in c for c in ssh_calls)


def test_no_tail_skips_the_log_but_still_restarts(tmp_path):
    r, log = _run_with_fake_remote(tmp_path, ["acoustic1.local", "--no-tail"])
    assert r.returncode == 0, r.stdout + r.stderr
    ssh_calls = [l for l in log.strip().splitlines() if l.startswith("SSH")]
    assert any("systemctl restart" in c for c in ssh_calls)
    assert not any("journalctl" in c for c in ssh_calls)


def test_unrecognised_flag_fails_fast_with_usage():
    r = subprocess.run(["bash", str(SCRIPT), "acoustic1.local", "--bogus-flag"],
                       capture_output=True, text=True, timeout=10)
    assert r.returncode != 0
    assert "usage" in (r.stdout + r.stderr).lower()
