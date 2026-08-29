"""
tests/test_wifi_provision.py — backlog T5.1, `tools/wifi_provision.py`.

Same evidence split `tests/test_deploy_node.py` used for T2.2, because the
constraint is identical: no real target to run against (there, no ssh host;
here, no wifi radio and no `nmcli` binary at all — checked directly,
`which nmcli` fails in this sandbox).

1. `_parse_terse` — nmcli's `-t` colon-escaping, tested directly against
   real sample output copied from `man nmcli`'s own terse-format examples,
   no subprocess involved.
2. `is_connected` / `scan_networks` / `start_ap` / `stop_ap` /
   `connect_to_network` — a FAKE `nmcli` placed first on PATH, logging its
   own argv and returning canned `-t` output, so the REAL functions run as
   a real subprocess against it. Proves the exact command lines this module
   sends to NetworkManager.
3. `tick` — the monitor-loop's one decision function, tested with plain
   fake callables (no subprocess at all): connected/disconnected x
   AP-was-up/AP-was-down, all four transitions.
4. `save_scan_cache` — the "never overwrite a good cache with an empty
   scan" property from the single-radio-problem docstring, pinned directly.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "tools"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wifi_provision as wp  # noqa: E402


# --------------------------------------------------------------------------
# 1. Terse-format parsing, no subprocess
# --------------------------------------------------------------------------

def test_parse_terse_plain_fields():
    assert wp._parse_terse("wlan0:connected", 2) == ["wlan0", "connected"]


def test_parse_terse_handles_escaped_colon_in_ssid():
    # A real, legal SSID containing a colon; nmcli -t escapes ONLY the
    # colon (and a literal backslash) — not the apostrophe, which needs no
    # escaping in this format.
    line = r"Dave's Cafe\: Guest:60:WPA2"
    ssid, signal, security = wp._parse_terse(line, 3)
    assert ssid == "Dave's Cafe: Guest"
    assert signal == "60"
    assert security == "WPA2"


def test_parse_terse_pads_missing_trailing_fields():
    # An open network: nmcli sometimes emits a trailing empty SECURITY
    # column, sometimes omits it outright depending on version — both must
    # come out the same shape.
    assert wp._parse_terse("OpenCafe:45:", 3) == ["OpenCafe", "45", ""]
    assert wp._parse_terse("OpenCafe:45", 3) == ["OpenCafe", "45", ""]


# --------------------------------------------------------------------------
# 2. Real subprocess against a fake nmcli on PATH
# --------------------------------------------------------------------------

_FAKE_NMCLI = r'''#!/usr/bin/env python3
import sys, os

logfile = os.environ["FAKE_NMCLI_LOG"]
args = sys.argv[1:]
with open(logfile, "a") as f:
    f.write(" ".join(args) + "\n")

def out(s=""):
    sys.stdout.write(s)

if args[:4] == ["-t", "-f", "DEVICE,STATE", "device"] and args[4:] == ["status"]:
    out("wlan0:connected\neth0:unmanaged\n")
elif args[:3] == ["-t", "-f", "SSID,SIGNAL,SECURITY"] and "wifi" in args:
    out("HomeNet:80:WPA2\nOpenCafe:45:\n")
elif "hotspot" in args:
    pass  # logged above; nothing to print
elif args[:2] == ["connection", "down"]:
    pass
elif args[:3] == ["device", "wifi", "connect"]:
    # simulate a wrong-password failure for a magic SSID name
    if "WRONGPASS" in args:
        sys.exit(1)
else:
    pass
sys.exit(0)
'''


def _install_fake_nmcli(tmp_path: Path) -> tuple[dict, Path]:
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    script = fake_bin / "nmcli"
    script.write_text(_FAKE_NMCLI)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logfile = tmp_path / "nmcli_calls.log"
    logfile.write_text("")
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
           "FAKE_NMCLI_LOG": str(logfile)}
    return env, logfile


def _run_module_fn_in_subprocess(env: dict, code: str) -> subprocess.CompletedProcess:
    """Runs a snippet importing the REAL wifi_provision module with the
    fake nmcli on PATH, so subprocess.run(["nmcli", ...]) inside the real
    functions hits the fake — this is what actually proves the module
    shells out correctly, not just that it would in theory."""
    full = (f"import sys; sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
            "import wifi_provision as wp\n" + code)
    return subprocess.run([sys.executable, "-c", full], env=env,
                          capture_output=True, text=True, timeout=15)


def test_is_connected_true_for_a_connected_device(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'print(wp.is_connected("wlan0"))')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"
    assert "device status" in log.read_text()


def test_is_connected_false_for_unmanaged_device(tmp_path):
    env, _ = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'print(wp.is_connected("eth0"))')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_is_connected_false_for_unknown_device(tmp_path):
    env, _ = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'print(wp.is_connected("wlan9"))')
    assert r.stdout.strip() == "False"


def test_scan_networks_parses_fake_output(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'import json; print(json.dumps(wp.scan_networks("wlan0")))')
    assert r.returncode == 0, r.stderr
    nets = json.loads(r.stdout)
    assert {"ssid": "HomeNet", "signal": 80, "security": "WPA2"} in nets
    assert {"ssid": "OpenCafe", "signal": 45, "security": ""} in nets
    assert "device" in log.read_text() and "wifi" in log.read_text()


def test_start_ap_sends_expected_nmcli_command(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'wp.start_ap("MySetup", "setup1234", "wlan0")')
    assert r.returncode == 0, r.stderr
    call = log.read_text().strip()
    assert "hotspot" in call
    assert "ssid MySetup" in call
    assert "password setup1234" in call
    assert "ifname wlan0" in call


def test_start_ap_without_password_omits_it(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'wp.start_ap("OpenSetup", None, "wlan0")')
    assert r.returncode == 0, r.stderr
    assert "password" not in log.read_text()


def test_stop_ap_calls_connection_down(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(env, 'wp.stop_ap()')
    assert r.returncode == 0, r.stderr
    assert "connection down wifi-provision-ap" in log.read_text()


def test_connect_to_network_success(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'print(wp.connect_to_network("HomeNet", "hunter2", "wlan0"))')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"
    assert "device wifi connect HomeNet ifname wlan0 password hunter2" in log.read_text()


def test_connect_to_network_wrong_password_returns_false_not_raise(tmp_path):
    """The most common real-world path through this function: a customer
    mistypes their own Wi-Fi password. Must return False for the portal to
    show "try again", not raise and 500 the page."""
    env, _ = _install_fake_nmcli(tmp_path)
    r = _run_module_fn_in_subprocess(
        env, 'print(wp.connect_to_network("WRONGPASS", "bad", "wlan0"))')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


# --------------------------------------------------------------------------
# 3. tick() — pure state-transition logic, no subprocess
# --------------------------------------------------------------------------

def _fake_deps(connected, nets, calls):
    def is_connected_fn(iface):
        calls.append(("is_connected", iface))
        return connected

    def scan_fn(iface):
        calls.append(("scan", iface))
        return nets

    def start_ap_fn(ssid, password, iface):
        calls.append(("start_ap", ssid, password, iface))

    def stop_ap_fn():
        calls.append(("stop_ap",))

    return is_connected_fn, scan_fn, start_ap_fn, stop_ap_fn


def test_tick_connected_and_ap_was_down_just_caches_scan(tmp_path):
    calls = []
    is_c, scan, start, stop = _fake_deps(
        True, [{"ssid": "X", "signal": 1, "security": ""}], calls)
    cache = tmp_path / "cache.json"
    new_state = wp.tick("wlan0", cache, "Setup", "pw", ap_active=False,
                        is_connected_fn=is_c, scan_fn=scan,
                        start_ap_fn=start, stop_ap_fn=stop)
    assert new_state is False
    assert ("stop_ap",) not in calls
    assert json.loads(cache.read_text()) == [{"ssid": "X", "signal": 1, "security": ""}]


def test_tick_connected_and_ap_was_up_tears_it_down(tmp_path):
    calls = []
    is_c, scan, start, stop = _fake_deps(True, [], calls)
    cache = tmp_path / "cache.json"
    new_state = wp.tick("wlan0", cache, "Setup", "pw", ap_active=True,
                        is_connected_fn=is_c, scan_fn=scan,
                        start_ap_fn=start, stop_ap_fn=stop)
    assert new_state is False
    assert ("stop_ap",) in calls


def test_tick_disconnected_and_ap_was_down_starts_it(tmp_path):
    calls = []
    is_c, scan, start, stop = _fake_deps(False, [], calls)
    cache = tmp_path / "cache.json"
    new_state = wp.tick("wlan0", cache, "Setup", "pw", ap_active=False,
                        is_connected_fn=is_c, scan_fn=scan,
                        start_ap_fn=start, stop_ap_fn=stop)
    assert new_state is True
    assert ("start_ap", "Setup", "pw", "wlan0") in calls
    assert not any(c[0] == "scan" for c in calls), (
        "must NOT scan while disconnected — see single-radio-problem docstring")


def test_tick_disconnected_and_ap_already_up_is_a_noop(tmp_path):
    calls = []
    is_c, scan, start, stop = _fake_deps(False, [], calls)
    cache = tmp_path / "cache.json"
    new_state = wp.tick("wlan0", cache, "Setup", "pw", ap_active=True,
                        is_connected_fn=is_c, scan_fn=scan,
                        start_ap_fn=start, stop_ap_fn=stop)
    assert new_state is True
    assert not any(c[0] == "start_ap" for c in calls), "already up, don't restart it"


# --------------------------------------------------------------------------
# 4. save_scan_cache — never overwrite good data with an empty scan
# --------------------------------------------------------------------------

def test_save_scan_cache_writes_nonempty(tmp_path):
    cache = tmp_path / "c.json"
    wrote = wp.save_scan_cache(cache, [{"ssid": "A", "signal": 1, "security": ""}])
    assert wrote is True
    assert wp.load_scan_cache(cache) == [{"ssid": "A", "signal": 1, "security": ""}]


def test_save_scan_cache_refuses_to_blank_existing_data(tmp_path):
    cache = tmp_path / "c.json"
    wp.save_scan_cache(cache, [{"ssid": "A", "signal": 1, "security": ""}])
    wrote = wp.save_scan_cache(cache, [])
    assert wrote is False, "an empty scan must not destroy the last good cache"
    assert wp.load_scan_cache(cache) == [{"ssid": "A", "signal": 1, "security": ""}]


def test_save_scan_cache_allows_first_write_to_be_empty(tmp_path):
    cache = tmp_path / "c.json"
    wrote = wp.save_scan_cache(cache, [])
    assert wrote is True
    assert wp.load_scan_cache(cache) == []


def test_load_scan_cache_missing_file_returns_empty_list(tmp_path):
    assert wp.load_scan_cache(tmp_path / "does_not_exist.json") == []


def test_load_scan_cache_corrupt_file_returns_empty_list_not_raise(tmp_path):
    cache = tmp_path / "c.json"
    cache.write_text("{not valid json")
    assert wp.load_scan_cache(cache) == []


# --------------------------------------------------------------------------
# 5. run_daemon — the tick loop terminates correctly under --once / max_iterations
# --------------------------------------------------------------------------

def test_run_daemon_once_calls_tick_exactly_one_time(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(wp, "is_connected", lambda iface: True)
    monkeypatch.setattr(wp, "scan_networks", lambda iface: [])
    sleeps = []
    wp.run_daemon("wlan0", tmp_path / "c.json", "Setup", "pw",
                  once=True, sleep_fn=sleeps.append)
    assert sleeps == [], "must not sleep when once=True"


def test_run_daemon_max_iterations_stops_and_sleeps_between(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, "is_connected", lambda iface: True)
    monkeypatch.setattr(wp, "scan_networks", lambda iface: [])
    sleeps = []
    wp.run_daemon("wlan0", tmp_path / "c.json", "Setup", "pw",
                  poll_interval=5.0, max_iterations=3, sleep_fn=sleeps.append)
    assert sleeps == [5.0, 5.0], "sleeps between iterations, not after the last one"


# --------------------------------------------------------------------------
# 6. CLI smoke test — argparse wiring, no real nmcli needed for --help
# --------------------------------------------------------------------------

def test_cli_help_lists_all_subcommands():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "wifi_provision.py"), "--help"],
        capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    for cmd in ("status", "scan", "start-ap", "stop-ap", "connect", "daemon"):
        assert cmd in r.stdout


def test_cli_status_uses_fake_nmcli(tmp_path):
    env, log = _install_fake_nmcli(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "wifi_provision.py"), "status"],
        env=env, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    # default --iface is wlan0, which the fake nmcli reports as connected
    assert r.stdout.strip() == "connected"
    assert "device status" in log.read_text()


def test_cli_status_respects_explicit_iface_flag(tmp_path):
    env, _ = _install_fake_nmcli(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "wifi_provision.py"),
         "--iface", "eth0", "status"],
        env=env, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "disconnected"  # fake reports eth0 as unmanaged
