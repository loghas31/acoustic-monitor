#!/usr/bin/env python3
"""
tools/wifi_provision.py — backlog T5.1, headless Wi-Fi provisioning.

WHAT THIS IS
--------------------------------------------------------------------------
The onboarding promise in README.md is "scan the QR card, pick your
network, done with hardware" (30 s, no keyboard/monitor/SSH). Today that
step does not exist — the node either has Wi-Fi credentials baked into the
SD image at flash time (fine for Logan's own bench units, not fine for a
customer who has to give you their SSID/password before you can image
anything) or it doesn't connect at all. This file is the control logic for
closing that gap: when the node has no working Wi-Fi link, it opens its own
access point with a captive portal (`wifi_portal_app.py`) so a customer's
phone can hand it real credentials, then it joins that network and the AP
disappears. Full design rationale, the options considered, and why this
approach was chosen over `balena wifi-connect` / `comitup` / `RaspAP` are in
`docs/WIFI_PROVISIONING.md` — read that first if you're wondering "why not
just use X".

WHY nmcli AND NOT hostapd/dnsmasq DIRECTLY
--------------------------------------------------------------------------
the build guide (not in this public copy) §3 already established that current Raspberry Pi OS
(Bookworm and Trixie) ships NetworkManager as the active network manager,
not `dhcpcd`, and that the old "drop a wpa_supplicant.conf on the boot
partition" trick no longer works for that reason. NetworkManager's own
`nmcli device wifi hotspot` subcommand starts an access point with a
`shared` IPv4 method — meaning NM itself runs the DHCP server for AP
clients, no separate `dnsmasq` unit needed for that part. This keeps the
dependency list at "NetworkManager", which is already a hard requirement
of everything else in the build guide (not in this public copy), instead of adding hostapd and
a hand-rolled dnsmasq config that could disagree with what NM thinks the
interface is doing.

THE SINGLE-RADIO PROBLEM (this is real, not speculative — same finding an
independent 2026 project building the identical Pi Zero 2 W + NetworkManager
stack hit and documented; see docs/WIFI_PROVISIONING.md's Evidence section)
--------------------------------------------------------------------------
The IEEE 802.11 radio on the Pi Zero 2 W's onboard chip cannot scan for
nearby networks while it is simultaneously running as an access point — it
is one radio, doing one job at a time. So the portal cannot show "networks
near you" live once the AP is already up; the network list has to be
scanned and CACHED *before* the AP starts, while the node still has (or
last had) a working uplink. `save_scan_cache` below deliberately never
overwrites a non-empty cache with an empty one, for exactly this reason —
an empty scan while already in AP mode is not "there are no networks", it
is "we can't see any right now", and destroying the last good list would
make the portal show nothing to a customer who is standing right next to
their own router.

WHAT IS AND ISN'T VERIFIED HERE
--------------------------------------------------------------------------
CANNOT be verified end to end in this sandbox: there is no wifi radio, no
NetworkManager, no `nmcli` binary at all (checked: `which nmcli` fails).
Every function that shells out to `nmcli` is verified the same way
`scripts/deploy_node.sh` (T2.2) verified rsync/ssh — a fake `nmcli`
executable placed first on PATH that logs its own argv and returns
canned, syntactically-real `nmcli -t` output, so the REAL functions below
run as a real subprocess against it. That proves the exact command lines
this module sends to NetworkManager, not that NetworkManager behaves the
way its own docs say it does. The parsing logic (colon-escaping) is
exercised directly with real `nmcli -t -f ...` sample output, byte for
byte, taken from NetworkManager's own documented `-t` format.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

AP_IFACE_DEFAULT = "wlan0"
AP_CON_NAME = "wifi-provision-ap"
DEFAULT_CACHE_PATH = Path("/var/lib/acoustic-monitor/wifi_scan_cache.json")
DEFAULT_POLL_INTERVAL_S = 30.0


class NmcliError(RuntimeError):
    """nmcli returned nonzero. Carries stderr so callers can log it."""


def run_nmcli(args: list[str], timeout: float = 20.0) -> str:
    """Run `nmcli <args>` and return stdout. Raises NmcliError on failure.

    A thin wrapper, not a re-implementation — every other function in this
    file goes through this one, so a test faking `nmcli` on PATH exercises
    every code path that talks to NetworkManager.
    """
    proc = subprocess.run(
        ["nmcli", *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise NmcliError(
            f"nmcli {' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}")
    return proc.stdout


def _parse_terse(line: str, n_fields: int) -> list[str]:
    """Split one line of `nmcli -t` output into `n_fields` columns.

    nmcli's terse (`-t`) mode separates fields with `:` and escapes a
    literal `:` inside a field as `\\:` (documented in `man nmcli`, §Terse
    output format) — an SSID containing a colon is legal and not rare
    ("Dave's Cafe: Guest" is a real SSID pattern). A plain `str.split(":")`
    would silently misalign every column after the first colon inside such
    an SSID. This does the escape-aware split by hand rather than pulling
    in a CSV-with-custom-escape dependency for one function.
    """
    fields: list[str] = []
    current = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] in (":", "\\"):
            current.append(line[i + 1])
            i += 2
            continue
        if ch == ":":
            fields.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    fields.append("".join(current))
    # A trailing empty SECURITY field etc. is legitimate (open network) —
    # pad rather than fail if nmcli emits fewer columns than asked for.
    while len(fields) < n_fields:
        fields.append("")
    return fields[:n_fields]


def is_connected(iface: str = AP_IFACE_DEFAULT) -> bool:
    """True if `iface` currently has a working NetworkManager connection.

    Reads `nmcli -t -f DEVICE,STATE device status` rather than
    `device show <iface>` — the latter errors on an interface NetworkManager
    doesn't know about yet (e.g. right after boot, before udev renames it),
    where the status table always exists and simply won't list it.
    """
    out = run_nmcli(["-t", "-f", "DEVICE,STATE", "device", "status"])
    for line in out.splitlines():
        if not line.strip():
            continue
        device, state = _parse_terse(line, 2)
        if device == iface:
            return state.strip() == "connected"
    return False


def scan_networks(iface: str = AP_IFACE_DEFAULT) -> list[dict]:
    """Return `[{"ssid": ..., "signal": int, "security": ...}, ...]`.

    Deliberately does NOT deduplicate or sort — that's a portal-rendering
    concern (`wifi_portal_app.py`), and doing it here would make this
    function's output depend on a policy decision instead of just being
    "what nmcli said".
    """
    out = run_nmcli(
        ["-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list",
         "ifname", iface])
    networks = []
    for line in out.splitlines():
        if not line.strip():
            continue
        ssid, signal, security = _parse_terse(line, 3)
        if not ssid:
            continue                       # hidden network, nothing to show
        try:
            signal_i = int(signal)
        except ValueError:
            signal_i = 0
        networks.append(
            {"ssid": ssid, "signal": signal_i, "security": security})
    return networks


def load_scan_cache(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_scan_cache(cache_path: Path, networks: list[dict]) -> bool:
    """Write `networks` to `cache_path`. Returns whether it actually wrote.

    Never overwrites a non-empty cache with an empty scan — see the module
    docstring's "single-radio problem" section for why an empty result
    while already in AP mode must not be trusted.
    """
    if not networks and load_scan_cache(cache_path):
        return False
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(networks))
    return True


def start_ap(ssid: str, password: Optional[str] = None,
             iface: str = AP_IFACE_DEFAULT,
             con_name: str = AP_CON_NAME) -> None:
    """Bring up an access point named `con_name` broadcasting `ssid`.

    `nmcli device wifi hotspot` creates a `shared`-method connection: NM
    runs DHCP for AP clients itself. No password -> open network (NM's own
    behaviour, not something this function decides); WPA2 needs >= 8 chars,
    which the caller (the QR-card generator, `docs/WIFI_PROVISIONING.md`)
    is responsible for — this function does not second-guess the password
    NetworkManager is handed, only passes it through.
    """
    args = ["device", "wifi", "hotspot", "ifname", iface,
            "con-name", con_name, "ssid", ssid]
    if password:
        args += ["password", password]
    run_nmcli(args)


def stop_ap(con_name: str = AP_CON_NAME) -> None:
    """Tear down the hotspot connection. Safe to call if it isn't up."""
    try:
        run_nmcli(["connection", "down", con_name])
    except NmcliError:
        pass                                # already down — not an error


def connect_to_network(ssid: str, password: Optional[str] = None,
                        iface: str = AP_IFACE_DEFAULT) -> bool:
    """Try to join `ssid`. Returns True on success, False on failure.

    Does NOT raise on a wrong password / unreachable network — that is the
    single most common path through this function (a customer mistyping
    their own Wi-Fi password) and the portal needs a normal return value to
    show "that didn't work, try again", not an exception to catch.
    """
    args = ["device", "wifi", "connect", ssid, "ifname", iface]
    if password:
        args += ["password", password]
    try:
        run_nmcli(args)
        return True
    except NmcliError:
        return False


def tick(iface: str, cache_path: Path,
         ap_ssid: str, ap_password: Optional[str],
         ap_active: bool,
         is_connected_fn: Callable[[str], bool] = is_connected,
         scan_fn: Callable[[str], list[dict]] = scan_networks,
         start_ap_fn: Callable[..., None] = start_ap,
         stop_ap_fn: Callable[..., None] = stop_ap) -> bool:
    """One monitor-loop decision. Returns the new `ap_active` state.

    Pure-ish and injectable on purpose: this is the function
    `docs/WIFI_PROVISIONING.md`'s daemon loop calls every
    `DEFAULT_POLL_INTERVAL_S`, and it is also what `tests/test_wifi_provision.py`
    exercises directly with fake callables — no subprocess, no real nmcli,
    just the state-transition logic:

      connected, AP was up   -> tear the AP down, cache a fresh scan
      connected, AP was down -> just cache a fresh scan (the common case)
      not connected, AP down -> start the AP using the LAST cached scan
                                 (never scan here — see module docstring)
      not connected, AP up   -> do nothing, already provisioning
    """
    connected = is_connected_fn(iface)
    if connected:
        if ap_active:
            stop_ap_fn()
        nets = scan_fn(iface)
        if nets:
            save_scan_cache(cache_path, nets)
        return False
    if not ap_active:
        start_ap_fn(ap_ssid, ap_password, iface)
        return True
    return True


def run_daemon(iface: str, cache_path: Path, ap_ssid: str,
                ap_password: Optional[str],
                poll_interval: float = DEFAULT_POLL_INTERVAL_S,
                once: bool = False, max_iterations: Optional[int] = None,
                sleep_fn: Callable[[float], None] = time.sleep) -> None:
    """Run `tick` forever (or `max_iterations` times, or once with `once=True`).

    `once`/`max_iterations` exist purely so this function is testable
    without an infinite loop or a real 30 s sleep — production use is
    `run_daemon(...)` with neither set, from the systemd unit described in
    docs/WIFI_PROVISIONING.md.

    Passes `is_connected`/`scan_networks`/`start_ap`/`stop_ap` to `tick`
    explicitly by name rather than relying on `tick`'s own defaults: a
    Python default-argument value is bound ONCE, at `def tick(...)` time,
    to whatever object those names pointed to then — a test that
    monkeypatches `wifi_provision.is_connected` AFTER import would silently
    have no effect on `tick`'s default if `tick` were called bare. Naming
    them here means each call looks the names up fresh in this module's
    namespace, which is exactly what `tests/test_wifi_provision.py`'s
    `monkeypatch.setattr(wifi_provision, "is_connected", ...)` depends on.
    """
    ap_active = False
    iterations = 0
    while True:
        ap_active = tick(iface, cache_path, ap_ssid, ap_password, ap_active,
                         is_connected_fn=is_connected, scan_fn=scan_networks,
                         start_ap_fn=start_ap, stop_ap_fn=stop_ap)
        iterations += 1
        if once or (max_iterations is not None and iterations >= max_iterations):
            return
        sleep_fn(poll_interval)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Headless Wi-Fi provisioning control (backlog T5.1). "
                     "See docs/WIFI_PROVISIONING.md.")
    p.add_argument("--iface", default=AP_IFACE_DEFAULT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="print connected/disconnected")
    sub.add_parser("scan", help="scan now, print + cache the results")

    ap = sub.add_parser("start-ap", help="bring up the setup access point")
    ap.add_argument("--ssid", required=True)
    ap.add_argument("--password", default=None)

    sub.add_parser("stop-ap", help="tear down the setup access point")

    conn = sub.add_parser("connect", help="join a real Wi-Fi network")
    conn.add_argument("--ssid", required=True)
    conn.add_argument("--password", default=None)

    daemon = sub.add_parser(
        "daemon", help="monitor loop: AP when offline, cache scans when online")
    daemon.add_argument("--ap-ssid", required=True)
    daemon.add_argument("--ap-password", default=None)
    daemon.add_argument("--poll-interval", type=float,
                         default=DEFAULT_POLL_INTERVAL_S)
    daemon.add_argument("--once", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "status":
        print("connected" if is_connected(args.iface) else "disconnected")
    elif args.command == "scan":
        nets = scan_networks(args.iface)
        save_scan_cache(args.cache, nets)
        print(json.dumps(nets, indent=2))
    elif args.command == "start-ap":
        start_ap(args.ssid, args.password, args.iface)
    elif args.command == "stop-ap":
        stop_ap()
    elif args.command == "connect":
        ok = connect_to_network(args.ssid, args.password, args.iface)
        print("connected" if ok else "failed")
        if not ok:
            return 1
    elif args.command == "daemon":
        run_daemon(args.iface, args.cache, args.ap_ssid, args.ap_password,
                   poll_interval=args.poll_interval, once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
