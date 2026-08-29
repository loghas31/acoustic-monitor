#!/usr/bin/env python3
"""
tools/wifi_portal_app.py — backlog T5.1, the captive-portal web page.

`tools/wifi_provision.py` decides WHEN to run an access point; this is WHAT
a customer's phone sees once they've connected to it. FastAPI + uvicorn are
already a hard dependency of this repo (`backend/requirements.txt`), so this
reuses that stack rather than adding Flask or anything else — one Python web
framework in the whole project, not two.

WHAT "CAPTIVE PORTAL" MEANS HERE, PRECISELY
--------------------------------------------------------------------------
There are two separate mechanisms bundled under that name, and this file is
only the second one:

1. Getting the phone's OS to notice there's no real internet and pop the
   portal UI automatically, instead of the customer having to manually type
   an IP address into a browser. iOS/macOS probe
   `http://captive.apple.com/hotspot-detect.html` expecting an exact HTML
   body; Android probes `http://connectivitycheck.gstatic.com/generate_204`
   expecting a bare 204; Windows probes
   `http://www.msftconnecttest.com/connecttest.txt` expecting the literal
   text "Microsoft Connect Test". This app answers all three (`/hotspot-
   detect.html`, `/generate_204`, `/connecttest.txt`) with a redirect to
   `/`, so failing the OS's own check is what triggers the auto-popup — but
   ONLY IF those exact hostnames actually resolve to this device's IP
   instead of the real internet, which needs DNS on the AP interface to lie
   (every hostname -> this device). That is infrastructure this app cannot
   provide by itself: see docs/WIFI_PROVISIONING.md's "DNS/iptables" section
   for what's needed and why it is NOT implemented here (untestable without
   a real interface and root, unlike everything below).
2. Serving the actual page: a list of nearby networks (from the scan
   cache — see wifi_provision.py's single-radio-problem docstring for why
   it's a cache and not a live scan) and a form to submit credentials. THAT
   part is fully implemented and tested below with FastAPI's TestClient,
   the same tool `tests/test_api.py` already uses for the cloud backend —
   no real network, no real HTTP socket, the ASGI app driven in-process.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import wifi_provision  # noqa: E402  (see sys.path note above)

app = FastAPI(title="acoustic-monitor Wi-Fi setup")

# Overridable by tests so they don't touch the real filesystem path under
# /var/lib, mirroring how firmware/state.py's own tests parameterise the DB
# path rather than monkeypatching a module constant mid-suite.
CACHE_PATH = wifi_provision.DEFAULT_CACHE_PATH
IFACE = wifi_provision.AP_IFACE_DEFAULT


def _render_page(networks: list[dict], message: Optional[str] = None) -> str:
    rows = "".join(
        f'<label><input type="radio" name="ssid" value="{n["ssid"]}" '
        f'required> {n["ssid"]} '
        f'({"secured" if n["security"] else "open"}, signal {n["signal"]})'
        f"</label><br>"
        for n in sorted(networks, key=lambda n: -n["signal"])
    )
    if not rows:
        rows = ("<p>No networks in range yet — the list is from the last "
                "time this device had a working connection. Move it closer "
                "to your router and reconnect the phone to this setup "
                "network to refresh.</p>")
    banner = f"<p><strong>{message}</strong></p>" if message else ""
    return f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Set up your acoustic monitor</title></head>
<body>
<h1>Connect this sensor to your Wi-Fi</h1>
{banner}
<form method="post" action="/connect">
{rows}
<label>Password (leave blank for an open network):
  <input type="password" name="password"></label><br>
<button type="submit">Connect</button>
</form>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def portal_page():
    networks = wifi_provision.load_scan_cache(CACHE_PATH)
    return _render_page(networks)


@app.post("/connect", response_class=HTMLResponse)
def submit_credentials(ssid: str = Form(...), password: str = Form("")):
    ok = wifi_provision.connect_to_network(
        ssid, password or None, IFACE)
    if ok:
        return _render_page(
            wifi_provision.load_scan_cache(CACHE_PATH),
            message=f"Connected to {ssid}. This setup network will "
                    f"disappear shortly.")
    return _render_page(
        wifi_provision.load_scan_cache(CACHE_PATH),
        message=f"Could not connect to {ssid} — check the password and "
                f"try again.")


# --- OS captive-portal detection probes -----------------------------------
# See the module docstring: answering these correctly is necessary but not
# sufficient — DNS on the AP interface has to route the real hostnames here
# first, which is outside what this ASGI app controls.

@app.get("/hotspot-detect.html")   # Apple (iOS/macOS)
def apple_probe():
    return RedirectResponse("/")


@app.get("/generate_204")          # Android / ChromeOS
def android_probe():
    return RedirectResponse("/")


@app.get("/connecttest.txt")       # Windows NCSI
def windows_probe():
    return RedirectResponse("/")
