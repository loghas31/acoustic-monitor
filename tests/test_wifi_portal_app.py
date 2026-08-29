"""
tests/test_wifi_portal_app.py — backlog T5.1, `tools/wifi_portal_app.py`.

Driven the same way `tests/test_api.py` drives the cloud backend: FastAPI's
`TestClient` over the in-process ASGI app, no real HTTP socket, no real
`nmcli`. `wifi_provision.connect_to_network`/`load_scan_cache` are
monkeypatched at the module level the portal app imported them from — the
portal's own logic (what HTML it renders, which form field maps to which
call, what a failed connect shows the customer) is what's under test here,
not NetworkManager, which `tests/test_wifi_provision.py` already covers
with a fake `nmcli` on PATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "tools"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient  # noqa: E402

import wifi_portal_app as portal  # noqa: E402
import wifi_provision as wp  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the portal at a throwaway cache path, same "don't touch the
    # real filesystem" convention tests/test_severity_persistence.py uses
    # for firmware/state.py's own DB path.
    cache = tmp_path / "wifi_scan_cache.json"
    monkeypatch.setattr(portal, "CACHE_PATH", cache)
    return TestClient(portal.app), cache


def test_root_page_lists_cached_networks_by_signal_strength(client):
    c, cache = client
    wp.save_scan_cache(cache, [
        {"ssid": "Weak", "signal": 20, "security": "WPA2"},
        {"ssid": "Strong", "signal": 90, "security": ""},
    ])
    r = c.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Strong" in body and "Weak" in body
    # strongest signal listed first
    assert body.index("Strong") < body.index("Weak")
    assert "open" in body            # Strong has no security -> "open"
    assert "secured" in body         # Weak has WPA2 -> "secured"


def test_root_page_with_empty_cache_explains_instead_of_showing_nothing(client):
    c, _cache = client
    r = c.get("/")
    assert r.status_code == 200
    assert "No networks in range yet" in r.text


def test_connect_success_shows_confirmation(client, monkeypatch):
    c, _cache = client
    monkeypatch.setattr(wp, "connect_to_network", lambda ssid, pw, iface: True)
    r = c.post("/connect", data={"ssid": "HomeNet", "password": "hunter2"})
    assert r.status_code == 200
    assert "Connected to HomeNet" in r.text


def test_connect_failure_shows_retry_message_not_an_error_page(client, monkeypatch):
    c, _cache = client
    monkeypatch.setattr(wp, "connect_to_network", lambda ssid, pw, iface: False)
    r = c.post("/connect", data={"ssid": "HomeNet", "password": "wrong"})
    assert r.status_code == 200               # not a 4xx/5xx — customer stays on the page
    assert "Could not connect to HomeNet" in r.text


def test_connect_passes_blank_password_as_none_for_open_networks(client, monkeypatch):
    c, _cache = client
    seen = {}

    def fake_connect(ssid, password, iface):
        seen["ssid"], seen["password"], seen["iface"] = ssid, password, iface
        return True

    monkeypatch.setattr(wp, "connect_to_network", fake_connect)
    r = c.post("/connect", data={"ssid": "OpenCafe", "password": ""})
    assert r.status_code == 200
    assert seen == {"ssid": "OpenCafe", "password": None, "iface": portal.IFACE}


def test_connect_requires_an_ssid(client):
    c, _cache = client
    r = c.post("/connect", data={"password": "x"})
    assert r.status_code == 422                # FastAPI's own validation, ssid is required


@pytest.mark.parametrize("path", [
    "/hotspot-detect.html", "/generate_204", "/connecttest.txt"])
def test_os_captive_portal_probes_redirect_to_the_portal(client, path):
    c, _cache = client
    r = c.get(path, follow_redirects=False)
    assert r.status_code in (302, 303, 307)
    assert r.headers["location"] == "/"
