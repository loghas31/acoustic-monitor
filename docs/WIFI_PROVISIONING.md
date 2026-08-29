# Headless Wi-Fi provisioning (T5.1)

Companion code: `tools/wifi_provision.py` (NetworkManager control logic +
CLI), `tools/wifi_portal_app.py` (the captive-portal web page), tests in
`tests/test_wifi_provision.py` and `tests/test_wifi_portal_app.py`.

## The gap this closes

`README.md`'s customer-setup section promises:

> 2. Scan the QR card → Wi-Fi setup → done with hardware.

That step does not exist today. the execution plan (not in this public copy) already names it
plainly — Phase C, weeks 9–12: "Enclosure + zero-touch Wi-Fi onboarding —
**the 15-minute install promise is currently fiction**." Right now a node's
Wi-Fi credentials have to be written into the SD image at flash time
(the build guide (not in this public copy) §3, via Raspberry Pi Imager's own customisation
screen). That works for Logan's own bench units, where he knows the bench
Wi-Fi password before he images anything. It does not work for a customer:
you cannot ask someone to hand you their home or shop Wi-Fi password before
you can ship them a working sensor, and re-imaging a card in the field is
not a "15-minute install."

**Scope of this task, and what "implement if tractable" turned out to
mean.** The task text asked for a design plus an implementation if
tractable. The control logic and the captive-portal page are both fully
implemented and tested (unit + subprocess-level, see below) — that part
needed no hardware, only NetworkManager's documented CLI contract. The
parts that genuinely need a radio, root, and a real interface (DNS
hijacking so a phone's captive-portal probe actually reaches this device,
the systemd wiring, `iptables`) are designed and written up but **not**
implemented as runnable artefacts, because there is nothing in this sandbox
that could execute or test them honestly. That boundary is drawn
explicitly in each section below, not left implicit.

## Why NetworkManager, not `wpa_supplicant`/`dhcpcd` directly

This was already decided, not by this task — the build guide (not in this public copy) §3
established that current Raspberry Pi OS (Bookworm and the current
release, Trixie/Debian 13) uses NetworkManager as the active network
manager, and that the classic "drop a `wpa_supplicant.conf` on the boot
partition" trick **no longer works** for that reason; Imager's own
Wi-Fi/hostname customisation screen writes an NM connection profile
instead. Any provisioning approach for this project has to speak to
NetworkManager or fight it. `nmcli` (NM's own CLI, always present
alongside NM) is the natural interface — no extra system package beyond
what the build guide (not in this public copy) already requires.

## Options considered

| Option | What it is | Why not chosen |
|---|---|---|
| **`balena wifi-connect`** | A compiled Rust binary + captive portal, the most widely used tool for exactly this problem. Talks to NetworkManager over D-Bus, tested on Raspberry Pi 3's onboard Wi-Fi. | Ships as a prebuilt binary/Docker image aimed at balenaOS; the plain-Debian install path is a `raspbian-install.sh` script last aimed at Stretch. A single physics undergraduate maintaining this alone is better served by ~250 lines of Python he can read end to end than by vendoring an unfamiliar Rust binary's failure modes. Also: its own README lists the **official Raspberry Pi Wi-Fi dongle (BCM43143) as known not to work** with NM/AP mode — a reminder to verify AP mode on the Zero 2 W's *onboard* chip specifically at bring-up (H2), not assume it from a name-brand tool's general reputation. |
| **`comitup`** | A well-established, actively-used Pi captive-portal project (D-Bus + NetworkManager, similar goal). | Current research (Aug 2026) turned up **at least one independent project reporting comitup, balena wifi-connect, and RaspAP all hitting compatibility problems with newer Raspberry Pi OS's `netplan` + NetworkManager combination** on Trixie. This project already targets current Raspberry Pi OS (the build guide (not in this public copy) explicitly tracks Trixie); inheriting a third-party tool's unresolved compatibility gap on the exact OS release this project ships on is a worse position than writing ~250 lines against `nmcli` directly, which is the same interface Imager itself uses. |
| **`RaspAP`** | A popular web-based Pi AP/router management UI. | Built for "turn a Pi into a permanent router/AP", not "briefly provision then get out of the way" — wrong shape for this product (the AP must disappear once real Wi-Fi is joined; RaspAP's default posture is the opposite). Same Trixie-compatibility caveat as comitup, above. |
| **DIY: `nmcli` + a small FastAPI portal (chosen)** | `nmcli device wifi hotspot` for the AP (NM's own `shared` IPv4 method runs DHCP for AP clients — no separate `dnsmasq` service needed for that part), a small FastAPI app for the portal page (FastAPI/uvicorn are already hard dependencies of this repo — one Python web framework in the whole project, not two). | Smallest dependency footprint, fully inside what one person can read, debug, and fix on a Pi with a terminal and no internet. The tradeoff, stated honestly: this project owns the maintenance burden RaspAP/comitup/wifi-connect otherwise carry for you — acceptable here because the resulting code is ~250 lines, not thousands. |

**Evidence for the Trixie-compatibility concern and the single-radio
scanning limit**, both load-bearing for the design below, came from a
recent (2026), narrowly-scoped independent project building the identical
stack this repo targets — Raspberry Pi (down to "also works with... Pi
Zero 2 W" explicitly) + NetworkManager + a captive portal, written because
existing tools didn't fit. It corroborates two things from a different
codebase, the same kind of independent-corroboration evidence
`docs/DOC_SELF_REVIEW.md` treats as more trustworthy than a single source:
(1) the Trixie/netplan+NM compatibility gap in the established tools, and
(2) that a Pi's onboard Wi-Fi radio cannot scan for networks while running
as an access point — one radio, one job. That project's own name for
finding this out the hard way: "mass emotional damage during development."
Source, and the balena wifi-connect README's dongle-compatibility table,
are both worth reading before touching real hardware at H2:
- <https://github.com/imadash/headless-wifi-provision-pi-trixie>
- <https://github.com/balena-os/wifi-connect/blob/master/README.md>

## How it works

```
Node has Wi-Fi?
   │
   ├── YES → cache a fresh scan every 30 s (tools/wifi_provision.py `daemon`
   │         mode → tick()), so there is always a recent network list on
   │         disk even though the device usually isn't near a screen.
   │
   └── NO  → bring up an access point (nmcli device wifi hotspot),
             broadcasting a fixed SSID printed on a card shipped with the
             unit, as a WIFI: QR code (§QR card below).
             │
             Customer's phone joins the AP → OS captive-portal probe
             (ideally) auto-opens tools/wifi_portal_app.py's `/` page →
             shows the CACHED network list (never a live scan — the radio
             is busy being the AP) → customer picks their network, enters
             its password → POST /connect → nmcli device wifi connect.
             │
             Success → next `daemon` tick sees "connected", tears the AP
             down. Failure → portal shows "could not connect, try again",
             AP stays up.
```

This is exactly `tools/wifi_provision.py`'s `tick()` function, and it is
unit-tested for all four connected/disconnected × AP-up/AP-down
transitions in `tests/test_wifi_provision.py` — including the specific
assertion that the disconnected branch never calls `scan_networks`, which
is the single-radio constraint enforced in code, not just described in
prose.

## The QR card

the original brief (not in this public copy)/`README.md` call it "scan the QR card" already — the
mechanism is the standard `WIFI:` URI scheme every modern phone camera app
recognises without an extra download:

```
WIFI:S:AcousticMonitor-<serial>;T:WPA;P:<per-unit setup password>;;
```

Two design choices worth stating plainly, because they were not obvious on
first read of the balena wifi-connect precedent (which defaults to a
single shared SSID/open network for every unit):

1. **The SSID should be per-unit** (`AcousticMonitor-<serial>`, matching
   whatever the device's own ID/serial scheme ends up being at H5), not one
   shared name — if two units are being set up in the same building (a
   plausible pilot scenario, §the customer research (not in this public copy)), a shared SSID means
   the customer's phone can't tell them apart or might join the wrong one.
2. **The setup AP should require a password**, not be open — `nmcli device
   wifi hotspot` supports this directly (`start_ap`'s `password` argument);
   an open AP means anyone within range during the setup window (which
   could be minutes, if the customer doesn't act on it immediately) could
   connect and either see the portal or, worse, feed it credentials for a
   network they don't own. WPA2 needs ≥ 8 characters; the QR generator
   (not yet written — see Not built, below) is responsible for producing a
   compliant per-unit password, not `wifi_provision.py`, which only passes
   whatever password it's given straight to `nmcli`.

## What's implemented and tested

- `tools/wifi_provision.py`: `is_connected`, `scan_networks`, `start_ap`,
  `stop_ap`, `connect_to_network` — each a thin `nmcli` wrapper, verified
  the same way `scripts/deploy_node.sh` (T2.2) verified rsync/ssh: a fake
  `nmcli` executable placed first on PATH, logging its own argv and
  returning canned `-t`-format output, with the REAL functions run as a
  real subprocess against it. `_parse_terse` (nmcli's `\:`-escaped
  colon-separated format) is tested directly against realistic sample
  lines, including an SSID containing a literal colon. `tick()` (the
  one-iteration decision function) and `run_daemon()` (the `--once`/
  `--max-iterations`-testable loop around it) are tested with plain fake
  callables, no subprocess. `save_scan_cache`'s "never overwrite a good
  cache with an empty scan" property — the single-radio problem's direct
  consequence — is pinned as its own test.
- `tools/wifi_portal_app.py`: a FastAPI app (reusing this repo's existing
  FastAPI/uvicorn dependency rather than adding Flask) serving the network
  list + credentials form, plus the three OS captive-portal detection
  probes (`/hotspot-detect.html` for Apple, `/generate_204` for
  Android/ChromeOS, `/connecttest.txt` for Windows NCSI), each redirecting
  to `/`. Tested with FastAPI's `TestClient`, the same in-process ASGI
  approach `tests/test_api.py` already uses for the cloud backend —
  success/failure connect flows, empty-cache messaging, and that a blank
  password form field is passed through as `None` (open network) rather
  than the literal empty string.

## What's designed but NOT built, and why

**DNS hijacking on the AP interface.** The OS captive-portal probes
(`/generate_204` etc.) only trigger the auto-popup if the hostnames those
probes target (`connectivitycheck.gstatic.com`, `captive.apple.com`,
`www.msftconnecttest.com`) actually resolve to this device's own IP while
the AP is up — normally they'd fail to resolve at all (no real internet
behind this AP) or resolve to the real internet's servers if some upstream
DNS leaks through, and most phones interpret "can't reach it" as "maybe
there's a portal, maybe there's no internet" inconsistently across
vendors, which is a materially worse experience than a clean redirect.
Making every DNS query on the AP interface answer with this device's own
address needs a DNS server on that interface (`dnsmasq` is the standard
choice here, run only while the AP is active) plus, ideally, `iptables`
NAT rules to also catch clients that hardcode a resolver. **Not
implemented**: this needs a real wireless interface in AP mode and root to
test meaningfully — a unit test asserting "this iptables rule string
contains DROP" tests that a string was typed correctly, not that the
network behaves as claimed, and this repo's whole discipline (`docs/
DOC_SELF_REVIEW.md`) is not to write that kind of test and call it
verification. Do this at H2 or H5, against the real interface, following
the imadash project's own worked example (`captive-iptables.sh`) as a
starting point, not a copy — reread it against whatever's actually
happening on the real device rather than trusting it blind.

**systemd wiring.** `tools/wifi_provision.py daemon` is a foreground
process by design (mirrors `firmware/main.py`'s own loop shape) and needs
a unit file (`ExecStart=... daemon --ap-ssid ... --ap-password ...`,
`Restart=on-failure`) to run unattended, analogous to `firmware/
acoustic-monitor.service` which `scripts/provision_pi.sh` already
installs. **Not written yet, deliberately** — `provision_pi.sh` is not
touched by this task. Wiring a new unit into it risks the exact kind of
scope creep this project's rules warn against (the task backlog (not in this public copy) rule 1:
"one task per run"), and `tests/test_provision_scripts.py` pins specific
strings inside that script that a careless edit could break for no reason
connected to Wi-Fi provisioning. Do this as its own small task once H2
hardware exists to test the resulting unit against.

**AP-mode compatibility of the Zero 2 W's onboard chip, specifically.**
The balena wifi-connect precedent's own dongle table is a warning, not a
guarantee: it lists the *official Raspberry Pi USB dongle* (a different,
older chip) as broken in AP mode while onboard Pi 3 Wi-Fi worked. The Zero
2 W's onboard chip (Cypress/Infineon CYW43436, the same family used across
several Pi boards) is *expected* to support pure AP mode based on that
family's general track record (many "Pi Zero W as an access point"
tutorials use it with `hostapd` directly), but this project's own
standing rule applies here as everywhere else: **that is a prediction,
not a measurement, until H2 runs `nmcli device wifi hotspot` on the real
board and it either works or it doesn't.**

## Honest status

Everything under "What's implemented and tested" above was executed this
run — real subprocess calls against a fake `nmcli`, real FastAPI
`TestClient` requests — and the numbers are in the Run log entry for T5.1
in the task backlog (not in this public copy). Nothing about whether a real Pi Zero 2 W, a real
NetworkManager, and a real customer's thumb actually produces "scan the QR
card → done" in under a minute has been tested, because none of those
three things exist in this sandbox. This is designed to be the fast path
once H2 hardware exists: `tools/wifi_provision.py status` /
`tools/wifi_provision.py start-ap --ssid test --password test1234` are
both one-line manual checks against the real board, no new code needed to
try them.
