#!/usr/bin/env bash
# check_provision.sh — verify a system scripts/provision_pi.sh (T2.1) has
# already run on. Run it on the Pi after provisioning, and after every
# reboot (a missed reboot is the single most common reason config.txt
# changes silently don't take effect — see the build guide (not in this public copy) §4.4).
#
#   bash scripts/check_provision.sh
#
# CANNOT BE RUN IN THIS PROJECT'S DEVELOPMENT SANDBOX, for the same reason
# provision_pi.sh can't: no real Pi, no /boot/firmware/config.txt, no
# monitor system user, no /dev/spidev0.0. Verified with `bash -n` and a
# read-through against provision_pi.sh's own steps, not executed against a
# real provisioned system. Every check below prints PASS/FAIL/WARN and
# WHY, and the script's own exit code is 1 if anything failed — safe to
# use in a boot-time health check, not just interactively.

set -uo pipefail        # NOT -e: a single failed check must not abort the rest

FAILED=0
WARNED=0

pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAILED=1; }
warn() { echo "  WARN  $1"; WARNED=1; }

check() {
  # check "description" "test command"
  local desc="$1" cmd="$2"
  if eval "$cmd" >/dev/null 2>&1; then pass "$desc"; else fail "$desc"; fi
}

echo "== system user =="
check "monitor user exists" "id monitor"
check "monitor is in the spi group (IIS3DWB access)" "id -nG monitor | grep -qw spi"
check "monitor is in the audio group (ALSA capture access)" "id -nG monitor | grep -qw audio"

echo "== directories and ownership =="
for d in /opt/acoustic-monitor /var/lib/acoustic-monitor /etc/acoustic-monitor; do
  if [ -d "$d" ]; then
    owner="$(stat -c '%U' "$d" 2>/dev/null || echo '?')"
    if [ "$owner" = "monitor" ]; then
      pass "$d exists, owned by monitor"
    else
      fail "$d exists but is owned by '$owner', not monitor"
    fi
  else
    fail "$d does not exist"
  fi
done

echo "== code + venv =="
check "firmware/main.py deployed" "[ -f /opt/acoustic-monitor/firmware/main.py ]"
check "venv exists" "[ -x /opt/acoustic-monitor/venv/bin/python3 ]"
check "numpy importable in the venv" \
  "/opt/acoustic-monitor/venv/bin/python3 -c 'import numpy'"
check "scipy importable in the venv" \
  "/opt/acoustic-monitor/venv/bin/python3 -c 'import scipy'"
check "sklearn importable in the venv" \
  "/opt/acoustic-monitor/venv/bin/python3 -c 'import sklearn'"

echo "== config =="
if [ -f /etc/acoustic-monitor/config.yaml ]; then
  pass "config.yaml present"
  if grep -q 'id: "dev-0001"' /etc/acoustic-monitor/config.yaml 2>/dev/null; then
    warn "config.yaml still has the TEMPLATE device.id (dev-0001) — edit it before deploying"
  fi
else
  fail "config.yaml missing at /etc/acoustic-monitor/config.yaml"
fi

echo "== systemd =="
check "acoustic-monitor.service installed" "[ -f /etc/systemd/system/acoustic-monitor.service ]"
check "unit ExecStart points at the venv (not the bare system python3)" \
  "grep -q '/opt/acoustic-monitor/venv/bin/python3' /etc/systemd/system/acoustic-monitor.service"
check "acoustic-monitor.service is enabled" "systemctl is-enabled --quiet acoustic-monitor"
if systemctl is-active --quiet acoustic-monitor 2>/dev/null; then
  pass "acoustic-monitor.service is running"
else
  warn "acoustic-monitor.service is not running — expected before the learn " \
       "period (the operations runbook (not in this public copy) §1) has produced a baseline.npz"
fi
check "journald size cap installed" \
  "[ -f /etc/systemd/journald.conf.d/acoustic-monitor.conf ]"

echo "== hardware interfaces (from config.txt / kernel, not the sensors themselves) =="
if [ -f /boot/firmware/config.txt ]; then
  check "dtparam=spi=on in config.txt" "grep -q '^dtparam=spi=on' /boot/firmware/config.txt"
  check "I2S mic overlay in config.txt" \
    "grep -q '^dtoverlay=googlevoicehat-soundcard' /boot/firmware/config.txt"
else
  warn "/boot/firmware/config.txt not found (older Raspberry Pi OS uses " \
       "/boot/config.txt — see the build guide (not in this public copy) §4.2)"
fi
check "/dev/spidev0.0 exists (SPI enabled AND a reboot has happened)" \
  "[ -e /dev/spidev0.0 ]"
if command -v arecord >/dev/null 2>&1; then
  check "an ALSA capture device is visible" "arecord -l 2>&1 | grep -qi card"
else
  warn "arecord not found — cannot check for an ALSA capture device"
fi

echo
echo "This checks the SYSTEM is provisioned correctly, not that the sensors" \
     "themselves work — run firmware/bench/selftest.py for that " \
     "(the handover notes (not in this public copy)), and WHO_AM_I / gravity checks before trusting any" \
     "reading."
if [ "$FAILED" -eq 1 ]; then
  echo "RESULT: at least one check FAILED — see above."
  exit 1
elif [ "$WARNED" -eq 1 ]; then
  echo "RESULT: all checks passed, with warnings — see above."
  exit 0
else
  echo "RESULT: all checks passed."
  exit 0
fi
