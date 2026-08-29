#!/usr/bin/env bash
# provision_pi.sh — idempotent one-shot Pi setup (backlog T2.1).
#
#   sudo bash scripts/provision_pi.sh [/path/to/checked-out/acoustic-monitor]
#
# WHAT THIS DOES NOT DO: clone the repo. the build guide (not in this public copy)'s manual
# walkthrough clones under the `pi` user's home directory; a provisioning
# script has no business assuming a git URL or network access at setup
# time. Get the code onto the Pi first — `scp`, `git clone` by hand, or
# `scripts/deploy_node.sh` (T2.2) — then point this script at it. Default:
# the directory two levels up from this script (i.e. running it from an
# already-checked-out copy, same convention as `dev_up.sh`).
#
# CANNOT BE RUN IN THIS PROJECT'S DEVELOPMENT SANDBOX. It needs a real
# Raspberry Pi (raspi-config, /boot/firmware/config.txt, apt, systemd,
# a `monitor` system account) — none of which exist in an agent's Linux
# container. Verified here with `bash -n` (syntax) and read against
# the build guide (not in this public copy) §4 (the manual walkthrough this automates) line by
# line; NOT verified end to end on real hardware. Companion script
# `scripts/check_provision.sh` audits a system this script has already run
# on — run that on the Pi after provisioning, and after every reboot.
#
# EVERY STEP IS IDEMPOTENT: safe to re-run after a partial failure, an OS
# update, or to pick up a firmware/requirements.txt or config.yaml change.
# Nothing here overwrites an already-deployed /etc/acoustic-monitor/config.yaml
# (it holds this device's own id/api_key) or restarts a running service.
#
# What it does, matching the build guide (not in this public copy) §4 in order:
#   1. apt packages (§4.5): git python3-venv python3-full libportaudio2.
#   2. Enable SPI via `raspi-config nonint do_spi 0` (§4.3's automated
#      equivalent) — creates /dev/spidev0.0 for the IIS3DWB.
#   3. Add the I2S mic overlay to /boot/firmware/config.txt (§4.4) —
#      `dtoverlay=googlevoicehat-soundcard`, the verified-correct overlay
#      for the INMP441/SPH0645 on current Raspberry Pi OS (NOT
#      dtparam=i2s=on — the overlay claims the I2S pins itself; see §4.4's
#      own long note on why the hardware design notes (not in this public copy) was wrong about this).
#   4. Create the `monitor` system user + /opt/acoustic-monitor (code) and
#      /var/lib/acoustic-monitor (state.db + baseline.npz, per
#      firmware/config.yaml's storage.* paths) and /etc/acoustic-monitor
#      (config + MQTT CA cert — mode 700, it holds device secrets).
#   5. A dedicated venv at /opt/acoustic-monitor/venv (§4.5's
#      --system-site-packages recipe), `pip install -r firmware/requirements.txt`.
#      **A real bug this script's own design caught, not fixed after the
#      fact:** the shipped `firmware/acoustic-monitor.service`'s ExecStart
#      ran `/usr/bin/python3` — the SYSTEM interpreter, which has none of
#      these packages, PEP 668 blocks installing them there directly
#      (`error: externally-managed-environment`, §4.5), and systemd does
#      not source a venv's activate script. Following BUILD_GUIDE's own
#      venv recipe to the letter while installing the ORIGINAL unit file
#      unmodified would have produced a service that fails at every start
#      with `ModuleNotFoundError`. Fixed by pointing ExecStart at the
#      venv's own interpreter (see the sed line below) — recorded in
#      docs/DOC_SOAK_DB_GROWTH.md-style honesty in the task backlog (not in this public copy) T2.1,
#      not silently patched with no explanation.
#   6. Install the systemd unit (with the ExecStart fix above applied),
#      `daemon-reload`, `enable` (NOT `--now` — there is no baseline.npz
#      yet; the operations runbook (not in this public copy) §1 covers the learn period that has to
#      happen before the service can usefully start).
#   7. Cap the systemd journal's own disk use via a drop-in — T4.2's
#      database-growth audit measured this application's OWN writes as a
#      non-issue for SD wear (~483 MB/year, ~100,000+ years of headroom
#      against any real card); an UNBOUNDED systemd journal is a separate,
#      real, and much more common cause of a Pi's SD card filling up or
#      wearing out, and costs one config line to bound.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "must run as root (sudo bash scripts/provision_pi.sh ...)" >&2
  exit 1
fi

REPO_SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO_SRC/firmware/main.py" ]; then
  echo "'$REPO_SRC' does not look like an acoustic-monitor checkout " \
       "(no firmware/main.py) — pass the checked-out directory as \$1." >&2
  exit 1
fi

MONITOR_USER="monitor"
OPT_DIR="/opt/acoustic-monitor"
VAR_DIR="/var/lib/acoustic-monitor"
ETC_DIR="/etc/acoustic-monitor"
VENV_DIR="$OPT_DIR/venv"
BOOT_CONFIG="/boot/firmware/config.txt"
UNIT_SRC="$REPO_SRC/firmware/acoustic-monitor.service"
UNIT_DST="/etc/systemd/system/acoustic-monitor.service"

echo "== 1. apt packages =="
apt-get update -y
apt-get install -y git python3-venv python3-full libportaudio2

echo "== 2. enable SPI (IIS3DWB) =="
# `raspi-config nonint do_spi 0` (0=enable) is the standard scripted
# equivalent of `raspi-config` -> Interface Options -> SPI -> Yes
# (the build guide (not in this public copy) §4.3, done interactively there). UNVERIFIED THIS
# RUN — a web search to confirm current syntax against raspi-config's own
# docs was unavailable when this script was written (rate-limited). Also
# belt-and-braces handled directly in step 3 below (`dtparam=spi=on` is
# added to config.txt even if this command silently no-ops), and
# `check_provision.sh` verifies the outcome that actually matters
# (`/dev/spidev0.0` exists) rather than trusting this command succeeded.
if command -v raspi-config >/dev/null 2>&1; then
  raspi-config nonint do_spi 0
else
  echo "raspi-config not found — not a Raspberry Pi OS image? " \
       "Skipping (add 'dtparam=spi=on' to $BOOT_CONFIG by hand)." >&2
fi

echo "== 3. I2S mic overlay =="
if [ -f "$BOOT_CONFIG" ]; then
  if ! grep -q "^dtparam=spi=on" "$BOOT_CONFIG"; then
    printf '\n# --- acoustic-monitor (added by provision_pi.sh) ---\ndtparam=spi=on\n' >> "$BOOT_CONFIG"
  fi
  if ! grep -q "^dtoverlay=googlevoicehat-soundcard" "$BOOT_CONFIG"; then
    printf 'dtoverlay=googlevoicehat-soundcard\n' >> "$BOOT_CONFIG"
    echo "config.txt updated — a REBOOT is required before the mic overlay loads."
  fi
else
  echo "WARNING: $BOOT_CONFIG not found (old Raspberry Pi OS uses /boot/config.txt " \
       "instead — see the build guide (not in this public copy) §4.2). Add the overlay lines by hand." >&2
fi

echo "== 4. system user + directories =="
if ! id "$MONITOR_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$MONITOR_USER"
fi
# gpio/spi/audio group membership so the monitor user can reach the devices
# raspi-config/the overlay create, without running the service as root.
for grp in gpio spi audio; do
  if getent group "$grp" >/dev/null 2>&1; then
    usermod -aG "$grp" "$MONITOR_USER"
  fi
done

install -d -o "$MONITOR_USER" -g "$MONITOR_USER" -m 750 "$VAR_DIR"
install -d -o "$MONITOR_USER" -g "$MONITOR_USER" -m 700 "$ETC_DIR"
install -d -o "$MONITOR_USER" -g "$MONITOR_USER" -m 750 "$OPT_DIR"

echo "== 5. deploy code + venv =="
# Only what the device actually runs: firmware/ (main.py and everything it
# imports) and ml/ (firmware/main.py inserts ml/ onto sys.path). frontend/
# and backend/ are the CLOUD side and never run on the Pi; docs/, tests/
# and .git are for a developer, not a 512 MB embedded box — T4.2's own
# database audit is the reminder that this device's SD card is not a place
# to be careless with space.
rsync -a --delete --exclude='__pycache__' --exclude='*.db' --exclude='baseline.npz' \
    "$REPO_SRC/firmware/" "$OPT_DIR/firmware/"
rsync -a --delete --exclude='__pycache__' "$REPO_SRC/ml/" "$OPT_DIR/ml/"
chown -R "$MONITOR_USER:$MONITOR_USER" "$OPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
  chown -R "$MONITOR_USER:$MONITOR_USER" "$VENV_DIR"
fi
echo "installing firmware/requirements.txt — this can take 15-40 minutes on a " \
     "Pi Zero 2W even with piwheels (the build guide (not in this public copy) §4.5). Not a hang."
sudo -u "$MONITOR_USER" "$VENV_DIR/bin/pip" install -q -r "$OPT_DIR/firmware/requirements.txt"

echo "== 6. config template =="
if [ ! -f "$ETC_DIR/config.yaml" ]; then
  install -o "$MONITOR_USER" -g "$MONITOR_USER" -m 600 \
      "$OPT_DIR/firmware/config.yaml" "$ETC_DIR/config.yaml"
  echo "wrote a TEMPLATE config to $ETC_DIR/config.yaml — edit device.id, " \
       "device.name and mqtt.api_key before starting the service."
else
  echo "$ETC_DIR/config.yaml already exists — left untouched (holds this device's own settings)."
fi

echo "== 7. systemd unit =="
# The venv fix this script's own header comment explains: the checked-in
# unit's ExecStart runs the system python3, which cannot see the venv's
# packages. sed the ExecStart line to the venv's interpreter when
# installing, rather than editing the checked-in file — the repo's copy
# stays a correct DESCRIPTION of what to install, this is the one place
# environment-specific like a real install path enters.
sed "s#^ExecStart=.*#ExecStart=$VENV_DIR/bin/python3 main.py --config $ETC_DIR/config.yaml#" \
    "$UNIT_SRC" > "$UNIT_DST"
systemctl daemon-reload
systemctl enable acoustic-monitor
echo "acoustic-monitor.service installed and enabled, NOT started — " \
     "run the learn period first (the operations runbook (not in this public copy) §1), then " \
     "'systemctl start acoustic-monitor'."

echo "== 8. bound the systemd journal =="
install -d -m 755 /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/acoustic-monitor.conf <<'EOF'
# Added by provision_pi.sh. The application's own SD-card writes are not a
# real wear risk (see docs/DOC_SOAK_DB_GROWTH.md, T4.2) — an UNBOUNDED
# systemd journal is a separate, common cause of a Pi's SD card filling up.
[Journal]
SystemMaxUse=200M
EOF
systemctl restart systemd-journald

echo
echo "Provisioning complete. Next: edit $ETC_DIR/config.yaml, run a learn " \
     "period (the operations runbook (not in this public copy) §1), then 'systemctl start acoustic-monitor'."
echo "Verify with: bash $(dirname "${BASH_SOURCE[0]}")/check_provision.sh"
