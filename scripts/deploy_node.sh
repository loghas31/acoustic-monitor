#!/usr/bin/env bash
# deploy_node.sh — push a code update to an ALREADY-PROVISIONED Pi over ssh,
# restart the service, tail the log (backlog T2.2).
#
#   scripts/deploy_node.sh <host> [ssh-user] [--no-restart] [--no-tail]
#
#   scripts/deploy_node.sh acoustic1.local
#   scripts/deploy_node.sh 192.168.1.42 pi
#
# WHAT THIS DOES NOT DO: create the `monitor` user, /opt/acoustic-monitor,
# the venv, or the systemd unit — that is scripts/provision_pi.sh (T2.1),
# a one-time step. This script is the DAY-TWO tool: you changed
# firmware/ or ml/ on your laptop, and want that code on the node without
# re-running the whole provisioning script (which is safe to re-run too,
# but slower — apt-get update alone costs real time on a Pi Zero 2 W, see
# the build guide (not in this public copy) §4.5). Run provision_pi.sh first, always.
#
# CANNOT BE VERIFIED END TO END IN THIS PROJECT'S DEVELOPMENT SANDBOX —
# there is no real Pi and no ssh target to deploy to. Verified here with
# `bash -n` (syntax) and `tests/test_deploy_node.py`, which points ssh/
# rsync's `-e`/target arguments at a FAKE remote made of a local directory
# plus a stub `ssh` executable on PATH that logs every invocation instead
# of opening a real connection — that proves the rsync source/exclude list
# and the ssh command sequence are exactly what they claim to be, without
# needing network access. NOT verified against a real sshd or a real
# systemd unit; do that once hardware exists (the handover notes (not in this public copy)'s week-1
# bring-up order covers when).
#
# Mirrors provision_pi.sh's own choices on purpose, so the two scripts
# don't disagree about what "the deployed code" means:
#   - Same two directories synced (firmware/, ml/) — nothing else runs on
#     the device. frontend/, backend/, docs/, tests/, .git are a
#     developer's machine's concern, not a 512 MB embedded box's.
#   - Same excludes (__pycache__, *.db, baseline.npz) — this script must
#     NEVER overwrite the device's own state.db or learned baseline.npz;
#     a code deploy is not a factory reset. rsync --delete only ever
#     applies inside firmware/ and ml/ on the remote, and those two files
#     are excluded from the delete pass specifically because --delete
#     would otherwise remove them for not existing in the source tree.
#   - Same remote path, /opt/acoustic-monitor, and the same venv
#     interpreter for anything this script needs to run remotely.

set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <host> [ssh-user] [--no-restart] [--no-tail]" >&2
  exit 1
}

[ "$#" -ge 1 ] || usage

HOST="$1"; shift || true
SSH_USER="pi"
NO_RESTART=0
NO_TAIL=0

# The second positional argument is the ssh user ONLY if it doesn't start
# with '--' — everything else here is a flag, order-independent after that.
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  SSH_USER="$1"; shift
fi
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-restart) NO_RESTART=1 ;;
    --no-tail) NO_TAIL=1 ;;
    *) echo "unrecognised argument: $1" >&2; usage ;;
  esac
  shift
done

REPO_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$REPO_SRC/firmware/main.py" ]; then
  echo "'$REPO_SRC' does not look like an acoustic-monitor checkout " \
       "(no firmware/main.py) — run this script from inside the repo." >&2
  exit 1
fi

OPT_DIR="/opt/acoustic-monitor"
REMOTE="$SSH_USER@$HOST"

echo "== deploying to $REMOTE:$OPT_DIR =="

echo "-- 1. syncing firmware/ --"
rsync -az --delete \
  --exclude='__pycache__' --exclude='*.db' --exclude='baseline.npz' \
  -e ssh \
  "$REPO_SRC/firmware/" "$REMOTE:$OPT_DIR/firmware/"

echo "-- 2. syncing ml/ --"
rsync -az --delete --exclude='__pycache__' \
  -e ssh \
  "$REPO_SRC/ml/" "$REMOTE:$OPT_DIR/ml/"

if [ "$NO_RESTART" -eq 1 ]; then
  echo "-- 3. --no-restart given, service left as-is --"
else
  echo "-- 3. restarting acoustic-monitor.service --"
  # sudo on the remote, not locally — the code above ran as whatever local
  # user invoked this script; the remote restart needs root (systemctl),
  # same convention provision_pi.sh's own comments describe for the unit
  # install. Requires passwordless sudo for systemctl on the Pi, or an
  # interactive password prompt over the ssh session (ssh -t keeps that
  # working rather than swallowing the prompt).
  ssh -t "$REMOTE" "sudo systemctl restart acoustic-monitor"
fi

if [ "$NO_TAIL" -eq 1 ]; then
  echo "-- 4. --no-tail given, skipping log --"
else
  echo "-- 4. tailing the log (Ctrl-C to stop watching; the deploy already happened) --"
  echo "   showing the last 20 lines then following new ones:"
  ssh -t "$REMOTE" "sudo journalctl -u acoustic-monitor -n 20 -f"
fi
