#!/usr/bin/env bash
# dev_up.sh — bring up the backend WITHOUT Docker (backlog T2.3).
#
#   bash scripts/dev_up.sh
#
# `backend/docker-compose.yml` brings up Postgres + Mosquitto + the API + the
# MQTT bridge, four containers, for the full production topology. Day one —
# before there is a Pi, a broker, or a reason to run four containers — you
# just need the API up so the dashboard has something to talk to and
# `firmware/main.py --simulate --no-mqtt` (or a real device over HTTP) has
# somewhere to POST readings. This script is that: uvicorn + SQLite, one
# process, no Docker daemon required.
#
# What this deliberately does NOT start: Mosquitto, or `mqtt_bridge.py`. The
# API works fully without either — `POST /readings` and `POST /anomalies` are
# the HTTP twin of the MQTT ingest path and share the same handler code
# (backend/mqtt_bridge.py's handle_telemetry/handle_anomaly, called directly
# by backend/main.py) — but the ONE feature that needs a live broker is the
# "this was normal" downlink (`publish_cmd` in backend/main.py): the verdict
# is still recorded in the database either way, but the device will not hear
# about it until you have a broker running and `mqtt_bridge.py` beside it, or
# until you run `firmware/baseline.py --retrain` by hand. Said plainly here so
# nobody spends an hour debugging a downlink that was never going to arrive.
#
# What this script does:
#   1. Checks python3 is on PATH (first, before anything else that shells
#      out — see the comment at that check for why the order matters).
#   2. Installs backend/requirements.txt if fastapi/uvicorn aren't already
#      importable (skipped if they are — most runs after the first).
#   3. Points DATABASE_URL at a SQLite file under /tmp (pass --fresh to
#      delete it and start clean) unless the caller already exported one.
#   4. Runs `python3 -m uvicorn main:app` in the foreground, in backend/, on
#      0.0.0.0:8000 by default. Ctrl+C stops it, same as docker-compose.
#
# After it's up, in another terminal:
#   curl http://localhost:8000/openapi.json            # readiness check
#   curl -X POST http://localhost:8000/auth/register \
#        -H 'Content-Type: application/json' \
#        -d '{"email":"you@example.com","password":"change-me-now"}'
#   cd frontend && VITE_API_URL=http://localhost:8000 npm run dev
#
# tests/test_dev_up.py runs this script for real (background, health-checked,
# then killed) as part of the normal pytest suite, so "does it actually
# serve" is not a claim resting on someone having run it by hand once.

set -euo pipefail

# python3 check FIRST, before anything else touches an external command.
# `command -v` is a bash builtin (no PATH-search binary of its own needed),
# so this is the one check that still works even when PATH is broken badly
# enough that ordinary coreutils like `dirname` are missing too. It used to
# run after REPO_DIR was computed with `dirname` below — tests/test_dev_up.py
# caught that ordering bug for real: a PATH broken enough to hide `python3`
# is also broken enough to hide `dirname`, so the script died on `dirname:
# command not found` two lines before ever reaching this check, instead of
# printing its own clear message.
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH — install Python 3.10+ first." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_DIR/backend"
HOST="${DEV_UP_HOST:-0.0.0.0}"
PORT="${DEV_UP_PORT:-8000}"
# /tmp, not backend/: SQLite needs POSIX file locks that a network-mounted
# repo directory (a dev container, an NFS home, this project's own CI/agent
# sandboxes) often does not provide — `sqlite3.OperationalError: disk I/O
# error` on the very first CREATE TABLE. tests/test_api.py hit this exact
# thing first (see its own comment) and dev_up.sh reproduced it during
# T2.3's own verification before this line existed. A plain local disk
# would have worked either way, so this loses nothing there.
#
# uid-qualified filename, not a bare `acoustic-monitor-dev.db`: this repo's
# own test suite hit a real bug from exactly this shortcut (F12 in
# the commit log (not in this public copy)) — a fixed filename under shared /tmp collides across
# different-uid containers, and whichever uid didn't create the file gets a
# permission error instead of a fresh database. Same fix applied here as was
# applied to conftest.py / tests/test_api.py: qualify by uid so two
# different users (or two different agent-sandbox containers) on the same
# /tmp never fight over the same inode.
#
# $EUID (bash builtin), not `$(id -u)`: the python3-not-on-PATH test in
# tests/test_dev_up.py caught this the hard way — shelling out to `id`
# happened before the python3-on-PATH check below, so a broken-enough PATH
# (one missing `id` too, not just `python3`) produced a confusing
# "id: command not found" instead of this script's own clear error message.
# $EUID needs no external command and no PATH at all.
DB_FILE="${TMPDIR:-/tmp}/acoustic-monitor-dev-$EUID.db"

if [ "${1:-}" = "--fresh" ]; then
  echo "--fresh: removing $DB_FILE (if present)"
  rm -f "$DB_FILE"
fi

echo "Backend dir: $BACKEND_DIR"
cd "$BACKEND_DIR"

if ! python3 -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "fastapi/uvicorn not importable — installing backend/requirements.txt..."
  python3 -m pip install -r requirements.txt
fi

# Respect an already-exported DATABASE_URL (e.g. someone pointing this at a
# real Postgres for a hybrid dev setup); default to the SQLite file above
# otherwise.
if [ -z "${DATABASE_URL:-}" ]; then
  export DATABASE_URL="sqlite:///$DB_FILE"
fi

echo "DATABASE_URL: $DATABASE_URL"
echo "Starting API on http://$HOST:$PORT  (Ctrl+C to stop)"
echo "No MQTT broker started — HTTP ingest only. See this script's header"
echo "comment for what that does and does not affect."
echo

exec python3 -m uvicorn main:app --host "$HOST" --port "$PORT"
