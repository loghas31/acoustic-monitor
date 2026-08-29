# Backend — the cloud half

Companion to the system overview (not in this public copy) §6. Directory: `backend/`.

---

## Shape

```
device ──MQTT/TLS──► Mosquitto ──► mqtt_bridge.py ──► PostgreSQL
                                          │
                                          └──► alerts.py ──► email / webhook
browser ──HTTPS──► FastAPI (main.py) ──────┘
```

Four containers in `docker-compose.yml`: `db`, `mqtt`, `api`, `bridge`.

**Why the bridge is a separate process from the API:** their failure modes
differ. The API should restart and scale freely. The bridge must hold a
persistent MQTT session (`clean_session=False`, QoS 1) so anomaly events queue
at the broker while it is down rather than being lost.

## Files

| File | Role |
|---|---|
| `main.py` | FastAPI app, all HTTP endpoints |
| `models.py` | SQLAlchemy ORM: users, devices, readings, anomaly_events, alert_configs, alerts |
| `mqtt_bridge.py` | subscribes `devices/+/+`, writes to DB, fans out alerts |
| `alerts.py` | SendGrid → SMTP fallback, plus user webhooks |
| `auth.py` | device API keys + user JWTs |
| `docker-compose.yml`, `Dockerfile`, `mosquitto/` | deployment |

## Two auth domains, deliberately separate

- **Devices** authenticate with an `X-API-Key` header. The key is 32 random
  bytes, shown **once** at registration, stored only as SHA-256. Plain SHA-256
  is correct here — salting and stretching defend *low-entropy* secrets, and a
  256-bit random key cannot be dictionary-attacked.
- **Humans** authenticate with a Bearer JWT from email/password login.
  Passwords use `hashlib.scrypt` (memory-hard, stdlib, no native dependency).

A device can only write its own data; a user can only read their own devices.
Cross-user access returns **404, not 403** — a 403 would confirm the resource
exists.

## Endpoints

| Method | Path | Who |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | public |
| POST | `/devices/register` | user — returns the API key once |
| GET | `/devices/{id}/status`, `/readings`, `/anomalies` | user |
| POST | `/readings`, `/anomalies` | device (HTTP twin of MQTT) |
| POST | `/anomalies/{id}/feedback` | user — **the false-alarm kill switch** |
| POST | `/alerts/configure` | user |
| GET | `/alerts/log/{id}`, `/dashboard/summary` | user |

HTTP ingest duplicates the MQTT path on purpose: some customer networks block
non-HTTP egress, and `curl` beats a broker during development. Both paths call
the *same* handler functions in `mqtt_bridge.py` — one source of truth.

## The feedback endpoint

`POST /anomalies/{id}/feedback {"verdict": "normal"}` does three things:

1. records the verdict and marks the event acknowledged,
2. publishes `mark_normal` to `devices/{id}/cmd` so the node banks those
   windows for retraining,
3. de-escalates device health from red to amber.

Step 2 is **best-effort**: if the broker is unreachable the verdict is still
recorded in the database, which remains the source of truth. Losing a
notification must never lose a customer's decision.

## Health tiers at ingest

`health_from_score()` computes green/amber/red **when telemetry arrives**, so
the fleet page is a single indexed scan rather than an aggregate query per
device on every poll.

## Alert dispatch

SendGrid first, SMTP fallback, skip if neither is configured — and the webhook
fires regardless, so email misconfiguration cannot silence the system. Every
attempt is written to the `alerts` table with status and detail, because
*"did the customer get warned?"* is an auditable business question.

Alert wording is deliberately non-diagnostic: it reports how far above the
learned threshold the machine has been and for how long, recommends
inspection, and invites the "this was normal" press if the behaviour was
legitimate.

## Running it

```bash
cd backend && docker-compose up --build     # full stack
# or, without Docker:
DATABASE_URL=sqlite:////tmp/dev.db python -m uvicorn main:app --port 8000
```

## Status

13 API tests pass (`tests/test_api.py`), covering login, ingest auth, readings,
device-reported health tiers, the severity trend on `/readings` and `/status`,
the sparkline's field choice, the anomaly + feedback flow, alert config and
cross-user isolation. A further 12 in `tests/test_severity_persistence.py`
cover the bridge's coercion rules and the schema migration.

**Schema changes need `models.add_missing_columns()`, not just `create_all`.**
`create_all` adds missing *tables* and silently ignores missing *columns*, so
T1.11's severity columns would have existed on a fresh test database and been
absent on any database that already had a `readings` table — the insert
succeeds and the value disappears. `init_db()` now calls the migration helper,
which ALTERs in any nullable column the ORM declares and the database lacks,
is idempotent, and raises rather than guessing a backfill for a NOT NULL
column. Anything more than that needs Alembic and a human.

A live `uvicorn` + SQLite run was verified end to end: register user → register
device → ingest reading → ingest anomaly → GET anomalies → POST feedback →
device health red → amber.

⚠ **`docker-compose` has never been run** — no Docker daemon in the
development sandbox. Expect small fixes on first use. Mosquitto is configured
anonymous for development; the production checklist (TLS listener, per-device
passwords, ACLs) is in `mosquitto/mosquitto.conf`.
