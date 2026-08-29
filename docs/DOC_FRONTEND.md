# Dashboard — what the customer sees

Companion to the system overview (not in this public copy) §6. Directory: `frontend/`.

React + Vite + Tailwind + Recharts. Builds to a static site (Vercel/Netlify)
that talks to the FastAPI backend.

---

## Views

### Fleet overview (`pages/Overview.jsx`)
A card per machine: name, green/amber/red badge, a 7-day score sparkline,
online state and last-seen. Polls every 30 s.

The colour is precomputed server-side at ingest, so this page is one query no
matter how many machines a site has.

**What the amber badge means — corrected 2026-08-18 (T1.7 / F5).** This
document previously described amber as "score in the top 30 % of the headroom
below threshold", the state that makes the dashboard feel alive between alerts.
Measured, that rule fired on **16.5 % of 200 healthy windows** and only
**12.5 %** of the windows of a developing fault — a badge more likely on a
healthy machine than a failing one. Amber is now computed on the device from
**state**: above threshold, but the 30-minute persistence gate has not fired
(0.5 % of healthy windows, 40 % of ramp windows). See
[DOC_ALERTING.md](DOC_ALERTING.md) §Health tiers.

**Sparkline units.** Plot the `display_index` (0–100, 70 = this machine's own
threshold), not the raw score. The raw distance spans 2.46 decades between a
healthy machine and severity 0.5, so a linear sparkline of it is a flat line
with one spike. `display_index` covers the same range in 47.0 → 91.3 and gives
healthy machines ~9 points of visible day-to-day movement.

**Built, T1.11.** All seven reported fields (`tier`, `display_index`,
`score_percentile` and the four `severity_*`) are now columns on
`models.Reading`, are written by `handle_telemetry`, and are returned by
`GET /devices/{id}/readings` and `/devices/{id}/status`. The fleet sparkline
returns `sparkline_field` saying which quantity it summarised, so the UI never
has to guess whether "5.0" is a score or an index. Verified end to end by
`python tools/e2e_severity_trend.py`: 40 firmware-published windows, 40
arriving with their index intact.

Adding the columns needed a migration, not just `create_all` — see
`models.add_missing_columns()`, which ALTERs in any nullable column the ORM has
and the database does not, and refuses to invent a NOT NULL one.

### Device detail (`pages/DeviceDetail.jsx`)
- **Sound picture** — a 24 h Mel-spectrogram heatmap (64 bins, 20 Hz–8 kHz)
- **Health index** over time, 0–100, with the alert line at a fixed **70**.
  This replaces the raw-score chart when the device reports an index, and the
  reason is not cosmetic: the raw score's threshold is *per regime*, so the
  old single dashed line was only ever correct for whichever regime happened
  to run last. Falls back to the raw score + that regime's threshold for
  pre-T1.7 firmware.
- **Operating regime** as a step chart — this is the view that *shows* the
  customer that mode changes are understood rather than alarmed at
- **Severity trend** — the physical, trendable pair, one line **per regime**:
  band RMS in dB (absolute level) and envelope peak / background as a **×
  ratio on a log axis** (contrast). Not two views of one thing: the ratio is
  exactly gain-invariant while band RMS moves 6.02 dB per doubling (measured),
  so a mic knocked 10 cm closer moves one panel and not the other. Per regime
  because at equal fault severity the same machine measured 434.6× contrast in
  one regime and 192.9× in the other.
- **Alert history** with the two feedback buttons

The transforms behind these charts (`src/lib/trend.js`) are separated from the
component so they can be tested without a DOM: `node
frontend/src/lib/trend.test.mjs`, which `tests/test_frontend_trend.py` runs as
part of the Python suite. **The rendering itself is still unverified** — no
browser exists in the development sandbox, so `npm run build` compiling is all
that has been observed.

### Alert config (`pages/AlertConfig.jsx`)
Email and webhook targets. **There is deliberately no sensitivity slider:**
thresholds are cross-validated per machine and per regime during the learn
period, and the customer's lever on false alarms is the feedback button, which
*retrains* rather than globally blunting detection.

### Onboarding (`pages/Onboarding.jsx`)
Four steps — plug in → name the machine → learn period → confirm — in plain
language, with each step explaining what is happening and why no alerts arrive
during learning.

## The feedback buttons

On every alert row:

| Button | Effect |
|---|---|
| **This was normal** | records verdict → device banks those windows → next retrain absorbs them → health red → amber (de-escalation only; the device's own tier takes over at the next telemetry window) |
| **Real problem** | records a confirmed catch |

This is the most commercially important control in the product, and it is two
buttons. See [DOC_ALERTING.md](DOC_ALERTING.md).

## Spectrogram rendering

`components/SpectrogramHeatmap.jsx` draws to a `<canvas>` via a single
`putImageData`, not SVG. 288 columns × 64 rows ≈ 18 000 cells would crawl as
DOM nodes. The colour map is a compact piecewise-linear approximation of
magma.

The device only uploads **64 column means per window** (~300 bytes), not the
full spectrogram — enough for a recognisable picture at negligible bandwidth.

## Demo mode

`api/client.js` falls back to `api/mock.js` on a network error or a 401, so
`npm run dev` renders a full simulated fleet with **no backend at all** — one
healthy machine, one with a developing fault, one amber, one offline. A banner
says "demo mode" so nobody mistakes it for real data.

This exists because you will demo this on a laptop in a room with bad wifi.

## Running and building

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, demo mode if no backend
npm run build    # static bundle in dist/
```

Verified: `vite build` completes in ~3.3 s. The bundle is ~610 kB (~180 kB
gzipped), dominated by Recharts — acceptable for a dashboard, and code
splitting is the obvious first optimisation if it ever matters.

## Status

Builds and renders against mock data. **Never tested against a live backend in
a browser** — the API contract is verified by `tests/test_api.py`, and the
mock mirrors those response shapes, but the two have not been exercised
together end to end. That is a week-4 task.
