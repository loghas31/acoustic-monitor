// contract.test.mjs — backlog T3.6, the frontend<->backend JSON contract
// check. Run via `node frontend/src/api/contract.test.mjs <real-backend.json>`;
// tests/test_frontend_backend_integration.py drives it as part of the normal
// pytest suite (same pattern T1.11 established for trend.test.mjs: plain node,
// no DOM, no new dependency — this repo has no JS test runner on purpose).
//
// mock.js's own header comment makes a promise: "Shapes mirror the v2 FastAPI
// responses exactly — swap-in/swap-out guarantee." This is the first thing
// that checks it. Two directions matter, and they are different bugs:
//   (a) a real-backend field a component READS is missing from mock.js
//       -> the zero-backend `npm run dev` demo is broken (or was never
//          exercising a real code path).
//   (b) a real-backend field a component READS is missing from the REAL
//       backend's response -> the app breaks the moment a real device
//       reports in, and the mock never would have caught it.
// The specific field names below are not invented — they are grepped
// directly out of Overview.jsx, DeviceDetail.jsx, AlertConfig.jsx,
// Onboarding.jsx and lib/trend.js, so this file cannot drift from what the
// components actually do without someone noticing it was never updated.

import { readFileSync } from "node:fs";
import { mockApi } from "./mock.js";

const realPath = process.argv[2];
if (!realPath) {
  console.error("usage: node contract.test.mjs <real-backend.json>");
  process.exit(1);
}
const real = JSON.parse(readFileSync(realPath, "utf8"));

let failures = 0;
const notes = [];

function fail(msg) {
  failures += 1;
  console.error(`FAIL: ${msg}`);
}

function note(msg) {
  notes.push(msg);
}

/** Every key in `obj`, or [] for null/undefined (a legitimately absent
 * `latest`, for instance, must not throw here). */
function keys(obj) {
  return obj && typeof obj === "object" ? Object.keys(obj) : [];
}

/** Assert every field in `required` is present (key exists, not necessarily
 * non-null — `severity_env_peak_hz` on a legacy device is a real, meaningful
 * null, not a bug) in both `mockSample` and `realSample`. `label` names the
 * endpoint; `readBy` names the component/function that actually reads each
 * field, so a failure message points straight at the code to fix. */
function checkRequiredFields(label, required, mockSample, realSample) {
  for (const [field, readBy] of Object.entries(required)) {
    const inMock = keys(mockSample).includes(field);
    const inReal = keys(realSample).includes(field);
    if (!inMock) fail(`${label}: mock.js is missing "${field}", but ${readBy} reads it — the zero-backend demo would crash or silently show nothing here`);
    if (!inReal) fail(`${label}: the REAL backend is missing "${field}", but ${readBy} reads it — this breaks the moment a real device reports in`);
  }
}

// ----------------------------------------------------------------------------
// 1. GET /dashboard/summary  ->  Overview.jsx, lib/trend.js#sparklineCaption
// ----------------------------------------------------------------------------
{
  const mockSummary = await mockApi.summary();
  const mockDevice = mockSummary.devices[0];
  const realDevices = real.summary.devices;
  if (realDevices.length === 0) {
    fail("summary: real backend returned zero devices — the fixture seeded three");
  } else {
    for (const d of realDevices) {
      checkRequiredFields("summary device", {
        device_id: "Overview.jsx (<Link>/key), Onboarding.jsx",
        name: "Overview.jsx (<h3>)",
        health: "Overview.jsx (StatusBadge, red-count filter)",
        online: "Overview.jsx (online/offline caption)",
        last_seen_ts: "Overview.jsx (timeAgo)",
        anomalies_7d: "lib/trend.js#sparklineCaption",
        sparkline: "Overview.jsx (<Sparkline data=.../>)",
        sparkline_field: "lib/trend.js#sparklineCaption",
      }, mockDevice, d);
    }
  }
}

// ----------------------------------------------------------------------------
// 2. GET /devices/{id}/status  ->  DeviceDetail.jsx
// ----------------------------------------------------------------------------
{
  const mockStatus = await mockApi.status("mock-1");
  const required = {
    name: "DeviceDetail.jsx (<h2>{status.name}</h2>)",
    health: "DeviceDetail.jsx (StatusBadge)",
    online: "DeviceDetail.jsx (online/offline caption)",
    last_seen_ts: "DeviceDetail.jsx (timeAgo)",
  };
  const requiredLatest = {
    threshold: "DeviceDetail.jsx (status.latest?.threshold)",
    fr_hz: "DeviceDetail.jsx (status.latest.fr_hz.toFixed)",
    fr_reliable: "DeviceDetail.jsx (speed estimate reliable/unreliable branch)",
    severity_env_peak_hz: "DeviceDetail.jsx (peakHz caption)",
  };
  for (const [name, realStatus] of Object.entries(real.status)) {
    checkRequiredFields(`status (${name})`, required, mockStatus, realStatus);
    if (!("latest" in realStatus)) {
      fail(`status (${name}): real backend response has no "latest" key at all`);
    } else if (realStatus.latest !== null) {
      checkRequiredFields(`status (${name}).latest`, requiredLatest,
        mockStatus.latest, realStatus.latest);
    }
  }
}

// ----------------------------------------------------------------------------
// 3. GET /devices/{id}/readings  ->  DeviceDetail.jsx, lib/trend.js
// ----------------------------------------------------------------------------
{
  const mockReadings = await mockApi.readings("mock-1");
  const required = {
    ts: "DeviceDetail.jsx (fmtT(r.ts))",
    score: "DeviceDetail.jsx (scoreData score fallback)",
    regime: "DeviceDetail.jsx (regime chart), lib/trend.js#byRegime",
    mel_mean: "DeviceDetail.jsx (heatCols / SpectrogramHeatmap)",
  };
  for (const [name, rows] of Object.entries(real.readings)) {
    if (rows.length === 0) {
      fail(`readings (${name}): real backend returned zero rows — the fixture seeded some`);
      continue;
    }
    checkRequiredFields(`readings (${name})[0]`, required, mockReadings[0], rows[0]);
    // display_index/severity_* are legitimately ABSENT (key present, value
    // null) on the legacy device — hasDisplayIndex/byRegime are built
    // specifically to handle that, so only check the key exists on rows
    // that are supposed to have it.
    if (name !== "legacy") {
      checkRequiredFields(`readings (${name})[0] (T1.11 fields)`, {
        display_index: "lib/trend.js#hasDisplayIndex, DeviceDetail.jsx scoreData",
        severity_band_rms_db: "lib/trend.js#byRegime('severity_band_rms_db')",
        severity_env_peak_ratio: "lib/trend.js#byRegime('severity_env_peak_ratio')",
      }, mockReadings[0], rows[0]);
    } else if (rows[0].display_index !== null && rows[0].display_index !== undefined) {
      fail("readings (legacy): fixture/backend gave the legacy device a display_index — hasDisplayIndex() cannot be exercised honestly");
    }
  }
}

// ----------------------------------------------------------------------------
// 4. GET /devices/{id}/anomalies  ->  DeviceDetail.jsx (alert history table)
// ----------------------------------------------------------------------------
{
  const mockAnomalies = await mockApi.anomalies("mock-2");   // mock-2 is the red device
  const required = {
    id: "DeviceDetail.jsx (row key, feedback button target)",
    ts: "DeviceDetail.jsx (new Date(e.ts*1000))",
    score: "DeviceDetail.jsx (e.score?.toFixed)",
    threshold: "DeviceDetail.jsx (e.threshold?.toFixed)",
    persisted_minutes: "DeviceDetail.jsx (persisted column)",
    feedback: "DeviceDetail.jsx (verdict branch: normal/fault/pending buttons)",
  };
  const realFaultyAnomalies = real.anomalies.faulty;
  if (realFaultyAnomalies.length === 0) {
    fail("anomalies (faulty): real backend returned zero events — the fixture posted one");
  } else {
    checkRequiredFields("anomalies (faulty)[0]", required, mockAnomalies[0] ?? {}, realFaultyAnomalies[0]);
    if (realFaultyAnomalies[0].feedback !== "normal") {
      fail(`anomalies (faulty)[0]: expected feedback "normal" after the test posted it, got ${JSON.stringify(realFaultyAnomalies[0].feedback)}`);
    }
  }
  if (real.anomalies.healthy.length !== 0) {
    fail("anomalies (healthy): the healthy device fixture should have zero anomaly events");
  }
}

// ----------------------------------------------------------------------------
// 5. POST /anomalies/{id}/feedback  ->  DeviceDetail.jsx#sendFeedback (return
//    value is not read by the component — it just triggers a reload — but
//    client.js's own JSDoc-free contract with the backend is still checked)
// ----------------------------------------------------------------------------
{
  const mockFeedback = await mockApi.feedback("ev-0", "normal");
  checkRequiredFields("feedback", {
    ok: "client.js (POST /anomalies/{id}/feedback caller)",
    verdict: "client.js response shape",
    device_notified: "client.js response shape",
  }, mockFeedback, real.feedback);
}

// ----------------------------------------------------------------------------
// 6. POST /devices/register  ->  Onboarding.jsx
// ----------------------------------------------------------------------------
{
  const mockReg = await mockApi.registerDevice("x");
  checkRequiredFields("registerDevice", {
    device_id: "Onboarding.jsx (Device ID: {device?.device_id})",
  }, mockReg, real.registerDevice);
}

// ----------------------------------------------------------------------------
// 7. GET /alerts/log/{id}  ->  fetched by client.js, but NOT rendered by any
//    page today (grepped: no .jsx file calls api.alertLog). Recorded as a
//    finding, not a failure — the shape is still checked against mock.js
//    since a future page presumably will read it.
// ----------------------------------------------------------------------------
{
  const mockLog = await mockApi.alertLog();
  if (real.alertLog.length === 0) {
    fail("alertLog: real backend returned zero rows — the fixture configured a webhook and posted an anomaly specifically to populate this");
  } else {
    checkRequiredFields("alertLog[0]", {
      ts: "(unused by any current page — see note below)",
      channel: "(unused by any current page — see note below)",
      target: "(unused by any current page — see note below)",
      status: "(unused by any current page — see note below)",
    }, mockLog[0], real.alertLog[0]);
  }
  note("api.alertLog() / GET /alerts/log/{id} is fetched by client.js but no .jsx page currently calls or renders it (grepped: 0 matches for \"alertLog\" outside api/). Dead frontend API surface, not a contract bug — flagged so it isn't mistaken for one.");
}

// ----------------------------------------------------------------------------
console.log("");
for (const n of notes) console.log(`NOTE: ${n}`);
console.log("");
if (failures > 0) {
  console.log(`FAIL: ${failures} contract mismatch(es) above.`);
  process.exit(1);
}
console.log("PASS: real backend and mock.js both satisfy every field this repo's own components read.");
