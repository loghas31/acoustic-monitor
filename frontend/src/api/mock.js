// Deterministic fake fleet so `npm run dev` demos with zero backend.
// Shapes mirror the v2 FastAPI responses exactly — swap-in/swap-out guarantee.

const now = Date.now() / 1000;
const rand = (() => { let s = 42; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();

function scoreSeries(n, anomalyAfter = null) {
  // Healthy Mahalanobis scores sit ~4-7 under a threshold of ~8.7;
  // a developing fault climbs into the tens.
  return Array.from({ length: n }, (_, i) => {
    let v = 4.5 + 2.0 * rand();
    if (anomalyAfter !== null && i > anomalyAfter)
      v += 1.5 * (i - anomalyAfter) * (1 + 0.3 * rand());
    return v;
  });
}

// T1.11. The display index the firmware reports: log-linear in
// score/threshold and pinned to exactly 70.0 at threshold, bounded 0-100
// (firmware/reporting.py). Reproduced here so the mock's shape matches the
// real API rather than merely resembling it; measured healthy windows against
// the repo baseline span roughly 31-70, which is what this produces.
const THRESH = 8.7;
const indexFromScore = (s) =>
  Math.max(0, Math.min(100, 70 + 30 * (Math.log10(s / THRESH) / 1.5)));

function melSketch(n, fault = false) {
  // 64-bin log-Mel column means; faults add energy in the high bins
  // (resonance band ~4.5 kHz lives around bin 50 of 64 at 8 kHz fmax).
  return Array.from({ length: n }, (_, t) =>
    Array.from({ length: 64 }, (_, m) => {
      let v = 2.2 * Math.exp(-((m - 8) ** 2) / 60) + 0.35 * rand();
      if (fault && t > n * 0.6) v += 1.8 * Math.exp(-((m - 50) ** 2) / 30) * ((t - n * 0.6) / (n * 0.4));
      return v;
    })
  );
}

const DEVICES = [
  { device_id: "mock-1", name: "Compressor A", health: "green", online: true,
    anomalies_7d: 0, last_seen_ts: now - 12, scores: scoreSeries(168) },
  { device_id: "mock-2", name: "Injection moulder 3", health: "red", online: true,
    anomalies_7d: 2, last_seen_ts: now - 31, scores: scoreSeries(168, 120) },
  { device_id: "mock-3", name: "HVAC roof unit", health: "amber", online: true,
    anomalies_7d: 0, last_seen_ts: now - 8, scores: scoreSeries(168, 158) },
  // Deliberately left on pre-T1.7 firmware: exercises the API's per-device
  // fallback to the raw score, which is otherwise never seen in the demo.
  { device_id: "mock-4", name: "Conveyor B (packing)", health: "green", online: false,
    anomalies_7d: 0, last_seen_ts: now - 7200, scores: scoreSeries(168),
    legacyFirmware: true },
].map((d) => ({
  ...d,
  sparkline: d.legacyFirmware ? d.scores : d.scores.map(indexFromScore),
  sparkline_field: d.legacyFirmware ? "score" : "display_index",
}));

const FEEDBACK = {};   // event_id -> verdict (in-memory; mock only)

export const mockApi = {
  summary: async () => ({ devices: DEVICES }),

  status: async (id) => {
    const d = DEVICES.find((x) => x.device_id === id) ?? DEVICES[0];
    const s = d.scores.at(-1);
    return { ...d, latest: { ts: d.last_seen_ts,
      score: s, threshold: 8.7,
      regime: 0, fr_hz: 49.6, fr_reliable: true, band: [3866, 5420],
      ...(d.legacyFirmware ? {} : {
        tier: d.health, display_index: indexFromScore(s),
        severity_band_rms_db: -23.8, severity_env_peak_hz: 152.5,
        severity_env_peak_ratio: 19.8, severity_env_db_re_learn: 0.0 }) } };
  },

  readings: async (id) => {
    const d = DEVICES.find((x) => x.device_id === id) ?? DEVICES[0];
    const fault = d.health === "red";
    const n = 288;
    const scores = scoreSeries(n, fault ? 170 : null);
    const mel = melSketch(n, fault);
    return scores.map((s, i) => {
      const row = {
        ts: now - (n - i) * 300, score: s, threshold: 8.7,
        regime: i % 16 < 8 ? 0 : 1,                // visible regime alternation
        anomalous: fault && i > 200, fr_hz: 49.6 + 0.1 * rand(), mel_mean: mel[i],
      };
      if (d.legacyFirmware) return row;            // pre-T1.7: no reportable layer
      // T1.11 severity trend. Regime 1 sits ~3 dB louder than regime 0 for
      // entirely healthy reasons — that offset is exactly why the chart draws
      // one line per regime instead of one line for the machine.
      const growth = fault ? Math.max(0, (i - 170) / (n - 170)) : 0;
      const offset = row.regime === 1 ? 3.0 : 0.0;
      return { ...row,
        tier: row.anomalous ? "red" : "green",
        display_index: indexFromScore(s),
        severity_band_rms_db: -23.8 + offset + 17.8 * growth + 0.3 * rand(),
        severity_env_peak_hz: 152.5,
        severity_env_peak_ratio: 19.8 + offset + 45.0 * growth + 0.4 * rand(),
        severity_env_db_re_learn: 20.9 * growth };
    });
  },

  anomalies: async (id) => {
    const d = DEVICES.find((x) => x.device_id === id) ?? DEVICES[0];
    if (d.health !== "red") return [];
    return [0, 1].map((k) => ({
      id: `ev-${k}`, ts: now - 3600 * (8 - 4 * k), score: 45 + 20 * k,
      threshold: 8.7, regime: 0,
      ts_from: now - 3600 * (8 - 4 * k) - 1800, ts_to: now - 3600 * (8 - 4 * k),
      persisted_minutes: 30, feedback: FEEDBACK[`ev-${k}`] ?? "",
      acknowledged: Boolean(FEEDBACK[`ev-${k}`]) }));
  },

  feedback: async (eventId, verdict) => {
    FEEDBACK[eventId] = verdict;
    return { ok: true, verdict, device_notified: false };
  },

  alertLog: async () => [
    { ts: now - 7200, channel: "email", target: "ops@example.com", status: "sent", detail: "sendgrid" },
    { ts: now - 7200, channel: "webhook", target: "https://hooks.slack.com/…", status: "sent", detail: "http 200" },
  ],

  configureAlerts: async () => ({ ok: true }),
  registerDevice: async () => ({ device_id: "mock-new", api_key: "demo-key-not-real" }),
};
