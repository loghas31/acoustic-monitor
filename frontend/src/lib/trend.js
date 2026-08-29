// Chart-data transforms for the device page (backlog T1.11).
//
// These live outside the component so they can be executed by a test without a
// browser or a DOM: `node frontend/src/lib/trend.test.mjs`, which
// tests/test_frontend_trend.py runs as part of the normal pytest suite. The
// arithmetic that decides WHAT a customer sees is worth having under test even
// though the rendering itself is not.

/** Colour per operating regime; index by `regime % length`. */
export const REGIME_COLOR = ["#38bdf8", "#a78bfa", "#fbbf24", "#f472b6"];

/**
 * Split one severity field into one recharts series per operating regime.
 *
 * Severity is only comparable WITHIN a regime — an idling machine and a loaded
 * one differ in band RMS for entirely healthy reasons — so a single line for
 * the machine would show a sawtooth every time it changed gear and hide the
 * slow trend that actually matters. Each row carries only the key for its own
 * regime, so recharts leaves a gap in the other lines (`connectNulls={false}`)
 * rather than interpolating across a period when that mode was not running.
 *
 * Rows with a null/undefined/non-finite value are dropped: the backend stores
 * NULL for a window whose severity was non-finite (a silent window gives
 * -Infinity dB), and plotting that would blow up the axis domain.
 *
 * @param {Array<object>} readings  rows from GET /devices/{id}/readings
 * @param {string} field            e.g. "severity_band_rms_db"
 * @returns {{rows: Array<object>, regimes: number[]}}
 */
export function byRegime(readings, field) {
  const rows = [];
  const seen = new Set();
  for (const r of readings ?? []) {
    const v = r?.[field];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    const g = r.regime ?? 0;
    seen.add(g);
    rows.push({ t: r.t ?? r.ts, [`v${g}`]: v });
  }
  return { rows, regimes: [...seen].sort((a, b) => a - b) };
}

/**
 * Does this device report the display index, or only a raw score?
 *
 * The index is 70.0 at the alert threshold in every regime; the raw score's
 * threshold changes with the regime, so a single reference line on a raw-score
 * chart is only correct for whichever regime ran last. Pre-T1.7 firmware sends
 * no index at all, hence the check rather than an assumption.
 */
export function hasDisplayIndex(readings) {
  return (readings ?? []).some(
    (r) => r?.display_index !== null && r?.display_index !== undefined);
}

/** Caption for the fleet sparkline. The backend says which field it summed;
 *  never guess, or a raw score of 5 reads as "far below the 70 alert line". */
export function sparklineCaption(device) {
  if (device?.anomalies_7d) return `${device.anomalies_7d} anomalies / 7d`;
  return device?.sparkline_field === "score"
    ? "raw score, 7d"
    : "health index, 7d (70 = alert)";
}
