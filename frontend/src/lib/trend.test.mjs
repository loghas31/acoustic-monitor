// Plain-node tests for the T1.11 chart transforms. No DOM, no test runner, no
// new dependency — `node frontend/src/lib/trend.test.mjs` and it prints
// PASS/FAIL. tests/test_frontend_trend.py runs this inside the pytest suite so
// it cannot rot unnoticed.
//
// This covers the arithmetic that decides what a customer SEES. It does not
// render anything: nothing in this sandbox has a browser, and pretending
// otherwise is exactly what DOC_STATUS exists to prevent.

import { byRegime, hasDisplayIndex, sparklineCaption } from "./trend.js";

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name} ${detail}`);
  }
}

// -- byRegime -----------------------------------------------------------------

const READINGS = [
  { ts: 1, regime: 0, severity_band_rms_db: -23.8 },
  { ts: 2, regime: 1, severity_band_rms_db: -20.1 },
  { ts: 3, regime: 0, severity_band_rms_db: -21.0 },
  { ts: 4, regime: 1, severity_band_rms_db: -19.4 },
];

{
  const { rows, regimes } = byRegime(READINGS, "severity_band_rms_db");
  check("one series per regime", JSON.stringify(regimes) === "[0,1]", JSON.stringify(regimes));
  check("every reading kept", rows.length === 4, `got ${rows.length}`);
  // Each row carries ONLY its own regime's key, so the other line has a gap
  // there instead of being interpolated across a mode it was not running in.
  check("rows are exclusive per regime",
        rows.every((r) => ("v0" in r) !== ("v1" in r)), JSON.stringify(rows));
  check("regime 0 values", rows[0].v0 === -23.8 && rows[2].v0 === -21.0);
  check("regime 1 values", rows[1].v1 === -20.1 && rows[3].v1 === -19.4);
}

{
  // The backend stores NULL where the device reported a non-finite severity
  // (a silent window gives -Infinity dB). Those must not reach an axis domain.
  const withHoles = [
    { ts: 1, regime: 0, severity_band_rms_db: -23.8 },
    { ts: 2, regime: 0, severity_band_rms_db: null },
    { ts: 3, regime: 0 },                                    // key absent
    { ts: 4, regime: 0, severity_band_rms_db: -Infinity },   // belt and braces
    { ts: 5, regime: 0, severity_band_rms_db: -18.0 },
  ];
  const { rows, regimes } = byRegime(withHoles, "severity_band_rms_db");
  check("nulls, missing keys and infinities are dropped", rows.length === 2,
        JSON.stringify(rows));
  check("a regime with no plottable point does not appear",
        JSON.stringify(regimes) === "[0]", JSON.stringify(regimes));
}

{
  const { rows, regimes } = byRegime([], "severity_band_rms_db");
  check("empty input is empty output", rows.length === 0 && regimes.length === 0);
  const u = byRegime(undefined, "x");
  check("undefined input does not throw", u.rows.length === 0);
}

{
  // A device with no regime field at all (defensive: old rows) folds into 0.
  const { regimes } = byRegime([{ ts: 1, severity_band_rms_db: -5 }],
                              "severity_band_rms_db");
  check("missing regime defaults to 0", JSON.stringify(regimes) === "[0]");
}

// -- hasDisplayIndex ----------------------------------------------------------

check("index detected when present",
      hasDisplayIndex([{ display_index: 0 }]) === true);      // 0 is a real index
check("no index on pre-T1.7 rows",
      hasDisplayIndex([{ score: 5.0 }, { score: 6.0 }]) === false);
check("one indexed row is enough (unit upgraded mid-week)",
      hasDisplayIndex([{ score: 5.0 }, { display_index: 63 }]) === true);
check("empty readings claim no index", hasDisplayIndex([]) === false);

// -- sparklineCaption ---------------------------------------------------------

check("index caption names the alert line",
      sparklineCaption({ sparkline_field: "display_index" })
        === "health index, 7d (70 = alert)");
check("raw score is labelled as such, never as an index",
      sparklineCaption({ sparkline_field: "score" }) === "raw score, 7d");
check("anomaly count wins when there is one",
      sparklineCaption({ anomalies_7d: 2, sparkline_field: "score" })
        === "2 anomalies / 7d");
check("unknown field falls back to the index caption, not to silence",
      sparklineCaption({}) === "health index, 7d (70 = alert)");

console.log(failures === 0 ? "\nPASS trend.test.mjs" : `\nFAIL ${failures} check(s)`);
process.exit(failures === 0 ? 0 : 1);
