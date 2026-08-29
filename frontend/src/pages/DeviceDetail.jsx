import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client.js";
import SpectrogramHeatmap from "../components/SpectrogramHeatmap.jsx";
import { Card, StatusBadge, timeAgo } from "../components/widgets.jsx";
// T1.11 chart transforms live in their own module so they can be executed by
// a test with no DOM — see frontend/src/lib/trend.test.mjs.
import { REGIME_COLOR, byRegime, hasDisplayIndex } from "../lib/trend.js";

const fmtT = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export default function DeviceDetail() {
  const { id } = useParams();
  const [status, setStatus] = useState(null);
  const [readings, setReadings] = useState([]);
  const [events, setEvents] = useState([]);

  const reload = useCallback(() => {
    api.status(id).then(setStatus);
    api.readings(id, Date.now() / 1000 - 86400).then(setReadings);
    api.anomalies(id).then(setEvents);
  }, [id]);

  useEffect(reload, [reload]);

  async function sendFeedback(eventId, verdict) {
    await api.feedback(eventId, verdict);
    reload();
  }

  if (!status) return <p className="text-slate-500">Loading…</p>;

  const threshold = status.latest?.threshold;
  const scoreData = readings.map((r) => ({
    t: fmtT(r.ts), score: r.score, regime: r.regime, index: r.display_index,
  }));
  const heatCols = readings.map((r) => r.mel_mean).filter(Boolean);

  // Prefer the display index: it is 70.0 at the alert threshold in EVERY
  // regime, so one horizontal reference line is honest for the whole series.
  // The raw score's threshold changes with the regime, so the dashed line on
  // that chart was only ever correct for whichever regime happened to be
  // running last. Fall back to the score for pre-T1.7 firmware.
  const hasIndex = hasDisplayIndex(readings);
  const stamped = readings.map((r) => ({ ...r, t: fmtT(r.ts) }));
  const rms = byRegime(stamped, "severity_band_rms_db");
  const env = byRegime(stamped, "severity_env_peak_ratio");
  const hasSeverity = rms.rows.length > 0 || env.rows.length > 0;
  const peakHz = status.latest?.severity_env_peak_hz;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-xl font-semibold text-white">{status.name}</h2>
        <StatusBadge health={status.health} />
        <span className="text-sm text-slate-500">
          {status.latest?.fr_reliable
            ? `running at ≈ ${status.latest.fr_hz?.toFixed(1)} Hz`
            : "speed estimate unreliable"}
          {" · "}{status.online ? "online" : "offline"} · seen {timeAgo(status.last_seen_ts)}
        </span>
      </div>

      <Card>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Sound picture (last 24 h)</h3>
        <SpectrogramHeatmap columns={heatCols} />
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <h3 className="text-sm font-medium text-slate-300 mb-2">
            {hasIndex ? "Health index" : "Anomaly score"}
            <span className="text-slate-500 font-normal">
              {hasIndex
                ? " — how far from this machine's own normal; 70 is the alert line"
                : " — distance from this machine's own normal"}
            </span>
          </h3>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={scoreData}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 11 }} minTickGap={40} />
                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} width={44}
                       domain={hasIndex ? [0, 100] : ["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                {hasIndex ? (
                  <ReferenceLine y={70} stroke="#fb7185" strokeDasharray="4 4"
                                 label={{ value: "alert threshold", fill: "#fb7185", fontSize: 10 }} />
                ) : threshold ? (
                  <ReferenceLine y={threshold} stroke="#fb7185" strokeDasharray="4 4"
                                 label={{ value: "alert threshold", fill: "#fb7185", fontSize: 10 }} />
                ) : null}
                <Line dataKey={hasIndex ? "index" : "score"} stroke="#38bdf8" dot={false}
                      strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-medium text-slate-300 mb-2">
            Operating regime
            <span className="text-slate-500 font-normal"> — learned modes; switching is normal, not an alert</span>
          </h3>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={scoreData}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 11 }} minTickGap={40} />
                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} width={44}
                       allowDecimals={false} domain={[0, 3]} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                <Line dataKey="regime" stroke="#a78bfa" dot={false} strokeWidth={1.5}
                      type="stepAfter" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* T1.11 — the physical severity trend. This is the chart a customer is
          actually shown to justify a repair: unlike the score, these are in
          physical units (dB), they do not reset when the baseline is
          retrained, and they were measured monotone in fault severity
          (Spearman rho = +1.000 over a simulated ramp, T1.7). One line per
          operating regime, because severity is only comparable within one. */}
      {hasSeverity && (
        <Card>
          <h3 className="text-sm font-medium text-slate-300 mb-2">
            Severity trend
            <span className="text-slate-500 font-normal">
              {" "}— physical loudness of the impacts, per operating regime
              {peakHz ? ` · repeating at ≈ ${peakHz.toFixed(1)} Hz` : ""}
            </span>
          </h3>
          <div className="grid lg:grid-cols-2 gap-4">
            {[
              { d: rms, label: "Band RMS (dB) — loudness of the impact band",
                unit: " dB", scale: "linear" },
              // A LINEAR ratio, not dB (firmware/reporting.physical_severity),
              // and it spans ~3x to ~580x over one fault ramp — on a linear
              // axis the whole healthy history is squashed onto the baseline.
              { d: env, label: "Envelope peak / background (×, log scale) — impact contrast",
                unit: "×", scale: "log" },
            ].map(({ d, label, unit, scale }) => (
              <div key={label}>
                <p className="text-xs text-slate-500 mb-1">{label}</p>
                <div className="h-48">
                  <ResponsiveContainer>
                    <LineChart data={d.rows}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 11 }} minTickGap={40} />
                      <YAxis tick={{ fill: "#64748b", fontSize: 11 }} width={44} unit={unit}
                             scale={scale} domain={scale === "log" ? ["auto", "auto"] : undefined}
                             allowDataOverflow={false} />
                      <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                      {d.regimes.map((g) => (
                        <Line key={g} dataKey={`v${g}`} name={`regime ${g}`}
                              stroke={REGIME_COLOR[g % REGIME_COLOR.length]}
                              dot={false} strokeWidth={1.5} connectNulls={false}
                              isAnimationActive={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-600 mt-2">
            Left panel is absolute level, right panel is contrast, and they are
            independent: multiplying the whole signal by 8 moves the left panel
            +18 dB and leaves the right panel unchanged (measured). Both rising
            together over days is the degradation signature. A step inside one
            window is a change of load, not a fault.
          </p>
        </Card>
      )}

      <Card>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Alert history</h3>
        {events.length === 0 ? (
          <p className="text-sm text-slate-500">No anomalies recorded.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-slate-500 text-left text-xs">
              <tr>
                <th className="py-1">Time</th><th>Score / threshold</th>
                <th>Persisted</th><th>Your verdict</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id} className="border-t border-slate-800">
                  <td className="py-2">{new Date(e.ts * 1000).toLocaleString()}</td>
                  <td>{e.score?.toFixed(1)} / {e.threshold?.toFixed(1)}</td>
                  <td>{e.persisted_minutes} min</td>
                  <td>
                    {e.feedback === "normal" && <span className="text-sky-300">✓ marked normal — sensor will learn it</span>}
                    {e.feedback === "fault" && <span className="text-rose-300">✓ confirmed fault</span>}
                    {!e.feedback && (
                      <span className="flex gap-2">
                        <button onClick={() => sendFeedback(e.id, "normal")}
                                className="text-xs px-2 py-1 rounded bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/40 hover:bg-sky-500/30">
                          This was normal
                        </button>
                        <button onClick={() => sendFeedback(e.id, "fault")}
                                className="text-xs px-2 py-1 rounded bg-rose-500/15 text-rose-300 ring-1 ring-rose-500/40 hover:bg-rose-500/30">
                          Real problem
                        </button>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
