// Small shared components. One file — they're each a few lines.

import { Line, LineChart, ResponsiveContainer } from "recharts";

export const HEALTH = {
  green: { label: "Healthy", cls: "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30" },
  amber: { label: "Watch", cls: "bg-amber-500/15 text-amber-400 ring-amber-500/30" },
  red: { label: "Anomaly", cls: "bg-rose-500/15 text-rose-400 ring-rose-500/30" },
  unknown: { label: "No data", cls: "bg-slate-500/15 text-slate-400 ring-slate-500/30" },
};

export function StatusBadge({ health }) {
  const h = HEALTH[health] ?? HEALTH.unknown;
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ring-1 ${h.cls}`}>
      {h.label}
    </span>
  );
}

export function Sparkline({ data, color = "#34d399" }) {
  const pts = (data ?? []).map((v, i) => ({ i, v }));
  if (!pts.length) return <div className="h-10" />;
  return (
    <div className="h-10">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={pts}>
          <Line dataKey="v" stroke={color} dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 90000) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function Card({ children, className = "" }) {
  return (
    <div className={`bg-slate-900 ring-1 ring-slate-800 rounded-xl p-4 ${className}`}>
      {children}
    </div>
  );
}
