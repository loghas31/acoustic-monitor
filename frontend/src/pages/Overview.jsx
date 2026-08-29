import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client.js";
import { Card, Sparkline, StatusBadge, timeAgo } from "../components/widgets.jsx";
import { sparklineCaption } from "../lib/trend.js";

const SPARK_COLOR = { green: "#34d399", amber: "#fbbf24", red: "#fb7185", unknown: "#64748b" };

export default function Overview() {
  const [devices, setDevices] = useState(null);

  useEffect(() => {
    api.summary().then((d) => setDevices(d.devices));
    const t = setInterval(() => api.summary().then((d) => setDevices(d.devices)), 30000);
    return () => clearInterval(t);
  }, []);

  if (!devices) return <p className="text-slate-500">Loading fleet…</p>;

  const red = devices.filter((d) => d.health === "red").length;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h2 className="text-xl font-semibold text-white">Fleet overview</h2>
        <p className="text-sm text-slate-400">
          {devices.length} machines ·{" "}
          {red ? <span className="text-rose-400">{red} need attention</span> : "all nominal"}
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {devices.map((d) => (
          <Link key={d.device_id} to={`/device/${d.device_id}`}>
            <Card className="hover:ring-slate-600 transition">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-white">{d.name}</h3>
                <StatusBadge health={d.health} />
              </div>
              <Sparkline data={d.sparkline} color={SPARK_COLOR[d.health] ?? "#64748b"} />
              <div className="flex justify-between text-xs text-slate-500 mt-2">
                <span>{d.online ? "online" : "offline"} · seen {timeAgo(d.last_seen_ts)}</span>
                {/* T1.11: the sparkline is now the display index (70 = this
                    machine's alert threshold, in every operating regime), not
                    the raw Mahalanobis score, whose scale differs per regime
                    and per machine. Old-firmware units still send only the
                    score, and the backend says which one it gave us — so the
                    label is derived, never assumed. */}
                <span>{sparklineCaption(d)}</span>
              </div>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
