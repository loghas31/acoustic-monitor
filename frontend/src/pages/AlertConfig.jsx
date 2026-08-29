import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import { Card } from "../components/widgets.jsx";

export default function AlertConfig() {
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState("");
  const [email, setEmail] = useState("");
  const [webhook, setWebhook] = useState("");
  const [saved, setSaved] = useState(false);
  // No sensitivity slider in v1, deliberately: thresholds are CV-calibrated
  // per machine and per regime during the learn period; the customer's lever
  // on false alarms is the "this was normal" button, which retrains rather
  // than blunting detection globally.

  useEffect(() => {
    api.summary().then((d) => {
      setDevices(d.devices);
      if (d.devices[0]) setDeviceId(d.devices[0].device_id);
    });
  }, []);

  async function save() {
    await api.configureAlerts(deviceId, {
      email: email || null,
      webhook_url: webhook || null,
      enabled: true,
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  return (
    <div className="max-w-xl space-y-4">
      <h2 className="text-xl font-semibold text-white">Alert configuration</h2>
      <Card className="space-y-4">
        <label className="block text-sm">
          <span className="text-slate-400">Machine</span>
          <select value={deviceId} onChange={(e) => setDeviceId(e.target.value)}
                  className="mt-1 w-full bg-slate-800 rounded-md px-3 py-2 ring-1 ring-slate-700">
            {devices.map((d) => <option key={d.device_id} value={d.device_id}>{d.name}</option>)}
          </select>
        </label>

        <label className="block text-sm">
          <span className="text-slate-400">Alert email</span>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                 placeholder="maintenance@yourcompany.co.uk"
                 className="mt-1 w-full bg-slate-800 rounded-md px-3 py-2 ring-1 ring-slate-700" />
        </label>

        <label className="block text-sm">
          <span className="text-slate-400">Webhook URL (Slack / Teams / PagerDuty)</span>
          <input value={webhook} onChange={(e) => setWebhook(e.target.value)}
                 placeholder="https://hooks.slack.com/services/…"
                 className="mt-1 w-full bg-slate-800 rounded-md px-3 py-2 ring-1 ring-slate-700" />
        </label>

        <button onClick={save}
                className="bg-sky-500 hover:bg-sky-400 text-slate-950 font-medium px-4 py-2 rounded-md">
          {saved ? "Saved ✓" : "Save"}
        </button>
      </Card>
    </div>
  );
}
