// 4-step onboarding. The target persona has no IT team — every step is one
// action, in plain language, with the "what's happening" explained in a line.

import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client.js";
import { Card } from "../components/widgets.jsx";

const STEPS = ["Plug in", "Name it", "Learn normal", "Done"];

export default function Onboarding() {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [device, setDevice] = useState(null);

  async function register() {
    setDevice(await api.registerDevice(name || "My machine"));
    setStep(2);
  }

  return (
    <div className="max-w-xl mx-auto">
      <ol className="flex gap-2 mb-6">
        {STEPS.map((s, i) => (
          <li key={s} className={`flex-1 text-center text-xs py-2 rounded-md ${
              i < step ? "bg-emerald-500/15 text-emerald-400"
              : i === step ? "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/40"
              : "bg-slate-900 text-slate-500"}`}>
            {i + 1}. {s}
          </li>
        ))}
      </ol>

      <Card className="space-y-4">
        {step === 0 && (
          <>
            <h3 className="text-lg font-medium text-white">Stick the sensor on your machine</h3>
            <p className="text-sm text-slate-400">
              The base is magnetic — place it on a flat metal surface near a bearing
              (motor end-shield is ideal). Plug the USB-C cable into mains. The light
              turns blue when it finds your Wi-Fi (set up via the QR card in the box).
            </p>
            <button onClick={() => setStep(1)}
                    className="bg-sky-500 hover:bg-sky-400 text-slate-950 font-medium px-4 py-2 rounded-md">
              The light is blue →
            </button>
          </>
        )}

        {step === 1 && (
          <>
            <h3 className="text-lg font-medium text-white">What machine is this?</h3>
            <input autoFocus value={name} onChange={(e) => setName(e.target.value)}
                   placeholder="e.g. Compressor A"
                   className="w-full bg-slate-800 rounded-md px-3 py-2 ring-1 ring-slate-700" />
            <button onClick={register} disabled={!name.trim()}
                    className="bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-slate-950 font-medium px-4 py-2 rounded-md">
              Register device →
            </button>
          </>
        )}

        {step === 2 && (
          <>
            <h3 className="text-lg font-medium text-white">Learning what “normal” sounds like</h3>
            <p className="text-sm text-slate-400">
              Run the machine as you normally would for the next 24–72 hours. The
              sensor is building a baseline of its healthy sound and vibration.
              You'll get no alerts during this period — that's expected.
            </p>
            <p className="text-xs text-slate-500">
              Device ID: <code className="text-slate-300">{device?.device_id}</code> — keep the
              API key from your welcome email safe; it's shown only once.
            </p>
            <button onClick={() => setStep(3)}
                    className="bg-sky-500 hover:bg-sky-400 text-slate-950 font-medium px-4 py-2 rounded-md">
              Start learning →
            </button>
          </>
        )}

        {step === 3 && (
          <>
            <h3 className="text-lg font-medium text-white">You're protected 🎉</h3>
            <p className="text-sm text-slate-400">
              When the baseline is ready, the machine appears green on your fleet page.
              If its sound ever drifts from normal, you'll get an email within 90
              seconds — typically days or weeks before a failure you could hear yourself.
            </p>
            <Link to="/" className="inline-block bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-medium px-4 py-2 rounded-md">
              Go to fleet overview
            </Link>
          </>
        )}
      </Card>
    </div>
  );
}
