import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview.jsx";
import DeviceDetail from "./pages/DeviceDetail.jsx";
import AlertConfig from "./pages/AlertConfig.jsx";
import Onboarding from "./pages/Onboarding.jsx";
import { isMock } from "./api/client.js";

const tab = ({ isActive }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium ${
    isActive ? "bg-slate-700 text-white" : "text-slate-400 hover:text-white"}`;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 px-6 py-3 flex items-center gap-6">
        <h1 className="text-lg font-semibold text-white">
          <span className="text-emerald-400">●</span> Acoustic Monitor
        </h1>
        <nav className="flex gap-2">
          <NavLink to="/" className={tab} end>Fleet</NavLink>
          <NavLink to="/alerts" className={tab}>Alerts</NavLink>
          <NavLink to="/onboarding" className={tab}>Add a machine</NavLink>
        </nav>
        {isMock() && (
          <span className="ml-auto text-xs bg-amber-900/60 text-amber-300 px-2 py-1 rounded">
            demo mode — simulated data (backend not connected)
          </span>
        )}
      </header>
      <main className="p-6 max-w-7xl mx-auto">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/device/:id" element={<DeviceDetail />} />
          <Route path="/alerts" element={<AlertConfig />} />
          <Route path="/onboarding" element={<Onboarding />} />
        </Routes>
      </main>
    </div>
  );
}
