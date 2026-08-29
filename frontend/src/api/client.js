// API client. Real backend when VITE_API_URL is set (or the dev proxy answers);
// otherwise falls back to the mock so the dashboard always renders something.
// JSDoc typedefs stand in for TypeScript — same documentation value, no build
// chain changes (decision logged in README).

import axios from "axios";
import { mockApi } from "./mock.js";

/** @typedef {{device_id:string,name:string,health:string,online:boolean,
 *  anomalies_7d:number,last_seen_ts:number,sparkline:number[]}} DeviceSummary */

const base = import.meta.env.VITE_API_URL || "/api";
const http = axios.create({ baseURL: base, timeout: 8000 });

let mock = false;
export const isMock = () => mock;

let token = null;
export function setToken(t) { token = t; http.defaults.headers.Authorization = `Bearer ${t}`; }

async function tryReal(fn, fallback) {
  if (mock) return fallback();
  try {
    return await fn();
  } catch (e) {
    // Network error (backend absent) or 401 (no login page in the MVP yet):
    // fall back to demo data instead of a blank screen. A real login flow is
    // week-2 work; until then the dashboard must always render something.
    if (!e.response || e.response.status === 401) {
      mock = true;
      return fallback();
    }
    throw e;                     // other HTTP errors: surface them, don't mask bugs
  }
}

export const api = {
  /** @returns {Promise<{devices: DeviceSummary[]}>} */
  summary: () => tryReal(async () => (await http.get("/dashboard/summary")).data,
                         mockApi.summary),

  status: (id) => tryReal(async () => (await http.get(`/devices/${id}/status`)).data,
                          () => mockApi.status(id)),

  readings: (id, since = 0) =>
    tryReal(async () => (await http.get(`/devices/${id}/readings`, { params: { since } })).data,
            () => mockApi.readings(id)),

  anomalies: (id, since = 0) =>
    tryReal(async () => (await http.get(`/devices/${id}/anomalies`, { params: { since } })).data,
            () => mockApi.anomalies(id)),

  alertLog: (id) => tryReal(async () => (await http.get(`/alerts/log/${id}`)).data,
                            mockApi.alertLog),

  configureAlerts: (deviceId, body) =>
    tryReal(async () =>
      (await http.post("/alerts/configure", body, { params: { device_id: deviceId } })).data,
      mockApi.configureAlerts),

  // The false-alarm kill switch: verdict "normal" retrains the device baseline.
  feedback: (eventId, verdict) =>
    tryReal(async () =>
      (await http.post(`/anomalies/${eventId}/feedback`, { verdict })).data,
      () => mockApi.feedback(eventId, verdict)),

  registerDevice: (name) =>
    tryReal(async () => (await http.post("/devices/register", { name })).data,
            () => mockApi.registerDevice(name)),
};
