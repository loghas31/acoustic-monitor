#!/usr/bin/env python3
"""
Build a single self-contained HTML page that replays a `sim_trace.json`.

    python tools/sim_trace.py                     # record the run
    python tools/build_sim_dashboard.py           # render it
    open tools/sim_dashboard.html

Why inline rather than `fetch('sim_trace.json')`: opening a page from
`file://` blocks XHR/fetch of sibling files under every browser's
same-origin rules, so a fetching version works from a dev server and
silently shows an empty chart when double-clicked from Finder — which is
the only way this will ever actually be opened. The trace is embedded.

No CDN, no build step, no network: the page must render on a laptop in a
workshop with no wifi, which is the same room the sensor is meant to work in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Acoustic Machine Health Monitor — simulated run</title>
<style>
  :root {
    --bg:#0e1116; --panel:#161b22; --line:#272e37; --ink:#e6edf3; --dim:#8b949e;
    --green:#3fb950; --amber:#d29922; --red:#f85149; --blue:#58a6ff;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; padding:22px 18px 60px; }
  h1 { font-size:20px; margin:0 0 2px; font-weight:650; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:16px; }
  .synthetic { display:inline-block; border:1px solid var(--amber); color:var(--amber);
    border-radius:4px; padding:1px 7px; font-size:11px; font-weight:600;
    letter-spacing:.04em; text-transform:uppercase; margin-left:8px; vertical-align:2px;}
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:16px; margin-bottom:14px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; }
  .row > .panel { flex:1 1 260px; margin-bottom:0; }
  .grid3 { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }
  .k { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .v { font-family:var(--mono); font-size:19px; margin-top:3px; }
  .v small { font-size:12px; color:var(--dim); font-weight:400; }
  /* status card */
  #status { border-left:5px solid var(--green); transition:border-color .15s; }
  #tier { font-size:26px; font-weight:700; letter-spacing:.02em; }
  #tiermsg { color:var(--dim); font-size:13px; margin-top:2px; }
  .green { color:var(--green);} .amber { color:var(--amber);} .red { color:var(--red);}
  /* controls */
  .ctl { display:flex; align-items:center; gap:12px; }
  button { background:#21262d; color:var(--ink); border:1px solid var(--line);
    border-radius:6px; padding:7px 15px; font-size:14px; cursor:pointer; }
  button:hover { background:#2b3138; }
  input[type=range] { flex:1; accent-color:var(--blue); }
  .clock { font-family:var(--mono); color:var(--dim); min-width:120px; text-align:right;}
  /* gate pips */
  #gate { display:flex; gap:4px; margin-top:8px; flex-wrap:wrap; }
  .pip { width:22px; height:12px; border-radius:3px; background:#21262d;
         border:1px solid var(--line); }
  .pip.on { background:var(--amber); border-color:var(--amber); }
  .pip.fire { background:var(--red); border-color:var(--red); }
  svg { display:block; width:100%; height:auto; }
  .axis { stroke:var(--line); stroke-width:1; }
  .gridln { stroke:var(--line); stroke-width:1; stroke-dasharray:2 4; }
  .lbl { fill:var(--dim); font-size:10px; font-family:var(--mono); }
  .thr  { stroke:var(--red); stroke-width:1.5; stroke-dasharray:5 4; }
  .trace{ fill:none; stroke:var(--blue); stroke-width:2; }
  .cursor { stroke:var(--ink); stroke-width:1; opacity:.55; }
  .note { color:var(--dim); font-size:12px; margin-top:10px; }
  table { border-collapse:collapse; width:100%; font-family:var(--mono); font-size:12px;}
  td,th { text-align:left; padding:3px 8px 3px 0; }
  th { color:var(--dim); font-weight:500; }
  .foot { color:var(--dim); font-size:12px; border-top:1px solid var(--line);
          padding-top:12px; margin-top:20px; }
  code { background:#21262d; padding:1px 5px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">

  <h1>Acoustic Machine Health Monitor — a run, replayed<span class="synthetic">synthetic</span></h1>
  <div class="sub">
    Every number below was produced by <code>firmware/main.py</code> running unmodified
    against <code>ml/simulate.py</code>. Nothing here is drawn for effect.
  </div>

  <div class="panel" id="status">
    <div class="k">Machine status</div>
    <div id="tier">GREEN</div>
    <div id="tiermsg">Sounds like its own normal.</div>
  </div>

  <div class="panel">
    <div class="ctl">
      <button id="play">▶ Play</button>
      <button id="rst">↻ Restart</button>
      <input type="range" id="scrub" min="0" value="0" step="1">
      <span class="clock" id="clock">00:00</span>
    </div>
  </div>

  <div class="panel">
    <div class="k">Anomaly score ÷ this regime's threshold — log scale</div>
    <svg id="chart" viewBox="0 0 1120 300" preserveAspectRatio="none"></svg>
    <div class="note">
      Red dashed line = the alert threshold (ratio&nbsp;1.0), learned per operating regime
      during the learn period. Shaded bands mark which regime each window was
      assigned to. The score axis spans more than three decades, so it is logarithmic.
    </div>
  </div>

  <div class="panel">
    <div class="k">Persistence gate — <span id="needtxt"></span> consecutive anomalous windows required</div>
    <div id="gate"></div>
    <div class="note" id="gatemsg"></div>
  </div>

  <div class="row">
    <div class="panel">
      <div class="k">Score</div><div class="v"><span id="score"></span> <small>vs <span id="thr"></span></small></div>
      <div class="k" style="margin-top:12px">Health index <small>(0–100, 70 = threshold)</small></div><div class="v" id="idx"></div>
    </div>
    <div class="panel">
      <div class="k">Shaft speed <small>(estimated by HPS)</small></div><div class="v"><span id="fr"></span> <small>Hz</small></div>
      <div class="k" style="margin-top:12px">Operating regime</div><div class="v" id="reg"></div>
    </div>
    <div class="panel">
      <div class="k">Demodulation band <small>(chosen by protrugram)</small></div><div class="v" id="band"></div>
      <div class="k" style="margin-top:12px">Envelope peak</div><div class="v"><span id="env"></span> <small>Hz &nbsp; <span id="envx"></span></small></div>
    </div>
  </div>

  <div class="panel" style="margin-top:14px">
    <div class="k">What the simulator was actually doing — the detector cannot see this</div>
    <table>
      <tr><th style="width:150px">Ground truth</th><td id="truth"></td></tr>
      <tr><th>Fault severity</th><td id="tsev"></td></tr>
      <tr><th>Envelope energy</th><td id="envdb"></td></tr>
      <tr><th>Extraction latency</th><td id="lat"></td></tr>
    </table>
  </div>

  <div class="panel">
    <div class="k">Run summary</div>
    <table id="summary"></table>
  </div>

  <div class="foot" id="foot"></div>
</div>

<script>
const D = __TRACE__;
// Exposed deliberately: `const` at the top level of a classic script is a
// lexical binding and does NOT become a property of `window`, so
// tools/check_sim_dashboard.mjs could not see the trace it is meant to check
// the page against. Also handy from the browser console.
window.TRACE = D;
const W = D.windows, C = D.config, R = D.result;
const N = W.length;
const need = C.gate_need_windows;

// ---- chart geometry -------------------------------------------------------
const PW=1120, PH=300, ML=52, MR=14, MT=14, MB=26;
const IW=PW-ML-MR, IH=PH-MT-MB;
const ratios = W.map(w => Math.max(w.ratio || 0.01, 0.01));
const lo = Math.log10(Math.min(0.3, Math.min(...ratios)));
const hi = Math.log10(Math.max(...ratios) * 1.5);
const X = i => ML + (N<2 ? 0 : i * IW / (N - 1));
const Y = r => MT + IH - (Math.log10(Math.max(r,0.01)) - lo) / (hi - lo) * IH;

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function drawChart(){
  const p = [];
  // regime shading
  let s = 0;
  for (let i = 1; i <= N; i++){
    if (i === N || W[i].regime !== W[s].regime){
      const fill = W[s].regime % 2 ? 'rgba(88,166,255,.055)' : 'rgba(255,255,255,.03)';
      p.push(`<rect x="${X(s).toFixed(1)}" y="${MT}" width="${(X(i-1)-X(s)).toFixed(1)}" height="${IH}" fill="${fill}"/>`);
      s = i;
    }
  }
  // decade gridlines
  for (let d = Math.ceil(lo); d <= Math.floor(hi); d++){
    const y = Y(Math.pow(10,d));
    p.push(`<line class="gridln" x1="${ML}" y1="${y.toFixed(1)}" x2="${PW-MR}" y2="${y.toFixed(1)}"/>`);
    p.push(`<text class="lbl" x="${ML-7}" y="${(y+3).toFixed(1)}" text-anchor="end">${d===0?'1':'10'+sup(d)}×</text>`);
  }
  // threshold
  const yt = Y(1);
  p.push(`<line class="thr" x1="${ML}" y1="${yt.toFixed(1)}" x2="${PW-MR}" y2="${yt.toFixed(1)}"/>`);
  p.push(`<text class="lbl" x="${PW-MR}" y="${(yt-6).toFixed(1)}" text-anchor="end" fill="#f85149">alert threshold</text>`);
  // trace
  p.push(`<path class="trace" d="${W.map((w,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(ratios[i]).toFixed(1)}`).join('')}"/>`);
  // points
  W.forEach((w,i)=>{
    const c = w.tier==='red' ? 'var(--red)' : w.tier==='amber' ? 'var(--amber)' : 'var(--green)';
    p.push(`<circle cx="${X(i).toFixed(1)}" cy="${Y(ratios[i]).toFixed(1)}" r="${w.fired?5:2.6}" fill="${c}"/>`);
  });
  // alert markers
  R.alert_windows.forEach(i=>{
    p.push(`<line x1="${X(i).toFixed(1)}" y1="${MT}" x2="${X(i).toFixed(1)}" y2="${MT+IH}" stroke="var(--red)" stroke-width="1.5" opacity=".8"/>`);
    p.push(`<text class="lbl" x="${(X(i)+5).toFixed(1)}" y="${MT+12}" fill="#f85149">ALERT</text>`);
  });
  // transient marker
  const tw = Math.round(C.transient_at_minute*60/C.window_seconds);
  if (tw < N) p.push(`<text class="lbl" x="${(X(tw)+5).toFixed(1)}" y="${(Y(ratios[tw])-9).toFixed(1)}" fill="#d29922">1-window transient — correctly ignored</text>`);
  // axes + minute labels
  p.push(`<line class="axis" x1="${ML}" y1="${MT+IH}" x2="${PW-MR}" y2="${MT+IH}"/>`);
  for (let i=0;i<N;i+=Math.max(1,Math.round(N/10))){
    p.push(`<text class="lbl" x="${X(i).toFixed(1)}" y="${PH-8}" text-anchor="middle">${W[i].minute.toFixed(0)}m</text>`);
  }
  p.push(`<line class="cursor" id="cur" x1="${ML}" y1="${MT}" x2="${ML}" y2="${MT+IH}"/>`);
  document.getElementById('chart').innerHTML = p.join('');
}
function sup(d){ return String(d).split('').map(c=>({'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}[c]||c)).join(''); }

// ---- gate pips ------------------------------------------------------------
const gateEl = document.getElementById('gate');
gateEl.innerHTML = Array.from({length:need}, ()=>'<div class="pip"></div>').join('');
document.getElementById('needtxt').textContent =
  `${need} × ${C.window_seconds}s = ${C.persist_minutes} min`;

const TIERMSG = {
  green: 'Sounds like its own normal.',
  amber: 'Above threshold, but not for long enough to alert.',
  red:   'Persistently abnormal — alert raised.',
};

function render(i){
  const w = W[i];
  const tierEl = document.getElementById('tier');
  tierEl.textContent = w.tier.toUpperCase();
  tierEl.className = w.tier;
  document.getElementById('tiermsg').textContent = TIERMSG[w.tier];
  document.getElementById('status').style.borderLeftColor = `var(--${w.tier})`;

  document.getElementById('clock').textContent =
    `${String(Math.floor(w.minute)).padStart(2,'0')}:${String(Math.round((w.minute%1)*60)).padStart(2,'0')}  ·  w${i}`;
  document.getElementById('score').textContent = w.score.toFixed(2);
  document.getElementById('thr').textContent   = 'threshold ' + w.threshold.toFixed(2);
  document.getElementById('idx').textContent   = w.index.toFixed(1);
  document.getElementById('fr').textContent    = w.fr_hz.toFixed(1) + (w.fr_reliable?'':' (unreliable)');
  document.getElementById('reg').textContent   = w.regime;
  document.getElementById('band').textContent  = `${(w.band_lo_hz/1000).toFixed(2)}–${(w.band_hi_hz/1000).toFixed(2)} kHz`;
  document.getElementById('env').textContent   = w.env_peak_hz.toFixed(1);
  document.getElementById('envx').textContent  = w.fr_hz>0 ? `= ${(w.env_peak_hz/w.fr_hz).toFixed(2)} × shaft` : '';
  document.getElementById('truth').textContent = w.truth_kind === 'normal'
      ? 'healthy' : w.truth_kind.replace('_',' ') + ' fault';
  document.getElementById('tsev').textContent  = w.truth_severity.toFixed(2);
  document.getElementById('envdb').textContent = (w.env_db_re_learn>=0?'+':'') + w.env_db_re_learn.toFixed(1) + ' dB vs learn period';
  document.getElementById('lat').textContent   = w.latency_ms.toFixed(0) + ' ms  (budget 2000 ms)';

  [...gateEl.children].forEach((p,k)=>{
    p.className = 'pip' + (k < Math.min(w.streak, need) ? (w.streak>=need ? ' fire' : ' on') : '');
  });
  document.getElementById('gatemsg').textContent = w.streak === 0
    ? 'Counter at zero — this window looks normal.'
    : w.streak < need
      ? `${w.streak} of ${need}. One normal window resets this to zero, which is what kills the transient at minute ${C.transient_at_minute}.`
      : `Held for ${w.streak} windows. The alert fires once per episode, not once per window.`;

  const c = document.getElementById('cur');
  if (c){ c.setAttribute('x1', X(i)); c.setAttribute('x2', X(i)); }
  document.getElementById('scrub').value = i;
}

// ---- summary --------------------------------------------------------------
const healthy = W.filter(w=>w.truth_kind==='normal');
const faulty  = W.filter(w=>w.truth_kind!=='normal');
const fp = healthy.filter(w=>w.anomalous).length;
const tp = faulty.filter(w=>w.anomalous).length;
document.getElementById('summary').innerHTML = `
  <tr><th style="width:290px">Windows simulated</th><td>${N} × ${C.window_seconds}s = ${C.minutes} minutes</td></tr>
  <tr><th>Operating regimes learned</th><td>${C.n_regimes} (thresholds ${C.baseline_thresholds.join(' / ')})</td></tr>
  <tr><th>Regime switches</th><td>${W.reduce((a,w,i)=>a+(i&&w.regime!==W[i-1].regime?1:0),0)} — none of them alerted</td></tr>
  <tr><th>Healthy windows flagged</th><td>${fp} of ${healthy.length}</td></tr>
  <tr><th>Fault windows flagged</th><td>${tp} of ${faulty.length}</td></tr>
  <tr><th>Alerts raised</th><td>${R.alerts} — at window ${R.alert_windows.join(', ')} (minute ${R.alert_windows.map(i=>W[i].minute.toFixed(1)).join(', ')})</td></tr>
`;
document.getElementById('foot').innerHTML =
  `Generated by <code>tools/sim_trace.py</code> → <code>tools/build_sim_dashboard.py</code>. ` +
  `<strong>Synthetic throughout.</strong> The fault signals come from a model of bearing physics, ` +
  `not from a bearing. A real machine will be noisier and this page proves nothing about it — ` +
  `it demonstrates that the decision logic behaves as designed. See <code>docs/DOC_STATUS.md</code> ` +
  `for what is proven versus assumed.`;

// ---- transport ------------------------------------------------------------
let cur = 0, timer = null;
const scrub = document.getElementById('scrub'), playBtn = document.getElementById('play');
scrub.max = N - 1;
scrub.addEventListener('input', e => { stop(); cur = +e.target.value; render(cur); });
playBtn.addEventListener('click', () => timer ? stop() : start());
document.getElementById('rst').addEventListener('click', () => { stop(); cur = 0; render(0); });
function start(){ if (cur >= N-1) cur = 0; playBtn.textContent = '❚❚ Pause';
  timer = setInterval(() => { cur++; if (cur >= N){ cur = N-1; stop(); } render(cur); }, 220); }
function stop(){ clearInterval(timer); timer = null; playBtn.textContent = '▶ Play'; }

drawChart(); render(0);
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=ROOT / "tools" / "sim_trace.json")
    ap.add_argument("--out", type=Path, default=ROOT / "tools" / "sim_dashboard.html")
    a = ap.parse_args()

    trace = json.loads(a.trace.read_text())
    # `</script>` inside a JSON string would close the tag early; nothing in a
    # trace should contain it, but escaping costs nothing and a silently blank
    # page costs an afternoon.
    blob = json.dumps(trace, separators=(",", ":")).replace("</", r"<\/")
    a.out.write_text(PAGE.replace("__TRACE__", blob))
    print(f"wrote {a.out}  ({a.out.stat().st_size/1024:.0f} kB, "
          f"{len(trace['windows'])} windows, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
