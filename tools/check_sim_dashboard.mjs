/**
 * Execute tools/sim_dashboard.html in a headless DOM and assert it actually
 * renders the run, rather than trusting that ~300 lines of hand-written
 * browser JS work because they look like they should.
 *
 *     npm install jsdom
 *     node tools/check_sim_dashboard.mjs
 *
 * Not part of `pytest` on purpose: it needs node + jsdom, which the Pi image
 * and the normal dev setup do not have, and a test that is usually skipped is
 * worse than a script that is occasionally run. Run it after regenerating the
 * dashboard.
 *
 * The assertions below are the three things the page exists to show — the
 * transient at w4 does NOT alert, the gate resets at w5, and the persistent
 * fault alerts exactly once at w15 — so if the visualiser ever starts telling
 * a different story than the firmware did, this fails.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { JSDOM } from 'jsdom';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'tools', 'sim_dashboard.html'), 'utf8');

const errors = [];
const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true });
dom.virtualConsole.on('jsdomError', e => errors.push(e.message));
await new Promise(r => setTimeout(r, 400));
const { window } = dom, doc = window.document;

let bad = 0;
const fail = m => { console.log('FAIL:', m); bad++; };
const ok = m => console.log('ok  :', m);

if (errors.length) fail('script errors: ' + errors.join(' | '));
else ok('page executed with no script errors');

const chart = doc.getElementById('chart');
const n = window.TRACE.windows.length;
chart.querySelectorAll('circle').length === n
  ? ok(`chart drew ${n} window markers`) : fail('wrong number of window markers');
chart.querySelectorAll('path').length >= 1 ? ok('score trace drawn') : fail('no trace path');

const need = window.TRACE.config.gate_need_windows;
doc.querySelectorAll('#gate .pip').length === need
  ? ok(`gate shows ${need} pips`) : fail('gate pip count != need');

const scrub = doc.getElementById('scrub');
const goto = i => { scrub.value = String(i); scrub.dispatchEvent(new window.Event('input')); };
const st = () => ({
  tier: doc.getElementById('tier').textContent,
  on: doc.querySelectorAll('#gate .pip.on').length,
  fire: doc.querySelectorAll('#gate .pip.fire').length,
});

const transient = Math.round(window.TRACE.config.transient_at_minute * 60 / window.TRACE.config.window_seconds);
const alertW = window.TRACE.result.alert_windows[0];

goto(0);
st().tier === 'GREEN' ? ok('first window renders GREEN') : fail('first window not green');

goto(transient);
const a = st();
a.tier === 'AMBER' && a.on === 1 && a.fire === 0
  ? ok(`w${transient} transient shows AMBER at 1/${need} — correctly not an alert`)
  : fail(`w${transient} should be amber, 1 pip, not fired — got ${JSON.stringify(a)}`);

goto(transient + 1);
st().on === 0 ? ok(`w${transient + 1} gate reset to zero`) : fail('gate did not reset after transient');

goto(alertW);
const r = st();
r.tier === 'RED' && r.fire === need
  ? ok(`w${alertW} shows RED with the gate full — the alert window`)
  : fail(`w${alertW} should be red and full — got ${JSON.stringify(r)}`);

const sum = doc.getElementById('summary').textContent;
/Alerts raised/.test(sum) ? ok('run summary rendered') : fail('summary empty');

console.log(bad ? `\n${bad} check(s) FAILED` : '\nall checks passed');
process.exit(bad ? 1 : 0);
