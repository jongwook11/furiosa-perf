// Shared by every page: data loading, the page router, formatting and the metrics the pages compare.
'use strict';

// ---- data ----------------------------------------------------------------------------------------
// data/index.json lists every run; data/runs/<id>.json holds one run's points (loaded on demand).
const DATA = { index: null, points: {} };

async function loadIndex() {
  if (!DATA.index) {
    const r = await fetch('data/index.json');
    if (!r.ok) throw new Error('data/index.json: HTTP ' + r.status);
    DATA.index = (await r.json()).runs;
  }
  return DATA.index;
}

// One point of a fixed-length run, with the fields the pages use. `raw` keeps every recorded value.
function toPoint(raw) {
  const out = raw['Output Throughput(tok/s)'];
  return {
    isl: raw['Input Tokens'], osl: raw['Output Tokens'], conc: raw['Concurrent'], out: out,
    ttft50: raw['Median TTFT(s)'], ttft99: raw['P99 TTFT(s)'], tpot50: raw['Median TPOT(ms)'], tpot99: raw['P99 TPOT(ms)'],
    power: raw['mean_power'],
    // Results of the presets judge each point (ok); older ones only tell a failed point by no output.
    ok: (raw.ok === undefined || !!raw.ok) && out > 0,
    raw: raw,
  };
}

async function loadPoints(runs) {
  await Promise.all(runs.filter((run) => !DATA.points[run.id]).map(async (run) => {
    const r = await fetch('data/runs/' + encodeURIComponent(run.id) + '.json');
    if (!r.ok) throw new Error(run.id + ': HTTP ' + r.status);
    const d = await r.json();
    DATA.points[run.id] = d.rows.map((row) => toPoint(Object.fromEntries(d.cols.map((c, i) => [c, row[i]]))));
  }));
}

// Points of one ISL/OSL pair, by concurrency.
function pointsAt(run, isl, osl) {
  return (DATA.points[run.id] || []).filter((p) => p.isl === isl && p.osl === osl).sort((a, b) => a.conc - b.conc);
}

// The newest run of each identical measurement (same platform, layout, task, model and server options).
function latestRuns(runs) {
  const best = {};
  runs.forEach((r) => {
    const k = [r.hw, r.n, r.layout, r.backend, r.ver, r.kind, r.task, r.model, r.variant].join('|');
    if (!best[k] || (r.date || '') > (best[k].date || '')) best[k] = r;
  });
  return Object.values(best);
}

// Fixed-length generation runs as configured (no --server-arg): the ones the Summary, Explorer and Releases pages compare.
async function generationRuns() {
  return latestRuns((await loadIndex()).filter((r) => r.kind === 'baseline' && !r.variant && GEN_TASKS[r.task]));
}
const GEN_TASKS = { 'text': 'Text', 'image-text': 'Image + text' };

function shapeOptions(runs) {
  const set = {};
  runs.forEach((r) => (r.shapes || []).forEach((s) => { set[s] = s.split('/').map(Number); }));
  return Object.keys(set).sort((a, b) => set[a][0] - set[b][0] || set[a][1] - set[b][1]);
}
function pickShape(want, options) {
  return options.indexOf(want) >= 0 ? want : options.indexOf('1024/1024') >= 0 ? '1024/1024' : options[0];
}

// ---- platforms -----------------------------------------------------------------------------------
function verKey(v) {
  const m = /^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?/.exec(v) || [];
  const pre = { a: 0, b: 1, rc: 2 }[m[4]];
  return [+m[1] || 0, +m[2] || 0, +m[3] || 0, pre === undefined ? 3 : pre, +m[5] || 0];
}
function cmpVer(a, b) {
  const x = verKey(a), y = verKey(b);
  for (let i = 0; i < 5; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
}
function family(hw) {
  if (hw.indexOf('RNGD') >= 0) return 'rngd';
  if (hw.indexOf('H200') >= 0) return 'h200';
  if (hw.indexOf('H100') >= 0) return 'h100';
  if (hw.indexOf('RTX') >= 0) return 'rtx';
  return 'other';
}
const FAMILY_ORDER = { rngd: 0, h100: 1, h200: 2, rtx: 3, other: 4 };
function famOrder(hw) { return FAMILY_ORDER[family(hw)]; }
function backendLabel(b) { return b === 'vllm' ? 'vLLM' : b; }
function platformKey(r) { return r.hw + '|' + r.backend + '|' + r.ver; }
function platformLabel(k) { const p = k.split('|'); return p[0] + ' · ' + backendLabel(p[1]) + ' ' + p[2]; }
function seriesLabel(r) { return r.hw + ' ×' + r.n + (r.layout ? ' ' + r.layout : '') + ' · ' + backendLabel(r.backend) + ' ' + r.ver; }
// Newest version measured on each device: its series get the strong colour, older ones the light one.
function newestVersions(runs) {
  const newest = {};
  runs.forEach((r) => { if (!newest[r.hw] || cmpVer(r.ver, newest[r.hw]) > 0) newest[r.hw] = r.ver; });
  return newest;
}

// ---- metrics -------------------------------------------------------------------------------------
// Power of a standard 8-card server, kW: RNGD and RTX PRO 6000 as in the published efficiency chart.
// ponytail: H100/H200 assumed 10.2 kW (HGX), replace with the official figure when there is one.
const SERVER_KW = { rngd: 3.0, rtx: 6.5, h100: 10.2, h200: 10.2 };

function okPts(pts) { return pts.filter((p) => p.ok); }
function peakPoint(pts) { return okPts(pts).reduce((b, p) => (!b || p.out > b.out ? p : b), null); }
function c1(pts) { return okPts(pts).filter((p) => p.conc === 1)[0]; }

// Largest concurrency whose TPS/user is still >= slo, interpolated between measured points.
// ge: the sweep's top concurrency still met it, so the true value is at least that.
function usersAtSlo(pts, slo) {
  const ok = okPts(pts), c = ok.map((p) => p.conc), t = ok.map((p) => p.out / p.conc);
  if (!c.length || !(slo > 0)) return null;
  if (!t.some((v) => v >= slo)) return { v: 0, ge: false };
  if (t[t.length - 1] >= slo) return { v: c[c.length - 1], ge: true };
  let best = null;
  for (let i = 0; i < c.length - 1; i++) {
    const y0 = t[i], y1 = t[i + 1];
    if (y0 !== y1 && (y0 - slo) * (y1 - slo) <= 0) {
      const x = c[i] + (slo - y0) / (y1 - y0) * (c[i + 1] - c[i]);
      best = best === null ? x : Math.max(best, x);
    }
  }
  return { v: best === null ? 0 : best, ge: false };
}

// Tokens per second per kW of an 8-card server, read off the throughput curve where TPS/user = slo.
// A curve faster than the SLO even at its highest concurrency only gives a lower bound.
function tpsPerKwAtSlo(pts, run, slo) {
  const kw = SERVER_KW[family(run.hw)];
  const c = okPts(pts).map((p) => ({ x: p.out / p.conc, y: p.out / run.n * 8 / kw })).sort((a, b) => a.x - b.x);
  if (!kw || !c.length || !(slo > 0)) return null;
  if (slo > c[c.length - 1].x) return { v: 0 };
  if (slo < c[0].x) return { v: c[0].y, ge: true };
  for (let i = 0; i < c.length - 1; i++) {
    if (c[i].x <= slo && slo <= c[i + 1].x) return { v: c[i].y + (c[i + 1].y - c[i].y) * (slo - c[i].x) / ((c[i + 1].x - c[i].x) || 1) };
  }
  return { v: c[c.length - 1].y };
}

// One number per run and ISL/OSL pair, for the Summary and Releases pages.
const METRICS = {
  tpkw: { label: 'TPS per kW at SLO', title: 'Token Throughput per Kilowatt', higher: true, perDevice: true, slo: true, f: tpsPerKwAtSlo },
  users: { label: 'Users at SLO / device', higher: true, perDevice: true, slo: true, f: (pts, r, slo) => { const u = usersAtSlo(pts, slo); return u ? { v: u.v / r.n, ge: u.ge } : null; } },
  peak: { label: 'Peak output TPS / device', higher: true, perDevice: true, f: (pts, r) => { const p = peakPoint(pts); return p ? { v: p.out / r.n } : null; } },
  tpw: { label: 'Output TPS / W at peak', higher: true, perDevice: true, f: (pts) => { const p = peakPoint(pts); return p && p.power > 0 ? { v: p.out / p.power } : null; } },
  speed1: { label: 'TPS / User at c1', higher: true, f: (pts) => { const p = c1(pts); return p ? { v: p.out } : null; } },
  tpot1: { label: 'TPOT p50 at c1 (ms)', higher: false, f: (pts) => { const p = c1(pts); return p ? { v: p.tpot50 } : null; } },
  ttft1: { label: 'TTFT p50 at c1 (s)', higher: false, f: (pts) => { const p = c1(pts); return p ? { v: p.ttft50 } : null; } },
};

function median(xs) {
  xs = xs.slice().sort((a, b) => a - b);
  if (!xs.length) return null;
  return xs.length % 2 ? xs[(xs.length - 1) / 2] : (xs[xs.length / 2 - 1] + xs[xs.length / 2]) / 2;
}

// ---- formatting ----------------------------------------------------------------------------------
function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000 || Number.isInteger(v)) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
function day(iso) { return iso ? iso.slice(0, 10) : ''; }
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// <select> bound to state[key]; options are strings or {v, label}.
function select(label, key, value, options, attrs) {
  const opts = options.map((o) => {
    const v = typeof o === 'object' ? o.v : o, l = typeof o === 'object' ? o.label : o;
    return '<option value="' + esc(v) + '"' + (String(v) === String(value) ? ' selected' : '') + '>' + esc(l) + '</option>';
  }).join('');
  return '<label class="ctl">' + esc(label) + '<select data-k="' + key + '"' + (attrs || '') + '>' + opts + '</select></label>';
}
function sloInput(slo) {
  return '<label class="ctl">SLO · tok/s/user<input type="number" min="1" step="1" data-k="slo" data-num value="' + slo + '"></label>';
}
function kpi(label, value, color) {
  return '<div class="kpi"><div class="label">' + esc(label) + '</div><div class="value"' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</div></div>';
}

// ---- router --------------------------------------------------------------------------------------
// The page and its controls live in the URL (#explorer?model=…&scen=…), so a view can be linked.
// Pages register an async render(state) that returns HTML; changing a control re-renders the page.
const PAGES = {};
let STATE = {}, PAGE = 'summary', RENDERING = 0;

function readHash() {
  const [page, query] = location.hash.replace(/^#/, '').split('?');
  PAGE = PAGES[page] ? page : 'summary';
  STATE = Object.fromEntries(new URLSearchParams(query || ''));
  ['slo'].forEach((k) => { if (STATE[k] !== undefined) STATE[k] = +STATE[k]; });
}
function setState(patch) {
  Object.assign(STATE, patch);
  Object.keys(STATE).forEach((k) => { if (STATE[k] === undefined || STATE[k] === null) delete STATE[k]; });
  const q = new URLSearchParams(STATE).toString();
  history.replaceState(null, '', '#' + PAGE + (q ? '?' + q : ''));
  render();
}
async function render() {
  const ticket = ++RENDERING, app = document.getElementById('app');
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + PAGE));
  try {
    const html = await PAGES[PAGE](STATE);
    if (ticket === RENDERING) app.innerHTML = html;  // a newer render started while loading: drop this one
  } catch (e) {
    if (ticket === RENDERING) app.innerHTML = '<div class="status">Could not load the data: ' + esc(e.message) + '</div>';
  }
}
function start() {
  const app = document.getElementById('app');
  app.addEventListener('change', (e) => {
    const el = e.target;
    if (el.dataset.k) setState({ [el.dataset.k]: el.dataset.num !== undefined ? (+el.value > 0 ? +el.value : undefined) : el.value });
    else if (el.dataset.toggle !== undefined) PAGES[PAGE].toggle(el.dataset.toggle, el.checked);
  });
  window.addEventListener('hashchange', () => { readHash(); render(); });
  readHash();
  render();
}
