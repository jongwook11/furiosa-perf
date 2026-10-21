// Explorer: any two metrics of one model's runs plotted against each other, one curve per run.
'use strict';

const AXES = {
  tps_user: { label: () => 'TPS / User', axis: 'TPS per user (interactivity)', f: (p) => p.out / p.conc },
  out: { label: (norm) => (norm === 'dev' ? 'Output TPS / Device' : 'Output TPS / System'), f: (p, r, norm) => (norm === 'dev' ? p.out / r.n : p.out) },
  tps_kw: { label: () => 'Tokens per second (TPS) per kW', title: 'Token Throughput per Kilowatt', f: (p, r) => { const kw = SERVER_KW[family(r.hw)]; return kw ? p.out / r.n * 8 / kw : null; } },
  conc: { label: () => 'Concurrency', f: (p) => p.conc },
  ttft50: { label: () => 'TTFT p50 (s)', f: (p) => p.ttft50 },
  ttft99: { label: () => 'TTFT p99 (s)', f: (p) => p.ttft99 },
  tpot50: { label: () => 'TPOT p50 (ms)', f: (p) => p.tpot50 },
  tpot99: { label: () => 'TPOT p99 (ms)', f: (p) => p.tpot99 },
  tpw: { label: () => 'Output TPS / W', f: (p) => (p.power > 0 ? p.out / p.power : null) },
  power: { label: (norm) => (norm === 'dev' ? 'Power / Device (W)' : 'Power / System (W)'), f: (p, r, norm) => (norm === 'dev' ? p.power / r.n : p.power) },
};
// Y metrics where a taller RNGD curve means RNGD is better, so a ×N multiplier reads naturally.
const HIGHER_IS_BETTER = { out: 1, tps_kw: 1, tpw: 1 };
const SHORT_NAME = { rngd: 'FURIOSA RNGD', h100: 'NVIDIA H100', h200: 'NVIDIA H200', rtx: 'NVIDIA RTX PRO 6000' };
const W = 900, H = 490, L = 70, R = 880, T = 20, B = 440;  // chart viewBox and plot area

function dashOf(n) { return n === 1 ? 'none' : n === 2 ? '8 5' : n === 4 ? '2 4' : '12 4 2 4'; }
function niceTicks(max) {
  if (!(max > 0)) return { top: 1, list: [0, 1] };
  const raw = max / 6, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p, top = Math.ceil(max / step) * step, list = [];
  for (let v = 0; v <= top + step / 1000; v += step) list.push(+v.toFixed(6));
  return { top: top, list: list };
}
function smoothPath(pts) {
  let d = 'M ' + pts[0].x.toFixed(1) + ' ' + pts[0].y.toFixed(1);
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    d += ' C ' + [p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6, p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6, p2.x, p2.y].map((n) => n.toFixed(1)).join(' ');
  }
  return d;
}
function yAtX(vals, x) {
  const s = vals.slice().sort((a, b) => a.x - b.x);
  for (let i = 0; i < s.length - 1; i++) if (s[i].x <= x && x <= s[i + 1].x) return s[i].y + (s[i + 1].y - s[i].y) * (x - s[i].x) / ((s[i + 1].x - s[i].x) || 1);
  return null;
}

let SHOWN = [];  // run ids drawn in the last render, for the series checkboxes

PAGES.explorer = async function (st) {
  const gen = await generationRuns();
  const models = [...new Set(gen.map((r) => r.model))].sort((a, b) => a.localeCompare(b));
  const model = models.indexOf(st.model) >= 0 ? st.model : models[0];
  const mRuns = gen.filter((r) => r.model === model);
  const tasks = Object.keys(GEN_TASKS).filter((t) => mRuns.some((r) => r.task === t));
  const task = tasks.indexOf(st.task) >= 0 ? st.task : tasks[0];
  const tRuns = mRuns.filter((r) => r.task === task);
  const scens = scenarioOptions(tRuns);
  const scen = pickScenario(st.scen, scens);
  const norm = st.norm === 'sys' ? 'sys' : 'dev';
  const xKey = AXES[st.x] ? st.x : 'tps_user', yKey = AXES[st.y] ? st.y : 'tps_kw';
  const slo = st.slo > 0 ? st.slo : 20, showFailed = st.failed === '1';
  if (!scen) return '<div class="status">No fixed-length generation results.</div>';
  const [isl, osl] = scen.split('/').map(Number);

  await loadPoints(tRuns);
  const all = tRuns.map((r) => ({ r: r, pts: pointsAt(r, isl, osl), label: seriesLabel(r), dash: dashOf(r.n) }))
    .filter((o) => o.pts.length)
    .sort((a, b) => famOrder(a.r.hw) - famOrder(b.r.hw) || a.r.n - b.r.n || cmpVer(b.r.ver, a.r.ver));
  // The newest version measured here gets the strong colour, older ones the light one.
  const newestHere = newestVersions(all.map((o) => o.r));
  all.forEach((o) => { o.color = 'var(--c-' + family(o.r.hw) + (newestHere[o.r.hw] === o.r.ver ? '-a' : '-b') + ')'; });
  // Shown by default: the newest RNGD release at its smallest device count, and each GPU's newest
  // framework at the device count closest to it. The checkboxes override this (state.on).
  let rngdN = null;
  all.forEach((o) => { if (family(o.r.hw) === 'rngd' && o.r.ver === newestHere[o.r.hw] && (rngdN === null || o.r.n < rngdN)) rngdN = o.r.n; });
  const nearestN = {};
  all.forEach((o) => {
    if (o.r.ver !== newestHere[o.r.hw]) return;
    const cur = nearestN[o.r.hw], d = (n) => Math.abs(n - (rngdN === null ? 1 : rngdN));
    if (cur === undefined || d(o.r.n) < d(cur) || (d(o.r.n) === d(cur) && o.r.n < cur)) nearestN[o.r.hw] = o.r.n;
  });
  // A selection left over from another model or task matches nothing here: fall back to the defaults.
  const on = st.on && all.some((o) => st.on.split('~').indexOf(o.r.id) >= 0) ? st.on.split('~') : null;
  all.forEach((o) => { o.on = on ? on.indexOf(o.r.id) >= 0 : o.r.ver === newestHere[o.r.hw] && o.r.n === nearestN[o.r.hw]; });
  SHOWN = all.filter((o) => o.on).map((o) => o.r.id);
  const visible = all.filter((o) => o.on);

  const mx = AXES[xKey], my = AXES[yKey];
  let failedCount = 0, xmax = 0, ymax = 0;
  const plotted = visible.map((o) => {
    const vals = [];
    o.pts.forEach((p) => {
      if (!p.ok) failedCount++;
      if (!p.ok && !showFailed) return;
      const x = mx.f(p, o.r, norm), y = my.f(p, o.r, norm);
      if (x === null || y === null || !isFinite(x) || !isFinite(y)) return;
      xmax = Math.max(xmax, x); ymax = Math.max(ymax, y);
      vals.push({ p: p, x: x, y: y, failed: !p.ok });
    });
    return { o: o, vals: vals };
  });
  const okOf = (g) => g.vals.filter((v) => !v.failed);
  const sloOnX = xKey === 'tps_user', sloOnY = yKey === 'tps_user';
  // On the interactivity axis, cut every curve at the highest TPS/user all shown systems reached,
  // so no curve runs on where another has no data.
  let clipAt = null;
  const withData = plotted.filter((g) => okOf(g).length);
  if (sloOnX && withData.length > 1) {
    clipAt = Math.min(...withData.map((g) => Math.max(...okOf(g).map((v) => v.x))));
    xmax = 0; ymax = 0;
    plotted.forEach((g) => {
      const ok = okOf(g).sort((a, b) => a.x - b.x), kept = g.vals.filter((v) => v.x <= clipAt + 1e-9);
      const below = ok.filter((v) => v.x <= clipAt).pop(), above = ok.filter((v) => v.x > clipAt)[0];
      if (below && above) kept.push({ p: above.p, x: clipAt, y: below.y + (above.y - below.y) * (clipAt - below.x) / (above.x - below.x), failed: false, interp: true });
      g.vals = kept.sort((a, b) => a.x - b.x);
      g.vals.forEach((v) => { xmax = Math.max(xmax, v.x); ymax = Math.max(ymax, v.y); });
    });
  }
  if (sloOnX) xmax = Math.max(xmax, slo);
  if (sloOnY) ymax = Math.max(ymax, slo);
  const xt = niceTicks(xmax * 1.02), yt = niceTicks(ymax * 1.05);
  const px = (x) => L + x / xt.top * (R - L), py = (y) => B - y / yt.top * (B - T);

  let svg = '<line x1="' + L + '" y1="' + B + '" x2="' + R + '" y2="' + B + '" stroke="var(--faint)"/><line x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + B + '" stroke="var(--faint)"/>';
  xt.list.forEach((v) => { svg += '<text x="' + px(v) + '" y="' + (B + 20) + '" text-anchor="middle">' + fmt(v) + '</text>'; });
  yt.list.forEach((v) => { svg += '<text x="' + (L - 8) + '" y="' + (py(v) + 4) + '" text-anchor="end">' + fmt(v) + '</text>'; });
  const axisTitle = (a) => (a.axis || a.label(norm)).toUpperCase().replace(/KW\b/g, 'kW');
  svg += '<text class="axis-title" x="' + (L + R) / 2 + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(axisTitle(mx)) + '</text>';
  svg += '<text class="axis-title" transform="translate(14 ' + (T + B) / 2 + ') rotate(-90)" text-anchor="middle">' + esc(axisTitle(my)) + '</text>';
  if (sloOnX) svg += '<line x1="' + px(slo) + '" x2="' + px(slo) + '" y1="' + T + '" y2="' + B + '" stroke="var(--faint)" stroke-dasharray="2 4"/><text x="' + (px(slo) + 6) + '" y="' + (B - 8) + '">SLO ' + slo + '</text>';
  if (sloOnY) svg += '<line x1="' + L + '" x2="' + R + '" y1="' + py(slo) + '" y2="' + py(slo) + '" stroke="var(--text)" stroke-width="1.5" stroke-dasharray="6 5"/><text x="' + (R - 4) + '" y="' + (py(slo) - 6) + '" text-anchor="end">SLO ' + slo + '</text>';

  const names = [];
  plotted.forEach((g) => {
    const ok = okOf(g);
    if (ok.length > 1) svg += '<path d="' + smoothPath(ok.map((v) => ({ x: px(v.x), y: py(v.y) }))) + '" fill="none" stroke="' + g.o.color + '" stroke-width="2" stroke-dasharray="' + g.o.dash + '"/>';
    g.vals.forEach((v) => {
      if (v.interp) return;
      const p = v.p, tip = g.o.label + ' · c' + p.conc + '\nTPS/User ' + fmt(p.out / p.conc) + '\n' + (norm === 'dev' ? 'TPS/Device ' + fmt(p.out / g.o.r.n) : 'TPS/System ' + fmt(p.out))
        + '\nTPOT p50 ' + fmt(p.tpot50) + ' ms\nTTFT p50 ' + fmt(p.ttft50) + ' s' + (v.failed ? '\nfailed' : '');
      svg += '<circle cx="' + px(v.x) + '" cy="' + py(v.y) + '" r="3" fill="' + (v.failed ? 'var(--bg)' : g.o.color) + '" stroke="' + (v.failed ? 'var(--faint)' : 'var(--bg)') + '"><title>' + esc(tip) + '</title></circle>'
        + '<text class="conc" x="' + px(v.x) + '" y="' + (py(v.y) - 7) + '" text-anchor="middle" style="fill:' + (v.failed ? 'var(--faint)' : g.o.color) + '">c' + p.conc + '</text>';
    });
    // Name the curve where it starts (its leftmost point).
    if (ok.length) {
      const first = ok.reduce((b, v) => (!b || v.x < b.x ? v : b), null), fam = family(g.o.r.hw);
      names.push({ x: Math.min(px(first.x), R - 160), y: py(first.y) - 36, color: g.o.color, text: SHORT_NAME[fam] || g.o.r.hw, full: g.o.label });
    }
  });
  // Curve names: the full series label when two shown curves would share a name, then pushed up until they stop overlapping.
  names.forEach((d) => { if (names.filter((e) => e.text === d.text).length > 1) d.alt = d.full; });
  names.forEach((d) => { if (d.alt) d.text = d.alt; });
  names.sort((a, b) => b.y - a.y).forEach((d, i) => {
    for (let j = 0; j < i; j++) {
      const e = names[j];
      if (d.x < e.x + e.text.length * 8 && e.x < d.x + d.text.length * 8 && d.y > e.y - 18) d.y = e.y - 18;
    }
    svg += '<text class="curve-name" x="' + d.x + '" y="' + Math.max(14, d.y + 14) + '" style="fill:' + d.color + '">' + esc(d.text) + '</text>';
  });

  // RNGD-vs-GPU multiplier at the SLO: an arrow from the chosen GPU curve up (or down) to the RNGD curve.
  const rngdG = plotted.filter((g) => family(g.o.r.hw) === 'rngd')[0], gpuGs = plotted.filter((g) => family(g.o.r.hw) !== 'rngd');
  const vsG = gpuGs.filter((g) => g.o.r.id === st.vs)[0] || gpuGs.filter((g) => yAtX(okOf(g), slo) !== null)[0] || gpuGs[0];
  const showVs = sloOnX && !!HIGHER_IS_BETTER[yKey] && !!rngdG && gpuGs.length > 0;
  if (showVs && vsG) {
    const yr = yAtX(okOf(rngdG), slo), yg = yAtX(okOf(vsG), slo);
    if (yr !== null && yg !== null && yg > 0) {
      const X = px(slo), Pr = py(yr), Pg = py(yg), up = Pr < Pg, head = up ? Pr + 1 : Pr - 1, dir = up ? 6 : -6;
      svg += '<path d="M ' + X + ' ' + Pg + ' L ' + X + ' ' + head + ' M ' + (X - 4) + ' ' + (head + dir) + ' L ' + X + ' ' + head + ' L ' + (X + 4) + ' ' + (head + dir) + '" fill="none" stroke="var(--muted)" stroke-width="1.2"/>'
        + '<text class="mult" x="' + X + '" y="' + (up ? Pr - 12 : Pr + 22) + '" text-anchor="middle">' + (yr / yg).toFixed(1) + '×</text>';
    }
  }

  const tableRows = visible.map((o) => {
    const u = usersAtSlo(o.pts, slo), peak = peakPoint(o.pts), one = c1(o.pts), nFail = o.pts.filter((p) => !p.ok).length, div = norm === 'dev' ? o.r.n : 1;
    return '<tr><td><span style="color:' + o.color + '">●</span> ' + esc(o.label) + '</td>'
      + '<td class="num">' + (u ? (u.ge ? '≥' : '') + fmt(u.v / div) : '—') + '</td><td class="num">' + (peak ? fmt(peak.out / div) + ' @c' + peak.conc : '—') + '</td>'
      + '<td class="num">' + (one ? fmt(one.tpot50) + ' ms' : '—') + '</td><td class="num">' + (one ? fmt(one.ttft50) + ' s' : '—') + '</td>'
      + '<td class="num">' + (peak && peak.power > 0 ? fmt(peak.out / peak.power) : '—') + '</td>'
      + '<td class="num" style="color:' + (nFail ? 'var(--fail)' : 'var(--faint)') + '">' + nFail + ' / ' + o.pts.length + '</td><td class="num date muted">' + day(o.r.date) + '</td></tr>';
  }).join('');

  const seriesList = all.map((o) => '<label class="series"><input type="checkbox" data-toggle="' + esc(o.r.id) + '"' + (o.on ? ' checked' : '') + '>'
    + '<svg width="28" height="10" aria-hidden="true"><line x1="0" y1="5" x2="28" y2="5" stroke="' + o.color + '" stroke-width="2" stroke-dasharray="' + o.dash + '"/></svg><span>' + esc(o.label) + '</span></label>').join('');
  const axisOptions = Object.keys(AXES).map((k) => ({ v: k, label: AXES[k].label(norm) }));
  const caption = model + ', ' + isl + '/' + osl + ' ISL/OSL. ' + visible.map((o) => o.label).join(', ') + '.'
    + (yKey === 'tps_kw' ? ' Per-device throughput scaled to 8-card server power: RNGD 3.0 kW, RTX PRO 6000 6.5 kW, H100/H200 10.2 kW.' : '')
    + (clipAt !== null ? ' Curves are shown up to ' + fmt(clipAt) + ' TPS per user, within the range measured on all shown systems.' : '');

  return '<div class="explorer"><aside class="panel">'
    + select('Model', 'model', model, models) + select('Task', 'task', task, tasks.map((t) => ({ v: t, label: GEN_TASKS[t] + ' · ' + t })))
    + select('ISL / OSL', 'scen', scen, scens)
    + '<div class="ctl">Hardware · backend' + seriesList + '</div>'
    + select('Normalize', 'norm', norm, [{ v: 'dev', label: 'Per device' }, { v: 'sys', label: 'Per system' }])
    + select('Failed points', 'failed', showFailed ? '1' : '0', [{ v: '0', label: 'Hide (' + failedCount + ')' }, { v: '1', label: 'Show (' + failedCount + ')' }])
    + '</aside><section><div class="controls">' + select('X axis', 'x', xKey, axisOptions) + select('Y axis', 'y', yKey, axisOptions) + sloInput(slo)
    + (showVs ? select('× RNGD vs', 'vs', vsG ? vsG.o.r.id : '', gpuGs.map((g) => ({ v: g.o.r.id, label: g.o.label }))) : '') + '</div>'
    + '<div class="panel"><div><h2 style="font-weight:400;font-size:18px">' + esc(my.title || my.label(norm)) + '</h2><div class="muted">ISL ' + isl + ' / OSL ' + osl + ' · ' + esc(GEN_TASKS[task].toLowerCase()) + '</div></div>'
    + '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="' + esc(my.label(norm) + ' against ' + mx.label(norm) + ' for ' + model) + '">' + svg + '</svg>'
    + '<p class="caption">' + esc(caption) + '</p></div>'
    + '<div class="panel scroll"><table class="list" style="min-width:820px"><thead><tr><th>Hardware · backend</th><th class="num">Users at SLO</th><th class="num">'
    + (norm === 'dev' ? 'Peak TPS / device' : 'Peak TPS / system') + '</th><th class="num">TPOT p50 · c1</th><th class="num">TTFT p50 · c1</th><th class="num">tok/W at peak</th><th class="num">Failed</th><th class="num">Measured</th></tr></thead><tbody>'
    + tableRows + '</tbody></table></div></section></div>';
};

PAGES.explorer.toggle = function (id, checked) {
  const shown = SHOWN.filter((x) => x !== id).concat(checked ? [id] : []);
  setState({ on: shown.join('~') });
};
