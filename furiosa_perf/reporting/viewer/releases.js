// Releases: one RNGD metric per model across furiosa-llm releases, tinted by the change from the previous one.
'use strict';

const FLAG_LABEL = { 'enable-overlap-scheduling': 'overlap', 'experimental-enable-npu-sampling': 'npu-sampling' };

// Cell tint from a ratio (> 1 = better): improvements in the accent, regressions in grey;
// full strength at 1.2× / 0.8×, none within 0.98–1.02×.
function tint(r) {
  if (r === null || r === undefined || Math.abs(r - 1) <= 0.02) return 'transparent';
  const a = Math.min(Math.abs(r - 1), 0.2) / 0.2;
  return r > 1 ? 'rgba(226,21,0,' + (0.12 + a * 0.73).toFixed(2) + ')' : 'rgba(169,169,169,' + (0.06 + a * 0.39).toFixed(2) + ')';
}
function times(v) { return v === null || v === undefined ? '—' : v.toFixed(2) + '×'; }

PAGES.releases = async function (st) {
  const gen = await generationRuns();
  const tasks = Object.keys(GEN_TASKS).filter((t) => gen.some((r) => r.task === t && family(r.hw) === 'rngd'));
  const task = tasks.indexOf(st.task) >= 0 ? st.task : tasks[0];
  const all = gen.filter((r) => family(r.hw) === 'rngd' && r.task === task);
  const devs = [...new Set(all.map((r) => r.n))].sort((a, b) => a - b);
  const dev = st.dev && devs.indexOf(+st.dev) >= 0 ? st.dev : 'all';
  const sel = all.filter((r) => dev === 'all' || r.n === +dev);
  const scens = scenarioOptions(sel);
  const scen = pickScenario(st.scen, scens);
  const metric = METRICS[st.metric] ? st.metric : 'tpkw';
  const M = METRICS[metric];
  const slo = st.slo > 0 ? st.slo : 20;
  const controls = '<div class="controls">'
    + select('Devices', 'dev', dev, [{ v: 'all', label: 'All device counts' }].concat(devs.map((n) => ({ v: String(n), label: 'RNGD ×' + n }))))
    + select('Task', 'task', task, tasks.map((t) => ({ v: t, label: GEN_TASKS[t] + ' · ' + t })))
    + select('ISL / OSL', 'scen', scen, scens)
    + select('Metric', 'metric', metric, Object.keys(METRICS).map((k) => ({ v: k, label: METRICS[k].label })))
    + (M.slo ? sloInput(slo) : '') + '</div>';
  if (!scen) return '<h1>Releases</h1>' + controls + '<div class="status">No RNGD results.</div>';

  await loadPoints(sel);
  const [isl, osl] = scen.split('/').map(Number);
  // One row per model × device count, one column per release.
  const vers = [], byRow = {}, flags = {}, dates = {};
  sel.forEach((r) => {
    const pts = pointsAt(r, isl, osl), m = pts.length ? M.f(pts, r, slo) : null;
    if (!m || !(m.v > 0)) return;
    if (vers.indexOf(r.ver) < 0) vers.push(r.ver);
    const k = r.model + '|' + r.n;
    (byRow[k] = byRow[k] || {})[r.ver] = m;
    flags[r.ver] = flags[r.ver] || {};
    (r.flags || []).forEach((f) => { flags[r.ver][f] = 1; });
    if (r.date && (!dates[r.ver] || r.date < dates[r.ver])) dates[r.ver] = r.date;
  });
  vers.sort(cmpVer);
  // Ratio of the newer release to the older, oriented so that > 1 is an improvement.
  const better = (a, b) => (M.higher ? b.v / a.v : a.v / b.v);
  const rows = [];
  Object.keys(byRow).forEach((k) => {
    const vals = byRow[k], steps = {};
    let prev = null, first = null;
    vers.forEach((v) => {
      if (!vals[v]) return;
      if (prev) steps[v] = better(vals[prev], vals[v]);
      if (!first) first = v;
      prev = v;
    });
    if (first === prev) return;  // measured on one release only
    const [model, n] = k.split('|');
    rows.push({ model: model, n: +n, vals: vals, steps: steps, total: better(vals[first], vals[prev]), first: first, last: prev });
  });
  rows.sort((a, b) => b.total - a.total || a.model.localeCompare(b.model));

  const stepMed = {};
  vers.forEach((v) => { stepMed[v] = median(rows.filter((r) => r.steps[v] !== undefined).map((r) => r.steps[v])); });
  const latest = vers[vers.length - 1], prevVer = vers[vers.length - 2];
  const latestSteps = rows.filter((r) => r.steps[latest] !== undefined).map((r) => r.steps[latest]);
  const totMed = median(rows.map((r) => r.total));
  const short = (v) => v.replace(/^\d{4}\./, '');

  const cell = (main, o) => {
    o = o || {};
    return '<div class="' + (o.cls || '') + '" style="background:' + (o.bg || 'transparent') + (o.color ? ';color:' + o.color : '') + (o.weight ? ';font-weight:' + o.weight : '') + '">'
      + '<span>' + main + '</span>' + (o.sub ? '<span class="sub" style="color:' + (o.subColor || 'var(--muted)') + '">' + o.sub + '</span>' : '') + '</div>';
  };
  let cells = cell('Model', { cls: 'hdr left' }) + cell('RNGD', { cls: 'hdr' });
  vers.forEach((v, j) => {
    // Release, date and the server flags it turned on (new ones in white).
    const fl = Object.keys(flags[v] || {}).sort(), pf = j ? flags[vers[j - 1]] || {} : {};
    const added = fl.filter((f) => !pf[f]).map((f) => '+' + (FLAG_LABEL[f] || f));
    cells += cell(esc(short(v)), { cls: 'hdr', weight: 600, color: v === latest ? 'var(--title)' : 'var(--muted)',
      sub: esc(day(dates[v]).slice(0, 7) + (added.length ? '  ' + added.join(' ') : '')), subColor: added.length ? 'var(--text)' : 'var(--faint)' });
  });
  cells += cell('Total', { cls: 'hdr' });
  cells += cell('Median', { cls: 'hdr left' }) + cell('', { cls: 'hdr' });
  vers.forEach((v) => { cells += cell(stepMed[v] === null ? '' : times(stepMed[v]), { cls: 'hdr', weight: 600, color: 'var(--title)', bg: tint(stepMed[v]) }); });
  cells += cell(totMed === null ? '' : times(totMed), { cls: 'hdr', weight: 600, color: 'var(--title)', bg: tint(totMed) });
  rows.forEach((r) => {
    cells += cell(esc(r.model), { cls: 'left' }) + cell('×' + r.n, { color: 'var(--muted)' });
    vers.forEach((v) => {
      const x = r.vals[v], b = r.steps[v];
      cells += x ? cell((x.ge ? '≥' : '') + fmt(x.v), { bg: tint(b), sub: b === undefined ? '' : times(b), subColor: b > 1.02 ? 'var(--title)' : 'var(--muted)' })
        : cell('—', { color: 'var(--faint)' });
    });
    cells += cell(times(r.total), { weight: 600, color: r.total > 1.02 ? 'var(--accent)' : r.total < 0.98 ? 'var(--muted)' : 'var(--text)',
      sub: esc(short(r.first) + ' → ' + short(r.last)), subColor: 'var(--faint)' });
  });

  const unit = M.label + (M.slo ? ' · TPS per user ' + slo : '');
  const red = (v) => (v > 1.02 ? 'var(--accent)' : 'var(--title)');
  return '<div class="head"><h1>Releases</h1>' + controls + '</div>'
    + '<section class="kpis">'
    + kpi('Latest release · ' + (prevVer || '—') + ' → ' + (latest || '—') + ' · median', times(stepMed[latest]), red(stepMed[latest]))
    + kpi('Models up / down beyond ±2% in ' + (latest || '—') + ' (of ' + latestSteps.length + ')',
      '<span style="color:var(--accent)">' + latestSteps.filter((v) => v > 1.02).length + '</span><span class="faint" style="font-size:22px"> up · </span>'
      + '<span class="muted">' + latestSteps.filter((v) => v < 0.98).length + '</span><span class="faint" style="font-size:22px"> down</span>')
    + kpi('First measured → latest · median of ' + rows.length + ' models', times(totMed), red(totMed)) + '</section>'
    + '<section class="panel"><div class="panel-head"><h2>' + esc('furiosa-llm releases · ' + unit + ' · ' + scen) + '</h2>'
    + '<div class="heat-scale"><span class="mono">0.80×</span><span class="ramp"></span><span class="mono">1.20×</span><span>vs previous release</span></div></div>'
    + (rows.length ? '<div class="scroll"><div class="heat" role="table" style="grid-template-columns:minmax(220px,1.6fr) 56px repeat(' + vers.length + ',minmax(110px,1fr)) 120px">' + cells + '</div></div>'
      : '<div class="empty">No RNGD model was measured on two or more releases for this selection.</div>')
    + '<p class="caption">' + esc('Each cell is ' + unit + ' (' + (M.higher ? 'higher' : 'lower') + ' is better); the small figure is the multiple over the previous release that measured the model. The header marks server flags a release turned on (e.g. +overlap = --enable-overlap-scheduling).') + '</p></section>';
};
