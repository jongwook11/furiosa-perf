// Data: every run in the site, filterable, with its points as JSON and the filtered list as CSV.
'use strict';

const RUN_COLUMNS = ['kind', 'task', 'model_id', 'hw', 'n', 'layout', 'backend', 'ver', 'date', 'num_points', 'failed', 'flags', 'id'];

function filteredRuns(runs, st) {
  return runs.filter((r) => (!st.kind || r.kind === st.kind) && (!st.dtask || r.task === st.dtask) && (!st.hw || r.hw === st.hw)
    && (!st.q || (r.model_id + ' ' + r.ver).toLowerCase().indexOf(st.q.toLowerCase()) >= 0));
}

function csvOf(runs) {
  const q = (v) => { v = Array.isArray(v) ? v.join(' ') : v === null || v === undefined ? '' : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  return [RUN_COLUMNS.join(',')].concat(runs.map((r) => RUN_COLUMNS.map((c) => q(r[c])).join(','))).join('\n') + '\n';
}

PAGES.data = async function (st) {
  const runs = await loadIndex();
  const uniq = (key) => [...new Set(runs.map((r) => r[key]))].sort();
  const rows = filteredRuns(runs, st).sort((a, b) => famOrder(a.hw) - famOrder(b.hw) || a.hw.localeCompare(b.hw) || a.n - b.n
    || cmpVer(b.ver, a.ver) || a.task.localeCompare(b.task) || a.model.localeCompare(b.model));
  const points = rows.reduce((s, r) => s + r.num_points, 0), failed = rows.reduce((s, r) => s + r.failed, 0);
  const any = (label) => [{ v: '', label: label }];
  const table = rows.map((r) => '<tr><td>' + esc(r.kind) + '</td><td>' + esc(r.task) + '</td><td>' + esc(r.model_id) + '</td>'
    + '<td>' + esc(r.hw + ' ×' + r.n + (r.layout ? ' ' + r.layout : '')) + '</td><td>' + esc(backendLabel(r.backend) + ' ' + r.ver) + '</td>'
    + '<td class="date">' + day(r.date) + '</td><td class="num">' + r.num_points + '</td>'
    + '<td class="num" style="color:' + (r.failed ? 'var(--fail)' : 'var(--faint)') + '">' + r.failed + '</td>'
    + '<td class="muted">' + esc((r.flags || []).join(' ')) + '</td>'
    + '<td><a href="data/runs/' + encodeURIComponent(r.id) + '.json" download>JSON</a></td></tr>').join('');
  return '<div class="head"><h1>Data</h1><div class="controls">'
    + select('Kind', 'kind', st.kind || '', any('All').concat(uniq('kind')))
    + select('Task', 'dtask', st.dtask || '', any('All').concat(uniq('task')))
    + select('Hardware', 'hw', st.hw || '', any('All').concat(uniq('hw')))
    + '<label class="ctl">Model or version<input type="search" data-k="q" value="' + esc(st.q || '') + '" placeholder="e.g. Qwen3 or 2026.4"></label>'
    + '<button class="button" id="csv">Download CSV</button></div></div>'
    + '<section class="kpis">' + kpi('Runs', rows.length.toLocaleString('en-US')) + kpi('Points', points.toLocaleString('en-US'))
    + kpi('Models', new Set(rows.map((r) => r.model)).size) + kpi('Failed points', failed.toLocaleString('en-US'), 'var(--fail)') + '</section>'
    + '<section class="panel"><div class="scroll"><table class="list" style="min-width:1000px"><thead><tr><th>Kind</th><th>Task</th><th>Model</th><th>Hardware</th><th>Backend</th>'
    + '<th>Measured</th><th class="num">Points</th><th class="num">Failed</th><th>Server flags</th><th></th></tr></thead><tbody>' + table + '</tbody></table></div>'
    + '<p class="caption">A failed point is one the run judged invalid (client error, failure rate over 5%, no output) or, for results from before the presets, one with no output. '
    + 'JSON has every recorded value of a run\'s points; the CSV lists the runs shown here.</p></section>';
};

// The CSV is built from the filters in the URL when the button is clicked.
document.addEventListener('click', async (e) => {
  if (e.target.id !== 'csv') return;
  const blob = new Blob([csvOf(filteredRuns(await loadIndex(), STATE))], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'furiosa-perf-runs.csv';
  a.click();
  URL.revokeObjectURL(a.href);
});
