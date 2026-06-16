/* PA-Bench Console — M-FE1 static mode + M-FE2 server mode (fe-rq.md §13)
 * Views: Overview / Runs (list+wizard+progress) / Run results / Episode browser / Episode debug / Metric registry
 * Data: probe /api/ping on startup —
 *   server mode: platform API (launch runs + SSE progress, python serve.py)
 *   static mode: web/data/ (build_web_data.py artifacts), no backend, M-FE1 behavior unchanged.
 */
"use strict";

const $main = document.getElementById("main");
const GAP_MM = { T1: 1.0, T2: 0.5, T3: 0.2 };
const PHASE_COLORS = { approach: "#eef3fb", grasp: "#fdf2dd", transfer: "#e9f7ee",
                       align: "#f3eafa", insert: "#fdeaea", fasten: "#eaf6fa" };
const ATTR_TEXT = { model: "Model", hardware: "Hardware",
                    environment: "Environment", ambiguous: "Manual review" };
// Sankey node prefixes — must stay in sync with webdata.build_sankey (backend).
const SANKEY = { phase: "Phase:", attribution: "Attribution:", rule: "Rule:" };

/* ---------------- Data layer (server/static dual mode, with cache) ---------------- */
let MODE = "static";   // "server" = M-FE2 platform API; "static" = M-FE1 web/data/
let RUN_CTX = null;    // current run context in server mode (carried by drill-down links, FR-FE-4.1)

const cache = {};
async function fetchJSON(url, { fresh = false } = {}) {
  if (!fresh && cache[url]) return cache[url];
  const resp = await fetch(url);
  if (!resp.ok) {
    let msg = `HTTP ${resp.status} — ${url}`;
    try { msg = (await resp.json()).detail || msg; } catch { /* non-JSON error body */ }
    throw new Error(msg);
  }
  const data = await resp.json();
  if (!fresh) cache[url] = data;
  return data;
}

async function detectMode() {
  try { MODE = (await fetch("/api/ping")).ok ? "server" : "static"; }
  catch { MODE = "static"; }
}
const fetchRuns = () => fetchJSON("/api/runs", { fresh: true }).then((d) => d.runs);
async function latestDoneRun() {
  const r = (await fetchRuns()).find((x) => x.status === "done");
  return r ? r.run_id : null;
}
async function resolveRun(rid) {
  if (MODE !== "server") return null;
  rid = rid || RUN_CTX || await latestDoneRun();
  if (!rid) throw new Error("No completed run yet — launch one from the Runs page first");
  RUN_CTX = rid;
  return rid;
}
async function loadIndex(rid) {
  if (MODE !== "server") return fetchJSON("data/index.json");
  rid = await resolveRun(rid);
  return fetchJSON(`/api/runs/${encodeURIComponent(rid)}/index`);
}
async function loadReport(rid) {
  if (MODE !== "server") return fetchJSON("data/report.json");
  rid = await resolveRun(rid);
  return fetchJSON(`/api/runs/${encodeURIComponent(rid)}/summary`);
}
const loadRegistry = () =>
  fetchJSON(MODE === "server" ? "/api/metric-registry" : "data/registry.json");
const loadEpisode = (id) =>
  fetchJSON(MODE === "server" ? `/api/episodes/${encodeURIComponent(id)}`
                              : `data/episodes/${encodeURIComponent(id)}.json`);
// already-loaded index (either mode), used by the debug page to read meta.seed
const idxCache = () => cache["data/index.json"]
  || (RUN_CTX && cache[`/api/runs/${encodeURIComponent(RUN_CTX)}/index`]) || null;
// drill-down links carry run context (server mode)
const withRun = (params) =>
  (MODE === "server" && RUN_CTX) ? { run: RUN_CTX, ...params } : params;

/* ---------------- Helpers ---------------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined) ? "N/A" : Number(v).toFixed(d);
const lightOf = (sr) => sr >= 0.8 ? "green" : sr >= 0.5 ? "yellow" : "red";
const lightIcon = (sr) => sr >= 0.8 ? "🟢" : sr >= 0.5 ? "🟡" : "🔴";
function qs(params) {
  const p = Object.entries(params).filter(([, v]) => v !== "" && v != null);
  return p.length ? "?" + p.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&") : "";
}
function parseHash() {
  const h = location.hash.slice(1) || "/overview";
  const [path, query = ""] = h.split("?");
  const params = Object.fromEntries(new URLSearchParams(query));
  const seg = path.split("/").filter(Boolean);
  return { seg, params };
}
let charts = [];
function mkChart(dom) { const c = echarts.init(dom); charts.push(c); return c; }
function disposeCharts() { charts.forEach((c) => c.dispose()); charts = []; }
let stopPlayer = null;
let stopWatcher = null;  // cleanup hook for the run-progress page SSE/polling

function setView(html) {
  disposeCharts();
  if (stopPlayer) { stopPlayer(); stopPlayer = null; }
  if (stopWatcher) { stopWatcher(); stopWatcher = null; }
  $main.innerHTML = html;
}

/* toast (§6: success auto-dismiss 3s / failure stays, closable) */
function toast(msg, type = "ok") {
  let box = document.getElementById("toasts");
  if (!box) {
    box = document.createElement("div"); box.id = "toasts";
    document.body.appendChild(box);
  }
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `${esc(msg)}${type === "fail"
    ? ' <button onclick="this.parentElement.remove()">×</button>' : ""}`;
  box.appendChild(el);
  if (type !== "fail") setTimeout(() => el.remove(), 3000);
}
window.toast = toast;
function errorView(err, retry) {
  setView(`<div class="error-box">Failed to load: ${esc(err.message)}
    <button onclick="${retry}">Retry</button>
    <div style="font-size:12px;margin-top:6px">Tip: run
    <code>python demo.py && python build_web_data.py</code> first to generate data.</div></div>`);
}

/* ---------------- ① Overview (customer layer) ---------------- */
async function renderOverview() {
  setView(`<h1>Overview</h1><div class="meta">Loading…</div>
    <div class="cards">${'<div class="card skeleton" style="height:130px"></div>'.repeat(4)}</div>`);
  let idx, registry, report;
  try {
    [idx, registry, report] = await Promise.all([loadIndex(), loadRegistry(), loadReport()]);
  } catch (e) {
    if (MODE === "server")  // empty state: no completed run yet → guide to launch (§4.1)
      return setView(`<h1>Overview</h1><div class="empty-box">No runs yet
        <div style="margin-top:10px"><button onclick="location.hash='#/runs/new'">New run</button></div></div>`);
    return errorView(e, "router()");
  }
  const results = report.results;
  if (!results.length)
    return setView(`<h1>Overview</h1><div class="empty-box">No runs yet — run
      <code>python demo.py</code> in a terminal, then refresh</div>`);

  // Δ arrow: SR of same combo vs the previous completed run (§4.1, server mode only has run history)
  const prevSR = {};
  if (MODE === "server") {
    try {
      const done = (await fetchRuns()).filter((r) => r.status === "done");
      if (done.length > 1) {
        const prev = await fetchJSON(`/api/runs/${encodeURIComponent(done[1].run_id)}/summary`);
        prev.results.forEach((r) => { prevSR[`${r.model_id}@${r.hw_config_id}`] = r.sr; });
      }
    } catch { /* Δ is enhancement-only, ignore failures */ }
  }
  const delta = (r) => {
    const p = prevSR[`${r.model_id}@${r.hw_config_id}`];
    if (p === undefined || p === r.sr) return "";
    const up = r.sr > p;
    return ` <span style="color:${up ? "#237a36" : "#b3261e"};font-size:14px"
      title="vs previous run ${fmt(p)} → ${fmt(r.sr)}">${up ? "▲" : "▼"}${((r.sr - p) * 100).toFixed(0)}%</span>`;
  };
  const cardTarget = MODE === "server" && RUN_CTX
    ? `#/runs/${encodeURIComponent(RUN_CTX)}` : "#/run";

  const cards = results.slice().sort((a, b) => b.sr - a.sr).map((r) => {
    const dom = Object.entries(r.attribution_counts).sort((a, b) => b[1] - a[1])[0];
    const short = dom ? `Main weakness: ${ATTR_TEXT[dom[0]] || dom[0]} (${dom[1]} failures)`
                      : "No failed episodes";
    return `<div class="card ${lightOf(r.sr)}" onclick="location.hash='${cardTarget}'">
      <span class="light">${lightIcon(r.sr)}</span>
      <div class="title">${esc(r.model_id)}</div>
      <div class="hw">${esc(r.hw_config_id)}</div>
      <div class="sr">${(r.sr * 100).toFixed(0)}%${delta(r)}</div>
      <div class="ci">95% CI [${fmt(r.ci95[0])}, ${fmt(r.ci95[1])}] · n=${r.n}</div>
      <div class="short">${esc(short)}${r.mr1_violated ? " · ⚠️ MR-1 equivariance violated" : ""}</div>
    </div>`;
  }).join("");

  // Top weaknesses: 3 lowest-SR combos, bound to the registry's improvement actions (FR-5.1)
  const weak = results.slice().sort((a, b) => a.sr - b.sr).slice(0, 3).map((r) => {
    const dom = Object.entries(r.attribution_counts).sort((a, b) => b[1] - a[1])[0];
    let action = "—", filter = { success: "false" };
    if (r.mr1_violated && dom && dom[0] === "model") {
      action = registry["l2.plan_margin_ratio"].improvement_actions[0];
      filter.attribution = "model";
    } else if (dom && dom[0] === "hardware") {
      action = registry["l3.e_track_rms"].improvement_actions[0];
      filter.attribution = "hardware";
    } else if (dom) {
      action = registry["l2.plan_margin_ratio"].improvement_actions[0];
      filter.attribution = dom[0];
    }
    filter.model = r.model_id; filter.hw = r.hw_config_id;
    return `<li onclick="location.hash='#/episodes${qs(withRun(filter))}'">
      ${lightIcon(r.sr)} <b>${esc(r.model_id)} @ ${esc(r.hw_config_id)}</b>
      (SR ${(r.sr * 100).toFixed(0)}%) → suggested action: ${esc(action)}</li>`;
  }).join("");

  const meta = idx.meta;
  setView(`<h1>Overview <span style="font-weight:400;font-size:13px;color:#888">for readers without a robotics background</span></h1>
    <div class="meta">benchmark: ${esc(meta.benchmark_version)} · seed ${meta.seed} ·
      ${meta.total_episodes} episodes · attribution rules ${esc(meta.attr_rules_version)} ·
      🟢≥80% usable / 🟡≥50% at risk / 🔴 not usable</div>
    <div class="cards">${cards}</div>
    <h2>Top-3 weaknesses & improvement actions (click to drill down)</h2><ul class="weak-list">${weak}</ul>`);
}

/* ---------------- ② Run results (engineering layer) ---------------- */
let runSort = { key: "sr", dir: -1 };
let runCustom = { specs: [], vals: {} };  // user-registered metrics for the current run (server mode)
async function renderRun(rid) {
  setView(`<h1>Run results</h1><div class="loading">Loading…</div>`);
  let idx, report;
  try {
    idx = await loadIndex(rid);
    report = await loadReport(rid);
  } catch (e) { return errorView(e, "router()"); }
  if (!report.results.length)
    return setView(`<h1>Run results</h1><div class="empty-box">No episode data in this run
      ${MODE === "server" ? '<div style="margin-top:8px"><button onclick="location.hash=\'#/runs/new\'">Restart</button></div>' : ""}</div>`);
  const agg = idx.aggregates, meta = idx.meta;
  runCustom = { specs: idx.custom_metric_specs || [], vals: idx.custom_metrics || {} };
  const jsonHref = MODE === "server" && RUN_CTX
    ? `/api/runs/${encodeURIComponent(RUN_CTX)}/summary` : "data/report.json";

  setView(`<h1>Run results — screw cap (screw_cap · T1)</h1>
    <div class="meta">${MODE === "server" && RUN_CTX ? `run <code>${esc(RUN_CTX)}</code> · ` : ""}benchmark ${esc(meta.benchmark_version)} · seed ${meta.seed} ·
      ${meta.total_episodes} episodes · status <b style="color:#237a36">done</b> ·
      <a class="plain" href="${jsonHref}" download="report.json">Export JSON</a></div>
    <div id="combo-table"></div>
    <div class="chart-grid" style="margin-top:14px">
      <div class="chart-box"><h3>C1 Capability radar (min-max normalized, baseline = this run)</h3>
        <div id="c1" class="chart"></div></div>
      <div class="chart-box"><h3>C2 Robustness decay: SR vs lighting intensity (click a bucket to drill down)</h3>
        <div id="c2" class="chart"></div></div>
      <div class="chart-box"><h3>C3 Failure-attribution Sankey: failure→phase→attribution→rule (click to drill down)</h3>
        <div id="c3" class="chart"></div></div>
      <div class="chart-box"><h3>C5 First-failure phase histogram (click to drill down)</h3>
        <div id="c5" class="chart"></div></div>
    </div>`);
  drawComboTable(report.results);
  drawRadar(agg.radar); drawRobustness(agg.robustness);
  drawSankey(agg.sankey); drawFailureHist(agg.failure_hist);
}

function drawComboTable(results) {
  const cols = [
    ["combo", "Model @ hardware", null],
    ["sr", "SR [95%CI]", "successful/total episodes, Wilson CI (FR-3.1)"],
    ["plan_margin_mean", "e_plan margin", ">1 = model plan guaranteed to fail (FR-3.5)"],
    ["e_track_rms_mean_mm", "e_track mm", "steady-state tracking error, hardware quantity (FR-3.12)"],
    ["jitter_band_mean", "Jitter energy", "5–50Hz PSD (FR-3.11)"],
    ["jerk_cmd_median", "Command jerk", "dimensionless jerk integral (FR-3.6)"],
    ["uncertainty_auroc", "AUROC", "ability of uncertainty to predict failure (FR-3.8)"],
    ["mr1_violated", "MR-1", "equivariance metamorphic test (FR-1.3)"],
    ["attr", "Attribution mix", "failure-responsibility breakdown (FR-4)"],
    ["oracle_replays_used", "Oracle calls", "controlled-experiment cost (FR-4.3)"]];
  // user-registered metrics → extra columns (server mode); values precomputed per combo by the backend
  const customCols = runCustom.specs.map((s) => ({
    id: s.metric_id, label: s.metric_id.replace(/^custom\./, ""),
    tip: `custom · ${s.agg}(${s.expr})` }));
  const rows = results.slice().sort((a, b) => {
    const k = runSort.key, av = a[k] ?? -1, bv = b[k] ?? -1;
    return (av > bv ? 1 : av < bv ? -1 : 0) * runSort.dir;
  });
  const customVal = (r, id) => {
    const v = (runCustom.vals[`${r.model_id} @ ${r.hw_config_id}`] || {})[id];
    return v === null || v === undefined ? "N/A" : fmt(v, 3);
  };
  const html = `<table class="grid"><tr>${cols.map(([k, label, tip]) =>
    `<th ${tip ? `title="${esc(tip)}"` : ""} onclick="sortRun('${k}')">${label}
     ${runSort.key === k ? `<span class="arrow">${runSort.dir > 0 ? "▲" : "▼"}</span>` : ""}</th>`).join("")}
    ${customCols.map((c) => `<th title="${esc(c.tip)}" class="custom-col">✦ ${esc(c.label)}</th>`).join("")}</tr>
    ${rows.map((r) => `<tr class="clickable"
        onclick="location.hash='#/episodes${qs(withRun({ model: r.model_id, hw: r.hw_config_id }))}'">
      <td>${lightIcon(r.sr)} ${esc(r.model_id)}<br><span style="color:#888;font-size:11px">${esc(r.hw_config_id)}</span></td>
      <td><b>${fmt(r.sr)}</b> [${fmt(r.ci95[0])}, ${fmt(r.ci95[1])}]</td>
      <td>${fmt(r.plan_margin_mean)}</td><td>${fmt(r.e_track_rms_mean_mm)}</td>
      <td>${fmt(r.jitter_band_mean, 1)}</td><td>${fmt(r.jerk_cmd_median, 0)}</td>
      <td>${r.uncertainty_auroc === null ? '<span class="badge attr-ambiguous">N/A</span>' : fmt(r.uncertainty_auroc)}</td>
      <td>${r.mr1_violated ? "❌ violated" : "✅ passed"} (${fmt(r.mr1_median_dist_mm, 1)}mm)</td>
      <td>${Object.entries(r.attribution_counts).map(([k, v]) =>
            `<span class="badge attr-${k}">${k}:${v}</span>`).join(" ") || "—"}</td>
      <td>${r.oracle_replays_used}</td>
      ${customCols.map((c) => `<td class="custom-col">${customVal(r, c.id)}</td>`).join("")}</tr>`).join("")}</table>`;
  document.getElementById("combo-table").innerHTML = html;
}
window.sortRun = (k) => {
  if (k === "combo" || k === "attr") return;
  runSort = { key: k, dir: runSort.key === k ? -runSort.dir : -1 };
  loadReport(RUN_CTX).then((r) => drawComboTable(r.results));
};

function drawRadar(radar) {
  const c = mkChart(document.getElementById("c1"));
  c.setOption({
    tooltip: { trigger: "item", formatter: (p) => {
      const combo = radar.combos[p.dataIndex];
      return `<b>${esc(combo.name)}</b><br>` + radar.axes.map((a, i) =>
        `${a}: ${combo.raw[i] === null ? "N/A (model emits no uncertainty)" : combo.raw[i]}`).join("<br>");
    } },
    legend: { type: "scroll", bottom: 0, textStyle: { fontSize: 11 } },
    radar: { indicator: radar.axes.map((n) => ({ name: n, max: 1 })), radius: "62%" },
    series: [{ type: "radar",
      data: radar.combos.map((cb) => ({ name: cb.name, value: cb.norm })) }],
  });
}

function drawRobustness(rb) {
  const c = mkChart(document.getElementById("c2"));
  const combos = Object.keys(rb.series);
  c.setOption({
    tooltip: { trigger: "axis", formatter: (ps) => ps.map((p) => {
      const row = rb.series[p.seriesName][p.dataIndex];
      return row && row.sr !== null
        ? `${p.marker}${esc(p.seriesName)}: SR ${fmt(row.sr)} [${row.ci_lo}, ${row.ci_hi}] (n=${row.n})`
        : `${p.marker}${esc(p.seriesName)}: no samples`; }).join("<br>") },
    legend: { type: "scroll", bottom: 0, textStyle: { fontSize: 11 } },
    grid: { top: 30, left: 45, right: 15, bottom: 55 },
    xAxis: { type: "category", data: rb.bucket_labels, name: "lux" },
    yAxis: { type: "value", min: 0, max: 1, name: "SR" },
    series: combos.map((name) => ({
      name, type: "line", connectNulls: true,
      data: rb.series[name].map((r) => r.sr) })),
  });
  c.on("click", (p) => {
    const row = rb.series[p.seriesName] && rb.series[p.seriesName][p.dataIndex];
    if (!row) return;
    const [model, hw] = p.seriesName.split(" @ ");
    location.hash = "#/episodes" + qs(withRun({ model, hw, lux_min: row.lux_min, lux_max: row.lux_max }));
  });
}

function drawSankey(sk) {
  if (!sk.links.length) {
    document.getElementById("c3").innerHTML = '<div class="empty-box">No failed episodes</div>';
    return;
  }
  const c = mkChart(document.getElementById("c3"));
  c.setOption({
    tooltip: { formatter: (p) => p.dataType === "edge"
      ? `${esc(p.data.source)} → ${esc(p.data.target)}: ${p.data.value} episodes`
      : `${esc(p.name)}` },
    series: [{ type: "sankey", data: sk.nodes, links: sk.links,
      emphasis: { focus: "adjacency" }, lineStyle: { color: "gradient", opacity: 0.35 },
      label: { fontSize: 11 } }],
  });
  c.on("click", (p) => {
    if (p.dataType !== "node") return;
    const n = p.name, f = { success: "false" };
    if (n.startsWith(SANKEY.phase)) f.failure_phase = n.slice(SANKEY.phase.length);
    else if (n.startsWith(SANKEY.attribution)) f.attribution = n.slice(SANKEY.attribution.length);
    else if (n.startsWith(SANKEY.rule)) f.rule = n.slice(SANKEY.rule.length);
    location.hash = "#/episodes" + qs(withRun(f));
  });
}

function drawFailureHist(fh) {
  const c = mkChart(document.getElementById("c5"));
  const combos = Object.keys(fh.series);
  c.setOption({
    tooltip: {}, legend: { type: "scroll", bottom: 0, textStyle: { fontSize: 11 } },
    grid: { top: 25, left: 40, right: 10, bottom: 55 },
    xAxis: { type: "category", data: fh.phases, name: "First-failure phase" },
    yAxis: { type: "value", name: "Episodes" },
    series: combos.map((name) => ({ name, type: "bar", data: fh.series[name] })),
  });
  c.on("click", (p) => {
    const [model, hw] = p.seriesName.split(" @ ");
    location.hash = "#/episodes" + qs(withRun({ model, hw, success: "false", failure_phase: p.name }));
  });
}

/* ---------------- ③ Episode browser ---------------- */
let epSort = { key: "episode_id", dir: 1 };
function applyFilters(records, f) {
  return records.filter((r) =>
    (!f.model || r.model_id === f.model) &&
    (!f.hw || r.hw_config_id === f.hw) &&
    (!f.success || String(r.success) === f.success) &&
    (!f.attribution || r.attribution === f.attribution) &&
    (!f.generation_method || r.generation_method === f.generation_method) &&
    (!f.failure_phase || r.failure_phase === f.failure_phase) &&
    (!f.rule || (r.attribution_reason || "").includes(f.rule)) &&
    (!f.lux_min || r.lux >= Number(f.lux_min)) &&
    (!f.lux_max || r.lux <= Number(f.lux_max)));
}

async function renderEpisodes(params) {
  setView(`<h1>Episode browser</h1><div class="loading">Loading…</div>`);
  let idx;
  try { idx = await loadIndex(params.run); } catch (e) { return errorView(e, "router()"); }
  const recs = idx.episodes;
  if (!recs.length)
    return setView(`<h1>Episode browser</h1><div class="empty-box">No episode data — run the demo first</div>`);

  const distinct = (k) => [...new Set(recs.map((r) => r[k]).filter((v) => v != null))].sort();
  const sel = (name, label, opts, cur) => `<label>${label}
    <select onchange="setFilter('${name}', this.value)">
      <option value="">All</option>
      ${opts.map((o) => `<option value="${esc(o)}" ${String(o) === cur ? "selected" : ""}>${esc(o)}</option>`).join("")}
    </select></label>`;

  const filtered = applyFilters(recs, params).slice().sort((a, b) => {
    const av = a[epSort.key] ?? -Infinity, bv = b[epSort.key] ?? -Infinity;
    return (av > bv ? 1 : av < bv ? -1 : 0) * epSort.dir;
  });
  sessionStorage.setItem("epList", JSON.stringify(filtered.map((r) => r.episode_id)));

  const cols = [["episode_id", "Episode"], ["model_id", "Model"], ["hw_config_id", "Hardware"],
    ["success", "Result"], ["failure_phase", "First-failure phase"], ["attribution", "Attribution"],
    ["plan_margin_ratio", "e_plan margin"], ["e_track_steady_rms_mm", "e_track mm"],
    ["peak_uncertainty", "Peak uncertainty"], ["lux", "lux"], ["generation_method", "Generation"]];

  const body = filtered.length === 0
    ? `<div class="empty-box">No matches for the filters (of ${recs.length} episodes)
        <div style="margin-top:8px"><button class="linkish" onclick="location.hash='#/episodes'">Clear all filters</button></div></div>`
    : `<table class="grid"><tr>${cols.map(([k, label]) =>
        `<th onclick="sortEp('${k}')">${label}${epSort.key === k ? `<span class="arrow">${epSort.dir > 0 ? "▲" : "▼"}</span>` : ""}</th>`).join("")}</tr>
      ${filtered.map((r) => `<tr class="clickable"
          onclick="location.hash='#/episode/${encodeURIComponent(r.episode_id)}'">
        <td title="${esc(r.episode_id)}" style="max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.episode_id)}</td>
        <td>${esc(r.model_id)}</td><td>${esc(r.hw_config_id)}</td>
        <td>${r.success ? '<span class="badge ok">success</span>' : '<span class="badge fail">failure</span>'}</td>
        <td>${esc(r.failure_phase || "—")}</td>
        <td>${r.attribution ? `<span class="badge attr-${esc(r.attribution)}" title="${esc(r.attribution_reason || "")}">${esc(r.attribution)}</span>` : "—"}</td>
        <td>${fmt(r.plan_margin_ratio)}</td><td>${fmt(r.e_track_steady_rms_mm)}</td>
        <td>${fmt(r.peak_uncertainty)}</td><td>${fmt(r.lux)}</td>
        <td><span class="badge gen">${esc(r.generation_method)}</span>${r.mr_id ? ` <span class="badge gen">${esc(r.mr_id)}</span>` : ""}</td>
      </tr>`).join("")}</table>`;

  setView(`<h1>Episode browser</h1>
    <div class="meta">Filter state is written to the URL — shareable directly (FR-FE-4.1)</div>
    <div class="filters">
      ${sel("model", "Model", distinct("model_id"), params.model || "")}
      ${sel("hw", "Hardware", distinct("hw_config_id"), params.hw || "")}
      ${sel("success", "Result", ["true", "false"], params.success || "")}
      ${sel("attribution", "Attribution", distinct("attribution"), params.attribution || "")}
      ${sel("failure_phase", "First-failure phase", distinct("failure_phase"), params.failure_phase || "")}
      ${sel("generation_method", "Generation", distinct("generation_method"), params.generation_method || "")}
      <label>lux ≥ <input type="number" step="0.05" min="0" max="1" style="width:64px"
        value="${esc(params.lux_min || "")}" onchange="setFilter('lux_min', this.value)"></label>
      <label>lux ≤ <input type="number" step="0.05" min="0" max="1" style="width:64px"
        value="${esc(params.lux_max || "")}" onchange="setFilter('lux_max', this.value)"></label>
      ${params.rule ? `<label>rule <span class="badge gen">${esc(params.rule)}</span></label>` : ""}
      <button class="linkish" onclick="location.hash='#/episodes'">Clear</button>
      <span class="count">${filtered.length} / ${recs.length} episodes</span>
    </div>${body}`);
}
window.setFilter = (k, v) => {
  const { params } = parseHash();
  if (v === "") delete params[k]; else params[k] = v;
  location.hash = "#/episodes" + qs(params);
};
window.sortEp = (k) => {
  epSort = { key: k, dir: epSort.key === k ? -epSort.dir : 1 };
  router();
};

/* ---------------- Episode debug page ---------------- */
async function renderEpisode(id) {
  setView(`<h1>Episode debug</h1><div class="loading">Loading episode data…</div>`);
  let ep;
  try { ep = await loadEpisode(id); }
  catch (e) {
    return setView(`<h1>Episode debug</h1><div class="error-box">Episode not found or failed to load: ${esc(e.message)}
      <button class="linkish" onclick="location.hash='#/episodes'">Back to episode list</button></div>`);
  }
  const o = ep.outcome, scene = ep.scene;
  const t = ep.robot.t, n = t.length, dt = t[1] - t[0];
  const cmd = ep.model.actions.cmd_xyz, act = ep.robot.ee_xyz_actual;
  const entropy = ep.model.actions.entropy;
  const target = scene.target_pose_gt.xyz, part = scene.part_pose_gt.xyz;
  const gapMM = GAP_MM[scene.tolerance_class];
  const dist2 = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const ePlan = cmd.map((p) => dist2(p, target) * 1000);
  const eTrack = cmd.map((p, i) => Math.hypot(p[0] - act[i][0], p[1] - act[i][1], p[2] - act[i][2]) * 1000);
  const force = ep.robot.ft_wrench.map((w) => Math.hypot(w[0], w[1], w[2]));
  const grip = ep.robot.gripper_width.map((g) => g * 1000);
  const spans = ep.robot.phase_spans; // [[phase,i0,i1],...]
  const phaseAt = (i) => { for (const [p, i0, i1] of spans) if (i >= i0 && i < i1) return p; return "done"; };

  const list = JSON.parse(sessionStorage.getItem("epList") || "[]");
  const pos = list.indexOf(id);
  const navBtn = (d, label) => pos >= 0 && list[pos + d]
    ? `<button class="linkish" onclick="location.hash='#/episode/${encodeURIComponent(list[pos + d])}'">${label}</button>`
    : "";

  setView(`<h1 style="font-size:16px">Episode debug <span style="color:#888;font-weight:400">${esc(id)}</span></h1>
    <div class="meta">${navBtn(-1, "← Prev episode")} ${pos >= 0 ? `${pos + 1}/${list.length}` : ""} ${navBtn(1, "Next episode →")}
      &nbsp;·&nbsp; <button class="linkish" onclick="navigator.clipboard.writeText('python demo.py --seed ${esc(String(idxCache()?.meta.seed ?? ""))}')">Copy reproduce command</button>
      ${scene.parent_episode_id ? `&nbsp;·&nbsp; lineage: <span class="badge gen">${esc(scene.generation_method)}${scene.mr_id ? " " + esc(scene.mr_id) : ""}</span> ← ${esc(scene.parent_episode_id)}` : ""}</div>
    <div class="conclusion">
      ${o.success ? '<span class="badge ok" style="font-size:14px">✅ success</span>'
                  : `<span class="badge fail" style="font-size:14px">❌ failure · ${esc(o.failure_label || "")} (${esc(o.failure_phase || "")} phase)</span>`}
      ${o.attribution ? `<span class="badge attr-${esc(o.attribution)}" style="font-size:13px">attribution: ${esc(o.attribution)}</span>` : ""}
      <span class="reason">${esc(o.attribution_reason || "no attribution for a successful episode")}</span>
      <span style="font-size:12px;color:#888">model ${esc(ep.model.model_id)} · hardware ${esc(ep.robot.hw_config_id)} · tolerance ${gapMM}mm</span>
    </div>
    <div class="debug-layout">
      <div class="canvas-stack">
        <div class="canvas-box"><h3>Top-down panorama (XY) —
          <span class="legend-dot" style="background:#2f63d8"></span>command (cmd)
          <span class="legend-dot" style="background:#d33;margin-left:8px"></span>actual</h3>
          <canvas id="cv-full" class="traj" height="300"></canvas></div>
        <div class="canvas-box"><h3>Target magnifier (±10mm window · green circle = tolerance ${gapMM}mm)</h3>
          <canvas id="cv-zoom" class="traj" height="260"></canvas></div>
        <div class="na-panel">Video area: no video for this episode (FakeSim does not render) — trajectory replay is the authoritative data (O-F2)</div>
      </div>
      <div class="canvas-box"><h3>Synchronized timeline (red line = playhead, click the plot to jump to a time)</h3>
        <div id="ts-wrap"><div id="ts" style="width:100%;height:560px"></div>
        <div id="playhead"></div></div></div>
    </div>
    <div class="player">
      <button id="btn-play">▶ Play</button>
      <select id="speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option>
        <option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select>
      <input type="range" id="seek" min="0" max="${n - 1}" value="0">
      <span class="t" id="t-disp">t=0.00s</span><span id="phase-disp" class="badge gen">approach</span>
    </div>`);

  // ---- timeline chart (C4: 4 panels sharing X, linked) ----
  const tsChart = mkChart(document.getElementById("ts"));
  const grids = [
    { top: "4%", height: "17%" }, { top: "29%", height: "17%" },
    { top: "54%", height: "17%" }, { top: "79%", height: "15%" }];
  const markArea = { silent: true, data: spans.map(([p, i0, i1]) =>
    [{ xAxis: t[i0], itemStyle: { color: PHASE_COLORS[p] || "#f5f5f5" },
       label: { show: true, position: "insideTop", fontSize: 9, color: "#999", formatter: p } },
     { xAxis: t[Math.min(i1, n - 1)] }]) };
  const mkAxis = (i, yName) => ({
    x: { type: "value", gridIndex: i, min: 0, max: t[n - 1],
         axisLabel: { show: i === 3, formatter: "{value}s" } },
    y: { type: "value", gridIndex: i, name: yName,
         nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } } });
  const axes = [mkAxis(0, "error mm"), mkAxis(1, "entropy"), mkAxis(2, "force N"), mkAxis(3, "gripper mm")];
  const zip = (arr) => arr.map((v, i) => [t[i], v]);
  const series = [
    { name: "e_plan (cmd→target)", type: "line", xAxisIndex: 0, yAxisIndex: 0, showSymbol: false,
      data: zip(ePlan), lineStyle: { width: 1.5 }, markArea,
      markLine: { silent: true, symbol: "none", label: { formatter: "tolerance", fontSize: 9 },
                  lineStyle: { color: "#c00", type: "dotted" }, data: [{ yAxis: gapMM }] } },
    { name: "e_track (actual−cmd)", type: "line", xAxisIndex: 0, yAxisIndex: 0, showSymbol: false,
      data: zip(eTrack), lineStyle: { width: 1.5 } },
    ...(entropy ? [{ name: "uncertainty (entropy)", type: "line", xAxisIndex: 1, yAxisIndex: 1,
      showSymbol: false, data: zip(entropy), lineStyle: { width: 1.5, color: "#8a5fc8" } }] : []),
    { name: "‖F‖ contact force", type: "line", xAxisIndex: 2, yAxisIndex: 2, showSymbol: false,
      data: zip(force), lineStyle: { width: 1.5, color: "#d97f2c" },
      markLine: { silent: true, symbol: "none", label: { formatter: "safety threshold", fontSize: 9 },
                  lineStyle: { color: "#c00", type: "dotted" }, data: [{ yAxis: 3 }] } },
    { name: "gripper opening", type: "line", xAxisIndex: 3, yAxisIndex: 3, showSymbol: false,
      data: zip(grip), lineStyle: { width: 1.5, color: "#3a9d8f" } }];
  tsChart.setOption({
    animation: false,
    tooltip: { trigger: "axis", textStyle: { fontSize: 11 },
      valueFormatter: (v) => Array.isArray(v) ? v[1].toFixed(3) : Number(v).toFixed(3) },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    grid: grids, xAxis: axes.map((a) => a.x), yAxis: axes.map((a) => a.y),
    series,
    title: entropy ? [] : [{ text: "this model emits no uncertainty — N/A (FR-FE-5.2)",
      top: "32%", left: "center", textStyle: { fontSize: 12, color: "#999", fontWeight: 400 } }],
  });

  // ---- canvas replay ----
  const cvF = document.getElementById("cv-full"), cvZ = document.getElementById("cv-zoom");
  const setupCanvas = (cv) => {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = Number(cv.getAttribute("height"));
    cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    return { ctx, w, h };
  };
  const F = setupCanvas(cvF), Z = setupCanvas(cvZ);
  // panorama viewport range
  const xs = cmd.map((p) => p[0]).concat(act.map((p) => p[0]), [target[0], part[0]]);
  const ys = cmd.map((p) => p[1]).concat(act.map((p) => p[1]), [target[1], part[1]]);
  const pad = 0.05;
  const bx = [Math.min(...xs) - pad, Math.max(...xs) + pad];
  const by = [Math.min(...ys) - pad, Math.max(...ys) + pad];
  const mapFull = (p) => [ (p[0] - bx[0]) / (bx[1] - bx[0]) * F.w,
                           F.h - (p[1] - by[0]) / (by[1] - by[0]) * F.h ];
  const ZWIN = 0.010; // ±10mm
  const mapZoom = (p) => [ (p[0] - target[0] + ZWIN) / (2 * ZWIN) * Z.w,
                           Z.h - (p[1] - target[1] + ZWIN) / (2 * ZWIN) * Z.h ];

  function drawPath(ctx, mapFn, pts, upto, color, dash) {
    ctx.beginPath(); ctx.setLineDash(dash);
    let started = false;
    for (let i = 0; i <= upto; i++) {
      const [x, y] = mapFn(pts[i]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke(); ctx.setLineDash([]);
  }
  function dot(ctx, mapFn, p, color, r) {
    const [x, y] = mapFn(p);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  }
  function drawFrame(i) {
    // panorama
    F.ctx.clearRect(0, 0, F.w, F.h);
    dot(F.ctx, mapFull, part, "#e6a700", 6);
    F.ctx.fillStyle = "#888"; F.ctx.font = "10px sans-serif";
    F.ctx.fillText("cap (bin)", mapFull(part)[0] + 8, mapFull(part)[1]);
    dot(F.ctx, mapFull, target, "#2e9e44", 6);
    F.ctx.fillText("bottle mouth (target)", mapFull(target)[0] + 8, mapFull(target)[1]);
    drawPath(F.ctx, mapFull, cmd, i, "#2f63d8", [5, 4]);
    drawPath(F.ctx, mapFull, act, i, "#d33", []);
    dot(F.ctx, mapFull, cmd[i], "#2f63d8", 4); dot(F.ctx, mapFull, act[i], "#d33", 4);
    // magnifier
    Z.ctx.clearRect(0, 0, Z.w, Z.h);
    const [tx, ty] = mapZoom(target);
    const rPix = gapMM / 1000 / (2 * ZWIN) * Z.w;
    Z.ctx.beginPath(); Z.ctx.arc(tx, ty, rPix, 0, Math.PI * 2);
    Z.ctx.fillStyle = "rgba(46,158,68,.12)"; Z.ctx.fill();
    Z.ctx.strokeStyle = "#2e9e44"; Z.ctx.lineWidth = 1.5; Z.ctx.stroke();
    drawPath(Z.ctx, mapZoom, cmd, i, "#2f63d8", [5, 4]);
    drawPath(Z.ctx, mapZoom, act, i, "#d33", []);
    dot(Z.ctx, mapZoom, cmd[i], "#2f63d8", 4); dot(Z.ctx, mapZoom, act[i], "#d33", 4);
    const inTol = dist2(act[i], target) * 1000 <= gapMM;
    Z.ctx.fillStyle = inTol ? "#237a36" : "#b3261e"; Z.ctx.font = "11px sans-serif";
    Z.ctx.fillText(`measured offset ${ (dist2(act[i], target) * 1000).toFixed(2) }mm ${inTol ? "≤ tolerance ✓" : "> tolerance ✗"}`, 8, 16);
  }

  // ---- playback control + playhead linkage (FR-FE-5.1) ----
  const playheadEl = document.getElementById("playhead");
  const seek = document.getElementById("seek"), btn = document.getElementById("btn-play");
  const tDisp = document.getElementById("t-disp"), phDisp = document.getElementById("phase-disp");
  let cur = 0, playing = false, raf = null, lastTs = null;
  function setIdx(i) {
    cur = Math.max(0, Math.min(n - 1, Math.round(i)));
    seek.value = cur;
    tDisp.textContent = `t=${t[cur].toFixed(2)}s`;
    phDisp.textContent = phaseAt(cur);
    drawFrame(cur);
    try {
      const px = tsChart.convertToPixel({ xAxisIndex: 0 }, t[cur]);
      if (Number.isFinite(px)) {
        playheadEl.style.display = "block";
        playheadEl.style.left = px + "px";
        playheadEl.style.height = document.getElementById("ts").clientHeight + "px";
      }
    } catch { /* chart not ready */ }
  }
  function tick(ts) {
    if (!playing) return;
    if (lastTs !== null) {
      const speed = Number(document.getElementById("speed").value);
      cur += ((ts - lastTs) / 1000) * speed / dt;
      if (cur >= n - 1) { cur = n - 1; toggle(false); }
      setIdx(cur);
    }
    lastTs = ts;
    if (playing) raf = requestAnimationFrame(tick);
  }
  function toggle(v) {
    playing = v === undefined ? !playing : v;
    btn.textContent = playing ? "⏸ Pause" : "▶ Play";
    lastTs = null;
    if (playing) raf = requestAnimationFrame(tick);
    else if (raf) cancelAnimationFrame(raf);
  }
  btn.onclick = () => { if (cur >= n - 1) cur = 0; toggle(); };
  seek.oninput = () => { toggle(false); setIdx(Number(seek.value)); };
  tsChart.getZr().on("click", (e) => {
    const tv = tsChart.convertFromPixel({ xAxisIndex: 0 }, e.offsetX);
    if (Number.isFinite(tv)) { toggle(false); setIdx(tv / dt); }
  });
  const keyHandler = (e) => {
    if (e.key === " ") { e.preventDefault(); toggle(); }
    if (e.key === "ArrowLeft" && pos > 0) location.hash = `#/episode/${encodeURIComponent(list[pos - 1])}`;
    if (e.key === "ArrowRight" && pos >= 0 && pos < list.length - 1)
      location.hash = `#/episode/${encodeURIComponent(list[pos + 1])}`;
  };
  window.addEventListener("keydown", keyHandler);
  stopPlayer = () => { toggle(false); window.removeEventListener("keydown", keyHandler); };
  setIdx(0);
  toggle(true); // auto-start replay
}

/* ---------------- ④ Metric registry ---------------- */
let metricFields = null;  // {fields, aggregations} cache for the registration form

async function renderMetrics(params) {
  setView(`<h1>Metric registry</h1><div class="loading">Loading…</div>`);
  let reg, custom = [], fields = null;
  try {
    reg = await loadRegistry();
    if (MODE === "server") {
      [custom, fields] = await Promise.all([
        fetchJSON("/api/custom-metrics", { fresh: true }).then((d) => d.metrics),
        metricFields ? Promise.resolve(metricFields)
          : fetchJSON("/api/metric-fields").then((d) => (metricFields = d))]);
    }
  } catch (e) { return errorView(e, "router()"); }

  const q = (params.q || "").toLowerCase();
  const builtin = Object.entries(reg).filter(([k, v]) =>
    !q || k.includes(q) || v.definition.toLowerCase().includes(q) ||
    v.improvement_actions.join(" ").toLowerCase().includes(q));
  const customMatch = custom.filter((m) =>
    !q || m.metric_id.toLowerCase().includes(q) || m.definition.toLowerCase().includes(q) ||
    (m.improvement_actions || []).join(" ").toLowerCase().includes(q));

  const ownerBadge = (o) => `<span class="badge attr-${o === "hardware" ? "hardware" : o === "model" ? "model" : "ambiguous"}">${esc(o)}</span>`;
  const customRows = customMatch.map((m) => `<tr>
      <td><code>${esc(m.metric_id)}</code> <span class="badge gen">custom</span></td>
      <td>${esc(m.level)}</td><td>${ownerBadge(m.owner)}</td>
      <td>${esc(m.definition)}<div style="color:#888;font-size:11px;margin-top:2px">
        <code>${esc(m.agg)}(${esc(m.expr)})</code></div></td>
      <td>${(m.improvement_actions || []).map((a) => `• ${esc(a)}`).join("<br>")}
        <div style="margin-top:4px"><button class="linkish" onclick="deleteCustomMetric('${esc(m.metric_id)}')">delete</button></div></td>
    </tr>`).join("");
  const builtinRows = builtin.map(([k, v]) => `<tr><td><code>${esc(k)}</code></td>
      <td>${esc(v.level)}</td><td>${ownerBadge(v.owner)}</td>
      <td>${esc(v.definition)}</td>
      <td>${v.improvement_actions.map((a) => `• ${esc(a)}`).join("<br>")}</td></tr>`).join("");

  const total = Object.keys(reg).length + custom.length;
  const shown = builtin.length + customMatch.length;
  setView(`<h1>Metric registry <span style="font-size:12px;color:#888;font-weight:400">
      R-8 machine check: every metric must bind ≥1 improvement action</span></h1>
    ${MODE === "server" ? registerFormHtml(fields) : `<div class="meta">Registering metrics needs server mode:
      <code>python serve.py</code></div>`}
    <div class="filters"><label>Search
      <input style="width:240px" value="${esc(params.q || "")}" placeholder="metric id / definition / action"
        onchange="location.hash='#/metrics'+(this.value?('?q='+encodeURIComponent(this.value)):'')"></label>
      <span class="count">${shown} / ${total} items</span></div>
    ${shown === 0 ? '<div class="empty-box">No matching metrics</div>' :
      `<table class="grid"><tr><th>Metric</th><th>Level</th><th>Owner</th><th>Definition</th><th>Improvement actions (FR-5.1)</th></tr>
      ${customRows}${builtinRows}</table>`}`);
}

function registerFormHtml(fields) {
  const f = fields || { fields: [], aggregations: [] };
  const fieldChips = f.fields.map((x) =>
    `<code title="${esc(x.description)}" class="field-chip" onclick="insertField('${esc(x.name)}')">${esc(x.name)}</code>`).join(" ");
  const aggOpts = f.aggregations.map((a) => `<option value="${esc(a)}">${esc(a)}</option>`).join("");
  return `<details class="reg-form"><summary>＋ Register a metric</summary>
    <div class="reg-grid">
      <label>id <input id="cm-id" placeholder="custom.my_metric"></label>
      <label>level <select id="cm-level"><option>L0</option><option>L1</option><option selected>L2</option><option>L3</option></select></label>
      <label>owner <select id="cm-owner"><option>model</option><option>hardware</option><option>both</option><option>environment</option></select></label>
      <label>aggregation <select id="cm-agg">${aggOpts}</select></label>
      <label class="wide">formula (per episode) <input id="cm-expr" placeholder="e.g. e_track_steady_rms_mm > 0.5"></label>
      <label class="wide">definition <input id="cm-def" placeholder="what this metric means"></label>
      <label class="wide">improvement actions (one per line) <textarea id="cm-actions" rows="2" placeholder="bind ≥1 action (R-8)"></textarea></label>
    </div>
    <div class="field-hints">fields: ${fieldChips} · functions: <code>abs min max</code> · aggregations: <code>${esc(f.aggregations.join(" "))}</code></div>
    <button onclick="addCustomMetric()">Register metric</button>
  </details>`;
}

window.insertField = (name) => {
  const el = document.getElementById("cm-expr");
  if (!el) return;
  const s = el.selectionStart ?? el.value.length;
  el.value = el.value.slice(0, s) + name + el.value.slice(el.selectionEnd ?? s);
  el.focus();
};

window.addCustomMetric = async () => {
  const val = (id) => (document.getElementById(id)?.value || "").trim();
  const body = {
    metric_id: val("cm-id"), level: val("cm-level"), owner: val("cm-owner"),
    agg: val("cm-agg"), expr: val("cm-expr"), definition: val("cm-def"),
    improvement_actions: val("cm-actions"),
  };
  try {
    const resp = await fetch("/api/custom-metrics", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
    toast(`Registered ${body.metric_id}`);
    router();  // re-render the registry
  } catch (e) { toast(`Register failed: ${e.message}`, "fail"); }
};

window.deleteCustomMetric = async (id) => {
  try {
    const resp = await fetch(`/api/custom-metrics/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
    toast(`Deleted ${id}`);
    router();
  } catch (e) { toast(`Delete failed: ${e.message}`, "fail"); }
};

/* ================= M-FE2 server-mode views (fe-rq.md §4.2/§4.3) ================= */

const STATUS_BADGE = {
  running: '<span class="badge st-running">● running</span>',
  done: '<span class="badge ok">done</span>',
  failed: '<span class="badge fail">failed</span>',
  cancelled: '<span class="badge attr-ambiguous">cancelled</span>',
};

/* ---------- ②a Run list ---------- */
async function renderRuns() {
  if (MODE !== "server")
    return setView(`<h1>Runs</h1><div class="empty-box">Launching runs requires server mode (M-FE2):
      <div style="margin-top:8px"><code>python serve.py</code> then open <code>http://127.0.0.1:8000/</code></div></div>`);
  setView(`<h1>Runs</h1><div class="loading">Loading…</div>`);
  let runs;
  try { runs = await fetchRuns(); } catch (e) { return errorView(e, "router()"); }
  const rows = runs.map((r) => `<tr class="clickable"
      onclick="location.hash='#/runs/${encodeURIComponent(r.run_id)}'">
    <td><code>${esc(r.run_id)}</code></td>
    <td>${STATUS_BADGE[r.status] || esc(r.status)}</td>
    <td>${esc(r.created_at)}</td>
    <td>${r.config.seed}</td>
    <td>${r.combos}</td>
    <td>${r.done_episodes}/${r.total_episodes}</td>
  </tr>`).join("");
  setView(`<h1>Runs</h1>
    <div class="meta">Launch from the browser → live SSE progress → consume results at the same URL (M-FE2)
      <button style="float:right" onclick="location.hash='#/runs/new'">＋ New run</button></div>
    ${runs.length === 0
      ? `<div class="empty-box">No runs yet<div style="margin-top:10px">
          <button onclick="location.hash='#/runs/new'">New run</button></div></div>`
      : `<table class="grid"><tr><th>Run</th><th>Status</th><th>Created</th>
          <th>seed</th><th>Combos</th><th>Episode progress</th></tr>${rows}</table>`}`);
}

/* ---------- ②b New-run wizard (3 steps, no scene editor — that's M-FE3) ---------- */
const DRAFT_KEY = "pabench-run-draft";
const WIZ_DEFAULT = () => ({
  step: 1, model_ids: [], hw_ids: [], backend: "fake",
  nominal: true, mutation_on: true, mutation_episodes: 24,
  pos_range_mm: 15, yaw_range_rad: 0.3, lux_min: 0.3, lux_max: 1.0,
  friction_min: 0.6, friction_max: 1.2,
  metamorphic: true, pace_ms: 0,
  seed: Math.floor(Math.random() * 100000),  // random by default and shown; can be fixed for reproducibility (NFR-1)
});
let wiz = null;
function wizSave() { localStorage.setItem(DRAFT_KEY, JSON.stringify(wiz)); }
window.wizSet = (k, v) => { wiz[k] = v; wizSave(); renderRunsNew(); };
window.wizNum = (k, v) => window.wizSet(k, Number(v));
window.wizToggle = (k, list, v) => {
  const i = wiz[list].indexOf(k);
  if (v && i < 0) wiz[list].push(k); else if (!v && i >= 0) wiz[list].splice(i, 1);
  wizSave(); renderRunsNew();
};

function wizCombos() { return wiz.model_ids.length * wiz.hw_ids.length; }
function wizPerCombo() {
  return (wiz.nominal ? 1 : 0) + (wiz.mutation_on ? wiz.mutation_episodes : 0)
       + (wiz.metamorphic ? 3 : 0);
}
function wizTotal() { return wizCombos() * wizPerCombo(); }
function wizBody() {
  return {
    model_ids: wiz.model_ids, hw_ids: wiz.hw_ids, backend: wiz.backend || "fake", seed: wiz.seed,
    nominal: wiz.nominal,
    mutation_episodes: wiz.mutation_on ? wiz.mutation_episodes : 0,
    metamorphic: wiz.metamorphic,
    pos_range_m: wiz.pos_range_mm / 1000, yaw_range_rad: wiz.yaw_range_rad,
    lux_range: [wiz.lux_min, wiz.lux_max],
    friction_range: [wiz.friction_min, wiz.friction_max],
    pace_s: wiz.pace_ms / 1000,
  };
}
function wizProblems() {
  const p = [];
  if (!wiz.model_ids.length) p.push("Step ①: select at least 1 model");
  if (!wiz.hw_ids.length) p.push("Step ①: select at least 1 hardware profile");
  if (!wiz.nominal && wiz.metamorphic) p.push("Step ②: MR-1 needs a nominal source episode");
  if (!wiz.nominal && !wiz.mutation_on) p.push("Step ②: enable at least one scene strategy");
  if (wiz.mutation_on && wiz.mutation_episodes < 1) p.push("Step ②: mutation budget ≥1");
  return p;
}

async function renderRunsNew() {
  if (MODE !== "server") return renderRuns();
  if (!wiz) {
    try { wiz = JSON.parse(localStorage.getItem(DRAFT_KEY)) || WIZ_DEFAULT(); }
    catch { wiz = WIZ_DEFAULT(); }
  }
  let models, hardware, backends;
  try {
    [models, hardware, backends] = await Promise.all([
      fetchJSON("/api/models").then((d) => d.models),
      fetchJSON("/api/hardware").then((d) => d.hardware),
      fetchJSON("/api/backends").then((d) => d.backends)]);
  } catch (e) { return errorView(e, "router()"); }
  // if the saved draft picked a backend that's no longer available, fall back to fake
  if (!backends.some((b) => b.id === wiz.backend && b.available)) wiz.backend = "fake";
  if (!models.length)
    return setView(`<h1>New run</h1><div class="empty-box">Model list is empty —
      register a model in MODEL_REGISTRY in <code>pabench/pipeline.py</code> (FR-2.4)</div>`);

  const stepNav = [1, 2, 3].map((s) => `<div class="step ${wiz.step === s ? "active" : ""} ${wiz.step > s ? "done" : ""}"
      onclick="wizSet('step', ${s})">${wiz.step > s ? "✓" : s}. ${["Subjects", "Scenes & strategy", "Confirm"][s - 1]}</div>`).join("");

  let stepHtml = "";
  if (wiz.step === 1) {
    stepHtml = `
      <h3>Models (multi-select)</h3>
      ${models.map((m) => `<label class="pick">
        <input type="checkbox" ${wiz.model_ids.includes(m.model_id) ? "checked" : ""}
          onchange="wizToggle('${esc(m.model_id)}', 'model_ids', this.checked)">
        <code>${esc(m.model_id)}</code>
        ${m.provides_uncertainty ? '<span class="badge gen">uncertainty ✓</span>' : '<span class="badge attr-ambiguous">no uncertainty</span>'}
        ${m.fake ? '<span class="badge attr-ambiguous">FAKE</span>' : ""}
      </label>`).join("")}
      <h3>Hardware profiles (multi-select)</h3>
      ${hardware.map((h) => `<label class="pick ${h.stale ? "stale" : ""}">
        <input type="checkbox" ${wiz.hw_ids.includes(h.hw_config_id) ? "checked" : ""}
          onchange="wizToggle('${esc(h.hw_config_id)}', 'hw_ids', this.checked)">
        <code>${esc(h.hw_config_id)}</code>
        <span class="badge gen">calibrated ${esc(h.calibrated || "?")}</span>
        ${h.stale ? '<span class="badge warn">⚠ calibration stale (NFR-2)</span>' : ""}
      </label>`).join("")}
      <h3>Physics backend (NFR-5 plug-and-play)</h3>
      ${backends.map((b) => `<label class="pick ${b.available ? "" : "disabled"}" title="${esc(b.note)}">
        <input type="radio" name="backend" ${b.id === wiz.backend ? "checked" : ""}
          ${b.available ? "" : "disabled"} onchange="wizSet('backend', '${esc(b.id)}')">
        <code>${esc(b.id)}</code>
        ${b.available ? '<span class="badge gen">available</span>' : `<span class="badge attr-ambiguous">unavailable</span>`}
        <span style="color:#888;font-size:11px">${esc(b.note)}</span>
      </label>`).join("")}
      <h3>Task × tolerance matrix</h3>
      <label class="pick"><input type="checkbox" checked disabled> screw_cap × T1 (cap fastening · tolerance 1.0mm)</label>
      <label class="pick disabled"><input type="checkbox" disabled> insert_part × T2 <span class="badge attr-ambiguous">M4</span></label>
      <label class="pick disabled"><input type="checkbox" disabled> snap_fit × T3 <span class="badge attr-ambiguous">M4</span></label>`;
  } else if (wiz.step === 2) {
    const slider = (k, label, min, max, step, unit) => `<label class="param">${label}
      <input type="range" min="${min}" max="${max}" step="${step}" value="${wiz[k]}"
        oninput="this.nextElementSibling.textContent=this.value+'${unit}'" onchange="wizNum('${k}', this.value)">
      <span class="val">${wiz[k]}${unit}</span></label>`;
    stepHtml = `
      <div class="strategy-cards">
        <div class="scard ${wiz.nominal ? "on" : ""}" onclick="wizSet('nominal', ${!wiz.nominal})">
          <b>nominal</b><p>nominal-scene baseline (MR-1 source episode)</p>
          <span class="budget">1 episode/combo</span></div>
        <div class="scard ${wiz.mutation_on ? "on" : ""}" onclick="wizSet('mutation_on', ${!wiz.mutation_on})">
          <b>mutation</b><p>parametric pose/lighting/friction perturbation (FR-1.2)</p>
          <span class="budget" onclick="event.stopPropagation()">budget
            <input type="number" min="1" max="500" value="${wiz.mutation_episodes}"
              onchange="wizNum('mutation_episodes', this.value)"> episodes/combo</span></div>
        <div class="scard ${wiz.metamorphic ? "on" : ""}" onclick="wizSet('metamorphic', ${!wiz.metamorphic})">
          <b>metamorphic (MR-1)</b><p>rotational-equivariance metamorphic test (FR-1.3)</p>
          <span class="budget">3 episodes/combo</span></div>
        <div class="scard disabled"><b>adversarial_opt</b><p>CMA-ES failure-boundary search</p>
          <span class="badge attr-ambiguous">M3</span></div>
      </div>
      ${wiz.mutation_on ? `<h3>mutation parameters (= MutationGenerator args, FR-FE-2.1)</h3>
      <div class="params">
        ${slider("pos_range_mm", "pose perturbation ±", 1, 30, 1, "mm")}
        ${slider("yaw_range_rad", "yaw perturbation ±", 0.05, 0.6, 0.05, "rad")}
        ${slider("lux_min", "lighting min", 0.1, 1.0, 0.05, "")}
        ${slider("lux_max", "lighting max", 0.1, 1.0, 0.05, "")}
        ${slider("friction_min", "friction min", 0.2, 1.5, 0.05, "")}
        ${slider("friction_max", "friction max", 0.2, 1.5, 0.05, "")}
      </div>
      <button class="linkish" onclick="wizPreview()">Preview sampling (first 12 scenes, no execution)</button>
      <div id="preview-box"></div>` : ""}
      <h3>Demo throttle (optional)</h3>
      <label class="param">per-episode interval
        <input type="number" min="0" max="2000" step="50" value="${wiz.pace_ms}"
          onchange="wizNum('pace_ms', this.value)"> ms
        <span style="color:#888;font-size:11px">FakeSim runs ~1s total; slow it down to watch progress; 0 = full speed</span></label>`;
  } else {
    const est = (wizTotal() / 110 + wizTotal() * wiz.pace_ms / 1000).toFixed(1);
    const problems = wizProblems();
    stepHtml = `
      <h3>Confirm and launch</h3>
      <label class="param">seed
        <input type="number" value="${wiz.seed}" onchange="wizNum('seed', this.value)">
        <button class="linkish" onclick="wizNum('seed', Math.floor(Math.random()*100000))">Random</button>
        <span style="color:#888;font-size:11px">a fixed seed makes two runs produce identical artifacts (NFR-1 / FR-FE-2.1)</span></label>
      <div class="confirm-grid">
        <div>Combos</div><div>${wiz.model_ids.length} models × ${wiz.hw_ids.length} hardware = ${wizCombos()}</div>
        <div>Episodes/combo</div><div>${wizPerCombo()}</div>
        <div>Total episodes</div><div><b>${wizTotal()}</b></div>
        <div>Estimated time</div><div>≈ ${est}s (FakeSim throughput ~110 episodes/s${wiz.pace_ms ? ` + throttle ${wiz.pace_ms}ms/episode` : ""})</div>
      </div>
      ${problems.length ? `<div class="error-box">${problems.map(esc).join("<br>")}</div>` : ""}
      <button id="btn-launch" ${problems.length ? "disabled" : ""} onclick="wizLaunch()">🚀 Launch run</button>`;
  }

  const sum = `
    <h3>Selection summary</h3>
    <div class="sum-row">Models</div>${wiz.model_ids.map((m) => `<code>${esc(m)}</code>`).join(" ") || "<i>none</i>"}
    <div class="sum-row">Hardware</div>${wiz.hw_ids.map((h) => `<code>${esc(h)}</code>`).join(" ") || "<i>none</i>"}
    <div class="sum-row">Backend</div><code>${esc(wiz.backend || "fake")}</code>
    <div class="sum-row">Strategy</div>${[wiz.nominal && "nominal",
      wiz.mutation_on && `mutation×${wiz.mutation_episodes}`,
      wiz.metamorphic && "MR-1×3"].filter(Boolean).join(" + ") || "<i>none</i>"}
    <div class="sum-row">Budget</div>total <b>${wizTotal()}</b> episodes · seed ${wiz.seed}
    <div style="margin-top:14px">
      ${wiz.step > 1 ? `<button onclick="wizSet('step', ${wiz.step - 1})">← Back</button>` : ""}
      ${wiz.step < 3 ? `<button onclick="wizSet('step', ${wiz.step + 1})">Next →</button>` : ""}
      <button class="linkish" onclick="wizReset()">Reset draft</button>
    </div>`;

  setView(`<h1>New run <span style="font-size:12px;color:#888;font-weight:400">
      3-step config and launch · draft auto-saved (localStorage)</span></h1>
    <div class="wizard">
      <div class="wiz-nav">${stepNav}</div>
      <div class="wiz-main">${stepHtml}</div>
      <div class="wiz-sum">${sum}</div>
    </div>`);
}

window.wizReset = () => {
  localStorage.removeItem(DRAFT_KEY);
  wiz = null;
  renderRunsNew();
};

window.wizPreview = async () => {
  const box = document.getElementById("preview-box");
  box.innerHTML = '<div class="loading">Sampling…</div>';
  try {
    const resp = await fetch("/api/scenes/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...wizBody(), n: 12 }) });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
    const scenes = (await resp.json()).scenes;
    box.innerHTML = `<div class="preview-grid">${scenes.map((s) => {
      const p = s.perturbation, lum = Math.round(40 + p.lux_factor * 60);
      return `<div class="pcard" title="${esc(s.scene_id)}">
        <div class="lux" style="background:hsl(45 80% ${lum}%)">lux ${p.lux_factor.toFixed(2)}</div>
        <div>dx ${(p.part_dx * 1000).toFixed(1)}mm · dy ${(p.part_dy * 1000).toFixed(1)}mm</div>
        <div>dyaw ${p.part_dyaw.toFixed(2)}rad · μ ${p.friction.toFixed(2)}</div></div>`;
    }).join("")}</div>`;
  } catch (e) { box.innerHTML = `<div class="error-box">Preview failed: ${esc(e.message)}</div>`; }
};

window.wizLaunch = async () => {
  const btn = document.getElementById("btn-launch");
  btn.disabled = true; btn.textContent = "Submitting…";
  try {
    const resp = await fetch("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(wizBody()) });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
    const d = await resp.json();
    toast(`Run launched: ${d.run_id} (${d.total_episodes} episodes)`);
    location.hash = `#/runs/${encodeURIComponent(d.run_id)}`;
  } catch (e) {
    // launch failed: toast shows the backend message, config is kept (§4.2 error state)
    toast(`Launch failed: ${e.message}`, "fail");
    btn.disabled = false; btn.textContent = "🚀 Launch run";
  }
};

/* ---------- ②c Run detail: progress → results (§4.3) ---------- */
async function renderRunDetail(id) {
  if (MODE !== "server") return renderRuns();
  setView(`<h1>Run detail</h1><div class="loading">Loading…</div>`);
  let run;
  try { run = await fetchJSON(`/api/runs/${encodeURIComponent(id)}`, { fresh: true }); }
  catch (e) { return errorView(e, "router()"); }

  if (run.status === "done") { RUN_CTX = id; return renderRun(id); }

  const rebuildBtn = `<button onclick="rebuildRun('${esc(id)}')">Rebuild with same config</button>`;
  if (run.status === "failed")
    return setView(`<h1>Run detail <code>${esc(id)}</code></h1>
      <div class="meta">${STATUS_BADGE.failed} · seed ${run.config.seed} · ${run.done_episodes}/${run.total_episodes} episodes</div>
      <div class="error-box">Run failed
        <details style="margin-top:8px"><summary>Backend error log</summary>
          <pre style="white-space:pre-wrap;font-size:11px">${esc(run.error || "(no log)")}</pre></details>
        <div style="margin-top:10px">${rebuildBtn}</div></div>`);
  if (run.status === "cancelled")
    return setView(`<h1>Run detail <code>${esc(id)}</code></h1>
      <div class="meta">${STATUS_BADGE.cancelled} · seed ${run.config.seed}</div>
      <div class="empty-box">Run cancelled (completed ${run.done_episodes}/${run.total_episodes} episodes, artifacts not persisted)
        <div style="margin-top:10px">${rebuildBtn}</div></div>`);

  // ---- running: progress area + live SSE event stream (FR-FE-3.1) ----
  const combos = [];
  run.config.model_ids.forEach((m) => run.config.hw_ids.forEach((h) => combos.push(`${m} @ ${h}`)));
  const perCombo = Math.round(run.total_episodes / combos.length);
  setView(`<h1>Run detail <code>${esc(id)}</code></h1>
    <div class="meta">${STATUS_BADGE.running} · benchmark ${esc(run.benchmark_version)} ·
      seed ${run.config.seed} · ${combos.length} combos × ${perCombo} episodes
      <button style="float:right" onclick="cancelRun('${esc(id)}')">Cancel run</button></div>
    <div id="sse-banner" class="banner" style="display:none">⚠ Progress stream offline, reconnecting… (auto-fallback to 5s polling)</div>
    <div class="progress-wrap">
      <div class="pg-label">Total progress <span id="pg-text">${run.done_episodes}/${run.total_episodes}</span></div>
      <div class="pg-bar"><div id="pg-fill" style="width:${run.total_episodes ? run.done_episodes / run.total_episodes * 100 : 0}%"></div></div>
      ${combos.map((c, i) => `<div class="pg-label small">${esc(c)}
          <span id="pg-combo-t-${i}">${run.combo_progress[c] || 0}/${perCombo}</span></div>
        <div class="pg-bar small"><div id="pg-combo-${i}" style="width:${(run.combo_progress[c] || 0) / perCombo * 100}%"></div></div>`).join("")}
    </div>
    <h2>Live event stream (last 20)</h2>
    <div id="ev-list" class="ev-list"><div class="loading">Waiting for events…</div></div>`);

  watchRun(id, combos, perCombo);
}

window.cancelRun = async (id) => {
  if (!confirm(`Cancel run ${id}? Completed episodes will not have artifacts persisted.`)) return;  // §6 one of only two confirm dialogs
  try {
    const resp = await fetch(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
    toast("Cancellation requested");
  } catch (e) { toast(`Cancel failed: ${e.message}`, "fail"); }
};

window.rebuildRun = async (id) => {
  try {
    const run = await fetchJSON(`/api/runs/${encodeURIComponent(id)}`, { fresh: true });
    const resp = await fetch("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(run.config) });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
    const d = await resp.json();
    toast(`Rebuilt: ${d.run_id}`);
    location.hash = `#/runs/${encodeURIComponent(d.run_id)}`;
  } catch (e) { toast(`Rebuild failed: ${e.message}`, "fail"); }
};

function watchRun(id, combos, perCombo) {
  let es = null, pollTimer = null, retries = 0, closed = false;
  const comboDone = {};
  const $ = (i) => document.getElementById(i);

  function applyProgress(done, total) {
    if ($("pg-text")) $("pg-text").textContent = `${done}/${total}`;
    if ($("pg-fill")) $("pg-fill").style.width = `${total ? done / total * 100 : 0}%`;
  }
  function applyCombo(combo, n) {
    const i = combos.indexOf(combo);
    if (i < 0) return;
    if ($(`pg-combo-${i}`)) $(`pg-combo-${i}`).style.width = `${n / perCombo * 100}%`;
    if ($(`pg-combo-t-${i}`)) $(`pg-combo-t-${i}`).textContent = `${n}/${perCombo}`;
  }
  function pushEvent(html) {
    const list = $("ev-list");
    if (!list) return;
    if (list.firstElementChild?.classList.contains("loading")) list.innerHTML = "";
    const el = document.createElement("div");
    el.className = "ev"; el.innerHTML = html;
    list.prepend(el);
    while (list.children.length > 20) list.lastChild.remove();
  }
  function finish() {
    cleanup();
    if (!closed) { closed = true; renderRunDetail(id); }
  }
  function handle(e) {
    if (e.type === "episode_done") {
      applyProgress(e.done, e.total);
      comboDone[e.combo] = (comboDone[e.combo] || 0) + 1;
      applyCombo(e.combo, comboDone[e.combo]);
      pushEvent(`<code>${esc(e.episode_id.split("__")[0])}</code> ${
        e.success ? '<span class="badge ok">success</span>'
                  : `<span class="badge fail">failure</span> ${esc(e.failure_label || "")}`}`);
    } else if (e.type === "combo_start") {
      pushEvent(`▶ start combo <b>${esc(e.combo)}</b> (${e.index + 1}/${e.total_combos})`);
    } else if (e.type === "combo_done") {
      pushEvent(`✓ combo done <b>${esc(e.combo)}</b> · SR ${fmt(e.summary.sr)}`);
    } else if (e.type === "run_done") {
      toast("Run complete", "ok"); finish();
    } else if (e.type === "cancelled" || e.type === "error") {
      finish();
    }
  }
  function startSSE() {
    // SSE replays full history from seq=0 on every connect → zero local counts before reconnect to avoid doubling
    Object.keys(comboDone).forEach((k) => delete comboDone[k]);
    const list = $("ev-list");
    if (list) list.innerHTML = '<div class="loading">Waiting for events…</div>';
    es = new EventSource(`/api/runs/${encodeURIComponent(id)}/events`);
    ["episode_done", "combo_start", "combo_done", "run_done", "cancelled", "error", "scenes_ready"]
      .forEach((t) => es.addEventListener(t, (ev) => handle({ type: t, ...JSON.parse(ev.data) })));
    es.addEventListener("run_closed", finish);
    es.onopen = () => { retries = 0; if ($("sse-banner")) $("sse-banner").style.display = "none"; };
    es.onerror = () => {
      es.close();
      if ($("sse-banner")) $("sse-banner").style.display = "block";
      retries += 1;
      if (retries <= 3) setTimeout(() => { if (!closed) startSSE(); }, 1000 * 2 ** (retries - 1));
      else startPolling();  // after 3 exponential-backoff retries, fall back to 5s polling (§11)
    };
  }
  function startPolling() {
    pollTimer = setInterval(async () => {
      try {
        const p = await fetchJSON(`/api/runs/${encodeURIComponent(id)}/progress`, { fresh: true });
        applyProgress(p.done, p.total);
        Object.entries(p.combo_progress).forEach(([c, n]) => applyCombo(c, n));
        if (p.status !== "running") finish();
      } catch { /* service unreachable, retry next cycle */ }
    }, 5000);
  }
  function cleanup() { if (es) es.close(); if (pollTimer) clearInterval(pollTimer); }
  stopWatcher = () => { closed = true; cleanup(); };
  startSSE();
}

function updateNav() {
  const server = MODE === "server";
  document.getElementById("brand-mode").innerHTML =
    server ? "Console · M-FE2 server mode" : "Console · M-FE1";
  document.getElementById("nav-runs").style.display = server ? "" : "none";
  document.getElementById("nav-run").style.display = server ? "none" : "";
  document.getElementById("nav-note").innerHTML = server
    ? "Server mode: platform API<br><code>/api/*</code> + SSE progress"
    : "Static mode: data from<br><code>out/</code> benchmark artifacts";
}

/* ---------------- Router ---------------- */
async function router() {
  const { seg, params } = parseHash();
  document.querySelectorAll("#sidebar a").forEach((a) =>
    a.classList.toggle("active", a.dataset.route === (seg[0] === "episode" ? "episodes" : seg[0])));
  try {
    if (seg[0] === "run") await renderRun(null);
    else if (seg[0] === "runs" && seg[1] === "new") await renderRunsNew();
    else if (seg[0] === "runs" && seg[1]) await renderRunDetail(decodeURIComponent(seg[1]));
    else if (seg[0] === "runs") await renderRuns();
    else if (seg[0] === "episodes") await renderEpisodes(params);
    else if (seg[0] === "episode" && seg[1]) await renderEpisode(decodeURIComponent(seg[1]));
    else if (seg[0] === "metrics") await renderMetrics(params);
    else await renderOverview();
  } catch (e) { errorView(e, "router()"); }
}
window.router = router;
window.addEventListener("resize", () => { clearTimeout(window.__rs);
  window.__rs = setTimeout(() => charts.forEach((c) => c.resize()), 200); });
// startup: probe mode (server=M-FE2 / static=M-FE1) before entering the router
detectMode().then(() => {
  updateNav();
  window.addEventListener("hashchange", router);
  router();
});