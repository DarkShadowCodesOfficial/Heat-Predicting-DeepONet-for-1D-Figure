/* Heat-operator demo front-end (vanilla JS + Plotly). */
"use strict";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TABS = ["solution", "spacetime", "intervals", "speed", "training"];
const COL = { fem: "#111827", op: "#2563eb", band: "rgba(37,99,235,.16)", ana: "#d97706",
              ic: "#9ca3af", err: "#dc2626" };
const INFERNO = [[0, "#000004"], [0.13, "#1f0c48"], [0.25, "#550f6d"], [0.38, "#88226a"], [0.5, "#a83655"],
                 [0.63, "#e35933"], [0.75, "#f9950a"], [0.88, "#f8c932"], [1, "#fcffa4"]];
const ERRSCALE = [[0, "#fff7ec"], [0.5, "#fdbb84"], [1, "#e34a33"]];   // light enough for dark cell labels
const PCFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

const S = {
  cfg: null, ic: [], alpha: 1.5e-3, mode: "linear", t: 600, reference: "fem",
  sim: null, snap: null, simId: 0, snapId: 0, snapTimer: null, runTimer: null,
  tab: "solution", dirty: new Set(TABS), play: null,
};

/* ----------------------------------------------------------------- helpers */
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
function fmtDur(s) {
  if (s === null || s === undefined) return "–";
  if (s < 1e-3) return (s * 1e6).toFixed(0) + " µs";
  if (s < 1) return (s * 1e3).toFixed(s < 0.01 ? 2 : 1) + " ms";
  return s.toFixed(2) + " s";
}
function fmtX(v) { return v === null || v === undefined ? "–" : (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "×"; }
const mins = (s) => (s / 60).toFixed(s % 60 ? 1 : 0) + " min";
function stat(k, v, unit = "", sub = "", cls = "") {
  return `<div class="stat ${cls}"><div class="k">${k}</div><div class="v">${v}<span class="u">${unit}</span></div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}
function binLabel([lo, hi]) { return `${lo / 60}–${hi / 60} min`; }

/* ------------------------------------------------------------ boot / status */
async function boot() {
  let st = null;
  for (;;) {
    try {
      st = await (await fetch("/api/status")).json();
      renderStatus(st);
      if (st.state === "ready" || st.state === "error") break;
    } catch (e) { /* server still starting */ }
    await sleep(700);
  }
  if (st.state === "ready") {
    S.cfg = await (await fetch("/api/config")).json();
    init();
  }
}

function renderStatus(st) {
  const pill = $("statusPill");
  const p = st.progress;
  pill.className = "status " + (st.state === "ready" ? "ready" : st.state === "error" ? "error" : "");
  $("statusText").textContent = st.state === "ready" ? "operator ready"
    : st.state === "error" ? "error" : st.state === "training" ? `training ${(p.fraction * 100).toFixed(0)}%` : "loading…";
  const ov = $("overlay");
  if (st.state === "ready") { ov.classList.add("hidden"); return; }
  ov.classList.remove("hidden");
  $("ovTitle").textContent = st.state === "training" ? "Training the operator" : "Loading the operator";
  $("ovBar").style.width = (p.fraction * 100).toFixed(1) + "%";
  $("ovPct").textContent = (p.fraction * 100).toFixed(0) + "%";
  $("ovStage").textContent = p.stage;
  $("ovMsg").textContent = p.message;
  $("ovEta").textContent = p.eta_s ? `~${Math.max(1, Math.round(p.eta_s))} s left · ${Math.round(p.elapsed_s)} s elapsed` : `${Math.round(p.elapsed_s)} s elapsed`;
  const log = $("ovLog");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = p.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
  if (st.state === "error") { $("ovError").hidden = false; $("ovError").textContent = st.error; }
}

/* -------------------------------------------------------------------- init */
function init() {
  const c = S.cfg;
  const a = $("alpha");
  a.min = c.alpha_range[0]; a.max = c.alpha_range[1];
  S.alpha = 0.5 * (c.alpha_range[0] + c.alpha_range[1]);
  a.value = S.alpha;
  $("alphaLo").textContent = fmtAlpha(c.alpha_range[0]);
  $("alphaHi").textContent = fmtAlpha(c.alpha_range[1]);
  $("tSlider").max = c.max_time;
  S.t = 600;
  $("tSlider").value = S.t;

  applyPreset("bump", false);
  updateReadouts();
  wireControls();
  renderEditor();
  renderTraining();
  runSimulate();
}
function fmtAlpha(a) { return (a * 1e3).toFixed(2) + "×10⁻³"; }

function updateReadouts() {
  $("alphaOut").textContent = fmtAlpha(S.alpha);
  $("tOut").textContent = `${S.t} s  (${mins(S.t)})`;
  const c = S.cfg;
  $("modeHint").innerHTML = S.mode === "linear"
    ? `Constant diffusivity. Ensemble of ${c.training ? c.training.config.ens_n : 3} active-learned DeepONets, shown with a ±2σ uncertainty band.`
    : `α(T) = α₀(1 + β(T − 20)) with <b>β = ${c.beta}</b>, fixed at training time. Single DeepONet trained on Picard-FEM data, so no uncertainty band. No closed form exists.`;
}

/* --------------------------------------------------------------- IC editor */
const ED = { W: 330, H: 230, l: 38, r: 12, t: 12, b: 26, ymin: -10, ymax: 110 };
const px = (i) => ED.l + (i / (S.cfg.sensors - 1)) * (ED.W - ED.l - ED.r);
const py = (v) => ED.t + (1 - (v - ED.ymin) / (ED.ymax - ED.ymin)) * (ED.H - ED.t - ED.b);
const vy = (y) => ED.ymin + (1 - (y - ED.t) / (ED.H - ED.t - ED.b)) * (ED.ymax - ED.ymin);

function renderEditor(sel = -1) {
  const n = S.cfg.sensors, ic = S.ic;
  let g = "";
  for (let v = 0; v <= 100; v += 20) {
    g += `<line x1="${ED.l}" x2="${ED.W - ED.r}" y1="${py(v)}" y2="${py(v)}" stroke="#e4e7ec"/>` +
         `<text x="${ED.l - 6}" y="${py(v) + 3.5}" text-anchor="end" font-size="10" fill="#667085">${v}</text>`;
  }
  for (const xv of [0, 0.5, 1]) {
    const i = xv * (n - 1);
    g += `<text x="${px(i)}" y="${ED.H - 9}" text-anchor="middle" font-size="10" fill="#667085">${xv}</text>`;
  }
  const pts = ic.map((v, i) => `${px(i)},${py(v)}`).join(" ");
  g += `<polyline points="${pts}" fill="none" stroke="#2563eb" stroke-width="2"/>`;
  ic.forEach((v, i) => {
    if (i === 0) g += `<rect x="${px(0) - 5}" y="${py(v) - 5}" width="10" height="10" rx="2" fill="#111827"><title>heater (fixed)</title></rect>`;
    else g += `<circle cx="${px(i)}" cy="${py(v)}" r="${i === sel ? 7 : 5}" fill="${i === sel ? "#1d4ed8" : "#fff"}" stroke="#2563eb" stroke-width="2"/>`;
  });
  $("icEditor").innerHTML = g;
}

function wireEditor() {
  const svg = $("icEditor");
  let drag = null;
  const locate = (e) => {
    const r = svg.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * ED.W, sy = ((e.clientY - r.top) / r.height) * ED.H;
    const n = S.cfg.sensors;
    const idx = Math.max(0, Math.min(n - 1, Math.round(((sx - ED.l) / (ED.W - ED.l - ED.r)) * (n - 1))));
    return { idx, v: Math.max(ED.ymin, Math.min(ED.ymax, vy(sy))) };
  };
  const readout = (i) => {
    $("icReadout").textContent = i < 0 ? "\u00a0" :
      `x = ${(i / (S.cfg.sensors - 1)).toFixed(2)}   T = ${S.ic[i].toFixed(1)} °C` + (i === 0 ? "   (heater, fixed)" : "");
  };
  svg.addEventListener("pointerdown", (e) => {
    const { idx, v } = locate(e);
    if (idx === 0) { readout(0); return; }
    svg.setPointerCapture(e.pointerId);
    drag = { idx, base: S.ic.slice(), smooth: e.shiftKey || $("brush").checked };
    move(e);
  });
  const move = (e) => {
    const { idx: hover, v } = locate(e);
    if (!drag) { readout(hover); return; }
    const dv = Math.round(v * 10) / 10 - drag.base[drag.idx];
    for (let j = 1; j < S.cfg.sensors; j++) {
      const w = drag.smooth ? Math.exp(-((j - drag.idx) ** 2) / (2 * 1.8 ** 2)) : (j === drag.idx ? 1 : 0);
      if (w > 1e-3) S.ic[j] = Math.round((drag.base[j] + dv * w) * 10) / 10;
    }
    renderEditor(drag.idx);
    readout(drag.idx);
  };
  svg.addEventListener("pointermove", move);
  const end = () => { if (drag) { drag = null; renderEditor(); scheduleRun(); } };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("pointerleave", () => { if (!drag) readout(-1); });
}

/* ----------------------------------------------------------------- presets */
function profile(fn) {
  const n = S.cfg.sensors;
  const ic = Array.from({ length: n }, (_, i) => fn(i / (n - 1)));
  ic[0] = S.cfg.heater;
  return ic.map((v) => Math.round(Math.max(ED.ymin, Math.min(ED.ymax, v)) * 10) / 10);
}
const gauss = (x, c, w) => Math.exp(-((x - c) ** 2) / (2 * w * w));

async function applyPreset(name, run = true) {
  switch (name) {
    case "random": {
      const d = await (await fetch("/api/random_ic")).json();
      S.ic = d.ic; S.alpha = d.alpha; $("alpha").value = d.alpha; break;
    }
    case "uniform": S.ic = profile(() => 20); break;
    case "bump": S.ic = profile((x) => 22 + 35 * gauss(x, 0.5, 0.09)); break;
    case "dip": S.ic = profile((x) => 30 - 18 * gauss(x, 0.6, 0.1)); break;
    case "ramp": S.ic = profile((x) => 20 + 45 * x); break;
    case "two": S.ic = profile((x) => 20 + 30 * gauss(x, 0.35, 0.06) + 22 * gauss(x, 0.75, 0.08)); break;
    case "smooth": {
      const a = S.ic.slice();
      for (let i = 1; i < a.length; i++) {
        const l = a[i - 1], r = a[Math.min(i + 1, a.length - 1)];
        S.ic[i] = Math.round(((l + 2 * a[i] + r) / 4) * 10) / 10;
      }
      break;
    }
    case "jagged": S.ic = profile((x, i) => 20 + ((Math.round(x * 20) % 2) ? 25 : -5)); break;
  }
  if (run) { renderEditor(); updateReadouts(); scheduleRun(); }
}

/* ---------------------------------------------------------------- controls */
function wireControls() {
  wireEditor();
  $("presets").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) applyPreset(b.dataset.p); });
  $("alpha").addEventListener("input", (e) => { S.alpha = parseFloat(e.target.value); updateReadouts(); scheduleRun(); });
  document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", (e) => {
    S.mode = e.target.value; updateReadouts(); scheduleRun(true);
  }));
  $("reference").addEventListener("change", (e) => { S.reference = e.target.value; scheduleRun(true); });
  $("runBtn").addEventListener("click", () => runSimulate());
  $("tSlider").addEventListener("input", (e) => { stopPlay(); S.t = parseInt(e.target.value, 10); timeChanged(); queueSnapshot(); });
  $("playBtn").addEventListener("click", togglePlay);
  $("hmWindow").addEventListener("change", () => renderSpacetime());
  $("tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    S.tab = b.dataset.tab;
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + S.tab));
    if (S.dirty.has(S.tab)) renderTab(S.tab);
    document.querySelectorAll("#tab-" + S.tab + " .plot").forEach((g) => { try { Plotly.Plots.resize(g); } catch (_) {} });
  });
}

function scheduleRun(force = false) {
  if (!force && !$("autorun").checked) return;
  clearTimeout(S.runTimer);
  S.runTimer = setTimeout(runSimulate, 250);
}

function togglePlay() { S.play ? stopPlay() : startPlay(); }
function startPlay() {
  if (!S.sim) return;
  if (S.t >= S.cfg.max_time) S.t = 0;
  $("playBtn").textContent = "⏸";
  S.play = setInterval(() => {
    S.t = Math.min(S.cfg.max_time, S.t + 30);
    $("tSlider").value = S.t;
    timeChanged();
    if (S.t >= S.cfg.max_time) stopPlay();
  }, 90);
}
function stopPlay() {
  if (!S.play) return;
  clearInterval(S.play); S.play = null;
  $("playBtn").textContent = "▶";
  queueSnapshot();
}

/* ------------------------------------------------------------- API calls */
const body = () => ({ ic: S.ic, alpha: S.alpha, mode: S.mode, t: S.t, reference: S.reference });
async function post(url) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()) });
  if (!r.ok) throw new Error((await r.text()).slice(0, 300));
  return r.json();
}

async function runSimulate() {
  const id = ++S.simId;
  $("busy").hidden = false; $("busy").className = "busy"; $("busy").textContent = "computing FEM + operator…";
  try {
    const d = await post("/api/simulate");
    if (id !== S.simId) return;
    S.sim = d; S.snap = null;
    const ana = document.querySelector("#reference option[value=analytical]");
    ana.disabled = !d.analytical_available;
    S.reference = d.reference; $("reference").value = d.reference;
    $("busy").hidden = true;
    TABS.forEach((t) => S.dirty.add(t));
    renderTab(S.tab);
    queueSnapshot(50);
  } catch (err) {
    if (id !== S.simId) return;
    $("busy").className = "busy error"; $("busy").textContent = "Error: " + err.message;
  }
}

function queueSnapshot(delay = 220) {
  clearTimeout(S.snapTimer);
  S.snapTimer = setTimeout(fetchSnapshot, delay);
}
async function fetchSnapshot() {
  if (!S.sim) return;
  const id = ++S.snapId, simId = S.simId;
  try {
    const d = await post("/api/snapshot");
    if (id !== S.snapId || simId !== S.simId) return;
    S.snap = d.snapshot;
    S.dirty.add("speed"); S.dirty.add("solution");
    if (S.tab === "speed" || S.tab === "solution") renderTab(S.tab);
  } catch (_) { /* a newer request superseded this one */ }
}

/* ---------------------------------------------------------- derived data */
function gridIndex() {
  const n = S.sim.times.length;
  return Math.round(S.t / (S.cfg.max_time / (n - 1)));
}
function localSnap() {
  const d = S.sim, k = gridIndex();
  const fem = d.fem_grid[k], op = d.op_grid[k], std = d.std_grid[k], err = d.err_grid[k];
  const ana = d.analytical_grid ? d.analytical_grid[k] : null;
  const ref = d.reference === "analytical" ? ana : fem;
  let sa = 0, sq = 0, mx = 0, rn = 0, cov = 0;
  for (let i = 0; i < err.length; i++) {
    const a = Math.abs(err[i]); sa += a; sq += err[i] * err[i]; mx = Math.max(mx, a); rn += ref[i] * ref[i];
    if (a <= 2 * std[i]) cov++;
  }
  const n = err.length;
  return { k, fem, op, std, err, ana, ref, mae: sa / n, rmse: Math.sqrt(sq / n), max: mx,
           rel: 100 * Math.sqrt(sq / rn), cov: (100 * cov) / n };
}
function yRange() {
  const d = S.sim;
  let lo = Math.min(...S.ic), hi = Math.max(...S.ic);
  for (const row of d.fem_grid) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = 0.06 * (hi - lo || 1);
  return [lo - pad, hi + pad];
}
function segIndex(x) { return Math.min(S.cfg.n_x_segments - 1, Math.floor(x / (S.cfg.length / S.cfg.n_x_segments) + 1e-9)); }
function segLabels() {
  const n = S.cfg.n_x_segments, w = S.cfg.length / n;
  return Array.from({ length: n }, (_, i) => `${(i * w).toFixed(1)}–${((i + 1) * w).toFixed(1)}`);
}

/* ---------------------------------------------------------------- render */
function renderTab(name) {
  if (!S.sim) return;
  S.dirty.delete(name);
  ({ solution: renderSolution, spacetime: renderSpacetime, intervals: renderIntervals,
     speed: renderSpeed, training: renderTraining })[name]();
}

function timeChanged() {
  updateReadouts();
  if (!S.sim) return;
  ["solution", "spacetime", "intervals", "speed"].forEach((t) => S.dirty.add(t));
  if (S.tab === "solution") renderSolution();
  else if (S.tab === "spacetime") updateHmLines();
  else if (S.tab === "intervals") renderIntervals();
}

/* --- Solution tab ------------------------------------------------------- */
function renderSolution() {
  const d = S.sim, s = localSnap(), x = d.x;
  const refName = d.reference === "analytical" ? "analytical" : "FEM";
  const tm = S.snap && S.snap.t === S.t ? S.snap.timing : null;
  $("stats").innerHTML =
    stat("MAE @ t", fmt(s.mae), "°C", `vs ${refName}`) +
    stat("RMSE @ t", fmt(s.rmse), "°C") +
    stat("Max error @ t", fmt(s.max), "°C") +
    stat("Rel. L2 error", fmt(s.rel, 2), "%") +
    (d.has_uncertainty ? stat("Within ±2σ", fmt(s.cov, 0), "%", "ensemble calibration") : "") +
    stat("Speed-up @ t", tm ? (tm.speedup ? fmtX(tm.speedup) : "n/a") : "…", "",
         tm ? `FEM ${fmtDur(tm.fem_s)} · operator ${fmtDur(tm.op_s)}` : "measuring…", "good");

  const traces = [];
  traces.push({ x: S.cfg.sensor_x, y: S.ic, mode: "lines+markers", name: "initial condition", line: { color: COL.ic, dash: "dash", width: 1.5 },
                marker: { size: 4 }, hovertemplate: "IC %{y:.1f}°C<extra></extra>" });
  if (d.has_uncertainty) {
    traces.push({ x, y: s.op.map((v, i) => v + 2 * s.std[i]), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false });
    traces.push({ x, y: s.op.map((v, i) => v - 2 * s.std[i]), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: COL.band,
                  name: "operator ±2σ", hoverinfo: "skip" });
  }
  traces.push({ x, y: s.fem, mode: "lines", name: "FEM (ground truth)", line: { color: COL.fem, width: 2.5 } });
  if (s.ana) traces.push({ x, y: s.ana, mode: "lines", name: "analytical", line: { color: COL.ana, width: 2, dash: "dot" } });
  traces.push({ x, y: s.op, mode: "lines", name: d.has_uncertainty ? "operator (ensemble mean)" : "operator", line: { color: COL.op, width: 2, dash: "dash" } });

  Plotly.react("plotProfile", traces, {
    margin: { l: 55, r: 15, t: 40, b: 45 }, hovermode: "x unified",
    title: { text: `Temperature along the rod at t = ${S.t} s (${mins(S.t)})`, font: { size: 14 }, x: 0.02 },
    xaxis: { title: "x (position along rod)", range: [0, 1], zeroline: false },
    yaxis: { title: "Temperature (°C)", range: yRange() },
    legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 1, xanchor: "right", font: { size: 11 } },
    paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  // error profile
  let em = 0;
  for (let i = 0; i < s.err.length; i++) em = Math.max(em, Math.abs(s.err[i]), d.has_uncertainty ? Math.min(2 * s.std[i], 3 * s.max + 0.05) : 0);
  em = Math.max(em * 1.15, 0.05);
  const et = [{ x, y: s.err, mode: "lines", name: `operator − ${refName}`, line: { color: COL.err, width: 2 }, fill: "tozeroy", fillcolor: "rgba(220,38,38,.12)" }];
  if (d.has_uncertainty) {
    et.push({ x, y: s.std.map((v) => 2 * v), mode: "lines", name: "±2σ", line: { color: COL.op, width: 1, dash: "dot" } });
    et.push({ x, y: s.std.map((v) => -2 * v), mode: "lines", showlegend: false, line: { color: COL.op, width: 1, dash: "dot" } });
  }
  Plotly.react("plotErrProfile", et, {
    margin: { l: 55, r: 15, t: 34, b: 42 }, hovermode: "x unified",
    title: { text: "Pointwise error (operator − reference)", font: { size: 14 }, x: 0.02 },
    xaxis: { title: "x", range: [0, 1], zeroline: false },
    yaxis: { title: "°C", range: [-em, em], zeroline: true, zerolinecolor: "#9ca3af" },
    legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 1, xanchor: "right", font: { size: 11 } },
    paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  let note = `Reference for the errors: ${refName}.`;
  if (d.analytical_available) note += ` The initial condition is uniform (${fmt(d.uniform_T0, 1)} °C), so the closed-form series solution is also shown; FEM itself differs from it by ${fmt(d.fem_vs_analytical.mae, 3)} °C MAE.`;
  if (S.mode === "nonlinear") note += " No analytical solution exists for α(T).";
  $("refNote").textContent = note;
}

/* --- Space-time tab ----------------------------------------------------- */
function tLine() {
  return [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: S.t, y1: S.t, line: { color: "#22d3ee", width: 2, dash: "dot" } }];
}
function updateHmLines() {
  for (const id of ["hmFem", "hmOp", "hmErr", "hmStd"]) Plotly.relayout(id, { shapes: tLine() });
}
function hm(id, z, title, scale, extra = {}) {
  const d = S.sim;
  Plotly.react(id, [Object.assign({ type: "heatmap", x: d.x, y: extra.y || d.times, z, colorscale: scale,
    colorbar: { thickness: 12, len: 0.9, title: { text: extra.unit || "°C", side: "right" } },
    hovertemplate: "x=%{x:.2f}<br>t=%{y:.0f}s<br>%{z:.3f}<extra></extra>" }, extra.trace || {})], {
    margin: { l: 52, r: 8, t: 36, b: 40 }, title: { text: title, font: { size: 13 }, x: 0.03 },
    xaxis: { title: "x" }, yaxis: { title: "t (s)", range: extra.yrange || [0, d.times[d.times.length - 1]] }, shapes: tLine(), paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);
}
function renderSpacetime() {
  const d = S.sim, win = parseFloat($("hmWindow").value);
  const rows = d.times.map((t, i) => i).filter((i) => d.times[i] <= win + 1e-9);
  const sub = (g) => rows.map((i) => g[i]);
  const fem = sub(d.fem_grid), op = sub(d.op_grid), err = sub(d.err_grid);
  let lo = Infinity, hi = -Infinity;
  for (const g of [d.fem_grid, d.op_grid]) for (const row of g) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const T = d.times.filter((t) => t <= win + 1e-9);
  const H = (id, z, title, scale, trace, unit) => hm(id, z, title, scale, { trace, unit, y: T, yrange: [0, win] });
  H("hmFem", fem, "FEM (ground truth)", INFERNO, { zmin: lo, zmax: hi });
  H("hmOp", op, d.has_uncertainty ? "Operator (ensemble mean)" : "Operator", INFERNO, { zmin: lo, zmax: hi });
  // colour range clipped at the 99th percentile so the t≈0 transient does not wash out the rest
  const q = (arr, p) => arr[Math.min(arr.length - 1, Math.floor(p * arr.length))];
  const inWin = err.flat().map(Math.abs).sort((a, b) => a - b);
  const em = Math.max(q(inWin, 0.99), 1e-6);
  const refName = d.reference === "analytical" ? "analytical" : "FEM";
  H("hmErr", err, `Error: operator − ${refName} (colour clipped at 99th pct.)`, "RdBu", { zmin: -em, zmax: em, reversescale: true });
  if (d.has_uncertainty) H("hmStd", sub(d.std_grid), "Ensemble std (uncertainty)", "Viridis", {});
  else H("hmStd", err.map((r) => r.map(Math.abs)), "Absolute error", ERRSCALE, { zmin: 0, zmax: em });
}

/* --- Error-interval tab ------------------------------------------------- */
function renderIntervals() {
  const d = S.sim, iv = d.intervals, m = d.metrics, s = localSnap();
  const refName = d.reference === "analytical" ? "analytical" : "FEM";
  $("fullStats").innerHTML =
    stat("MAE (all x, t)", fmt(m.mae), "°C", `vs ${refName}`) +
    stat("RMSE", fmt(m.rmse), "°C") +
    stat("Max error", fmt(d.max_error_at.value), "°C", `at x=${fmt(d.max_error_at.x, 2)}, t=${fmt(d.max_error_at.t, 0)} s`) +
    stat("Rel. L2 error", fmt(m.rel_l2_pct, 2), "%") +
    stat("Bias", fmt(m.bias), "°C", "mean signed error") +
    (d.has_uncertainty ? stat("Within ±2σ", fmt(m.coverage_2std_pct, 0), "%", `mean σ = ${fmt(m.mean_std, 3)} °C`) : "");

  const xl = segLabels(), tl = iv.time_bins.map(binLabel);
  const text = iv.matrix.map((r) => r.map((v) => (v === null ? "" : v.toFixed(3))));
  Plotly.react("plotMatrix", [{ type: "heatmap", z: iv.matrix, x: tl, y: xl, colorscale: ERRSCALE, text, texttemplate: "%{text}",
    colorbar: { thickness: 12, title: { text: "MAE °C", side: "right" } }, hovertemplate: "x∈%{y}<br>%{x}<br>MAE %{z:.4f} °C<extra></extra>" }], {
    margin: { l: 70, r: 8, t: 36, b: 44 }, title: { text: "MAE by x-segment × time window", font: { size: 13 }, x: 0.03 },
    xaxis: { title: "time window" }, yaxis: { title: "x segment", autorange: "reversed" }, paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  // snapshot MAE per x segment at the selected time
  const nseg = S.cfg.n_x_segments, acc = Array(nseg).fill(0), cnt = Array(nseg).fill(0);
  d.x.forEach((xv, i) => { const k = segIndex(xv); acc[k] += Math.abs(s.err[i]); cnt[k]++; });
  const snapSeg = acc.map((a, i) => a / cnt[i]);
  Plotly.react("plotByX", [
    { type: "bar", x: xl, y: iv.by_x_segment, name: "all times", marker: { color: "#93c5fd" } },
    { type: "bar", x: xl, y: snapSeg, name: `t = ${S.t} s`, marker: { color: COL.op } },
  ], {
    margin: { l: 55, r: 8, t: 36, b: 44 }, barmode: "group", title: { text: "MAE by x-segment", font: { size: 13 }, x: 0.03 },
    xaxis: { title: "x segment" }, yaxis: { title: "MAE (°C)" }, legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 1, xanchor: "right" },
    paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  Plotly.react("plotByTime", [{ type: "bar", x: tl, y: iv.by_time_bin, marker: { color: "#7c3aed" },
    hovertemplate: "%{x}<br>MAE %{y:.4f} °C<extra></extra>" }], {
    margin: { l: 55, r: 8, t: 36, b: 44 }, title: { text: "MAE by time window (all x)", font: { size: 13 }, x: 0.03 },
    xaxis: { title: "time window" }, yaxis: { title: "MAE (°C)" }, paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  Plotly.react("plotCurves", [
    { x: d.times, y: iv.mae_curve_vs_time, mode: "lines", name: "MAE vs t (avg over x)", line: { color: COL.err, width: 2 }, xaxis: "x", yaxis: "y" },
    { x: d.x, y: iv.mae_curve_vs_x, mode: "lines", name: "MAE vs x (avg over t)", line: { color: COL.op, width: 2 }, xaxis: "x2", yaxis: "y2" },
  ], {
    margin: { l: 55, r: 8, t: 36, b: 44 }, title: { text: "Error curves", font: { size: 13 }, x: 0.03 },
    grid: { rows: 1, columns: 2, pattern: "independent" },
    xaxis: { title: "t (s)", domain: [0, 0.46] }, yaxis: { title: "MAE (°C)" },
    xaxis2: { title: "x", domain: [0.56, 1] }, yaxis2: { anchor: "x2" },
    shapes: [{ type: "line", xref: "x", yref: "paper", x0: S.t, x1: S.t, y0: 0, y1: 1, line: { color: "#22d3ee", width: 2, dash: "dot" } }],
    legend: { orientation: "h", y: -0.28 }, paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);
}

/* --- Speed tab ---------------------------------------------------------- */
function renderSpeed() {
  const d = S.sim, tm = d.timing, sn = S.snap && S.snap.t === S.t ? S.snap.timing : null;
  $("speedStats").innerHTML =
    stat("FEM @ t", sn ? fmtDur(sn.fem_s) : "…", "", sn ? `${sn.fem_steps.toLocaleString()} time steps` : "") +
    stat("Operator @ t", sn ? fmtDur(sn.op_s) : "…", "", "one forward pass, whole rod") +
    stat("Speed-up @ t", sn ? (sn.speedup ? fmtX(sn.speedup) : "n/a") : "…", "", "", "good") +
    stat("FEM, full field", fmtDur(tm.fem_full_s), "", `0 → ${S.cfg.max_time} s`) +
    stat("Operator, full field", fmtDur(tm.op_full_s), "", `${tm.grid_points.toLocaleString()} (x,t) points`) +
    stat("Speed-up, full field", fmtX(tm.speedup_full), "", "", "good");

  const names = [], vals = [], cols = [];
  if (sn) { names.push(`FEM @ t=${S.t}s`, `Operator @ t=${S.t}s`); vals.push(sn.fem_s, sn.op_s); cols.push(COL.fem, COL.op); }
  names.push("FEM full field", "Operator full field"); vals.push(tm.fem_full_s, tm.op_full_s); cols.push(COL.fem, COL.op);
  Plotly.react("plotSpeed", [{ type: "bar", x: names, y: vals, marker: { color: cols }, text: vals.map(fmtDur), textposition: "outside",
    hovertemplate: "%{x}<br>%{text}<extra></extra>" }], {
    margin: { l: 60, r: 10, t: 36, b: 50 }, title: { text: "Wall-clock time (log scale)", font: { size: 13 }, x: 0.02 },
    yaxis: { type: "log", title: "seconds", exponentformat: "power",
             range: [Math.log10(Math.min(...vals)) - 0.4, Math.log10(Math.max(...vals)) + 0.5] }, paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  const tr = S.cfg.training, tt = tr ? tr.timings_s.total : null;
  let be = "";
  if (tt && tm.fem_full_s > tm.op_full_s) {
    const n = Math.ceil(tt / (tm.fem_full_s - tm.op_full_s));
    be = `<p><b>Break-even.</b> Building the operator took ${tt.toFixed(0)} s (including the FEM runs that generated its training data). At this problem size and this setting, that is the cost of about <b>${n.toLocaleString()}</b> full-field FEM solves, so the operator only pays off if you need more queries than that.</p>`;
  }
  $("speedNotes").innerHTML =
    `<p><b>How the timings work.</b> The FEM marches through every time step from t = 0 up to the requested time (Δt = 0.5 s, dense LU solve per step${S.mode === "nonlinear" ? ", plus a Picard iteration loop per step because α depends on T" : ""}), so its cost grows with t. The operator evaluates any (x, t) directly in a single forward pass (${d.has_uncertainty ? S.cfg.training.config.ens_n + " networks" : "one network"}; the branch net runs once per scenario, the trunk net once per query point), so its cost does not depend on t.</p>` +
    `<p>Operator times are the median of repeated calls after a warm-up; FEM times are a single run. Both run on the server CPU. The FEM here is the notebook's dense-LU implementation of a small 1-D problem; a banded or tridiagonal solver would be considerably faster, so treat the speed-up as relative to this particular baseline.</p>` + be;
}

/* --- Training tab ------------------------------------------------------- */
function renderTraining() {
  const tr = S.cfg && S.cfg.training;
  if (!tr) return;
  S.dirty.delete("training");
  const cfg = tr.config, tm = tr.timings_s;
  $("trainStats").innerHTML =
    stat("Test MAE (final)", fmt(tr.final_test_mean_err), "°C", "held-out FEM scenarios") +
    stat("Test max error", fmt(tr.final_test_max_err), "°C") +
    stat("α(T) operator MAE", fmt(tr.nonlinear_test_mae), "°C", "vs nonlinear FEM") +
    stat("Training time", tm.total.toFixed(0), "s", `preset “${S.cfg.preset}”`) +
    stat("Training examples", tr.initial_train_examples.toLocaleString() + " → " + tr.final_train_examples.toLocaleString(), "", "before → after active learning");

  const al = tr.al_history, bl = tr.baseline_history;
  const traces = [{ x: al.map((h) => h.round), y: al.map((h) => h.err), mode: "lines+markers", name: "uncertainty-guided",
    line: { color: COL.op, width: 2 }, customdata: al.map((h) => h.size), hovertemplate: "round %{x}<br>MAE %{y:.3f} °C<br>%{customdata} examples<extra>uncertainty</extra>" }];
  if (bl) traces.push({ x: bl.map((h) => h.round), y: bl.map((h) => h.err), mode: "lines+markers", name: "random selection",
    line: { color: "#9ca3af", width: 2, dash: "dash" }, marker: { symbol: "square" }, customdata: bl.map((h) => h.size),
    hovertemplate: "round %{x}<br>MAE %{y:.3f} °C<br>%{customdata} examples<extra>random</extra>" });
  Plotly.react("plotAL", traces, {
    margin: { l: 55, r: 10, t: 36, b: 44 }, title: { text: "Active learning: held-out test error per round", font: { size: 13 }, x: 0.03 },
    xaxis: { title: "active-learning round", dtick: 1 }, yaxis: { title: "mean abs. error (°C)", rangemode: "tozero" },
    legend: { orientation: "h", y: -0.25 }, paper_bgcolor: "rgba(0,0,0,0)",
  }, PCFG);

  const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  $("trainTable").innerHTML =
    `<h3>Configuration</h3><table>` +
    row("Ensemble size", cfg.ens_n) + row("Initial / candidate / test scenarios", `${cfg.inits} / ${cfg.choices} / ${cfg.tests}`) +
    row("Initial epochs", cfg.epochs) + row("Rounds × scenarios per round", `${cfg.rounds} × ${cfg.k}`) +
    row("Fine-tune epochs per round", cfg.epochs_round) + row("Loss weights (data / physics)", `${cfg.w_data} / ${cfg.w_phy}`) +
    row("α(T) scenarios / epochs / β", `${cfg.nl_scenarios} / ${cfg.nl_epochs} / ${cfg.beta}`) + `</table>` +
    `<h3>Timing</h3><table>` + row("FEM data generation", tm.fem_data.toFixed(1) + " s") +
    row("Active-learning ensemble", tm.active_learning.toFixed(1) + " s") + row("α(T) operator (data + training)", tm.nonlinear.toFixed(1) + " s") +
    row("Total (incl. random baseline if run)", tm.total.toFixed(1) + " s") + `</table>` +
    `<h3>Environment</h3><table>` + row("Trained at", tr.trained_at) + row("PyTorch", tr.torch_version) + `</table>`;
}

boot();
