/* ============================ CityPulse frontend ============================ */
"use strict";

const API = ""; // same origin (served by FastAPI)
const BK = [40.655, -73.95];           // Brooklyn centre
const GROUP_COLORS = {
  noise: "#f59e0b", road: "#38bdf8", flood: "#818cf8",
  sewer_backup: "#c084fc", normal: "#64748b", mixed: "#5eead4",
};

const state = {
  zones: [], incidents: [], reports: [],
  livability: {},        // zone_id -> 0..1 (1 = best)
  predict: null,         // last /predict response
  reportLoc: null,       // {lat,lon}
  mapMode: "report",     // "report" | "route"
  route: { a: null, b: null },
  colorMode: "livability",
};

let map, layerIncidents, layerZones, layerReports, layerRoute, livChart;
let routeMarkers = [];

/* ----------------------------- helpers ----------------------------- */
const $ = (id) => document.getElementById(id);
async function getJSON(url) { const r = await fetch(API + url); if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }
function fmt(n) { return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); }
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove("show"), 2600); }

// green(1) -> amber(.5) -> red(0)
function scoreColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[248, 113, 113], [251, 191, 36], [52, 211, 153]]; // bad, mid, good
  const seg = t < 0.5 ? [stops[0], stops[1], t / 0.5] : [stops[1], stops[2], (t - 0.5) / 0.5];
  const [a, b, f] = seg;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function norm(vals) {
  const mn = Math.min(...vals), mx = Math.max(...vals);
  return (v) => (mx === mn ? 0.5 : (v - mn) / (mx - mn));
}

/* ----------------------------- map init ----------------------------- */
function initMap() {
  map = L.map("map", { zoomControl: true, attributionControl: false }).setView(BK, 12);
  // Esri "Dark Gray Canvas" — keyless, reliable dark basemap (note {z}/{y}/{x} order)
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 16 }
  ).addTo(map);
  layerZones = L.layerGroup().addTo(map);
  layerIncidents = L.layerGroup().addTo(map);
  layerReports = L.layerGroup().addTo(map);
  layerRoute = L.layerGroup().addTo(map);
  map.on("click", onMapClick);
}

function onMapClick(e) {
  const { lat, lng } = e.latlng;
  if (state.mapMode === "route") {
    if (!state.route.a) { state.route.a = [lat, lng]; $("ptA").textContent = `A: ${lat.toFixed(4)}, ${lng.toFixed(4)}`; }
    else { state.route.b = [lat, lng]; $("ptB").textContent = `B: ${lat.toFixed(4)}, ${lng.toFixed(4)}`; }
    $("routeBtn").disabled = !(state.route.a && state.route.b);
    drawRoutePins();
  } else {
    state.reportLoc = { lat, lon: lng };
    $("locChip").textContent = `${lat.toFixed(3)}, ${lng.toFixed(3)}`;
    $("submitBtn").disabled = false;
    drawReportCursor(lat, lng);
  }
}

let cursorMarker;
function drawReportCursor(lat, lng) {
  if (cursorMarker) map.removeLayer(cursorMarker);
  cursorMarker = L.marker([lat, lng], {
    icon: L.divIcon({ className: "", html: `<div style="width:18px;height:18px;border-radius:50%;
      background:#22d3ee55;border:2px solid #22d3ee;box-shadow:0 0 14px #22d3ee"></div>`, iconSize: [18, 18] })
  }).addTo(map);
}

/* ----------------------------- zones + livability ----------------------------- */
function computeLivability() {
  const z = state.zones;
  const groups = ["noise", "road", "flood", "sewer_backup"];
  const normFns = {};
  groups.forEach((g) => { normFns[g] = norm(z.map((x) => x.counts_by_group?.[g] || 0)); });
  state.livability = {};
  z.forEach((x) => {
    const problem = groups.reduce((s, g) => s + normFns[g](x.counts_by_group?.[g] || 0), 0) / groups.length;
    state.livability[x.zone_id] = 1 - problem; // 1 = most livable
  });
}

function zoneColorValue(z) {
  if (state.colorMode === "livability") return state.livability[z.zone_id] ?? 0.5;
  if (state.colorMode === "density") {
    const counts = state.zones.map((x) => x.count);
    return norm(counts)(z.count); // high data = 1 (shown), low = red-ish
  }
  // flood / noise risk from /predict
  if (state.predict) {
    const key = state.colorMode;
    const arr = state.predict[key]?.zones || [];
    const risks = arr.map((a) => a.risk);
    const nf = norm(risks);
    const hit = arr.find((a) => a.zone_id === z.zone_id);
    return hit ? 1 - nf(hit.risk) : 0.5; // high risk -> low score -> red
  }
  return 0.5;
}

function renderZones() {
  layerZones.clearLayers();
  if (!$("tglZones").checked) return;
  state.zones.forEach((z) => {
    const score = zoneColorValue(z);
    const col = scoreColor(score);
    const lowData = state.colorMode === "density" && score < 0.2;
    L.circle([z.centre_lat, z.centre_lon], {
      radius: z.radius_m_p90 || 1500,
      color: col, weight: 2, fillColor: col,
      fillOpacity: lowData ? 0.08 : 0.22, dashArray: lowData ? "4 6" : null,
    }).bindPopup(zonePopup(z, score)).addTo(layerZones);
  });
}
function zonePopup(z, score) {
  const cb = z.counts_by_group || {};
  const rows = Object.keys(cb).map((g) =>
    `<span style="color:${GROUP_COLORS[g] || "#fff"}">●</span> ${g}: <b>${fmt(cb[g])}</b>`).join("<br>");
  const label = state.colorMode === "livability" ? "Livability"
    : state.colorMode === "density" ? "Data density" : state.colorMode + " (inv.)";
  return `<b>Zone ${z.zone_id}</b> — ${z.dominant_group}<br>
    ${label}: <b>${(score * 100).toFixed(0)}%</b><br>Total: <b>${fmt(z.count)}</b><br>${rows}`;
}

/* ----------------------------- incidents ----------------------------- */
function renderIncidents() {
  layerIncidents.clearLayers();
  if (!$("tglIncidents").checked) return;
  state.incidents.forEach((i) => {
    const r = Math.min(9, 3 + Math.sqrt(i.count));
    L.circleMarker([i.lat, i.lon], {
      radius: r, color: GROUP_COLORS[i.group] || "#fff", weight: 1,
      fillColor: GROUP_COLORS[i.group] || "#fff", fillOpacity: 0.75,
    }).bindPopup(`<b>${i.incident_id}</b><br>${i.group} · <b>${i.count}</b> reports<br>zone ${i.zone_id}`)
      .addTo(layerIncidents);
  });
}

/* ----------------------------- reports (my pins) ----------------------------- */
function renderReports() {
  layerReports.clearLayers();
  if (!$("tglReports").checked) return;
  state.reports.forEach((rp) => addReportPin(rp, false));
}
function addReportPin(rp, open) {
  const col = GROUP_COLORS[rp.group] || GROUP_COLORS[rp.type] || "#22d3ee";
  const m = L.marker([rp.lat, rp.lon], {
    icon: L.divIcon({ className: "", iconSize: [26, 26], html:
      `<div style="width:24px;height:24px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);
        background:${col};border:2px solid #fff;box-shadow:0 0 12px ${col}"></div>` })
  }).bindPopup(reportPopup(rp)).addTo(layerReports);
  if (open) m.openPopup();
}
function reportPopup(rp) {
  const extra = rp.intensity != null ? `<br>Intensity: <b>${(rp.intensity * 100).toFixed(0)}%</b>` : "";
  const tr = rp.trust != null ? `<br>Trust: <b>${(rp.trust * 100).toFixed(0)}%</b>` : "";
  return `<b style="text-transform:capitalize">${rp.type}</b> (${rp.modality})<br>
    Confidence: <b>${(rp.confidence * 100).toFixed(0)}%</b>${extra}${tr}`;
}

/* ----------------------------- stats + chart ----------------------------- */
function renderStats() {
  const totals = {};
  state.zones.forEach((z) => Object.entries(z.counts_by_group || {}).forEach(([g, c]) => totals[g] = (totals[g] || 0) + c));
  const topGroup = Object.entries(totals).sort((a, b) => b[1] - a[1])[0];
  const totalSignals = state.zones.reduce((s, z) => s + z.count, 0);
  const avgLiv = Object.values(state.livability).reduce((s, v) => s + v, 0) / (state.zones.length || 1);
  $("statIncidents").textContent = fmt(totalSignals);
  $("statIncidents").nextElementSibling.textContent = "Signals";
  $("statZones").textContent = state.zones.length;
  $("statLiv").textContent = (avgLiv * 100).toFixed(0) + "%";
  $("statHot").textContent = topGroup ? topGroup[0] : "—";
}

function renderChart() {
  const labels = state.zones.map((z) => "Z" + z.zone_id);
  const data = state.zones.map((z) => +((state.livability[z.zone_id] ?? 0.5) * 100).toFixed(0));
  const colors = data.map((d) => scoreColor(d / 100));
  if (livChart) { livChart.data.labels = labels; livChart.data.datasets[0].data = data; livChart.data.datasets[0].backgroundColor = colors; livChart.update(); return; }
  livChart = new Chart($("livChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "Livability %", data, backgroundColor: colors, borderRadius: 6 }] },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { min: 0, max: 100, grid: { color: "#243150" }, ticks: { color: "#8ea0c4" } },
        y: { grid: { display: false }, ticks: { color: "#8ea0c4" } },
      },
    },
  });
}

/* ----------------------------- admin table ----------------------------- */
async function loadAdmin() {
  const rows = await getJSON("/admin?limit=40");
  $("adminCount").textContent = rows.length + " shown";
  $("adminBody").innerHTML = rows.map((r) => {
    const col = GROUP_COLORS[r.group] || "#fff";
    return `<tr><td>${r.incident_id}</td>
      <td><span class="gtag" style="background:${col}22;color:${col}">${r.group}</span></td>
      <td>${r.count}</td><td class="prio">${r.priority}</td></tr>`;
  }).join("");
}

/* ----------------------------- legend ----------------------------- */
function renderLegend() {
  const el = $("legend");
  if (["livability", "flood", "noise", "density"].includes(state.colorMode)) {
    const title = state.colorMode === "livability" ? "Livability"
      : state.colorMode === "density" ? "Report density" : state.colorMode + " risk";
    const ends = state.colorMode === "livability" || state.colorMode === "density"
      ? ["worse / low", "better / high"] : ["high risk", "low risk"];
    el.innerHTML = `<h4>${title}</h4><div class="lg-grad"></div>
      <div class="lg-ends"><span>${ends[0]}</span><span>${ends[1]}</span></div>
      <h4 style="margin-top:10px">Incidents</h4>
      ${Object.entries(GROUP_COLORS).filter(([g]) => ["noise", "road", "flood", "sewer_backup"].includes(g))
        .map(([g, c]) => `<div class="lg-row"><span class="lg-swatch" style="background:${c}"></span>${g}</div>`).join("")}`;
  }
}

/* ----------------------------- forecast / predict ----------------------------- */
async function runPredict() {
  const rain = +$("rain").value, hour = +$("hour").value, weekday = +$("weekday").value;
  try {
    state.predict = await (await fetch(API + "/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rain_mm: rain, hour, weekday }),
    })).json();
  } catch (e) { state.predict = null; }
  if (state.colorMode === "flood" || state.colorMode === "noise") { renderZones(); }
}

/* ----------------------------- report submit ----------------------------- */
function currentModality() { return document.querySelector(".seg-btn.active").dataset.mod; }

async function submitReport() {
  if (!state.reportLoc) { toast("Click the map to set a location first"); return; }
  const opt = $("demoSelect").selectedOptions[0];
  if (!opt || !opt.value) { toast("Pick a demo sample"); return; }
  const meta = JSON.parse(opt.value);
  $("submitBtn").disabled = true; $("submitBtn").textContent = "Analyzing…";
  try {
    const blob = await (await fetch(API + meta.url)).blob();
    const fd = new FormData();
    fd.append("file", blob, meta.name);
    fd.append("lat", state.reportLoc.lat); fd.append("lon", state.reportLoc.lon);
    fd.append("modality", meta.modality);
    const res = await (await fetch(API + "/report", { method: "POST", body: fd })).json();
    showResult(res); state.reports.unshift(res); addReportPin(res, true);
    toast(`Detected: ${res.type} (${(res.confidence * 100).toFixed(0)}%)`);
  } catch (e) { toast("Report failed: " + e.message); }
  $("submitBtn").disabled = false; $("submitBtn").textContent = "Analyze & drop pin";
}

function meter(label, val, color) {
  return `<div class="meter"><div class="meter-lbl"><span>${label}</span><span>${(val * 100).toFixed(0)}%</span></div>
    <div class="meter-bar"><div class="meter-fill" style="width:${val * 100}%;background:${color}"></div></div></div>`;
}
function showResult(r) {
  const col = GROUP_COLORS[r.group] || "#22d3ee";
  let html = `<h3><span style="color:${col}">●</span><span class="rtype">${r.type}</span></h3>
    <p class="hint" style="margin:4px 0 8px">Group: ${r.group} · ${r.modality}</p>`;
  html += meter("Confidence", r.confidence, "#2dd4bf");
  if (r.intensity != null) html += meter("Relative intensity", r.intensity, "#f59e0b");
  if (r.trust != null) html += meter("Trust", r.trust, scoreColor(r.trust));
  const rc = $("resultCard"); rc.innerHTML = html; rc.classList.remove("hidden");
}

/* ----------------------------- live mic ----------------------------- */
async function recordMic() {
  if (currentModality() !== "audio") { toast("Switch to Audio to record"); return; }
  if (!state.reportLoc) { toast("Click the map to set a location first"); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { toast("Mic blocked (needs localhost/HTTPS)"); return; }
  const rec = new MediaRecorder(stream), chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: "audio/webm" });
    const fd = new FormData();
    fd.append("file", blob, "live.webm");
    fd.append("lat", state.reportLoc.lat); fd.append("lon", state.reportLoc.lon);
    fd.append("modality", "audio");
    try {
      const res = await (await fetch(API + "/report", { method: "POST", body: fd })).json();
      showResult(res); state.reports.unshift(res); addReportPin(res, true);
      toast(`Live: ${res.type}`);
    } catch (e) { toast("Upload failed"); }
    $("recBtn").classList.remove("active"); $("recBtn").textContent = "● Rec";
  };
  $("recBtn").classList.add("active"); $("recBtn").textContent = "● 5s…";
  rec.start(); setTimeout(() => rec.stop(), 5000);
}

/* ----------------------------- route ----------------------------- */
function drawRoutePins() {
  routeMarkers.forEach((m) => map.removeLayer(m)); routeMarkers = [];
  [["a", "#22d3ee", "A"], ["b", "#f59e0b", "B"]].forEach(([k, c, lbl]) => {
    if (!state.route[k]) return;
    routeMarkers.push(L.marker(state.route[k], {
      icon: L.divIcon({ className: "", iconSize: [22, 22], html:
        `<div style="width:20px;height:20px;border-radius:50%;background:${c};color:#042027;
          font:700 12px Inter;display:grid;place-items:center;border:2px solid #fff">${lbl}</div>` })
    }).addTo(map));
  });
}
async function planRoute() {
  if (!state.route.a || !state.route.b) return;
  layerRoute.clearLayers();
  $("routeBtn").disabled = true; $("routeBtn").textContent = "Routing…";
  try {
    const r = await fetch(API + "/route", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_lat: state.route.a[0], start_lon: state.route.a[1],
        end_lat: state.route.b[0], end_lon: state.route.b[1],
      }),
    });
    const res = await r.json();
    if (!r.ok || !res.normal_route) { toast(res.detail || "Route failed"); }
    else {
      L.polyline(res.normal_route, { color: "#64748b", weight: 5, dashArray: "8 8", opacity: .7 }).addTo(layerRoute);
      L.polyline(res.safe_route, { color: "#2dd4bf", weight: 5, opacity: .95 }).addTo(layerRoute);
      const km = (m) => (m / 1000).toFixed(2) + " km";
      toast(`Safe ${km(res.safe_len_m)} vs normal ${km(res.normal_len_m)} · +${res.detour_m} m to avoid ${fmt(res.hazards)} hazards`);
    }
  } catch (e) { toast("Route failed: " + e.message); }
  $("routeBtn").disabled = false; $("routeBtn").textContent = "Plan route";
}

/* ----------------------------- demo list + UI wiring ----------------------------- */
async function loadDemo() {
  const data = await getJSON("/demo_list");
  state._demo = data;
  fillDemoSelect(currentModality());
}
function fillDemoSelect(mod) {
  const items = (state._demo && state._demo[mod]) || [];
  $("demoSelect").innerHTML = items.map((it) =>
    `<option value='${JSON.stringify(it)}'>${it.cls} — ${it.name}</option>`).join("")
    || "<option value=''>no samples</option>";
}

function wireUI() {
  $("modSeg").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); fillDemoSelect(b.dataset.mod);
  });
  $("submitBtn").addEventListener("click", submitReport);
  $("recBtn").addEventListener("click", recordMic);

  $("rain").addEventListener("input", (e) => { $("rainVal").textContent = e.target.value + " mm"; });
  $("rain").addEventListener("change", runPredict);
  $("hour").addEventListener("input", (e) => { $("hourVal").textContent = String(e.target.value).padStart(2, "0") + ":00"; });
  $("hour").addEventListener("change", runPredict);
  $("weekday").addEventListener("change", runPredict);

  $("colorMode").addEventListener("change", async (e) => {
    state.colorMode = e.target.value;
    if ((state.colorMode === "flood" || state.colorMode === "noise") && !state.predict) await runPredict();
    renderZones(); renderLegend();
  });

  $("routeMode").addEventListener("click", (e) => {
    state.mapMode = state.mapMode === "route" ? "report" : "route";
    e.target.classList.toggle("active", state.mapMode === "route");
    toast(state.mapMode === "route" ? "Route mode: click start then end" : "Report mode: click to place reports");
  });
  $("routeClear").addEventListener("click", () => {
    state.route = { a: null, b: null }; $("ptA").textContent = "A: —"; $("ptB").textContent = "B: —";
    $("routeBtn").disabled = true; layerRoute.clearLayers(); drawRoutePins();
  });
  $("routeBtn").addEventListener("click", planRoute);

  ["tglIncidents", "tglZones", "tglReports"].forEach((id) =>
    $(id).addEventListener("change", () => { renderIncidents(); renderZones(); renderReports(); }));
}

/* ----------------------------- boot ----------------------------- */
async function boot() {
  initMap(); wireUI();
  try {
    const h = await getJSON("/health");
    $("conn").classList.add("ok"); $("connText").textContent = `online · ${h.zones} zones`;
  } catch (e) { $("conn").classList.add("bad"); $("connText").textContent = "backend offline"; }

  try {
    state.zones = await getJSON("/zones");
    computeLivability(); renderZones(); renderStats(); renderChart(); renderLegend();
  } catch (e) { toast("Could not load zones"); }

  try { state.incidents = await getJSON("/incidents?limit=600"); renderIncidents(); } catch (e) {}
  try { state.reports = await getJSON("/reports"); renderReports(); } catch (e) {}
  try { await loadAdmin(); } catch (e) {}
  try { await loadDemo(); } catch (e) {}
  await runPredict();
}
document.addEventListener("DOMContentLoaded", boot);
