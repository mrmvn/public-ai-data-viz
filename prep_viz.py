#!/usr/bin/env python3
"""Render graph.json into a self-contained index.html (vis-network from CDN).

Bipartite layout: datasets pinned to the left column, models to the right column, edges =
`trained on`. Deterministic vertical spacing (no node overlap), hover-to-highlight a node's
neighborhood, a stats line, and a filter toggle (with sources / GPAI notice). Orphan
datasets (no model) are dropped at render time; graph.json stays the complete data artifact.

Data is embedded inline so the HTML is a single self-contained file (works on GitHub Pages
and from file://). Re-run after pull_kg.py.

Usage:
    python render_viz.py
"""

import os
import json
import string

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = HERE

TEMPLATE = string.Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI training data — models &amp; datasets</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  #net { width: 100%; height: 100vh; background: #fafafa; }
  #panel {
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: #fff; border: 1px solid #ddd;
    border-radius: 8px; padding: 10px 14px; font-size: 13px; line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,.08); max-width: 260px;
  }
  #panel h1 { font-size: 14px; margin: 0 0 6px; }
  #stats { color: #444; margin: 6px 0; font-variant-numeric: tabular-nums; }
  #filter { margin: 4px 0 2px; padding: 0; border: 0; }
  #filter label { display: block; cursor: pointer; }
  #filter input { margin-right: 6px; vertical-align: middle; cursor: pointer; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 6px; vertical-align: middle; }
  .ln  { display: inline-block; width: 18px; height: 0; border-top: 3px solid;
         margin-right: 6px; vertical-align: middle; }
  .muted { color: #888; }
</style>
</head>
<body>
<div id="panel">
  <h1>Models &amp; their training datasets</h1>
  <div id="stats"></div>
  <fieldset id="filter">
    <label><input type="radio" name="filter" value="gpai" checked>with a GPAI notice</label>
    <label><input type="radio" name="filter" value="sources">with public data sources</label>
  </fieldset>
  <label style="display:block;margin-top:6px;cursor:pointer">
    <input type="checkbox" id="timeaxis" style="margin-right:6px;vertical-align:middle;cursor:pointer">time axis (by publication date)</label>
  <div style="margin-top:8px">
    <div><span class="dot" style="background:#e8923a"></span>dataset (left)</div>
    <div><span class="dot" style="background:#3a72e8"></span>model (right)</div>
    <div><span style="display:inline-block;width:9px;height:9px;background:#3a72e8;border:2px solid #e02424;margin-right:5px;vertical-align:middle"></span>no disclosed public data sources</div>
    <div><span class="ln" style="border-color:#4caf50"></span>pre-training</div>
    <div><span class="ln" style="border-color:#9c27b0"></span>post-training</div>
    <div><span class="ln" style="border-color:#bbbbbb"></span>stage unspecified</div>
    <div><span class="ln" style="border-color:#777; border-top-style:dashed"></span>based on (lineage)</div>
  </div>
  <div id="axislegend" style="display:none; margin-top:6px; border-top:1px solid #eee; padding-top:6px">
    <div><b>right axis</b>: model publication date (older top &rarr; newer bottom)</div>
    <div><span class="ln" style="border-color:#e02424; border-top-style:dashed"></span>EU AI Act / GPAI milestone dates</div>
  </div>
  <div class="muted" style="margin-top:6px">hover to focus &middot; drag &middot; scroll to
    zoom &middot; click a node to open it</div>
</div>
<div id="net"></div>
<script>
const GRAPH = $graph;
const MODELS_WITH_SOURCES = new Set(GRAPH.edges.map(e => e.model));   // disclosed >=1 dataset

const STEP = 38;                       // vertical px between nodes in a column (no overlap)
const COL_X = 620;                     // half the column separation
const INDENT = 60;                     // derived datasets nudged toward centre, under source
const COL_DATASET = "#e8923a", COL_MODEL = "#3a72e8";
const EDGE_COLOR = { "pre-training": "#4caf50", "post-training": "#9c27b0" };
const DEFAULT_EDGE = "#cccccc", BASED_EDGE = "#777777";
const FADE_NODE = "#e6e6e6", FADE_EDGE = "#ececec";
const AXIS_GAP = 200;                  // px from model column to the time axis
// EU AI Act / GPAI milestones drawn as horizontal reference lines on the time axis
const GPAI_DATES = [
  { date: "2024-08-01", label: "EU AI Act enters into force" },
  { date: "2025-08-02", label: "GPAI rules apply" }
];

const nodesDS = new vis.DataSet();
const edgesDS = new vis.DataSet();
let nodeBase = {};                     // id -> base color (for un-highlighting)
let edgeBase = {};                     // edge id -> {color, width}
let neighbors = {};                    // node id -> Set of adjacent node + edge ids
let AXIS = null;                       // time-axis scale params, set per build()
let TIME_AXIS = false;                  // toggled by the checkbox; default off

function ypos(i, n) { return (i - (n - 1) / 2) * STEP; }

function build(mode) {
  const visibleModels = GRAPH.models.filter(m =>
    mode === "gpai" ? !!m.gpai : MODELS_WITH_SOURCES.has(m.id));
  const modelIds = new Set(visibleModels.map(m => m.id));

  // --- merge edition families: map every dataset to its family root (follow edition_of) ---
  const dsById = {}; GRAPH.datasets.forEach(d => dsById[d.id] = d);
  function root(id) {
    let d = dsById[id], guard = 0;
    while (d && d.edition_of && dsById[d.edition_of] && guard++ < 20) d = dsById[d.edition_of];
    return d ? d.id : id;
  }
  const members = {};                    // root id -> [member labels] (for tooltip)
  GRAPH.datasets.forEach(d => { const r = root(d.id); (members[r] = members[r] || []).push(d.label); });

  // trained-on edges: remap each dataset to its family root, dedup per model (mixed stage -> null)
  const seen = {}, edges = [];
  GRAPH.edges.filter(e => modelIds.has(e.model)).forEach(e => {
    const ds = root(e.dataset), k = e.model + "|" + ds;
    if (seen[k] === undefined) { seen[k] = edges.length; edges.push({ model: e.model, dataset: ds, stage: e.stage }); }
    else if (edges[seen[k]].stage !== e.stage) edges[seen[k]].stage = null;
  });

  const deg = {};
  edges.forEach(e => { deg[e.model] = (deg[e.model]||0)+1; deg[e.dataset] = (deg[e.dataset]||0)+1; });
  // degree drives label font size (more connections = visually heavier), capped to keep
  // box height predictable so the fixed STEP never overlaps.
  const fsize = id => Math.min(20, 12 + 2 * Math.sqrt(deg[id] || 0));

  const dsUsed = new Set(edges.map(e => e.dataset));
  const datasets = GRAPH.datasets.filter(d => dsUsed.has(d.id));   // only family roots survive
  const models = visibleModels.slice();

  // group datasets by `based on` lineage: build a forest among *visible* roots (derived -> source),
  // keep each lineage component contiguous, and offset derived datasets in X by their depth.
  const visDs = new Set(datasets.map(d => d.id));
  const parentOf = {};
  datasets.forEach(d => {
    const p = d.based_on && root(d.based_on);
    if (p && p !== d.id && visDs.has(p)) parentOf[d.id] = p;
  });
  const depth = id => { let n = 0, c = id, g = 0; while (parentOf[c] && g++ < 20) { c = parentOf[c]; n++; } return n; };
  const ancestor = id => { let c = id, g = 0; while (parentOf[c] && g++ < 20) c = parentOf[c]; return c; };
  const compDeg = {};
  datasets.forEach(d => { const a = ancestor(d.id); compDeg[a] = (compDeg[a]||0) + (deg[d.id]||0); });
  datasets.sort((a,b) => {
    const aa = ancestor(a.id), ab = ancestor(b.id);
    if (aa !== ab) return (compDeg[ab] - compDeg[aa]) || (aa < ab ? -1 : 1);
    const da = depth(a.id), db = depth(b.id);
    if (da !== db) return da - db;          // source above its derivatives
    return (deg[b.id]||0) - (deg[a.id]||0);
  });
  // model vertical positions: by publication date (time axis on) or by degree (off)
  const modelY = {};
  let yTop, yBottom, axisX;
  if (TIME_AXIS) {
    const T = m => Date.parse(m.publication_date);
    const dated = models.filter(m => m.publication_date);
    const times = dated.map(T);
    const minT = times.length ? Math.min(...times) : 0;
    const maxT = times.length ? Math.max(...times) : 1;
    const half = Math.max(datasets.length, models.length * 1.6, 10) * STEP / 2;
    yTop = -half; yBottom = half;
    const scaleY = t => (maxT === minT) ? 0 : yTop + (t - minT) / (maxT - minT) * (yBottom - yTop);
    // place in date order, nudging apart so clustered releases don't overlap
    let prevY = -Infinity; const MINGAP = STEP * 0.92;
    dated.slice().sort((a,b) => T(a) - T(b)).forEach(m => {
      let y = scaleY(T(m));
      if (y < prevY + MINGAP) y = prevY + MINGAP;
      modelY[m.id] = y; prevY = y;
    });
    models.filter(m => !m.publication_date).forEach(m => { prevY += MINGAP; modelY[m.id] = prevY; });
    axisX = COL_X + AXIS_GAP;
    AXIS = { minT, maxT, yTop, yBottom, axisX, xLeft: COL_X - AXIS_GAP };
  } else {
    models.sort((a,b) => (deg[b.id]||0) - (deg[a.id]||0));
    models.forEach((m,i) => modelY[m.id] = ypos(i, models.length));
    AXIS = null;
  }

  const nodes = [];
  datasets.forEach((d,i) => {
    const mem = members[d.id] || [d.label];
    nodes.push({
      id: d.id, label: d.label,
      title: d.label + "  (used by " + (deg[d.id]||0) + ")" +
             (mem.length > 1 ? " — merges " + mem.length + " editions: " + mem.join(", ") : ""),
      url: d.url, x: -COL_X + depth(d.id) * INDENT, y: ypos(i, datasets.length), fixed: true,
      shape: "box", color: { background: COL_DATASET, border: "#c9742a" },
      font: { color: "#1a1a1a", size: fsize(d.id) }
    });
  });
  models.forEach((m,i) => {
    const noSrc = !MODELS_WITH_SOURCES.has(m.id);
    const extra = [m.gpai && "GPAI notice", m.paper && "paper"].filter(Boolean);
    nodes.push({
      id: m.id, label: m.label,
      title: m.label + "  (" + (deg[m.id]||0) + " datasets)" +
             (extra.length ? " — " + extra.join(", ") : "") +
             (noSrc ? " — no disclosed public data sources" : ""),
      url: m.url, x: COL_X, y: modelY[m.id], fixed: true,
      shape: "box",
      color: { background: COL_MODEL, border: noSrc ? "#e02424" : (m.gpai ? "#1a3c8a" : "#2a55b0") },
      borderWidth: (noSrc || m.gpai) ? 4 : 1, borderWidthSelected: 4,
      font: { color: "#ffffff", size: fsize(m.id) }
    });
  });
  // invisible spacers so network.fit() keeps the axis + labels in view
  if (TIME_AXIS) [[yTop, "_axT"], [yBottom, "_axB"]].forEach(([y,id]) => nodes.push({
    id, x: axisX + 180, y, fixed: true, shape: "text", label: " ",
    color: "rgba(0,0,0,0)", font: { color: "rgba(0,0,0,0)" }
  }));

  // trained-on edges (dataset family -> model)
  const visEdges = edges.map((e, i) => ({
    id: "t" + i, from: e.dataset, to: e.model,
    color: { color: EDGE_COLOR[e.stage] || DEFAULT_EDGE, opacity: 0.55 }, width: 1,
    smooth: { type: "cubicBezier", forceDirection: "horizontal", roundness: 0.4 }
  }));

  // `based on` lineage: dashed within-column links (datasets remapped through family root,
  // models direct). Drawn only when both endpoints are visible; self-links skipped.
  const visN = new Set(nodes.map(n => n.id));
  let lj = 0;
  const lineage = (src, dst) => {
    if (src && dst && src !== dst && visN.has(src) && visN.has(dst))
      visEdges.push({ id: "b" + (lj++), from: src, to: dst,
        color: { color: BASED_EDGE, opacity: 0.85 }, width: 1.5, dashes: true,
        smooth: { type: "curvedCW", roundness: 0.3 } });
  };
  GRAPH.datasets.forEach(d => { if (d.based_on) lineage(root(d.based_on), root(d.id)); });
  GRAPH.models.forEach(m => { if (m.based_on) lineage(m.based_on, m.id); });

  // adjacency for hover-highlight
  neighbors = {};
  nodes.forEach(n => neighbors[n.id] = new Set());
  visEdges.forEach(e => {
    neighbors[e.from].add(e.to); neighbors[e.from].add("e" + e.id);
    neighbors[e.to].add(e.from);  neighbors[e.to].add("e" + e.id);
  });
  nodeBase = {}; nodes.forEach(n => nodeBase[n.id] = n.color);
  edgeBase = {}; visEdges.forEach(e => edgeBase[e.id] = { color: e.color.color, width: e.width });

  nodesDS.clear(); nodesDS.add(nodes);
  edgesDS.clear(); edgesDS.add(visEdges);

  document.getElementById("stats").innerHTML =
    "<b>" + models.length + "</b> models &middot; <b>" + datasets.length +
    "</b> datasets &middot; <b>" + edges.length + "</b> trained-on links";
}

function highlight(id) {
  const keepN = neighbors[id],
        nUpd = nodesDS.map(n => ({ id: n.id,
          color: (n.id === id || keepN.has(n.id)) ? nodeBase[n.id] : FADE_NODE }));
  const eUpd = edgesDS.map(e => {
    const on = keepN.has("e" + e.id);
    return { id: e.id, color: { color: on ? edgeBase[e.id].color : FADE_EDGE, opacity: on ? 0.9 : 0.25 },
             width: on ? 2.5 : 1 };
  });
  nodesDS.update(nUpd); edgesDS.update(eUpd);
}

function clearHighlight() {
  nodesDS.update(nodesDS.map(n => ({ id: n.id, color: nodeBase[n.id] })));
  edgesDS.update(edgesDS.map(e => ({ id: e.id,
    color: { color: edgeBase[e.id].color, opacity: 0.55 }, width: edgeBase[e.id].width })));
}

build("gpai");

const network = new vis.Network(document.getElementById("net"),
  { nodes: nodesDS, edges: edgesDS },
  { physics: false, interaction: { hover: true, tooltipDelay: 120 } });

network.on("hoverNode", p => highlight(p.node));
network.on("blurNode", clearHighlight);
network.on("click", p => {
  if (!p.nodes.length) return;
  const n = nodesDS.get(p.nodes[0]);
  if (n && n.url) window.open(n.url, "_blank");
});

// draw the publication-date axis + GPAI reference lines (in network coordinates)
network.on("afterDrawing", ctx => {
  if (!AXIS) return;
  const { minT, maxT, yTop, yBottom, axisX, xLeft } = AXIS;
  const Y = t => (maxT === minT) ? 0 : yTop + (t - minT) / (maxT - minT) * (yBottom - yTop);
  ctx.lineWidth = 1;

  // GPAI / AI Act milestone lines
  GPAI_DATES.forEach(g => {
    const t = Date.parse(g.date); if (t < minT || t > maxT) return;
    const y = Y(t);
    ctx.strokeStyle = "#e02424"; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(xLeft, y); ctx.lineTo(axisX, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#c81e1e"; ctx.font = "12px system-ui";
    ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText("\\u2696 " + g.label, xLeft + 4, y - 3);
  });

  // vertical axis line
  ctx.strokeStyle = "#888"; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(axisX, yTop - 30); ctx.lineTo(axisX, yBottom + 30); ctx.stroke();

  // year ticks + labels
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = "13px system-ui";
  const y0 = new Date(minT).getUTCFullYear(), y1 = new Date(maxT).getUTCFullYear();
  for (let yr = y0; yr <= y1; yr++) {
    const t = Date.parse(yr + "-01-01T00:00:00Z"); if (t < minT || t > maxT) continue;
    const y = Y(t);
    ctx.strokeStyle = "#bbb"; ctx.beginPath(); ctx.moveTo(axisX - 5, y); ctx.lineTo(axisX + 5, y); ctx.stroke();
    ctx.fillStyle = "#555"; ctx.fillText(String(yr), axisX + 10, y);
  }
});

const currentMode = () => document.querySelector('input[name=filter]:checked').value;

document.getElementById("filter").addEventListener("change", () => {
  build(currentMode());
  network.fit();
});

document.getElementById("timeaxis").addEventListener("change", e => {
  TIME_AXIS = e.target.checked;
  document.getElementById("axislegend").style.display = TIME_AXIS ? "block" : "none";
  build(currentMode());
  network.fit();
});
</script>
</body>
</html>
""")


def main():
    with open(os.path.join(SITE, "graph.json")) as f:
        graph = json.load(f)
    html = TEMPLATE.substitute(graph=json.dumps(graph, ensure_ascii=False))
    out = os.path.join(SITE, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote %s  (%d models, %d datasets, %d edges)"
          % (out, len(graph["models"]), len(graph["datasets"]), len(graph["edges"])))


if __name__ == "__main__":
    main()
