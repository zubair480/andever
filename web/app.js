"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v == null || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
const signed = (v, d = 2) => (v == null ? "-" : (v > 0 ? "+" : "") + Number(v).toFixed(d));

const PROFILE_FIELDS = [
  "age", "sex", "bmi", "pack_years", "exercise_minutes_per_week", "sleep_hours",
  "alcohol_units_per_week", "stress_level", "diet_quality", "glucose",
  "c_reactive_protein", "albumin", "creatinine", "lymphocyte_percent",
  "mean_cell_volume", "red_blood_cell_distribution_width", "alkaline_phosphate",
  "white_blood_cell_count",
];

const state = {
  meta: null, mcp: null, source: null, report: null,
  history: [], lastProfileAt: null, connected: false, attachedRunId: null,
  // The hosted build is stateless: the loop streams out of the POST that
  // starts it, there is no run to attach to later and no shared session.
  hosted: false,
};

/* ------------------------------------------------------------ presets */
const PRESETS = {
  "Metabolic risk": {
    age: 54, sex: 1, bmi: 30.5, smoking_status: "former", pack_years: 14,
    exercise_minutes_per_week: 60, sleep_hours: 6, alcohol_units_per_week: 12,
    stress_level: 7, diet_quality: 4, glucose: 104, c_reactive_protein: 3.1,
    albumin: 42, creatinine: 0.95, lymphocyte_percent: 28, mean_cell_volume: 90,
    red_blood_cell_distribution_width: 13.4, alkaline_phosphate: 72,
    white_blood_cell_count: 7.2,
  },
  "Current smoker": {
    age: 47, sex: 0, bmi: 26.2, smoking_status: "current", pack_years: 26,
    exercise_minutes_per_week: 90, sleep_hours: 6.5, alcohol_units_per_week: 16,
    stress_level: 8, diet_quality: 4, glucose: 94, c_reactive_protein: 4.4,
    albumin: 40, creatinine: 0.8, lymphocyte_percent: 24, mean_cell_volume: 93,
    red_blood_cell_distribution_width: 14.1, alkaline_phosphate: 84,
    white_blood_cell_count: 9.1,
  },
  "Already optimised": {
    age: 61, sex: 1, bmi: 22.8, smoking_status: "never", pack_years: 0,
    exercise_minutes_per_week: 320, sleep_hours: 8, alcohol_units_per_week: 1,
    stress_level: 2, diet_quality: 9, glucose: 84, c_reactive_protein: 0.4,
    albumin: 46, creatinine: 0.9, lymphocyte_percent: 34, mean_cell_volume: 88,
    red_blood_cell_distribution_width: 12.4, alkaline_phosphate: 58,
    white_blood_cell_count: 4.9,
  },
};

/* --------------------------------------------------------------- boot */
async function boot() {
  bindForm();
  bindTabs();
  bindModal();
  buildPresets();
  try {
    state.meta = await (await fetch("/api/meta")).json();
    state.hosted = !!state.meta.hosted;
    renderBadges();
    renderBackends();
    renderHarness();
  } catch (err) {
    $("#badges").append(el("span", "badge", "metadata unavailable"));
  }
  try {
    state.mcp = await (await fetch("/api/mcp-setup")).json();
    renderModal();
  } catch (err) { /* modal falls back to a message */ }

  if (state.hosted) {
    applyHostedMode();
  } else {
    pollConnection();
    setInterval(pollConnection, 2500);
  }
}

function applyHostedMode() {
  const cap = state.meta.max_iterations || 14;
  const slider = $("#form").elements.iterations;
  slider.max = String(cap);
  if (Number(slider.value) > cap) {
    slider.value = String(cap);
    slider.dispatchEvent(new Event("input"));
  }
  $("#agentCard").querySelector("p").textContent =
    "Point Claude Code, Codex or any MCP client at this URL and ask it about "
    + "your longevity. It reads your health data from wherever it lives and "
    + "calls the loop. Nothing you send is stored.";
}

function bindTabs() {
  $$("#tabs button").forEach((btn) => btn.addEventListener("click", () => showTab(btn.dataset.tab)));
}

function showTab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  ["loop", "report", "harness"].forEach((t) => { $("#tab-" + t).hidden = t !== name; });
}

function bindForm() {
  $$("input[type=range]").forEach((input) => {
    const out = $(`.hint[data-for="${input.name}"]`);
    if (out) input.addEventListener("input", () => { out.textContent = input.value; });
  });
  $("#form").addEventListener("submit", (ev) => { ev.preventDefault(); startRun(); });
}

function buildPresets() {
  const host = $("#presets");
  host.append(el("span", "badge", "or try"));
  Object.keys(PRESETS).forEach((name) => {
    const b = el("button", null, esc(name));
    b.type = "button";
    b.addEventListener("click", () => applyPreset(PRESETS[name]));
    host.append(b);
  });
}

function applyPreset(values) {
  const form = $("#form");
  Object.entries(values).forEach(([k, v]) => {
    const field = form.elements[k];
    if (!field) return;
    // An agent sends real measurements, so clamp them into each control's own
    // range and snap sliders to their step. Otherwise the form silently fails
    // validation and the run button does nothing.
    if (field.type === "range" || field.type === "number") {
      let n = Number(v);
      if (Number.isNaN(n)) return;
      if (field.min !== "") n = Math.max(n, Number(field.min));
      if (field.max !== "") n = Math.min(n, Number(field.max));
      if (field.type === "range") n = Math.round(n);
      v = n;
    }
    field.value = v;
    field.dispatchEvent(new Event("input"));
  });
}

function renderBadges() {
  const m = state.meta;
  const host = $("#badges");
  host.innerHTML = "";
  const ref = m.reference || {};
  host.append(el("span", "badge", `cohort <b>${esc(ref.cohort || "-")}</b> n=${ref.samples ?? "-"}`));
  host.append(el("span", "badge", `<b>${ref.cpgs ?? "-"}</b> CpGs`));
  host.append(el("span", "badge", `<b>${Object.keys(m.panel).length}</b> biolearn models`));
}

function renderBackends() {
  const sel = $("#backend");
  sel.innerHTML = "";
  const labels = {
    auto: "Auto", claude: `Claude (${state.meta.model})`,
    reasoner: "Built-in optimiser (no API key)",
  };
  ["auto", ...state.meta.backends].filter((v, i, a) => a.indexOf(v) === i)
    .forEach((b) => {
      const o = el("option", null, esc(labels[b] || b));
      o.value = b;
      sel.append(o);
    });
  $("#backend-note").textContent = state.meta.claude_available
    ? "ANTHROPIC_API_KEY found. Auto uses Claude to generate hypotheses."
    : "No ANTHROPIC_API_KEY found, so Auto uses the built-in optimiser. "
      + "It probes each axis, fits credit over the scored history and evolves "
      + "combinations, so the loop still improves.";
}

/* ------------------------------------------------------- MCP connect */
function bindModal() {
  const open = () => { $("#overlay").hidden = false; };
  const close = () => { $("#overlay").hidden = true; };
  $("#connectBtn").addEventListener("click", open);
  $("#agentCardBtn").addEventListener("click", open);
  $("#closeModal").addEventListener("click", close);
  $("#overlay").addEventListener("click", (ev) => {
    if (ev.target === $("#overlay")) close();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") close();
  });
}

function renderModal() {
  const mcp = state.mcp;
  const tabs = $("#clientTabs");
  const body = $("#clientBody");
  tabs.innerHTML = "";
  if (!mcp || !mcp.clients) {
    body.innerHTML = '<p class="step">The MCP endpoint did not start. '
      + 'Check the server console.</p>';
    return;
  }

  const show = (client) => {
    $$("#clientTabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.id === client.id));
    body.innerHTML = "";
    client.steps.forEach((s) => body.append(el("p", "step", esc(s))));
    body.append(codeBlock(client.command));
    if (client.verify) {
      body.append(el("p", "step", "Check it registered:"));
      body.append(codeBlock(client.verify));
    }
    body.append(el("p", "step",
      "Then ask your agent something like: <em>read my health data and run the "
      + "longevity loop on it</em>."));
  };

  mcp.clients.forEach((client, i) => {
    const b = el("button", i === 0 ? "active" : null, esc(client.name));
    b.dataset.id = client.id;
    b.addEventListener("click", () => show(client));
    tabs.append(b);
  });
  show(mcp.clients[0]);

  const tools = $("#mcpTools");
  tools.innerHTML = "";
  (mcp.tools || []).forEach((t) => tools.append(el("span", "chip", esc(t))));
}

function codeBlock(text) {
  const wrap = el("div", "codeblock");
  const bar = el("div", "bar");
  const btn = el("button", null, "Copy");
  btn.type = "button";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    btn.textContent = "Copied";
    btn.classList.add("done");
    setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("done"); }, 1600);
  });
  bar.append(btn);
  wrap.append(bar, el("pre", null, esc(text)));
  return wrap;
}

async function pollConnection() {
  let snap;
  try {
    snap = await (await fetch("/api/connection")).json();
  } catch (err) { return; }

  const live = !!snap.connected;
  state.connected = live;
  $("#connectDot").classList.toggle("live", live);
  $("#modalDot").classList.toggle("live", live);
  $("#connectBtn").classList.toggle("live", live);
  $("#connectLabel").textContent = live ? "Agent connected" : "Connect agent";
  $("#mcpLive").classList.toggle("on", live);
  $("#modalStatus").textContent = live
    ? `Connected, ${snap.calls} tool call${snap.calls === 1 ? "" : "s"}`
    : "Waiting for a client";

  // A profile pushed by the agent fills the form so it can be seen and edited.
  const fresh = snap.profile && snap.profile_at
    && snap.profile_at !== state.lastProfileAt
    && snap.profile_source !== "browser";
  if (fresh) {
    state.lastProfileAt = snap.profile_at;
    applyPreset(snap.profile);
    $("#moreDetail").open = true;
  }

  const status = $("#agentStatus");
  if (live) {
    status.hidden = false;
    const head = `Agent connected &middot; ${snap.calls} `
      + `tool call${snap.calls === 1 ? "" : "s"}`;
    const note = snap.profile
      ? `<div>Profile from ${esc(snap.profile_source || "agent")} is filled in below.</div>`
      : "";
    const trail = (snap.log || []).slice(-3)
      .map((e) => `<div class="trail">${esc(e.message)}</div>`).join("");
    status.innerHTML = head + note + trail;
  }

  // Attach to a run the agent started, so it streams into this tab too.
  if (snap.active_run_id && snap.active_run_id !== state.attachedRunId) {
    attachRun(snap.active_run_id);
  }
}

/* ------------------------------------------------------------ run     */
function readProfile() {
  const form = $("#form");
  const profile = {};
  PROFILE_FIELDS.forEach((k) => {
    const field = form.elements[k];
    const raw = field ? field.value : "";
    if (raw !== "" && raw != null && !Number.isNaN(Number(raw))) {
      profile[k] = Number(raw);
    }
  });
  profile.smoking_status = form.elements.smoking_status.value;
  return profile;
}

async function startRun() {
  const form = $("#form");
  if (!form.checkValidity()) {
    // Without this the browser blocks the submit and nothing visible happens.
    form.reportValidity();
    return;
  }
  const payload = {
    profile: readProfile(),
    iterations: Number(form.elements.iterations.value),
    backend: form.elements.backend.value,
  };

  prepareStage();
  if (state.hosted) return runStreaming(payload);

  let res;
  try {
    res = await (await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })).json();
  } catch (err) {
    return failRun(String(err));
  }
  if (res.error) return failRun(res.error);
  attachRun(res.run_id);
}

/* On the hosted build there is no run to come back to: the loop happens inside
   this one request and the events arrive on its response body. */
async function runStreaming(payload) {
  let response;
  try {
    response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    return failRun(String(err));
  }
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch (_) {}
    return failRun(message);
  }
  if (!response.body) {
    return failRun("this browser cannot read a streamed response");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split;
      while ((split = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw || raw === "{}") continue;
          try { handle(JSON.parse(raw)); } catch (_) { /* partial frame */ }
        }
      }
    }
  } catch (err) {
    failRun(String(err));
  }
  resetButton();
}

function prepareStage() {
  $("#runBtn").disabled = true;
  $("#runBtn").textContent = "Running";
  $("#loopEmpty").hidden = true;
  $("#loopLive").hidden = false;
  $("#iterations").innerHTML = "";
  $("#baselineCards").innerHTML = "";
  $("#loadBars").innerHTML = "";
  $("#lifespanNow").innerHTML = "";
  $("#reportBody").hidden = true;
  $("#reportEmpty").hidden = false;
  state.history = [];
  setStage("input");
  showTab("loop");
}

function attachRun(runId) {
  if (state.source) state.source.close();
  state.attachedRunId = runId;
  prepareStage();
  const source = new EventSource(`/api/stream/${runId}`);
  state.source = source;
  source.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch (_) { return; }
    handle(event);
  };
  source.addEventListener("end", () => {
    source.close();
    state.source = null;
    resetButton();
  });
  // Do NOT close on error. EventSource reconnects by itself, and the server
  // replays a run's events from the beginning, so a dropped connection costs
  // the view and not the run. Closing here defeats that and shows a failed run
  // that is in fact still going, which matters most on a host that sleeps.
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) {
      state.source = null;
      resetButton();
      status("connection closed before the run finished; reload to reattach");
    } else {
      status("connection dropped, reconnecting...");
    }
  };
}

function resetButton() {
  $("#runBtn").disabled = false;
  $("#runBtn").textContent = "Run the loop";
}

function failRun(message) {
  resetButton();
  status(`failed: ${message}`);
  $("#iterations").prepend(el("div", "banner error",
    `<b>Run failed.</b> ${esc(message)}`));
}

/* --------------------------------------------------------- events     */
function handle(ev) {
  switch (ev.type) {
    case "status": return status(ev.message);
    case "baseline": return renderBaseline(ev);
    case "iteration_start": return openIteration(ev);
    case "hypothesis": return fillHypothesis(ev);
    case "evaluation": return fillEvaluation(ev);
    case "warning": return status("note: " + ev.message);
    case "error": return failRun(ev.message);
    case "complete": return renderReport(ev);
  }
}

function status(message) { $("#statusline").textContent = message; }

function setStage(step) {
  const order = ["input", "agent", "hypothesis", "eval", "score", "store"];
  const at = order.indexOf(step);
  $$("#pipeline span").forEach((s) => {
    const i = order.indexOf(s.dataset.step);
    s.classList.toggle("on", i === at);
    s.classList.toggle("done", i < at);
  });
}

/* -------------------------------------------------------- lifespan    */
function lifespanBlock(life, treated) {
  const box = el("div", "lifespan" + (treated ? "" : " single"));
  const grid = el("div", "lifespan-grid");

  const side = (question, value, band, cls) => {
    const d = el("div");
    d.append(el("div", "life-q", question));
    d.append(el("div", "life-n " + cls, `${num(value.median_age, 0)}<small>years old</small>`));
    d.append(el("div", "life-band", band));
    return d;
  };

  grid.append(side(
    "<b>1. When are you gonna die?</b>Median age at death for a cohort carrying this biomarker profile.",
    life,
    `middle half ${num(life.quartile_low, 0)} to ${num(life.quartile_high, 0)} `
    + `&middot; hazard ratio ${num(life.hazard_ratio, 2)}`,
    "now"));

  if (treated) {
    const gap = el("div", "life-gap");
    gap.append(el("div", "arrow", "&rarr;"));
    gap.append(el("div", "delta", signed(treated.median_age - life.median_age, 1)));
    gap.append(el("div", "sub2", "years"));
    grid.append(gap);
    grid.append(side(
      "<b>2. And if you do something about it?</b>Same cohort, holding the protocol below for life.",
      treated,
      `middle half ${num(treated.quartile_low, 0)} to ${num(treated.quartile_high, 0)} `
      + `&middot; hazard ratio ${num(treated.hazard_ratio, 2)}`,
      "after"));
  } else {
    grid.append(el("div", "life-gap",
      '<div class="arrow">&rarr;</div><div class="sub2">question 2 needs<br>the loop to finish</div>'));
    grid.append(el("div", "life-q",
      "<b>2. And if you do something about it?</b>The loop is searching for "
      + "the intervention that moves that date furthest."));
  }

  box.append(grid);
  const drivers = Object.entries(life.drivers || {})
    .map(([k, v]) => `${esc(k)} ${signed(v.value, 2)}`).join(" &middot; ") || "no measurable acceleration";
  box.append(el("div", "lifespan-foot",
    `Driven by: ${drivers}. This is a cohort projection from published mortality `
    + `hazard ratios, not a prediction about one person, and the quartile band is `
    + `the honest width of it.`));
  return box;
}

/* -------------------------------------------------------- baseline    */
function renderBaseline(ev) {
  const chrono = Number($("#form").elements.age.value);

  if (ev.lifespan) {
    $("#lifespanNow").innerHTML = "";
    $("#lifespanNow").append(lifespanBlock(ev.lifespan, null));
  }

  const host = $("#baselineCards");
  host.innerHTML = "";
  const add = (k, v, s, cls) => {
    const c = el("div", "metric" + (cls ? " " + cls : ""));
    c.append(el("div", "k", esc(k)), el("div", "v", v), el("div", "s", esc(s || "")));
    host.append(c);
  };

  add("Chronological", num(chrono, 0), "years");
  ["GrimAgeV2", "PhenoAge", "Horvathv1", "DunedinPACE"].forEach((name) => {
    const value = ev.readout[name];
    if (value == null) return;
    const spec = ev.panel[name];
    if (spec.kind === "age") {
      // Acceleration is measured against a neutral-lifestyle peer of the same
      // age, not against chronological age: several models carry a platform
      // offset that would otherwise read as the subject's own acceleration.
      const accel = (ev.acceleration || {})[name];
      add(spec.label, num(value, 1), `${signed(accel, 1)} yr vs a neutral peer`,
          accel > 1.5 ? "bad" : accel < -1.5 ? "good" : "");
    } else {
      add(spec.label, num(value, 3),
          value > 1 ? "aging faster than 1 yr/yr" : "aging slower than 1 yr/yr",
          value > 1 ? "bad" : "good");
    }
  });
  if (ev.clinical_phenoage) {
    const c = ev.clinical_phenoage;
    add("Clinical PhenoAge", num(c.value, 1),
        `${signed(c.acceleration, 1)} yr, from blood panel`,
        c.acceleration > 1.5 ? "bad" : c.acceleration < -1.5 ? "good" : "");
  }

  const bars = $("#loadBars");
  bars.innerHTML = "";
  const loads = Object.entries(ev.loads || {}).sort((a, b) => b[1] - a[1]);
  if (!loads.length) {
    bars.append(el("p", "note",
      "No modifiable burden detected from the submitted parameters. The only "
      + "headroom left is intrinsic age drift, so scores will be small."));
  }
  loads.forEach(([axis, value]) => {
    const row = el("div", "bar-row");
    row.append(el("div", "name", esc((ev.axes[axis] || {}).label || axis)));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill" + (value > 0.8 ? " warn" : value < 0 ? " good" : ""));
    fill.style.width = Math.min(100, (Math.abs(value) / 1.5) * 100) + "%";
    if (value < 0) fill.style.marginLeft = "auto";
    track.append(fill);
    row.append(track, el("div", "val", signed(value, 2)));
    bars.append(row);
  });

  drawChart([]);
  status(`Baseline scored on ${Object.keys(ev.readout).length} biolearn models. `
    + `Agent: ${ev.agent}`);
}

/* ------------------------------------------------------- iterations   */
function openIteration(ev) {
  setStage("agent");
  const card = el("div", "iter");
  card.id = "iter-" + ev.iteration;
  const head = el("div", "iter-head");
  head.append(
    el("span", "iter-n", String(ev.iteration).padStart(2, "0")),
    el("span", "mode " + ev.mode, esc(ev.mode)),
    el("span", "iter-title", "thinking..."),
    el("span", "spinner"),
  );
  card.append(head, el("div", "iter-body", ""));
  $("#iterations").append(card);
}

function fillHypothesis(ev) {
  setStage("hypothesis");
  const card = $("#iter-" + ev.iteration);
  if (!card) return;
  const h = ev.hypothesis;
  $(".iter-title", card).textContent = h.title;

  const left = el("div");
  left.append(el("p", "rationale", esc(h.rationale)));
  left.append(el("div", "sub", "Mechanism axes targeted"));
  const bars = el("div", "bars");
  h.targets.forEach((t) => {
    const row = el("div", "bar-row");
    row.append(el("div", "name", esc((state.meta?.axes[t.axis] || {}).label || t.axis)));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = t.intensity * 100 + "%";
    track.append(fill);
    row.append(track, el("div", "val", num(t.intensity, 2)));
    bars.append(row);
  });
  left.append(bars);

  const right = el("div");
  right.append(el("div", "sub", "Composed protocol"));
  const chips = el("div", "chips");
  (h.protocol || []).forEach((p) => chips.append(el("span", `chip grade-${p.grade}`,
    `<b>${esc(p.name)}</b> &middot; ${p.grade}`)));
  if (!(h.protocol || []).length) chips.append(el("span", "chip", "none"));
  right.append(chips);
  right.append(el("div", "sub", "Primary endpoint"));
  right.append(el("p", "rationale", esc(h.primary_endpoint || "-")));
  right.append(el("p", "note", `proposed by ${esc(h.source || "agent")} in ${ev.think_ms} ms`));

  const body = $(".iter-body", card);
  body.innerHTML = "";
  body.append(left, right);
  setStage("eval");
}

function fillEvaluation(ev) {
  setStage("score");
  const card = $("#iter-" + ev.iteration);
  if (!card) return;
  $(".spinner", card)?.remove();
  const s = ev.scored;

  $(".iter-head", card).append(
    el("span", "pill", `${signed(ev.reward, 3)} reward &middot; `
      + `${signed(s.years_reversed, 2)} yr &middot; ${ev.eval_ms} ms`),
    el("span", "score-chip " + (ev.reward > 0 ? "good" : "bad"), num(ev.score, 1)));
  if (ev.new_best) card.classList.add("best");

  const rows = Object.entries(s.gains)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6)
    .map(([model, gain]) => {
      const spec = state.meta?.panel[model] || {};
      const held = (s.held_out || []).includes(model);
      const cls = gain > 0.15 ? "up" : gain < -0.15 ? "down" : "flat";
      return `<tr class="${held ? "heldout" : ""}"><td>${esc(spec.label || model)}`
        + `${held ? ' <span class="tag">held out</span>' : ""}</td>`
        + `<td class="num ${cls}">${signed(gain, 2)}</td></tr>`;
    }).join("");
  const table = el("div", "tablewrap",
    `<table><thead><tr><th>Panel model</th><th class="num">Year-equivalent</th>`
    + `</tr></thead><tbody>${rows}</tbody></table>`);

  const penalties = Object.entries(s.penalties || {})
    .filter(([, v]) => v > 0.005).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<span class="chip">${esc(k)} <b>-${num(v, 2)}</b></span>`)
    .join("") || '<span class="chip">none</span>';

  const right = $(".iter-body", card).lastChild;
  right.append(el("div", "sub", "Largest panel movements"), table);
  right.append(el("div", "sub", "Penalties charged"), el("div", "chips", penalties));
  $(".iter-body", card).after(el("div", "insight", esc(ev.insight)));

  state.history.push({ iteration: ev.iteration, reward: ev.reward });
  drawChart(state.history);
  setStage("store");
  status(`iteration ${ev.iteration} scored ${num(ev.score, 1)} `
    + `(best so far ${num(ev.best_so_far, 1)})`);
}

/* ------------------------------------------------------------ chart   */
function drawChart(history) {
  const svg = $("#rewardChart");
  const W = 640, H = 220, padL = 42, padR = 12, padT = 14, padB = 26;
  if (!history.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" fill="#5d7189" font-size="13"
      text-anchor="middle">waiting for the first score</text>`;
    return;
  }
  const values = history.map((h) => h.reward);
  let best = -Infinity;
  const running = values.map((v) => (best = Math.max(best, v)));
  let lo = Math.min(...values, 0), hi = Math.max(...values, 0.5);
  const pad = (hi - lo) * 0.15 || 0.5;
  lo -= pad; hi += pad;

  const x = (i) => padL + (history.length === 1 ? 0.5 : i / (history.length - 1))
    * (W - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  const grid = [lo, (lo + hi) / 2, hi].map((v) =>
    `<line x1="${padL}" x2="${W - padR}" y1="${y(v)}" y2="${y(v)}" stroke="#1a2532"/>`
    + `<text x="${padL - 6}" y="${y(v) + 4}" fill="#5d7189" font-size="10"
       text-anchor="end" font-family="monospace">${v.toFixed(1)}</text>`).join("");
  const zero = lo < 0 && hi > 0
    ? `<line x1="${padL}" x2="${W - padR}" y1="${y(0)}" y2="${y(0)}"
        stroke="#2c3d54" stroke-dasharray="3 3"/>` : "";
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  const dots = values.map((v, i) =>
    `<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="${v >= 0 ? "#6aa8f0" : "#f0645e"}"/>`).join("");

  svg.innerHTML = grid + zero
    + `<path d="${path(running)}" fill="none" stroke="#46d6a4" stroke-width="2"/>`
    + `<path d="${path(values)}" fill="none" stroke="#6aa8f0" stroke-width="1.4"
        stroke-opacity=".85"/>` + dots
    + `<text x="${W - padR}" y="${H - 6}" fill="#5d7189" font-size="10"
        text-anchor="end">iteration ${history.length}</text>`;
}

/* ----------------------------------------------------------- report   */
function renderReport(ev) {
  setStage("store");
  const r = ev.report;
  state.report = r;
  $("#reportEmpty").hidden = true;
  const host = $("#reportBody");
  host.hidden = false;
  host.innerHTML = "";

  const life = r.lifespan || {};
  if (life.current && life.treated) {
    host.append(lifespanBlock(life.current, life.treated));
    $("#lifespanNow").innerHTML = "";
    $("#lifespanNow").append(lifespanBlock(life.current, life.treated));
  }

  const h = r.headline || {};
  const metric = (k, v, s, cls) => {
    const c = el("div", "metric" + (cls ? " " + cls : ""));
    c.append(el("div", "k", esc(k)), el("div", "v", v), el("div", "s", esc(s)));
    return c;
  };
  const cards = el("div", "cards");
  cards.append(
    metric("Mortality hazard", `-${num((life.hazard_reduction || 0) * 100, 0)}%`,
           "against the current trajectory", "good"),
    metric("GrimAge V2", signed(h.grimage_years, 2) + " yr", "mortality clock"),
    metric("DunedinPACE", num(h.pace_after, 3), `from ${num(h.pace_before, 3)} yr/yr`,
           h.pace_after < h.pace_before ? "good" : ""),
    metric("Panel mean", signed(h.years_reversed, 2) + " yr",
           h.above_noise ? "above the assay noise floor" : `below the ${h.noise_floor} yr noise floor`,
           h.above_noise ? "good" : "warn"),
    metric("Best score", num(r.best.score, 1),
           `iteration ${r.best.iteration} of ${r.learning.per_iteration.length}`,
           r.best.score >= 60 ? "good" : ""),
    metric("Loop gain", signed(r.learning.improvement, 2),
           `reward, first ${num(r.learning.first, 2)} to best ${num(r.learning.best, 2)}`,
           r.learning.improved_over_first ? "good" : "warn"),
  );
  host.append(cards);

  const confidenceGood = /coherent across families/.test(h.confidence || "");
  host.append(el("div", "banner" + (confidenceGood ? "" : " error"),
    `<b>Confidence:</b> ${esc(h.confidence)}. Coherence across clock families `
    + `${num(h.coherence * 100, 0)}%, generalisation gap ${signed(h.generalisation_gap, 2)} `
    + `year-equivalents.`));

  const s1 = section("Winning hypothesis");
  s1.append(el("h3", null, esc(r.best.title)));
  s1.append(el("p", "rationale", esc(r.best.rationale)));
  const chips = el("div", "chips");
  r.best.targets.forEach((t) => chips.append(el("span", "chip",
    `${esc((state.meta?.axes[t.axis] || {}).label || t.axis)} <b>${num(t.intensity, 2)}</b>`)));
  s1.append(chips);
  s1.append(el("p", "note",
    `Primary endpoint: ${esc(r.best.primary_endpoint)} | Falsifier: ${esc(r.best.falsifier)}`));
  host.append(s1);

  const s2 = section("Recommended protocol");
  r.protocol.forEach((p) => {
    const item = el("div", "protocol-item");
    item.append(el("div", "grade " + p.grade, p.grade));
    const body = el("div");
    const tags = (p.stance === "maintain" ? ' <span class="tag">already in place, maintain</span>' : "")
      + (p.prescription_only ? ' <span class="tag rx">prescription</span>' : "")
      + (p.category === "experimental" ? ' <span class="tag exp">experimental</span>' : "");
    body.append(el("div", "name", esc(p.name) + tags));
    body.append(el("div", "detail", esc(p.detail)));
    body.append(el("div", "note", esc(p.note)));
    if (p.axes.length) body.append(el("div", "note", "acts on: " + p.axes.map(esc).join(", ")));
    item.append(body);
    s2.append(item);
  });
  const q = r.protocol_quality || {};
  s2.append(el("p", "note",
    `Mean evidence weight ${num(q.evidence, 2)} of 1.00, highest single-item risk `
    + `${num(q.risk, 2)}, ${q.experimental || 0} experimental item(s).`));
  host.append(s2);

  const s3 = section("Bio-eval panel, before and after");
  s3.append(el("div", "tablewrap",
    `<table><thead><tr><th>Model</th><th>Family</th><th class="num">Before</th>
      <th class="num">After</th><th class="num">Delta</th>
      <th class="num">Year-equivalent</th></tr></thead><tbody>`
    + r.panel.map((row) => {
      const cls = row.years_equivalent > 0.15 ? "up"
        : row.years_equivalent < -0.15 ? "down" : "flat";
      return `<tr class="${row.held_out ? "heldout" : ""}">
        <td>${esc(row.label)}${row.held_out ? ' <span class="tag">held out</span>' : ""}</td>
        <td><small>${esc(row.family)}</small></td>
        <td class="num">${num(row.before, 3)}</td>
        <td class="num">${num(row.after, 3)}</td>
        <td class="num">${signed(row.delta, 3)}</td>
        <td class="num ${cls}">${signed(row.years_equivalent, 2)}</td></tr>`;
    }).join("") + "</tbody></table>"));
  host.append(s3);

  if ((r.surrogates || []).length) {
    const s4 = section("GrimAge DNAm surrogate biomarkers");
    s4.append(el("div", "tablewrap",
      `<table><thead><tr><th>Surrogate</th><th class="num">Before</th>
        <th class="num">After</th><th class="num">Change</th></tr></thead><tbody>`
      + r.surrogates.map((row) => {
        const cls = row.percent < -1 ? "up" : row.percent > 1 ? "down" : "flat";
        return `<tr><td>${esc(row.label)}</td>
          <td class="num">${num(row.before, 2)}</td>
          <td class="num">${num(row.after, 2)}</td>
          <td class="num ${cls}">${signed(row.percent, 1)}%</td></tr>`;
      }).join("") + "</tbody></table>"));
    s4.append(el("p", "note",
      "These are DNAm predictors of measured plasma proteins, not the proteins "
      + "themselves. A drop in the CRP surrogate is a hypothesis to confirm with "
      + "an actual hs-CRP draw."));
    host.append(s4);
  }

  const s5 = section("How to check this in a real subject");
  s5.append(el("div", "tablewrap",
    `<table><thead><tr><th>Measure</th><th>When</th><th>Why</th></tr></thead><tbody>`
    + r.monitoring.map((m) => `<tr><td>${esc(m.what)}</td>
        <td><small>${esc(m.when)}</small></td><td><small>${esc(m.why)}</small></td></tr>`)
      .join("") + "</tbody></table>"));
  host.append(s5);

  const s6 = section("Loop training output");
  s6.append(el("p", "rationale",
    `The run wrote <b>${ev.training.pairs}</b> preference pairs and `
    + `<b>${ev.training.sft}</b> supervised examples. Each pair is the same `
    + "prompt with a higher-scoring hypothesis as chosen and a lower-scoring "
    + "one as rejected, which is the format DPO consumes. This is the step "
    + "that turns harness results back into training signal."));
  const downloads = el("div", "downloads");
  if (ev.training.inline) {
    // Stateless build: the dataset came back in the response rather than
    // being written to a file on a server, so it is turned into a download
    // here in the browser.
    downloads.append(
      blobLink("dpo_pairs.jsonl", ev.training.dpo || []),
      blobLink("sft_dataset.jsonl", ev.training.sft_rows || []));
    s6.append(el("p", "note",
      "This deployment stores nothing, so the dataset is built in your browser "
      + "from the response. Nothing you submitted was written down."));
  } else {
    downloads.innerHTML =
      `<a href="/api/dataset/dpo" download>Download dpo_pairs.jsonl</a>
       <a href="/api/dataset/sft" download>Download sft_dataset.jsonl</a>`;
  }
  s6.append(downloads);
  host.append(s6);

  const s7 = section("What this is and is not");
  const ul = el("ul", "caveats");
  r.caveats.forEach((c) => ul.append(el("li", null, esc(c))));
  s7.append(ul);
  host.append(s7);

  status(`complete in ${ev.elapsed_s}s. Report ready.`);
  showTab("report");
}

function section(title) {
  const s = el("section", "section");
  s.append(el("h2", null, esc(title)));
  return s;
}

/** A download link built from data already in the page, as JSON Lines. */
function blobLink(filename, rows) {
  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  const link = el("a", null, `Download ${esc(filename)} (${rows.length})`);
  link.href = URL.createObjectURL(new Blob([body], {type: "application/x-ndjson"}));
  link.download = filename;
  return link;
}

/* ---------------------------------------------------------- harness   */
function renderHarness() {
  const m = state.meta;
  const host = $("#harnessBody");
  host.innerHTML = "";

  const ref = m.reference || {};
  host.append(el("div", "banner",
    `<b>Reference epigenome.</b> Fitted per CpG on GEO series `
    + `<b>${esc(ref.cohort)}</b> (n=${ref.samples}, ages ${ref.age_min} to `
    + `${ref.age_max}) loaded through biolearn, restricted to the `
    + `${ref.cpgs} CpGs the panel and the axes use. Each CpG gets a regression `
    + `on age, sex and disease status, which is what defines both the subject's `
    + `expected methylome and the young-adult target an intervention aims at.`));

  const s1 = section("Evaluation panel");
  s1.append(el("div", "tablewrap",
    `<table><thead><tr><th>biolearn model</th><th>Family</th><th>Output</th>
      <th class="num">Reward weight</th><th>Direction</th></tr></thead><tbody>`
    + Object.entries(m.panel).map(([name, spec]) =>
      `<tr class="${m.holdout.includes(name) ? "heldout" : ""}">
        <td><b>${esc(name)}</b><br><small>${esc(spec.label)}</small>
        ${m.holdout.includes(name) ? '<span class="tag">holdout</span>' : ""}</td>
        <td><small>${esc(spec.family)}</small></td>
        <td><small>${esc(spec.unit)}</small></td>
        <td class="num">${num(spec.weight, 1)}</td>
        <td><small>${spec.direction > 0 ? "lower is better" : "higher is better"}</small></td>
      </tr>`).join("") + "</tbody></table>"));
  s1.append(el("p", "note",
    "Improvements are divided by each model's own drift per year of age, "
    + "measured on real cohort samples, so a telomere index, a mortality "
    + "z-score and a methylation clock all end up in the same unit: years of "
    + "biological aging undone. The holdout rows are dropped from the reward on "
    + "alternating iterations and reported back as a generalisation gap."));
  host.append(s1);

  const s2 = section("Mechanism axes");
  Object.entries(m.axes).forEach(([name, spec]) => {
    const item = el("div", "axis-item");
    item.append(el("div", "name", esc(spec.label)));
    item.append(el("div", "bio", esc(spec.biology)));
    item.append(el("div", "meta",
      `${esc(name)} | source: ${esc(spec.source)} | burden ${num(spec.burden, 2)} `
      + `| ${(spec.mechanisms || []).map(esc).join(", ")}`));
    s2.append(item);
  });
  host.append(s2);

  const s3 = section("Intervention catalogue");
  s3.append(el("div", "tablewrap",
    `<table><thead><tr><th>Intervention</th><th>Category</th>
      <th class="num">Evidence</th><th class="num">Risk</th></tr></thead><tbody>`
    + m.interventions.map((i) => `<tr><td>${esc(i.name)}</td>
        <td><small>${esc(i.category)}</small></td>
        <td class="num"><span class="grade ${i.grade}"
          style="display:inline-grid;width:22px;height:22px;font-size:12px">${i.grade}</span></td>
        <td class="num">${num(i.risk, 2)}</td></tr>`).join("") + "</tbody></table>"));
  s3.append(el("p", "note",
    "Grades: A randomised human trials with hard or validated surrogate "
    + "endpoints; B human trials with biomarker endpoints or large consistent "
    + "cohorts; C early or small human data; D animal or in-vitro only."));
  host.append(s3);
}

boot();
