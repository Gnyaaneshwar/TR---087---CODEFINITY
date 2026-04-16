/**
 * GraspAI — Premium Frontend JS v2.0
 * Neural canvas · Custom cursor · Bento animations ·
 * Pipeline polling · Gauges · Terminal · Results render
 */

/* ═══════════════════════════════════════════════════
   1. NEURAL NETWORK CANVAS
   ═══════════════════════════════════════════════════ */
(function NeuralCanvas() {
  const canvas = document.getElementById("neural-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W, H, nodes = [], mouse = { x: -999, y: -999 };
  const NODE_COUNT = 80, LINK_DIST = 160, MOUSE_DIST = 120;

  class Node {
    constructor() { this.reset(); this.y = Math.random() * H; }
    reset() {
      this.x  = Math.random() * W;
      this.y  = Math.random() * H;
      this.vx = (Math.random() - .5) * .35;
      this.vy = (Math.random() - .5) * .35;
      this.r  = Math.random() * 1.8 + .6;
      this.alpha = Math.random() * .5 + .2;
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > W) this.vx *= -1;
      if (this.y < 0 || this.y > H) this.vy *= -1;

      // attract to mouse softly
      const dx = mouse.x - this.x, dy = mouse.y - this.y;
      const d  = Math.sqrt(dx*dx + dy*dy);
      if (d < MOUSE_DIST) {
        this.vx += (dx / d) * .015;
        this.vy += (dy / d) * .015;
      }
      // speed cap
      const spd = Math.sqrt(this.vx*this.vx + this.vy*this.vy);
      if (spd > .8) { this.vx /= spd; this.vy /= spd; }
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,229,255,${this.alpha})`;
      ctx.fill();
    }
  }

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function init() {
    resize();
    nodes = Array.from({ length: NODE_COUNT }, () => new Node());
  }

  function drawLinks() {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < LINK_DIST) {
          const alpha = (1 - d / LINK_DIST) * .35;
          // gradient cyan→purple
          const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
          grad.addColorStop(0, `rgba(0,229,255,${alpha})`);
          grad.addColorStop(1, `rgba(191,90,242,${alpha * .6})`);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = grad;
          ctx.lineWidth = .7;
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    nodes.forEach(n => { n.update(); });
    drawLinks();
    nodes.forEach(n => n.draw());
    requestAnimationFrame(loop);
  }

  window.addEventListener("resize", resize);
  document.addEventListener("mousemove", e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  init();
  loop();
})();


/* ═══════════════════════════════════════════════════
   2. CUSTOM CURSOR
   ═══════════════════════════════════════════════════ */
(function Cursor() {
  const dot  = document.getElementById("cursor-dot");
  const ring = document.getElementById("cursor-ring");
  if (!dot || !ring) return;

  let mx = 0, my = 0, rx = 0, ry = 0;
  document.addEventListener("mousemove", e => { mx = e.clientX; my = e.clientY; });

  function animateCursor() {
    rx += (mx - rx) * .12;
    ry += (my - ry) * .12;
    dot.style.left  = mx + "px"; dot.style.top  = my + "px";
    ring.style.left = rx + "px"; ring.style.top = ry + "px";
    requestAnimationFrame(animateCursor);
  }
  animateCursor();

  // hover enlargement
  document.querySelectorAll("a,button,.bento-card,.drop-card").forEach(el => {
    el.addEventListener("mouseenter", () => ring.classList.add("hover"));
    el.addEventListener("mouseleave", () => ring.classList.remove("hover"));
  });
  document.addEventListener("mousedown", () => ring.classList.add("click"));
  document.addEventListener("mouseup",   () => ring.classList.remove("click"));
})();


/* ═══════════════════════════════════════════════════
   3. NAV SCROLL EFFECT
   ═══════════════════════════════════════════════════ */
window.addEventListener("scroll", () => {
  document.getElementById("nav").classList.toggle("scrolled", window.scrollY > 40);
});


/* ═══════════════════════════════════════════════════
   4. COUNTER ANIMATION (hero metrics)
   ═══════════════════════════════════════════════════ */
function animateCounter(el) {
  const target = parseInt(el.dataset.count);
  if (isNaN(target)) return;
  const dur = 1400, start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.floor(ease * target);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = target;
  }
  requestAnimationFrame(step);
}
const heroObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll("[data-count]").forEach(animateCounter);
      heroObserver.unobserve(e.target);
    }
  });
}, { threshold: .5 });
const heroContent = document.querySelector(".hero-content");
if (heroContent) heroObserver.observe(heroContent);


/* ═══════════════════════════════════════════════════
   5. BENTO CARD ENTRANCE (Intersection Observer)
   ═══════════════════════════════════════════════════ */
const cardObserver = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      setTimeout(() => e.target.classList.add("visible"), i * 60);
      cardObserver.unobserve(e.target);
    }
  });
}, { threshold: .1 });
document.querySelectorAll(".bento-card").forEach(c => cardObserver.observe(c));


/* ═══════════════════════════════════════════════════
   6. DRAG & DROP UPLOAD
   ═══════════════════════════════════════════════════ */
let rgbFile = null, depthFile = null;

setupDrop("rgb-zone",   "rgb-input",   "rgb-preview",   "rgb-thumb",   "rgb-name",
          f => { rgbFile   = f; checkReady(); });
setupDrop("depth-zone", "depth-input", "depth-preview", "depth-thumb", "depth-name",
          f => { depthFile = f; checkReady(); });

function setupDrop(zoneId, inputId, prevId, thumbId, nameId, onFile) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const prev  = document.getElementById(prevId);
  const thumb = document.getElementById(thumbId);
  const nameEl= document.getElementById(nameId);

  zone.addEventListener("click",    () => input.click());
  zone.addEventListener("keydown",  e => { if(e.key==="Enter"||e.key===" ") input.click(); });

  zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if(input.files[0]) loadFile(input.files[0]); });

  function loadFile(f) {
    onFile(f);
    zone.classList.add("has-file");
    document.querySelector(`#${zoneId} .drop-body`).style.opacity = "0";
    const url = URL.createObjectURL(f);
    thumb.src = url; nameEl.textContent = f.name;
    prev.hidden = false;
  }
}

function checkReady() {
  document.getElementById("run-custom-btn").disabled = !(rgbFile && depthFile);
}


/* ═══════════════════════════════════════════════════
   7. RUN BUTTONS
   ═══════════════════════════════════════════════════ */
document.getElementById("run-custom-btn").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("rgb", rgbFile); fd.append("depth", depthFile);
  const res  = await fetch("/api/run", { method: "POST", body: fd });
  const data = await res.json();
  if (data.job_id) startJob(data.job_id);
});

document.getElementById("run-sample-btn").addEventListener("click", async () => {
  const res  = await fetch("/api/samples");
  const data = await res.json();
  if (data.job_id) startJob(data.job_id);
});


/* ═══════════════════════════════════════════════════
   8. PIPELINE POLLING
   ═══════════════════════════════════════════════════ */
const STEP_NAMES = [
  "Input", "Point Cloud", "Segmentation",
  "Candidates", "Scoring", "Collision",
  "Stability", "Ranking", "Physics",
  "Output", "Visualize"
];

let currentJobId = null, pollTimer = null, seenLogs = 0, activeStep = -1;

function startJob(jobId) {
  currentJobId = jobId;
  seenLogs = 0; activeStep = -1;
  document.getElementById("term-job-id").textContent = jobId;
  setNavStatus("running", "Running...");

  // Show progress section
  const ps = document.getElementById("progress-section");
  const rs = document.getElementById("output");
  ps.hidden = false; rs.hidden = true;

  buildStepTrack();
  document.getElementById("log-body").innerHTML = "";

  setTimeout(() => ps.scrollIntoView({ behavior: "smooth" }), 100);

  pollTimer = setInterval(() => poll(jobId), 700);
}

function buildStepTrack() {
  const track = document.getElementById("step-track");
  track.innerHTML = STEP_NAMES.map((name, i) => `
    <div class="st-item" id="st-${i}">
      <div class="st-circle">${String(i+1).padStart(2,"0")}</div>
      <div class="st-label">${name}</div>
      <div class="st-time" id="st-time-${i}"></div>
    </div>`).join("");
}

async function poll(jobId) {
  let data;
  try { data = await (await fetch(`/api/status/${jobId}`)).json(); }
  catch(e) { return; }

  // Append new logs
  const newLogs = (data.logs || []).slice(seenLogs);
  const logBody = document.getElementById("log-body");
  newLogs.forEach(l => {
    const cls = logClass(l.msg);
    const div = document.createElement("div");
    div.className = "log-line";
    div.innerHTML = `<span class="log-ts">${l.time}</span><span class="${cls}">${esc(l.msg)}</span>`;
    logBody.appendChild(div);
    logBody.scrollTop = logBody.scrollHeight;

    // detect step
    const m = l.msg.match(/\[Step\s+(\d+)\]/);
    if (m) {
      const idx = parseInt(m[1]) - 1;
      if (idx !== activeStep) {
        if (activeStep >= 0) markStep(activeStep, "done");
        activeStep = idx; markStep(idx, "active");
        document.getElementById("prog-title").textContent =
          `Running: ${STEP_NAMES[idx] || "..."}`;
      }
    }
  });
  seenLogs = data.logs.length;

  if (data.status === "done") {
    clearInterval(pollTimer);
    if (activeStep >= 0) markStep(activeStep, "done");
    document.getElementById("term-tail").textContent = "✓ complete";
    document.getElementById("term-tail").style.color = "var(--green)";
    document.getElementById("prog-title").textContent = "Pipeline Complete";
    setNavStatus("done", "Done");
    buildResults(jobId);
  } else if (data.status === "error") {
    clearInterval(pollTimer);
    if (activeStep >= 0) markStep(activeStep, "error");
    setNavStatus("error", "Error");
    document.getElementById("prog-title").textContent = "Pipeline Error";
  }
}

function markStep(i, state) {
  const el = document.getElementById(`st-${i}`);
  if (!el) return;
  el.classList.remove("active","done","error");
  el.classList.add(state);
  const dot = el.querySelector(".st-circle");
  if (state === "done") dot.innerHTML = "&#10003;";
}

function logClass(msg) {
  if (/\[Step/.test(msg))           return "log-step";
  if (/done|saved|complete|OK/i.test(msg)) return "log-ok";
  if (/warn|skip|fall/i.test(msg))  return "log-warn";
  if (/error|fail/i.test(msg))      return "log-err";
  return "log-plain";
}


/* ═══════════════════════════════════════════════════
   9. BUILD RESULTS
   ═══════════════════════════════════════════════════ */
let resultPayload = null;

async function buildResults(jobId) {
  const data = await (await fetch(`/api/result/${jobId}`)).json();
  resultPayload = data;

  const grasps  = data.top_grasps || [];
  const timings = data.timings || {};
  const total   = timings.total || 0;

  // Summary
  document.getElementById("results-summary").textContent =
    `Pipeline completed in ${total.toFixed(2)}s — ${grasps.length} top grasps ranked.`;

  // Visualization
  if (data.vis_b64) {
    const src = `data:image/png;base64,${data.vis_b64}`;
    document.getElementById("vis-img").src = src;
    const dlBtn = document.getElementById("dl-img-btn");
    dlBtn.href = src;
  }

  // Timings
  const timingList = document.getElementById("timing-list");
  timingList.innerHTML = "";
  const tEntries = Object.entries(timings).filter(([k]) => k !== "total");
  const maxT = Math.max(...tEntries.map(([,v]) => v), 0.001);
  tEntries.forEach(([key, val]) => {
    const label = key.replace(/step\d+_?/i,"").replace(/_/g," ");
    const pct   = Math.round((val / maxT) * 100);
    const row   = document.createElement("div");
    row.className = "timing-row";
    row.innerHTML = `
      <div class="timing-row-header">
        <span class="timing-name">${cap(label)}</span>
        <span class="timing-val">${val.toFixed(3)}s</span>
      </div>
      <div class="timing-bar-wrap">
        <div class="timing-bar-fill" style="width:0%" data-pct="${pct}%"></div>
      </div>`;
    timingList.appendChild(row);
  });

  // Total row
  const totalRow = document.createElement("div");
  totalRow.className = "timing-row";
  totalRow.style.cssText = "border-top:1px solid rgba(255,255,255,.06);padding-top:8px;margin-top:4px;";
  totalRow.innerHTML = `
    <div class="timing-row-header">
      <span class="timing-name" style="color:var(--cyan)">TOTAL</span>
      <span class="timing-val" style="color:var(--cyan)">${total.toFixed(3)}s</span>
    </div>
    <div class="timing-bar-wrap">
      <div class="timing-bar-fill" data-pct="100%"></div>
    </div>`;
  timingList.appendChild(totalRow);

  // Spotlight — best grasp
  buildSpotlight(grasps[0] || null);

  // Grasp table
  buildGraspTable(grasps);

  // JSON viewer
  const clean = { ...data }; delete clean.vis_b64;
  document.getElementById("json-body").textContent = JSON.stringify(clean, null, 2);

  // Mark all steps done
  STEP_NAMES.forEach((_, i) => markStep(i, "done"));
  // Fill timing labels on step track
  tEntries.forEach(([, v], i) => {
    const el = document.getElementById(`st-time-${i}`);
    if (el) el.textContent = v.toFixed(2) + "s";
  });

  // Show results
  const rs = document.getElementById("output");
  rs.hidden = false;
  setTimeout(() => rs.scrollIntoView({ behavior: "smooth" }), 200);

  // Animate bars after DOM paint
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".timing-bar-fill[data-pct]").forEach(el => {
      el.style.width = el.dataset.pct;
    });
  }));

  // Download JSON
  document.getElementById("dl-json-btn").onclick = () => downloadJSON(clean);

  // Toggle JSON
  document.getElementById("toggle-json").onclick = function() {
    const pre = document.getElementById("json-body");
    pre.hidden = !pre.hidden;
    this.textContent = pre.hidden ? "Show JSON" : "Hide JSON";
  };
}


/* ─────────────────── Spotlight ─────────────────── */
function buildSpotlight(g) {
  const el = document.getElementById("spotlight-body");
  if (!g) { el.innerHTML = "<p style='color:var(--t3)'>No grasp data.</p>"; return; }

  const p = g.position;
  const collCls = g.collision_status === "clear" ? "clear" : "collision";
  const collTxt = g.collision_status === "clear" ? "✓ Collision-Free" : "✗ Collision";

  el.innerHTML = `
    <div class="sg-pos">
      (${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) m
    </div>
    <div class="sg-gauges">
      ${gauge("Conf",  g.confidence,     "var(--cyan)")}
      ${gauge("Stab",  g.stability_score,"var(--purple)")}
      ${gauge("Score", g.final_score,    "var(--green)")}
    </div>
    <span class="sg-status ${collCls}">${collTxt}</span>`;

  // Animate gauge fills after render
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.querySelectorAll(".gauge-fill").forEach(arc => {
      const target = parseFloat(arc.dataset.target);
      // stroke-dasharray 132 = full circle (r=21, 2π*21≈132)
      arc.style.strokeDashoffset = 132 - (target * 132);
    });
  }));
}

function gauge(label, value, color) {
  const pct = Math.min(Math.max(value, 0), 1);
  const offset = 132; // start fully hidden
  return `
    <div class="gauge-wrap">
      <svg class="gauge-svg" viewBox="0 0 60 60">
        <circle class="gauge-arc" cx="30" cy="30" r="21"/>
        <circle class="gauge-fill" cx="30" cy="30" r="21"
          stroke="${color}" data-target="${pct}"
          style="stroke-dashoffset:${offset};transform:rotate(-90deg);transform-origin:50% 50%"/>
        <text class="gauge-val" x="30" y="30" style="fill:${color};font-size:9px;font-family:'JetBrains Mono'">${(value*100).toFixed(0)}%</text>
      </svg>
      <div class="gauge-label">${label}</div>
    </div>`;
}


/* ─────────────────── Grasp Table ─────────────────── */
function buildGraspTable(grasps) {
  const tbody = document.getElementById("grasp-tbody");
  tbody.innerHTML = "";
  grasps.forEach((g, i) => {
    const p = g.position;
    const isBest = i === 0;
    const rb = isBest
      ? `<span class="rank-badge gold">${g.rank}</span>`
      : `<span class="rank-badge">${g.rank}</span>`;
    const coll = g.collision_status === "clear"
      ? `<span class="coll-clear">Clear</span>`
      : `<span class="coll-hit">Collision</span>`;
    const phys = g.physics_validated
      ? `<span style="color:var(--green);font-weight:700">YES</span>`
      : `<span style="color:var(--t3)">—</span>`;
    const scoreW = Math.round(g.final_score * 80);
    const tr = document.createElement("tr");
    if (isBest) tr.className = "best-row";
    tr.innerHTML = `
      <td>${rb}</td>
      <td>${p.x.toFixed(4)}</td><td>${p.y.toFixed(4)}</td><td>${p.z.toFixed(4)}</td>
      <td>${g.confidence.toFixed(3)}</td>
      <td>${g.stability_score.toFixed(3)}</td>
      <td>${coll}</td>
      <td>
        <div class="score-cell">
          <div class="score-track"><div class="score-fill" style="width:${scoreW}px"></div></div>
          ${g.final_score.toFixed(4)}
        </div>
      </td>
      <td>${phys}</td>`;
    tbody.appendChild(tr);
  });
}


/* ═══════════════════════════════════════════════════
   10. HELPERS
   ═══════════════════════════════════════════════════ */
function setNavStatus(state, text) {
  const pill = document.getElementById("nav-status");
  const dot  = pill.querySelector(".status-dot");
  const txt  = pill.querySelector(".status-text");
  pill.className = `status-pill ${state}`;
  dot.className  = `status-dot ${state}`;
  txt.textContent = text;
}

function downloadJSON(data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "grasps.json"; a.click();
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function resetUI() { location.reload(); }
