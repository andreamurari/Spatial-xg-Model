(() => {
  const svg = document.getElementById("pitch");
  const markersLayer = document.getElementById("markers-layer");
  const hint = document.getElementById("hint");
  const NS = "http://www.w3.org/2000/svg";

  const state = {
    mode: "ball",
    shot: null, // {x, y}
    defenders: [], // [{x, y}]
    keeper: null, // {x, y}
  };

  const HINTS = {
    ball: "Click the pitch to place (or move) the shot.",
    defender: "Click empty grass to add a defender · click a red dot to remove it.",
    keeper: "Click to place the goalkeeper · click the dot again to remove it.",
  };

  // ---------------- mode toolbar ----------------
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      hint.textContent = HINTS[state.mode];
    });
  });

  document.getElementById("reset-btn").addEventListener("click", () => {
    state.shot = null;
    state.defenders = [];
    state.keeper = null;
    render();
    scheduleUpdate();
  });

  // ---------------- coordinate helpers ----------------
  function clientToPitch(evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    return {
      x: Math.min(120, Math.max(0, loc.x)),
      y: Math.min(80, Math.max(0, loc.y)),
    };
  }

  svg.addEventListener("pointerdown", (evt) => {
    if (evt.target.closest(".marker")) return; // handled by marker itself
    const p = clientToPitch(evt);
    if (state.mode === "ball") {
      state.shot = p;
    } else if (state.mode === "defender") {
      state.defenders.push(p);
    } else if (state.mode === "keeper") {
      state.keeper = p;
    }
    render();
    scheduleUpdate();
  });

  // ---------------- marker rendering + dragging ----------------
  function makeMarker(cls, p, onRemove, onMove) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "marker");
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", p.x);
    c.setAttribute("cy", p.y);
    c.setAttribute("r", 1.6);
    c.setAttribute("class", cls);
    g.appendChild(c);

    let moved = false;
    let dragging = false;

    g.addEventListener("pointerdown", (evt) => {
      evt.stopPropagation();
      dragging = true;
      moved = false;
      g.setPointerCapture(evt.pointerId);
    });
    g.addEventListener("pointermove", (evt) => {
      if (!dragging) return;
      moved = true;
      const np = clientToPitch(evt);
      c.setAttribute("cx", np.x);
      c.setAttribute("cy", np.y);
      onMove(np);
      renderCone();
    });
    g.addEventListener("pointerup", (evt) => {
      evt.stopPropagation();
      dragging = false;
      if (!moved && onRemove) {
        onRemove();
        render();
      }
      scheduleUpdate();
    });
    return g;
  }

  function render() {
    markersLayer.innerHTML = "";

    if (state.shot) {
      markersLayer.appendChild(
        makeMarker("m-ball", state.shot, null, (np) => (state.shot = np))
      );
    }

    state.defenders.forEach((d, i) => {
      markersLayer.appendChild(
        makeMarker(
          "m-defender",
          d,
          () => state.defenders.splice(i, 1),
          (np) => (state.defenders[i] = np)
        )
      );
    });

    if (state.keeper) {
      markersLayer.appendChild(
        makeMarker(
          "m-keeper",
          state.keeper,
          () => (state.keeper = null),
          (np) => (state.keeper = np)
        )
      );
    }

    renderCone();
  }

  function renderCone() {
    const coneLayer = document.getElementById("cone-layer");
    coneLayer.innerHTML = "";
    if (!state.shot) return;
    const poly = document.createElementNS(NS, "polygon");
    poly.setAttribute("points", `${state.shot.x},${state.shot.y} 120,36 120,44`);
    poly.setAttribute("class", "shot-cone");
    coneLayer.appendChild(poly);
  }

  const style = document.createElement("style");
  style.textContent = `
    .m-ball { fill: #ffffff; stroke: #1f2937; stroke-width: .3; cursor: grab; }
    .m-defender { fill: var(--danger); stroke: #7a1414; stroke-width: .25; cursor: grab; }
    .m-keeper { fill: var(--gold); stroke: #78350f; stroke-width: .25; cursor: grab; }
    .shot-cone { fill: rgba(230, 41, 58, 0.14); stroke: rgba(230, 41, 58, 0.45); stroke-width: .2; }
  `;
  document.head.appendChild(style);

  // ---------------- form ----------------
  const form = document.getElementById("controls");
  const hasAssist = document.getElementById("has-assist");
  const assistFields = document.getElementById("assist-fields");

  hasAssist.addEventListener("change", () => {
    assistFields.classList.toggle("hidden", !hasAssist.checked);
    scheduleUpdate();
  });

  const rangeOutputs = {
    assist_lunghezza: "assist-len-out",
  };
  Object.entries(rangeOutputs).forEach(([name, outId]) => {
    const input = form.querySelector(`[name="${name}"]`);
    const out = document.getElementById(outId);
    if (input && out) input.addEventListener("input", () => (out.textContent = input.value));
  });

  form.addEventListener("input", scheduleUpdate);
  form.addEventListener("change", scheduleUpdate);

  // ---------------- backend call ----------------
  let debounceTimer = null;
  function scheduleUpdate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(update, 200);
  }

  function fd() {
    const data = new FormData(form);
    const obj = {};
    data.forEach((v, k) => (obj[k] = v));
    return obj;
  }

  async function update() {
    const resultEmpty = document.getElementById("result-empty");
    const resultBody = document.getElementById("result-body");

    if (!state.shot) {
      resultEmpty.classList.remove("hidden");
      resultBody.classList.add("hidden");
      return;
    }

    const f = fd();
    const payload = {
      shot: state.shot,
      defenders: state.defenders,
      keeper: state.keeper,
      details: {
        body_part: f.body_part,
        shot_type: f.shot_type,
        shot_technique: f.shot_technique,
        play_pattern: f.play_pattern,
        position_tiratore: f.position_tiratore,
        primo_tocco: !!f.primo_tocco,
        porta_sguarnita: !!f.porta_sguarnita,
        tiro_deviato: !!f.tiro_deviato,
        sotto_pressione: !!f.sotto_pressione,
      },
      assist: hasAssist.checked
        ? {
            altezza: f.assist_altezza,
            tecnica: f.assist_tecnica,
            lunghezza: f.assist_lunghezza,
            cross: !!f.assist_cross,
            cutback: !!f.assist_cutback,
            through_ball: !!f.assist_through_ball,
          }
        : null,
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      renderResult(data);
    } catch (err) {
      resultEmpty.textContent = `Error: ${err.message}`;
      resultEmpty.classList.remove("hidden");
      resultBody.classList.add("hidden");
    }
  }

  function renderResult(data) {
    document.getElementById("result-empty").classList.add("hidden");
    const body = document.getElementById("result-body");
    body.classList.remove("hidden");

    document.getElementById("xg-number").textContent = data.xg.toFixed(2);
    document.getElementById("xg-fill").style.width = `${Math.min(100, data.xg * 100)}%`;

    const feat = data.features;
    document.getElementById("stat-distance").textContent = `${feat.distanza.toFixed(1)} yd`;
    document.getElementById("stat-angle").textContent = `${feat.angolo.toFixed(0)}°`;
    document.getElementById("stat-free-angle").textContent = `${feat.angolo_libero.toFixed(0)}°`;
    document.getElementById("stat-defenders").textContent = feat.difensori_cono;
    document.getElementById("stat-min-def").textContent =
      feat.distanza_min_difensore >= 99 ? "none nearby" : `${feat.distanza_min_difensore.toFixed(1)} yd`;
    document.getElementById("stat-gk-cone").textContent = feat.portiere_cono ? "yes" : "no";
  }

  render();
})();
