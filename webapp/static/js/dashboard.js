(() => {
  const NS = "http://www.w3.org/2000/svg";
  const DATA = window.DASHBOARD_DATA || [];
  const SERIES_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceMax(value) {
    if (value <= 0) return 1;
    const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
    const steps = [1, 2, 2.5, 5, 10];
    for (const s of steps) {
      const candidate = s * magnitude;
      if (candidate >= value) return candidate;
    }
    return 10 * magnitude;
  }

  // ==================================================================
  // Conversion rate grouped bar chart
  // ==================================================================
  function buildConversionChart() {
    const box = document.getElementById("conversion-chart");
    if (!box) return;

    const seriesDefs = [
      { key: "real_conversion", name: "Real" },
      { key: "model_conversion", name: "Our model" },
      { key: "sb_conversion", name: "StatsBomb" },
    ];

    const W = 640, H = 340;
    const margin = { top: 16, right: 16, bottom: 56, left: 56 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const allValues = DATA.flatMap((d) =>
      seriesDefs.map((s) => d.metrics[s.key]).filter((v) => v !== null && v !== undefined)
    );
    const yMax = niceMax(Math.max(...allValues, 1));

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
    const g = el("g", { transform: `translate(${margin.left},${margin.top})` }, svg);

    // gridlines + y ticks
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const v = (yMax / ticks) * i;
      const y = innerH - (v / yMax) * innerH;
      el("line", { x1: 0, x2: innerW, y1: y, y2: y, class: "chart-grid-line" }, g);
      el(
        "text",
        { x: -8, y: y + 3, class: "chart-axis-label", "text-anchor": "end" },
        g
      ).textContent = `${v.toFixed(0)}%`;
    }

    const groupWidth = innerW / DATA.length;
    const barGap = 3;
    const barWidth = (groupWidth - barGap * (seriesDefs.length + 1)) / seriesDefs.length;

    DATA.forEach((d, gi) => {
      const groupX = gi * groupWidth;
      seriesDefs.forEach((s, si) => {
        const value = d.metrics[s.key];
        const x = groupX + barGap + si * (barWidth + barGap);
        if (value === null || value === undefined) {
          el(
            "text",
            { x: x + barWidth / 2, y: innerH - 8, class: "chart-axis-label", "text-anchor": "middle" },
            g
          ).textContent = "n/a";
          return;
        }
        const barH = (value / yMax) * innerH;
        const rect = el(
          "rect",
          {
            x,
            y: innerH - barH,
            width: barWidth,
            height: Math.max(barH, 0.5),
            rx: 2,
            fill: SERIES_COLORS[si],
          },
          g
        );
        el("title", {}, rect).textContent = `${d.label} · ${s.name}: ${value.toFixed(2)}%`;
      });
      el(
        "text",
        { x: groupX + groupWidth / 2, y: innerH + 24, class: "chart-axis-label", "text-anchor": "middle" },
        g
      ).textContent = d.label;
    });

    box.innerHTML = "";
    box.appendChild(svg);

    const legend = document.getElementById("conversion-legend");
    legend.innerHTML = seriesDefs
      .map((s, i) => `<span><i style="background:${SERIES_COLORS[i]}"></i>${s.name}</span>`)
      .join("");
  }

  buildConversionChart();
})();
