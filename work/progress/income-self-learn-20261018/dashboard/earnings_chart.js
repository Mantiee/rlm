/**
 * Synta EarningsChart — "Zarobki w czasie"
 * Pokazuje łączne przychody: paper trading P&L + gig income, osobno każde źródło.
 * Wymaga: Chart.js (https://cdn.jsdelivr.net/npm/chart.js@4)
 *
 * Użycie:
 *   <canvas id="earningsChart" width="800" height="350"></canvas>
 *   <script src="earnings_chart.js"></script>
 *   SyntaEarningsChart.init('earningsChart', '/api/earnings');
 */

const SyntaEarningsChart = (() => {
  "use strict";

  // -------------------------------------------------------------------------
  // Data model
  // -------------------------------------------------------------------------

  /**
   * Expected API response shape:
   * {
   *   "timestamps": ["2026-10-01T00:00:00Z", ...],
   *   "sources": {
   *     "paper_pnl":  [0.0, 1.2, -0.5, ...],    // cumulative PLN
   *     "gig_income": [0.0, 0.0, 5.0, ...]       // cumulative PLN
   *   }
   * }
   */

  const PALETTE = {
    paper_pnl:  { color: "#4ade80", label: "Paper Trading P&L" },
    gig_income: { color: "#60a5fa", label: "Gig Income" },
    total:      { color: "#f59e0b", label: "Łącznie" },
  };

  const DEFAULT_API = "/api/earnings_history";

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function fmtDate(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString("pl-PL", { day: "2-digit", month: "2-digit" });
    } catch {
      return iso;
    }
  }

  function buildDatasets(data) {
    const { timestamps = [], sources = {} } = data;
    const datasets = [];

    // Individual sources
    for (const [key, values] of Object.entries(sources)) {
      const palette = PALETTE[key] || { color: "#a78bfa", label: key };
      datasets.push({
        label: palette.label,
        data: values,
        borderColor: palette.color,
        backgroundColor: palette.color + "22",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
        borderWidth: 2,
      });
    }

    // Computed total
    if (Object.keys(sources).length > 1) {
      const len = timestamps.length;
      const totals = Array(len).fill(0).map((_, i) =>
        Object.values(sources).reduce((s, arr) => s + (arr[i] || 0), 0)
      );
      datasets.push({
        label: PALETTE.total.label,
        data: totals,
        borderColor: PALETTE.total.color,
        backgroundColor: "transparent",
        fill: false,
        tension: 0.3,
        pointRadius: 2,
        borderWidth: 2.5,
        borderDash: [6, 3],
      });
    }

    return { labels: timestamps.map(fmtDate), datasets };
  }

  // -------------------------------------------------------------------------
  // Mock data (fallback when API unreachable)
  // -------------------------------------------------------------------------

  function mockData() {
    const now = Date.now();
    const n = 30;
    const timestamps = Array.from({ length: n }, (_, i) => {
      const d = new Date(now - (n - 1 - i) * 86400000);
      return d.toISOString().slice(0, 10) + "T00:00:00Z";
    });

    // Simulate small paper P&L drift ± noise
    let pnl = 0, gig = 0;
    const paper_pnl = [], gig_income = [];
    for (let i = 0; i < n; i++) {
      pnl += (Math.random() - 0.48) * 2.5;
      gig += Math.random() < 0.2 ? Math.random() * 8 : 0;
      paper_pnl.push(parseFloat(pnl.toFixed(2)));
      gig_income.push(parseFloat(gig.toFixed(2)));
    }

    return { timestamps, sources: { paper_pnl, gig_income }, is_mock: true };
  }

  // -------------------------------------------------------------------------
  // Chart instance
  // -------------------------------------------------------------------------

  let _chart = null;
  let _canvas = null;
  let _apiUrl = DEFAULT_API;
  let _pollInterval = null;

  function renderChart(data) {
    const { labels, datasets } = buildDatasets(data);
    const isMock = data.is_mock;

    const cfg = {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top", labels: { color: "#e2e8f0", font: { size: 12 } } },
          tooltip: {
            mode: "index",
            intersect: false,
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y >= 0 ? "+" : ""}${ctx.parsed.y.toFixed(2)} PLN`,
            },
          },
          title: {
            display: isMock,
            text: "⚠ Dane demonstracyjne — brak danych runtime",
            color: "#f59e0b",
            font: { size: 12 },
          },
        },
        scales: {
          x: {
            ticks: { color: "#94a3b8", maxRotation: 45 },
            grid: { color: "#1e293b" },
          },
          y: {
            ticks: {
              color: "#94a3b8",
              callback: (v) => (v >= 0 ? "+" : "") + v.toFixed(2) + " PLN",
            },
            grid: { color: "#1e293b" },
          },
        },
        interaction: { mode: "nearest", axis: "x", intersect: false },
      },
    };

    if (_chart) {
      _chart.data = cfg.data;
      _chart.options = cfg.options;
      _chart.update("active");
    } else {
      _chart = new Chart(_canvas, cfg);
    }
  }

  async function fetchAndRender() {
    try {
      const resp = await fetch(_apiUrl, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      renderChart(data);
    } catch {
      renderChart(mockData());
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  function init(canvasIdOrEl, apiUrl, pollMs) {
    _canvas = typeof canvasIdOrEl === "string"
      ? document.getElementById(canvasIdOrEl)
      : canvasIdOrEl;
    if (!_canvas) {
      console.error("[EarningsChart] canvas not found:", canvasIdOrEl);
      return;
    }
    _apiUrl = apiUrl || DEFAULT_API;

    if (typeof Chart === "undefined") {
      console.error("[EarningsChart] Chart.js not loaded. Add: <script src='https://cdn.jsdelivr.net/npm/chart.js@4'></script>");
      return;
    }

    fetchAndRender();

    if (pollMs && pollMs > 0) {
      if (_pollInterval) clearInterval(_pollInterval);
      _pollInterval = setInterval(fetchAndRender, pollMs);
    }
  }

  function destroy() {
    if (_chart) { _chart.destroy(); _chart = null; }
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }
  }

  /** Update chart data directly (e.g. from WebSocket push). */
  function updateData(data) {
    renderChart(data);
  }

  return { init, destroy, updateData, mockData };
})();

if (typeof module !== "undefined") module.exports = SyntaEarningsChart;
