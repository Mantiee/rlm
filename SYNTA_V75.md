# v75: readable evidence dashboard and execution accounting

Run `tools/update-synta-dashboard.sh COMMIT` on v74. Reload the existing Tailscale HTTPS URL. Only dashboard and gateway services restart; mission, Windows tasks, balances and archived results remain intact.

- `rlm/v100/dashboard_insights.py`: six navigation views, overview distinguishing revenue, tool work, CPU shadow learning and Gemma updates; domain coverage and source reread charts; labelled Brier comparison of the same replay samples. Counts are not earnings. Unknown revenue stays unknown.
- `rlm/v100/dashboard_layout.py`: host navigation applies to existing guest HTML; repairs six table headers; proposals deduplicated by domain/title/mechanism and limited to 12 visible items; eligibility folded; latest five sources. Full receipts and action archive remain accessible. Blue metric curve, green entry triangles, red exit squares, dashed reference line; selected execution shows friction and basis-point units. Tabs, zoom and receipts survive refresh.
- `rlm/v100/dashboard_snapshot.py`: bounded active-goal CPU policy receipt read; no extra model inference or training.
- `rlm/v100/trade_explorer.py`: separate historical spot equity (paid entry friction, no future exit reserve) and liquidation equity; explicit final spot liquidation point and per-leg cost assumptions. Paper spot equity is not invented from incomplete records.
- `tools/update-synta-dashboard.sh`: targeted pinned update; preserves guest layout, goals, weights, ledgers and original reports.
- `tests/test_v100_dashboard_insights.py`, `tests/test_v100_trade_explorer.py`: accounting and interface regression checks.

## What the immediate drop means

Simulator arithmetic uses fee and slippage in basis points divided by 10,000. At a constant price with 50 bps fee and 10 bps slippage, an all-capital purchase leaves spot equity `1 / (1.001 * 1.005)`, about 99.40% of starting capital. Estimated liquidation equity also reserves exit friction, leaving about 98.81%. This reserve is not a second paid entry fee. Later candle-close movement is separate. At final sale it becomes paid exit friction, not another deduction from already-reserved liquidation equity.

A deterministic constant-price replay checks this accounting. It does not certify the operator's unseen prices, specific graph or exchange fees. Existing losses are not rewritten. Current trading fees depend on product/pair/tier and must come from a valid source; historical assumption labels remain explicit.

Flat paper equity with no fills is expected, not a validated strategy. Missing/stale observations cannot prove activity. Research coverage is not successful task completion. No verified payment series is connected, so charts for other domains show measured proposal/source activity and explicitly missing revenue outcomes rather than invented profits. CPU shadow Brier is separate from Gemma optimizer/accepted-update receipts and money. This display update does not create profitable strategies or grant external authority.

## Validation

Full suite: 1,278 passed, 65 skipped. Final focused dashboard/accounting suite: 52 passed; Ruff and shell syntax checks passed. Browser screenshot validation was attempted but the Chromium download failed in this environment. Phone appearance and operator live receipts therefore remain to be checked after install. JavaScript navigation, receipt selection and state-preserving updates were exercised through deterministic Node tests.
