# Synta v76: strategy evidence and dashboard recovery

The v75 package installed successfully in the supplied log, but its immediate HTTP check ran before port 8765 answered. v76 waits for the expected renderer for up to 60 seconds and prints service diagnostics on failure. This verifies local HTTP readiness only.

## Changes

- The trade explorer resets its caption when switching series. Recorded executions no longer inherit an empty-paper caption.
- Historical series offer a zero-cost diagnostic using the same selected rule, lookback, signals and execution times. It separates signal returns from cost drag without changing archived reports or claiming that cost-free execution is available.
- Execution receipts record the previous completed bar and signal used for the next opening fill. Spot equity, liquidation value and subsequent price movement remain distinct.
- Fixed momentum/mean-reversion commissioning runs once per goal rather than generating another losing baseline every hour. The accepted master retains access to explicit backtest and sandbox tools to choose different experiments.
- Goal work receives the actual previous hypothesis, test and tool states, plus recorded commissioning failures, instead of only a filename. Model-led decisions remain subject to evidence and resource checks.
- Registering a proposal is not labeled an executed test. Forecast commitments, process exits and historical test receipts are reported separately.
- Other-income cards show one representative per domain with the remaining receipts retained. This is presentation grouping, not proof that all ideas in a domain are identical.
- Learning compares the shadow policy with the strongest recorded reference. Source-disjoint CPU training gates remain intact; blockers include group counts.
- Known legacy Polish dashboard headings and descriptions render in English.

## What the screenshots establish

The selected historical example shows net loss around 8.21% and modeled costs around 0.0801 normalized capital units. This suggests costs are material; the new diagnostic measures their return impact using the archived report. With assumed 50 bps fees and 10 bps slippage per leg, entry spot friction is about 0.6%, not 3%. Later candle movement and estimated exit costs must be inspected separately.

The shadow replay Brier score of 0.5819 is worse than the prior-frequency reference of 0.5780. There is no evidence of accepted production weight updates in these screenshots. A completed research loop does not establish profitable behavior.

The RTX endpoint is still unreachable in the supplied Debian receipts. This release does not establish a repaired Windows connection, external income, phone routing or a profitable strategy. Windows policies and balances are preserved.

## Install

Use `tools/update-synta-v76.sh` pinned to this release commit. It supports installed v74-v76, preserves the current profile, stops the mission once, installs the package, starts the supervisor and waits for the dashboard. It retains goals, weights, ledgers, archived reports and external-action policy. Reload the browser once.

For a dashboard-only update, `tools/update-synta-dashboard.sh` remains available; it does not restart the mission and therefore does not activate the controller changes above.

## Changed code

`rlm/v100/{backtesting,trade_explorer,dashboard_layout,dashboard_insights,dashboard_health,market_research,income_work,research_contract,goal_compute}.py`, `tools/update-synta-v76.sh`, `tools/update-synta-dashboard.sh`, version metadata and regression tests. No new dependency.
