# Synta recovery, v59

This fixes the uploaded dashboard snapshot's demonstrated failures, not a claim
that every strategy is profitable or every candidate must be accepted.

- Aggregate timeout retains only the latest same-run report, with explicit stale
  state and timestamp. New-run optimizer counters never inherit old-run values.
- Verified goal/plans and worker file snapshots remain readable independently.
  Missing ledger/blocker data is unavailable, not an empty ledger or no blockers.
- Dedicated report entry point avoids inference CLI initialization. Goal status
  uses a bounded read-only SQLite snapshot, without competing writer transactions.
  Failure to collect goal metrics does not discard other mission evidence.
- Public tool receipts show actual state/ID/value/process exit code. Desktop
  completion means process execution, with task outcome explicitly unverified.
  Nonzero desktop exit is failed, not completed. The known placeholder/pass
  simulation pattern is refused before queuing; other code still needs validation.
- Upgrade archives UI-dominated income plans and restores observable financial
  research tasks while retaining the operator goal. Model researchers cannot
  replace financial plans with UI-only work. Explicit operator UI tasks stay allowed.
- Financial missions collect up to two public reference series per observer tick,
  rotated across eight separate Coinbase spot-price paths. Uniform controller
  forecasts are precommitted for 300 seconds and resolved using independently
  fetched future prices. They are labelled commissioning baselines, not model
  discoveries, actionable signals or profit. Source outages create failure receipts,
  never fabricated prices or labels. Other goals need their own measurable adapters.
- Model forecasts and observation discovery remain available. The independent
  reference loop supplies task labels even when a model makes invalid tool calls.
  Existing source-disjoint CPU/master admission, retention and independent quality
  gates remain mandatory. Insufficient/division-inadequate labels, unavailable
  workers, fees or failed candidate gates remain explicit legitimate blockers.

No change executes real orders, purchases, paid APIs or account creation. It does
not certify the historical placeholder scalping/arbitrage scripts. Production
execution and measured progress must be checked after installation on Debian.
