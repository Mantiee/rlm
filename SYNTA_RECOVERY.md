# Synta recovery, v60

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

## v60: native startup and concrete income research

- Managed native serving executes llama-server directly, with the original argv
  and library environment. CUDA loading overlaps artifact hashing; the protected
  receipt is bound to the actual child PID/start identity and unchanged files.
  Protected evaluation still waits for hashes and health. Native health startup
  is bounded to at most 180 seconds; failed children are retired. Hash I/O itself
  is visible but not interruptible by that deadline. No second inference CLI is
  launched. This removes a redundant startup path; the user's precise stalled
  stack was not supplied, so a successful V100 start remains a hardware check.
- `mission-status.server_startup` and dashboard show startup stage/PID/error.
  Isolated/remote transports are labelled separately, not claimed as native GPU.
- Operator helper duty percent is preserved by research policy. The old policy
  silently capped 50% at 30%; no hard utilization or board power cap is claimed.
- UI-dominated and known previous commissioning income plans are archived and
  replaced with evidence-backed opportunities/tests across domains. Automatic
  financial plans cannot smuggle UI work in by mentioning income/forecasts.
  The long-term goal and explicit operator authority stay unchanged.
- One bounded, deduplicated RTX research assignment runs every ten minutes
  independently of the V100 training phase when income research is enabled.
  Queue failures are recorded. Allocation alone is not discovery or success.
- `income_opportunities` and `register_income_opportunity` expose fresh
  host-archived source IDs, conservative gross/net estimates before personal
  tax, labor, first-income delay, eligibility, blockers and falsification.
  Zero upfront spend is mandatory. Estimates cannot become actual revenue,
  training labels or accepted master weights. Earlier versions are retained;
  repeated titles/domain update a candidate instead of multiplying it.
- Actual public fetch counters distinguish distinct URLs/content and unchanged
  rereads. Event/source IDs, checksums and acquisition times are separate from
  the shared journal filename. Historical finished jobs are labelled. Old local
  stream fragments are cleared when a new inference starts. Counters cover only
  instrumented calls, not unseen OS/network activity.
- Dashboard adds opportunity/test and source-acquisition panels through the
  existing transactional renderer; no guest script permission is expanded.

No payment ledger is connected. Nothing here guarantees an undiscovered edge,
fast large income, validated strategy, automatic sale or universal continual
learning. Prepared useful work, actual outcomes and payment remain distinct.
