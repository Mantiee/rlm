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

## v61: demonstrated benchmark, research and dashboard faults

- Official benchmark finalization previously overwrote `cases` with a count,
  causing `'int' object is not iterable`. Counts now use `case_count`; all case
  rows, including zero scores, remain intact. A restart resumes the newest
  matching original baseline prefix after checking model, generation, immutable
  snapshot and row identities. It never chooses a run by score or skips gates.
- The operator explicitly stopped Windows RTX after a black-screen hang. This
  update records that preference, cancels its periodic benchmark and sends
  resident model research to the protected accepted V100 already serving. GUI
  observations and dashboard probes do not contact the disabled helper. Windows
  CPU jobs retain their existing two-thread / four-GiB child configuration.
  No Windows GPU process, board clock, power limit or task is started here.
- Bounded primary-terms collection runs independently of the master training
  phase every ten minutes, rotating starting sources across human research,
  usability and explicitly authorized software-bounty programs. It archives an
  actual source even when model analysis is unavailable. A protected V100 then
  prepares a source-linked feasibility dossier with eligibility, labor, costs,
  delay, blockers and a mechanism-specific test. Sources are starting points,
  not recommendations or a closed discovery list. Human participation is not
  automated; no account, application, scan, submission or payment occurs.
- Known microtask/affiliate hypotheses tested with BTC prices are rejected before
  review. Exact repeated notes without a new recorded operation are marked as
  repeated advisory. These conservative lexical/receipt checks are not semantic
  proofs. Old findings are annotated with a backup; original texts are retained.
- Token deltas no longer displace tool calls/results/errors from the action
  timeline. Historical actor cards are collapsed; queued/running job cards put
  old attempts in a separate disclosure, never the current result. New requests
  clear their actor's old error, evidence and token usage. An unchanged journal
  filename is not treated as shared source evidence.
- Updates preserve inner reader scroll and anchor the visible expanded detail
  when new content is inserted above it. Tests execute the renderer in a Node
  DOM fixture; actual browser acceptance remains a hardware check.
- Existing crypto prices are refreshed before fee parsing. A fee-layout failure
  archives the exact source and blocker instead of suppressing price observation.
  Product boundaries accept case changes. Unknown or ambiguous fees still block
  paper fills, and published fee assumptions remain distinct from account rights.

Inspect `research/income-work/status.json`, its source-linked dossiers,
`research/research-quality/latest.json`, public benchmark resume receipts and
`research/paper/fee-source-status.json` for measured execution. Feasibility
estimates are neither real income nor training labels. Actual master training
and promotion still need independently verified outcomes and all quality gates.

## v62: recover the supplied startup race without discarding evidence

The uploaded v61 report shows an income analysis hitting `/proc/515286/stat`
after that old server exited. Source acquisition had succeeded. This is a
serving readiness failure, not a failed income experiment.

- `income_work.accepted_master` waits for current-run startup readiness and
  requires the protected process/model receipt before generation. Missing process,
  receipt or endpoint becomes an explicit deferral. An identity mismatch remains
  a rejection. No candidate or disabled Windows RTX is substituted.
- `income_work.execute` resumes a pending dossier under the same goal for up to
  24 hours without refetching or counting its source twice. Startup retries use
  30-120-second backoff for six attempts, then the normal ten-minute period.
  Analysis attempts are separately limited to three; failures stay archived.
  Goal changes during generation block registration under the new goal.
- `drones.finish` changes only the next due time for these income retries;
  the configured periodic interval is retained. The waiting job releases its
  worker slot rather than sleeping inside it.
- `research_policy.apply` cannot expand the bounded resident client's 1024-token
  ceiling or enable its thinking mode. Ordinary research policy remains separate.
- The research audit rejects actual-labor income comparisons against simulated
  paper returns, recognizes the supplied Prolific/microtask variants, and flags
  the same advisory test under paraphrased hypotheses. Lexical checking is not
  semantic proof. Researchers receive the last four rejected tests and their
  reasons so they can revise them. Historical migration cannot replace the
  latest live audit.
- New worker notes carry the originating mission run. Prior-run and migrated
  historical notes remain archived and are excluded from the live ideas list.
- Cached official baselines publish finished progress for the current run.
  Complete current-run reports take precedence over stale global counters;
  previous-run counters are presented as history, never current computation.
- `mission-report` includes owned CPU dispatch gates and the archived fee-source
  blocker. Dispatch is goal-attributed and rechecks promptly after a goal change.
  No training is forced with unverified labels or inadequate independent splits.

This release does not certify fees unavailable from the actual retrieved page,
income, demand, novelty, CUDA utilization or browser behavior on the operator's
machines. The existing Windows CPU limit and disabled RTX preference are retained.

## v63 rebuilt recovery

The interrupted publication did not reach the remote branch. The local workspace
was subsequently restored from the exact public v62 tree before these fixes were
rebuilt. Old test counts are not acceptance evidence for this rebuilt release.

- Financial read-only questions use current host reports without waiting for a
  model, modifying plans or inventing ROI. Historical losses, scoring metrics,
  zero forward paper fills, hypotheses and verified actual income are separate.
  An uninitialized paper ledger reports unknown fills rather than fabricated zero.
- Explicit test-now/history requests execute a transparent BTC hourly commissioning
  comparison with one shared source acquisition, two predetermined rules and
  documented assumed costs. Cash, buy-and-hold, doubled costs and chronological
  windows reject weak candidates. This is not broad autonomous strategy discovery.
  A comparison is reused for one hour with report digest checks; source failures
  have their own blocked receipt and no invented ROI. Actual income remains unknown.
- Hypothetical-capital research is an archived operator policy, distinct from
  spending/account/outreach/order authorization. Existing explicit goals remain
  literal. Research jobs are goal-bound, deduplicated and bounded. Windows RTX
  remains disabled and the bounded CPU mailbox is retained.
- Native serving readiness waits have durable SQLite timing and a limit of 600
  seconds or 20 checks. Other requests can complete during the wait. Actual model
  generation timeouts stay terminal. A plan, inspection or queued job cannot prove
  implementation. Receipts distinguish failures, reads, queues and publication.
- Full admitted drone assignments replace the old 300-character SQL preview.
  Backend finish_reason and measured tokens are separate; length marks an
  incomplete response. Archive previews remain bounded and labelled.
- Internal generation instructions and built-in UI labels use English. Chat follows
  the current operator language. Archived evidence and literal goals stay verbatim.
- Layout publication cannot reload the reading page automatically. A manual button
  loads a new layout when the reader is ready. Existing keyed DOM and disclosure
  preservation remain; viewport anchoring is expanded. Reload once after upgrade
  to replace already-loaded old JavaScript.
- The separate n-gram memory pilot is documented in SYNTA_NGRAM_MEMORY.md. It is
  not a modified Gemma GGUF and cannot activate without measured gates.

Runtime evidence is stored under research/market-research, research/backtests,
research/income-policy-history and research/architecture-candidates. Code tests do
not assert operator-browser, Windows/CUDA runtime acceptance or actual income.

## v64: delivered replies, independent financial visibility and remote access

- Financial questions including the operator's `zarobiles` wording read measured
  historical/ledger facts directly. They do not substitute a no-real-orders
  explanation for an unprofitable simulation. Explicit `assume capital` commands
  update hypothetical-capital policy without authorizing spending. The upgrade
  applies the current operator's explicit paper-capital authorization.
- Read-only financial/status questions can bypass the GPU work queue. Terminal
  chat monitors requests still pending after its wait window, delivers eventual
  saved results and errors, and offers `/results` and `/pending`. Real commands
  retain their bounded queue and tool receipts.
- Completed research cycles, last entered loop stage, admitted data, optimizer
  steps and accepted weights are separate. The loop records preparation,
  research, memory work, data admission, waits and candidate trials. A checkpoint
  no longer runs a synchronous full financial audit inside the controller.
- Dashboard inference/readiness HTTP polling uses already-collected events;
  browser requests have eight-second deadlines and one in-flight request per
  endpoint. The action archive uses live-document delegated events and a retained
  DOM container, preserving expanded records across updates and on errors.
- Stored historical tests load independently of the aggregate report. A separate
  bounded read checks the current paper state and its last event, exposing equity,
  positions, quote/fee blockers and original timestamps. It is visibly distinct
  from a full-history audit; it does not invent fill counts. Saved paper reports
  retain their own period/time rather than pretending to be live or current-run
  training evidence. Negative results remain negative.
- `install-synta-remote.sh` prepares owner-only Tailscale HTTPS dashboard/chat and
  existing SSH access, without port forwarding. It requires actual operator
  login/HTTPS authorization and the same account on mobile/laptop. See
  SYNTA_REMOTE_ACCESS.md for boundaries and shutdown. Windows RTX stays disabled;
  the existing conservative Windows CPU worker is unchanged.

The finite code tests do not prove a profitable strategy, continuous optimizer
updates, a successful phone login, or runtime performance on the operator's GPUs.

## v63: independent live refresh and conservative Windows CPU

- Live file snapshots refresh every five seconds without running `mission-report`
  in that loop. A separate serial collector performs the aggregate audit with its
  existing 20-second bound. Delayed audits have their own state and timestamp;
  GPU/activity/mission reads continue. Guest SSH layout polling is a third loop,
  so its timeout cannot stop live data. Historical aggregate values stay scoped
  to the same run and visibly stale; unavailable values are never invented.
- Financial questions in chat read stored same-run evidence and its original
  timestamp without launching another aggregate audit. Missing fills and
  blockers remain unknown. Old-run counters are discarded.
- Paper and competition reports open existing SQLite ledgers read-only with a
  one-second lock wait. They no longer request WAL/schema writer locks just to
  display a report. Paper hash-chain validation is retained.
- `install-synta-low-load.ps1` replaces only the isolated owned CPU worker, retains
  its authenticated mailbox, sets autostart, and opens a visible log console.
  It retires the isolated RTX helper without changing board power, clocks,
  unrelated processes, firewall rules or desktop applications.
- Windows children get idle priority and at most two CPU affinity slots, plus
  two-thread BLAS/Torch limits. One job runs at a time. The worker checks host
  CPU, free RAM and foreground browser/media/game every two seconds and kills
  its owned experiment when CPU exceeds 40%, free RAM falls below 6 GiB, or
  foreground activity needs the computer. Child RSS is sampled against 4 GiB;
  this is a monitored cutoff, not a Windows kernel allocation quota.
- The earlier code noticed CPU pressure but ignored it during running jobs.
  This is fixed and tested with an owned child cancellation receipt.

Per-process duty cycles cannot bound a 3090's instantaneous board power. After
the reported black-screen hang, this preset uses Windows CPU and leaves RTX off;
it does not promise to diagnose hardware or prevent all crashes. The V100
remains the main model/GPU. N-gram memory runs as a bounded experimental candidate
with paired host-scored ablation and existing independent promotion gates.
