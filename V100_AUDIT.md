## v100.45 current-request resource contract

A reported v44 resource request received an unrelated trading/dashboard answer,
changed short/mid plans and queued a placeholder VM script. Those receipts did not
prove Windows allocation or useful execution. This revision fixes that path:

- Resource-only instructions do not replay unrelated dashboard conversation history.
  Their tool allowlist exposes actual resource status, owned compute proposals and
  bounded drone scheduling, without HTML edits or GUI/desktop jobs.
- Unrelated plan/sandbox actions are rejected before application. The displayed
  answer is composed from host resource evidence and executed tool receipts, not
  from an unverified model success declaration. Tool receipts are retained in chat.
- `compute_resources` distinguishes pinned RTX model load, CPU worker heartbeat,
  local worker RAM/disk budgets and actual job states. Windows RAM/VRAM is not pooled
  into the Debian address space. Worker readiness alone is not execution.
- The stopped upgrade archives completed misrouted resource requests, restores only
  still-current short/mid versions they replaced, cancels only matching still-queued
  desktop scripts and requeues the original resource request. Later user plans and
  the operator long-term goal are preserved. Already executed scripts are not
  claimed to be undone.
- Expired fee profiles retained solely as accounting history are separate from
  expired profiles still referenced by active instruments. Historical records stay
  intact; currently expired instrument fees still block paper execution.

This correction does not certify that a model will choose a useful CPU experiment,
that its job passes independent validation, or that distributing work is faster.
Those outcomes require actual job receipts and measurements.

## v100.44 integration and observed limits

This revision connects the existing capabilities; it does not certify live hardware,
accepted model updates, profit, or universal prevention of forgetting.

- The host supplies a trusted live renderer with separate master, V100, Debian CPU,
  resident drone, RTX helper and Windows CPU cards, actual assignments, operator
  long/mid/short plans and a bounded event timeline. Board utilization is not per-model
  utilization. Stale worker/chat receipts are explicitly marked. RTX cooldown and
  loaded model are observed separately from board utilization, which remains unknown.
- `dashboard_layout.py` is the shared HTML/CSS contract. `read_dashboard`,
  `write_dashboard`, and `dashboard_status` expose bounded, backed-up guest edits.
  The chat host verifies the exact guest/publication hash and overrides unsupported
  success claims. A write is not publication. Trusted refreshing stays outside the
  model-editable HTML. Old valid layouts gain live cards automatically; the upgrade
  backs up and repairs an invalid script-containing layout.
- The installed dashboard tool and package now come from the same pinned revision.
  The upgrade preserves goals, accepted weights, checkpoints, guest disk and valid
  custom layouts. It does not repeat training calibration.
- `background_observer.py` refreshes registered paper quotes/primary fee evidence
  independently of GPU evaluation and GUI readiness, with visible errors and backoff.
  A failed goal-observation endpoint does not suppress quote refresh. It retains
  instrument selection, fee provenance and paper risk checks; it submits no orders.
- The private VM lifecycle owns and restarts the loopback-only SSH reverse tunnel
  to the existing public HTTP broker. Existing guest APT/browser proxy configuration
  is updated. Only the tunnel child created by this lifecycle is stopped. A manual
  old tunnel may already hold its port; this is logged rather than killed.
- Goal-linked development panels now cover alternate foundation and scratch master
  trials as well as LoRA updates. Where enough independent resolved observations
  exist, parent/candidate weights must pass the exact panel and beat its best constant
  predictor. Activation rechecks the proof and the current operator goal. Insufficient
  labels mean no goal-specific improvement is certified. Protected/fresh/public gates
  remain mandatory where configured; development panels are not untouched audits.
- Master and resident research agents can propose owned CPU compute trials through
  their actual tool allowlists. Idle workers are capacity, not proof of execution.
  Their constrained proof-domain trials are experimental; a useful goal-linked
  hypothesis still needs matching independent evidence before master promotion.
- Internal income/self-upgrade research assignments fit the enforced 400-character
  admission limit. The prior over-budget assignment no longer aborts every round.

Operational acceptance remains on the user's devices: guest networking/GUI after a
restart, VRAM fit, model numerical health, real worker execution and accepted update
counters. The optional resource/kernel tests requiring unavailable local process or
GPU facilities are not represented as passed hardware checks.

The requested always-on free managed Colab worker farm is not implemented. Google's
current FAQ explicitly disallows distributed workers on free managed runtimes.
The existing interactive notebook and authenticated operator-owned compute paths
remain available. No account/session rotation or quota bypass is added.

Sources checked 2026-10-09:
- https://research.google.com/colaboratory/faq.html
- https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html
- Retention literature and implemented experiment limits: RETENTION_RESEARCH.md.

# V100 mission audit v100.39

The optional `tools/start-v100-dashboard.sh` installs a separate read-only LAN
viewer without updating or stopping the mission. It binds explicitly to
192.168.0.68:8765, exports bounded status/log snapshots and allowlisted paper
report assets, and has no command or training endpoint. Its data collection is
resource-limited, and old-run reports are not attributed to a new mission.
Serving a dashboard on the operator host does not relax the private VM's network
or filesystem isolation. Reachability from another LAN device still needs an
on-device check.

The master can edit `/workspace/dashboard/index.html` through the existing chat
`sandbox` action. The viewer pulls only this fixed guest file, validates bounded
HTML/CSS and component IDs, preserves prior versions and retains the last valid
layout on failure. The data renderer remains host-owned and CSP permits only its
exact script hash. The user preferences carry this presentation permission
without changing the long-term goal or existing directives. Status is saved in
`research/dashboard/layout-status.json`. Browser polling refreshes data and
reloads a changed validated layout automatically. This is presentation editing,
not permission to change financial evidence or run arbitrary host commands.

This inventory separates source implementation from configuration, execution and
demonstrated improvement. Passing controller tests does not certify GPU stability,
profitable trading or a successful model promotion. It is not a declaration that
every requested feature is complete or optimal.

## Current requirements inventory

| Requirement | Source mechanism | Limit or evidence still needed |
|---|---|---|
| Autonomous self-upgrade loop | `mission.py`, `continuous.py`, `supervisor.py` | Accepted updates and numerical health, not loop count alone |
| Operator-owned long-term objective | `goals.py`, `chat_goals.py`, `planning.py` | Only a clear current operator request can change it; upgrades preserve old goals |
| Short/mid-term plans from ordinary chat | `mission_chat.py`, `planning.py` | Versioned host receipts; a written plan does not mean its experiment ran |
| Chat while research continues | Persistent queue, accepted native master on GPU or RAM-admitted CPU, explicit RTX delegate | CPU copy uses the same accepted weights, is slower and may be deferred by host RAM pressure |
| Factual status in chat | `mission_evidence.py`, supplied automatically and as a research tool | Report paths, successful optimizer counters and accepted versions are distinct; missing counters mean unknown |
| Long context, Flash Attention, KV | Native configuration and fit checks | Device execution and long-input accuracy; free VRAM alone does not establish speed |
| Adaptive thinking/output | Validated `research_policy.py` budgets | Research only; fixed evaluations retain comparable settings |
| Adaptive batching/speed | MTP/calibration/performance measurements | Master batch is a stopped-server benchmark proposal, not unrestricted live reconfiguration |
| Parallel master/helper research | `paper_agents.py`, resident drones | One RTX request and bounded CPU jobs; concurrent master training/inference models on the single V100 are not enabled |
| Independent A/B evolution/crossbreeding | `lineages.py`, `competition.py`, `crossbreeding.py` | Accepted separate ancestors require passing updates; initial branches share the base |
| Architecture activation/rollback | `architecture_promotion.py`, foundation/custom trials | Fixed/public/fresh gates and actual inference probe; no hardware promotion inferred |
| New independent training/audit tasks | `curriculum.py`, split ledgers, `fresh_audit.py` | Fresh calculations, sequences and typed extraction; finite synthetic coverage is not all-domain robustness |
| Official AI benchmark comparisons | Pinned LiveBench tasks/scorers, same-question published references | A selected panel is not the full leaderboard; provider settings differ; scorer errors block certification |
| Source/CPU/RTX drones | Bounded persistent job queue | Configured and running jobs differ; limits and game pauses can defer work |
| Full tool evidence and compact context | SHA256 archives, paged reads and compact capability routing | Previews are incomplete; oversized original input can still fail explicitly |
| Original memory and summaries | Archive, hierarchical summaries, CPU lexical/embedding retrieval | Retrieval/provenance quality; recursive model transcripts are not facts |
| Read/copy own code, sandbox files | `self_code.py`, private namespace and guest | Host controller protected; arbitrary shell and modified copies operate in private guest |
| Linux GUI/browser/tools/vision | Debian VM, public HTTP(S) broker, RTX vision | Guest/device verification; GUI is not required for native training or calculators |
| Resource/error recovery and uptime | Bounded resources, deadlines, restart backoff, rollback | Best-effort recovery, not unconditional 24/7 uptime or stability |
| Restrict helper only on Windows | Isolated runtime, smaller batches/context, pacing, LoL pause | No hard per-process GPU-utilization/power cap; blackscreen cause unproven |
| Use Windows CPU/RAM/owned machines | Authenticated shared-folder workers, leases and tensor validation | Worker launcher must run there; mailbox configuration alone is insufficient |
| Additional Colab compute | Bounded interactive notebook and safe result import | No autonomous managed-free worker farm or cookie/account quota evasion |
| Free external model consultation | Documented services, zero-price catalog, guest browser, explicit key routing | Existing session/key and allowance needed; stop or hand off at authentication/payment/limit walls |
| Research beyond fixed assets | Catalog discovery, model-selected spot and configurable feeds | Equities/sports require verified fees, rights, liquidity and settlement semantics |
| Historical/event-timed backtests | Chronological splits, events, cash/buy-hold and cost stress | Historical availability/leakage; past sample performance does not prove future income |
| Learn from profitable/losing outcomes | Audited paper accounting, causal DPO, shadow reward head | Resolved valid outcomes required; not unbiased profit-maximizing policy RL |
| Net results and local alerts | Audited paper reports and labeled rejected/accepted proposals | No real orders or demonstrated income edge inferred from signals |

## v100.39 additions

`chat_backend.py` can serve the identical accepted native GGUF on a separate
CPU-only loopback endpoint while the V100 performs an exclusive training or
candidate-evaluation job. It never adopts the model found on the candidate endpoint.
The model digest and native launch receipt are checked, including every cached
reuse. Two low-priority threads, an 8192-token context, 1024 output tokens and
no draft GPU model bound this fallback. Admission requires at least 16 GiB
available host RAM, or a larger weight-size-dependent estimate. An owned-process
watchdog retires it below 4 GiB available RAM, above its RSS budget, or on mission
shutdown. Idle copies close after 180 seconds. A labelled RTX delegate remains
the fallback if CPU admission fails; otherwise requests stay queued during outages.
This adds concurrency through CPU serving, not simultaneous training/inference on
the single GPU or a guarantee of low chat latency. Actual device fit/speed remains
an on-device check. `accepted-cpu-chat-status.json` and chat responder receipts
identify the backend and accepted model digest.

Stopped campaign setup refreshes the guest's readonly code ISO to the newly
pinned own-source revision instead of retaining the original v32 code indefinitely.
It preserves the writable guest disk, refuses a live-guest replacement, checks the
ISO digest and leaves the old ISO untouched if building the replacement fails.
GUI health now reports separate cloud-init, workspace, source, service and display
checks, bounded guest installation/service logs, and loaded versus expected source
revision. A stale source mount cannot report full readiness. This diagnoses startup
failures; it does not certify that this user's VM has completed installation.

Campaign upgrades preserve the previous mission input profile even before its
first completed learning cycle. Validated learning checkpoints still take priority.
The user-owned goals and prior accepted weights are not reset by this addition.

## v100.38 additions

Campaign setup automatically creates a separate public benchmark snapshot when
the installed official-grading adapter changed. A baseline from a different
snapshot is retained as historical evidence, while accepted weights must be
evaluated on the new snapshot before comparison/promotion. Existing input hashes
are checked first; corrupted benchmark inputs are not silently replaced.

`mission_evidence` reads bounded current-run health records and aggregate baseline
summaries. Hidden audit answers never enter this tool. Attempted/global steps are
not successful optimizer updates; old benchmark errors are labeled historical.
Ordinary chat receives evidence even when it does not select the tool. The prompt
rejects invented scorer causes, a GUI prerequisite for unrelated work, and claims
that arithmetic practice is necessary for income.

`read_tool_result` reads complete archived outputs in UTF-8-safe pages, checking
SHA256 on every page. Arbitrary file paths and symlinks are rejected. A preview is
still explicitly incomplete evidence. Original task/resource limits remain enforced.

Fresh training and post-freeze audits add independently checked sequence operations
and typed JSON extraction to the existing three calculation domains. Generation
never reads benchmark answers. Audits stay single-use, drawn after candidate
identities are frozen and reserved against training. Owned-worker examples are
tested to fit without truncation. This adds coverage, not an income policy or
universal no-forgetting proof.

Earlier device logs established native inference, a small CUDA backward probe,
private code namespaces and RTX identity/loading. They did not establish complete
GUI readiness, master architecture promotion, latest-release GPU training or
repeatable net profit. Those claims require fresh device reports.

The official [NVIDIA MPS documentation](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
limits MPS support to Linux/QNX; it is not a Windows per-helper utilization cap.
The [Colab FAQ](https://research.google.com/colaboratory/faq.html) distinguishes
ordinary interactive work from distributed workers on free managed runtimes and
disallows multiple accounts to bypass limits. Owned workers and interactive
notebooks remain the implemented routes; no quota-evasion mechanism is added.

## Historical release notes and earlier scope

The sections below record earlier releases. Their test counts and resource defaults
are historical, not evidence that every current feature is active.

v100.24 normalizes Transformers 5 chat-tokenization mappings and handles Gemma's
exact empty-thought inference suffix when encoding answer-only training records.
The full token prefix must still match; prompts are never supervised and records
are never silently truncated. Tokenization is checked before large weight hashes,
and training logs explicitly announce tokenization, hashing, loading, validation
and optimizer stages. Child Python processes use unbuffered output.

v100.23 fixes Kraken's geolocated Polish fee page: the parser recognizes the
published English and Polish spot sections, decimal separators and optional
futures-volume column. It still rejects missing or duplicate lowest-volume rows
and excludes cross-platform and maker-rebate tables. No fees are hardcoded.

This is an implementation and evidence audit, not a claim of best possible speed,
a money-making policy or zero catastrophic forgetting. Install only in the existing
`ai-v100/venvs/v100-continual` environment. The original model, V100 lab, desktop
Ollama and system GPU power settings remain separate.

## Requirements and actual scope

| Request | Implemented mechanism | Evidence still required |
|---|---|---|
| Work in a loop | Owned background mission, source research, paper observer, candidate learning cycles | Completed cycles and accepted updates in `mission-report` |
| Talk while it works | Persistent `chat` / `chat-status` queue serviced by current V100 server | On-device chat responsiveness; training queues messages |
| Obey new directions | User directives in next R&D context; host receipts for thinking/token changes and local paper alerts | Inspect receipts; free text is guidance, not a guaranteed executable automation |
| A/B evolution | Separate accepted adapter parents, immutable quality evidence, replay/KL and resumable optimizer checkpoints | Accepted branch improvements on unseen tasks |
| Crossbreeding | One bounded 0.5 blend of compatible accepted adapters; export and independent regression gate | Child must improve over both parent scores; originals remain available |
| Persistent memory | Original source SQLite archive, hierarchical summaries, CPU multilingual embeddings and lexical retrieval | Encoder preparation and full-context retrieval quality |
| Fast master/helper | Independent V100/RTX research overlap, fast JSON tool routing, adaptive R&D thinking and token limits | End-to-end hardware measurements, not short token rate alone |
| Source drones | Two concurrent bounded public-page fetchers, at most eight URLs per call | Valid sources; excerpts are not certified training labels |
| Tiny submodels | Existing isolated CPU architecture pilots and tests | Tiny models do not automatically replace pretrained Gemma |
| Own code changes | Pinned own source, exact candidate edits, namespace checks and independent training/quality testing | Working Debian namespaces and passing trusted checks; no live controller replacement |
| Training speed | Optional exclusive NF4/FP16 pilots, finite losses and observed allocated-VRAM checks | Actual V100 pilot times and representative sequence lengths |
| Speculative decoding | Optional official assistant MTP sweep 2/4/8/16, measured speed and fixed-suite quality gates | Requires the device sweep; no speculative speedup claimed before it passes |
| Long context | Requested 131072-token V100 context, Flash Attention on, native KV cache; fit fallback | Native logs must confirm actual execution; long-context accuracy not established |
| RTX crashes | Smaller 32768 context/batch16, 30% request active-time pacing, guardian stops only isolated helper during LoL | Crash cause unproven; this is not a hard GPU/power cap or stability guarantee |
| Paper outcome learning | Host-audited realized trade accounting, including losses, can enter verified SFT | This is not profit-policy RL; no evidence of a repeatable income edge yet |
| No fixed asset list | Kraken USD spot catalog discovery and model-selected registration, maximum eight in this resource budget; other income research allowed | Sports/equities need dedicated verified feeds and settlement simulators |
| Backtests and events | Chronological walk-forward windows, cost stress, cash/buy-hold comparisons, optional timestamped events, duplicate caching | Historical availability, market liquidity and future independent paper outcomes |
| Costs | Primary spot fee/rule snapshots, quantity/notional minimums, documented FX/slippage assumptions | Personal tax and real account/regional entitlements are not certified |
| Autosave | Best candidate adapter plus optimizer/trainer/RNG checkpoints and immutable accepted versions | Check completed checkpoint markers and serving selection receipts |
| Colab | Short interactive tiny-model notebook with heldout loss and local checkpoint download | Manual user session; no distributed free workers or quota resets |

Research explores other income opportunities without a fixed financial universe.
BTC/ETH are bootstrap examples, not forced investments. Models may hold cash or reject
all proposals. Adding a public pair never bypasses fee, risk, freshness or future-quote
execution checks. No real-money orders, paid APIs, sales, accounts or spending are
executed automatically.

Finite regression tests detect some forgetting, not all possible forgetting. Frozen
base weights and prior adapters preserve recoverable versions, while an updated
adapter can still regress on untested tasks. Retaining old files is not proof that
new behavior remembers everything.

## Research decisions

The [RLM paper](https://arxiv.org/abs/2512.24601) motivates decomposing external
context and reading original snippets rather than filling every request with the
entire archive. This release uses bounded source work, original-source retrieval
and hierarchical summaries; it does not train a new recursive transformer from scratch.

[Replay to Remember](https://arxiv.org/abs/2504.17780) reports partial retention
benefits from replay with LoRA, not a universal zero-forgetting guarantee. Replay,
KL and versioned quality gates are combined here; accepted parents remain recoverable.

[Ollama thinking](https://docs.ollama.com/capabilities/thinking) supports a per-request
switch. [Its concurrency documentation](https://docs.ollama.com/faq) states that memory
scales with parallel requests and context. Given repeated Windows blackscreens, the
helper keeps one slot and pauses for LoL rather than increasing concurrency there.
V100 batch changes are proposals until a stopped-server measurement validates them.

[Kraken AssetPairs](https://docs.kraken.com/api-reference/market-data/get-tradable-asset-pairs)
provides actual market rules; [fees](https://www.kraken.com/features/fee-schedule)
are re-read and archived. Simulated trading volume never earns fictitious discounts.
The [Colab FAQ](https://research.google.com/colaboratory/faq.html) disallows free
runtime distributed workers and account switching to evade limits. The provided
notebook is an ordinary interactive, optional pilot.

## Recovery and use

1. On Windows, download the pinned release `tools/start-rtx3090-helper.ps1`, run
   `-Stop`, then `-Context 32768 -BatchTokens 16 -ActiveTimePercent 30 -DebugLogs`
   in administrator PowerShell **before opening LoL**. The runtime window is hidden.
   Guardian pauses its own server while the game runs and warms it after a cooldown.
   During transport outages, serving research uses the already loaded V100 master;
   it retries the external helper next cycle and records the fallback explicitly.
   It does not stop the desktop Ollama or change global clocks/power.
2. On Debian, run the pinned release `tools/upgrade-v100-mission.sh RELEASE_SHA`.
   `--optimize --calibrate` adds exclusive MTP and training measurements before starting
   the mission. Those optional stages can take a long time; their failures are recorded
   as deferred. Required helper/source setup failures stop preparation.
3. `~/ai-v100/bin/v100-continual mission-watch` follows model/tool events.
   `mission-report` shows accepted updates, losses/metrics, actual paper P&L and blockers.
   `mission-audit` distinguishes configured mechanisms from observed evidence.
4. `~/ai-v100/bin/v100-continual chat` opens the chat in another PuTTY session.
   Try: “Wyłącz thinking helpera i użyj 1024 tokenów. Szukaj także usług bez wpłaty.”
   Then: “Włącz lokalne alerty paper kupna/sprzedaży.” Check the separate host receipts.
   `/exit` closes chat only. Requests survive terminal closure; inspect an ID with
   `chat-status ID`. A single V100 slot shares its request queue; chat may wait.
5. Local signals: `research/alerts/signals.jsonl`. They label rejected signals explicitly
   and are model paper proposals, not certified opportunities. Arbitrary textual alert
   conditions guide R&D but are not hard guarantees of immediate detection.
6. Windows logs: `ai-v100-helper/logs/guardian.jsonl`, `server.stderr.log` and
   `server.stdout.log`. Debian preparation logs and stages persist under `research`.
   Closing PuTTY does not stop the owned background mission; use `mission-stop`.

## Verification limits

The deterministic Python suite and Ruff run in the development workspace.
Release validation: 365 tests passed, 12 optional ML tests skipped; Ruff, Python 3.11
grammar, notebook-cell syntax, shell syntax and diff checks passed. GPU training,
MTP conversion, full 131072-token tests, Windows PowerShell execution, guardian/game
interaction and the Colab notebook runtime need device verification. The notebook is
syntax checked. This workspace cannot reach either LAN computer and has no GPU runtime.
No new accepted weight update or income result should be inferred from installation.

## v100.35 process lifecycle recovery

Exited Linux workers (including zombies with retained start ticks) no longer count
as live missions or protected servers. This lets the supervisor restart failed
children and lets the updater proceed without signalling an exited process.
The updater includes this check before installing, so recovery also works with
older installed packages. Live command/session ownership mismatches still abort
without killing the process or replacing packages. Checkpoints and failure reports
are retained. A successful update is not evidence of a successful weight promotion.

## v100.36 native server restart ports

Native and MTP availability probes now use SO_REUSEADDR and actually listen before
closing the probe. Linux TIME_WAIT after a completed acceptance request no longer
looks like a live server occupying the port. An existing loopback or wildcard
listener still blocks launch, with the port included in the error. No listener is
killed or reused, and no second model is started against an occupied endpoint.

## v100.37 official grader dispatch and research context

Instruction-following questions before 2025-11-25 now use the same legacy official
IFEval route as the pinned LiveBench CLI, rather than being sent to IFBench's
incompatible instruction registry. New IFBench questions retain their official
route, and grader failures still block certification instead of becoming scores.
Changed adapter hashes require a new frozen benchmark snapshot.

When a full JSON capability catalog cannot fit the client's measured or conservative
context budget, a short validated capability selector precedes the selected tool's
argument schema. It can also decline tools with a real concise answer. Task scopes,
argument validation, token limits and the original input remain enforced.
Oversized tool results are retained in SHA-256-addressed files and activity traces;
model-visible previews explicitly mark truncation and do not certify missing facts.
# Retention experiments v100.40

`RETENTION_RESEARCH.md` distinguishes existing replay/KL/fresh regression gates
from new bounded LoRA L2, empirical diagonal Fisher EWC, experimental delta-A
orthogonality, A-GEM projection and exact-initial-delta rank expansion. Model
planning can compare them; historical methods require previous verified training
references, never audit labels. Anchors/Fisher are pinned across resume, and
retention/capacity artifacts appear in current-run training evidence. Existing
ancestor/public/one-use audit gates and original accepted checkpoints remain.

`capacity_growth.py` implements small isolated frozen residual-column and appended
embedding prototypes. Their old routes/IDs stay unchanged in tensor tests; this
does not implement live Gemma layer/vocabulary surgery or prove retention on
unseen tasks. Full NF4/FP16 training and grown-adapter native export require V100
acceptance. The dedicated upgrade preserves setup and skips calibration sweeps.


## v100.41 goal-linked learning

Public host-timestamped source observations and preregistered forecasts now connect to
verified supervised direction labels, including failed predictions. Resident network
observation runs independently of GUI/paper feeds and GPU training. Current long/mid/short
plans guide researchers and curricula. Token admission defers oversized records. Exported
LoRA A/B candidates receive an additional goal-linked development gate when enough current-goal
validation data exists. It is not an untouched financial audit or causal/profit proof.
See GOAL_PATTERN_LEARNING.md for tool schemas, artifact locations and limits.

Validation for v100.44: full repository tests completed with 956 passed, 65 skipped,
1 deselected, and 1 PEFT configuration warning. The deselected owned-worker process
integration requires process/RAM introspection unavailable in this environment.
The JavaScript live renderer was executed against an older layout with missing live
sections. Ruff, formatting, diff whitespace checks and both installer shell syntax
checks passed. No V100/RTX hardware, LAN, Windows process or guest boot is certified
by those results.

Validation for v100.45: 968 repository tests passed, 65 skipped, 1 deselected,
with one PEFT configuration warning. The excluded owned-worker process integration
requires process/RAM introspection unavailable here. Resource request tests cover
history isolation, host receipts, rejection before desktop execution, archive/requeue
recovery, preservation of later plans and refusal to repair a running mission.
Ruff, formatting, shell syntax and diff whitespace checks passed. Hardware/LAN
allocation and useful execution still require the operator's deployment receipts.

## v100.46 observed agent activity and readable chat

The trusted live renderer adds per-agent cards for observed state, assigned task,
inference device/model, declared conclusion or next step, last tool result, recorded
 time and evidence file. Declarations remain explicitly unverified; no private
reasoning/thinking fields are selected. Missing declarations stay absent. The read
budget remains two daily tails of 128 KiB, 40 events and 16 recent jobs. Hardware
readiness is not attributed as agent execution. Host CSS improves old saved layouts
without replacing operator/master guest HTML. The same validated guest editor remains
available. Browsers already open during installation need a full page refresh.

Terminal chat separates user, master and action receipts, displays queue position
or active processing elapsed time every 15 seconds, and exits cleanly on Ctrl+C
without cancelling mission work. Identical pending messages reuse their request ID.
These UI changes do not raise helper pacing/resource caps or prove useful allocation.

Validation: 972 tests passed, 65 skipped, 1 owned-worker integration deselected and
one PEFT configuration warning. Tests execute the trusted renderer on an old layout,
cover pending-request deduplication, queue/processing labels, clean interruption,
source attribution and omission of private reasoning fields. Ruff, shell syntax
and diff whitespace checks passed. No V100/Windows/LAN execution is certified here.

## v100.47 repairable source and executable capability discovery

read_dashboard now returns invalid HTML as untrusted source plus the validation
error. Validation still gates writes and publication. repair_dashboard removes script
elements and preserves a valid passive design; if further errors remain it restores
the default layout. Both paths back up the original guest file. Dashboard sync
retries after guest boot and repairs invalid layouts before publication, retaining
the repair receipt. A model does not need an operator to paste guest source.

capabilities lists only the current request's allowed operations and their scope.
Chat instructions require tools and verification for concrete implementation requests,
not a plan-only success claim. Repair/write tool calls trigger publication checks even
when the user says only a short follow-up. Simple recognized greetings are answered
by the explicitly labelled controller without model/tool calls or plan changes.
This does not guarantee arbitrary task completion or useful workload allocation.

Validation: 977 passed, 65 skipped, 1 process/RAM-dependent owned-worker integration
deselected, one PEFT configuration warning. Tests cover invalid-source reads without
publication, design-preserving script removal, deferred boot repair, scoped capability
catalogs, greeting isolation and accepted CPU chat for substantive requests. Ruff,
formatting, shell syntax and whitespace checks passed. LAN/GPU/Windows/guest execution
requires deployment evidence.

## v100.48 valid dynamic action objects and direct mission progress

The action validator incorrectly rejected fields in open nested object schemas
such as predict_goal_pattern.specification. It now follows additionalProperties:
unspecified/true allows dynamic fields, false rejects extras, and a schema validates
each extra value. Required and typed known properties remain validated; tool-level
and business admission constraints remain unchanged. This is a reproducible cause
of the reported schema error, not proof of the exact failed on-host tool call.

Plain progress/learning-status questions receive controller evidence rather than
model improvisation or plan actions. Terminal queries bypass the inference queue.
Reports distinguish missing optimizer counters from zero, current benchmark counters
from historical runs, and host-supplied live rendering from passive guest HTML.
Official benchmark progress is written before generation, during grading, after each
case and at completion, both beside its report and in the legacy shared location.
Per-run evidence prefers the local counter so concurrent RTX runs do not overwrite it.
The dashboard's main progress bar now shows the current official benchmark in that phase.

Validation: full repository run 986 passed, 65 skipped, 1 process/RAM-dependent owned
worker integration deselected and one PEFT configuration warning. The final focused
10-test run also covers an added concurrent-helper/local-counter regression and direct
terminal status bypass. Ruff, formatting, shell syntax and diff checks passed.
No successful on-host forecast, training update or profitable outcome is certified.

Continuation requests preserve the current mission goal and report the controller
state without replacing research with HTML plans. Explicit basic dashboard repair
requests write the validated default view with backup and separate publication
receipt. Host styling improves panel separation and background contrast. A concrete
implementation response consisting solely of plans without tool receipts is rejected
instead of saving the plans as progress. Arbitrary future tasks still depend on the
available scoped tools; this does not certify universal autonomous completion.


## v49: readable operation view and operator-selected local reasoning

The host renderer places readable long/mid/short plans first, groups hardware
resources separately, renders agents in two broad columns with expandable tasks,
outputs, results and evidence, and expands action timeline rows to show their
recorded arguments, output, failures, counts and timings. Latest response token
counts retain their observation time; missing values remain unknown. The timeline
is explicitly limited to the latest 200 events from bounded daily log tails.

The operator explicitly requested local model chain-of-thought display. Upgrade
sets capture_local_model_trace=true. LlamaCppClient records only reasoning_content
actually returned by owned inference backends, in bounded journal chunks, using
existing secret redaction. The viewer labels these as unverified local output.
Defaults without this preference still omit reasoning; public outputs are separate.
No thinking mode or token budget is enabled automatically. This is post-response
capture, not token streaming; older omitted traces cannot be reconstructed. Devices
and scheduling infrastructure have telemetry rather than model reasoning.

The compact tool selector now validates that a selected tool name is a string
before dictionary membership, so a model returning a dict is rejected with a
schema error rather than an unhashable-type exception. Regression covers it.

Validation: 990 passed, 65 skipped, one resource-dependent owned-worker test
deselected; one existing PEFT warning. Node runtime exercises the trusted renderer
on older guest layouts. Browser visual QA could not run because the local browser
binary is unavailable. Ruff, shell syntax and diff checks passed. Hardware execution
and LAN publication of v49 await operator installation.


## v50: Synta, paginated actions and active-goal learning evidence

Synta is the system name in host-rendered HTML, terminal chat, chat instructions,
preferences and dashboard service description. Existing binary names and model
aliases stay compatible. The read-only /api/actions endpoint pages exact recorded,
already-redacted daily journals by byte cursor, with bounded pages, date validation,
symlink exclusion and retry-safe incomplete-tail handling. The dashboard exposes
day selection and all recorded fields, arguments, outputs and correlation IDs.
Activity events record the operator goal ID at event time; this does not override
an experiment's original goal snapshot. No missing historical events are fabricated.

The goal ML panel exposes task inputs, observed targets, metrics, training admission,
optimizer evidence and accepted weights. New goal examples default to the active
goal; older verified labels remain archived and independently revalidated when
used for retention replay. Old or unattributed admission counters are not shown as
current goal admission. Malformed evidence/target objects fail validation before
set membership or database binding. This is a generic source-based numerical
forecast task and independently verified auxiliary tasks, not a claim that any
natural-language goal automatically supplies labels, a reward oracle or profit.

Validation: 996 passed, 65 skipped, one resource-dependent owned-worker integration
deselected and one existing PEFT warning. Final focused renderer/telemetry checks
passed after replacing missing ML counters with explicitly unknown values. Tests
cover archive pagination, goal attribution, incomplete tails, path/symlink rejection,
HTTP action routes and goal-change separation while preserving verifiable replay.
No hardware training improvement or universal goal oracle is certified.
