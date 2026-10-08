# V100 mission audit v100.38

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
| Chat while research continues | Persistent queue, accepted master and explicit RTX delegate | One V100 slot shares inference; exclusive training prevents accepted-master chat temporarily |
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
