# V100 mission release v100.24

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
