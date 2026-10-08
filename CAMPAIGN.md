# V100 campaign v32

Use `tools/upgrade-v100-campaign.sh RELEASE_COMMIT` on Debian after any existing training/calibration/MTP sweep completes. It updates only the existing isolated continual venv, repairs the launcher atomically, retains checkpoints, prepares optional components, runs mandatory short CUDA/master acceptance probes, and starts the owned supervisor only after those probes pass. Failed acceptance retains a report and leaves the mission stopped. Preparation prints each stage and records failures as `deferred`; a deferred feature is not operational.

## Operating from one console

- `~/ai-v100/bin/v100-continual chat`: persistent chat while research continues. During exclusive V100 experiments the pinned RTX researcher may answer as a visibly identified delegate. Model requests remain subject to server queues and configured resource budgets.
- Normal chat, e.g. `Mój główny cel to ...`, `Plan średnioterminowy: ...`, `Plan na dziś: ...`: changes the stated goal/plan. `/cel` remains an optional shortcut. Short/mid plans may be revised by the model; long-term changes require the current operator request.
- `mission-report`: accepted weight updates, A/B lineages, paper results, plans, drone queue and benchmark status.
- `mission-watch`: model/tool events. Ctrl+C in chat/watch does not stop the mission.
- `mission-stop`: intentional pause; supervisor respects it and checkpoints remain.
- `supervisor-start`: resume. User service can start after boot only when user lingering is enabled; detached fallback survives SSH disconnect but not reboot.

## Implemented mechanisms

A/B keep separately accepted adapters and serve their own accepted versions sequentially on the V100. Verified source/CPU jobs and one bounded RTX job run concurrently with local work. Persistent queues resume after restart. New adapters, merged children and algorithm candidates remain unpromoted until their relevant gates pass. Retrospective backtest labels include losses and independently recomputed accounting. Fully resolved audited paper decisions additionally yield earlier-context/action preference pairs for LoRA DPO; a small CPU reward-trained decision head remains shadow-only. These observational objectives do not establish future profitability or unbiased policy reinforcement learning.

Chat/master can schedule source jobs, research/critic jobs, bounded CPU experiments, desktop experiments and public benchmark jobs. It can read its pinned full source, copy it inside the guest and propose changes. Resource controls can adjust thinking/output/batch within validated budgets. RTX request pacing is not a hard GPU utilization, transient power or VRAM cap. Existing Windows game guard pauses the helper during League; recurring black screens require hardware diagnosis independently of this software.

## Private Linux desktop

Optional rootless QEMU guest: 2 CPU cores, 3 GiB RAM, 24 GiB persistent virtual disk, no host home/model/credential mounts or GPU passthrough. A read-only source ISO exposes its own code; guest root can install tools and create files inside its disk. Public HTTP(S) egress is brokered and private/loopback destinations are rejected. GUI tools use the guest display; image analysis uses the pinned Qwen vision helper. RAM/disk pressure stops only this guest and retries later. CPU-only emulation can be slow when KVM is unavailable.

Initial preparation downloads a Debian cloud image and private signed Debian packages; first boot additionally installs the desktop. Readiness is checked through key-pinned SSH, the cloud-init completion marker, mounted source and the guest GUI service/display. A running QEMU process alone is not reported as ready. GPU inference, actual VM boot and Windows PowerShell monitoring require validation on the operator's hardware. No sandbox can promise protection against every kernel/hypervisor vulnerability.

## Benchmarks and learning limits

Separate pinned LiveBench environment, official grader and dataset snapshots. Default balanced 20-case non-coding panel covers five categories. Reports compare previous accepted versions and published model results only where matching cases are available. This panel is not the full official leaderboard or Artificial Analysis index, and published providers may use different inference budgets. A failed scorer is an infrastructure error, not a zero model score. Benchmark material is not fed to research memory or training.

Immutable originals, replay, preservation losses, independently verified examples, fixed regression suites and rollback reduce forgetting; finite tests cannot prove universal zero forgetting. Accepted updates are counted explicitly; a research cycle or saved hypothesis is not a weight update. Paper outcomes remain distinct from exploratory backtests. Real order execution is not installed.

Colab remains an operator-started bounded notebook pilot with fresh pinned inputs, safe-tensor output and independent host evaluation. Free managed Colab explicitly prohibits distributed workers; use the owned-computer system below instead. No account/cookie quota rotation or paid API calls are used.

## Windows visibility

`tools/watch-rtx3090-helper.ps1` shows whole-board GPU readings, CPU/RAM, loaded helper and recent server logs. Its readings are not per-process power measurements. Ctrl+C stops only the viewer.

## Goal-directed self-upgrade and fresh checks

Self-upgrade under the operator's goal is the default mission; financial research is optional (`campaign-prepare --paper`). Existing explicitly stored long-term goals are retained, never silently overwritten. Set the goal through normal chat, e.g. `Mój główny cel to ...`; `/cel` remains an optional shortcut. Only the current operator message authorizes the change. Short/mid-term plans and tools remain available without financial instruments or a paper ledger.

`request_fresh_curriculum` generates up to 32 new calculator-verified examples when a demonstrated weakness makes them useful. No benchmark answers are admitted. Campaign profiles require a fresh audit before accepting an A/B adapter, a bred child or an alternative foundation expert: freeze exported model identities first, draw fresh random arithmetic/equation/decimal tasks, reserve their source IDs against training, compare predecessor/candidate under the same generation settings, and consume the audit once. Interrupted attempts cannot reuse exposed cases. Detailed cases remain in operator files, not model feedback. This is a finite general-skill gate, not a hidden test for every possible user goal; goal-specific independent future outcomes and operator tests are still necessary.

Pinned official benchmarks remain comparison/regression measurements. `benchmark-prepare --limit 0 --coding` selects all tasks in the pinned release and uses private-VM coding graders. Agentic repository repair requires a separately provisioned privileged grading VM; its absence leaves the full report incomplete, never a fabricated leaderboard score. The smaller panel remains the default to keep ordinary upgrade cycles bounded.

`propose_foundation_trial` can queue a different library-supported pretrained architecture, with pinned revision/download/context budgets, LoRA/export tests and the same quality gates. Successful alternatives become immutable independent experts and can automatically become the master after a strict task improvement, no ancestor regressions, a passed fresh audit against the current master, public checks when configured, and an actual native boot probe. The preceding native model is copied into an immutable rollback expert; A/B adapter histories remain archived by base. No adapters from different bases are combined. Two repeated owned mission failures can restore the predecessor and quarantine the failed architecture. Arbitrary guest-side architecture research and tiny all-weight pilots remain separate from trusted host loaders.

Normalized public JSON feeds can cover any already registered crypto/equity/sports instrument, with documented mappings, source timestamps and settlement semantics. Provider documentation/observations are archived; no retrieval-time quote rejuvenation, broker-rule certification or new fee discounts are invented. At most four registered feeds are read per observer tick, rotating across the registry. `consult_browser_model` opens a researched free service in the guest; authentication/captcha requires operator handoff, and quotas must be respected.

## v28 reliability fixes

Resident worker crashes requeue interrupted bounded jobs and restart after cooldown; a full persistent queue does not block mission startup. SQLite handles close after transactions. The supervisor user service sources the isolated environment and refreshes its owned predecessor on upgrade. `/status` in chat returns locally without inference. Replies identify V100 or the RTX delegate. Pinned complete public baselines are reused only for identical weights, datasets and generation; measured MTP can be adopted without replacing accepted training ancestry.

Full requirement and evidence audit: [REQUIREMENTS.md](REQUIREMENTS.md).

## v31 architecture switching and owned distributed compute

`propose_foundation_trial` is connected to automatic activation in the learning loop, with immutable rollback models and hash-bound ancestor gates retained across restarts. An equally scoring architecture remains an expert, rather than replacing the master without a demonstrated improvement. Alternative architectures disable predecessor-bound MTP until measured again.

The operator configures an authenticated dedicated shared folder with `compute-configure`. `tools/start-owned-compute-worker.ps1` and `.sh` create separate CPU-only environments and connect Windows/Linux computers to that folder. No listener, firewall rule, board power setting or original Ollama install is changed. The model can propose and cancel `propose_compute_trial` jobs and inspect `compute_trial_status`. Jobs choose built-in GRU/transformer shape, steps and learning rate under 2M parameter / 2 thread / 4 GiB child RAM / 150 second wall limits. The worker pauses for host pressure or League gameplay. Atomic exclusive claims prevent duplicate work, expired claims retry up to three times, and remote results remain untrusted. A host drone independently evaluates safe-tensor weights against the pinned heldout dataset before reporting local improvement. This is small-model R&D, not distributed Gemma full-weight training or automatic master promotion.

`colab-import` can import pinned safe-tensor notebook output into this same local validation queue. Each proposal uses new calculator-verified examples. Notebook GPU usage is explicitly interactive, bounded to 120 seconds and separate from the owned worker service. A smaller heldout loss alone is neither a mastered general ability nor evidence of income.

The feature-branch CI performs CPU tensor/training tests and parses Windows PowerShell scripts on standard public-repository runners. CUDA performance, guest boot, the LAN mount and RTX stability still require the operator's hardware; CI does not certify them.

Verified CPU/Windows checks: [run 37791661812](https://github.com/Mantiee/rlm/actions/runs/37791661812), 504 tests passed with none skipped, and all PowerShell scripts parsed successfully. This includes real tiny GRU/transformer training, the owned worker-to-host tensor round trip, adapter gradients, breeding and checkpoint resume. Native V100/RTX execution and private-VM boot were not run in CI.


## v32 custom full-weight masters and on-device acceptance

`create_submodel` freezes arbitrary `build(config)` PyTorch network code implementing `[batch, length, 257]` UTF-8 byte logits. `propose_scratch_master` queues a bounded full-weight experiment (joint trials use `support_scratch_master` for both branches): at most 1B parameters, 24 GiB host RAM, 30 GiB configured GPU budget and 7200 seconds. Namespace isolation, process/file limits and sampled RAM/VRAM/artifact guards apply; native Gemma is stopped during exclusive CUDA trials. Candidate Python is never imported into the host controller. Admission preserves the predecessor's context/output conditions and requires complete ancestor, configured public and post-freeze fresh checks plus a real isolated inference probe. Activation snapshots source, launcher, budget and safe-tensor weights into an immutable expert. Equal scores retain the predecessor; a failed serving version is quarantined and rolled back.

A selected scratch master can queue A/B full-weight warm-start children only when newly admitted verified examples arrive. Each is a new bounded experiment; validation loss chooses its best weights and the host gates decide promotion. Tiny full-weight training tests demonstrate optimizer changes, not that a new network will outperform pretrained Gemma. Byte counts are not BPE-token throughput. This backend reloads inside the sandbox per turn and recomputes its prefix, so it is an architecture research path, not an assumed faster default or native KV/MTP backend. Existing native LoRA experiments retain their separate optimizer-resumable checkpoints. Built-in remote CPU pilots do not bypass master promotion gates.

`hardware-acceptance` produces a durable report, verifies Volta CUDA backward and a short master response, then checks the sandbox and RTX identity opportunistically. `--desktop` additionally checks an already provisioned guest. Short probes do not certify a full 131k-token workload, prove Flash Attention kernel selection, diagnose RTX black screens or measure financial performance. Optional infrastructure failures remain visible and deferred; mandatory GPU/master failures prevent automatic startup.

## Read-only free market adapters

`market-adapter-register` accepts an immutable JSON object with `id`, `provider`, `settings` and `free_only: true`; the corresponding `configure_free_market_adapter` tool lets the model propose mappings. Provider endpoints are fixed read-only data origins, snapshots are archived with hashes, quota usage is reserved before requests, and no paid/history/order endpoint is called. Observer ticks rotate at most four adapters and pace repeated source requests. Credentials belong in the operator's controller environment, not model-visible JSON. Missing keys and provider outages defer that feed rather than stopping the mission.

* `alpaca-iex`: operator-provided free `V100_ALPACA_KEY` and `V100_ALPACA_SECRET`, IEX latest quotes only. Settings specify ticker, registered symbol/feed and independently documented round-lot size. Source timestamps and raw prices are retained; IEX is one venue, not consolidated NBBO. PLN conversion uses an archived NBP reference and explicitly modeled FX costs, not a claim of executable FX. Corporate actions require independent records.
* `sports-odds`: an existing free `V100_ODDS_KEY`, one region, h2h event, bookmaker and outcome. Current odds can be archived as research without claiming free historical odds. Executable paper mappings must exactly match the operator-verified instrument's event, bookmaker, outcome and final-score settlement rule. The provider supplies no stake liquidity: `paper_stake_cap` is a simulation ceiling. Once the event begins, final scores can settle the paper position even when odds disappear. Final-score grading supports documented draw/void-tie rules; bookmaker-specific exceptions and later score corrections are not automatically certified. A local 100-credit monthly budget prevents paid fallback.
* `bybit-linear`: public contract ticker research without a key. Predicted funding and instantaneous marks cannot certify settled funding or the worst intrabar margin loss, so this adapter emits research evidence only. Generic independently documented interval/funding feeds can drive the existing isolated-linear paper engine; this ticker alone cannot.

No new symbols, volume discounts, regional eligibility, operator credentials or bookmaker rules are inferred as certified. Stocks/sports still need registered instruments and actual documented fee profiles before paper fills. Fees, FX, drawdowns and losses remain in accounting. No real orders or guaranteed income are enabled.

## Two-console launch

`tools/start-windows-lab.ps1 -Revision RELEASE_COMMIT` starts the isolated RTX helper at 32768 context, batch 16 and 15% active-request time target, then shows helper/native logs and whole-board GPU/CPU/RAM readings in the same console. League pauses the helper. This is request pacing, not a hard per-process GPU/power cap or a crash fix; it never changes the global board power limit.

The wrapper also starts a hidden CPU worker launcher waiting up to two hours for the authenticated mailbox. Its default UNC is `\\192.168.0.68\dane\tests\v100-owned-compute`; pass `-Mailbox` if the actual Samba share differs. Debian's updater configures `/srv/samba/dane/tests/v100-owned-compute` when the known parent folder exists. No Samba share or stored credentials are created. Until the UNC is reachable and the worker reports ready, external CPU compute is deferred, not operational. Windows CPU jobs keep the isolated 2-thread/4-GiB/150-second limits and pause for games or pressure.

Run the pinned Debian updater in one SSH console, then use that same console for `chat`. The supervisor runs separately, so closing chat does not stop research. Boot autostart requires the user systemd service plus operator-enabled lingering. Power outages, unavailable workers and unprovisioned infrastructure cannot be described as uninterrupted operation.
