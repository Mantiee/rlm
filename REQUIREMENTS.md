# Conversation requirements: implementation and evidence

This audit covers the practical requests from the V100/RTX discussion. Code presence does not mean an operator's hardware has run it. `campaign-prepare` records ready/deferred setup stages; `mission-audit`, `mission-report` and chat `/status` report observed runtime evidence.

| Request | Implemented mechanism / status |
|---|---|
| Keep working environments intact | Isolated continual, benchmark, MTP and model/tool folders; atomic launcher; original weights/checkpoints retained. |
| Fast V100 serving with large context | Native Q6_K serving, Flash Attention/KV configuration, context fallback from 131072, bounded output budgets. Actual speed and long-context accuracy require local measurements. |
| Speculative decoding | Target-bound MTP speed sweep plus fixed quality checks; enable only a verified gain. A changed model/context requires new evidence. |
| Persistent learning / weight changes | Verified examples, actual optimizer steps, saved training state/checkpoints, finite preservation and task gates before accepted serving updates. Each cycle may finish without a weight update. |
| Prevent forgetting | Immutable originals and accepted versions, replay, preservation KL, separate adapter lineages and regression/rollback. Universal zero forgetting is not provable with finite tests. |
| Competing A/B and crossbreeding | Independent accepted lineages, sequential GPU experiments and parent-bound adapter children; shared evidence and memory. Two entire training models are not simultaneously loaded into one V100. |
| Testers during training | Resident source/CPU jobs and bounded RTX researchers/critics share persistent queues and explicit resource budgets. Worker crashes retry interrupted jobs; external CPU jobs use exclusive leases, three bounded retries and independent host safe-tensor validation. |
| Smart resource allocation | Request policy can change thinking/output limits; stopped-server experiments handle native batch changes. RTX game guard/pacing is not a hard per-process GPU peak cap. |
| Durable memory / compression | Archived originals, hierarchical summaries, hybrid CPU embeddings/retrieval; hypotheses retain unverified status. This does not extend the model's native attention window losslessly. |
| Chat while work continues | Persistent controls, V100 accepted serving model when available, labeled RTX delegate while V100 is busy; `/status` and `/cel` do not need inference. Without any available LLM, ordinary chat waits. |
| Long / mid / short objectives | Operator-requested long goal in ordinary chat; versioned model/user short and mid plans. User can steer next research and local paper alerts through chat. |
| Sandbox files, installs, internet and GUI | Private bounded Debian VM, writable guest disk, read-only source ISO, public HTTP(S) broker, guest GUI commands and Qwen vision. Real guest boot remains an on-device check. |
| Self-code and architecture research | Full own-source read/copy, sandbox algorithm candidates, small all-weight CPU model pilots. Host prompts, accounting and promotion gates cannot be replaced by candidates. Pinned library-supported alternatives and isolated custom byte-token networks can automatically replace the master only after improvement, finite ancestor/public/fresh gates and a real inference probe; immutable predecessor rollback and base-specific adapter histories are retained. |
| 24/7 and SSH disconnect | Owned restart supervisor/user service with cooldown, deliberate-stop protection; user lingering needed for boot autostart. Host power failures and hardware faults remain external. |
| Visible steps and results | Activity timeline, native logs, training metrics, paper reports, worker status, benchmarks and Windows helper hardware viewer. Hardware readings measure the board, not helper-only power. |
| Profit outcome learning | Independently audited paper postmortems and recomputed historical reviews, including losses, can enter supervised training. Fully resolved audited decisions additionally produce earlier-context LoRA DPO preferences; a reward-trained CPU head remains shadow-only. Neither establishes an income edge or unbiased policy RL. |
| Research beyond fixed symbols | Public-source/backtest tools and source drones, Kraken market discovery/registration, SEC metadata and other-income research. New instruments require valid cost/feed/product rules. |
| Stocks/sports/leveraged trading | Generic research supported; normalized documented public JSON mappings support already registered equity/sports/leverage instruments; read-only free IEX quotes, current sports odds/final-score settlement and public Bybit ticker research adapters are implemented. Operator credentials and verified execution/fee/rule registration remain necessary; automatic broker certification and a complete live derivative risk feed are not supplied. Never treat odds as stock prices or assume leverage cannot create debt. |
| Official AI comparisons | Pinned official LiveBench panel and optional full-release/coding grading in the private VM; agentic Docker grading requires a separate provisioned VM. same-case published model references where available. The default panel is not a full leaderboard score or private commercial benchmark index. |
| Free Colab / other LLMs | Fresh bounded interactive Colab pilot with safe-tensor import and local heldout evaluation; distributed CPU training workers run on authenticated operator-owned Windows/Linux machines; existing free HF consultation and researched guest-browser model consultation, with operator auth handoff. Free managed Colab prohibits distributed computing workers. Account/cookie quota bypass is excluded. |
| Best profitable result, no losses | Goal for investigation; no guarantee. Exploratory backtests, fees and forward paper outcomes are reported separately. |

## Anti-overfit and goal priority

The main mission follows the operator's goal and measured self-upgrades. Finance is an opt-in module; it does not determine every experiment. Campaign profiles add post-freeze randomized one-use audit gates, source-ledger exclusion from training and no detailed candidate feedback. Fixed/public benchmarks compare versions; they are not training data. The trusted curriculum tool draws new independent examples, not audit/benchmark exercises. Fresh random general-skill tests do not certify arbitrary domain goals: future independent outcome tests are still required.

## On-device acceptance

1. Finish any active exclusive training/MTP sweep before the pinned updater.
2. The installer runs mandatory short CUDA/master probes and retains `hardware-acceptance` reports. Inspect preparation stage results; deferred sandbox/benchmark features are unavailable until their reported failure is corrected.
3. Check `mission-report` and `/status`: actual loop phase, worker readiness, guest health, completed evaluations and accepted weight updates.
4. Treat the Windows helper's existing black-screen problem as unresolved until hardware-side testing confirms stability.

Primary implementation references: [QEMU user networking](https://www.qemu.org/docs/master/system/qemu-manpage.html), [LiveBench official source](https://github.com/LiveBench/LiveBench), [Colab resource and usage FAQ](https://research.google.com/colaboratory/faq.html).


## Explicit practical limits

The implementation covers native learning, guarded architecture replacement, persistent chat/goals, owned worker queues, sandbox/GUI paths, memory, benchmark comparisons and paper research. Hardware probes and queued experiments are not evidence of an accepted self-upgrade or income. Custom architectures must implement the bounded byte-logit interface; a different tokenizer/runtime or unbounded cluster is not silently supported. Free managed Colab is an interactive pilot, not a distributed worker farm. Account/cookie quota evasion, paid APIs and real order execution are excluded. Universal zero forgetting, guaranteed gains, indefinite third-party availability and a software-only cure for RTX black screens cannot be promised.
