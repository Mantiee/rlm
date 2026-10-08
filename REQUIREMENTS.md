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
| Testers during training | Resident source/CPU jobs and bounded RTX researchers/critics share persistent queues and explicit resource budgets. Worker crashes retry interrupted jobs. |
| Smart resource allocation | Request policy can change thinking/output limits; stopped-server experiments handle native batch changes. RTX game guard/pacing is not a hard per-process GPU peak cap. |
| Durable memory / compression | Archived originals, hierarchical summaries, hybrid CPU embeddings/retrieval; hypotheses retain unverified status. This does not extend the model's native attention window losslessly. |
| Chat while work continues | Persistent controls, V100 accepted serving model when available, labeled RTX delegate while V100 is busy; `/status` and `/cel` do not need inference. Without any available LLM, ordinary chat waits. |
| Long / mid / short objectives | Operator-only long goal; versioned model/user short and mid plans. User can steer next research and local paper alerts through chat. |
| Sandbox files, installs, internet and GUI | Private bounded Debian VM, writable guest disk, read-only source ISO, public HTTP(S) broker, guest GUI commands and Qwen vision. Real guest boot remains an on-device check. |
| Self-code and architecture research | Full own-source read/copy, sandbox algorithm candidates, small all-weight CPU model pilots. Host prompts, accounting and promotion gates cannot be replaced by candidates. Arbitrary pretrained Gemma architecture replacement is not implemented. |
| 24/7 and SSH disconnect | Owned restart supervisor/user service with cooldown, deliberate-stop protection; user lingering needed for boot autostart. Host power failures and hardware faults remain external. |
| Visible steps and results | Activity timeline, native logs, training metrics, paper reports, worker status, benchmarks and Windows helper hardware viewer. Hardware readings measure the board, not helper-only power. |
| Profit outcome learning | Independently audited paper postmortems and recomputed historical reviews, including losses, can enter supervised training. This is not profit-policy reinforcement learning or proof of an income edge. |
| Research beyond fixed symbols | Public-source/backtest tools and source drones, Kraken market discovery/registration, SEC metadata and other-income research. New instruments require valid cost/feed/product rules. |
| Stocks/sports/leveraged trading | Generic research supported; automatic provider-specific equity/sports/leverage execution and settlement adapters are not implemented. Never treat odds as stock prices or assume leverage cannot create debt. |
| Official AI comparisons | Pinned official LiveBench non-coding panel and graders; same-case published model references where available. The default panel is not a full leaderboard score or private commercial benchmark index. |
| Free Colab / other LLMs | Bounded manual Colab pilot with pinned import/export and local reproduction; existing free HF consultation tool and guest browser capability. Automatic free-tier distributed workers and account/cookie quota rotation are not implemented. |
| Best profitable result, no losses | Goal for investigation; no guarantee. Exploratory backtests, fees and forward paper outcomes are reported separately. |

## On-device acceptance

1. Finish any active exclusive training/MTP sweep before the pinned updater.
2. Inspect preparation stage results; deferred sandbox/benchmark features are unavailable until their reported failure is corrected.
3. Check `mission-report` and `/status`: actual loop phase, worker readiness, guest health, completed evaluations and accepted weight updates.
4. Treat the Windows helper's existing black-screen problem as unresolved until hardware-side testing confirms stability.

Primary implementation references: [QEMU user networking](https://www.qemu.org/docs/master/system/qemu-manpage.html), [LiveBench official source](https://github.com/LiveBench/LiveBench), [Colab resource and usage FAQ](https://research.google.com/colaboratory/faq.html).
