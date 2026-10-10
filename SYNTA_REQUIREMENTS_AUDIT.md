# Synta: requirement and verification audit, v60

This audits the visible conversation. "Implemented" means code exists and the
listed checks pass, not that it has run successfully on the operator's computers.
Hardware-dependent checks are separate. No real orders, spending, paid APIs,
account creation or sales are authorized by this implementation.

| Requirement | Code / checks | Result and boundary |
| --- | --- | --- |
| Name Synta; distinguish user/master/controller | `mission_chat.py`, `tests/test_v100_chat_backend.py` | Implemented; controller capability/status answers are identified, not claimed as model generation. |
| Chat answers; finite timeouts and no endless queue retries | `mission_chat.py`, chat/backend tests | Implemented; ordinary questions use one generation, identity probes bounded, connection retries finite, incomplete replies fail explicitly. An offline backend cannot supply a genuine model answer. |
| Master executes requests instead of only writing plans | `research_tools.py`, `mission_chat.py`, action-object/resource tests | Named validated tools and receipts; plan-only implementation replies do not count as execution. Model tool choice is not guaranteed. |
| Goal editable by explicit chat request | `chat_goals.py`, `planning.py`, chat-goal tests | Implemented; old experiments preserve their original goal; fetched content cannot change the operator goal. |
| Dashboard accessible over LAN and updated automatically | `tools/serve-v100-dashboard.py`, `dashboard_editor.py` | Read-only host service with trusted renderer and validated passive guest layout. Host/guest publication hashes must match. |
| Master edits dashboard without pasting HTML into chat | `dashboard_editor.py`, dashboard tests | Scoped read/write/repair tools; source read is separate from publication validation. Guest scripts remain disallowed; host supplies refresh. |
| Clear resource, agent, plan, job and evidence sections | Dashboard renderer, activity-view tests | Implemented cards/timelines. No visual browser binary here, so aesthetic acceptance is not certified. |
| Reading position and expansion survive all refresh paths | Transactional/keyed renderer, dashboard Node tests | Same DOM reader/text nodes and scroll offsets retained across repeated refreshes; live browser on operator machine remains an acceptance check. |
| Show actions and results of controlled tools | `activity.py`, `research_tools.py`, `activity_browser.py` | Correlated start/result/error events and paged journal. Redaction, bounds and omitted secrets are intentional. This is not an OS-wide syscall recorder. |
| Show each model's returned text/token measurements | `llamacpp.py`, `remote_helper.py`, `streaming.py`, streaming tests | Returned fragments and actual usage counters, with model/device/request provenance. Missing counters are unknown. Not every backend exposes raw token IDs. |
| See model reasoning | Opt-in local returned-trace capture | Only reasoning actually returned by owned local models can be recorded. Unavailable internal reasoning is not fabricated or inferred from GPU usage. |
| GPU/CPU utilization is distinct from actual useful work | Activity/resource cards; job receipts | Implemented. Readiness, loaded VRAM and idle heartbeat do not prove a task was executed. |
| RTX helper survives Windows logon and recovers | `install-rtx3090-autostart.ps1`, guardian/firewall policies | Current-user logon task, existing guardian/game guard retained. Matching narrowly scoped firewall rules reused, unrelated rules unchanged. Windows runtime test required. |
| Visible PowerShell monitor | `watch-rtx3090-helper.ps1`, `Synta-Helper-Monitor` task | Separate visible read-only window at logon; closing it does not stop compute. LAN binding corrected. |
| Owned Windows CPU worker autostart | `install-owned-compute-autostart.ps1` | Added; existing authenticated mailbox/settings recovered; credentials/share unchanged; running worker preserved, new revision used after it exits/logon. CPU child remains 2 threads / 4 GiB. |
| 50% helper target consistently | RTX autostart and `start-windows-lab.ps1` | Both use 50% active wall-time pacing, not a hard board utilization/power/temperature cap. Exact installed CPU dependencies avoid repeated network resolution. |
| Use Windows RAM/disk/CPU and RTX | Owned mailbox, distributed compute, remote helper | Worker-local bounded jobs and safe artifact transfer. RAM/VRAM is not pooled into Debian; available hardware need not run dummy work to raise utilization. |
| Automatically allocate useful owned CPU work | `goal_compute.py`, dispatch tests | A/B tasks queued for fresh idle workers when enough verified active-goal outcomes exist. No valid outcomes means an explicit blocker. |
| Goal-oriented ML from newly discovered signals | `goal_learning.py`, `goal_compute.py`, `continuous.py` | Precommitted numeric signal/target forecasts, observed later outcomes, source-disjoint splits, verified master examples plus separate tiny CPU baselines. Tools support operator directions and model proposals. Arbitrary goals need an independently verifiable task/label definition. |
| Social/news/filings evidence linked to outcomes | Public-source/JSON feed registration, `paper_agents.py`, goal observation tools | Available primitives; a measured association does not prove causality or profit. No promise of universal social-feed access or discovery of an income edge. |
| Compare lawful opportunities, sports, crypto, equities | Research tools, provider registry, paper ledger | Research and bounded paper simulations; execution requires valid fees, feeds and domain settlement rules. Stale/unavailable sources remain blockers, not fabricated substitutes. |
| Scalping/news/arbitrage and high leverage | Earlier assistant-generated plans | Not falsely certified as implemented profitable algorithms. These names came from model replies; placeholders and promises are not executable evidence. Leveraged live trading remains outside the authorized goal. |
| Retention experiments including capacity growth | `retention.py`, `capacity_growth.py`, training/distillation, actual CPU PEFT tests | Replay, KL, reference-gradient projection, Fisher/drift constraints and optional adapter growth; residual-column/embedding extension experiments. Activation requires independent finite gates. No universal zero-forgetting guarantee or claim every published method was implemented. |
| Prior weights/checkpoints and A/B histories survive | Protection, branch-lineage, activation/rollback tests | Implemented; candidates do not overwrite immutable originals. Tiny worker weights are not automatically substituted for master weights. |
| Honest training progress and optimizer counters | `mission_evidence.py`, training-health tests | Attempted/global steps, completed optimizer updates and accepted versions remain separate; missing means unknown. GUI readiness is not a prerequisite for native inference/training. |
| End-to-end task execution and acceptance decision | Real child-process goal test and independent local tensor validation | Observations -> frozen job -> actual optimizer -> safe weights -> host reload -> locally validated/rejected decision. Hardware sensors are simulated in sandbox; old goal labels cannot enter a new goal pool. Master activation separately tested through quality/retention/goal gates; no actual V100 training here. |
| Code size / number of changed lines | Git tracked source/diff history | Reproducible with Git; temporary test directories are excluded. Line count is not a completion or quality measure. |

## Hardware acceptance still required

After installing the pinned Debian and Windows revisions, verify a real model
answer, stable browser reading position, fresh Windows CPU heartbeat plus actual
job receipt, RTX inference logs, and a logon recovery. Verify GPU training and
accepted weights only from new run artifacts and independent gate results.
Neither screenshots nor passing mocked transport tests establish these facts.

## Operator goal is retained

The financial goal is not replaced by arithmetic, dashboard maintenance or a
small forecast task. Arithmetic is an auxiliary experiment. Failed trading tests,
losses and rejected updates remain evidence. No result here establishes income.

Source-size snapshot: 46130 physical lines in 190 tracked/new Python, PowerShell and shell files under `rlm/v100`, `tools` and `tests`. Includes comments/tests, excludes temporary files.

Verification result: full test suite 1028 passed, 65 skipped, no deselections;
ruff and upgrade shell checks pass. Windows startup policy has static contract
checks, not a Windows runtime execution. Real CPU child-training/tensor-validation
integration is included with simulated resource readings.

## v58 structural evolution

`morphology.py` and named research tools add bounded typed GRU/Transformer shape
proposals and compatible identity-initialized residual growth. Existing queued
master trials execute them in isolation. Activation additionally requires verified
goal outcomes and strict measured task improvement. Native GGUF architecture is
not reshaped in place. See `SYNTA_MORPHOLOGY.md` for the executable contract.

## v59 runtime recovery

See `SYNTA_RECOVERY.md`: same-run stale-report preservation, independent verified
goal reads, read-only goal-status SQLite, actual process/forecast receipts, archived
UI-plan recovery and independent forward-only financial reference labels. These
labels commission supervised task data, not model skill or profitable strategies.

## v60 recovery additions

Native startup now bypasses a second controller CLI and reports stages with a
child-bound protected receipt. Research respects the operator's 50% pacing.
Income candidates/tests have their own bounded, versioned evidence ledger and
periodic RTX allocation, including non-market mechanisms. UI work cannot replace
automatic financial plans. Public-source counters distinguish rereads and fresh
content; event evidence is separate from a shared journal path.

Validation includes a real child HTTP server standing in for the native binary,
actual SQLite/host source archives with mocked network, concurrent fetch-counter
updates, invalid/stale evidence rejection and explicit unknown payment. It does
not test V100 CUDA, profitability, actual demand or the operator's Windows run.
