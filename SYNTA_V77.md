# Synta v77: measured inference experiments and curriculum evidence

This release includes the v76 dashboard and strategy-evidence fixes. The operator's last supplied installer output refused v76 because trainer PID 967599 was still running. That output establishes neither v76 installation nor a completed training update.

## Changes

- Native serving accepts model-free `ngram-simple`, `ngram-cache` and `ngram-mod` configuration. Profiles with disabled legacy draft-model settings continue to launch without speculation. N-gram proposals still require target-model verification; no synthetic acceptance flags are used.
- A standalone benchmark compares the same model/context with no speculation, n-gram, q8 KV cache when applicable, their combination, and partial CPU layer offload. It saves model SHA-256, complete outputs, timings, logs and selection evidence. Unsupported native n-gram flags are reported as blocked. Arithmetic and structured-output probes must pass; truncated or empty answers are rejected.
- A faster candidate must match all eight baseline probe outputs and improve their total measured request time by at least 10%. This is a small deterministic inference experiment, not a broad quality evaluation, training outcome or financial-performance test.
- The existing `compute_resources` tool exposes the saved local benchmark selections, blocked states and configured speculative mode to the master. It distinguishes a configured profile from a currently verified serving process.
- The Debian updater promotes only the measured n-gram candidate with unchanged model, context, cache and GPU layout. Cache/offload winners remain proposals for independent task-quality evaluation. Failed or pressured benchmarks restart the preserved mission without a promotion.
- Windows uses official pinned b10683 CUDA 12.4 archives with published SHA-256 checks, and reuses the existing qwen3.5:9b-q8_0 GGUF blob after verifying its manifest digest. No new model, driver or scheduler configuration is installed. The native benchmark is separate from the existing Ollama helper; Ollama does not acquire native n-gram flags from it. Switching the production Windows transport requires another evaluated change after inspecting results.
- Resource admission on Windows requires free RAM, free VRAM, CPU and GPU temperature/power headroom. A browser is allowed. Known game processes and a fresh generating helper receipt interrupt the benchmark. Monitoring terminates only its own native process. Thresholds are sampled admission/interruption policies, not peak power or crash guarantees.
- Each captured historical execution, including final liquidation, checks that its same-reference-price spot-equity loss equals modeled fee plus slippage. Receipts expose the percentage of pre-fill equity and the reconciliation result. Existing archived reports are retained; the explorer reconstructs diagnostics from their recorded inputs.
- The curriculum catalog exposes verification kind and actual task scope. Numeric source-change labels are not evidence of learned bug discovery, web operations or income skill. Each duel records available versus selected training examples, held-out validation count, planned steps and approximate example passes. Selection remains model-led; held-out source roles are unchanged.
- General dashboard edit requests no longer restore a fixed default layout automatically. Only `/reset-dashboard` or an explicit restore-default command does so. Combined model/n-gram/offload requests go to the master rather than the bounded CPU-concurrency shortcut.

## Accounting interpretation

At constant prices, assuming 50 bps fee and 10 bps slippage, a full-capital buy leaves spot equity `1 / (1.001 * 1.005)`, a drop of about 0.596%. Immediate liquidation value reserves the future sell leg as well. Later candle movement is a separate component. Selling before a subsequent price fall avoids that later fall; it is not itself evidence of a reversed strategy.

The regression fixture independently calculates constant-price buy and sell cash flows and reconciles total costs, including a zero-cost control. That fixture does not certify a provider's prices, assumed fee tier, archived inputs or live paper observations. Archived same-signal zero-cost replay remains a diagnostic, not available execution conditions or a profitable strategy.

## Install and run

Use `tools/update-synta-v77.sh` pinned to this release SHA on Debian. It supports installed v74-v77, refuses an active exclusive training/preparation process before changing services, preserves the current profile, installs v77, runs its isolated benchmark, restarts the supervisor and checks local dashboard readiness for up to 60 seconds. Inspect `research/inference-recovery-update/<timestamp>/inference-benchmark/results.json` or `benchmark-blocked.json`. A running training cycle must finish first; this script does not kill it to enable an update.

On Windows, run `tools/benchmark-synta-windows.ps1 -Revision <release SHA>` using the existing isolated CPU Python environment. Initial official native runtime download is approximately 642 MB plus extraction space. Results are under `%USERPROFILE%\ai-v100-helper\performance\<timestamp>\results\results.json`. Port 8097 is loopback-only and must be unused. The experiment has no production helper promotion or autostart. Browser use is allowed; resource pressure may block or interrupt a run. This result is needed before recommending a native production transport or larger model.

The Debian V100 and Windows RTX are separate devices. This implementation tests layer offload within each machine, not shared RAM/VRAM or distributed tensor inference over Tailscale. It does not repair the previously unreachable `192.168.0.61:11435` endpoint or establish current remote connectivity.

## Validation and limits

Full local suite: 1298 passed, 65 skipped, one existing PEFT Conv1D warning. Ruff, whitespace checks and Debian updater Bash syntax checks passed. A PowerShell runtime and the operator GPUs are unavailable in this workspace.

Local deterministic tests cover model-free native arguments, legacy profile compatibility, same-output selection, incomplete/wrong outputs, child-process cleanup, pressure interruption, model checksum refusal, independent cost accounting, combined-request routing and selected-curriculum exposure. Native V100/RTX performance, Windows execution and current financial outcomes require host results; they cannot be measured in this workspace.

No larger model is selected, no loss is hidden by changing costs, and no profitable strategy or accepted weight update is asserted. Existing web/sandbox tools, goals, balances, archives and Windows production policy remain available and preserved.

## Exact code locations

| Area | Files |
| --- | --- |
| Native n-gram serving and measurement | `rlm/v100/cli.py`, `rlm/v100/inference_benchmark.py`, `rlm/v100/chat_resources.py` |
| Curriculum and exposure | `rlm/v100/experiments.py` |
| Chat command routing | `rlm/v100/mission_chat.py`, `rlm/v100/resource_optimization.py` |
| Execution accounting and UI | `rlm/v100/backtesting.py`, `rlm/v100/dashboard_layout.py` |
| Debian update / Windows benchmark | `tools/update-synta-v77.sh`, `tools/benchmark-synta-windows.ps1` |
| Regressions | `tests/test_v100_inference_benchmark.py`, `tests/test_v100_action_objects.py` |
