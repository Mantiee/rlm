# Synta v80: large-model and long-context qualification

This release provides executable candidate acquisition and experiments. It does not claim that a candidate is already running on the operator's hosts, faster, financially useful, or admitted as the master.

Changes:
- `rlm/v100/model_candidates.py`: small Qwen3.5 4B Q4_K_M and larger Qwen3.5 35B-A3B MoE Q4_K_M candidates. Resolve repository revision and publisher LFS SHA-256 before downloading; verify cached/downloaded bytes; retain immutable artifact receipts. Require artifact size plus 8 GiB disk headroom. Models are separate from accepted Gemma and its LoRAs.
- `rlm/v100/inference_benchmark.py`: capability-checked YaRN flags for 524288 context, CPU MoE experts, n-gram, Q8 KV cache, batch/thread/Flash Attention comparisons. A long-context candidate must process at least 500000 prompt tokens and retrieve three markers. Allocating KV cache is not success. This test is synthetic and does not certify complex reasoning.
- `rlm/v100/wiki_ingest.py`: acquire a completed, dated official Polish or English Wikipedia export, verify publisher checksum, then stream all current main-namespace nonredirect pages into sourced FTS5 RAG. Requires disk capacity before download. Whole corpus throughput and publisher endpoint availability remain host checks. Current official export-format changes fail explicitly rather than ingesting wrong data. Export receipt and licensing attribution are retained. No weight training approval from unverified Wikipedia text.
- `tools/qualify-synta-models-windows.ps1`, `tools/qualify-synta-models-debian.sh`: executable host experiments. Production services are not silently replaced. Windows uses the already installed pinned b10683 CUDA12.4 runtime; Debian uses its existing V100 runtime and rejects exclusive active mission/training work.
- `tools/update-synta-v80.sh`: preserve accepted profile and activate package changes when no exclusive experiment is running. Includes v79 queue/RAG/dashboard fixes for a host still on v78.

## Host use after package installation

Debian:
```
$HOME/ai-v100/venvs/v100-continual/bin/python -m rlm.v100.wiki_ingest --root "$HOME/ai-v100" --language pl
bash "$HOME/ai-v100/qualify-synta-models-debian.sh" fast 32768
bash "$HOME/ai-v100/qualify-synta-models-debian.sh" planner 524288
```
The qualification launcher must be downloaded from this exact revision first. GPU experiments require an idle mission after current training finishes. Do not kill a training process to qualify a candidate. Wikipedia may require many hours and substantial disk space; retrieval is not model pretraining.

Windows launcher parameters: `-Revision REV -Role fast -Context 32768`, then separate `-Role planner -Context 524288`. It downloads separate candidates and writes benchmark results under `ai-v100-helper/performance`. The 35B model can exceed available RTX VRAM, especially alongside existing services. Expert offload also requires host RAM and may be slower.

## Remaining admission blockers

LAN helper TCP still timed out after the actual-Python firewall rule. That rule is not claimed to solve the remaining route/VPN/firewall problem. Current trainer PID receipt means updater did not install v79. Host execution cannot be verified from the repository environment.

A master switch requires independent tool-use/goal-task quality, context correctness, memory admission and a training model/base alignment plan. Gemma LoRA weights cannot be attached to Qwen. Native eight-probe throughput is insufficient evidence to auto-promote a new master, and no such promotion occurs.

## Engram

Official DeepSeek Engram repository currently exposes a standalone demonstration with mocked backbone components (https://github.com/deepseek-ai/Engram). It is not an available pretrained Gemma/Qwen extension. Existing `rlm/v100/ngram_memory.py` is a separately trainable bounded causal-memory pilot, exposed through `propose_ngram_memory` and existing candidate tests. It remains a pilot, not the official Engram-27B, and is not relabeled or substituted for a trained master. Native n-gram speculative decoding is a separate implemented experiment.

## Validation

Full suite after model/benchmark changes: 1311 passed, 65 skipped; existing PEFT Conv1D warning. Added official export/checksum and measured-context gate tests pass. Ruff/Bash checks pass. PowerShell execution, download publisher endpoints, whole corpus scale, model kernel compatibility, 500K prefill and speedups need actual host receipts. No profit guarantee or accepted weight improvement is claimed.
