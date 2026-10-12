# Synta managed rollout (v87)

The bootstraps install an operator-authorized updater on each existing owned host. They fetch the `v100-rtx-reconnect` GitHub channel, resolve it to an immutable commit, verify the release artifact SHA256 values and preserve model/goal/accounting evidence. They do not create a remote shell for ChatGPT.

## Order and prerequisites

Run the Windows bootstrap first, in elevated PowerShell under the existing helper account. Run the Debian bootstrap second under `marek`, without sudo. Both require the existing Synta Python environments. The Windows task runs under the current interactive user and requires that user to be signed in. Debian uses a user systemd timer. A disconnected SSH session does not stop its task. Existing linger/boot policies are not changed.

Automatic checks occur every 15 minutes on both hosts. Both avoid repeating a successfully prepared release within six hours. Partial preparation is retried after fifteen minutes. A Wikipedia or helper failure does not abort the other preparations. An active exclusive train causes a deferred package update. The dashboard shows updater, asset preparation, RAG, admitted roles and native qualification receipts.

## What executes

* Windows keeps the pinned 9B baseline and downloads Qwen3.5 2B Q4, 4B Q4 and Qwen3.6 35B-A3B Q4, retaining already downloaded identities. Pulls use the owned loopback Ollama, not a paid provider.
* The existing Tailscale peer named `debian1` can reach the existing Windows admission proxy through port 11437, restricted to the discovered Debian tailnet address. The proxy remains the resource admission controller. LAN rules are retained. Failure remains visible; no general firewall/VPN disabling occurs.
* Debian installs v87, preserves its profile and resumes the mission. It resumes/starts the hash-verified English Wikipedia importer unless an importer is already running or complete. The importer has two download/prefetch slots and one index writer. Indexed batches become queryable during the remaining import.
* RAG automatically contributes at most two 800-character excerpts to research tasks. These are archived data with source/revision references, not training approval or current market pricing.
* Debian acquires hash-pinned native Qwen3.5 4B/Qwen3.6 35B GGUF candidates. A library-supported Qwen4B foundation trial is queued for the continuous learner, with a separate adapter and retained task/fresh/public gates. A different foundation never inherits a Gemma adapter.
* Windows roles are evaluated against the pinned 9B with twenty arithmetic, accounting and evidence cases. Fast roles need no measured accuracy loss, every critical case passing, and at least ten percent faster responses. The planner needs no regression and a measured accuracy or speed improvement. This is a limited admission panel, not broad intelligence certification. The report contains every observed answer and elapsed time.
* A bounded independent CPU n-gram memory pilot runs on frozen goal inputs with memory enabled/disabled comparison. It does not retrofit pretrained Gemma/Qwen weights and is not the published DeepSeek Engram architecture.
* Automatic preparation keeps the active master/chat running. It does not start a native maintenance window or pause the mission for long benchmarks. Native maintenance now requires a separate explicit window; previous hardware results remain recorded.
* RAG uses two bounded read workers, a 600 ms caller budget and a read-only SQLite connection. Index activity cannot hold the chat indefinitely. Committed checkpoints and unchanged article revisions are reused after importer restart; updates receive new provenance. The index emits a separate live progress receipt.
* The operator excludes gpt-oss. Neither Windows acquisition nor role qualification includes it. Its clients and previously admitted role profiles are rejected after this package update. Existing downloaded files and historical receipts are retained. Qwen3.6 35B-A3B is a planner candidate and requires the same measured quality/speed gates before use.
* Both hosts receive a pinned public source snapshot. Debian exposes it to existing source-inspection and isolated code-proposal tools. The model may develop copies in its private Linux workspace; these proposals are not silently installed in the controller.

* Native comparisons cover n-gram speculation, q8 KV cache, CPU offload and supported CPU-MoE/threads/batches/Flash Attention flags. These are runtime configuration experiments, not new CUDA kernels or driver replacements. Unsupported flags produce blocked results.
* An explicitly scheduled 524288-token Qwen trial must complete an actual 500000-token prefill and marker retrieval. It does not silently increase the current Gemma learner's context. A synthetic prefill pass is not broad long-context task qualification. Windows Ollama does not inherit native llama.cpp n-gram flags.

## Recover a currently paused mission

`tools/recover-synta-live.sh REV` stops only the owned managed updater, resumes the preserved accepted mission when no exclusive training is active, installs v87 and restores the updater. It also retires the explicitly named old standalone Wikipedia units so the managed importer can resume their archived parts and committed index with the new code. It preserves checkpoints and does not kill an active train. Use it under the existing Debian operator account. It is a recovery/update operation, not evidence that model generation or remote networking passed.

## Receipts

Debian:

```
~/ai-v100/research/managed-update/status.json
~/ai-v100/research/managed-update/preparation.json
~/ai-v100/research/managed-update/native-qualification.json
~/ai-v100/research/model-roles/active.json
~/ai-v100/research/knowledge/import-status.json
```

Windows:

```
%USERPROFILE%\ai-v100-helper\logs\managed-update.json
%USERPROFILE%\ai-v100-helper\logs\managed-models.json
%USERPROFILE%\ai-v100-helper\logs\tailnet-helper.json
%USERPROFILE%\ai-v100-helper\logs\managed-native.json
```

From Synta chat, `managed_upgrade_status` reads receipts and `request_model_preparation` queues the installed Debian updater. Actual training, qualification and profit require their own receipts. A downloaded model, a queued request, a completed process or a loss backtest is not verified income.

## Material limits

No speedup, profitable strategy, accepted new weights, complete Wikipedia import or working tailnet route is certified before the host receipts exist. Free external computing still requires the provider's authorized access and quota. This updater does not acquire third-party accounts, bypass quotas, pool RTX/V100 VRAM over LAN or supply credentials to ChatGPT. Existing web drafts and account-specific operations remain subject to their concrete authorization mechanism.

## Dedicated Debian RTX 3060 Ti

The operator-owned `192.168.0.9` node has its own protected system services, code updater and fixed CPU job pool. Setup and verification are documented in `tools/SYNTA-3060-NODE.md`. Controller registration does not stop training; its worker coordinator starts after the deferred v87 package installation. Source is shared by immutable GitHub revision; private data, sessions and GPU memory are not pooled.
