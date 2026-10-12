# Synta managed rollout (v83)

The bootstraps install an operator-authorized updater on each existing owned host. They fetch the `v100-rtx-reconnect` GitHub channel, resolve it to an immutable commit, verify the release artifact SHA256 values and preserve model/goal/accounting evidence. They do not create a remote shell for ChatGPT.

## Order and prerequisites

Run the Windows bootstrap first, in elevated PowerShell under the existing helper account. Run the Debian bootstrap second under `marek`, without sudo. Both require the existing Synta Python environments. The Windows task runs under the current interactive user and requires that user to be signed in. Debian uses a user systemd timer. A disconnected SSH session does not stop its task. Existing linger/boot policies are not changed.

Automatic checks occur every six hours on Windows. Debian checks every 15 minutes and avoids repeating a successfully prepared release within six hours. An active exclusive train causes a deferred package update. The dashboard shows updater, asset preparation, RAG, admitted roles and native qualification receipts.

## What executes

* Windows keeps the pinned 9B baseline and downloads Qwen3.5 2B Q4, 4B Q4 and 35B-A3B Q4, retaining already downloaded identities. Pulls use the owned loopback Ollama, not a paid provider.
* The existing Tailscale peer named `debian1` can reach the existing Windows admission proxy through port 11437, restricted to the discovered Debian tailnet address. The proxy remains the resource admission controller. LAN rules are retained. Failure remains visible; no general firewall/VPN disabling occurs.
* Debian installs v83, preserves its profile and resumes the mission. It resumes/starts the hash-verified English Wikipedia importer unless an importer is already running or complete. The importer has two download/prefetch slots and one index writer. Indexed batches become queryable during the remaining import.
* RAG automatically contributes at most two 800-character excerpts to research tasks. These are archived data with source/revision references, not training approval or current market pricing.
* Debian acquires hash-pinned native 4B/35B GGUF candidates. A library-supported Qwen4B foundation trial is queued for the continuous learner, with a separate adapter and retained task/fresh/public gates. A different foundation never inherits a Gemma adapter.
* Windows roles are evaluated against the pinned 9B with twenty arithmetic, accounting and evidence cases. Fast roles need no measured accuracy loss, every critical case passing, and at least ten percent faster responses. The planner needs no regression and a measured accuracy or speed improvement. This is a limited admission panel, not broad intelligence certification. The report contains every observed answer and elapsed time.
* A bounded independent CPU n-gram memory pilot runs on frozen goal inputs with memory enabled/disabled comparison. It does not retrofit pretrained Gemma/Qwen weights and is not the published DeepSeek Engram architecture.
* A Debian native maintenance window waits for the CUDA lease and active exclusive training to finish. It temporarily pauses the owned mission, benchmarks the accepted GGUF and native candidates, and resumes the preserved profile even after a benchmark error. A faster setting for the same accepted model is activated only after the full current development suite and retained case gates pass. Candidate architecture replacement remains a separate foundation trial.
* Native comparisons cover n-gram speculation, q8 KV cache, CPU offload and supported CPU-MoE/threads/batches/Flash Attention flags. These are runtime configuration experiments, not new CUDA kernels or driver replacements. Unsupported flags produce blocked results.
* The 524288-token Qwen trial must complete an actual 500000-token prefill and marker retrieval. It does not silently increase the current Gemma learner's context. A synthetic prefill pass is not broad long-context task qualification. Windows Ollama does not inherit native llama.cpp n-gram flags.

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
