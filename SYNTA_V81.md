# Synta v81: bounded master-chat context

Concrete failure: a master-chat request assembled 39428 prompt tokens for a 32768-token backend. Large host evidence/settings/plan snapshots were appended verbatim. The existing oversized-tool-catalog selector could not solve an oversized conversation.

Changes:
- `rlm/v100/mission_chat.py`, `operator_context`: preserve the authenticated current message verbatim; retain small snapshots intact and archive large host snapshots as hash-verified, explicitly incomplete previews. Full data remains readable through existing `read_tool_result`.
- `rlm/v100/research_tools.py`, `compact_research_messages`: bound previous tool-result data before every routing turn and final analysis, keep all tool call IDs/pairs, retain full original evidence on disk. Tool execution is not replayed to recover context space. System and user instructions are not silently truncated.
- `rlm/v100/mission_chat.py` and `chat_resources.py`: expose the existing `propose_colab_trial` capability to interactive/resource master chat. It creates a real bounded interactive notebook proposal and artifact receipts, not allocated GPU time. Account/session and provider admission are still needed. No paid fallback, quotas bypass or fictional free compute.
- `tools/update-synta-v81.sh`: includes prior v79/v80 package fixes and model-candidate/Wikipedia tools for Debian; copies the Debian qualification launcher. Preserves the current accepted model and its learning state.
- `tools/wait-update-synta-v81.sh`: wait up to 24 hours for exclusive training/preparation experiments to finish, checking every 30 seconds. Run the updater once afterward, with its own final exclusive-process guard. Never kill a training process. Can run in a separate transient user service across SSH disconnects.

This fixes the observed source of context growth, not an arbitrary oversized user instruction or a catalog with no representable capability. Genuine irreducible context failures remain explicit. Completed actions and original receipts are retained.

## Deployment

Download the pinned wait script to `$HOME/ai-v100/wait-update-synta-v81.sh`, then invoke it through `systemd-run --user --unit=synta-update-v81 --collect --property=Type=exec /bin/bash SCRIPT REV`. Inspect `journalctl --user -u synta-update-v81`. The existing trainer continues. After install use the same free-compute request again as a new explicit request; historical failed requests are not automatically replayed.

## Debian model status

The release contains `rlm/v100/model_candidates.py` and `tools/qualify-synta-models-debian.sh` from v80: download Qwen3.5 4B or 35B-A3B MoE, verify artifact SHA-256, compare native n-gram/KV/batch/thread/Flash Attention and expert offload, test 500K context. The accepted Gemma master is not claimed to have been replaced on the operator's machine. Actual host benchmarks and independent task-quality admission are pending, and Gemma's LoRA cannot be reused on Qwen.

## Validation

Targeted context/evidence/chat tests: 47 passed, including large snapshot archiving, unchanged operator instruction, preserved tool pairs and unchanged original evidence. Full regression: 1316 passed, 65 skipped, one existing PEFT Conv1D warning. Host deployment remains pending. Ruff/Bash syntax checks pass. Windows LAN helper remains independently blocked; no host networking repair or financial improvement is claimed.
