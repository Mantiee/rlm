# Synta v79: availability recovery and disk-backed knowledge retrieval

Implemented:
- `rlm/v100/mission_chat.py`: backend admission failures keep requests queued with bounded retry pacing. Other eligible requests can proceed. `/cancel-chat ID` cancels a queued request without requiring a model. Failures after an inference/tool execution starts are not blindly replayed.
- `rlm/v100/live_status.py`, `dashboard_layout.py`, `remote_access.py`: unavailable RTX is visible in the dashboard header; failed and waiting chat states are distinct.
- `tools/repair-synta-proxy-firewall.ps1`: verify the live owned proxy executable before adding a source-address-scoped rule for the actual Python executable. Does not disable firewall/VPN or expose Ollama broadly.
- `tools/start-rtx3090-helper.ps1`: account for Windows venv Python redirector in future starts.
- `rlm/v100/inference_benchmark.py`: up to 60 seconds of admission cooldown between variants; preserve measured baseline and partial results if pressure persists. No weakened thermal/VRAM limits and no automatic speedup claim.
- `rlm/v100/knowledge_rag.py`: streaming XML/bz2 Wikipedia import, disk-backed SQLite FTS5 retrieval, bounded passages, source URL and revision attribution. Indexed replacement prevents duplicate reimports. Import commits every 100 articles and pauses below 4 GiB free disk. Wikitext is archived untrusted knowledge, not executable instructions or verified commercial facts. No GPU embeddings or added package dependencies.
- `rlm/v100/research_tools.py`: `search_knowledge(query)` available to research tools and interactive master chat.

## Deployment

Run `tools/repair-synta-proxy-firewall.ps1` in elevated Windows PowerShell. Then verify from Debian with `curl --noproxy '*' --connect-timeout 5 --max-time 10 http://192.168.0.61:11435/api/tags`. This is not verified by the repository tests.

Run the pinned `tools/update-synta-v79.sh REV` on Debian. It preserves the accepted profile and refuses to interrupt an exclusive training experiment. It restarts the mission to load changed queue code and verifies the fresh dashboard collector at its configured address. It does not run GPU benchmarks while mission work owns the device.

## Wikipedia import

Obtain a public Wikipedia content XML export from https://dumps.wikimedia.org/ and retain its attribution/licensing metadata. Full corpus download/indexing requires substantial disk space and time; neither is performed by the updater. Import a pilot first:

```bash
"$HOME/ai-v100/venvs/v100-continual/bin/python" -m rlm.v100.knowledge_rag \
  --root "$HOME/ai-v100" --dump /absolute/path/to/wiki.xml.bz2 \
  --language pl --max-pages 10000
```

Use the corresponding language code. Raise `--max-pages` for the full export after measuring host disk, index size and retrieval latency. Query via `search_knowledge` in master chat. This importer indexes current main-namespace nonredirect wikitext pages. It does not render templates, index images, teach model weights, or establish investment skill.

## Context and model qualification

Current Gemma 4 12B's documented maximum is 256K tokens (https://ai.google.dev/gemma/docs/core/model_card_4). 500K is not enabled. Current accepted context is retained, not automatically doubled while training is active.

No larger/MoE model, task specialist, Engram layer, distributed VRAM pool or new GPU kernel is promoted by this release. Existing pinned helper identity guards remain intact. Separate models may improve role quality, but must be measured on actual hardware and independent tasks. Engram is a trained architecture, not a switch that installs Wikipedia into existing Gemma weights. Native n-gram decoding and CPU offload remain benchmark candidates; no measured gain has yet been established on the user's machines.

## Validation

Full Python suite: 1306 passed, 65 skipped, one pre-existing PEFT Conv1D warning. Two additional cooldown tests passed along with the RAG test. Ruff and Bash syntax checks passed. Windows PowerShell execution, whole-Wikipedia scale, GPU throughput and LAN connectivity require host receipts. No financial profit or accepted LoRA improvement is claimed.
