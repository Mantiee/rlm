# Dedicated Debian RTX 3060 Ti node

This node is explicitly operator-owned: `192.168.0.9`, controller `192.168.0.68`. Install the worker on `.9`, register it on the existing Marek controller. Do not run PowerShell in Bash. Existing V100 training is preserved; package updates defer until its exclusive job finishes.

## Worker

`bootstrap-synta-3060.sh REV` runs with sudo on the 3060 Ti computer. It requires an existing working NVIDIA driver, x86_64 Debian, the exact `.9` address and 16 GiB free disk. It does not replace drivers, change clocks, partition disks, expose a remote shell or grant the worker sudo access.

It creates a non-login `synta3060` account, a separate `/opt/synta3060` Python environment and root-owned versioned source/runtime. Runtime Ollama 0.40.0 is verified against the publisher's release SHA256. GPU models are Qwen3.5 2B/4B Q4, loaded on demand, one at a time. gpt-oss remains excluded.

System services use `ProtectSystem=strict`, `ProtectHome=yes`, no capabilities, no new privileges and a closed NVIDIA device policy. Only `/var/lib/synta3060/work` and the dedicated Ollama cache are writable by their respective services. They share a 90% RAM slice, with no swap allowance; GPU service receives at most 50% and CPU/proxy service at most 40% of host RAM. CPU scheduling has no artificial duty cap. The CPU pool admits as many two-thread fixed-kernel jobs as cores and its memory allocation allow, with a 768 MiB per-job RSS limit and a 150 second wall deadline. The system keeps a RAM reserve, and sampled GPU temperature at 80 C blocks/cancels owned inference. This is process/filesystem isolation on the host kernel, not a claim of VM-level isolation or immunity to hardware failure.

Ollama listens only on `127.0.0.1:11436`. The `.9:11435` proxy permits only the specified controller and local requests. It exposes text inference, inventory, telemetry and the frozen GRU/Transformer CPU kernel. Model management, received code and arbitrary shell commands are not exposed. The operator's root installer manages model acquisition through loopback.

The installation checks actual loaded model GPU residency and 32768 context before marking installation prepared. The controller separately checks every arithmetic/accounting/evidence panel case before advisory routing. No broad intelligence or income claim follows from that small panel.

## Controller and independent recovery

`attach-synta-3060.sh` creates the user service `synta-owned-nodes.service` and registers the exact peer. It keeps an existing compute mailbox or creates a dedicated local one. CPU bytes travel through the scoped node API and normal host safe-tensor replay still precedes any acceptance. Source groups, original inputs, nonce and artifact hashes are retained. Remote scores remain untrusted.

Telemetry and CPU transport poll every 20 seconds; GPU qualification runs independently. A missing GPU does not stop CPU transport. CPU transport/admission errors do not disable a healthy qualified GPU. A missing node records an unavailable/stale dashboard state and leaves other devices running. A qualified node serves fast resident drones and can serve the existing chat fallback while the accepted V100 is occupied. It does not replace accepted V100 training weights. Different physical nodes hold independent inference leases.

Root system services start after reboot without interactive login. The controller user service follows the controller's existing user-manager boot/linger policy. Neither the node nor this installer provides a remote shell for ChatGPT.

## Shared code and updates

All authorized updaters follow `Mantiee/rlm`, branch `v100-rtx-reconnect`, resolve an exact commit and verify release-file hashes. This shares versioned source, not private chats, accounts, RAM or VRAM. Each computer keeps its own data and weights. The node keeps a pinned inspection snapshot under `/var/lib/synta3060/research/managed-source/REV`, separate from the immutable executable release. The node root updater retries every 15 minutes, deferring restart while CPU or GPU jobs run. Incomplete first installations are retried; failed code replacements attempt to restore the previous source pointer and owned services. It does not alter unrelated services.

Useful receipts on the worker:

* `/var/lib/synta3060/installation.json`
* `/var/lib/synta3060/managed-update.json`
* `/var/lib/synta3060/work/resource-status.json`
* `journalctl -u synta3060-node.service`

On the controller: `~/ai-v100/research/owned-node/status.json`, `qualification-state.json`, and the compute job archives. Published code and passing mocked tests are not proof that the operator's node has booted, passed residency checks or earned income.
