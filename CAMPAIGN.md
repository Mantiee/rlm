# V100 campaign v28

Use `tools/upgrade-v100-campaign.sh RELEASE_COMMIT` on Debian after any existing training/calibration/MTP sweep completes. It updates only the existing isolated continual venv, repairs the launcher atomically, retains checkpoints, prepares optional components and starts the owned supervisor. Preparation prints each stage and records failures as `deferred`; a deferred feature is not operational.

## Operating from one console

- `~/ai-v100/bin/v100-continual chat`: persistent chat while research continues. During exclusive V100 experiments the pinned RTX researcher may answer as a visibly identified delegate. Model requests remain subject to server queues and configured resource budgets.
- `/cel Your long-term goal`: explicit user authorization to change the long-term objective. Model tools may change short/mid plans, but cannot change that objective themselves.
- `mission-report`: accepted weight updates, A/B lineages, paper results, plans, drone queue and benchmark status.
- `mission-watch`: model/tool events. Ctrl+C in chat/watch does not stop the mission.
- `mission-stop`: intentional pause; supervisor respects it and checkpoints remain.
- `supervisor-start`: resume. User service can start after boot only when user lingering is enabled; detached fallback survives SSH disconnect but not reboot.

## Implemented mechanisms

A/B keep separately accepted adapters and serve their own accepted versions sequentially on the V100. Verified source/CPU jobs and one bounded RTX job run concurrently with local work. Persistent queues resume after restart. New adapters, merged children and algorithm candidates remain unpromoted until their relevant gates pass. Retrospective backtest labels include losses and independently recomputed accounting; they do not establish future profitability or constitute profit-policy reinforcement learning.

Chat/master can schedule source jobs, research/critic jobs, bounded CPU experiments, desktop experiments and public benchmark jobs. It can read its pinned full source, copy it inside the guest and propose changes. Resource controls can adjust thinking/output/batch within validated budgets. RTX request pacing is not a hard GPU utilization, transient power or VRAM cap. Existing Windows game guard pauses the helper during League; recurring black screens require hardware diagnosis independently of this software.

## Private Linux desktop

Optional rootless QEMU guest: 2 CPU cores, 3 GiB RAM, 24 GiB persistent virtual disk, no host home/model/credential mounts or GPU passthrough. A read-only source ISO exposes its own code; guest root can install tools and create files inside its disk. Public HTTP(S) egress is brokered and private/loopback destinations are rejected. GUI tools use the guest display; image analysis uses the pinned Qwen vision helper. RAM/disk pressure stops only this guest and retries later. CPU-only emulation can be slow when KVM is unavailable.

Initial preparation downloads a Debian cloud image and private signed Debian packages; first boot additionally installs the desktop. Readiness is checked through key-pinned SSH, the cloud-init completion marker, mounted source and the guest GUI service/display. A running QEMU process alone is not reported as ready. GPU inference, actual VM boot and Windows PowerShell monitoring require validation on the operator's hardware. No sandbox can promise protection against every kernel/hypervisor vulnerability.

## Benchmarks and learning limits

Separate pinned LiveBench environment, official grader and dataset snapshots. Default balanced 20-case non-coding panel covers five categories. Reports compare previous accepted versions and published model results only where matching cases are available. This panel is not the full official leaderboard or Artificial Analysis index, and published providers may use different inference budgets. A failed scorer is an infrastructure error, not a zero model score. Benchmark material is not fed to research memory or training.

Immutable originals, replay, preservation losses, independently verified examples, fixed regression suites and rollback reduce forgetting; finite tests cannot prove universal zero forgetting. Accepted updates are counted explicitly; a research cycle or saved hypothesis is not a weight update. Paper outcomes remain distinct from exploratory backtests. Real order execution is not installed.

Colab remains an operator-started bounded notebook pilot with pinned input/result manifests and local reproduction before promotion. No automatic free-tier distributed workers, account/cookie quota rotation or paid API calls are implemented.

## Windows visibility

`tools/watch-rtx3090-helper.ps1` shows whole-board GPU readings, CPU/RAM, loaded helper and recent server logs. Its readings are not per-process power measurements. Ctrl+C stops only the viewer.

## v28 reliability fixes

Resident worker crashes requeue interrupted bounded jobs and restart after cooldown; a full persistent queue does not block mission startup. SQLite handles close after transactions. The supervisor user service sources the isolated environment and refreshes its owned predecessor on upgrade. `/status` in chat returns locally without inference. Replies identify V100 or the RTX delegate. Pinned complete public baselines are reused only for identical weights, datasets and generation; measured MTP can be adopted without replacing accepted training ancestry.

Full requirement and evidence audit: [REQUIREMENTS.md](REQUIREMENTS.md).
