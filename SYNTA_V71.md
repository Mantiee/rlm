# Synta v71: resource optimization and truthful execution status

An operator request to optimize CPU/RAM/resources now executes a host-owned operation,
rather than asking the model to select a backtest. This includes mixed questions and
commands such as “CPU Debian or Windows? Increase usage, do it.” Explicit follow-ups
such as “do all of it” or “error, continue” resolve prior operator messages, not assistant
claims. An intervening unrelated request ends that binding.

The controller writes `research/state/resource-policy.json`, archives the previous policy
and receipts under `research/resource-optimization`, and reports the measured current
admission. Debian admits up to four CPU/network jobs, bounded by affinity and cgroup v2
CPU quotas, two threads per slot and a CPU reserved when possible. Concurrent owned Python experiments receive distinct CPU affinity slots instead of
all being pinned to the first two CPUs. Each result and live drone status records its
assigned CPUs. The tuned policy
reserves 4 GiB available RAM plus 3 GiB per slot. Host CPU at 65% reduces concurrency
and 85% stops new admission. Existing bounded jobs can finish. One model job remains
allowed; six executor workers accommodate four CPU tasks, the model and the observer.

Windows affinity/RAM and adaptive RTX guards are retained. No board power cap, GPU
clock or quality gate is changed. With a fresh idle owned CPU worker, the controller can
queue the existing verified goal trial under the shared dispatch lock. Its existing label,
source split and independent evaluation requirements still apply. Missing labels report
a blocker instead of launching synthetic busywork. Existing owned jobs retain priority.
No measured speedup, profit, continuous training or crash immunity is claimed.

Missing archived tool results return `archive_not_found` with the requested digest and
recovery guidance. No missing content is invented. A chat whose tools all failed is
stored as `failed` with its response and receipts; an empty reply is rejected. Successful
partial work retains its receipts. No automatic rerun of mutations is introduced.

## Install

Use `tools/update-synta-resources.sh` with this release's pinned commit on Debian.
It accepts v67-v71, requires the owned user service, retains the current accepted/live
profile and goals, checks for exclusive experiments, stops the owned mission, installs
v71, restarts it through the existing supervisor and applies the resource policy. It does
not reinstall Windows, reset the ledger or restart Tailscale routing. Inspect the printed
resource receipt for actual slot counts and Windows blockers. Windows does not need
another installer for this change.

Hardware throughput and responsiveness must be measured on the deployment. This
release does not promise that every arbitrary optimization is implemented, nor does
CPU concurrency alone improve GPU inference. Accepted model settings are unchanged.
