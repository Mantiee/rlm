# Synta v78: recover dashboard collection and diagnose native probes

## Supplied host evidence

v77 installed successfully on Debian. All five inference variants were blocked with an undifferentiated JSON parser error, so no n-gram promotion or speedup was established. The mission supervisor restarted its process.

The dashboard service was listening on `192.168.0.68:8765`, while the v77 readiness checker queried `127.0.0.1:8765`. The dashboard collector thread also exited with `TypeError` when concatenating a dictionary stream fragment. The supplied screenshots show an initial/stale snapshot with missing mission data, not evidence that the saved goals, weights, ledgers or archives were deleted. A historical trade series remains visible.

## Changes

- Activity redaction can decode a complete JSON string into a sanitized dictionary or list. Agent views now render both text and structured fragments, including returned traces. Secrets remain redacted. Request boundaries and bounded stream previews remain intact.
- The dashboard collector catches malformed-data `TypeError` and retains an explicitly stale snapshot for the next cycle rather than losing its refresh thread.
- Readiness uses the configured private dashboard origin in `research/remote-access/config.json`, or the existing systemd service bind if that configuration is absent. It does not rebind the server or guess a new address. Both the expected HTML renderer and a fresh successful `/api/status` collector snapshot are checked. An initial page or stale snapshot is insufficient.
- An unavailable mission-running field displays `unknown`, rather than `stopped`. Missing trade timestamps display `timestamp unavailable`, rather than `Invalid Date`.
- The structured benchmark prompt now requests a native JSON schema. Completed probe responses are written to per-variant `*-probes.jsonl` files before correctness checks. Non-JSON HTTP responses and non-JSON model content get different errors with a bounded preview. Incorrect, empty and truncated results remain blocked. This fixes the probe interface and diagnosis; the previous receipts do not identify which JSON parser failed, and no host benchmark speedup is claimed.
- The targeted updater installs v78 and refreshes the actual copied host launcher `bin/serve-v100-dashboard.py`, preserving its backup. Earlier package-only updates could leave that launcher old. Mission and training services are not restarted; profiles, goals, balances, weights, Windows policy and archives are unchanged.

## Install

Run `tools/update-synta-v78.sh` with this release's exact SHA on an installed v77-v78 system. It replaces the package and the copied dashboard launcher, restarts only the dashboard and existing remote gateway, then checks private-address HTML and fresh collector data. On failure it prints dashboard service diagnostics. Reload the browser once and inspect that the collected timestamp advances.

This update deliberately does not launch another exclusive GPU benchmark while the mission or training may be running. The fixed benchmark can be rerun in a separately controlled idle period after the dashboard recovery result is known. Windows has not supplied a native benchmark result; production Windows n-gram transport is not enabled by this update.

## Exact locations

| Fix | Files |
| --- | --- |
| Structured stream fragments | `rlm/v100/live_status.py` |
| Collector recovery | `tools/serve-v100-dashboard.py` |
| Configured bind and fresh-snapshot checks | `rlm/v100/dashboard_health.py` |
| Unknown mission state and timestamp labels | `rlm/v100/dashboard_layout.py` |
| Structured inference probes and response evidence | `rlm/v100/inference_benchmark.py` |
| Targeted installer | `tools/update-synta-v78.sh` |
| Regression fixtures | `tests/test_v100_dashboard_recovery.py`, `tests/test_v100_strategy_recovery.py` |

Host model performance and Tailscale phone routing still need operator verification. These fixes do not establish profitable strategies or accepted learning updates.

## Validation

Full local suite: 1305 passed, 65 skipped, one existing PEFT Conv1D warning. Ruff, whitespace checks, Bash syntax and embedded updater Python parsing passed. Regression fixtures reproduce the actual redacting journal producing structured stream values, collector recovery after a TypeError, private LAN readiness with initial/stale/fresh snapshots, constrained JSON probe requests and preserved failed model responses. Host deployment, fresh collector output and native GPU timing still require operator results.
