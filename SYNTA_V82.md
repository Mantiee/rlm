# Synta v82: updater recovery after exit 127

Observed host failure: transient systemd updater preserved the profile and stopped the mission, then exited 127. Package remained v78. Exit 127 means a command was unavailable; bare `uv` in the updater and the transient service's PATH are the leading diagnosis, not a proven full journal trace.

Changes in `tools/update-synta-v82.sh`:
- Add `$HOME/.local/bin` to PATH and resolve executable `uv` to an explicit path. Run `uv --version` and check Python/git/curl/systemctl before any service change.
- Permit recovery of a previously stopped updater, using its preserved profile under `research/availability-recovery-update` after validating its path. Do not require the service to be active in this recovery case.
- Save an invocation-specific restore marker before requesting a stop. On subsequent failure, if the mission is stopped, submit the preserved profile to `supervisor.install`; if already running, leave it alone. Preserve original error status.
- A recovery attempt is not a package rollback. If package imports or supervisor restart fail, print an explicit error and require journal inspection.
- Retain training exclusivity guard, prior v79-v81 fixes, model qualification tools and fresh dashboard verification. Correct the installed-version success message.

`tests/test_v100_updater_recovery.py`: real Bash execution with a simulated host proves that missing uv never invokes the mission-stop Python, that ~/.local/bin/uv works without an interactive shell, and that failed package installation calls the recovery path and preserves its error status.

Validation: two new recovery tests passed; Bash syntax and all embedded Python heredoc syntax checked; Ruff/diff checks passed. Prior v81 full Python regression: 1316 passed, 65 skipped, one existing PEFT warning. Live host install/restart still requires receipts. This release does not claim a new model was promoted, a GPU speedup, income, or a repaired Windows LAN path.
