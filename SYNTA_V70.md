# Synta v70: native startup verification

An operator stack trace showed the native server already `ready`, while the mission remained
in `loading-context` inside `assert_served_expert -> file_hash`. The launch receipt had already
hashed the model. The admission check repeated that full read.

- Cache SHA-256 results only in the current Python process, keyed by resolved path, device,
  inode, size, modification time and change time. Restarting starts with an empty cache.
- Reject a file changing during hashing. Restoring its modification time does not restore
  its change time. Native receipts also check these identities, process start time, expected
  expert hash, execution settings and the endpoint's actual model path.
- Artifact verification reports the current file and bytes read in status and the dashboard.
  After native readiness the mission shows `verifying-serving-identity` while admitting it.
  Startup cancellation,
  native process exit and the existing startup deadline apply during verification too.
  A daemon verifier may remain inside a blocked OS read until that read returns; the mission
  stops waiting and stops its own server. The verifier checks cancellation before publishing
  its receipt; a receipt for an exited server cannot pass admission.

The targeted `tools/update-synta-startup.sh` updates v67-v70 and restarts only the owned
mission using a saved copy of its current accepted/input profile, then reloads an active
dashboard service to show verification progress. It preserves goals,
Windows settings, archived runs, model files and checkpoints. It does not install a new
Windows policy or rerun setup. Existing processes must restart to load this fix.

Zero GPU utilization while reading files is expected. Utilization after admission depends
on actual inference or training work. This patch does not establish learning or income.
