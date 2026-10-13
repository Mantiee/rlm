# Private Synta control plane

Run `setup-synta-private-sync.sh` once on the V100 controller with an immutable published release. It uses GitHub CLI device login on the owned computer. No token is pasted into chat, stored in the repository or distributed to helpers.

The installer creates **Mantiee/synta-control** with private visibility, unless an existing private repository is available. A newly created repository has Actions disabled; public release CI remains the deployment validator. Grant the ChatGPT GitHub connection access to this new repository once.

Every five minutes the controller collects all three owned-host APIs and uploads a bounded, credential-redacted report to `hosts/latest.json`. Missing peers are recorded as unreachable; their absence does not stop other peers or the mission. Full home directories, datasets, model weights, credentials and raw chat history are not uploaded.

The `source` branch mirrors the public `Mantiee/rlm` deployment commit. It never force-overwrites private source edits. This is a private source copy, not a migration of package distribution away from the existing public release channel. Proposed private changes need publication and validation in that release channel before they can become a deployed commit.

`control/approved-release.json` on the private repository's default branch selects updates:

```json
{
  "schema": "synta-approved-release-v1",
  "repository": "Mantiee/rlm",
  "mode": "follow-published"
}
```

For a specific already validated release, set `mode` to `pinned` and add its exact 40-character `revision`. The controller publishes this bounded selector through `/api/release`; the Windows and 3060 Ti updaters read it without needing the controller's GitHub credentials. If the controller is offline they retain a last known pin or continue the existing public channel when none was configured. Current tasks and accepted weights are not changed by telemetry sync.

Local status: `~/ai-v100/research/private-sync/status.json`.
Service: `synta-private-sync.service`. Timer: `synta-private-sync.timer`.

This provides a shared code copy, diagnostics and release selection. It does not create a remote shell, change a model foundation, certify profits or run this ChatGPT assistant continuously outside its active sessions.
