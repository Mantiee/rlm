# Synta v68

The private remote dashboard and master chat share a mobile navigation bar.
Dashboard link injection now handles body attributes, uppercase tags and quoted
attribute values containing `>`. It preserves the host renderer, script bytes
and Content Security Policy. Authentication and command routing stay unchanged.

For an existing v67 installation, run `tools/update-synta-navigation.sh` with
the published v68 commit. It replaces the package and restarts only
`synta-remote.service`. Production changes since v67 are limited to the remote
gateway. The mission, Windows workers, dashboard and Tailscale Serve settings
are retained. Older installations require the full retention upgrader.

HTTPS opens the dashboard or `/chat`. To open a shell on an iPhone, configure
an SSH client with host `debian1.tail68f677.ts.net`, port `22`, username `marek`
and the existing Debian SSH credentials. Keep Tailscale connected. Do not put
`https://` or `/chat` in the SSH host field. Existing OpenSSH access is separate
from the HTTPS gateway; no public port forwarding is required.

Regression coverage includes authenticated proxy navigation, unchanged script
bytes and security policy, body tag variants and idempotent insertion. This
environment cannot verify the operator's phone or Debian services directly.

Validation: 1210 tests passed, 65 skipped, one existing PEFT warning. Ruff and
the targeted updater's Bash syntax check passed.
