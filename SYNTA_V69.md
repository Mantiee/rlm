# Synta v69

Fixes v68 rejecting the built-in dashboard because its valid HTML omits the
optional body tag. Navigation placement now parses HTML, handles explicit and
implicit body content, and ignores fake body tags inside styles, scripts and
comments. It retains exact renderer script bytes and the upstream security
policy. Errors include their actual detail instead of hiding the cause.

Regression coverage proxies the actual built-in dashboard renderer through the
authenticated gateway, as well as body attributes, omitted body tags, Unicode
head content, misleading tags in metadata and idempotent navigation insertion.

Run `tools/update-synta-navigation.sh` with the published v69 commit on an
existing v67 or v68 installation. It restarts only `synta-remote.service`.
Before reporting success, it reads both authenticated local gateway pages and
requires HTTP 200 with their navigation links present.
Mission, Windows workers, dashboard and private Tailscale routing are retained.
The operator's actual Debian service and phone require post-update verification.

Validation: 1218 tests passed, 65 skipped, one existing PEFT warning. Ruff and
the updater's Bash syntax check passed.
