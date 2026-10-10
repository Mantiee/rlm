# Private remote Synta

Run `tools/install-synta-remote.sh` as the existing Debian operator after updating Synta. It installs Tailscale from its official signed Debian repository if absent. Login is an operator step: open the printed link, then use the same account on the phone/laptop. If HTTPS needs enabling, Tailscale Serve prints its authorization link.

The script prints the private dashboard, `/chat` and SSH address. Tailscale provides connections outside the LAN without router port forwarding. SSH uses the existing Debian SSH credentials. Install Tailscale separately on another home computer to access that computer; this installer does not expose the whole LAN or enable Windows RDP.

The web gateway binds only `127.0.0.1:8786`. Tailscale Serve terminates HTTPS and supplies verified identity headers. Every route requires the configured owner login. Chat writes require the exact HTTPS Origin and a per-process CSRF token; only the existing bounded mission queue is writable. Dashboard proxy routes and body sizes are restricted. There is no public Funnel, arbitrary URL proxy, browser terminal or unauthenticated command endpoint. Existing Serve and Funnel configuration conflicts are preserved and reported.

Browser chat keeps actual saved request states and replies visible across reconnects. It supports the same goal and work commands as the terminal. A submitted command is not a completed action. `/status` is read from host evidence by the mission service. Authentication, ACLs, HTTPS authorization and the first real phone connection must be completed/verified on the operator's devices.

Disable the gateway with `systemctl --user disable --now synta-remote.service`. Disable only its Serve listener with `sudo tailscale serve --https=443 off`; do not reset other Tailscale services.

Official references: [Serve and identity headers](https://tailscale.com/docs/features/tailscale-serve), [packages](https://pkgs.tailscale.com/stable/), [SSH access](https://tailscale.com/docs/features/tailscale-ssh).
