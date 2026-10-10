# Synta v66

The host dashboard now owns a Trade explorer. It reads the latest 600 immutable
paper events and the latest four independently replayed historical tests. It
shows equity, net returns, execution prices, costs and individual receipts.
Paper and historical samples are distinct; a portfolio without recorded fills
has no fabricated markers. Historical capital uses normalized units, not PLN.
Missing or inconsistent sources produce explicit errors and are excluded.

Selection, zoom sliders and opened receipts survive refreshes. The host API
serves cached chart data, including through the authenticated Tailscale gateway.
A visualization request in chat uses the controller, without a model tool loop
or VM SSH. Upgrade retires request 560e2c23686e46078fa691b7f505767a, preserves its
original database row in an archive, and leaves other requests alone.

Windows has an explicit adaptive CPU/RTX installer. Browsers are allowed.
CPU jobs remain at idle priority, one child, at most two affinity slots and
4 GiB RSS. Load changes affinity; pressure suspends/resumes progress. Critical
free RAM below 2 GiB releases the child; 150 active seconds and ten wall minutes
bound each experiment. Unfinished work cannot be accepted as a verified result.

The isolated GPU API runs through a Windows resource proxy, with Ollama bound
only to loopback behind it. Firewall access is restricted to the existing Debian
address. One inference is admitted; the active-time targets are 65/35/20 percent
and batches 64/32/16 according to measured headroom. Other work has priority:
heavy games, host CPU above 65%, free RAM below 6 GiB, free VRAM below 3 GiB,
GPU temperature at least 75 C or observed board draw above 250 W block new work
and can stop only the owned model runner. Decoder activity lowers the target.
The Debian profile requires a recent guard receipt before issuing inference.

These are sampled admission/interruption controls. They cannot hard-cap
instantaneous GPU watts or guarantee that previously unstable hardware will not
hang. Device-wide power limits and clocks are unchanged. Sampling errors disable
GPU admission instead of inventing safe readings. Resource state is shown in the
separate visible RTX monitor. Closing a log viewer leaves scheduled workers up.

Launchers capture complete native stderr and actual process exit codes instead
of treating the first traceback line as a fatal PowerShell error. CPU runtime
readiness is checked before enabling the guarded GPU API.

Validation is documented in the release response. Windows task execution,
actual RTX telemetry/thermals, operator browser and LAN endpoints must be checked
on the operator machines; this development environment does not provide them.
