# Synta v88 availability and throughput recovery

A stopped mission could remain stopped when its package version was current, or while
an updater was downloading optional assets. Availability now has a separate one-minute
user timer. It resumes the preserved accepted profile, defers exclusive training,
respects explicit operator pauses, and backs off failed restarts. An explicit recovery
request survives a deferred training window. Installation and recovery share a file
lock, so recovery cannot race a package replacement. Neither timer downloads models.

Windows preparation refreshes the owned proxy and launcher before model downloads.
Fresh active generation defers replacement; current proxies support an atomic drain.
Legacy unresponsive serving processes are replaced only after their executable path,
command line and creation time establish ownership. This can interrupt a request on an
old stale proxy; it does not terminate CPU workers or native benchmark processes.
Only the isolated Ollama runtime is restarted. Other applications and drivers remain.

The opted-in throughput profile allows continuous requests under spare headroom, batches
up to 256 and more inference CPU threads. It reserves 1 GiB for the GPU scheduler rather
than half the board. Games, CPU/RAM pressure, insufficient VRAM, heat and sampled board
power still stop or pace owned inference. These are sampled admission decisions, not
hard hardware limits or a speed guarantee. Power and clocks are not changed.

A telemetry write failure no longer kills the sensor loop. A heartbeat older than ten
seconds cannot authorize generation. The monitor console marks stale data explicitly.
The dashboard probes Windows separately from the dedicated 3060 Ti and shows recovery
receipts. Drone admission assigns one active job per inference lane rather than filling
all slots with jobs that wait on one GPU. Stable role qualification results are cached;
changed model inventory or baseline triggers new checks.

Main recovery also registers the explicitly owned 192.168.0.9 peer and enables verified
CPU goal trials. Installing the isolated node still requires the bootstrap command to
be run on that computer. Its services start at boot; the main coordinator reconnects.
No remote shell or pooled cross-host VRAM is created.

Qwen3.6 35B-A3B and the small Qwen roles are candidates until host measurements admit
them. Wiki ingestion and indexing remain parallel background work. Archived RAG
knowledge is not an accepted weight update. This release does not establish income,
profitable trading, a 500k active context, a fast 120B model, or a pretrained Engram.

Validation: full Python regression suite plus recovery/pressure/identity tests;
PowerShell runtime and physical GPU/boot/network measurements require host receipts.
