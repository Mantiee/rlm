# Wikipedia resumable transfer and bounded prefetch

Observed host receipt: one 2694239210-byte part, Accept-Ranges: bytes, approximately 1.8 MiB/s measured over an 11-second interval. This is not evidence of the limiting component and does not establish a speedup from additional transfers.

Changed `rlm/v100/wikimedia_exports.py`:
- Hash existing `.partial` prefix, request `Range: bytes=OFFSET-`, append only after exact HTTP206/Content-Range confirmation, verify final SHA256. Range ignored or mismatch preserves the prefix and fails explicitly. Already complete SHA-matching partials are promoted without downloading again.
- Log bytes, total, measured per-file average rate and estimated remaining time approximately every five seconds while data arrives.
- At most two download connections with bounded part prefetch; index serially in publisher-manifest order. `--connections 1` retains serial operation. Failures cancel other owned transfers; no identity/address rotation, retry storm or bypass of a 403/429.
- Singleton importer lock on Debian prevents multiple new importers writing the shared index concurrently.

The pinned standalone launcher uses the changed module without replacing the installed package or restarting the mission. Stop only the previous importer unit before starting the new launcher so its contiguous partial prefixes can be resumed. Existing indexed sources and cached complete parts remain available. After a crash, parts can be reindexed deterministically; restart is not advertised as skipping all prior indexing.

Validation: 11 targeted transfer/RAG tests passed, covering resumed prefix, ignored-range protection, checksum mismatch, explicit client identity, progress, bounded parallel download and serial indexing. Ruff/diff/Bash checks pass. Real combined throughput remains a host measurement; a twofold speedup is not claimed. No GPU or model weights changed.
