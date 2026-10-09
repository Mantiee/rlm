# v42: optional GUI vision must not block chat on RTX cooldown

`desktop.gui` saves the guest screenshot and requests a helper workload slot without waiting. If another helper call or its required idle reservation is active, the tool returns `vision_deferred: true`. No visual interpretation is accepted, no remote call is sent and the cooldown remains unchanged. Required research calls retain their existing serialization and pacing.

`mission_chat` exposes the existing `sandbox_run` tool to complete requested file edits inside the private guest, and instructs the master to read, edit and verify `/workspace/dashboard/index.html`. Reading HTML is not an edit; observing the GUI is unnecessary for HTML editing. Existing dashboard validation and host isolation remain in force.

This fixes optional vision admission, not every possible source of latency. An admitted image request still uses the configured remote timeout. A chat request can still show `queued` while executing. It is not proof that the dashboard was edited or that the latest layout passed host validation.

Verification: 123 remote-helper/campaign tests passed, including deferred cooldown, thread contention, preserved screenshot and unchanged idle reservation. Ruff and installer shell syntax checks passed. V100/RTX deployment requires the operator upgrade; no real hardware result is claimed here.

## v43: interactive budgets and visible processing

Research policy no longer overrides chat thinking/output settings. Chat uses no thinking, at most 2048 output tokens and at most 60 seconds per HTTP operation. A shared monotonic 180-second deadline follows the client's copies across template/token counting, tool routing, final generation and helper pacing. This is an inference/pacing budget, not a hard wall-clock guarantee for CPU model startup or arbitrary guest tools. Timeout ends the request with a visible failure instead of endlessly retrying the same timeout.

`chat-status` includes a `processing` record for the request currently handled by the live mission PID, including start time and time budget. The persisted state remains `queued` for compatibility while execution is active. The record is cleared when handling finishes. Completion updates only still-queued requests, preserving operator cancellation.

The dashboard request still requires actual execution and host layout validation. These changes do not claim that an HTML edit or profitable training has occurred.
