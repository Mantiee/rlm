# Synta v73: fee observer recovery

A missing or unrecognized fee publication previously raised a generic exception and terminated the paper observer. The controller could then restart despite a healthy inference server.

- `rlm/v100/spot_bootstrap.py`: requests the English Kraken publication, permits its canonical redirect, retains the actual source URL and archived body, and raises `FeeSourceUnavailable` for unavailable or unrecognized fee evidence. Rates are parsed, never substituted with guessed values.
- `rlm/v100/paper_learning.py`: records `fee-refresh-blocked`, retries after one hour, and continues price observations and independent verified work. Existing fee expiry and ledger execution gates remain enforced. Other integrity errors still propagate.
- `rlm/v100/competition.py`: reports `research-on-current-master` when an unavailable Windows helper causes immediate use of the accepted local master. No backend routing behavior changed.
- `tools/update-synta-fee-observer.sh`: installs a pinned v73 revision and preserves the active profile, goals, weights, ledgers, Windows policy and archived reports.

The operator's unrecognized archived HTML has not been inspected. The locale request does not guarantee recognition of every future page layout. Failure remains visible and blocks fee-dependent fills; it no longer terminates unrelated observations. This release does not establish strategy profitability or completed optimizer updates.
