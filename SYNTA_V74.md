# Synta v74: continuous goal work and measured shadow learning

This release gives the accepted master executable tools and separates actual learning evidence from research prose. It does not certify income, optimal settings or universal internet automation.

## Changes

- `rlm/v100/income_work.py`: after the three commissioning sources, the accepted model selects its next source, domain and experiment using the named tools. Six tool steps and a 180-second inference deadline. Complete tool traces are saved under `research/income-work/experiments`; preparation and HTTP/process receipts are never recorded as earned income.
- `rlm/v100/paper_agents.py`: researchers can inspect/open public web pages, prepare reviewed submissions, run guest experiments, inspect exact trade costs and consult the learned forecast policy. Real spendable capital is zero; the existing virtual paper portfolios are available for eligible funded tests. No resetting balances or weakening fee/quote/risk gates.
- `rlm/v100/drones.py`: transport failure on the Windows helper uses the protected, accepted V100 master. A changed model identity still fails. Windows benchmarks remain specific to Windows.
- `rlm/v100/research_tools.py`: explicitly bounded research tool sequences can use six steps instead of stopping after three. Existing tasks default to three; the maximum is eight. This is a capability budget, not keyword routing of user questions.
- `rlm/v100/researchers.py`: requests have an overall 180-second inference budget unless their caller supplies a shorter one. The model is told to use measured evidence for optimization claims.
- `rlm/v100/mission.py`: commissioning no longer repeats a long paper research round before entering the continuous research/training loop. Baselines and protected identity checks remain required.

## CPU shadow learning

`rlm/v100/goal_policy.py` trains a bounded softmax classifier using verified active-goal outcomes, including failures. Features use only past observations of the same numeric source. Replay scoring trains only on labels already available before a forecast was committed; overlapping horizons cannot leak future labels. Reference and shadow commitments on one target/horizon/threshold count as one training label. At most 512 archived commitments are used; history is bounded to eight observations. This is supervised online-style replay, separate from Gemma LoRA updates.

`rlm/v100/goal_observer.py` keeps the uniform A reference and commits a learned B shadow forecast on the same fresh observation. Both are independently settled by the existing host observer. The model can inspect learned probabilities through `suggest_goal_forecast`. Uniform probabilities are explicit cold-start/reference baselines, not master insight. No automatic trade execution follows these forecasts.

`research/goal-policy/status.json` reports trained outcomes, changed CPU weights and prequential Brier score against uniform, always-flat and prior-outcome-frequency baselines. Beating only uniform does not establish useful pattern learning. `goal_learning_status` also includes the CPU shadow state. A replay score is exploratory; prospective shadow results, paper net P&L and accepted Gemma updates remain different evidence. Losses do not become profitable labels.

## Internet and compute

Public sources, guest browser/shell, local deliverables, registered feeds, owned compute trials and eligible free-compute proposals are available. The master chooses operations through tools and sees their receipts. Existing network boundaries, service terms, credentials, eligibility and hardware limits still apply. Exact external submissions remain reviewable through the existing operator approval operation. No paid service, real trade, invented eligibility, human impersonation or bypass of service limits is enabled.

## Installation

`tools/update-synta-goal-work.sh` preserves the live profile, long-term goal, weights, ledgers, Windows policy and archived reports. It explicitly enables hypothetical capital research while keeping real spendable capital zero, spending disabled and real orders disabled. The old policy is archived. Startup verification waits up to 30 seconds for a running mission process; that is not proof of a completed training cycle.

Validation includes actual weight changes from failed forecasts, causal feature cutoffs, overlapping-horizon scoring, paired-label deduplication, altered-artifact rejection, extended tool sequences, beyond-seed execution, protected fallback and startup entering the learning loop. Live operator hardware and profitable strategies are not independently verified.
