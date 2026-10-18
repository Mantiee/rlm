# V1: MTP gate on the V100 is aware of sampling noise (Claude, 2026-10-18 ~20:55Z)

## Why MTP is still off on the V100

- The mission (`mission.setup_profile`) removes `draft_model` unless `mtp_gate.valid(profile)` is true. This is correct: there is no shortcut through the profile.
- The only gate run (08.10, `research/logs/mtp-ab-20261008-001455-da7bc852`) measured MTP n_max=2 at **×1.49–1.67 tok/s** (acceptance 0.72). It was then rejected by the quality check: baseline 70/81, candidate 68/81, with 4 regressions and 2 improvements.
- The quality suite runs at the served settings: `temperature 1.0, top_p 0.95, top_k 64, seed 42, thinking`. With sampling, `compare_reports` fails as soon as there is a single regression. In practice it can reject even a baseline compared with itself, so the gate measures noise.

## Change (only `rlm/v100/mtp_gate.py`)

- `quality_verdict(baseline, candidate, control=None)`:
  - Without a control run, the rule is the old strict one.
  - With a control run (the same baseline once more, after the candidate), MTP passes only if both hold:
    - the candidate has no more regressions than the number of baseline↔control flips;
    - the candidate passes at least as many cases as the weaker baseline run.
  - A deterministic suite gives 0 flips, so the result is identical to the strict rule.
  - The control must use the same model, suite, generation settings and memory (`compare_reports` checks this, plus a model SHA check).
- `optimize()` adds the control run (`quality-control.json`) to `mtp_validation.reports` (4 reports).
- `valid()` accepts the legacy 3-report proof (strict) or the 4-report proof (with control). Report SHAs and identity checks are unchanged.
- No change to served generation settings, the evaluation suite or thresholds. Nothing is hardcoded.

## SHA

- canonical/host `rlm/v100/mtp_gate.py` pre: `e1089a1b4713f1851a5c8700a99c58ababe6fbc52e1e6f2e48252a199e291dab` (local == host)
- post: `bcde048135b9030e053dc2a5ed1adb78c69cdfce3044cdde545b4cef8a792c9e`
- test: `test_mtp_gate_noise.py` `5af620aee368ad6e5e2897f6cdf05f2daa6ec8c1f8739a397680bf09a8bd3339`

## Tests

- `.venv` py3.12: `test_mtp_gate_noise.py` (7) + `tests/test_v100_speculative.py`: **40 passed**.
- `tests/test_v100_campaign.py` imports `pwd`, so on Windows it is to be run on the host or in CI.
- In-memory check in the host venv (py3.11): the module imports. On the 08.10 reports the strict verdict is still `False` (4 regressions / 2 improvements), and the verdict for identical reports is `True`.

## Deployment (Codex; copying the file does not need a restart)

1. Check host SHA == pre, then back up to `research/deploy/rollback/mtp-gate-noise-20261018/mtp_gate.py.pre` and copy the post file.
2. The running mission is not affected: `valid()` runs at the next `setup_profile` call, and no validated MTP proof exists yet.
3. **Maintenance window (Mantie's decision):** `optimize-mtp` requires a stopped mission and an idle GPU.
   - Sweep 4 draft sizes: ~32 min. Then 3 runs of the 81-case suite: ~70 + 40 + 70 min.
   - In total about **3.5 h without the mission**.
   - Command (the profile is the current `run-*/input-profile.json`; the suite is the one from the 08.10 run):
     `python -m rlm.v100.cli --root /home/marek/ai-v100 --profile <input-profile> optimize-mtp --suite <suite> --repeats 3`
   - If the result is `validated-profile.json`, the mission keeps MTP from the next start (supervisor `live_profile`/resume). Otherwise the baseline stays.
4. Rollback: restore `mtp_gate.py.pre`. A proof with 4 reports then becomes invalid under the old `len(paths) != 3` check, so the mission returns to baseline on its own.
