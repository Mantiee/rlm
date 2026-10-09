# Goal-linked pattern learning (v100.41)

The operator owns the long-term goal. Master chat may follow a direction in a normal
message; existing short/mid plan controls remain available. Researchers receive the
current plans and may independently choose patterns, public sources, numeric outcomes
and horizons. There is no fixed asset or social-media list.

## Connected workflow

1. `observe_goal_source(url, field)` retrieves public HTML/text/JSON using the existing
   restricted public reader, records host retrieval time and archives the response.
   An empty field means evidence text. A dotted JSON field (including array indexes)
   selects a finite number or numeric string. Host-local disk headroom and archive
   count limits apply. Login walls and blocked sources are not bypassed.
2. `predict_goal_pattern(specification)` precommits a question, rationale, 1-8 evidence
   IDs, a numeric target observed within 60 seconds, horizon of 60 seconds to 7 days,
   positive absolute change threshold in target units, and probabilities in order
   down/flat/up. The operator goal and plans are snapshotted. This does not update weights.
3. A resident CPU/network observer runs every 30 seconds independently of paper feeds,
   chat, GUI readiness and GPU training. It checks at most four due forecasts per tick.
   Only a matching target source/field observed within horizon + five minutes resolves
   a forecast. Missing or late evidence remains unresolved/expired, never synthetic.
4. Host code computes direction, multiclass Brier score and flat/uniform baselines.
   Correct and incorrect predictions both produce supervised corrections. Training
   prompts contain only precommit evidence and target value; subsequent values and
   the model's unverified rationale are excluded. The learner predicts one direction
   token, retaining its existing general reasoning skills through the retention machinery.
5. `extend_pool` automatically imports resolved examples. `load_records` independently
   recomputes labels against host archives; forged model-generated labels are rejected.
   Shared source paths and target series remain together in the existing split ledger.
   Query/fragment changes cannot create new independent sources. Canonical inputs use bounded excerpts. Before automatic pool admission a local CPU tokenizer checks the calibrated max_length; oversized examples remain archived with an admission report, without increasing GPU settings. New goal examples get
   catalog visibility; A/B still selects curricula within its fixed budgets.
6. With at least four validation examples from two independent target groups for the
   current goal, each exported candidate and its actual predecessor run a goal-linked
   development panel. A regression, incomplete panel or failure to beat the best constant-direction baseline prevents eligibility. The constant baseline is a hindsight development reference, not a live forecasting policy. Goal panel
   quality ranks eligible A/B candidates before generic quality and speed tie-breaks.
   Existing ancestor, official and fresh audit gates remain required as configured.

## Evidence and limits

- Implementation: `rlm/v100/goal_learning.py`; research/chat tools in `research_tools.py`,
  `mission_chat.py`, `researchers.py`, `drones.py`, `paper_agents.py`; training in
  `insights.py`, `training.py`, `experiments.py`, `competition.py`.
- Ledger: `research/goal-learning/ledger.sqlite3`; source snapshots under `sources/`.
- `mission_evidence.goal_learning` reports pending/expired/resolved questions, probabilities,
  outcomes and recent scores. The existing dashboard evidence section shows this data.
- Branch reports: `goal-development.jsonl`, `goal-parent.json`, `goal-candidate.json`.
- Observational prediction is not proof of causal influence, investment profitability,
  transaction eligibility or an optimal policy. Costs and realized rewards use the
  separate paper ledger. Retrieval time does not certify original publication time.
- Targets must currently be public JSON scalar outcomes. This does not provide
  authenticated social APIs, arbitrary event settlement or autonomous commerce.
- The panel is reused development validation, not an untouched final financial audit.
  Entity correlations across different domains/URLs may remain. New forward commitments
  supply chronological online evidence; returns require separate later paper outcomes.
- Resident researchers can choose and test hypotheses, but successful discovery, source
  availability, sufficient validation diversity and weight improvements are not guaranteed.
- Architecture activation paths retain their existing gates; this goal panel is currently
  integrated into exported LoRA A/B candidates, not every alternative architecture pilot.

The design uses test-before-update observational evidence rather than assuming stable
relationships. Online adaptation under changing distributions is an active research
problem; this implementation does not reproduce the neural architectures of these papers:

- [Learning under Concept Drift: A Review](https://arxiv.org/abs/2004.05785)
- [Handling Concept Drift in Global Time Series](https://arxiv.org/abs/2304.01512)
- [Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting](https://arxiv.org/abs/2412.08435)
