# Continual-learning retention portfolio (v100.40)

The recommended starting point for this hardware is a frozen pretrained base,
verified historical replay, predecessor KL, immutable accepted versions and
independent regression gates. Additional constraints should be compared as
separate candidates. Stacking every method is not evidence of improvement:
stability constraints can prevent useful new learning and add expensive forwards.
No finite test establishes universal zero forgetting.

## Primary research and implementation decisions

| Approach | Evidence / limitation | Implementation |
| --- | --- | --- |
| Historical replay | LLM CL surveys identify rehearsal as a major family; sampling coverage and distribution matter. | Existing mandatory verified prior-training records remain in A/B datasets. Existing split ledger excludes validation/audit sources. New reference sampler covers historical domains round-robin, at most eight examples. |
| Learning without Forgetting / KL | Li & Hoiem, ECCV 2016, https://arxiv.org/abs/1606.09282. Original work is vision, not a Gemma guarantee. | Existing shared frozen-base teacher adapter and supervised-token KL remain active; model chooses strength. No second 12B base allocation. |
| EWC | Kirkpatrick et al., PNAS 2017, https://doi.org/10.1073/pnas.1611835114. Empirical diagonal Fisher is an approximation and misses cross-parameter interactions. | New per-example squared-gradient Fisher on at most eight verified historical training records, quadratic anchor penalty, bounded LoRA parameters only. Reference weights/Fisher pinned and restored across checkpoint resume. Not original full-model EWC and not online-EWC. |
| Parameter anchoring | L2-SP, Li et al., ICML 2018, https://proceedings.mlr.press/v80/li18a.html. Parameter closeness does not imply identical outputs. | New L2 anchor to the initial predecessor adapter; frozen base unchanged. |
| Orthogonality | O-LoRA, EMNLP 2023, https://aclanthology.org/2023.findings-emnlp.715/; OGD, AISTATS 2020, https://proceedings.mlr.press/v108/farajtabar20a.html. | Experimental delta-A overlap penalty with predecessor row directions. Initialization penalty is zero. This is inspired by orthogonal methods, **not an O-LoRA/OGD reproduction**. |
| A-GEM | Chaudhry et al., ICLR 2019, https://arxiv.org/abs/1812.00420. Half-space protection is first-order on sampled replay; AdamW momentum/finite steps can still cause forgetting. | New conflicting-gradient projection after AMP unscale/clip and before optimizer step. Historical replay references rotate; adds one reference forward/backward per optimizer attempt. Nonfinite AMP attempts are left to the existing health gate. |
| Capacity growth | Progressive Neural Networks, Rusu et al., https://arxiv.org/abs/1606.04671, protect old columns when old task routes stay fixed. Route errors remain a problem. | New doubling of standard LoRA rank (maximum 64 and measured budget), zero new B columns, unchanged alpha/r and old tensors. It preserves the **initial** delta, not future behavior. Normal PEFT/GGUF candidate export and all quality gates remain required. |
| New layers / embeddings | Architectural growth can isolate interference, but changes export/tokenization and resource use. | `capacity_growth.py` supplies isolated tiny-model residual columns with zero output initialization, frozen predecessor route, and appended embedding rows with unchanged old IDs. Maximum two million old/new parameters for these prototypes. **Not live Gemma layer surgery, tokenizer expansion or a native GGUF serving feature.** |
| External memory / RAG | Retrieval avoids some weight updates but has stale/missing retrieval and routing failure modes. | Existing CPU hybrid persistent memory remains separate from weights. It is not proof that weights retain knowledge. |
| Evaluation / rollback | GEM describes backward transfer and continual-learning evaluation. Repeated public/development tests can overfit. | Existing case-by-case ancestor gates, official grading, one-use fresh audits and preserved accepted versions remain mandatory. Fresh audit answers are not training/replay/Fisher inputs. Old-version routing remains an available fallback, not proof a new version retained skills. |

Survey: Shi et al., *Continual Learning of Large Language Models: A Comprehensive
Survey*, https://arxiv.org/abs/2404.16789. Methods target different kinds of forgetting
and stages of training; results on small classification networks do not establish
performance on sequential Gemma 12B fine-tuning or financial outcomes.

## What the controller can select

With `resources.retention_experiments=true`, bounded A/B planning adds:

- `retention_mode`: 0 replay/KL; 1 L2; 2 EWC; 3 L2 + delta-A orthogonality;
  4 EWC + delta-A orthogonality; 5 A-GEM projection.
- `retention_strength`: 0.001, 0.01, 0.1, 1.0 (unused for projection).
- `retention_rank_growth`: 1 or 2, only when an existing rank can double within
  the measured maximum rank and the absolute limit of 64.

Delta-A orthogonality needs a predecessor adapter. EWC/projection are unavailable until a pinned predecessor and historical verified
training examples exist. Anchors/Fisher are capped at 160 million adapter
parameters. References are selected from the historical training portion of the
immutable combined pool, never the withheld validation set. These bounded
references are not comprehensive coverage of all pretrained capabilities.

Artifacts in each trial's training directory:

- `retention.json`, `retention-anchor.safetensors`: measured/reference provenance.
- `capacity-growth.json`, `expanded-initial/`: rank expansion evidence.
- Existing `training_health.json`, metrics, checkpoint manifests and quality
  judgments: actual successful optimizer counters and acceptance evidence.
- Activity events `retention-reference` and `replay-gradient-projection`.

The mission evidence collector exposes current-run reference/growth reports.
A configured method or a saved reference is **not** proof of a successful update,
accepted weights, improved retention, new task accuracy or profit.

## Experimental boundaries and omissions

Full GEM quadratic programming, full 12B Fisher/Jacobian matrices, unconstrained
layer replication, live vocabulary changes, and running many 12B model copies
are not enabled. They have unresolved compute/export/validation costs on a 32 GiB
V100. The implementation provides a testable portfolio, not every CL algorithm
or all published implementations. Newer variants require separate reproduction
and hardware measurements before being added to this pipeline.

Frozen-column and embedding prototypes have CPU tests of initial equality and
unchanged old-route/old-ID outputs after new-parameter training. LoRA growth has
structural factor/scaling tests. Fisher, penalties, projection and tamper-sensitive
resume have deterministic tensor tests. **Full NF4/FP16 Gemma training, retention
improvement and native export of an expanded adapter still need on-device tests.**

### v52 combined pilot

Mode 6 combines the existing delta Fisher EWC penalty, delta-A orthogonality penalty
and A-GEM-style reference gradient projection. Replay/KL and independently gated
rank or frozen auxiliary growth remain selectable. This is our experimental
composition, not a reproduction of an unnamed state-of-the-art method. Extra
regularization can impede adaptation; independent held-out and retention gates
remain mandatory, with the accepted predecessor retained on rejection.
