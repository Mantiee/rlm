# Synta structural experiments, v58

The master can call `morph_model` to create a typed structural candidate, inspect
`morph_model_status`, and choose an isolated CPU pilot (`test_submodel`) or queue
an exclusive full-budget master experiment (`propose_scratch_master`). These are
executable tools, not requests to manually paste Python. Existing mission
scheduling consumes the queued master experiments.

The bounded search space includes GRU or causal Transformer, width 16–256,
1–4 backbone layers, compatible attention heads, embedding width, and 0–4
residual bottleneck layers. The conservative pilot ceiling is two million
parameters. Vocabulary IDs remain the same 257 UTF-8 byte symbols. Changing
width/depth/backbone creates an independently trained model; it does not reshape
Gemma GGUF tensors in place.

Warm growth requires a trained typed predecessor with the same goal and backbone
shape. Added residual layers initially contribute zero. Existing parameters are
frozen; only new layers learn. Parent weights and source are hashed, retained and
checked before reuse. Separate safetensors checkpoints permit rollback.

Master replacement requires verified source-held-out goal outcomes before
training, strict improvement over the parent on that task, performance above the
best constant-label baseline, complete retained capability tests, fresh audit,
and official benchmark gates when configured. The current master remains active
when data or evidence is missing or a gate fails. An improved tiny specialist is
not automatically a better general master.

Pilot and full-master budgets differ. A completed pilot cannot be silently
relabelled as a full-budget master trial. Create a fresh candidate for the latter.
No architecture, throughput, profit or zero-forgetting guarantee is made.
GPU experiments use the existing isolated exclusive PyTorch path; Windows RAM
and VRAM are not pooled. Production hardware execution remains to be checked on
the operator's machines.
