# Synta conditional-memory pilot

Primary research: https://arxiv.org/abs/2601.07372 and
https://github.com/deepseek-ai/Engram. Published results use trained
Engram-equipped models and host-memory prefetching; they do not establish that
memory inserted into an existing Gemma GGUF improves a financial goal.

`rlm/v100/ngram_memory.py` implements an original smaller Engram-inspired
candidate: causal rolling byte n-gram addresses, bounded hashed embeddings,
contextual gates and a zero-initialized value projection. It does not reproduce
DeepSeek's tokenizer compression, multi-head hashing, convolution or distributed
prefetching implementation.

`propose_ngram_memory` accepts width 16-64, 1-2 GRU layers, 256-8192 power-of-two
memory slots and orders 2-4. The conservative total pilot parameter limit is two
million. Float32 tables occupy at most 6 MiB. Large disk mapping is not justified
for this pilot: NVMe stores checkpoints, not extra GPU compute or a measured RAM
expansion. The operator's Windows disk type and bandwidth have not been measured.

`test_submodel` uses the existing isolated CPU trial with 2 threads and bounded
time, memory and parameters. The host repeats prediction with memory disabled on
the same trained weights and fixed cases, retains both logs, scores them without
revealing gold answers to the worker, and verifies the weights checksum between
evaluations. `quality.json.memory_ablation` records scores and time. This measures
a component's contribution, not an independently trained equal-capacity baseline.

Scratch-master promotion refuses candidates without measured development
ablation improvement. Existing independent goal, retention, fresh and public
gates still apply. Main GGUF weights are untouched. The Windows RTX helper is
not started or enabled; the operator shutdown remains authoritative. No actual
goal improvement, profitability or RTX hardware acceptance is claimed.

Regressions cover future-token isolation, initial unchanged logits, actual
gradient updates, safetensors round-trip, dimension rejection before allocation
and host scoring with memory on/off on frozen weights.
