# Tracing — the v0.2 milestone

## Trace format (.npz)

One file, per layer:

- `layer_{L}_gate` — float32 array (n_tokens, num_experts), router gate weights, 0 where not in top-k
- `layer_{L}_output_norm` — float32 array (n_tokens, num_experts), L2 norm of expert output, 0 where not fired

You can produce this file yourself (e.g. from a colibri `.coli_usage`
file, or from your own HF hook) and then use the scorer / planner /
repacker today, without waiting for v0.2.

## Hook points to implement first (DeepSeek V4 Flash)

1. Load with `AutoModelForCausalLM`, `torch_dtype=float16`, `device_map="auto"`.
2. For each layer `model.model.layers[L]`:
   - Hook `mlp.gate` (router): capture `router_logits` per token.
   - Hook `mlp.experts`: capture per-expert output norm. The
     cleanest way is to hook each expert's `down_proj` output
     and take its L2 norm per token.
   - Convert router logits to gate weights the same way the
     model does (softmax + top-k + renormalize, per that
     family's `forward`).
3. Run calibration prompts in batches, accumulate gate /
   output_norm per layer, free the layer, move on.
   Do NOT hold all layers' traces in RAM at once for a 150B
   model — stream to disk per layer.

## Why this is not in v0.1

Hook wiring varies by family, and a wrong hook silently
produces a wrong saliency ranking — which then prunes the
*wrong* experts. That needs a GPU + a real checkpoint to
validate against the paper's numbers. Shipping a guessed
hook would be worse than shipping no hook.

## Alternative trace source: colibri usage

Colibri already tracks which experts your past conversations
used (`.coli_usage`). A `trace --from-colibri-usage` mode
that converts that file to the .npz format above is a
good first contribution — it gives you a personal trace
with zero GPU work.
