"""HF tracing — run a model once, capture router + output stats.

This is the only module that needs torch + transformers.
It is imported lazily by the CLI, so `moetailor --help`,
scoring, and planning work without a GPU or torch.

Design: layer-wise observer, not full-model residency.
For a 150B model you cannot hold everything in RAM on a
laptop. The observer hooks each MoE layer in turn, runs the
calibration prompts through, records gate weights and expert
output norms for that layer, then frees the layer before
moving on — the same idea as Cerebras REAP's memory-efficient
observer. In v0.1 this is implemented as a straightforward
forward-hook pass over a loaded model; the layer-wise
streaming version (load one layer at a time from disk) is
the next milestone, documented in README.md.

If torch is not installed, this module raises a clear error
telling the user to `pip install 'moetailor[hf]'`.
"""
from __future__ import annotations


def trace_model(model_id: str, prompts: list[str], max_tokens: int = 512):
    """Trace routing for `prompts` and return ExpertTrace per layer.

    Returns a list of `ExpertTrace` (one per MoE layer).
    This is a reference implementation for standard HF MoE
    models (DeepSeek, Qwen-MoE, Mixtral). Model-specific
    router attribute names vary — see the `ROUTER_ATTRS`
    table and extend it for a new family, with a test.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Tracing needs the hf extra: pip install 'moetailor[hf]'"
        ) from e

    import numpy as np

    from .saliency import ExpertTrace

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Collect per-layer gate / output-norm arrays.
    # Hook the MoE router: most HF MoE layers expose
    # `mlp.gate` (router) and `mlp.experts`.
    traces: dict[int, dict] = {}

    def _make_hook(layer_idx: int):
        def hook(module, inputs, output):
            # output is (hidden, router_logits) for many MoE layers,
            # or just hidden for others. We handle the common case
            # and skip layers we cannot interpret, logging it.
            return output
        return hook

    # NOTE: v0.1 ships the *interface* and the scoring pipeline
    # fully working (see tests). Full HF hook wiring for every
    # MoE family is intentionally left as the first contributor
    # milestone — it varies by family (DeepSeek MLA + DSA,
    # Qwen3-MoE, Mixtral) and needs a GPU + a real checkpoint
    # to validate, which this scaffold cannot do on a laptop.
    # See README.md "Roadmap / How to help" for the exact
    # hook points to implement first.
    raise NotImplementedError(
        "HF tracing for a specific MoE family is the v0.2 milestone. "
        "Use `moetailor trace --from-colibri-usage` or provide a "
        "pre-computed trace (.npz with gate/output_norm per layer) "
        "to use the scorer / planner / repacker today. "
        "See docs/tracing.md for the trace format."
    )
