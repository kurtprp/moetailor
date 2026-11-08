# moetailor — REAP, but for *your* prompts

> A generic REAP prunes the least-useful experts for everyone.
> moetailor prunes the least-useful experts **for you**.

MoE models are huge because they keep every expert. You only
use a few of them. Cerebras REAP (arXiv:2510.13999) proved you
can delete 25–50% of experts in one shot, no retraining, and
keep >96% of quality on coding/agentic tasks — Qwen3-Coder-480B
pruned 50% to 246B keeps 97.6% of coding ability.

But REAP uses a *generic* calibration mix. If you only use a
model for coding + writing + tool calls, experts for other
domains are dead weight *for you*. moetailor traces **your**
prompts, scores every expert per layer by router gate × output
magnitude on that trace, prunes adaptively per layer, guards
rare-but-important experts, renormalizes the router, repacks
the checkpoint, and then proves the result with a before/after
eval on held-out prompts from the same distribution.

Your proof case: DeepSeek V4 Flash REAP 150B — 132 of 256
experts, 43 layers, top-6, 85 GB instead of 167 GB — loads in
colibri with no conversion. moetailor builds *your* version of
that, from your prompts.

## What it does today (v0.1)

- **REAP saliency, pure NumPy** — `gate × ||output||`, tested.
- **Adaptive per-layer pruning** — concentrated layers prune
  harder, spread-out layers keep more. No flat 50% everywhere.
- **Rare-expert guard** — an expert that fires <1% of tokens
  but with max gate >0.5 is kept, up to a small reserve.
  This is the LOEO-lite check that saves rare languages,
  rare tools, rare math patterns.
- **Router renormalization** — softmax over kept experts
  renormalizes automatically; bias correction provided for
  unnormalized routers.
- **Safetensors repack** — drops pruned experts, rewrites
  `config.json` (256 → 132), writes a `moetailor.json` manifest.
  Never overwrites the input.
- **Holdout eval + report** — coverage loss on *unseen*
  prompts, with an explicit PASS / CAUTION / FAIL verdict
  and a list of what was *not* measured.
- **`moetailor demo`** — full pipeline end-to-end in ~10 seconds,
  no GPU, no model download.

## What it does NOT do yet (honest)

- **HF tracing for a specific MoE family is v0.2.**
  `moetailor trace` explains this instead of failing silently.
  Hook wiring varies by family (DeepSeek MLA + DSA,
  Qwen3-MoE, Mixtral) and needs a GPU + a real checkpoint
  to validate. See `docs/tracing.md` for the trace format
  and the exact hook points to implement first.
- **GGUF repacking is not in v0.1.** GGUF expert tensors
  are packed with quantisation blocks. moetailor writes
  `keep_indices.json` for GGUF and the pruned safetensors
  checkpoint is the primary output — that checkpoint loads
  directly in colibri and vLLM. See `docs/gguf.md`.
- **No full benchmarks.** v0.1 reports routing coverage
  and (HF path) perplexity delta. HumanEval / MMLU /
  SWE-Bench need a harness + GPU and are explicitly listed
  as not measured.

## Quickstart

```bash
pip install -e ".[hf,dev]"   # hf = torch + transformers, for real tracing/repack
# Demo — no model, no GPU:
moetailor demo --keep 0.5
# -> runs/demo/report.json

# Your prompts (one JSON object per line):
# {"text": "..."}  or  {"messages": [{"role": "user", "content": "..."}]}
# See examples/calibration-prompts.jsonl
moetailor trace --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --prompts your-prompts.jsonl --out trace.npz

moetailor prune --trace trace.npz --keep 0.52 --out plan.json
moetailor repack --model /path/to/DeepSeek-V4-Flash-0731 \
  --plan plan.json --out runs/my-reap-150b
```

## How pruning works

1. **Calibrate** — run your prompts through the model,
   record per-expert gate weight + output norm per layer.
   90% of prompts calibrate, 10% are held out for eval.
2. **Score** — REAP saliency = mean(gate × ||output||).
3. **Plan** — adaptive keep-rate per layer from saliency
   entropy, plus the rare-expert guard.
4. **Repack** — delete pruned experts, prune gate rows,
   rewrite config, write manifest.
5. **Prove** — coverage loss on holdout, perplexity delta,
   report.json. If loss >5%, lower the prune rate or add
   more calibration data from your distribution.

## Why personalized beats generic

|  | Generic REAP | moetailor |
|---|---|---|
| Calibration | Generic code/math/tools mix | Your prompts |
| Keep rate | Flat 50% / layer | Adaptive by entropy |
| Rare experts | Often pruned | Guarded |
| Output | Safetensors | Safetensors + colibri-ready + GGUF keep-list |
| Proof | Paper benchmarks | Your holdout + report |

## Roadmap

- [ ] v0.2 — HF tracing for DeepSeek V4 Flash, Qwen3-MoE, Mixtral
- [ ] v0.2 — Layer-wise streaming observer (prune 150B on a laptop)
- [ ] v0.3 — GGUF repack
- [ ] v0.3 — HOPE second-order pruning (pairwise interactions)
- [ ] v0.3 — Full eval harness (HumanEval, MMLU slice)

## Contributing

Start with `docs/tracing.md` — the first good issue is
implementing the DeepSeek hook. Tests are in `tests/`,
run with `pytest`.

## License

Apache 2.0. REAP method: Lasby et al., Cerebras Research.
This project is not affiliated with Cerebras, DeepSeek, or colibri.
