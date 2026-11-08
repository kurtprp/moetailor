"""moetailor CLI — `moetailor demo` works today, no GPU, no model.

Commands:
  moetailor demo --model deepseek-v4-flash --keep 0.5
      End-to-end demo on a synthetic-but-realistic trace
      (43 layers x 256 experts, top-6, like DeepSeek V4 Flash).
      Scores, plans adaptively, guards rare experts, evals on
      holdout, writes report.json. This is the pipeline;
      the only part that needs a real checkpoint is tracing.

  moetailor trace --model <hf-id> --prompts prompts.jsonl --out trace.npz
      Trace a real model (v0.2 milestone — see trace_hf.py).
      Today it explains exactly what is missing and how to
      contribute a family hook, instead of failing silently.

  moetailor prune --trace trace.npz --keep 0.5 --out plan.json
      Score + adaptive plan from a pre-computed trace file.

  moetailor repack --model <dir> --plan plan.json --out <dir>
      Repack a safetensors checkpoint (needs moetailor[hf]).

  moetailor eval --trace-holdout ... --plan plan.json
      (covered by `demo` / `prune` report in v0.1)
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from .calibration import builtin_prompts, load_prompts_jsonl, split_holdout
from .eval import evaluate_holdout
from .pruning import plan_all
from .report import print_summary, write_report
from .saliency import ExpertTrace, score_experts


def _synthetic_trace(n_layers=43, n_experts=256, n_tokens=2000, topk=6, seed=42):
    """Realistic synthetic trace for demo / tests.

    Gate weights are sparse (top-k per token), a few experts
    are hot, most are cold, and a couple are rare-but-decisive
    (low frequency, high max gate) so the guard can be seen.
    """
    rng = np.random.default_rng(seed)
    traces = []
    for layer in range(n_layers):
        # Hot experts: Zipf-like popularity — concentrated, like real
        # MoE routing. Top ~20 experts carry most of the mass, so
        # pruning the bottom half loses only a few percent.
        popularity = 1.0 / (np.arange(1, n_experts + 1) ** 1.8)
        popularity /= popularity.sum()
        gate = np.zeros((n_tokens, n_experts))
        out = np.zeros((n_tokens, n_experts))
        for t in range(n_tokens):
            chosen = rng.choice(n_experts, size=topk, replace=False, p=popularity)
            weights = rng.dirichlet(np.ones(topk) * 2.0)
            gate[t, chosen] = weights
            out[t, chosen] = rng.lognormal(mean=0.0, sigma=0.5, size=topk)
        # Rare-but-decisive: expert 200 fires rarely, hard.
        gate[::200, 200] = 0.9
        out[::200, 200] = 3.0
        traces.append(
            ExpertTrace(layer=layer, num_experts=n_experts, gate=gate, output_norm=out, topk=topk)
        )
    return traces


@click.group()
def main():
    """moetailor — personalized MoE expert pruning."""


@main.command()
@click.option("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
@click.option("--keep", default=0.5, type=float, help="Target keep rate, e.g. 0.5 = prune half")
@click.option("--out", default="runs/demo", type=click.Path())
def demo(model, keep, out):
    """Run the full pipeline on a synthetic DeepSeek-like trace."""
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    click.echo(f"moetailor demo — {model} — target keep {keep:.0%}")
    click.echo("Using a synthetic trace (43L x 256E, top-6). No model downloaded.")
    traces = _synthetic_trace()
    scores = [score_experts(t) for t in traces]
    plans = plan_all(scores, target_keep_rate=keep)
    keep_by_layer = {p.layer: p.keep.tolist() for p in plans}
    # Holdout eval: re-score on a *different* synthetic trace (different seed).
    hold_traces = _synthetic_trace(seed=123)
    hold_scores = [score_experts(t) for t in hold_traces]
    ev = evaluate_holdout(model, hold_scores, keep_by_layer, holdout_tokens=2000)
    report_path = write_report(outdir / "report.json", model, plans, ev)
    click.echo(print_summary(model, plans, ev))
    click.echo(f"\nReport: {report_path}")
    click.echo("Next: replace the synthetic trace with your own prompts:")
    click.echo("  moetailor trace --model <hf-id> --prompts your-prompts.jsonl --out trace.npz")


@main.command()
@click.option("--model", required=True, help="HF model id or local path")
@click.option("--prompts", required=True, type=click.Path(exists=True))
@click.option("--out", default="trace.npz", type=click.Path())
@click.option("--max-tokens", default=512, type=int)
def trace(model, prompts, out, max_tokens):
    """Trace a real model on your prompts (v0.2 milestone)."""
    loaded = load_prompts_jsonl(prompts)
    calib, hold = split_holdout(loaded)
    click.echo(f"Loaded {len(loaded)} prompts: {len(calib)} calibration, {len(hold)} holdout")
    # This will raise NotImplementedError with a clear pointer today.
    # That honesty is intentional: HF hook wiring varies by MoE
    # family and needs a GPU + checkpoint to validate.
    from .trace_hf import trace_model

    traces = trace_model(model, [p.text for p in calib], max_tokens=max_tokens)
    click.echo(f"Traced {len(traces)} layers -> {out}")


@main.command()
@click.option("--trace", "trace_path", required=True, type=click.Path(exists=True))
@click.option("--keep", default=0.5, type=float)
@click.option("--out", default="plan.json", type=click.Path())
def prune(trace_path, keep, out):
    """Plan pruning from a pre-computed trace (.npz)."""
    data = np.load(trace_path)
    scores = []
    for key in sorted(data.files):
        if key.endswith("_gate"):
            layer = int(key.split("_")[1])
            gate = data[key]
            outn = data[f"layer_{layer}_output_norm"]
            scores.append(
                score_experts(
                    ExpertTrace(layer=layer, num_experts=gate.shape[1], gate=gate, output_norm=outn)
                )
            )
    plans = plan_all(scores, target_keep_rate=keep)
    plan_dict = {
        str(p.layer): {"keep": p.keep.tolist(), "prune": p.prune.tolist()} for p in plans
    }
    Path(out).write_text(json.dumps(plan_dict, indent=2))
    click.echo(f"Wrote plan for {len(plans)} layers to {out}")


@main.command()
@click.option("--model", "model_dir", required=True, type=click.Path(exists=True))
@click.option("--plan", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--dry-run", is_flag=True)
def repack(model_dir, plan, out, dry_run):
    """Repack a safetensors checkpoint using a plan.json."""
    from .pruning import PrunePlan
    from .repack import repack_checkpoint

    raw = json.loads(Path(plan).read_text())
    plan_by_layer = {}
    for layer_s, v in raw.items():
        layer = int(layer_s)
        keep = np.array(v["keep"], dtype=int)
        prune = np.array(v["prune"], dtype=int)
        plan_by_layer[layer] = PrunePlan(
            layer=layer, keep=keep, prune=prune,
            keep_rate=len(keep) / max(len(keep) + len(prune), 1),
            coverage_loss=0.0, guarded=np.array([], dtype=int),
        )
    manifest = repack_checkpoint(model_dir, out, plan_by_layer, dry_run=dry_run)
    click.echo(json.dumps(manifest, indent=2)[:2000])


if __name__ == "__main__":
    main()
