"""Before/after eval — the proof, not the promise.

Two cheap, honest checks that run on a laptop (no GPU needed
for the routing checks, GPU needed only for full generation):

1. Routing coverage on held-out prompts: what fraction of
   routing mass on *unseen* prompts would have gone to pruned
   experts? This is the same coverage-loss metric as in
   saliency, but on holdout, so it is not optimistic.

2. Perplexity delta (optional, HF path): run the original and
   pruned model on the same held-out prompts and report
   perplexity before/after. A small rise (<5%) is expected at
   50% pruning; a large rise means the calibration distribution
   did not match the holdout, or the keep-rate was too aggressive.

Full benchmarks (HumanEval, MMLU, SWE-Bench) are out of scope
for v0.1 — they need a harness and a GPU. The report explicitly
says what was *not* measured, so a good routing score is never
presented as a quality guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .saliency import ExpertScores


@dataclass
class EvalReport:
    model: str
    keep_rate: float
    holdout_tokens: int
    mean_coverage_loss: float
    max_layer_coverage_loss: float
    perplexity_before: float | None = None
    perplexity_after: float | None = None
    verdict: str = ""

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "keep_rate": self.keep_rate,
            "holdout_tokens": self.holdout_tokens,
            "mean_coverage_loss": round(self.mean_coverage_loss, 4),
            "max_layer_coverage_loss": round(self.max_layer_coverage_loss, 4),
            "perplexity_before": self.perplexity_before,
            "perplexity_after": self.perplexity_after,
            "verdict": self.verdict,
            "not_measured": [
                "HumanEval / MMLU / SWE-Bench — needs a benchmark harness + GPU",
                "Generation quality on long-form tasks — check manually on your prompts",
            ],
        }
        return d


def evaluate_holdout(
    model: str,
    scores_by_layer: list[ExpertScores],
    keep_by_layer: dict[int, list[int]],
    holdout_tokens: int,
) -> EvalReport:
    losses = []
    for scores in scores_by_layer:
        keep = keep_by_layer.get(scores.layer, [])
        keep_set = np.zeros(len(scores.saliency), dtype=bool)
        if keep:
            keep_set[np.asarray(keep)] = True
        total = scores.saliency.sum()
        lost = scores.saliency[~keep_set].sum() if total > 0 else 0.0
        losses.append(float(lost / total) if total > 0 else 0.0)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    max_loss = float(np.max(losses)) if losses else 0.0
    keep_rate = float(
        np.mean([len(v) / max(len(scores_by_layer[0].saliency), 1) for v in keep_by_layer.values()])
    ) if keep_by_layer else 0.0

    if mean_loss < 0.02:
        verdict = "PASS — routing loss on holdout is tiny. Safe to try the pruned model."
    elif mean_loss < 0.05:
        verdict = "CAUTION — routing loss is moderate. Try it, but eval on your own prompts before trusting it."
    else:
        verdict = "FAIL — routing loss is high. Lower the prune rate or add more calibration data from this distribution."
    return EvalReport(
        model=model,
        keep_rate=keep_rate,
        holdout_tokens=holdout_tokens,
        mean_coverage_loss=mean_loss,
        max_layer_coverage_loss=max_loss,
        verdict=verdict,
    )
