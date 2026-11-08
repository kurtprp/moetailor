"""Adaptive per-layer pruning + rare-expert guard + router renorm.

Why not a flat 50% everywhere?
--------------------------------
Some layers are redundant, some are not. A flat cut wastes
budget on layers that need their experts and over-prunes layers
that could go further. moetailor allocates the keep-budget by
*saliency concentration* per layer:

- If a layer's saliency is concentrated in a few experts
  (low entropy), it can be pruned hard.
- If saliency is spread evenly (high entropy), it needs more
  experts kept.

This is implemented as: for a global target keep-rate, compute
each layer's entropy, then shift keep-rate toward low-entropy
layers, bounded so no layer goes below `min_keep` or above
`max_keep`.

Rare-but-important guard (LOEO-lite)
-------------------------------------
Frequency-only pruning kills experts that fire rarely but
decisively — e.g. an expert for a rare language, a rare tool,
or a rare math pattern. If an expert fired <1% of tokens but
its max gate was very high (>0.5), we keep it regardless of
its mean saliency, up to a small reserve budget. This mirrors
the leave-one-expert-out micro-check in progressive-pruning
pipelines, without the cost of a full LOEO sweep.

Router renormalization
----------------------
After pruning, the router's softmax is taken over the *kept*
experts only, so it renormalizes automatically in most
implementations (DeepSeek, Qwen, Mixtral all do softmax/top-k
over the remaining gate logits). No weight rescaling is needed
for softmax routers.

For sigmoid / unnormalized routers (some GGUF paths), we
provide `router_bias_correction`: add log(keep_ratio) to the
kept gate bias so the expected total gate mass is preserved.
It is off by default and logged when used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .saliency import ExpertScores, rank_experts


@dataclass
class PrunePlan:
    layer: int
    keep: np.ndarray  # sorted kept expert indices, original numbering
    prune: np.ndarray  # pruned expert indices
    keep_rate: float
    coverage_loss: float
    guarded: np.ndarray  # experts kept by the LOEO guard, not by saliency


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def adaptive_keep_rates(
    scores_by_layer: list[ExpertScores],
    target_keep_rate: float,
    min_keep: float = 0.25,
    max_keep: float = 0.85,
) -> list[float]:
    """Allocate a keep-rate per layer from saliency entropy.

    Low entropy -> concentrated -> can prune more.
    High entropy -> spread out -> keep more.
    The mean across layers is (approximately) target_keep_rate.
    """
    if not scores_by_layer:
        return []
    entropies = np.array([_entropy(s.saliency) for s in scores_by_layer])
    # Normalise entropy to [0, 1] across layers.
    lo, hi = entropies.min(), entropies.max()
    if hi - lo < 1e-9:
        return [float(target_keep_rate)] * len(scores_by_layer)
    norm = (entropies - lo) / (hi - lo)  # 0 = concentrated, 1 = spread
    # Shift: concentrated layers get -0.10, spread layers +0.10,
    # centred so the mean shift is ~0.
    shift = (norm - norm.mean()) * 0.20
    rates = np.clip(target_keep_rate + shift, min_keep, max_keep)
    # Renormalise the mean back toward the target (one pass).
    mean = rates.mean()
    if mean > 0:
        rates = np.clip(rates * (target_keep_rate / mean), min_keep, max_keep)
    return [float(r) for r in rates]


def plan_layer(
    scores: ExpertScores,
    keep_rate: float,
    guard_max_gate: float = 0.5,
    guard_max_freq: float = 0.01,
    guard_reserve: int = 4,
) -> PrunePlan:
    """Plan one layer: which experts to keep, which to prune."""
    n = len(scores.saliency)
    n_keep = int(round(n * keep_rate))
    n_keep = max(1, min(n, n_keep))

    order = rank_experts(scores)  # least -> most important
    keep = set(int(i) for i in order[-n_keep:])

    # LOEO-lite guard: rare but decisive experts.
    guarded: list[int] = []
    candidates = [
        int(i)
        for i in range(n)
        if i not in keep
        and scores.frequency[i] < guard_max_freq
        and scores.max_gate[i] >= guard_max_gate
    ]
    # Highest max-gate first, up to the reserve.
    candidates.sort(key=lambda i: scores.max_gate[i], reverse=True)
    for idx in candidates[:guard_reserve]:
        keep.add(idx)
        guarded.append(idx)

    keep_arr = np.array(sorted(keep), dtype=int)
    prune_arr = np.array(
        sorted(set(range(n)) - set(keep_arr.tolist())), dtype=int
    )
    # Coverage loss: saliency mass in pruned experts.
    total = scores.saliency.sum()
    lost = scores.saliency[prune_arr].sum() if len(prune_arr) else 0.0
    cov = float(lost / total) if total > 0 else 0.0

    return PrunePlan(
        layer=scores.layer,
        keep=keep_arr,
        prune=prune_arr,
        keep_rate=len(keep_arr) / n,
        coverage_loss=cov,
        guarded=np.array(guarded, dtype=int),
    )


def plan_all(
    scores_by_layer: list[ExpertScores],
    target_keep_rate: float = 0.5,
    **kwargs,
) -> list[PrunePlan]:
    rates = adaptive_keep_rates(scores_by_layer, target_keep_rate)
    return [
        plan_layer(s, r, **kwargs)
        for s, r in zip(scores_by_layer, rates)
    ]


def router_bias_correction(keep_rate: float) -> float:
    """Bias to add to kept gate logits for unnormalized routers."""
    if keep_rate <= 0:
        return 0.0
    return float(math.log(keep_rate))
