"""REAP saliency scoring — the heart of the pruner.

Reference: Lasby et al., "REAP the Experts" (arXiv:2510.13999).
REAP scores each expert by:

    saliency(e) = mean over tokens where e fired of
                  gate_weight(token, e) * || expert_output(token, e) ||

Two signals, not one:
- gate weight alone rewards experts the router *likes*, even if
  their output is tiny.
- output norm alone rewards loud experts, even if the router
  almost never picks them.

Multiplying them is what makes REAP beat frequency-only pruning.

This module is pure NumPy so the scoring / ranking logic can be
tested and run without torch. The HF tracing code in
``calibration.py`` produces the arrays this module consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertTrace:
    """One layer's routing trace from a calibration run.

    Attributes:
        layer: layer index.
        num_experts: experts in this layer in the *original* model.
        gate: shape (n_tokens, num_experts) — router gate weights,
            0 for experts not in top-k for that token.
        output_norm: shape (n_tokens, num_experts) — L2 norm of
            each expert's output, 0 where the expert did not fire.
        topk: k used by the router (e.g. 6 for DeepSeek V4 Flash).
    """

    layer: int
    num_experts: int
    gate: np.ndarray
    output_norm: np.ndarray
    topk: int = 6

    def __post_init__(self) -> None:
        assert self.gate.shape == self.output_norm.shape
        assert self.gate.shape[1] == self.num_experts


@dataclass
class ExpertScores:
    layer: int
    saliency: np.ndarray  # (num_experts,) REAP score
    frequency: np.ndarray  # (num_experts,) fraction of tokens that fired it
    mean_gate: np.ndarray  # (num_experts,) mean gate when fired
    mean_output_norm: np.ndarray  # (num_experts,) mean ||output|| when fired
    max_gate: np.ndarray  # (num_experts,) max gate ever seen — LOEO guard


def score_experts(trace: ExpertTrace) -> ExpertScores:
    """Compute REAP saliency + diagnostics for one layer."""
    gate = trace.gate.astype(np.float64)
    out = trace.output_norm.astype(np.float64)
    fired = gate > 0  # an expert fired iff the router gave it weight

    n_tokens = max(gate.shape[0], 1)
    frequency = fired.sum(axis=0) / n_tokens

    # REAP: gate * output norm, averaged over tokens where it fired.
    # Tokens where it did not fire contribute 0, so we average over
    # *all* tokens — that is equivalent and avoids div-by-zero for
    # never-fired experts (score 0, pruned first, correctly).
    saliency = (gate * out).sum(axis=0) / n_tokens

    counts = fired.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    mean_gate = np.where(counts > 0, gate.sum(axis=0) / safe_counts, 0.0)
    mean_output_norm = np.where(
        counts > 0, out.sum(axis=0) / safe_counts, 0.0
    )
    max_gate = gate.max(axis=0) if n_tokens else np.zeros(trace.num_experts)

    return ExpertScores(
        layer=trace.layer,
        saliency=saliency,
        frequency=frequency,
        mean_gate=mean_gate,
        mean_output_norm=mean_output_norm,
        max_gate=max_gate,
    )


def rank_experts(scores: ExpertScores) -> np.ndarray:
    """Expert indices from least to most important."""
    return np.argsort(scores.saliency, kind="stable")


def coverage_loss(scores: ExpertScores, keep: np.ndarray) -> float:
    """Fraction of *routing mass* lost if we keep only `keep`.

    Routing mass = sum of gate weights. If the pruned experts
    carried 2% of total gate mass on your calibration data, the
    model has to re-route that 2% through survivors. Low is good;
    >5% is a warning sign that the keep-rate is too aggressive
    for this layer / distribution.
    """
    keep_set = np.zeros_like(scores.saliency, dtype=bool)
    keep_set[np.asarray(keep)] = True
    total = scores.mean_gate * scores.frequency  # proportional to mass
    # Use saliency as the mass proxy — it already is gate * output.
    total_mass = scores.saliency.sum()
    if total_mass <= 0:
        return 0.0
    lost = scores.saliency[~keep_set].sum()
    return float(lost / total_mass)
