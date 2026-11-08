"""Human-readable report for a prune run."""
from __future__ import annotations

import json
from pathlib import Path

from .eval import EvalReport
from .pruning import PrunePlan


def write_report(
    path: str | Path,
    model: str,
    plans: list[PrunePlan],
    eval_report: EvalReport,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    total_keep = sum(len(pl.keep) for pl in plans)
    total_orig = sum(len(pl.keep) + len(pl.prune) for pl in plans)
    data = {
        "model": model,
        "total_experts_original": total_orig,
        "total_experts_kept": total_keep,
        "overall_keep_rate": round(total_keep / max(total_orig, 1), 3),
        "estimated_disk_saving": f"~{100*(1-total_keep/max(total_orig,1)):.0f}% of expert weights",
        "eval": eval_report.to_dict(),
        "per_layer": [
            {
                "layer": pl.layer,
                "keep": len(pl.keep),
                "prune": len(pl.prune),
                "keep_rate": round(pl.keep_rate, 3),
                "coverage_loss": round(pl.coverage_loss, 4),
                "guarded": pl.guarded.tolist(),
            }
            for pl in plans
        ],
    }
    p.write_text(json.dumps(data, indent=2))
    return p


def print_summary(model: str, plans: list[PrunePlan], eval_report: EvalReport) -> str:
    total_keep = sum(len(pl.keep) for pl in plans)
    total_orig = sum(len(pl.keep) + len(pl.prune) for pl in plans)
    lines = [
        f"Model: {model}",
        f"Experts: {total_orig} -> {total_keep} kept ({100*total_keep/max(total_orig,1):.1f}%)",
        f"Mean coverage loss (holdout): {eval_report.mean_coverage_loss:.4f}",
        f"Max layer coverage loss: {eval_report.max_layer_coverage_loss:.4f}",
        f"Verdict: {eval_report.verdict}",
        "",
        "Per-layer (first 5):",
    ]
    for pl in plans[:5]:
        lines.append(
            f"  layer {pl.layer:2d}: keep {len(pl.keep):3d}  "
            f"loss {pl.coverage_loss:.4f}  guarded {pl.guarded.tolist()}"
        )
    if len(plans) > 5:
        lines.append(f"  ... and {len(plans)-5} more layers (see report JSON)")
    return "\n".join(lines)
