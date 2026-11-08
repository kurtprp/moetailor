"""Calibration data — the 'personalized' part.

Three sources, in priority order:

1. ``--prompts your-prompts.jsonl`` — YOUR data. One JSON object
   per line: ``{"text": "..."}`` or ``{"messages": [{"role":..,
   "content":..}]}``. This is the whole point of moetailor: trace
   the prompts you actually send, not a generic benchmark mix.
   Export chat history, coding prompts, agent tool traces, etc.
   Keep it private — it never leaves your machine, and the
   trace files contain only aggregated per-expert statistics,
   never your prompt text.

2. ``--dataset code+math+tools`` — a built-in generic mix, for
   when you have no personal trace yet. Mirrors the Cerebras
   REAP calibration mix (code, math, science, tool-use).

3. ``--prompts`` + ``--holdout`` — split your prompts 90/10.
   Calibration uses 90%, eval uses the held-out 10%, so the
   before/after report is not measured on the data you pruned on.

The HF tracing itself lives in ``trace_hf.py`` (optional,
requires torch + transformers). This module only loads and
splits prompt files, so it works with zero heavy deps.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prompt:
    text: str
    source: str = "user"
    tag: str = ""


def load_prompts_jsonl(path: str | Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                text = obj["text"]
            elif "messages" in obj:
                parts = []
                for m in obj["messages"]:
                    parts.append(f"{m.get('role','user')}: {m.get('content','')}")
                text = "\n".join(parts)
            elif "prompt" in obj:
                text = obj["prompt"]
            else:
                continue
            prompts.append(
                Prompt(text=text, source=str(path), tag=str(obj.get("tag", "")))
            )
    return prompts


def split_holdout(
    prompts: list[Prompt], holdout_frac: float = 0.10, seed: int = 42
) -> tuple[list[Prompt], list[Prompt]]:
    """Deterministic 90/10 split. Holdout is for eval, never for pruning."""
    rng = random.Random(seed)
    idx = list(range(len(prompts)))
    rng.shuffle(idx)
    n_hold = int(round(len(prompts) * holdout_frac))
    hold_idx = set(idx[:n_hold])
    calib = [p for i, p in enumerate(prompts) if i not in hold_idx]
    hold = [p for i, p in enumerate(prompts) if i in hold_idx]
    return calib, hold


BUILTIN_MIX = [
    ("Write a Python function that parses a CSV and returns the median of a column, with tests.", "code"),
    ("Explain the difference between a Roth IRA and a traditional IRA for someone retiring early.", "writing"),
    ("Debug this: `TypeError: list indices must be integers` in a loop over a dict.", "code"),
    ("Summarise the trade-offs of pruning MoE experts vs quantizing them.", "reasoning"),
    ("Call the search tool to find the current price of NVDA, then summarise it in one sentence.", "tools"),
    ("What is 17 * 23 + 41? Show your working.", "math"),
]


def builtin_prompts() -> list[Prompt]:
    return [Prompt(text=t, source="builtin", tag=tag) for t, tag in BUILTIN_MIX]
