"""moetailor — personalized MoE expert pruning.

REAP, but for *your* prompts, on *your* hardware, with proof.

The idea: a generic REAP (Cerebras, arXiv:2510.13999) prunes the
least-useful experts using a generic calibration mix. That is safe
for everyone and optimal for no one.

moetailor traces *your* prompts — your code, your writing, your tool
calls — scores every expert per layer by router gate × output
magnitude on that trace, prunes adaptively per layer, guards
rare-but-important experts, renormalizes the router, repacks the
checkpoint, and then proves the result with a before/after eval
on held-out prompts from the same distribution.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
