import numpy as np

from moetailor.pruning import adaptive_keep_rates, plan_all, plan_layer
from moetailor.saliency import ExpertScores


def _scores(layer, saliency, freq=None, max_gate=None):
    n = len(saliency)
    return ExpertScores(
        layer=layer,
        saliency=np.array(saliency, dtype=float),
        frequency=np.array(freq if freq is not None else [0.5] * n),
        mean_gate=np.array([0.5] * n),
        mean_output_norm=np.array([1.0] * n),
        max_gate=np.array(max_gate if max_gate is not None else [0.5] * n),
    )


def test_plan_keeps_top_half():
    s = _scores(0, [1, 2, 3, 4, 5, 6, 7, 8])
    plan = plan_layer(s, keep_rate=0.5)
    assert len(plan.keep) == 4
    assert set(plan.keep.tolist()) == {4, 5, 6, 7}
    assert plan.coverage_loss == 10 / 36  # pruned 1+2+3+4 of total 36


def test_rare_but_decisive_expert_is_guarded():
    # Expert 0: saliency tiny, fires 0.5% of tokens, but max gate 0.9 when it does.
    s = _scores(
        0,
        saliency=[0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        freq=[0.005, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        max_gate=[0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    plan = plan_layer(s, keep_rate=0.5)
    assert 0 in plan.keep.tolist()
    assert 0 in plan.guarded.tolist()


def test_adaptive_rates_spread_around_target():
    # Layer 0 concentrated, layer 1 spread out.
    s0 = _scores(0, [10, 1, 1, 1, 1, 1, 1, 1])
    s1 = _scores(1, [5, 5, 5, 5, 5, 5, 5, 5])
    rates = adaptive_keep_rates([s0, s1], target_keep_rate=0.5)
    assert rates[0] < rates[1]  # concentrated layer pruned harder
    assert abs(sum(rates) / 2 - 0.5) < 0.1


def test_plan_all_covers_all_layers():
    plans = plan_all([_scores(i, [1, 2, 3, 4]) for i in range(3)], target_keep_rate=0.5)
    assert len(plans) == 3
    assert all(len(p.keep) + len(p.prune) == 4 for p in plans)
