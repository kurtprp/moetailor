import numpy as np

from moetailor.saliency import ExpertTrace, score_experts, rank_experts


def test_reap_score_is_gate_times_output():
    # 2 tokens, 3 experts. Expert 0 fires hard, expert 1 fires weakly, expert 2 never.
    gate = np.array([[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]])
    out = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    trace = ExpertTrace(layer=0, num_experts=3, gate=gate, output_norm=out)
    s = score_experts(trace)
    # saliency = mean(gate*out) = [(0.8*2+0.7*2)/2, (0.2+0.3)/2, 0]
    assert s.saliency[0] == 1.5
    assert abs(s.saliency[1] - 0.25) < 1e-9
    assert s.saliency[2] == 0.0
    assert rank_experts(s)[0] == 2  # least important first
    assert rank_experts(s)[-1] == 0


def test_never_fired_expert_scores_zero():
    gate = np.zeros((10, 4))
    out = np.zeros((10, 4))
    gate[:, 0] = 1.0
    out[:, 0] = 1.0
    s = score_experts(ExpertTrace(layer=1, num_experts=4, gate=gate, output_norm=out))
    assert s.saliency[1:].sum() == 0.0
    assert s.frequency[0] == 1.0
