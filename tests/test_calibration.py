import json

from moetailor.calibration import load_prompts_jsonl, split_holdout


def test_load_and_split(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"prompt {i}"}) for i in range(20)))
    prompts = load_prompts_jsonl(p)
    assert len(prompts) == 20
    calib, hold = split_holdout(prompts, holdout_frac=0.10, seed=42)
    assert len(hold) == 2
    assert len(calib) == 18
    # Holdout never overlaps calibration.
    assert not set(x.text for x in hold) & set(x.text for x in calib)


def test_messages_format(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    prompts = load_prompts_jsonl(p)
    assert "hi" in prompts[0].text
