import numpy as np

from vlm4ts.vlm_refine import refine


def test_refine_accepts_numpy_scores_and_filters(monkeypatch):
    monkeypatch.setattr("vlm4ts.vlm_refine.call_vlm", lambda *_: [1, 0])
    assert refine(np.array([0.1, 0.9, 0.8]), [1, 2], [0] * 4) == [1]


def test_refine_prompt_uses_candidate_indexed_scores(monkeypatch):
    captured = {}

    def capture_prompt(_image, prompt):
        captured["prompt"] = prompt
        return [1, 0]

    monkeypatch.setattr("vlm4ts.vlm_refine.call_vlm", capture_prompt)
    assert refine(np.array([0.111, 0.222, 0.333, 0.444]), [3, 1], [0] * 4) == [3]
    assert "[(3, 0.444), (1, 0.222)]" in captured["prompt"]
    assert "0.111" not in captured["prompt"]


def test_refine_empty_and_invalid_masks_fall_back(monkeypatch):
    assert refine(np.array([]), [], [0, 1]) == []
    monkeypatch.setattr("vlm4ts.vlm_refine.call_vlm", lambda *_: [True, False])
    assert refine(np.array([0.5, 0.6]), [0, 1], [0, 1, 2]) == [0, 1]


def test_refine_without_key_does_not_call_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert refine(np.array([0.5]), [0], [0, 1]) == [0]
