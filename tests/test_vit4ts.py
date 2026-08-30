import numpy as np

from vlm4ts.vit4ts import ViT4TS


def test_candidates_select_high_scores_only():
    model = type("Model", (), {"alpha": 0.2})()
    assert ViT4TS.candidates(model, np.array([0.1, 0.9, 0.8])) == [1]


def test_candidates_skip_empty_constant_and_short_series_scores():
    model = type("Model", (), {"alpha": 0.01})()
    assert ViT4TS.candidates(model, np.array([])) == []
    assert ViT4TS.candidates(model, np.zeros(10)) == []
    assert ViT4TS.candidates(model, np.full(10, 0.5)) == []


def test_short_series_predicts_constant_scores_with_no_candidates():
    model = type("Model", (), {"alpha": 0.01, "ws": 224})()
    scores, _ = ViT4TS.predict_scores(model, np.arange(10))
    assert ViT4TS.candidates(model, scores) == []


def test_candidates_skip_nan_padded_tail_after_final_evaluated_maximum():
    model = type("Model", (), {"alpha": 0.2})()
    scores = np.array([0.1, 0.2, 0.9, np.nan, np.nan])
    assert ViT4TS.candidates(model, scores) == [2]
