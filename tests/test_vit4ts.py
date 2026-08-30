def test_vit4ts_scores_shape():
    import numpy as np
    from src.vit4ts import ViT4TS

    m = ViT4TS(alpha=0.01, window_size=224)
    series = np.random.randn(500)
    scores, ts = m.predict_scores(series)
    assert len(scores) == 500 and len(ts) == 500
    assert scores.shape == (500,)


def test_vit4ts_candidates():
    import numpy as np
    from src.vit4ts import ViT4TS

    m = ViT4TS(alpha=0.01, window_size=224)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.95])
    cands = m.candidates(scores)
    # alpha 0.01 => top 1% -> at least max element qualifies
    assert 4 in cands or 1 in cands


def test_vit4ts_short_series():
    import numpy as np
    from src.vit4ts import ViT4TS

    m = ViT4TS(alpha=0.01, window_size=224)
    s = np.random.randn(10)
    scores, ts = m.predict_scores(s)
    assert len(scores) == 10
