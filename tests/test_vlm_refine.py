def test_vlm_refine_filters(monkeypatch):
    from src.vlm_refine import refine

    monkeypatch.setattr("src.vlm_refine.call_vlm", lambda *a, **k: [1, 0])
    out = refine(scores=[0.1, 0.9, 0.8], candidates=[1, 2], series=[0] * 4)
    assert out == [1]


def test_vlm_refine_empty():
    from src.vlm_refine import refine

    assert refine(scores=[], candidates=[], series=[0, 1]) == []


def test_vlm_refine_fallback(monkeypatch):
    from src.vlm_refine import refine

    monkeypatch.setattr("src.vlm_refine.call_vlm", lambda *a, **k: None)
    out = refine(scores=[0.5, 0.6], candidates=[0, 1], series=[0, 1, 2])
    assert out == [0, 1]


def test_vlm_refine_per_series_mask(monkeypatch):
    from src.vlm_refine import refine

    # per-series mask length == len(series)
    monkeypatch.setattr("src.vlm_refine.call_vlm", lambda *a, **k: [0, 0, 1, 0])
    out = refine(scores=[0.1, 0.9, 0.8], candidates=[1, 2], series=[0] * 4)
    assert out == [2]
