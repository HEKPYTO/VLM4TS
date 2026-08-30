def test_render_returns_image():
    import numpy as np

    from src.render import series_to_pil

    s = np.sin(np.linspace(0, 10, 224))
    img = series_to_pil(s, window_size=224)
    assert img.size == (224, 224)
    assert img.mode == "RGB"


def test_render_different_lengths():
    import numpy as np

    from src.render import series_to_pil

    for n in [10, 50, 224, 500]:
        s = np.random.randn(n)
        img = series_to_pil(s, window_size=224)
        assert img.size == (224, 224)
