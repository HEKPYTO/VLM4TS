import numpy as np

from vlm4ts.render import series_to_pil


def test_render_returns_rgb_image():
    image = series_to_pil(np.sin(np.linspace(0, 10, 224)), window_size=224)
    assert image.size == (224, 224)
    assert image.mode == "RGB"
