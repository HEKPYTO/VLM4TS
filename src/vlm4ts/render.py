import io

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series_to_pil(series, window_size=224):
    """Render a one-dimensional series as an RGB line-plot image."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.plot(np.asarray(series, dtype=float), linewidth=1.2, color="black")
    plt.axis("off")
    plt.tight_layout(pad=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").resize((window_size, window_size))
