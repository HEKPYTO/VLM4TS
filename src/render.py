import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def series_to_pil(series, window_size=224):
    """Render 1-D series as 224x224 RGB line plot."""
    series = np.asarray(series, dtype=float)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.plot(series, linewidth=1.2, color="black")
    plt.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((window_size, window_size))
