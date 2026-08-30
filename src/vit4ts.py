import numpy as np
import torch
import open_clip
from src.render import series_to_pil



class ViT4TS:
    def __init__(self, alpha=0.01, window_size=224, model_name="ViT-B-32", pretrained="openai"):
        self.alpha = alpha
        self.ws = window_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()

    @torch.no_grad()
    def predict_scores(self, series):
        series = np.asarray(series, dtype=float).ravel()
        n = len(series)
        if n < self.ws:
            return np.zeros(n), np.arange(n)
        embs = []
        for i in range(n - self.ws + 1):
            img = series_to_pil(series[i : i + self.ws], self.ws)
            t = self.preprocess(img).unsqueeze(0)
            e = self.model.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            embs.append(e)
        embs = torch.cat(embs)  # [N, 512]
        med = embs.median(dim=0).values
        med = med / med.norm().clamp(min=1e-8)
        sim = (embs @ med).detach().cpu().numpy()  # cosine to median
        scores = 1.0 - sim  # anomaly = dissimilar
        # align scores to series: window i -> score at i (start), pad tail with last value
        aligned = np.zeros(n)
        aligned[: len(scores)] = scores
        if len(scores) < n:
            aligned[len(scores) :] = scores[-1]
        return aligned, np.arange(n)

    def candidates(self, scores):
        """Return indices of top-alpha quantile as candidate anomalies."""
        scores = np.asarray(scores)
        if len(scores) == 0:
            return []
        thresh = np.quantile(scores, 1 - self.alpha)
        return np.where(scores >= thresh)[0].tolist()


def _demo():
    import numpy as np

    m = ViT4TS(alpha=0.01, window_size=224)
    s = np.sin(np.linspace(0, 20, 500))
    s[200:210] += 3  # synthetic spike
    scores, ts = m.predict_scores(s)
    assert len(scores) == 500
    cands = m.candidates(scores)
    print(f"demo: series len={len(s)}, scores mean={scores.mean():.4f} max={scores.max():.4f} candidates={len(cands)}")
    print(f"candidates sample: {cands[:10]}")
    return scores


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        _demo()
