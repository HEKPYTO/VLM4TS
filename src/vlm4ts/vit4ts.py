import numpy as np
import open_clip
import torch

from .render import series_to_pil


class ViT4TS:
    def __init__(
        self, alpha=0.01, window_size=224, model_name="ViT-B-32-quickgelu", pretrained="openai"
    ):
        self.alpha = alpha
        self.ws = window_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()

    @torch.no_grad()
    def predict_scores(self, series):
        series = np.asarray(series, dtype=float).ravel()
        if len(series) < self.ws:
            return np.zeros(len(series)), np.arange(len(series))
        embeddings = []
        for start in range(len(series) - self.ws + 1):
            image = series_to_pil(series[start : start + self.ws], self.ws)
            embedding = self.model.encode_image(self.preprocess(image).unsqueeze(0))
            embeddings.append(embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        embeddings = torch.cat(embeddings)
        median = embeddings.median(dim=0).values
        median = median / median.norm().clamp(min=1e-8)
        scores = 1.0 - (embeddings @ median).cpu().numpy()
        aligned = np.full(len(series), np.nan)
        aligned[: len(scores)] = scores
        return aligned, np.arange(len(series))

    def candidates(self, scores):
        """Return the top-alpha score indices, excluding uninformative scores."""
        scores = np.asarray(scores, dtype=float).ravel()
        finite = np.isfinite(scores)
        evaluated = scores[finite]
        if not evaluated.size or np.ptp(evaluated) == 0:
            return []
        threshold = np.quantile(evaluated, 1 - self.alpha)
        return np.flatnonzero(finite & (scores >= threshold)).tolist()


def _demo():
    model = ViT4TS(alpha=0.01, window_size=224)
    series = np.sin(np.linspace(0, 20, 256))
    series[120:130] += 3
    scores, _ = model.predict_scores(series)
    candidates = model.candidates(scores)
    evaluated = scores[np.isfinite(scores)]
    assert evaluated.size and np.isfinite(evaluated).all()
    print(f"demo: evaluated scores={len(evaluated)}, candidates={len(candidates)}")


if __name__ == "__main__":
    _demo()
