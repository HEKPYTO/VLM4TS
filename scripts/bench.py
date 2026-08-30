#!/usr/bin/env python3
"""Synthetic smoke check; this is not a paper benchmark."""

import numpy as np

from vlm4ts.vit4ts import ViT4TS


def main():
    series = np.sin(np.linspace(0, 20, 256))
    series[120:130] += 3
    model = ViT4TS(alpha=0.01, window_size=224)
    scores, _ = model.predict_scores(series)
    candidates = model.candidates(scores)
    evaluated = scores[np.isfinite(scores)]
    assert evaluated.size and np.isfinite(evaluated).all()
    print(f"synthetic smoke: evaluated_scores={len(evaluated)} candidates={len(candidates)}")


if __name__ == "__main__":
    main()
