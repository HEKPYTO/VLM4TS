import base64
import io
import json
import os

import numpy as np
from PIL import Image

from .render import series_to_pil


def call_vlm(full_img: Image.Image, prompt: str):
    """Return a JSON candidate mask from gpt-4o-mini, or None on fallback."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        buffer = io.BytesIO()
        full_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        response = OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("["), text.rfind("]") + 1
        return json.loads(text[start:end] if start != -1 and end > start else text)
    except Exception:
        return None


def refine(scores, candidates, series):
    """Filter candidate indices with one VLM call, preserving fallback candidates."""
    candidates = list(candidates)
    if not candidates:
        return []
    try:
        image = series_to_pil(series, 224)
    except (TypeError, ValueError):
        return candidates
    score_array = np.asarray(scores, dtype=float).ravel()
    score_preview = [
        (int(candidate), round(float(score_array[candidate]), 3))
        for candidate in candidates
        if isinstance(candidate, (int, np.integer))
        and 0 <= candidate < len(score_array)
        and np.isfinite(score_array[candidate])
    ][:20]
    prompt = (
        "You are a time-series anomaly expert. Full series plotted as a line chart. "
        f"Candidate anomaly indices (0-based): {candidates}. Candidate scores: {score_preview}. "
        "Which candidates are true anomalies considering global context? Return a JSON list "
        "of the same length with 1=keep and 0=drop, for example [1,0,1]."
    )
    mask = call_vlm(image, prompt)
    if (
        not isinstance(mask, list)
        or len(mask) != len(candidates)
        or any(type(value) is not int or value not in (0, 1) for value in mask)
    ):
        return candidates
    return [candidate for candidate, keep in zip(candidates, mask, strict=True) if keep]
