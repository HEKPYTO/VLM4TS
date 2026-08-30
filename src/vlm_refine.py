import base64
import io
import json
import os
from PIL import Image
from src.render import series_to_pil



def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_vlm(full_img: Image.Image, prompt: str):
    """Call gpt-4o-mini with image + prompt, parse JSON list of 0/1.
    Returns list[int] mask. Falls back to no filtering if no API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        b64 = _pil_to_b64(full_img)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=500,
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        # extract JSON list from text
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return json.loads(text)
    except Exception:
        return None


def refine(scores, candidates, series):
    """Single VLM call to filter candidates. Returns filtered list.
    If VLM unavailable or returns None/invalid, return candidates unchanged (ViT4TS-only fallback)."""
    if not candidates:
        return []
    # render global view
    try:
        img = series_to_pil(series, 224)
    except Exception:
        return list(candidates)
    prompt = (
        f"You are a time-series anomaly expert. Full series plotted as line chart. "
        f"Candidate anomaly indices (0-based timesteps): {candidates}. "
        f"Scores: {[round(float(s), 3) for s in (scores or [])][:20]}. "
        f"Which candidates are true anomalies considering global context? "
        f"Return JSON list of same length as candidates with 1=keep, 0=drop. Example: [1,0,1]"
    )
    mask = call_vlm(img, prompt)
    if mask is None:
        return list(candidates)
    # mask should be same length as candidates
    if not isinstance(mask, list) or len(mask) != len(candidates):
        # try to handle per-series mask (len == len(series))
        if isinstance(mask, list) and len(mask) == len(series):
            return [c for c in candidates if 0 <= c < len(mask) and mask[c] == 1]
        return list(candidates)
    return [c for c, m in zip(candidates, mask) if m == 1]
