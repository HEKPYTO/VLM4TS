# VLM4TS — Zero-Shot Time Series Anomaly Detection via Vision-Language Models

> **Paper:** He et al. *Harnessing Vision-Language Models for Time Series Anomaly Detection* — AAAI 2026 Oral (arXiv:2506.06836) · [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/39319) · [Original repo](https://github.com/ZLHe0/VLM4TS)

Training-free, CPU-only TSAD: render time series as line plots → **ViT-B/32** screening → **VLM** verification (single API call). No training, no fine-tuning.

## How it works

1. **ViT4TS screening** — sliding window (224) → plot → CLIP ViT-B/32 embedding → anomaly score = `1 - cosine(median)` → candidates at `α=0.01` top quantile.
2. **VLM verification** — global plot + candidate intervals → one `gpt-4o-mini` call → filtered intervals (36× token saving vs per-window VLM).

## Quick start

```bash
uv sync --python 3.11   # or pip install -e .
pytest -q
python -m src.vit4ts --demo
streamlit run app.py
```

Set `OPENAI_API_KEY` to enable VLM verification (otherwise ViT4TS-only mode).

## Project structure

```
src/render.py      # series → 224×224 PIL plot
src/vit4ts.py      # ViT-B/32 screening
src/vlm_refine.py  # VLM single-call verification
app.py             # Streamlit demo (upload CSV → scores → intervals)
scripts/bench.py   # quick benchmark (synthetic + TSB-AD-U if present)
```

## Benchmark

```bash
python scripts/bench.py --quick          # synthetic smoke test (<10 s)
python scripts/bench.py --dataset TSB-AD-U --metric F1-max
```

## References

- He et al., AAAI 2026 Oral. `10.1609/aaai.v40i26.39319`
- `ZLHe0/VLM4TS` (MIT)

