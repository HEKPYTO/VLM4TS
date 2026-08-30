# VLM4TS-inspired prototype

An independent educational prototype for time-series anomaly screening. It is inspired by [He et al.](https://arxiv.org/abs/2506.06836) and the [upstream VLM4TS repository](https://github.com/ZLHe0/VLM4TS), not an official implementation or a reproduction of its results.

The paper uses ViT-B/16, patch-level multi-scale cross-patch scoring, shared global y-limits, GPT-4o, and 11 NAB/NASA/YAHOO benchmark datasets. This prototype uses ViT-B/32 window embeddings, cosine distance to a median embedding, per-window autoscaling, and optional `gpt-4o-mini` verification.

## Run

Python 3.11 is the only supported runtime.

```bash
uv sync --extra dev
uv run python -m vlm4ts.vit4ts --demo
uv run python scripts/bench.py
uv run streamlit run app.py
```

The first scoring run downloads and caches the OpenAI CLIP checkpoint from Hugging Face; later runs can use that cache. `scripts/bench.py` is a synthetic smoke check only. It does not evaluate a public dataset or reproduce paper-reported results.

## VLM disclosure

VLM verification is optional. When enabled, the rendered uploaded series is sent to OpenAI using `gpt-4o-mini`; this may incur API costs. Without `OPENAI_API_KEY`, candidates are returned unchanged. By default, OpenAI documents up to 30-day abuse-monitoring retention; eligible customers may use modified or zero data-retention controls. See [Your data](https://developers.openai.com/api/docs/guides/your-data).

The app accepts a CSV `value` column, or its first column, provided all values are finite numbers.

## Layout

```
src/vlm4ts/       canonical package
app.py             Streamlit entry point
scripts/bench.py   synthetic smoke check
```

## Verification

```bash
uv lock --check && uv sync --extra dev --frozen
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev python -m pytest -q
uv run --extra dev pre-commit run --all-files
uv export --frozen --no-dev --no-emit-project --no-annotate --no-header | uv run --extra dev pip-audit -r /dev/stdin --disable-pip --no-deps
uv build
uv pip install --python .venv/bin/python --no-deps --reinstall dist/*.whl
uv run --no-sync python -c "import vlm4ts; print('wheel ok')"
docker compose config --quiet
docker build -t vlm4ts-public-readiness:check .
docker run --rm vlm4ts-public-readiness:check python -c "import vlm4ts; print('container ok')"
```

CI is the clean Docker build gate.
