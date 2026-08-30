# VLM4TS-inspired prototype

An independent educational prototype for time-series anomaly screening, inspired by [He et al.](https://arxiv.org/abs/2506.06836) and the [upstream VLM4TS repository](https://github.com/ZLHe0/VLM4TS). It is not an official implementation or a reproduction of the paper's results.

![Illustrative synthetic score curve](assets/score_curve.png)

*Illustrative synthetic output, not an evaluation result.*

## How it works

The prototype renders each sliding time-series window as a line plot, embeds it with CLIP ViT-B/32, and scores its cosine distance from the median window embedding. High-score windows become anomaly candidates. An optional `gpt-4o-mini` call receives one rendered full-series image and returns a keep/drop mask for those candidates.

## Quick start

[Python 3.11](https://www.python.org/) is the only supported runtime. Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --extra dev
uv run python -m vlm4ts.vit4ts --demo
uv run python scripts/bench.py
uv run streamlit run app.py
```

The first scoring run downloads and caches the OpenAI CLIP checkpoint from Hugging Face. Later runs reuse that cache.

## Input, privacy, and security

The Streamlit app accepts a CSV containing a numeric `value` column, or uses its first column. All values must be finite numbers.

VLM verification is optional. When enabled, the rendered uploaded series is sent to OpenAI using `gpt-4o-mini`; this may incur API costs. Without `OPENAI_API_KEY`, candidates are returned unchanged. Do not include secrets in uploaded data. OpenAI documents its retention controls in [Your data](https://developers.openai.com/api/docs/guides/your-data).

Use this repository's Forgejo issues for questions and non-sensitive bug reports. Report vulnerabilities privately to `zoonyanapat@gmail.com`; do not post exploit details publicly. The default branch is supported until releases exist, after which only the latest release is supported.

## Scope and limitations

| | Paper | This prototype |
| --- | --- | --- |
| Vision encoder | ViT-B/16 | CLIP ViT-B/32 |
| Scoring | Patch-level, multi-scale cross-patch | Window cosine distance from a median embedding |
| Plot scaling | Shared global y-limits | Per-window autoscaling |
| VLM | GPT-4o | Optional `gpt-4o-mini` |
| Evaluation | 11 NAB/NASA/YAHOO datasets | Synthetic smoke data only |

`scripts/bench.py` checks that the pipeline runs. It does not evaluate a public dataset, reproduce paper-reported results, or substantiate a performance claim.

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

## License

[MIT](LICENSE)
