FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir uv==0.12.7 && uv sync --frozen --no-dev --no-cache

COPY app.py ./
COPY scripts/ scripts/
COPY assets/ assets/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
