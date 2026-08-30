FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps for open-clip / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# use uv if available, else pip
RUN pip install --upgrade pip && pip install -e .

COPY src/ src/
COPY app.py ./
COPY scripts/ scripts/
COPY assets/ assets/ 2>/dev/null || true

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
