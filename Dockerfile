FROM python:3.13-slim AS base

WORKDIR /app

# System deps: LightGBM (libgomp), Prophet/Stan (gcc/g++), psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependency layer (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Source + artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY data/hdb_resale.parquet ./data/
COPY data/hdb_monthly_agg.parquet ./data/
COPY data/forecasts.parquet ./data/

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# ── API only ──────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Dashboard only ────────────────────────────────────────────────────────────
FROM base AS dashboard
EXPOSE 7860
CMD ["python", "-m", "src.dashboard.app"]

# ── Combined deploy (Cloud Run) — Gradio dashboard + embedded model serving ──
FROM base AS deploy
EXPOSE 7860
ENV PORT=7860
CMD ["python", "-m", "src.dashboard.app"]
