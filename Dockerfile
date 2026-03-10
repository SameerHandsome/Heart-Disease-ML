# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Builder
# Compiles native extensions: LightGBM, CatBoost, XGBoost, asyncpg.
# Cached until requirements.txt changes. Code changes skip this stage.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ cmake libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# requirements BEFORE code — pip install is only re-run when deps change
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Final image
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Non-root for security
RUN groupadd -r appgroup && useradd -r -u 10001 -g appgroup appuser

# Code LAST — never invalidates lib cache
COPY app/     ./app/
COPY ml/      ./ml/
COPY monitor/ ./monitor/

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]