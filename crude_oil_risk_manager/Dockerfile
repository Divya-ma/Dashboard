# syntax=docker/dockerfile:1

# ---------- Stage 1: build dependencies ----------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime image ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Bring in installed dependencies from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

EXPOSE 8050

ENV PYTHONUNBUFFERED=1

# ONE worker, several threads. The dashboard keeps its caches, locks and background downloads (OI, ATR, churn, curves, backups)
# inside the process, so a second worker would run every job twice, fight over the same files and split the shared API rate
# budget. Scale with threads, never with --workers. (WEB_CONCURRENCY, which gunicorn also reads, must stay unset or 1.)
ENV WEB_CONCURRENCY=1

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 CMD ["python", "tools/healthcheck.py"]

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", "--bind", "0.0.0.0:8050", "ui.app:server"]
