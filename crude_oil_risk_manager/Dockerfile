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

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "ui.app:server"]
