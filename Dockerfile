# Multi-stage production Dockerfile for Inference (Render & Cloud Ready)
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 user && \
    mkdir -p /app/data && \
    chown -R user:user /app

COPY --from=builder --chown=user:user /root/.local /home/user/.local
COPY --chown=user:user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

EXPOSE 8000

# Worker count. This service runs on Render's free plan (512MB RAM, 0.5 CPU).
# It was hard-coded to `--workers 4`, which forks four copies of a process that
# initialises a SQLite memory store and a provider key pool on startup. On a
# 512MB instance that is an out-of-memory kill waiting for load, and the OOM
# killer gives no useful log line — it is exactly the failure that made Forge's
# CI go red in Phase A1.
#
# Free-tier CPU is shared and already the bottleneck (measured p95 latency of
# ~12s), so extra workers buy no throughput here, they only multiply memory.
# One worker is the correct default; raise WEB_CONCURRENCY only on a paid
# instance that actually has the RAM for it.
ENV WEB_CONCURRENCY=1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
