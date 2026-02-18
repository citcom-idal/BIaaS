FROM ghcr.io/astral-sh/uv:0.10 AS uv

FROM python:3.13-slim-trixie AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --system --home /home/biaas --group biaas

FROM base AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /home/biaas/code

RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/home/biaas/code/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/home/biaas/code/pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

FROM base AS runtime

ENV STREAMLIT_SERVER_HEADLESS=true

WORKDIR /home/biaas/code

RUN mkdir -p /home/biaas/code/data && chown biaas:biaas /home/biaas/code/data

COPY --from=builder --chown=biaas:biaas /home/biaas/code/.venv ./.venv
COPY --chown=biaas:biaas app/ ./app/
COPY --chown=biaas:biaas ./streamlit_app.py ./

ENV PATH="/home/biaas/code/.venv/bin:$PATH"

USER biaas

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

CMD [ "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0" ]
