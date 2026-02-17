FROM ghcr.io/astral-sh/uv:0.10 AS uv

FROM python:3.13-slim-trixie AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM base AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /opt/biaas

RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/opt/biaas/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/opt/biaas/pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

FROM base AS runtime

ENV STREAMLIT_SERVER_HEADLESS=true

WORKDIR /opt/biaas

RUN adduser --system --home /home/biaas --group biaas

RUN mkdir -p /opt/biaas/data && chown biaas:biaas /opt/biaas/data

COPY --from=builder --chown=biaas:biaas /opt/biaas/.venv ./.venv
COPY --chown=biaas:biaas app/ ./app/
COPY --chown=biaas:biaas ./streamlit_app.py ./

ENV PATH="/opt/biaas/.venv/bin:$PATH"

USER biaas

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8501/ || exit 1

CMD [ "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0" ]
