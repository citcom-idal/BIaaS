FROM ghcr.io/astral-sh/uv:0.10 AS uv

FROM python:3.13-slim-trixie AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r -g 1001 biaas && \
    useradd -r -u 1001 -g biaas biaas

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

RUN mkdir -p /opt/biaas/data /opt/biaas/.cache && \
    chown -R biaas:biaas /opt/biaas/data /opt/biaas/.cache

COPY --from=builder --chown=biaas:biaas /opt/biaas/.venv ./.venv
COPY --chown=biaas:biaas app/ ./app/
COPY --chown=biaas:biaas ./streamlit_app.py ./
COPY --chown=biaas:biaas ./build_index.py ./

ENV PATH="/opt/biaas/.venv/bin:$PATH" \
    HF_HOME=/opt/biaas/.cache \
    XDG_CACHE_HOME=/opt/biaas/.cache

ARG ROOT_PATH
ENV ROOT_PATH=${ROOT_PATH:-}

USER biaas

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501${ROOT_PATH}/healthz || exit 1

CMD [ "sh", "-c", "exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.baseUrlPath=$ROOT_PATH" ]
