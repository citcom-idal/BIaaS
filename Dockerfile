FROM ghcr.io/astral-sh/uv:0.10 AS uv

FROM python:3.13-slim-trixie AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

RUN groupadd -r -g 1001 biaas; \
    useradd -r -g biaas -u 1001 --home-dir=/opt/biaas --shell=/bin/bash biaas; \
    install --verbose --directory --owner biaas --group biaas --mode 755 /opt/biaas

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

WORKDIR /opt/biaas

RUN mkdir -p /opt/biaas/data && chown -R biaas:biaas /opt/biaas/data

COPY --from=builder --chown=biaas:biaas /opt/biaas/.venv ./.venv
COPY --chown=biaas:biaas app/ ./app/
COPY --chown=biaas:biaas scripts/ ./scripts/
COPY --chown=biaas:biaas ./streamlit_app.py ./

ARG ROOT_PATH

ENV PATH="/opt/biaas/.venv/bin:$PATH" \
    STREAMLIT_LOGGER_ENABLE_RICH=1 \
    STREAMLIT_SERVER_HEADLESS=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_BASE_URL_PATH=${ROOT_PATH} \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=0

USER biaas

RUN python -m scripts.preload_models

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-m", "scripts.healthcheck"]

CMD [ "streamlit", "run", "streamlit_app.py" ]
