FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    LEGAL_QA_ROOT=/app HF_HOME=/models/huggingface HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
COPY config ./config
COPY prompts ./prompts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev && useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
