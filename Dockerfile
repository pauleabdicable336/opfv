# Reproducible CPU environment (extend with NVIDIA base + CUDA PyTorch for GPU OPL).
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src ./src
COPY tests ./tests

RUN uv sync --frozen --all-extras

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Smoke: quick synthetic OPE (override CMD for full experiments)
CMD ["uv", "run", "python", "-m", "opfv.run", "experiment=quick_synthetic_ope"]
