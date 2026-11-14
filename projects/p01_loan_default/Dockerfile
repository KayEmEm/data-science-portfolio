# Scoring API image. Build from the repository root:
#   docker build -f projects/p01_loan_default/Dockerfile -t loan-default-api .
# or run `make docker` in this folder. Requires artifacts/model.joblib from `make train`.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The same layout as the repository, so that the path source ../../common resolves.
WORKDIR /app/projects/p01_loan_default
COPY common /app/common
COPY projects/p01_loan_default/pyproject.toml projects/p01_loan_default/uv.lock \
     projects/p01_loan_default/README.md ./
COPY projects/p01_loan_default/src ./src

# Locked versions, runtime dependencies only, packages copied into the environment.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --locked --no-dev --no-editable
ENV PATH="/app/.venv/bin:$PATH"

COPY projects/p01_loan_default/artifacts/model.joblib /app/model.joblib
ENV MODEL_PATH=/app/model.joblib

RUN useradd --create-home scorer
USER scorer
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "loan_default.api:app", "--host", "0.0.0.0", "--port", "8000"]
