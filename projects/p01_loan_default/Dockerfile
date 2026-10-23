# Scoring API image. Build from the repository root:
#   docker build -f projects/p01_loan_default/Dockerfile -t loan-default-api .
# or run `make docker` in this folder. Requires artifacts/model.joblib from `make train`.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY common /app/common
COPY projects/p01_loan_default/pyproject.toml projects/p01_loan_default/README.md /app/project/
COPY projects/p01_loan_default/src /app/project/src
RUN pip install --no-cache-dir /app/common /app/project

COPY projects/p01_loan_default/artifacts/model.joblib /app/model.joblib
ENV MODEL_PATH=/app/model.joblib

RUN useradd --create-home scorer
USER scorer
EXPOSE 8000
CMD ["uvicorn", "loan_default.api:app", "--host", "0.0.0.0", "--port", "8000"]
