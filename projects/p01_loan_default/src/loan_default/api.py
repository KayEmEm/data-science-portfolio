"""Scoring API: one applicant's features in, calibrated probability of default and decision out."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from loan_default.config import ARTIFACTS_DIR

app = FastAPI(title="Loan default risk scoring", version="0.1.0")


class ScoreRequest(BaseModel):
    features: dict[str, float | int | str | bool | None] = Field(
        description=(
            "Feature values by column name, as in the feature table; missing columns are treated as unknown"
        )
    )


class ScoreResponse(BaseModel):
    probability_of_default: float
    decision: str
    threshold: float
    model: str
    unknown_features: list[str]


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    path = Path(os.environ.get("MODEL_PATH", ARTIFACTS_DIR / "model.joblib"))
    return joblib.load(path)


@app.get("/health")
def health() -> dict:
    bundle = load_bundle()
    return {"status": "ok", "model": bundle["model"].name, "data": bundle["data"]}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    bundle = load_bundle()
    model = bundle["model"]
    known = set(model.preprocessor.columns)
    frame = pd.DataFrame([request.features])
    p = float(model.predict(frame)[0])
    return ScoreResponse(
        probability_of_default=p,
        decision="approve" if p < bundle["threshold"] else "reject",
        threshold=bundle["threshold"],
        model=model.name,
        unknown_features=sorted(set(request.features) - known),
    )
