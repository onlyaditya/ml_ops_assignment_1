"""FastAPI service that serves the packaged heart-disease model.

Run locally:
    uvicorn src.api:app --reload                 # http://127.0.0.1:8000/docs

Endpoints:
    GET  /health       liveness/readiness (used by Docker and Kubernetes probes)
    GET  /model-info   what is being served and how well it is expected to do
    POST /predict      one patient record (JSON) -> prediction + confidence

Settings (environment variables):
    MODEL_PATH               path to model.joblib (default: models/model.joblib)
    PREDICTION_THRESHOLD     P(disease) cut-off for class 1 (default: 0.5)
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import METADATA_PATH, MODEL_PATH
from src.data import VALID_RANGES
from src.model_io import load_metadata, load_model
from src.predict import DEFAULT_THRESHOLD, score_records

logger = logging.getLogger("heart_api")


def _range(col: str) -> dict:
    """Reuse the cleaning step's plausible clinical ranges as API validation."""
    low, high = VALID_RANGES[col]
    return {"ge": low, "le": high}


class PatientRecord(BaseModel):
    """Raw inputs exactly as in the UCI data. `ca` and `thal` may be null
    (they are the fields missing in the real dataset; the model imputes them)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 1, "oldpeak": 2.3, "slope": 2,
        "ca": 2, "thal": 7,
    }})

    age: float = Field(..., **_range("age"), description="years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="chest pain: 1 typical angina, 2 atypical, 3 non-anginal, "
                         "4 asymptomatic")
    trestbps: float = Field(..., **_range("trestbps"), description="resting BP, mm Hg")
    chol: float = Field(..., **_range("chol"), description="serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="resting ECG result")
    thalach: float = Field(..., **_range("thalach"), description="max heart rate")
    exang: Literal[0, 1] = Field(..., description="exercise-induced angina")
    oldpeak: float = Field(..., **_range("oldpeak"), description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="slope of peak exercise ST")
    ca: Optional[int] = Field(None, **_range("ca"), description="major vessels 0-3")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect")


class Prediction(BaseModel):
    # allow the `model_version` field name (pydantic reserves `model_*` by default)
    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal[0, 1]
    label: str
    confidence: float = Field(..., description="probability of the predicted class")
    probability_disease: float
    risk_level: Literal["low", "moderate", "high"]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load once at start-up, not per request
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    app.state.model = load_model(model_path)
    meta_path = model_path.with_name(METADATA_PATH.name)
    app.state.metadata = load_metadata(meta_path) if meta_path.exists() else {}
    app.state.threshold = float(os.getenv("PREDICTION_THRESHOLD", DEFAULT_THRESHOLD))
    meta = app.state.metadata
    app.state.model_version = (
        f"{meta.get('model_name', 'unknown')}-v{meta.get('registered_version', '?')}")
    logger.info("Loaded %s from %s (threshold %.2f)",
                app.state.model_version, model_path, app.state.threshold)
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts presence of heart disease from 13 UCI patient features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"service": app.title, "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": app.state.model_version}


@app.get("/model-info")
def model_info():
    meta = app.state.metadata
    return {
        "model_version": app.state.model_version,
        "hyperparameters": meta.get("best_params"),
        "trained_on_rows": meta.get("trained_on_rows"),
        "features": meta.get("features"),
        "threshold": app.state.threshold,
        "estimated_performance": meta.get("estimated_performance"),
        "created_at": meta.get("created_at"),
    }


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientRecord):
    result = score_records(app.state.model, [patient.model_dump()],
                           app.state.threshold)[0]
    return {**result, "model_version": app.state.model_version}
