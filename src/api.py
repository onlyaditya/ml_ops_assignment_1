"""FastAPI service that serves the packaged heart-disease model.

Run locally:
    uvicorn src.api:app --reload                 # http://127.0.0.1:8000/docs

Endpoints:
    GET  /health       liveness/readiness (used by Docker and Kubernetes probes)
    GET  /model-info   what is being served and how well it is expected to do
    POST /predict      one patient record (JSON) -> prediction + confidence
    GET  /metrics      Prometheus metrics (see src/monitoring.py)

Settings (environment variables):
    MODEL_PATH               path to model.joblib (default: models/model.joblib)
    PREDICTION_THRESHOLD     P(disease) cut-off for class 1 (default: 0.5)
    LOG_LEVEL                default INFO; DEBUG also logs health checks
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response
from starlette.routing import Match

from src.config import METADATA_PATH, MODEL_PATH
from src.data import VALID_RANGES
from src.model_io import load_metadata, load_model
from src.monitoring import (
    LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    PROBABILITY,
    REQUESTS,
    VALIDATION_ERRORS,
    configure_logging,
)
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
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    # load once at start-up, not per request
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    app.state.model = load_model(model_path)
    meta_path = model_path.with_name(METADATA_PATH.name)
    app.state.metadata = load_metadata(meta_path) if meta_path.exists() else {}
    app.state.threshold = float(os.getenv("PREDICTION_THRESHOLD", DEFAULT_THRESHOLD))
    meta = app.state.metadata
    app.state.model_version = (
        f"{meta.get('model_name', 'unknown')}-v{meta.get('registered_version', '?')}")
    MODEL_INFO.labels(app.state.model_version, str(app.state.threshold)).set(1)
    logger.info("model loaded", extra={"fields": {
        "model_version": app.state.model_version, "model_path": str(model_path),
        "threshold": app.state.threshold}})
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts presence of heart disease from 13 UCI patient features.",
    version="1.0.0",
    lifespan=lifespan,
)

# endpoints whose every call would only add noise to the logs
QUIET_PATHS = {"/health", "/metrics"}


def _route_template(request: Request) -> str:
    """'/predict' etc. for known routes, 'other' for anything else, so random
    URLs (scanners, typos) cannot create unlimited metric label values."""
    for route in app.routes:
        if route.matches(request.scope)[0] == Match.FULL:
            return route.path
    return "other"


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Time, count and log every request; tag it with a request id."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = _route_template(request)
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        level = logging.DEBUG if endpoint in QUIET_PATHS and status < 400 else logging.INFO
        logger.log(level, "request", extra={"fields": {
            "request_id": request_id, "method": request.method, "path": request.url.path,
            "status": status, "latency_ms": round(elapsed * 1000, 2),
            "client": request.client.host if request.client else None}})


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Count and log which fields were rejected, then return FastAPI's usual 422."""
    fields = sorted({str(err["loc"][-1]) for err in exc.errors() if err.get("loc")})
    for field in fields:
        VALIDATION_ERRORS.labels(field).inc()
    logger.warning("validation error", extra={"fields": {
        "request_id": getattr(request.state, "request_id", None), "fields": fields}})
    return await request_validation_exception_handler(request, exc)


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
def predict(patient: PatientRecord, request: Request):
    result = score_records(app.state.model, [patient.model_dump()],
                           app.state.threshold)[0]
    PREDICTIONS.labels(str(result["prediction"]), result["risk_level"]).inc()
    PROBABILITY.observe(result["probability_disease"])
    # model output only - the patient's feature values are deliberately not logged
    logger.info("prediction", extra={"fields": {
        "request_id": request.state.request_id, "prediction": result["prediction"],
        "probability_disease": result["probability_disease"],
        "risk_level": result["risk_level"], "model_version": app.state.model_version}})
    return {**result, "model_version": app.state.model_version}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
