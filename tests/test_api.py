"""API contract tests against the packaged model (FastAPI TestClient, no server)."""
import json

import pytest
from fastapi.testclient import TestClient

from src.config import MODEL_PATH, PROJECT_ROOT
from src.predict import main as predict_cli
from src.predict import risk_level

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(),
                                reason="no packaged model - run `python -m src.train`")

SAMPLES = PROJECT_ROOT / "samples"


@pytest.fixture(scope="module")
def client():
    from src.api import app
    with TestClient(app) as c:  # `with` runs the lifespan, i.e. loads the model
        yield c


@pytest.fixture
def high_risk():
    return json.loads((SAMPLES / "patient_high_risk.json").read_text())


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("logistic_regression")


def test_model_info(client):
    body = client.get("/model-info").json()
    assert len(body["features"]) == 13
    assert body["threshold"] == 0.5


def test_predict_returns_prediction_and_confidence(client, patient):
    resp = client.post("/predict", json=patient)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["risk_level"] in ("low", "moderate", "high")


def test_high_risk_sample_predicts_disease(client, high_risk):
    body = client.post("/predict", json=high_risk).json()
    assert body["prediction"] == 1
    assert body["label"] == "heart disease"
    assert body["confidence"] == body["probability_disease"]


def test_ca_and_thal_may_be_missing(client, patient):
    patient.pop("ca")
    patient["thal"] = None
    assert client.post("/predict", json=patient).status_code == 200


@pytest.mark.parametrize("field, value", [
    ("age", 500),          # outside the plausible clinical range
    ("cp", 9),             # not a valid chest-pain code
    ("sex", "male"),       # wrong type
    ("chol", None),        # only ca/thal may be null
    ("smoker", 1),         # unknown field (typo protection)
])
def test_invalid_input_rejected(client, patient, field, value):
    patient[field] = value
    assert client.post("/predict", json=patient).status_code == 422


def test_missing_required_field_rejected(client, patient):
    patient.pop("age")
    assert client.post("/predict", json=patient).status_code == 422


@pytest.mark.parametrize("p, band", [(0.1, "low"), (0.5, "moderate"), (0.9, "high")])
def test_risk_level_bands(p, band):
    assert risk_level(p) == band


def test_inference_cli_matches_api(client, high_risk, capsys):
    predict_cli([str(SAMPLES / "patient_high_risk.json")])
    cli = json.loads(capsys.readouterr().out)
    api = client.post("/predict", json=high_risk).json()
    assert cli["probability_disease"] == api["probability_disease"]
