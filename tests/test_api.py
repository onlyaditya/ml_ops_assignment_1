"""API contract tests against the packaged model (FastAPI TestClient, no server)."""
import json

import pytest
from fastapi.testclient import TestClient

from src.config import MODEL_PATH, PROJECT_ROOT
from src.monitoring import JsonFormatter
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


# --- monitoring: /metrics and structured request logs -------------------------

def _metric(client, line_start: str) -> float:
    """Value of the first exposition line starting with `line_start` (0 if absent)."""
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_count_predictions(client, high_risk):
    key = 'heart_api_predictions_total{prediction="1",risk_level="high"}'
    before = _metric(client, key)
    client.post("/predict", json=high_risk)
    assert _metric(client, key) == before + 1
    assert _metric(client, "heart_api_prediction_probability_count") >= 1


def test_metrics_count_validation_errors_by_field(client, patient):
    key = 'heart_api_validation_errors_total{field="cp"}'
    before = _metric(client, key)
    patient["cp"] = 9
    client.post("/predict", json=patient)
    assert _metric(client, key) == before + 1


def test_unknown_paths_share_one_metric_label(client):
    client.get("/no-such-page")
    client.get("/another-typo")
    text = client.get("/metrics").text
    assert 'endpoint="other"' in text
    assert "no-such-page" not in text


def test_request_id_is_returned_and_propagated(client):
    assert len(client.get("/health").headers["X-Request-ID"]) == 12
    resp = client.get("/health", headers={"X-Request-ID": "trace-abc"})
    assert resp.headers["X-Request-ID"] == "trace-abc"


def test_prediction_log_is_json_without_patient_data(client, high_risk, caplog):
    caplog.set_level("INFO", logger="heart_api")
    client.post("/predict", json=high_risk, headers={"X-Request-ID": "log-test"})
    # render the captured records exactly as the stdout handler would
    records = [json.loads(JsonFormatter().format(r)) for r in caplog.records
               if r.name == "heart_api"]
    mine = [r for r in records if r.get("request_id") == "log-test"]
    assert {r["msg"] for r in mine} == {"prediction", "request"}
    request_log = next(r for r in mine if r["msg"] == "request")
    assert request_log["status"] == 200 and request_log["latency_ms"] > 0
    # no feature values (e.g. cholesterol) may leak into the logs
    assert all("chol" not in r for r in mine)
