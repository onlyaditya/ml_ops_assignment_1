"""Request logging and Prometheus metrics for the serving API.

Logs are one JSON object per line on stdout, so `kubectl logs` / `docker logs`
stay greppable and any log collector can parse them without regexes.
Patient feature values are never logged, only request metadata and the
model's output.

Metrics (scraped from GET /metrics):
    heart_api_requests_total{method, endpoint, status}
    heart_api_request_duration_seconds{endpoint}          histogram
    heart_api_predictions_total{prediction, risk_level}
    heart_api_prediction_probability                      histogram of P(disease)
    heart_api_validation_errors_total{field}              which inputs clients get wrong
    heart_api_model_info{model_version, threshold}        always 1, carries labels
"""
import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "heart_api_requests_total", "HTTP requests handled",
    ["method", "endpoint", "status"])
LATENCY = Histogram(
    "heart_api_request_duration_seconds", "Request latency in seconds",
    ["endpoint"],
    # model inference is ~1-5 ms, so start the buckets low
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
PREDICTIONS = Counter(
    "heart_api_predictions_total", "Predictions returned",
    ["prediction", "risk_level"])
PROBABILITY = Histogram(
    "heart_api_prediction_probability", "Predicted P(disease) - watch for drift",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
VALIDATION_ERRORS = Counter(
    "heart_api_validation_errors_total", "Rejected inputs by field", ["field"])
MODEL_INFO = Gauge(
    "heart_api_model_info", "Model currently served", ["model_version", "threshold"])


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON line; extra fields go in `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc)
                          .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    # our middleware logs every request; uvicorn's plain-text access log would duplicate it
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
