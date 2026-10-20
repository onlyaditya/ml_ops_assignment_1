"""Inference: score patient records with the packaged model.

`score_records` is the single place where raw records become predictions;
the API (src/api.py) and this CLI both call it, so they cannot drift apart.

Usage:
    python -m src.predict samples/patient_high_risk.json             # local model
    python -m src.predict samples/patients.json --url http://localhost:8000
                                                       # same input, via the API

The JSON file may hold one patient object or a list of them.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src.features import to_feature_frame

LABELS = {0: "no heart disease", 1: "heart disease"}
DEFAULT_THRESHOLD = 0.5


def risk_level(p_disease: float) -> str:
    """Coarse band for the disease probability, easier to read than a number."""
    if p_disease < 0.3:
        return "low"
    if p_disease < 0.7:
        return "moderate"
    return "high"


def score_records(model, records: list[dict],
                  threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Predict class, confidence and risk band for each raw patient record.

    Missing fields may be given as None; the pipeline imputes them.
    `confidence` is the probability of the *predicted* class, so it is
    always >= 0.5 at the default threshold.
    """
    X = to_feature_frame(pd.DataFrame.from_records(records))
    p_disease = model.predict_proba(X)[:, 1]

    results = []
    for p in p_disease:
        p = float(p)
        prediction = int(p >= threshold)
        results.append({
            "prediction": prediction,
            "label": LABELS[prediction],
            "confidence": round(p if prediction else 1 - p, 4),
            "probability_disease": round(p, 4),
            "risk_level": risk_level(p),
        })
    return results


def _predict_via_api(url: str, records: list[dict]) -> list[dict]:
    results = []
    for record in records:
        req = urllib.request.Request(
            url.rstrip("/") + "/predict",
            data=json.dumps(record).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            results.append(json.loads(resp.read()))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict heart disease risk.")
    parser.add_argument("input", type=Path, help="JSON file: one patient or a list")
    parser.add_argument("--url", help="score via a running API instead of the local model")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text())
    records = data if isinstance(data, list) else [data]

    if args.url:
        try:
            results = _predict_via_api(args.url, records)
        except urllib.error.HTTPError as err:
            # e.g. 422: show the API's validation message, not a traceback
            sys.exit(f"API returned {err.code}: {err.read().decode()}")
    else:
        from src.model_io import load_model
        results = score_records(load_model(), records, args.threshold)

    json.dump(results if isinstance(data, list) else results[0], sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
