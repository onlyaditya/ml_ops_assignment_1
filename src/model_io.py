"""Save and load the final packaged model.

The model is stored in two formats:
  - models/model.joblib     lightweight, loaded by the API (no MLflow needed)
  - models/mlflow_model/    MLflow format: MLmodel, signature and pinned
                            requirements, loadable with mlflow.pyfunc
plus models/model_metadata.json describing how it was produced.
"""
import json
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import sklearn
from mlflow.models import infer_signature

from src.config import (
    FEATURES,
    METADATA_PATH,
    MLFLOW_MODEL_DIR,
    MODEL_PATH,
    MODELS_DIR,
)


def save_final_model(model, X_sample, metadata: dict) -> dict:
    """Persist the fitted pipeline in joblib + MLflow formats with metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    if MLFLOW_MODEL_DIR.exists():
        shutil.rmtree(MLFLOW_MODEL_DIR)
    mlflow.sklearn.save_model(
        model,
        MLFLOW_MODEL_DIR,
        signature=infer_signature(X_sample, model.predict(X_sample)),
        input_example=X_sample.head(3),
    )

    metadata = {
        **metadata,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_model(path=MODEL_PATH):
    """Load the fitted preprocessing + classifier pipeline."""
    return joblib.load(path)


def load_metadata(path=METADATA_PATH) -> dict:
    return json.loads(path.read_text())
