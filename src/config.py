"""Central place for paths and dataset schema, shared by all modules."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"

# Final packaged model (what the API serves)
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

# Experiment tracking: local SQLite backend (enables the model registry);
# override with MLFLOW_TRACKING_URI to point at a remote server
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# Column order as documented in the UCI heart-disease.names file
COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG result (0-2)
    "thalach",   # max heart rate achieved
    "exang",     # exercise-induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment (1-3)
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # original diagnosis 0-4
]
TARGET = "target"  # binary: 0 = no disease, 1 = disease (num > 0)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
