"""Feature preprocessing shared by training and inference.

The preprocessor is part of the saved model pipeline, so raw patient
records (with possible missing values) can be passed straight to
`model.predict_proba` both in training and in the API.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and encode the three feature groups.

    - numeric:     median imputation (robust to outliers) + standard scaling
    - binary:      most-frequent imputation, passed through as 0/1
    - categorical: most-frequent imputation + one-hot encoding; unseen
                   codes at inference become all-zeros instead of an error
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", binary, BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs in a fixed order with a consistent float dtype."""
    return df[FEATURES].astype(float)


def load_xy(path=CLEAN_DATA_PATH):
    """Load the cleaned CSV and split it into features X and target y."""
    df = pd.read_csv(path)
    return to_feature_frame(df), df[TARGET].astype(int)
