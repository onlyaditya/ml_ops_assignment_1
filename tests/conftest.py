"""Shared fixtures: small synthetic datasets so tests never need the network."""
import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS, FEATURES, TARGET


def _random_raw(n: int, seed: int = 0) -> pd.DataFrame:
    """Raw-format rows (14 UCI columns incl. 0-4 `num`) with realistic values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 78, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(95, 190, n).astype(float),
        "chol": rng.integers(150, 400, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(90, 200, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    # make the label depend on a few features so models can learn something
    # (cast to int: adding boolean Series would be a logical OR, not a count)
    risk = ((df["cp"] == 4).astype(int) + (df["exang"] == 1).astype(int)
            + (df["thal"] == 7).astype(int) + (df["ca"] > 0).astype(int))
    df["num"] = np.where(risk >= 2, rng.integers(1, 5, n), 0)
    return df[COLUMNS]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return _random_raw(120)


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from src.data import clean_data
    return clean_data(raw_df)


@pytest.fixture
def xy(clean_df):
    from src.features import to_feature_frame
    return to_feature_frame(clean_df), clean_df[TARGET].astype(int)


@pytest.fixture
def patient() -> dict:
    """One valid patient record (first row of the real Cleveland data)."""
    return dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2,
                thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6)


@pytest.fixture
def patient_df(patient) -> pd.DataFrame:
    return pd.DataFrame([patient])[FEATURES].astype(float)
