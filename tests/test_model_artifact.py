"""Sanity checks on the packaged model in models/ (what the API will serve)."""
import pytest

from src.config import FEATURES, METADATA_PATH, MODEL_PATH
from src.model_io import load_metadata, load_model

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(),
                                reason="no packaged model - run `python -m src.train`")


def test_metadata_matches_feature_schema():
    meta = load_metadata(METADATA_PATH)
    assert meta["features"] == FEATURES
    assert meta["estimated_performance"]["cv_roc_auc_mean"] > 0.8


def test_model_predicts_probabilities(patient_df):
    model = load_model()
    proba = model.predict_proba(patient_df)
    assert proba.shape == (1, 2)
    assert proba.sum() == pytest.approx(1.0)


def test_model_tolerates_missing_and_unseen_values(patient_df):
    model = load_model()
    odd = patient_df.copy()
    odd[["chol", "ca", "thal"]] = float("nan")
    odd["cp"] = 9.0
    assert model.predict(odd)[0] in (0, 1)


def test_high_risk_patient_scores_higher_than_low_risk(patient_df):
    model = load_model()
    low = patient_df.copy()
    low[["cp", "exang", "oldpeak", "ca", "thal", "thalach"]] = [3, 0, 0.0, 0, 3, 175]
    high = patient_df.copy()
    high[["cp", "exang", "oldpeak", "ca", "thal", "thalach"]] = [4, 1, 3.5, 3, 7, 105]
    assert model.predict_proba(high)[0, 1] > model.predict_proba(low)[0, 1]
