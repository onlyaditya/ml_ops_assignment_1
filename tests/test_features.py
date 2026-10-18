"""Tests for the preprocessing pipeline (src/features.py)."""
import numpy as np
import pandas as pd
import pytest

from src.config import BINARY_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from src.features import build_preprocessor, load_xy, to_feature_frame


def test_to_feature_frame_orders_and_casts(clean_df):
    shuffled = clean_df[list(reversed(clean_df.columns))]
    X = to_feature_frame(shuffled)
    assert list(X.columns) == FEATURES
    assert all(dtype == np.float64 for dtype in X.dtypes)


def test_load_xy_splits_features_and_target(tmp_path, clean_df):
    path = tmp_path / "clean.csv"
    clean_df.to_csv(path, index=False)
    X, y = load_xy(path)
    assert list(X.columns) == FEATURES
    assert TARGET not in X.columns
    assert len(X) == len(y) == len(clean_df)


def test_preprocessor_output_has_no_missing_values(xy):
    X, _ = xy
    X = X.copy()
    X.loc[X.index[:5], ["chol", "ca", "thal", "fbs"]] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_preprocessor_scales_numeric_features(xy):
    X, _ = xy
    out = build_preprocessor().fit_transform(X)
    numeric = out[:, :len(NUMERIC_FEATURES)]
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-9)


def test_preprocessor_one_hot_encodes_categoricals(xy):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    names = list(pre.get_feature_names_out())
    # every observed category becomes its own column
    assert {"cat__cp_1.0", "cat__cp_4.0", "cat__thal_7.0"} <= set(names)
    expected_width = (len(NUMERIC_FEATURES) + len(BINARY_FEATURES)
                      + sum(X[c].nunique() for c in ["cp", "restecg", "slope", "thal"]))
    assert len(names) == expected_width


def test_preprocessor_ignores_unseen_categories(xy, patient_df):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    unseen = patient_df.copy()
    unseen["cp"] = 9.0  # chest-pain code never seen in training
    out = pre.transform(unseen)
    cp_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("cat__cp_")]
    assert out[0, cp_cols].sum() == 0


def test_preprocessor_rejects_missing_columns(xy):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    with pytest.raises((KeyError, ValueError)):
        pre.transform(pd.DataFrame(X.drop(columns=["age"])))
