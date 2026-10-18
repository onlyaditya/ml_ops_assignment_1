"""Tests for data acquisition and cleaning (src/data.py)."""
import numpy as np
import pandas as pd

from src import data
from src.config import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, TARGET


def test_load_raw_parses_question_marks_as_missing(tmp_path):
    raw_file = tmp_path / "raw.data"
    raw_file.write_text(
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,?,?,2\n"
    )
    df = data.load_raw(raw_file)
    assert df.shape == (2, 14)
    assert df.loc[1, ["ca", "thal"]].isna().all()
    assert df.loc[0, "age"] == 63.0


def test_clean_data_binarises_target(raw_df):
    clean = data.clean_data(raw_df)
    assert "num" not in clean.columns
    assert set(clean[TARGET].unique()) <= {0, 1}
    expected = (raw_df.drop_duplicates()["num"] > 0).astype(int).to_numpy()
    np.testing.assert_array_equal(clean[TARGET].to_numpy(), expected)


def test_clean_data_column_order_and_types(raw_df):
    clean = data.clean_data(raw_df)
    assert list(clean.columns) == FEATURES + [TARGET]
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        assert str(clean[col].dtype) == "Int64"


def test_clean_data_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(10)], ignore_index=True)
    assert len(data.clean_data(doubled)) == len(data.clean_data(raw_df))


def test_clean_data_nulls_out_of_range_values(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "chol"] = 5000      # impossible cholesterol
    bad.loc[1, "age"] = -3         # impossible age
    clean = data.clean_data(bad)
    assert pd.isna(clean.loc[0, "chol"])
    assert pd.isna(clean.loc[1, "age"])


def test_clean_data_keeps_missing_values_for_pipeline(raw_df):
    raw = raw_df.copy()
    raw.loc[:4, "thal"] = np.nan
    clean = data.clean_data(raw)
    assert clean["thal"].isna().sum() == 5
    assert len(clean) == len(raw)


def test_clean_data_does_not_modify_input(raw_df):
    before = raw_df.copy()
    data.clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_missing_report_lists_only_columns_with_gaps(raw_df):
    raw = raw_df.copy()
    raw.loc[:2, "ca"] = np.nan
    report = data.missing_report(data.clean_data(raw))
    assert report.to_dict() == {"ca": 3}


def test_download_skips_when_file_exists(tmp_path, monkeypatch):
    existing = tmp_path / "raw.data"
    existing.write_text("already here")
    monkeypatch.setattr(data.urllib.request, "urlretrieve",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    assert data.download_data(dest=existing) == existing
    assert existing.read_text() == "already here"


def test_download_fetches_when_forced(tmp_path, monkeypatch):
    dest = tmp_path / "sub" / "raw.data"
    calls = []

    def fake_retrieve(url, path):
        calls.append(url)
        path.write_text("downloaded")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake_retrieve)
    data.download_data(url="http://example/x", dest=dest, force=True)
    assert calls == ["http://example/x"]
    assert dest.read_text() == "downloaded"
