"""Tests for model training, evaluation and packaging (src/train.py, src/model_io.py)."""
import json

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import ci_report, model_io, train


@pytest.fixture
def tiny_specs():
    """One-candidate grids so training tests run in seconds."""
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=500),
            {"clf__C": [1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=20, random_state=0),
            {"clf__max_depth": [3]},
        ),
    }


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every file train.main() writes into a temp directory."""
    data_path = tmp_path / "clean.csv"
    models = tmp_path / "models"
    monkeypatch.setattr(train, "CLEAN_DATA_PATH", data_path)
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(train, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(train, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlruns")
    monkeypatch.setattr(model_io, "MODELS_DIR", models)
    monkeypatch.setattr(model_io, "MODEL_PATH", models / "model.joblib")
    monkeypatch.setattr(model_io, "METADATA_PATH", models / "model_metadata.json")
    monkeypatch.setattr(model_io, "MLFLOW_MODEL_DIR", models / "mlflow_model")
    monkeypatch.setattr(ci_report, "COMPARISON_PATH", tmp_path / "reports" / "model_comparison.csv")
    return tmp_path


def test_model_specs_have_at_least_two_models_with_valid_grids():
    specs = train.get_model_specs()
    assert len(specs) >= 2
    for name, (estimator, grid) in specs.items():
        assert hasattr(estimator, "predict_proba"), name
        assert grid and all(k.startswith("clf__") for k in grid), name


def test_pipeline_contains_preprocessor_and_classifier():
    pipe = train.make_pipeline(LogisticRegression())
    assert [name for name, _ in pipe.steps] == ["preprocess", "clf"]


def test_tune_model_and_cv_summary(xy):
    X, y = xy
    search = train.tune_model(LogisticRegression(solver="liblinear"),
                              {"clf__C": [0.1, 1.0]}, X, y)
    summary = train.cv_summary(search)
    for metric in train.SCORING:
        assert 0.0 <= summary[f"cv_{metric}_mean"] <= 1.0
        assert summary[f"cv_{metric}_std"] >= 0.0
    assert search.best_params_["clf__C"] in (0.1, 1.0)


def test_evaluate_returns_metrics_in_unit_range(xy):
    X, y = xy
    model = train.make_pipeline(LogisticRegression(solver="liblinear")).fit(X, y)
    metrics = train.evaluate(model, X, y)
    assert set(metrics) == {"test_accuracy", "test_precision", "test_recall",
                            "test_f1", "test_roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_model_learns_better_than_chance(xy):
    X, y = xy
    model = train.make_pipeline(LogisticRegression(solver="liblinear")).fit(X, y)
    assert train.evaluate(model, X, y)["test_roc_auc"] > 0.7


def test_strip_prefix():
    assert train.strip_prefix({"clf__C": 1, "clf__penalty": "l2"}) == {"C": 1, "penalty": "l2"}


def test_full_training_run_end_to_end(isolated_paths, clean_df, tiny_specs, monkeypatch,
                                      patient_df):
    """Train -> select -> refit on all rows -> save -> reload -> predict -> report."""
    clean_df.to_csv(isolated_paths / "clean.csv", index=False)
    monkeypatch.setattr(train, "load_xy", lambda: _load(isolated_paths / "clean.csv"))
    monkeypatch.setattr(train, "get_model_specs", lambda: tiny_specs)

    results, final_model = train.main()

    assert set(results["model"]) == set(tiny_specs)
    assert results["cv_roc_auc_mean"].is_monotonic_decreasing  # best first

    models_dir = isolated_paths / "models"
    for name in ("model.joblib", "model_metadata.json", "mlflow_model/MLmodel"):
        assert (models_dir / name).exists(), name

    metadata = json.loads((models_dir / "model_metadata.json").read_text())
    assert metadata["model_name"] == results.iloc[0]["model"]
    assert metadata["trained_on_rows"] == len(clean_df)  # option B: refit on all rows
    assert metadata["registered_version"] >= 1

    reloaded = model_io.load_model(models_dir / "model.joblib")
    proba = reloaded.predict_proba(patient_df)[0]
    assert proba.sum() == pytest.approx(1.0)
    assert (reloaded.predict(patient_df) == final_model.predict(patient_df)).all()

    report = ci_report.build_report(results, metadata, min_roc_auc=0.0)
    assert "Quality gate" in report and "PASSED" in report


def test_quality_gate_threshold():
    metadata = {"estimated_performance": {"cv_roc_auc_mean": 0.80}}
    assert ci_report.passes_gate(metadata, 0.75)
    assert not ci_report.passes_gate(metadata, 0.85)


def _load(path):
    from src.features import load_xy
    return load_xy(path)
