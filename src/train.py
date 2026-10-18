"""Train, tune and compare classifiers for heart disease prediction.

For each candidate model:
  1. grid-search hyperparameters with stratified 5-fold CV (refit on ROC-AUC)
  2. record the CV mean/std of accuracy, precision, recall, F1 and ROC-AUC
  3. evaluate the refitted best estimator once on a held-out 20% test set

The final model is chosen on CV ROC-AUC, not on the test set, so the test
score stays an unbiased estimate. The winning configuration is then refit
on ALL rows and packaged for serving.

Every invocation is tracked in MLflow as one parent run ("model-selection")
with one nested run per candidate model.

Usage:
    python -m src.train
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
"""
import hashlib
import json
import logging

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CLEAN_DATA_PATH,
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
)
from src.features import build_preprocessor, load_xy
from src.model_io import save_final_model

matplotlib.use("Agg")  # headless: works in CI and Docker
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# zero_division=0: heavily regularised candidates can predict no positives in
# a fold; score that as 0 precision explicitly instead of emitting warnings
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
SELECTION_METRIC = "roc_auc"

REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = PROJECT_ROOT / "screenshots" / "models"


def get_model_specs() -> dict:
    """Candidate estimators and their hyperparameter grids.

    Grid keys use the `clf__` prefix because each estimator sits at the
    end of a Pipeline behind the preprocessor.
    """
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=1000,
                               random_state=RANDOM_STATE),
            {
                "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.8, 1.0],
            },
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])


def tune_model(estimator, param_grid, X_train, y_train) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(estimator),
        param_grid,
        scoring=SCORING,
        refit=SELECTION_METRIC,
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def cv_summary(search: GridSearchCV) -> dict:
    """Mean and std of every CV metric for the best parameter set."""
    i = search.best_index_
    res = search.cv_results_
    summary = {}
    for metric in SCORING:
        summary[f"cv_{metric}_mean"] = float(res[f"mean_test_{metric}"][i])
        summary[f"cv_{metric}_std"] = float(res[f"std_test_{metric}"][i])
    return summary


def evaluate(model, X_test, y_test) -> dict:
    """Hold-out metrics for a fitted model."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def file_md5(path) -> str:
    """Fingerprint of the training data, logged so every run is traceable."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def strip_prefix(params: dict) -> dict:
    """'clf__C' -> 'C' for readable MLflow parameter names."""
    return {k.replace("clf__", ""): v for k, v in params.items()}


def plot_confusion_matrix(model, X_test, y_test, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    return fig


def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_title("ROC curves on held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_tuning(search: GridSearchCV, name):
    """CV ROC-AUC (± std) of every grid candidate, best first."""
    res = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_roc_auc", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(res.index + 1, res["mean_test_roc_auc"],
                yerr=res["std_test_roc_auc"], fmt="o", ms=4, capsize=2)
    ax.set_xlabel("Hyperparameter candidate (ranked)")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title(f"Grid search — {name} ({len(res)} candidates)")
    fig.tight_layout()
    return fig


def log_figure(fig, filename):
    """Save a figure to screenshots/models/ and attach it to the MLflow run."""
    fig.savefig(FIG_DIR / filename, dpi=150)
    mlflow.log_figure(fig, f"plots/{filename}")
    plt.close(fig)


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def configure_logging():
    """(Re)configure logging AFTER MLflow is initialised.

    Creating a fresh SQLite store runs alembic migrations, whose logging
    config silences existing loggers and raises the root level to WARNING.
    Without this, CI (always a fresh store) would lose all training logs.
    """
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.disabled = False


def run_candidate(name, estimator, grid, X_train, X_test, y_train, y_test):
    """Tune + evaluate one model family inside its own nested MLflow run."""
    with mlflow.start_run(run_name=name, nested=True) as run:
        logger.info("Tuning %s over %d combinations", name,
                    len(ParameterGrid(grid)))
        search = tune_model(estimator, grid, X_train, y_train)
        model = search.best_estimator_
        metrics = {**cv_summary(search), **evaluate(model, X_test, y_test)}

        mlflow.set_tag("model_family", name)
        mlflow.log_param("model_class", type(estimator).__name__)
        mlflow.log_param("grid_size", len(ParameterGrid(grid)))
        mlflow.log_params(strip_prefix(search.best_params_))
        mlflow.log_metrics(metrics)

        log_figure(plot_confusion_matrix(model, X_test, y_test, name),
                   f"confusion_matrix_{name}.png")
        log_figure(plot_tuning(search, name), f"tuning_{name}.png")

        cv_table = REPORT_DIR / f"cv_results_{name}.csv"
        pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(
            cv_table, index=False)
        mlflow.log_artifact(str(cv_table), artifact_path="cv_results")

        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
        )

        logger.info("%s best params: %s", name, search.best_params_)
        logger.info("%s CV ROC-AUC %.3f ± %.3f | test ROC-AUC %.3f",
                    name, metrics["cv_roc_auc_mean"],
                    metrics["cv_roc_auc_std"], metrics["test_roc_auc"])
        return model, {"model": name, **metrics, "run_id": run.info.run_id,
                       "best_params": json.dumps(search.best_params_)}


def main():
    setup_mlflow()
    configure_logging()

    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info("Train: %d rows, test: %d rows", len(X_train), len(X_test))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="model-selection") as parent:
        mlflow.log_params({
            "data_path": str(CLEAN_DATA_PATH.relative_to(PROJECT_ROOT)),
            "data_md5": file_md5(CLEAN_DATA_PATH),
            "n_rows": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "selection_metric": f"cv_{SELECTION_METRIC}_mean",
        })

        rows, fitted = [], {}
        for name, (estimator, grid) in get_model_specs().items():
            fitted[name], row = run_candidate(
                name, estimator, grid, X_train, X_test, y_train, y_test)
            rows.append(row)

        log_figure(plot_roc_curves(fitted, X_test, y_test), "roc_curves.png")

        results = (pd.DataFrame(rows)
                   .sort_values(f"cv_{SELECTION_METRIC}_mean", ascending=False)
                   .reset_index(drop=True))
        results_path = REPORT_DIR / "model_comparison.csv"
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

        best = results.iloc[0]
        logger.info("Selected model (by CV ROC-AUC): %s", best["model"])
        with pd.option_context("display.width", 200,
                               "display.float_format", "{:.3f}".format):
            logger.info("\n%s", results.drop(
                columns=["best_params", "run_id"]).to_string())

        # Option B: same winning configuration, refit on all 303 rows.
        # Its expected performance is the CV / test estimate above.
        final_model = clone(fitted[best["model"]]).fit(X, y)
        selection_metrics = {k: float(best[k]) for k in best.index
                             if k.startswith(("cv_", "test_"))}

        mlflow.set_tag("selected_model", best["model"])
        mlflow.log_metrics({f"selected_{k}": v
                            for k, v in selection_metrics.items()})
        model_info = mlflow.sklearn.log_model(
            final_model, artifact_path="final_model",
            signature=infer_signature(X, final_model.predict(X)),
            input_example=X.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        metadata = save_final_model(final_model, X, {
            "model_name": best["model"],
            "best_params": json.loads(best["best_params"]),
            "trained_on_rows": len(X),
            "data_md5": file_md5(CLEAN_DATA_PATH),
            "selection_metric": f"cv_{SELECTION_METRIC}_mean",
            "estimated_performance": selection_metrics,
            "mlflow_run_id": parent.info.run_id,
            "mlflow_model_uri": model_info.model_uri,
            "registered_model": REGISTERED_MODEL_NAME,
            "registered_version": model_info.registered_model_version,
        })
        logger.info("Final model (%s, refit on %d rows) saved to models/ — "
                    "registered as %s v%s", metadata["model_name"], len(X),
                    REGISTERED_MODEL_NAME, metadata["registered_version"])
    return results, final_model


if __name__ == "__main__":
    main()
