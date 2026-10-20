# Heart Disease Risk Prediction — End-to-End MLOps

Predicts the presence of heart disease from patient health data (UCI Heart Disease dataset)
and serves the model as a containerised, monitored API.

## Project structure

```
heart-disease-mlops/
├── data/
│   ├── raw/            # downloaded by script (not committed)
│   └── processed/      # cleaned dataset (committed)
├── notebooks/          # EDA notebook
├── src/                # data, training and API code
├── tests/              # pytest unit tests
├── models/             # final serialised model
├── k8s/                # Kubernetes manifests
├── monitoring/         # Prometheus / Grafana config
├── screenshots/        # images used in the report
├── reports/            # final report
└── .github/workflows/  # CI/CD pipeline
```

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — Cleveland subset
(303 patients, 13 features + diagnosis).

```bash
python -m src.data          # downloads to data/raw/ and writes data/processed/heart_clean.csv
```

Cleaning steps (`src/data.py`):
- `?` markers parsed as missing (`ca`: 4 rows, `thal`: 2 rows)
- duplicate rows dropped, impossible clinical values nulled out
- diagnosis `num` (0–4) converted to binary `target` (0 = no disease, 1 = disease) → 164 / 139 split
- missing values are kept in the CSV and **imputed inside the model pipeline**, so training and inference use identical logic and no test-set statistics leak into training

## Exploratory Data Analysis

`notebooks/01_eda.ipynb` — figures are exported to `screenshots/eda/`.

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Key findings: balanced classes (54/46); strongest signals are `thal`, `ca`, `exang`, `oldpeak`,
`cp` (positive) and `thalach` (negative); `fbs`/`chol` are weak; no strong multicollinearity
(max |r| = 0.58 between features); `cp`, `restecg`, `slope`, `thal` are nominal codes → one-hot encoding.

## Feature engineering & model training

```bash
python -m src.train
```

- **Preprocessing** (`src/features.py`) — one `ColumnTransformer`, saved inside the model pipeline:
  numeric → median impute + `StandardScaler`; binary → most-frequent impute;
  categorical (`cp`, `restecg`, `slope`, `thal`) → most-frequent impute + `OneHotEncoder(handle_unknown="ignore")`.
- **Validation**: stratified 80/20 train/test split; hyperparameters tuned with `GridSearchCV`
  using stratified 5-fold CV on the training set only.
- **Selection rule**: best mean CV ROC-AUC (the test set is only used for the final report).

| Model | CV ROC-AUC | CV accuracy | CV precision | CV recall | Test ROC-AUC | Test accuracy |
|---|---|---|---|---|---|---|
| **Logistic Regression** (C=0.3, L2, balanced) | **0.907 ± 0.019** | 0.855 | 0.882 | 0.801 | 0.961 | 0.869 |
| Random Forest (400 trees, min_leaf=3) | 0.901 ± 0.026 | 0.818 | 0.825 | 0.774 | 0.952 | 0.885 |
| Gradient Boosting (lr=0.03, depth=2) | 0.884 ± 0.027 | 0.822 | 0.851 | 0.756 | 0.960 | 0.902 |

Outputs: `reports/model_comparison.csv`, `reports/cv_results_<model>.csv` (every grid candidate),
ROC curves, confusion matrices and tuning plots in `screenshots/models/`.

## Experiment tracking (MLflow)

Every `python -m src.train` run is logged to a local MLflow store (`mlflow.db` + `mlruns/`, not committed):

- **Parent run `model-selection`** — data fingerprint (`data_md5`), split sizes, CV setup, the model
  comparison table, combined ROC curve, and the final packaged model.
- **One nested run per model family** — best hyperparameters, CV mean/std + test metrics,
  confusion matrix, grid-search plot, full `cv_results` table, and the fitted pipeline.
- **Model registry** — the final model is registered as `heart-disease-classifier` (new version per run).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # then open http://127.0.0.1:5000
```

Set `MLFLOW_TRACKING_URI` to log to a remote tracking server instead.

## Model packaging

The selected configuration is **refit on all 303 rows** and saved to `models/`:

| File | Purpose |
|---|---|
| `models/model.joblib` | full preprocessing + classifier pipeline, loaded by the API |
| `models/mlflow_model/` | MLflow format (`MLmodel`, input signature, pinned requirements) — `mlflow.pyfunc.load_model("models/mlflow_model")` |
| `models/model_metadata.json` | model name, hyperparameters, feature order, data hash, CV/test performance estimate, MLflow run id & registry version |

Because imputation, scaling and one-hot encoding live inside the saved pipeline, raw patient
records (including missing values or unseen category codes) can be passed directly to `predict_proba`.

## Testing

```bash
pytest                                   # 44 tests
pytest --cov=src --cov-report=term-missing
```

| File | Covers |
|---|---|
| `tests/test_data.py` | `?` parsing, target binarisation, duplicates, out-of-range values, download skip/force (network mocked) |
| `tests/test_features.py` | column order/dtypes, imputation leaves no NaNs, scaling, one-hot width, unseen categories, missing columns |
| `tests/test_train.py` | model grids, CV summary, metrics range, learns better than chance, **end-to-end training run** in a temp dir (train → select → refit → save → reload → predict → report), quality gate |
| `tests/test_model_artifact.py` | packaged `models/` model: schema, probabilities, missing/unseen inputs, high-risk > low-risk patient |
| `tests/test_api.py` | API contract: `/health`, `/model-info`, `/predict` output fields, null `ca`/`thal` accepted, invalid/unknown/missing fields → 422, risk bands, inference CLI = API |

Tests use small synthetic datasets (`tests/conftest.py`), so they need no network and never touch `models/` or the MLflow store.

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push to `main`, every pull request, and manually:

```
lint (flake8) ──► test (pytest + coverage) ──┬─► train  (data → train → quality gate)
                                             └─► docker (build image → run → smoke-test /predict)
```

| Job | Steps | Artifacts |
|---|---|---|
| **lint** | `flake8` (config in `.flake8`) | `lint-report` |
| **test** | install `requirements.txt` → `pytest` with JUnit XML + coverage | `test-results` (junit.xml, coverage.xml) |
| **docker** | `docker build` → run container → wait for `/health` → POST `samples/patient_high_risk.json` (expects class 1) → invalid input must return 422 | container logs in the job output |
| **train** | `python -m src.data --force` → verify cleaned CSV is identical to the committed one → `python -m src.train` → `python -m src.ci_report` (fails if CV ROC-AUC < 0.85; model table written to the job summary) | `model`, `training-reports` (log, CSVs, plots), `mlflow-tracking` (mlflow.db + mlruns) |

Each job depends on the previous one, so any lint error, failing test, data drift or below-threshold
model stops the pipeline with a red build and the failing step's log.

## Inference & serving API

### Inference script

```bash
python -m src.predict samples/patient_high_risk.json                        # local model
python -m src.predict samples/patients.json                                 # a list of patients
python -m src.predict samples/patients.json --url http://localhost:8000     # same, via the API
```

`src/predict.py::score_records` is the only place that turns raw records into predictions;
both the CLI and the API call it, so they always agree.

### FastAPI service (`src/api.py`)

```bash
uvicorn src.api:app --reload        # interactive docs at http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness/readiness check (Docker `HEALTHCHECK`, Kubernetes probes) |
| `GET /model-info` | served model version, hyperparameters, features, CV/test performance estimate |
| `POST /predict` | one patient (JSON) → prediction, confidence, risk level |

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" -d @samples/patient_high_risk.json
```
```json
{"prediction": 1, "label": "heart disease", "confidence": 0.9961,
 "probability_disease": 0.9961, "risk_level": "high", "model_version": "logistic_regression-v1"}
```

- `confidence` = probability of the predicted class; `risk_level` bands P(disease): < 0.3 low, < 0.7 moderate, otherwise high.
- **Input validation** (pydantic): categorical codes must be valid UCI codes (e.g. `cp` ∈ 1–4, `thal` ∈ {3, 6, 7}),
  numeric values must lie in the same plausible clinical ranges used during data cleaning, and unknown fields
  are rejected → HTTP 422 with a field-level message. Only `ca` and `thal` (missing in the real data) may be `null`;
  the model pipeline imputes them.
- Settings via environment variables: `MODEL_PATH`, `PREDICTION_THRESHOLD` (default 0.5).

### Docker

The image contains only the serving code and the model (`requirements-serve.txt` — no MLflow, Jupyter or plotting
libraries), runs as a non-root user and has a built-in health check.

```bash
docker build -t heart-disease-api:1.0 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0
curl http://localhost:8000/health
python -m src.predict samples/patients.json --url http://localhost:8000
docker logs heart-api
```

