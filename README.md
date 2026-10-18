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

_More sections (data, training, API, deployment, monitoring) will be added as the project progresses._
