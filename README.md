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

_More sections (data, training, API, deployment, monitoring) will be added as the project progresses._
