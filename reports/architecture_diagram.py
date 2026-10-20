"""Draw the end-to-end architecture diagram used in the report.

Usage:
    python reports/architecture_diagram.py      # -> screenshots/architecture/architecture.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "architecture" / "architecture.png"

# one colour per stage of the pipeline
BUILD, CICD, SERVE, MONITOR, NEUTRAL = "#2563eb", "#7c3aed", "#059669", "#d97706", "#475569"


def band(ax, x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=color, alpha=0.06, edgecolor=color, lw=1.2))
    ax.text(x + 1.5, y + h - 1.6, title, color=color, fontsize=12.5, weight="bold", va="top")


def box(ax, x, y, w, h, title, lines=(), color=NEUTRAL, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1",
                                facecolor="white", edgecolor=color, lw=1.6,
                                linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=9.6,
            weight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 4.6 - i * 2.55, line, ha="center", va="top",
                fontsize=7.9, color="#1f2937")
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}


def arrow(ax, a, b, label="", color=NEUTRAL, rad=0.0, label_xy=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if label:
        lx, ly = label_xy or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 1.2)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=7.4, color=color,
                style="italic", bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def main():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 110)
    ax.axis("off")
    ax.text(90, 108.5, "Heart Disease Risk Prediction — end-to-end MLOps architecture",
            ha="center", va="top", fontsize=16, weight="bold", color="#111827")

    # ---- 1. build the model -------------------------------------------------
    band(ax, 1, 72, 178, 32, "1  Data, training & experiment tracking", BUILD)
    uci = box(ax, 4, 76, 24, 21, "UCI repository", [
        "Heart Disease, Cleveland", "303 patients, 13 features", "+ diagnosis num 0-4"], NEUTRAL)
    data = box(ax, 33, 76, 28, 21, "src/data.py", [
        "download + parse '?' as NaN", "range checks, drop duplicates",
        "target = num > 0", "→ heart_clean.csv (in git)"], BUILD)
    train = box(ax, 66, 76, 38, 21, "src/features.py + src/train.py", [
        "ColumnTransformer: impute, scale, one-hot",
        "GridSearchCV, stratified 5-fold CV",
        "LogReg vs RandomForest vs GradBoost",
        "select by CV ROC-AUC → LogReg 0.907",
        "refit on all 303 rows"], BUILD)
    mlflow = box(ax, 109, 76, 32, 21, "MLflow (SQLite backend)", [
        "parent run + 1 nested run per model",
        "params, CV/test metrics, data md5",
        "plots, cv_results, fitted pipelines",
        "registry: heart-disease-classifier"], BUILD)
    model = box(ax, 146, 76, 30, 21, "models/", [
        "model.joblib (pipeline)", "mlflow_model/ (signature,",
        "pinned requirements)", "model_metadata.json"], BUILD)
    arrow(ax, uci["r"], data["l"], "download", BUILD)
    arrow(ax, data["r"], train["l"], "clean CSV", BUILD)
    arrow(ax, train["r"], mlflow["l"], "log runs", BUILD)
    arrow(ax, mlflow["r"], model["l"], "package", BUILD)

    # ---- 2. CI/CD ----------------------------------------------------------
    band(ax, 1, 40, 178, 29, "2  CI/CD — GitHub Actions on every push / pull request", CICD)
    repo = box(ax, 4, 44, 24, 19, "GitHub repository", [
        "code, tests, Dockerfile", "k8s/ + monitoring/ manifests",
        "cleaned data, model"], NEUTRAL)
    lint = box(ax, 36, 46, 22, 15, "lint", ["flake8"], CICD)
    test = box(ax, 66, 46, 26, 15, "test", ["pytest, 51 tests", "coverage 91%"], CICD)
    ci_train = box(ax, 102, 53, 40, 13, "train", [
        "re-download → data unchanged?", "train + MLflow → gate CV ROC-AUC ≥ 0.85"], CICD)
    ci_docker = box(ax, 102, 41.5, 40, 10.5, "docker", [
        "build image → /health → /predict → 422 check"], CICD)
    artifacts = box(ax, 150, 44, 26, 19, "run artifacts", [
        "lint + test reports", "model, training log, plots",
        "MLflow store", "job summary table"], CICD, dashed=True)
    arrow(ax, repo["r"], lint["l"], "push", CICD)
    arrow(ax, lint["r"], test["l"], "", CICD)
    arrow(ax, test["r"], ci_train["l"], "", CICD, rad=-0.15)
    arrow(ax, test["r"], ci_docker["l"], "", CICD, rad=0.15)
    arrow(ax, ci_train["r"], artifacts["l"], "", CICD)
    arrow(ax, ci_docker["r"], artifacts["l"], "", CICD)
    arrow(ax, (161, 76), (161, 63.5), "model in repo", NEUTRAL, label_xy=(167, 68.5))
    ax.text(64, 42.2, "any failing job stops the pipeline (red build, clear log)",
            ha="center", fontsize=7.6, color=CICD, style="italic")

    # ---- 3. serve & monitor --------------------------------------------------
    band(ax, 1, 1, 178, 36, "3  Serving & monitoring on Kubernetes", SERVE)
    clients = box(ax, 4, 6, 24, 25, "Clients", [
        "curl / Swagger UI /docs", "src/predict.py --url",
        "src/traffic.py (load)", "", "http://localhost:80"], NEUTRAL)
    image = box(ax, 33, 19, 26, 11, "Docker image 1.1", [
        "python:3.12-slim, non-root", "src/ + model.joblib"], SERVE)
    # namespace heart-disease
    ax.add_patch(FancyBboxPatch((62, 4), 62, 29.5, boxstyle="round,pad=0,rounding_size=1",
                                facecolor="none", edgecolor=SERVE, lw=1, linestyle=":"))
    ax.text(122.8, 4.6, "namespace heart-disease", fontsize=8, color=SERVE, ha="right")
    svc = box(ax, 64, 8, 22, 16, "Service", ["type LoadBalancer", "port 80 → 8000",
                                             "spreads requests"], SERVE)
    pods = box(ax, 91, 6, 31, 24, "Deployment heart-api", [
        "2 replicas, FastAPI + uvicorn", "POST /predict → class + confidence",
        "GET /health → probes", "GET /metrics → Prometheus",
        "JSON request logs (stdout)", "rolling update, preStop 10 s"], SERVE)
    ax.text(4, 2.6, "Kubernetes: Docker Desktop built-in cluster (kind), context docker-desktop",
            fontsize=7.4, color=SERVE, style="italic")
    # namespace monitoring
    ax.add_patch(FancyBboxPatch((127, 4), 50, 29.5, boxstyle="round,pad=0,rounding_size=1",
                                facecolor="none", edgecolor=MONITOR, lw=1, linestyle=":"))
    ax.text(128.2, 32.6, "namespace monitoring", fontsize=8, color=MONITOR, va="top")
    prom = box(ax, 129, 6, 22, 24, "Prometheus", [
        "pod discovery", "scrapes each pod", "every 10 s", "5 alert rules", ":9090"], MONITOR)
    graf = box(ax, 155, 6, 20, 24, "Grafana", [
        "provisioned", "dashboard:", "traffic, latency,", "risk levels, drift", ":3030"], MONITOR)
    arrow(ax, clients["r"], svc["l"], "JSON", SERVE, label_xy=(46, 12.5))
    arrow(ax, svc["r"], pods["l"], "", SERVE)
    arrow(ax, image["r"], (91, 28.5), "runs as 2 pods", SERVE, label_xy=(76, 28.4))
    arrow(ax, (pods["r"][0], 18), (prom["l"][0], 18), "scrape", MONITOR)
    arrow(ax, prom["r"], graf["l"], "query", MONITOR)
    arrow(ax, (112, 41.5), (54, 30), "docker build (Dockerfile + requirements-serve.txt)",
          SERVE, label_xy=(104, 36.9))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
