"""Summarise a training run and enforce a minimum quality bar.

Used by CI after `python -m src.train`: prints a Markdown report (appended
to the GitHub Actions job summary) and exits non-zero if the selected
model's CV ROC-AUC is below the threshold, so a bad model fails the build.

Usage:
    python -m src.ci_report [--min-roc-auc 0.85]
"""
import argparse
import sys

import pandas as pd

from src.config import PROJECT_ROOT
from src.model_io import load_metadata

COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
DEFAULT_MIN_ROC_AUC = 0.85
REPORT_COLUMNS = ["model", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_accuracy_mean",
                  "cv_precision_mean", "cv_recall_mean", "test_roc_auc", "test_accuracy"]


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def build_report(results: pd.DataFrame, metadata: dict, min_roc_auc: float) -> str:
    score = metadata["estimated_performance"]["cv_roc_auc_mean"]
    status = "PASSED" if score >= min_roc_auc else "FAILED"
    table = markdown_table(results[REPORT_COLUMNS].round(3))
    return "\n".join([
        "## Model training report",
        "",
        f"- **Selected model:** `{metadata['model_name']}` "
        f"(refit on {metadata['trained_on_rows']} rows)",
        f"- **Hyperparameters:** `{metadata['best_params']}`",
        f"- **Data MD5:** `{metadata['data_md5']}`",
        f"- **Quality gate:** CV ROC-AUC {score:.3f} vs minimum {min_roc_auc:.2f} "
        f"→ **{status}**",
        "",
        "### Candidate comparison",
        "",
        table,
        "",
    ])


def passes_gate(metadata: dict, min_roc_auc: float) -> bool:
    return metadata["estimated_performance"]["cv_roc_auc_mean"] >= min_roc_auc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-roc-auc", type=float, default=DEFAULT_MIN_ROC_AUC)
    args = parser.parse_args(argv)

    results = pd.read_csv(COMPARISON_PATH)
    metadata = load_metadata()
    print(build_report(results, metadata, args.min_roc_auc))
    return 0 if passes_gate(metadata, args.min_roc_auc) else 1


if __name__ == "__main__":
    sys.exit(main())
