"""Download, clean and save the UCI Heart Disease (Cleveland) dataset.

Usage:
    python -m src.data            # download (if needed) + clean
    python -m src.data --force    # re-download even if the raw file exists
"""
import argparse
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    CLEAN_DATA_PATH,
    COLUMNS,
    DATA_URL,
    FEATURES,
    RAW_DATA_PATH,
    TARGET,
)

logger = logging.getLogger(__name__)

# Plausible clinical ranges; values outside these are treated as data errors
VALID_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (40, 250),
    "oldpeak": (0, 10),
    "ca": (0, 3),
}


def download_data(url: str = DATA_URL, dest: Path = RAW_DATA_PATH,
                  force: bool = False) -> Path:
    """Download the raw data file unless it is already present."""
    dest = Path(dest)
    if dest.exists() and not force:
        logger.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest)
    logger.info("Saved raw data to %s", dest)
    return dest


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the header-less raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe.

    Steps:
      1. drop exact duplicate rows
      2. binarise the 0-4 diagnosis into `target` (presence/absence)
      3. null out physiologically impossible values
      4. cast categorical / binary codes to nullable integers

    Missing values are intentionally *kept* (as NaN) here. They are imputed
    inside the sklearn pipeline at training time so that the same logic runs
    at inference and no statistics leak from the test split.
    """
    df = df.copy().drop_duplicates()

    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    for col, (low, high) in VALID_RANGES.items():
        out_of_range = ~df[col].between(low, high) & df[col].notna()
        if out_of_range.any():
            logger.warning("%s: %d out-of-range values set to NaN",
                           col, out_of_range.sum())
            df.loc[out_of_range, col] = np.nan

    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        df[col] = df[col].round().astype("Int64")

    return df[FEATURES + [TARGET]].reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Count of missing values per column (only columns that have any)."""
    counts = df.isna().sum()
    return counts[counts > 0]


def main(force: bool = False) -> pd.DataFrame:
    raw_path = download_data(force=force)
    raw = load_raw(raw_path)
    clean = clean_data(raw)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_DATA_PATH, index=False)

    logger.info("Raw shape: %s -> clean shape: %s", raw.shape, clean.shape)
    logger.info("Missing values (imputed later in pipeline):\n%s",
                missing_report(clean).to_string())
    logger.info("Class balance:\n%s",
                clean[TARGET].value_counts().sort_index().to_string())
    logger.info("Clean data written to %s", CLEAN_DATA_PATH)
    return clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download the raw file")
    main(force=parser.parse_args().force)
