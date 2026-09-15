# ============================================================
# DATA LOADER — ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# Expects the Kaggle "Digit Recognizer" CSV format:
#   train.csv : columns = [label, pixel0, pixel1, ..., pixel783]
#   test.csv  : columns = [pixel0, pixel1, ..., pixel783]   (no label)
#
# Also supports the classic full MNIST CSV (same schema, 60k/10k rows) —
# both are handled by the same validation logic.
# ============================================================

import os
import numpy as np
import pandas as pd
import logging

from src.config import INPUT_DIM, NUM_CLASSES

logger = logging.getLogger(__name__)

PIXEL_COLS_PREFIX = "pixel"


# ============================================================
# VALIDATION
# ============================================================


def validate_raw_data(df: pd.DataFrame, has_label: bool = True) -> pd.DataFrame:
    """
    Validates a raw MNIST-style dataframe:
      - correct number of pixel columns (784)
      - pixel values in [0, 255]
      - label column present + values in [0, 9] (if has_label)
      - no fully-null rows
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    pixel_cols = [c for c in df.columns if c.startswith(PIXEL_COLS_PREFIX)]
    if len(pixel_cols) != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} pixel columns, found {len(pixel_cols)}. "
            "Check the CSV matches the Kaggle Digit Recognizer schema."
        )

    if has_label:
        if "label" not in df.columns:
            raise ValueError("Training data must contain a 'label' column.")
        bad_labels = ~df["label"].between(0, NUM_CLASSES - 1)
        if bad_labels.any():
            raise ValueError(f"Found {bad_labels.sum()} rows with label outside 0-9.")

    pixel_vals = df[pixel_cols].values
    if pixel_vals.min() < 0 or pixel_vals.max() > 255:
        raise ValueError("Pixel values must be within [0, 255].")

    n_nulls = df.isnull().sum().sum()
    if n_nulls > 0:
        raise ValueError(f"Found {n_nulls} null values — MNIST CSVs should have none.")

    logger.info(
        "Validation passed | rows=%d | pixel_cols=%d | has_label=%s",
        len(df),
        len(pixel_cols),
        has_label,
    )
    return df


# ============================================================
# LOAD
# ============================================================


def load_mnist_csv(path: str, has_label: bool = True):
    """
    Loads a Kaggle-format MNIST CSV and returns (X, y) or (X, None).
      X : (n_samples, 784) float64 raw pixel values (0-255, NOT yet scaled)
      y : (n_samples,) int labels, or None if has_label=False
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from "
            "https://www.kaggle.com/competitions/digit-recognizer/data "
            "and place train.csv / test.csv under data/."
        )

    df = pd.read_csv(path)
    df = validate_raw_data(df, has_label=has_label)

    pixel_cols = [c for c in df.columns if c.startswith(PIXEL_COLS_PREFIX)]
    # Preserve numeric pixel order (pixel0, pixel1, ..., pixel783), not lexical order
    pixel_cols = sorted(pixel_cols, key=lambda c: int(c.replace(PIXEL_COLS_PREFIX, "")))

    X = df[pixel_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int64) if has_label else None

    logger.info("Loaded %s | X=%s | y=%s", path, X.shape, None if y is None else y.shape)
    return X, y


def load_train_data(data_dir: str = "data", filename: str = "train.csv"):
    return load_mnist_csv(os.path.join(data_dir, filename), has_label=True)


def load_kaggle_test_data(data_dir: str = "data", filename: str = "test.csv"):
    """Kaggle's unlabeled test.csv — used only for competition submission, never for evaluation."""
    return load_mnist_csv(os.path.join(data_dir, filename), has_label=False)


# ============================================================
# CLASS BALANCE REPORT
# ============================================================


def class_distribution(y: np.ndarray) -> pd.Series:
    dist = pd.Series(y).value_counts().sort_index()
    dist.index.name = "digit"
    dist.name = "count"
    return dist
