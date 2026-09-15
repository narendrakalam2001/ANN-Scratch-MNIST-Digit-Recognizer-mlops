"""
extract_sample_image.py — pull a REAL digit out of your Kaggle CSV and
save it as a PNG, so you have something genuine to upload in the
Streamlit dashboard's "Predict a Digit" sidebar (Kaggle's train.csv /
test.csv are pure tabular pixel data -- there are no .png/.jpg files
in the dataset, so this script bridges that gap).

Usage:
    python extract_sample_image.py                     # random row from train.csv
    python extract_sample_image.py --index 17           # specific row
    python extract_sample_image.py --file test.csv --index 5 --out my_digit.png

Requires: pillow (pip install pillow --break-system-packages if missing)
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image


def extract_image(csv_path: str, index: int, out_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    pixel_cols = sorted(
        [c for c in df.columns if c.startswith("pixel")],
        key=lambda c: int(c.replace("pixel", "")),
    )
    if len(pixel_cols) != 784:
        raise ValueError(f"Expected 784 pixel columns, found {len(pixel_cols)} in {csv_path}")

    if index >= len(df):
        raise ValueError(f"Index {index} out of range (file has {len(df)} rows)")

    row = df.iloc[index]
    pixels = row[pixel_cols].values.astype(np.uint8).reshape(28, 28)

    label = None
    if "label" in df.columns:
        label = int(row["label"])

    img = Image.fromarray(pixels, mode="L")
    img.save(out_path)

    print(f"Saved {out_path} (28x28, from row {index} of {csv_path})")
    if label is not None:
        print(f"True label (from CSV): {label}  <-- use this to check if the API predicts correctly")
    else:
        print("This file has no label column (likely test.csv) -- no ground truth to compare against.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/train.csv", help="Path to train.csv or test.csv")
    parser.add_argument("--index", type=int, default=None, help="Row index to extract (default: random)")
    parser.add_argument("--out", type=str, default="sample_digit.png", help="Output PNG path")
    args = parser.parse_args()

    idx = args.index
    if idx is None:
        n_rows = len(pd.read_csv(args.file, nrows=0)) or 1
        # Cheap row count without loading full file into memory twice
        with open(args.file) as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
        idx = np.random.randint(0, n_rows)
        print(f"No --index given, picked random row {idx}")

    extract_image(args.file, idx, args.out)
