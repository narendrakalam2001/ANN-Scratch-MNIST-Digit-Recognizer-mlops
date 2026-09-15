"""Entry point: python scripts/train_model.py"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-tuning", action="store_true", help="Skip hyperparameter search, use config.py defaults"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Explicit model version tag (e.g. v5). If omitted, auto-increments "
        "past every existing model on disk AND the current registry entry.",
    )
    args = parser.parse_args()

    run_training_pipeline(run_tuning=not args.no_tuning, epochs=args.epochs, model_version=args.version)