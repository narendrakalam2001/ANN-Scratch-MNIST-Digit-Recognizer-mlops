# ============================================================
# MODEL CARD — Google Model Cards standard (JSON)
# ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import json
import os
import time
import logging

logger = logging.getLogger(__name__)


def build_model_card(
    model_name: str,
    version: str,
    architecture_config: dict,
    metrics: dict,
    thresholds: dict,
    training_config: dict,
    dataset_info: dict,
    limitations: list = None,
    ethical_considerations: list = None,
) -> dict:
    card = {
        "model_details": {
            "name": model_name,
            "version": version,
            "type": "Fully-connected Artificial Neural Network (from scratch, NumPy only)",
            "framework": "None — raw NumPy implementation of forward/backward propagation",
            "date": time.strftime("%Y-%m-%d"),
            "architecture": architecture_config,
        },
        "intended_use": {
            "primary_use": "Handwritten digit classification (0-9) on 28x28 grayscale images",
            "primary_users": "ML engineers demonstrating fundamentals; digitization pipelines (forms, cheques, postal codes)",
            "out_of_scope": [
                "Non-digit handwritten character recognition",
                "Natural scene text recognition",
                "Production use without further validation on target deployment distribution",
            ],
        },
        "training_data": dataset_info,
        "training_config": training_config,
        "metrics": metrics,
        "thresholds": thresholds,
        "limitations": limitations
        or [
            "Trained only on MNIST-style centered, normalized, black-background digit images.",
            "No convolutional structure — spatial locality is not explicitly modeled, unlike a CNN.",
            "Sensitive to input images that are not centered/cropped like MNIST (rotation, translation, scale).",
        ],
        "ethical_considerations": ethical_considerations
        or [
            "Misclassifications in downstream systems (e.g. cheque/form processing) should route to human review "
            "rather than auto-reject/auto-approve on low-confidence predictions.",
        ],
        "quantitative_analysis": {
            "notes": "See docs/plots/confusion_matrix.png and docs/plots/per_class_f1.png for per-digit breakdown.",
        },
    }
    return card


def save_model_card(card: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    logger.info("Model card saved -> %s", path)
