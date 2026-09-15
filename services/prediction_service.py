# ============================================================
# PREDICTION SERVICE — ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import time
import os
import csv
import logging
import numpy as np

from src.model_loader import load_latest_model
from src.preprocessing import PixelScaler
from src.config import INPUT_DIM, LOGS_DIR

logger = logging.getLogger(__name__)

PREDICTION_LOG = os.path.join(LOGS_DIR, "prediction_logs.csv")


class PredictionService:
    """
    Thin wrapper around the champion NeuralNetworkFromScratch that:
      - validates raw pixel input (784 values, 0-255)
      - scales it the same way training data was scaled
      - returns predicted digit + full softmax distribution
      - logs every prediction for monitoring / drift analysis
    """

    def __init__(self):
        self.network, self.registry = load_latest_model()
        self.scaler = PixelScaler()
        self.scaler.fitted_ = True
        self.scaler.scale_ = 255.0  # MNIST pixels are always 0-255
        logger.info("PredictionService ready | champion=%s", self.registry.get("model_prefix"))

    def _validate(self, pixels):
        arr = np.asarray(pixels, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected {INPUT_DIM} pixel values, got {arr.shape[1]}")
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError("Pixel values must be within [0, 255]")
        return arr

    def predict(self, pixels) -> dict:
        t0 = time.time()
        X = self._validate(pixels)
        X_scaled = self.scaler.transform(X)

        probs = self.network.predict_proba(X_scaled)[0]
        pred_digit = int(np.argmax(probs))
        confidence = float(probs[pred_digit])
        latency_ms = round((time.time() - t0) * 1000, 2)

        result = {
            "predicted_digit": pred_digit,
            "confidence": round(confidence, 4),
            "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probs)},
            "latency_ms": latency_ms,
            "model_name": self.registry.get("model_prefix"),
        }
        self._log_prediction(result, X)
        return result

    def predict_batch(self, pixel_batch) -> list:
        return [self.predict(row) for row in pixel_batch]

    def _log_prediction(self, result: dict, X: np.ndarray):
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_exists = os.path.exists(PREDICTION_LOG)
        with open(PREDICTION_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "predicted_digit",
                        "confidence",
                        "mean_pixel_intensity",
                        "latency_ms",
                        "model_name",
                    ]
                )
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    result["predicted_digit"],
                    result["confidence"],
                    round(float(X.mean()), 4),
                    result["latency_ms"],
                    result["model_name"],
                ]
            )
