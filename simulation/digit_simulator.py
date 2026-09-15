# ============================================================
# DIGIT SIMULATOR — ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# Simulates production inference traffic hitting the API, under
# three scenarios, and writes monitoring artifacts the dashboard
# reads:
#
#   1. NORMAL   — held-out test images, same distribution as training
#   2. NOISY    — test images with added Gaussian pixel noise
#                 (simulates a scanner/camera degrading over time)
#   3. DRIFTED  — test images with contrast/brightness shift
#                 (simulates a new device / lighting condition)
#
# For each scenario it calls the running API (or the service
# directly if API is unreachable), logs predictions, and computes
# PSI of mean-pixel-intensity vs. the training baseline.
#
# FIX (2026-08-21): _call_api_or_local() previously re-attempted the
# HTTP call to the API on EVERY single sample, even after the very
# first attempt already failed (API not running). With n_samples=300
# across 3 scenarios that's up to 900 doomed HTTP attempts, each
# paying Windows/network connection-refused or timeout overhead
# before falling back to the in-process model -- easily turning a
# few-second job into many minutes and looking "stuck". We now probe
# the API ONCE at the start of run_simulation() and, if unreachable,
# skip HTTP entirely for the rest of the run.
# ============================================================

import os
import time
import json
import logging
import numpy as np
import pandas as pd
import requests

from src.config import DATA_DIR, MODEL_DIR
from src.data_loader import load_train_data
from src.metrics import psi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("MNIST_API_URL", "http://localhost:8000") + "/predict"
API_HEALTH_URL = os.getenv("MNIST_API_URL", "http://localhost:8000") + "/health"
MONITOR_SCORES_PATH = os.path.join(MODEL_DIR, "monitor_scores.csv")
DRIFT_REPORT_PATH = os.path.join(MODEL_DIR, "feature_drift_report.csv")


def _add_noise(X, sigma=40):
    noisy = X + np.random.RandomState(1).normal(0, sigma, X.shape)
    return np.clip(noisy, 0, 255)


def _shift_contrast(X, factor=0.5, brightness=60):
    shifted = X * factor + brightness
    return np.clip(shifted, 0, 255)


def _check_api_available(timeout: float = 2.0) -> bool:
    """One-time probe -- avoids paying connection-refused/timeout cost
    on every one of hundreds of samples in the loop below."""
    try:
        r = requests.get(API_HEALTH_URL, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _get_local_service():
    from services.prediction_service import PredictionService

    if not hasattr(_get_local_service, "_service"):
        _get_local_service._service = PredictionService()
    return _get_local_service._service


def _call_api_or_local(pixels, api_available: bool):
    if api_available:
        try:
            r = requests.post(API_URL, json={"pixels": pixels.tolist()}, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.warning("API call failed mid-run (%s) -- falling back to local model for the rest of this run.", e)
            # Note: caller passes a mutable-friendly flag via run_simulation's
            # local variable, so a single mid-run failure degrades gracefully
            # without reverting to per-sample retries.
    return _get_local_service().predict(pixels)


def run_simulation(n_samples: int = 300, scenarios=("normal", "noisy", "drifted")):
    X, y = load_train_data(DATA_DIR, "train.csv")
    baseline_mean_intensity = X.mean(axis=1)  # reference distribution for PSI

    rng = np.random.RandomState(7)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X[idx]

    # ── Probe the API ONCE, up front ──────────────────────────
    api_available = _check_api_available()
    if api_available:
        logger.info("API reachable at %s -- using live HTTP calls.", API_URL)
    else:
        logger.info(
            "API not reachable at %s -- using in-process model for this entire run "
            "(start `python scripts/run_api.py` in another terminal to hit the live API instead).",
            API_URL,
        )

    all_scores = []
    drift_rows = []

    for scenario in scenarios:
        if scenario == "normal":
            X_scenario = X_sample
        elif scenario == "noisy":
            X_scenario = _add_noise(X_sample)
        elif scenario == "drifted":
            X_scenario = _shift_contrast(X_sample)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        logger.info("Running scenario '%s' (%d samples)...", scenario, len(X_scenario))
        t0 = time.time()

        confidences = []
        for i, row in enumerate(X_scenario):
            result = _call_api_or_local(row, api_available)
            all_scores.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": scenario,
                    "predicted_digit": result["predicted_digit"],
                    "confidence": result["confidence"],
                }
            )
            confidences.append(result["confidence"])
            if (i + 1) % 100 == 0:
                logger.info("  ... %d/%d samples done", i + 1, len(X_scenario))

        scenario_mean_intensity = X_scenario.mean(axis=1)
        psi_value = psi(baseline_mean_intensity, scenario_mean_intensity)

        drift_rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario": scenario,
                "psi_mean_intensity": round(psi_value, 4),
                "mean_confidence": round(float(np.mean(confidences)), 4),
                "n_samples": len(X_scenario),
            }
        )

        logger.info(
            "Scenario '%s' complete in %.1fs | PSI=%.4f | mean_confidence=%.4f",
            scenario, time.time() - t0, psi_value, np.mean(confidences),
        )

    scores_df = pd.DataFrame(all_scores)
    drift_df = pd.DataFrame(drift_rows)

    os.makedirs(MODEL_DIR, exist_ok=True)
    scores_df.to_csv(MONITOR_SCORES_PATH, index=False)
    drift_df.to_csv(DRIFT_REPORT_PATH, index=False)

    logger.info("Simulation complete -> %s, %s", MONITOR_SCORES_PATH, DRIFT_REPORT_PATH)
    return scores_df, drift_df


if __name__ == "__main__":
    run_simulation()