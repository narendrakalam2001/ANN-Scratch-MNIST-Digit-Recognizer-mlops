# ============================================================
# MNIST DIGIT RECOGNIZER API — FastAPI Serving
# ============================================================

import logging
import time
import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List

from services.prediction_service import PredictionService
from src.config import INPUT_DIM, MODEL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ANN From Scratch — MNIST Digit Recognizer API",
    description="Serves a fully-connected neural network implemented from scratch with NumPy (no ML frameworks).",
    version="1.0.0",
)

# ── Load champion model on startup ─────────────────────────────
try:
    service = PredictionService()
    logger.info("Prediction service initialized successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    service = None


# ============================================================
# INPUT SCHEMA
# ============================================================


class DigitImageInput(BaseModel):
    pixels: List[float] = Field(..., description=f"Flattened {INPUT_DIM}-length pixel array, values 0-255")

    @validator("pixels")
    def check_length(cls, v):
        if len(v) != INPUT_DIM:
            raise ValueError(f"pixels must have exactly {INPUT_DIM} values, got {len(v)}")
        if min(v) < 0 or max(v) > 255:
            raise ValueError("pixel values must be within [0, 255]")
        return v


class BatchDigitImageInput(BaseModel):
    images: List[List[float]]


# ============================================================
# ROUTES
# ============================================================


@app.get("/")
def home():
    return {
        "message": "ANN From Scratch — MNIST Digit Recognizer API is live",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "running", "model_loaded": service is not None}


@app.get("/model_info")
def model_info():
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
        card_path = registry.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                registry["model_card"] = json.load(f)
        return registry
    return {"error": "Model registry not found"}


@app.post("/predict")
def predict(image: DigitImageInput):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run scripts/train_model.py first")

    start = time.time()
    try:
        result = service.predict(image.pixels)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result["latency_seconds"] = round(time.time() - start, 4)
    return result


@app.post("/predict_batch")
def predict_batch(batch: BatchDigitImageInput):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run scripts/train_model.py first")

    for row in batch.images:
        if len(row) != INPUT_DIM:
            raise HTTPException(
                status_code=422, detail=f"Each image must have exactly {INPUT_DIM} pixel values"
            )

    results = service.predict_batch(batch.images)
    return {"predictions": results, "count": len(results)}
