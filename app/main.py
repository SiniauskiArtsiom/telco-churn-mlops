"""FastAPI application for Telco Churn prediction."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from app.model import get_model_name, load_metrics, load_model, predict_proba
from app.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telco-churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Прогреваем модель при старте
    try:
        load_model()
        logger.info("Model loaded: %s", get_model_name())
    except FileNotFoundError as e:
        logger.error("Model not found at startup: %s", e)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn probability for telco clients.",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus metrics: /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    metrics = load_metrics()
    trained_at = None
    if metrics:
        # просто отметим, что метрики есть
        trained_at = None

    # список признаков, которые ожидает API
    features = list(CustomerFeatures.model_json_schema()["properties"].keys())

    return ModelInfoResponse(
        model_name=model.__class__.__name__,
        features=features,
        trained_at=trained_at,
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: CustomerFeatures) -> PredictionResponse:
    start = time.perf_counter()
    try:
        features = payload.model_dump()
        proba = predict_proba(features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("predicted proba=%.4f in %.1f ms", proba, elapsed_ms)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= 0.5,
        threshold=0.5,
        model_name=get_model_name(),
    )
